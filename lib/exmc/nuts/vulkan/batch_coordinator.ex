defmodule Exmc.NUTS.Vulkan.BatchCoordinator do
  @moduledoc """
  N-way barrier for batched chain-shader dispatches.

  N concurrent `Sampler.sample` callers each issue per-instance leapfrog
  requests through this GenServer. When `batch_size` requests have
  accumulated in the same direction (forward / backward), the coordinator
  packs them into one `Dispatch.chain_batch` call and replies to each
  caller with its own trajectory slice.

  Step 2b constraint: all callers in a batch must share the same K and
  |epsilon|. The coordinator queues separately by direction; a future
  iteration can sub-key by (K, |eps|) if needed for heterogeneous
  warmup states.

  ## Usage

      {:ok, coord} = BatchCoordinator.start_link(batched_meta, 4)

      # From N concurrent tasks:
      result = BatchCoordinator.request_chain(coord, q, p, inv_mass, obs, +eps, K)
      # result :: {q_chain, p_chain, logp_chain, grad_chain}

  The result matches `Dispatch.chain` / `Dispatch.chain_batch[i]`'s
  per-instance shape exactly, so it slots into the existing speculative
  path's `multi_step_fn` boundary without further conversion.
  """

  use GenServer

  alias Exmc.NUTS.Vulkan.Dispatch
  alias Exmc.NUTS.Vulkan.Scheduler, as: GPUScheduler

  @default_flush_ms 25

  def start_link(batched_meta, batch_size, opts \\ []) do
    GenServer.start_link(__MODULE__, {batched_meta, batch_size, opts})
  end

  @doc """
  Submit a leapfrog request. Blocks until the coordinator has gathered
  `batch_size` requests in this direction (or `flush_ms` elapses).
  Returns the per-instance trajectory tuple from `chain_batch`.
  """
  def request_chain(pid, q, p, inv_mass, obs, eps_signed, k, timeout \\ 30_000)
      when is_number(eps_signed) and is_integer(k) do
    dir = if eps_signed >= 0, do: :forward, else: :backward
    eps_abs = abs(eps_signed)
    GenServer.call(pid, {:request, dir, q, p, inv_mass, obs, eps_abs, k}, timeout)
  end

  @doc """
  Task #171 Step 2: chain-shader-meta dispatch via batched coordinator.

  `Tree.route_chain` calls this when a caller Task has been tagged
  with `{coord_pid, obs}` in its process dict. The coordinator
  queues by `(phash2(meta), k, |eps|, dir_sign)`, dispatches each
  partition group via `Dispatch.chain_batch/5` in one
  `vkQueueSubmit`, and returns the per-instance trajectory tuple
  to the caller.

  Restricted to `{:synthesised, ...}` meta — the per-family chain
  shaders (`:normal`, `:exponential`, ...) don't have a batched
  variant. Other meta types return `{:fallback, :unsupported_meta_type}`
  so route_chain falls through to the single-instance direct path.

  Coordinator failures (GenServer timeout, dead coord, crash inside
  do_chain_flush_group) are caught and surfaced as `{:fallback, _}`
  too — the calling Task should not crash on coordinator pathology.

  Returns `{q_chain, p_chain, logp_chain, grad_chain}` on success,
  `{:fallback, reason}` on any non-batchable case.
  """
  def request_synth_chain(coord_pid, meta, q, p, inv_mass, obs, epsilon, k, dir_sign)
      when is_pid(coord_pid) do
    case meta do
      {:synthesised, _, _, _, _, _} ->
        eps_abs = abs(epsilon)

        try do
          GenServer.call(
            coord_pid,
            {:chain_request, meta, q, p, inv_mass, obs, eps_abs, k, dir_sign},
            60_000
          )
        catch
          :exit, reason -> {:fallback, {:coord_exit, reason}}
        end

      _ ->
        {:fallback, :unsupported_meta_type}
    end
  end

  def request_synth_chain(_coord_pid, _meta, _q, _p, _inv_mass, _obs, _epsilon, _k, _dir_sign) do
    {:fallback, :not_a_pid}
  end

  @doc "Force-flush any pending requests (test-only)."
  def flush_all(pid), do: GenServer.call(pid, :flush_all)

  @doc "Inspect pending request counts (test-only)."
  def pending_count(pid), do: GenServer.call(pid, :pending_count)

  @doc """
  Build a `multi_step_fn`-compatible adapter that routes leapfrog
  requests through this coordinator. Drop-in for the JIT'd multi_step_fn
  used in `Tree.build_speculative`'s speculative-precompute path.

  Closes over `(pid, obs)` so per-instance obs differs across callers
  but the coordinator's batched_meta is shared. Returned function
  matches the existing `(q, p, grad, eps_t, inv_mass, budget_t)` arity
  and produces tensors at `Exmc.JIT.precision()` to match the
  EXLA/EMLX path's dtype expectations.
  """
  def coordinator_step_fn(pid, obs) do
    fp = Exmc.JIT.precision()

    fn q, p, _grad, eps_t, inv_mass, budget_t ->
      eps_signed = Nx.to_number(eps_t)
      k = budget_t |> Nx.to_number() |> trunc()

      {q_chain, p_chain, logp_chain, grad_chain} =
        request_chain(pid, q, p, inv_mass, obs, eps_signed, k)

      {Nx.as_type(q_chain, fp), Nx.as_type(p_chain, fp), Nx.as_type(logp_chain, fp),
       Nx.as_type(grad_chain, fp)}
    end
  end

  ## GenServer callbacks

  @impl true
  def init({batched_meta, batch_size, opts}) do
    # Tag the coordinator process for DTrace consumers. No-op on
    # BEAMs built without dynamic-trace.
    :dyntrace.put_tag("BatchCoord")

    {:ok,
     %{
       meta: batched_meta,
       batch_size: batch_size,
       flush_ms: opts[:flush_ms] || @default_flush_ms,
       use_gpu_scheduler: opts[:use_gpu_scheduler] || false,
       pending: %{forward: [], backward: []},
       timers: %{forward: nil, backward: nil},
       # Task #171 Step 2: separate pending queue for chain-shader
       # dispatch. Entries carry per-request meta because chain dispatch
       # can mix multiple synthesised shaders (the coordinator stops
       # being single-shape per pid). Partition-by-(meta, k, |eps|) at
       # flush time means cross-meta cross-talk is impossible.
       chain_pending: %{forward: [], backward: []},
       chain_timers: %{forward: nil, backward: nil},
       stats: %{batches_fired: 0, requests_served: 0}
     }}
  end

  @impl true
  def handle_call({:request, dir, q, p, inv_mass, obs, eps_abs, k}, from, state) do
    entry = {from, q, p, inv_mass, obs, eps_abs, k}
    queue = state.pending[dir] ++ [entry]
    state = put_in(state.pending[dir], queue)

    if length(queue) >= state.batch_size do
      state = cancel_timer(state, dir)
      {:noreply, do_flush(dir, state)}
    else
      {:noreply, ensure_timer(state, dir)}
    end
  end

  def handle_call(:flush_all, _from, state) do
    state =
      [:forward, :backward]
      |> Enum.reduce(state, fn dir, acc ->
        if state.pending[dir] != [] do
          acc = cancel_timer(acc, dir)
          do_flush(dir, acc)
        else
          acc
        end
      end)

    {:reply, :ok, state}
  end

  def handle_call(:pending_count, _from, state) do
    {:reply, %{forward: length(state.pending.forward), backward: length(state.pending.backward)}, state}
  end

  # Task #171 Step 2: chain dispatch enqueue handler. Mirrors :request
  # but for the chain-shader pathway. dir_sign is carried explicitly
  # (rather than encoded in epsilon) since Dispatch.chain_batch takes
  # it separately.
  def handle_call({:chain_request, meta, q, p, inv_mass, obs, eps_abs, k, dir_sign}, from, state) do
    dir = if dir_sign >= 0, do: :forward, else: :backward
    entry = {from, meta, q, p, inv_mass, obs, eps_abs, k}
    queue = state.chain_pending[dir] ++ [entry]
    state = put_in(state.chain_pending[dir], queue)

    if length(queue) >= state.batch_size do
      state = chain_cancel_timer(state, dir)
      {:noreply, do_chain_flush(dir, state)}
    else
      {:noreply, chain_ensure_timer(state, dir)}
    end
  end

  @impl true
  def handle_info({:flush, dir}, state) do
    state = put_in(state.timers[dir], nil)

    if state.pending[dir] != [] do
      {:noreply, do_flush(dir, state)}
    else
      {:noreply, state}
    end
  end

  def handle_info({:chain_flush, dir}, state) do
    state = put_in(state.chain_timers[dir], nil)

    if state.chain_pending[dir] != [] do
      {:noreply, do_chain_flush(dir, state)}
    else
      {:noreply, state}
    end
  end

  ## Internals

  defp ensure_timer(state, dir) do
    case state.timers[dir] do
      nil ->
        ref = Process.send_after(self(), {:flush, dir}, state.flush_ms)
        put_in(state.timers[dir], ref)

      _ref ->
        state
    end
  end

  defp cancel_timer(state, dir) do
    case state.timers[dir] do
      nil ->
        state

      ref ->
        Process.cancel_timer(ref)
        put_in(state.timers[dir], nil)
    end
  end

  defp do_flush(dir, state) do
    queue = state.pending[dir]
    state = put_in(state.pending[dir], [])

    dir_sign = if dir == :forward, do: 1, else: -1

    # Partition by (K, |eps|). Independent samplers at different
    # warmup states naturally produce heterogeneous K/eps within
    # a single flush window; the batched leapfrog can only fuse
    # instances that share both. Group, dispatch each group
    # separately. Groups of size 1 still dispatch (no batching
    # win, but correct), and the common synchronized-dispatch
    # case (benchmark, vectorized sampling) collapses to a single
    # group of size = queue length.
    groups = Enum.group_by(queue, fn {_, _, _, _, _, eps, k} -> {k, eps} end)

    Enum.reduce(groups, state, fn {{k0, eps_abs0}, group_queue}, acc ->
      do_flush_group(group_queue, k0, eps_abs0, dir_sign, acc)
    end)
  end

  defp do_flush_group(queue, k0, eps_abs0, dir_sign, state) do
    instances =
      Enum.map(queue, fn {_from, q, p, im, obs, _eps, _k} -> {q, p, im, obs} end)

    n_instances = length(queue)

    # USDT probe — flush event. Lets DTrace consumers track
    # batch_size_actual vs batch_size_target (partial-flush detection
    # for the #159 go/no-go decision).
    :dyntrace.p(
      n_instances,
      state.batch_size,
      k0,
      dir_sign,
      "coord_flush",
      "",
      "",
      ""
    )

    dispatch = fn ->
      Dispatch.chain_batch(state.meta, instances, k0, dir_sign, eps_abs0)
    end

    t0 = :erlang.monotonic_time(:microsecond)

    results =
      if state.use_gpu_scheduler do
        GPUScheduler.run(fn _device -> dispatch.() end)
      else
        dispatch.()
      end

    dispatch_us = :erlang.monotonic_time(:microsecond) - t0

    # USDT probe — dispatch latency. Quantizable in DTrace for
    # per-instance and tail-latency views.
    :dyntrace.p(
      n_instances,
      k0,
      dispatch_us,
      0,
      "vk_dispatch",
      "",
      "",
      ""
    )

    queue
    |> Enum.zip(results)
    |> Enum.each(fn {{from, _, _, _, _, _, _}, result} ->
      GenServer.reply(from, result)
    end)

    %{
      state
      | stats: %{
          batches_fired: state.stats.batches_fired + 1,
          requests_served: state.stats.requests_served + n_instances
        }
    }
  end

  # ===== Task #171 Step 2: chain-shader dispatch path =====

  defp chain_ensure_timer(state, dir) do
    case state.chain_timers[dir] do
      nil ->
        ref = Process.send_after(self(), {:chain_flush, dir}, state.flush_ms)
        put_in(state.chain_timers[dir], ref)

      _ref ->
        state
    end
  end

  defp chain_cancel_timer(state, dir) do
    case state.chain_timers[dir] do
      nil ->
        state

      ref ->
        Process.cancel_timer(ref)
        put_in(state.chain_timers[dir], nil)
    end
  end

  defp do_chain_flush(dir, state) do
    queue = state.chain_pending[dir]
    state = put_in(state.chain_pending[dir], [])
    dir_sign = if dir == :forward, do: 1, else: -1

    # Partition by (meta-hash, K, |eps|). Different synthesised SPVs
    # can't share a batched dispatch (different shader binaries);
    # within the same meta, K and |eps| must also agree because the
    # batched chain shader is parameterized on a single K + signed eps
    # for the whole workgroup. The meta hash uses phash2 over the full
    # meta tuple — including spv_path — so two callers with identical
    # synthesised shaders batch together cleanly.
    groups =
      Enum.group_by(queue, fn {_from, meta, _q, _p, _im, _obs, eps, k} ->
        {:erlang.phash2(meta), k, eps}
      end)

    Enum.reduce(groups, state, fn {_partition_key, group_queue}, acc ->
      do_chain_flush_group(group_queue, dir_sign, acc)
    end)
  end

  defp do_chain_flush_group(queue, dir_sign, state) do
    [{_from, meta0, _q, _p, _im, _obs, eps_abs0, k0} | _] = queue

    instances =
      Enum.map(queue, fn {_from, _meta, q, p, im, obs, _eps, _k} -> {q, p, im, obs} end)

    n_instances = length(queue)

    # USDT probe — chain flush event. Distinct from coord_flush so
    # the dtrace harness can tell which pathway is firing.
    :dyntrace.p(
      n_instances,
      state.batch_size,
      k0,
      dir_sign,
      "coord_chain_flush",
      "",
      "",
      ""
    )

    # Dispatch.chain_batch takes the unsigned epsilon and computes
    # `dir_sign * epsilon` internally (chain_batch line 287). Passing
    # `dir_sign * eps_abs0` here would double-sign and silently invert
    # backward dispatches.
    dispatch = fn ->
      Dispatch.chain_batch(meta0, instances, k0, dir_sign, eps_abs0)
    end

    t0 = :erlang.monotonic_time(:microsecond)

    results =
      try do
        if state.use_gpu_scheduler do
          GPUScheduler.run(fn _device -> dispatch.() end)
        else
          dispatch.()
        end
      rescue
        e ->
          # Dispatch crashed — reply :fallback to every caller so they
          # retry via route_chain_direct. The coord process itself
          # survives.
          reason = {:dispatch_raise, Exception.message(e)}
          Enum.each(queue, fn {from, _, _, _, _, _, _, _} ->
            GenServer.reply(from, {:fallback, reason})
          end)
          :crashed
      end

    dispatch_us = :erlang.monotonic_time(:microsecond) - t0

    case results do
      :crashed ->
        state

      results when is_list(results) and length(results) == n_instances ->
        :dyntrace.p(
          n_instances,
          k0,
          dispatch_us,
          0,
          "vk_chain_dispatch",
          "",
          "",
          ""
        )

        queue
        |> Enum.zip(results)
        |> Enum.each(fn {{from, _, _, _, _, _, _, _}, result} ->
          GenServer.reply(from, result)
        end)

        %{
          state
          | stats: %{
              batches_fired: state.stats.batches_fired + 1,
              requests_served: state.stats.requests_served + n_instances
            }
        }

      other ->
        # Unexpected return shape — reply :fallback to every caller
        # rather than throw a match error that kills the coord.
        Enum.each(queue, fn {from, _, _, _, _, _, _, _} ->
          GenServer.reply(from, {:fallback, {:bad_result_shape, other}})
        end)

        state
    end
  end
end
