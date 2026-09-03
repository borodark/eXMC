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

  @default_flush_ms 25

  # The compute scheduler is a seam, not a fixed module.
  #
  # This line used to be `alias Exmc.NUTS.Vulkan.Scheduler, as: GPUScheduler`
  # here and `alias Exmc.Trading.GPUScheduler` in the applications tree — and
  # that alias was the *only* difference between the two copies of this file.
  # Core code naming an application module is what makes a file forkable, so
  # the application supplies its own scheduler through config instead:
  #
  #     config :exmc, :gpu_scheduler, MyApp.GPUScheduler
  #
  # The contract is one function: `run/1`, taking a zero- or one-arity function
  # and returning whatever that function returns. `Exmc.NUTS.Vulkan.Scheduler`
  # is the default and the reference implementation; it degrades to direct
  # execution when it is not started, so the seam costs nothing when unused.
  @doc false
  def scheduler,
    do: Application.get_env(:exmc, :gpu_scheduler, Exmc.NUTS.Vulkan.Scheduler)

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
    Exmc.Dyntrace.put_tag("BatchCoord")

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
    {:reply, %{forward: length(state.pending.forward), backward: length(state.pending.backward)},
     state}
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
    Exmc.Dyntrace.p(
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
      try do
        if state.use_gpu_scheduler do
          scheduler().run(fn _device -> dispatch.() end)
        else
          dispatch.()
        end
      rescue
        e ->
          # Dispatch crashed — reply :fallback to every caller so they
          # retry unbatched. The coord process itself survives.
          #
          # Without this the raise propagated out of the flush — reached
          # from handle_call on a size flush and handle_info on a timer
          # flush — and killed the coordinator, taking every other
          # in-flight caller's reply with it: a GenServer.call that should
          # have degraded to a slower unbatched draw became an exit in the
          # sampler instead.
          #
          # `rescue` catches exceptions only. An `exit` out of the GPU
          # scheduler still kills the coord; do_chain_flush_group has the
          # same gap, and closing it is a separate decision at both sites.
          reason = {:dispatch_raise, Exception.message(e)}

          Enum.each(queue, fn {from, _, _, _, _, _, _} ->
            GenServer.reply(from, {:fallback, reason})
          end)

          :crashed
      end

    dispatch_us = :erlang.monotonic_time(:microsecond) - t0

    case results do
      :crashed ->
        state

      results when is_list(results) and length(results) == n_instances ->
        # USDT probe — dispatch latency. Quantizable in DTrace for
        # per-instance and tail-latency views.
        Exmc.Dyntrace.p(
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

      other ->
        # Unexpected return shape — reply :fallback to every caller rather
        # than let Enum.zip/2 truncate silently and leave the unmatched
        # callers blocked until their call timeout.
        Enum.each(queue, fn {from, _, _, _, _, _, _} ->
          GenServer.reply(from, {:fallback, {:bad_result_shape, other}})
        end)

        state
    end
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

  @doc false
  # Public for direct testing. K is deliberately absent — see do_chain_flush/2.
  def partition_key({_from, meta, _q, _p, _im, _obs, eps, _k}),
    do: {:erlang.phash2(meta), eps}

  defp do_chain_flush(dir, state) do
    queue = state.chain_pending[dir]
    state = put_in(state.chain_pending[dir], [])
    dir_sign = if dir == :forward, do: 1, else: -1

    # Partition by (meta-hash, |eps|). Different synthesised SPVs cannot share
    # a batched dispatch, and |eps| is a single push field for the whole
    # workgroup, so both must agree. The meta hash is phash2 over the full
    # meta tuple — including spv_path — so identical synthesised shaders batch
    # together cleanly, and `d` rides along inside it.
    #
    # **K is deliberately NOT in this key**, and it used to be. The comment
    # that put it there was right about the shader — a single K really does
    # parameterise the whole workgroup — and drew the wrong conclusion from
    # it, because padding to the deepest was available and partitioning was
    # not the only way to satisfy the constraint. It was written when no
    # batched f64 NIF existed, so it reasoned about a capability nobody could
    # exercise.
    #
    # Measured cost of getting this wrong: of 300 draws with 4 chains, only
    # **48 had all four at the same n_steps**. Keying on K would batch fully
    # in 16% of draws and fall back to singletons in the other 84% — the
    # machinery paid for and almost none of the benefit collected.
    #
    # Padding is nearly free at our depths. Upstream's K-sweep on mac-248 puts
    # a chain call at 91.3 us intercept against 2.5 us/step over K <= 16, so
    # 86% is fixed cost and the padding lands on the 14%. Ragged depths
    # measured 3.4-3.5x against 4.3x for uniform ones.
    groups =
      Enum.group_by(queue, &partition_key/1)

    Enum.reduce(groups, state, fn {_partition_key, group_queue}, acc ->
      do_chain_flush_group(group_queue, dir_sign, acc)
    end)
  end

  defp do_chain_flush_group(queue, dir_sign, state) do
    [{_from, meta0, _q, _p, _im, _obs, eps_abs0, _k} | _] = queue

    instances =
      Enum.map(queue, fn {_from, _meta, q, p, im, obs, _eps, _k} -> {q, p, im, obs} end)

    n_instances = length(queue)

    # Pad to the deepest request in the group. Every instance is dispatched
    # for k0 steps; callers that asked for fewer are sliced back below.
    k0 = queue |> Enum.map(fn {_f, _m, _q, _p, _im, _o, _e, k} -> k end) |> Enum.max()

    # USDT probe — chain flush event. Distinct from coord_flush so
    # the dtrace harness can tell which pathway is firing.
    Exmc.Dyntrace.p(
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
          scheduler().run(fn _device -> dispatch.() end)
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
        Exmc.Dyntrace.p(
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
        |> Enum.each(fn {{from, _, _, _, _, _, _, k_req}, result} ->
          GenServer.reply(from, trim_to_requested(result, k_req, k0))
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

  # A padded instance runs MORE leapfrog steps than its caller asked for. The
  # extra steps are computed from valid state — they are trajectory the sampler
  # never requested, not garbage — so handing them back would be a plausible
  # wrong posterior rather than an error. Trim every buffer to the requested
  # depth.
  #
  # This is sound because the prefix property is exact: a K=k0 dispatch's first
  # k_req steps are BIT-identical to a K=k_req dispatch, on all four output
  # buffers. That is pinned upstream (nx_vulkan cccbd71) rather than assumed
  # here, verified at k_req = 1, 3 and 5 and again through the batched path
  # with an instance sliced out of a padded group.
  #
  # q/p/grad are {K, d}; logp is {K}, one scalar per step, not per dimension.
  @doc false
  # Public for direct testing: handing back untrimmed buffers is a silent
  # wrong posterior, so it is worth asserting on rather than reaching only
  # through a GenServer that needs a working batch NIF.
  def trim_to_requested(result, k_req, k0)

  def trim_to_requested(result, k_req, k0) when k_req >= k0, do: result

  def trim_to_requested({q, p, logp, grad}, k_req, _k0) do
    {slice_steps(q, k_req), slice_steps(p, k_req), slice_steps(logp, k_req),
     slice_steps(grad, k_req)}
  end

  # Anything that is not the 4-tuple of tensors passes through untouched —
  # a {:fallback, _} or an error shape must not be reshaped by this.
  def trim_to_requested(other, _k_req, _k0), do: other

  defp slice_steps(t, k_req) do
    case Nx.shape(t) do
      {k, d} when k > k_req -> Nx.slice(t, [0, 0], [k_req, d])
      {k} when k > k_req -> Nx.slice(t, [0], [k_req])
      _ -> t
    end
  end
end
