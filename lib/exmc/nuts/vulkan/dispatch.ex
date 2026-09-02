defmodule Exmc.NUTS.Vulkan.Dispatch do
  @moduledoc """
  Pure-function dispatch path for synthesised chain shaders.

  Extracted from `Exmc.NUTS.Tree.do_dispatch/10` so the generic
  `Nx.Vulkan.Node` has a single MCMC-shaped target to wrap. tree.ex
  routes here directly when the GPU node isn't started, or wraps a
  call to `chain/8` in `Nx.Vulkan.Node.with_node/2` when it is.

  The vulkano path allocates fresh buffers per dispatch (bytes-in /
  bytes-out), eliminating the stale-handle class of bugs from the
  legacy persistent-buffer approach.

  ## Dispatch counter (chain-shader coverage harness)

  Every call to `chain/8` and `chain_batch/5,6` records a tick in the
  per-process counter `:exmc_chain_dispatches`. Test harnesses use
  this to detect silent fallbacks — when sampling code expects the
  chain shader to fire but a config like `compiler=:none` quietly
  routes through `BinaryBackend.Evaluator` instead, the counter
  stays at zero and the test can fail loud.

  Helpers: `dispatch_count/0` returns the current count;
  `reset_dispatch_count/0` zeros it. Zero overhead when no caller
  reads it — the counter is a single process-dict entry.

  ## Precision

  Everything runs at f64. The synthesised chain path calls
  `Nx.Vulkan.NativeV.leapfrog_chain_synth_f64`, packing all buffers
  and push constants at f64 width. The legacy f32 family SPV shaders
  and the spirit (C++) backend have been removed.
  """

  alias Exmc.NUTS.CustomSynth.Push

  @dispatch_count_key :exmc_chain_dispatches
  @dispatch_micros_key :exmc_chain_dispatch_micros

  @doc """
  Read the per-process chain dispatch counter. Increments on every
  `chain/8` and `chain_batch/5,6` call within the calling process.
  Use with `reset_dispatch_count/0` to bound a measurement window.
  """
  def dispatch_count, do: Process.get(@dispatch_count_key) || 0

  @doc """
  Microseconds spent inside `chain/8` in this process since the last reset.

  Pairs with `dispatch_count/0` to split a sampling run into time spent in
  the GPU call and time spent everywhere else, **as a measurement rather than
  a subtraction**. The earlier decomposition of a run into ~1.2 ms GPU,
  ~1.0 ms in-NIF and ~1.9 ms tree logic came from subtracting a benchmark
  median from a wall-clock average, and a subtraction is where an upstream
  per-fence estimate went wrong by 3x.

  Costs two `monotonic_time` calls per dispatch — tens of nanoseconds against
  a dispatch that measures ~170 us on Kepler and ~2225 us on Tegra, so it is
  left always-on rather than gated behind a flag that would then need its own
  test to prove it was enabled.
  """
  def dispatch_micros, do: Process.get(@dispatch_micros_key) || 0

  @doc "Zero the per-process chain dispatch counter and timer."
  def reset_dispatch_count do
    Process.put(@dispatch_count_key, 0)
    Process.put(@dispatch_micros_key, 0)
    :ok
  end

  defp record_dispatch! do
    Process.put(@dispatch_count_key, (Process.get(@dispatch_count_key) || 0) + 1)
  end

  defp record_micros!(us) do
    Process.put(@dispatch_micros_key, (Process.get(@dispatch_micros_key) || 0) + us)
  end

  @doc """
  Dispatch one chain step. Returns `{q_chain_t, p_chain_t, logp_chain_t, grad_chain_t}`
  matching the historical tuple order from `Tree.chain_to_tensors/3`.

  - `meta` is the synthesised meta tuple from `CustomSynth.synthesise/1`.
  - `d` is the dimension; `epsilon` is the step size.
  - `inv_mass` is the inverse mass diagonal (Nx tensor).
  - `q`, `p` are the current position and momentum (Nx tensors).
  - `k` is the number of leapfrog steps in the chain (typically 32).
  - `dir_sign` is +1 or -1.

  Records a dispatch tick (see module doc) and delegates to the
  f64 vulkano synth path.
  """
  def chain(meta, d, epsilon, inv_mass, q, p, k, dir_sign) do
    record_dispatch!()
    t0 = :erlang.monotonic_time(:microsecond)

    try do
      do_chain(meta, d, epsilon, inv_mass, q, p, k, dir_sign)
    after
      record_micros!(:erlang.monotonic_time(:microsecond) - t0)
    end
  end

  # Synthesised chain shader dispatch. Meta tuple produced by
  # `Exmc.NUTS.CustomSynth.synthesise/1`. All models route through
  # the f64 synth path — the legacy f32 family SPVs are removed.
  #
  # On the `d <= 256` guard below: 256 is the SHADER's limit — the chain
  # templates declare `local_size_x = 256` with `shared double q_shared[256]`,
  # one thread per free RV — and it is now the binding one.
  #
  # It used to be described here as "almost never the one that bites", with
  # the 128-byte push block called authoritative at 13 prior floats. That was
  # backwards. `Push.pack/1` emits the header alone; the prior floats it once
  # appended were baked into the shader as literals and read by nothing, while
  # still counting against the NIF's `push.len() > 128` check. Removing them
  # took an 8-RV model from 0 chain dispatches to 2564.
  defp do_chain(
         {:synthesised, _sha, _layout, _push_spec, _spv_path, _obs_bin} = meta,
         d,
         epsilon,
         inv_mass,
         q,
         p,
         k,
         dir_sign
       )
       when is_integer(d) and d <= 256 do
    chain_synth_vulkano(meta, d, epsilon, inv_mass, q, p, k, dir_sign)
  end

  # Vulkano backend (Nx.Vulkan.NativeV): bytes-in / bytes-out at f64.
  # Allocates fresh buffers per dispatch. No tensor refs to keep alive
  # across calls — eliminates the stale-handle class of bugs.
  defp chain_synth_vulkano(
         {:synthesised, _sha, _layout, push_spec, spv_path, obs_bin},
         d,
         epsilon,
         inv_mass,
         q,
         p,
         k,
         dir_sign
       ) do
    signed_eps = dir_sign * epsilon

    {:ok, push, _bytes} =
      Exmc.NUTS.CustomSynth.Push.pack(%{push_spec | eps: signed_eps, K: k})

    q_bin = q |> Nx.as_type(:f64) |> Nx.to_binary()
    p_bin = p |> Nx.as_type(:f64) |> Nx.to_binary()
    inv_mass_bin = inv_mass |> Nx.as_type(:f64) |> Nx.to_binary()
    extras_bin = obs_bin <> inv_mass_bin

    # USDT probe (no-op unless BEAM built with --with-dynamic-trace).
    # Tag "vk_leap_in": entry-side hash of q_bin xor p_bin so a DTrace
    # consumer can pair this with the cpu_leap_in probe emitted by
    # Leapfrog.step.
    Exmc.Dyntrace.p(
      :erlang.phash2(q_bin),
      :erlang.phash2(p_bin),
      k,
      d,
      "vk_leap_in",
      "",
      "",
      ""
    )

    # Retry-on-error rather than stat-before-dispatch: checking File.exists?/1
    # on every call would cost a syscall on the hot path (~2600 dispatches in
    # a 60-sample run) to guard against a condition that is rare. The recovery
    # path costs nothing when the artifact is present.
    {:ok, {q_chain_bin, p_chain_bin, grad_chain_bin, logp_chain_bin}} =
      case leapfrog_f64(q_bin, p_bin, extras_bin, push, k, spv_path) do
        {:error, :dispatch_failed, msg} = err ->
          if is_binary(msg) and msg =~ "read spv" do
            case Exmc.NUTS.CustomSynth.Compile.ensure!(spv_path) do
              :ok ->
                leapfrog_f64(q_bin, p_bin, extras_bin, push, k, spv_path)

              {:error, reason} ->
                raise """
                chain dispatch failed: the compiled shader at

                    #{spv_path}

                could not be read and could not be rebuilt (#{inspect(reason)}).

                The artifact is content-addressed, so it is normally rebuilt
                from remembered GLSL. Recovery fails when synthesis happened in
                a different VM, so nothing in this one remembers the source.

                Re-run to re-synthesise. If it recurs, something is deleting
                #{Path.dirname(spv_path)} while the suite runs.
                """
            end
          else
            err
          end

        other ->
          other
      end

    Exmc.Dyntrace.p(
      :erlang.phash2(q_chain_bin),
      :erlang.phash2(p_chain_bin),
      :erlang.phash2(grad_chain_bin),
      :erlang.phash2(logp_chain_bin),
      "vk_leap_out",
      "",
      "",
      ""
    )

    bins_to_chain_tensors(
      {q_chain_bin, p_chain_bin, grad_chain_bin, logp_chain_bin},
      k,
      d,
      :f64
    )
  end

  defp leapfrog_f64(q_bin, p_bin, extras_bin, push, k, spv_path) do
    Nx.Vulkan.NativeV.leapfrog_chain_synth_f64(q_bin, p_bin, extras_bin, push, k, spv_path)
  end

  defp bins_to_chain_tensors({q_b, p_b, grad_b, logp_b}, k, d, _wire_type) do
    qd_shape = {k, d}

    {
      bin_to_tensor(q_b, qd_shape),
      bin_to_tensor(p_b, qd_shape),
      bin_to_tensor(logp_b, {k}),
      bin_to_tensor(grad_b, qd_shape)
    }
  end

  @doc """
  Task #154 Phase 3 — batched multi-instrument dispatch.

  `meta` must be the BATCHED synth meta from
  `Exmc.NUTS.CustomSynth.synthesise_batched/1` (different SPV from the
  single-instance synthesise — the batched shader uses
  `gl_WorkGroupID.x` for per-instance buffer offsets).

  `instances` is a list of `{q, p, inv_mass, obs}` tuples — each tensor
  in the per-instance natural shape (q, p, inv_mass = {d}; obs = {n_obs}).
  All instances must have the same `d` and `n_obs`.

  Returns a list of `{q_chain, p_chain, logp_chain, grad_chain}` tuples,
  one per input instance, in the same order.

  One `vkQueueSubmit` call regardless of N — that's the entire point of
  batching. Dispatch overhead amortizes across N independent inferences.

  ## Requires an f64 batch NIF that does not exist yet

  This function needs `Nx.Vulkan.NativeV.leapfrog_chain_synth_batch_f64/6`
  and raises without it — see `ensure_batch_nif!/0`. Every caller must
  therefore be prepared for a raise; `BatchCoordinator` converts it to
  `{:fallback, {:dispatch_raise, _}}` so sampling continues, unbatched,
  down the single-instance path.
  """
  def chain_batch(
        {:synthesised, _sha, _layout, push_spec, spv_path, _empty_obs},
        instances,
        k,
        dir_sign,
        epsilon \\ nil
      )
      when is_list(instances) and length(instances) > 0 do
    ensure_batch_nif!()
    record_dispatch!()
    n_instances = length(instances)
    [{q0, _, _, _} | _] = instances
    d = elem(Nx.shape(q0), 0)
    n_obs = push_spec.n_obs

    # Mirror chain_synth_vulkano: caller-provided epsilon overrides push_spec.eps.
    # If nil (legacy call), fall back to push_spec.eps.
    eps_used = epsilon || push_spec.eps
    signed_eps = dir_sign * eps_used

    # Batched push: K(4) + n_obs(4) + d(4) + n_instances(4) + eps(8) bytes
    # header. Prior floats follow (same as single-instance Push.pack).
    header =
      <<
        k::little-unsigned-32,
        n_obs::little-unsigned-32,
        d::little-unsigned-32,
        n_instances::little-unsigned-32,
        signed_eps::little-float-64
      >>

    prior_bin =
      push_spec.priors
      |> Enum.flat_map(&Push.prior_param_floats/1)
      |> Enum.reduce(<<>>, fn f, acc ->
        acc <> <<f * 1.0::little-float-64>>
      end)

    # D2: the batched header is not Push.pack/1's, so the 128-byte cap has to
    # be applied explicitly. Raising is what the coordinator's try/rescue turns
    # into {:fallback, _}; without it an oversized block reached the NIF and
    # came back as a MatchError on {:error, :bad_input} naming nothing.
    push = Push.ensure_fits!(header <> prior_bin, "chain_batch/5")

    # Pack inputs: instance-contiguous layout (f64)
    {q_bin, p_bin, extras_bin} =
      Enum.reduce(instances, {<<>>, <<>>, <<>>}, fn {q, p, inv_mass, obs}, {qa, pa, ea} ->
        q_b = q |> Nx.as_type(:f64) |> Nx.to_binary()
        p_b = p |> Nx.as_type(:f64) |> Nx.to_binary()
        obs_b = obs |> Nx.as_type(:f64) |> Nx.to_binary()
        inv_mass_b = inv_mass |> Nx.as_type(:f64) |> Nx.to_binary()
        # extras layout per instance: obs[0..n_obs-1] then inv_mass[0..d-1]
        {qa <> q_b, pa <> p_b, ea <> obs_b <> inv_mass_b}
      end)

    # `apply/3` rather than a direct call: the function does not exist, so a
    # literal call is a compile-time warning that has been ignored on every
    # build, and dialyzer reports it as a call to a missing function. This
    # form starts working the day the NIF lands, with no edit here.
    {:ok, {q_chain_bin, p_chain_bin, grad_chain_bin, logp_chain_bin}} =
      apply(Nx.Vulkan.NativeV, :leapfrog_chain_synth_batch_f64, [
        q_bin,
        p_bin,
        extras_bin,
        push,
        k,
        spv_path
      ])

    # Unpack per-instance slices
    chain_bytes_per_instance = k * d * 8
    logp_bytes_per_instance = k * 8

    for i <- 0..(n_instances - 1) do
      q_slice = binary_part(q_chain_bin, i * chain_bytes_per_instance, chain_bytes_per_instance)
      p_slice = binary_part(p_chain_bin, i * chain_bytes_per_instance, chain_bytes_per_instance)

      grad_slice =
        binary_part(grad_chain_bin, i * chain_bytes_per_instance, chain_bytes_per_instance)

      logp_slice =
        binary_part(logp_chain_bin, i * logp_bytes_per_instance, logp_bytes_per_instance)

      bins_to_chain_tensors({q_slice, p_slice, grad_slice, logp_slice}, k, d, :f64)
    end
  end

  # `leapfrog_chain_synth_batch_f64/6` has never existed. Not at the pinned
  # ref in mix.lock, and not at nx_vulkan HEAD — checked 2026-08-28, 78
  # commits ahead of the pin. The dep exports exactly three chain NIFs:
  #
  #     leapfrog_chain_synth/6        f32, single instance
  #     leapfrog_chain_synth_f64/6    f64, single instance — the live path
  #     leapfrog_chain_synth_batch/6  f32, batched
  #
  # The f32 batch NIF is not a substitute. It writes f32 chains
  # (`n_instances * K * d * 4` bytes) while everything in `chain_batch` packs
  # and slices f64 at 8 bytes per element, so routing to it would return
  # numerically plausible garbage — the one outcome worse than this raise.
  #
  # Until then `chain_batch/5` cannot run, and says so here rather than
  # surfacing as a bare UndefinedFunctionError inside a rescue that reports
  # it as a generic dispatch failure.
  defp ensure_batch_nif! do
    mod = Nx.Vulkan.NativeV

    unless Code.ensure_loaded?(mod) and
             function_exported?(mod, :leapfrog_chain_synth_batch_f64, 6) do
      raise """
      Batched chain dispatch is unavailable: \
      Nx.Vulkan.NativeV.leapfrog_chain_synth_batch_f64/6 is not exported.

      nx_vulkan provides an f32 batch NIF (leapfrog_chain_synth_batch/6) and \
      an f64 single-instance NIF (leapfrog_chain_synth_f64/6), but no f64 \
      batched variant. The f32 one is not interchangeable — chain_batch/5 \
      packs and unpacks f64.

      Sampling is unaffected: callers fall back to single-instance dispatch, \
      which loses the one-vkQueueSubmit-per-batch win but not correctness.
      """
    end
  end

  defp bin_to_tensor(bin, shape) do
    bin
    |> Nx.from_binary(:f64, backend: Nx.BinaryBackend)
    |> Nx.reshape(shape)
  end
end
