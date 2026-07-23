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

  @dispatch_count_key :exmc_chain_dispatches

  @doc """
  Read the per-process chain dispatch counter. Increments on every
  `chain/8` and `chain_batch/5,6` call within the calling process.
  Use with `reset_dispatch_count/0` to bound a measurement window.
  """
  def dispatch_count, do: Process.get(@dispatch_count_key) || 0

  @doc "Zero the per-process chain dispatch counter."
  def reset_dispatch_count do
    Process.put(@dispatch_count_key, 0)
    :ok
  end

  defp record_dispatch! do
    Process.put(@dispatch_count_key, (Process.get(@dispatch_count_key) || 0) + 1)
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
    do_chain(meta, d, epsilon, inv_mass, q, p, k, dir_sign)
  end

  # Synthesised chain shader dispatch. Meta tuple produced by
  # `Exmc.NUTS.CustomSynth.synthesise/1`. All models route through
  # the f64 synth path — the legacy f32 family SPVs are removed.
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
      :erlang.phash2(q_bin), :erlang.phash2(p_bin),
      k, d,
      "vk_leap_in", "", "", ""
    )

    {:ok, {q_chain_bin, p_chain_bin, grad_chain_bin, logp_chain_bin}} =
      Nx.Vulkan.NativeV.leapfrog_chain_synth_f64(
        q_bin, p_bin, extras_bin, push, k, spv_path
      )

    Exmc.Dyntrace.p(
      :erlang.phash2(q_chain_bin), :erlang.phash2(p_chain_bin),
      :erlang.phash2(grad_chain_bin), :erlang.phash2(logp_chain_bin),
      "vk_leap_out", "", "", ""
    )

    bins_to_chain_tensors(
      {q_chain_bin, p_chain_bin, grad_chain_bin, logp_chain_bin},
      k,
      d,
      :f64
    )
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
  """
  def chain_batch(
        {:synthesised, _sha, _layout, push_spec, spv_path, _empty_obs},
        instances,
        k,
        dir_sign,
        epsilon \\ nil
      )
      when is_list(instances) and length(instances) > 0 do
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
      |> Enum.flat_map(&prior_param_floats/1)
      |> Enum.reduce(<<>>, fn f, acc ->
        acc <> <<f * 1.0::little-float-64>>
      end)

    push = header <> prior_bin

    # Pack inputs: instance-contiguous layout (f64)
    {q_bin, p_bin, extras_bin} =
      Enum.reduce(instances, {<<>>, <<>>, <<>>}, fn {q, p, inv_mass, obs},
                                                    {qa, pa, ea} ->
        q_b = q |> Nx.as_type(:f64) |> Nx.to_binary()
        p_b = p |> Nx.as_type(:f64) |> Nx.to_binary()
        obs_b = obs |> Nx.as_type(:f64) |> Nx.to_binary()
        inv_mass_b = inv_mass |> Nx.as_type(:f64) |> Nx.to_binary()
        # extras layout per instance: obs[0..n_obs-1] then inv_mass[0..d-1]
        {qa <> q_b, pa <> p_b, ea <> obs_b <> inv_mass_b}
      end)

    {:ok, {q_chain_bin, p_chain_bin, grad_chain_bin, logp_chain_bin}} =
      Nx.Vulkan.NativeV.leapfrog_chain_synth_batch_f64(
        q_bin, p_bin, extras_bin, push, k, spv_path
      )

    # Unpack per-instance slices
    chain_bytes_per_instance = k * d * 8
    logp_bytes_per_instance = k * 8

    for i <- 0..(n_instances - 1) do
      q_slice = binary_part(q_chain_bin, i * chain_bytes_per_instance, chain_bytes_per_instance)
      p_slice = binary_part(p_chain_bin, i * chain_bytes_per_instance, chain_bytes_per_instance)
      grad_slice = binary_part(grad_chain_bin, i * chain_bytes_per_instance, chain_bytes_per_instance)
      logp_slice = binary_part(logp_chain_bin, i * logp_bytes_per_instance, logp_bytes_per_instance)

      bins_to_chain_tensors({q_slice, p_slice, grad_slice, logp_slice}, k, d, :f64)
    end
  end

  # Mirrors Push.prior_param_floats/1 (kept private there). Walking
  # priors here keeps chain_batch self-contained.
  defp prior_param_floats({_id, Exmc.Dist.Normal, params}),
    do: [scalar(params, :mu), scalar(params, :sigma)]

  defp prior_param_floats({_id, Exmc.Dist.HalfCauchy, params}),
    do: [scalar(params, :scale)]

  defp prior_param_floats({_id, Exmc.Dist.HalfNormal, params}),
    do: [scalar(params, :sigma)]

  defp prior_param_floats({_id, Exmc.Dist.Exponential, params}),
    do: [scalar(params, :lambda)]

  defp prior_param_floats({id, mod, _}),
    do: raise("chain_batch: no prior_param_floats clause for #{id} (#{inspect(mod)})")

  defp scalar(params, key) do
    case Map.fetch!(params, key) do
      v when is_number(v) -> v * 1.0
      %Nx.Tensor{} = t -> Nx.to_number(t) * 1.0
    end
  end

  defp bin_to_tensor(bin, shape) do
    bin
    |> Nx.from_binary(:f64, backend: Nx.BinaryBackend)
    |> Nx.reshape(shape)
  end
end
