defmodule Exmc.Compiler do
  @moduledoc """
  Compiles IR into a differentiable logp function.

  The compiler pre-dispatches at build time: it walks all IR nodes once,
  producing a list of term closures that are pure Nx ops at runtime. This
  means the returned `logp_fn` and `vag_fn` are ready for `Nx.Defn.grad`.
  """

  alias Exmc.{IR, Rewrite, PointMap, Transform}

  # Sentinel for models without observation data.
  # A scalar constant — adds one trivial XLA parameter, compiles once.
  @data_sentinel Nx.tensor(0.0, type: :f32, backend: Nx.BinaryBackend)

  @doc """
  Compile an IR into a logp function and its PointMap.

  Returns `{logp_fn, point_map}` where `logp_fn :: flat_tensor -> scalar_logp`.
  """
  def compile(%IR{} = ir, opts \\ []) do
    {logp_fn, pm, _ncp_info} = do_compile(ir, opts)
    # Wrap 2-arg logp_fn into 1-arg for public API (data baked in)
    obs_data = ir.data || @data_sentinel
    wrapped = fn flat -> logp_fn.(flat, obs_data) end
    {wrapped, pm}
  end

  @doc """
  Compile an IR into a value-and-grad function and its PointMap.

  Returns `{vag_fn, point_map}` where `vag_fn :: flat_tensor -> {logp, grad}`.
  """
  def value_and_grad(%IR{} = ir) do
    {logp_fn, pm, _ncp_info} = do_compile(ir, [])
    {build_vag_fn(logp_fn, ir.data), pm}
  end

  @doc """
  Compile an IR for sampling: returns `{vag_fn, step_fn, pm, ncp_info}`.

  `step_fn` is a fused leapfrog step: `(q, p, grad, epsilon, inv_mass_diag) -> {q, p, logp, grad}`.
  When EXLA is available, the entire leapfrog + gradient is one JIT-compiled call.
  `ncp_info` maps NCP'd variable ids to their original `%{mu:, sigma:}` sources.
  """
  def compile_for_sampling(%IR{} = ir, opts \\ []) do
    {logp_fn, pm, ncp_info} = do_compile(ir, opts)
    obs_data = ir.data
    device = Keyword.get(opts, :device, :host)
    jit_opts = if device == :host, do: [], else: [client: device]
    vag_fn = build_vag_fn(logp_fn, obs_data, jit_opts)
    step_fn = build_step_fn(logp_fn, obs_data, jit_opts)

    multi_step_fn =
      if pm.size > 0, do: Exmc.NUTS.BatchedLeapfrog.build(logp_fn, obs_data, pm.size, jit_opts), else: nil

    # Phases A+B: detect a fused-leapfrog chain meta from the IR.
    # Single-RV models in 6 supported families (Normal/StudentT/
    # Cauchy/Exponential/HalfNormal/Weibull) get auto-routed to
    # the corresponding chain shader at dispatch time. Hierarchical
    # models, observed-data models, and unsupported distributions
    # return :unsupported and fall through to multi_step_fn.
    detect = Exmc.NUTS.ChainShaderCodegen.detect_meta(ir)

    chain_meta =
      case detect do
        {:ok, meta} -> meta
        _ -> nil
      end

    # Plan B' guard: vulkan compiler must have a synthesisable IR.
    # Without chain_meta the sampler would silently fall through to
    # per-op CPU via Evaluator + VulkanoBackend host-fallbacks. Fail
    # loud at compile time instead.
    #
    # Exception: `{:unsupported, :push_too_large}` means the model *is*
    # shape-synthesisable but its prior params exceed the 128-byte f64
    # push-constants block. The model is valid, just too wide for the fused
    # chain path, so degrade to per-op sampling (slower but correct) instead
    # of raising — this is what "fall back on push_too_large" means.
    #
    # Escape hatch for the generic case:
    # `config :exmc, :allow_vulkan_perop_sampling, true` logs a warning
    # instead of raising. Default OFF.
    if Exmc.JIT.detect_compiler() == Nx.Vulkan and is_nil(chain_meta) do
      case detect do
        {:unsupported, :push_too_large} ->
          require Logger

          Logger.warning(
            "[Compiler] push-constants overflow (>128 B at f64) — falling back to " <>
              "per-op vulkan sampling for this IR (slower but correct). Reduce the " <>
              "free-RV count or reshape to fit the fused f64 chain shader."
          )

        _ ->
          msg = """
          Vulkan compiler requires a synthesisable model (f64 chain shader path).
          ChainShaderCodegen.detect_meta/1 returned :unsupported for this IR.
          Either:
            - use `compiler: :exla` (if EXLA is available on this platform)
            - use `compiler: :none` (pure CPU, slower but correct)
            - reshape the model so CustomSynth can emit a fused f64 chain shader
              (standard-family priors with optional Custom likelihood, d <= 256)
          """

          if Application.get_env(:exmc, :allow_vulkan_perop_sampling, false) do
            require Logger
            Logger.warning("[Compiler] Plan B' guard bypassed: #{msg}")
          else
            raise Exmc.SynthUnsupportedError, ir: ir, message: msg
          end
      end
    end

    {vag_fn, step_fn, pm, ncp_info, multi_step_fn, chain_meta}
  end

  @doc """
  Compile an IR for pointwise observation log-likelihood.

  Returns `{pointwise_fn, pm}` where `pointwise_fn :: flat_tensor -> %{obs_id => scalar_logp}`.
  Used by ModelComparison for WAIC/LOO.
  """
  def compile_pointwise(%IR{} = ir) do
    ir = Rewrite.apply(ir)
    ir = ensure_binary_backend(ir)
    pm = PointMap.build(ir)
    ncp_info = ir.ncp_info || %{}
    obs_terms = build_pointwise_obs_terms(ir, pm, ncp_info)

    pointwise_fn =
      if pm.size == 0 do
        evaluated =
          Map.new(obs_terms, fn {obs_id, terms} ->
            {obs_id, eval_terms(terms, %{})}
          end)

        fn _flat -> evaluated end
      else
        fn flat ->
          vm = PointMap.unpack(flat, pm)

          Map.new(obs_terms, fn {obs_id, terms} ->
            {obs_id, eval_terms(terms, vm)}
          end)
        end
      end

    {pointwise_fn, pm}
  end

  # --- Internal compile pipeline ---

  defp do_compile(%IR{} = ir, opts) do
    ir = Rewrite.apply(ir, opts)
    ir = ensure_binary_backend(ir)
    pm = PointMap.build(ir)
    ncp_info = ir.ncp_info || %{}
    terms = build_terms(ir, pm, ncp_info)

    has_data? = ir.data != nil

    # logp_fn :: (flat, data) -> scalar
    # When IR has data, inject it into vm as "__obs_data" so Custom dist
    # string refs resolve naturally. Data flows as a JIT argument (not a
    # captured constant), so EXLA caches by shape only.
    logp_fn =
      if pm.size == 0 do
        constant = eval_terms(terms, %{})
        fn _flat, _data -> constant end
      else
        if has_data? do
          fn flat, data ->
            vm = PointMap.unpack(flat, pm)
            vm = Map.put(vm, "__obs_data", data)
            eval_terms(terms, vm)
          end
        else
          fn flat, _data ->
            vm = PointMap.unpack(flat, pm)
            eval_terms(terms, vm)
          end
        end
      end

    {logp_fn, pm, ncp_info}
  end

  defp build_vag_fn(logp_fn, obs_data, jit_opts \\ []) do
    raw = Exmc.JIT.jit(
      fn flat, data ->
        Nx.Defn.value_and_grad(flat, fn f -> logp_fn.(f, data) end)
      end,
      jit_opts
    )

    data = obs_data || @data_sentinel
    fn flat -> raw.(flat, data) end
  end

  defp build_step_fn(logp_fn, obs_data, jit_opts) do
    fp = Exmc.JIT.precision()

    raw =
      Exmc.JIT.jit(
        fn q, p, grad, eps, inv_mass, data ->
          two = Nx.tensor(2.0, type: fp, backend: Nx.BinaryBackend)
          half = Nx.tensor(0.5, type: fp, backend: Nx.BinaryBackend)
          half_eps = Nx.divide(eps, two)
          p_half = Nx.add(p, Nx.multiply(half_eps, grad))
          q_new = Nx.add(q, Nx.multiply(eps, Nx.multiply(inv_mass, p_half)))
          {logp_new, grad_new} = Nx.Defn.value_and_grad(q_new, fn f -> logp_fn.(f, data) end)
          p_new = Nx.add(p_half, Nx.multiply(half_eps, grad_new))
          logp_scalar = Nx.reshape(logp_new, {})
          # Compute joint_logp = logp - 0.5 * p^T M^{-1} p inside JIT
          ke = Nx.multiply(half, Nx.sum(Nx.multiply(p_new, Nx.multiply(inv_mass, p_new))))
          joint_logp = Nx.subtract(logp_scalar, ke)
          {q_new, p_new, logp_scalar, grad_new, Nx.reshape(joint_logp, {})}
        end,
        jit_opts
      )

    data = obs_data || @data_sentinel
    fn q, p, grad, epsilon, inv_mass_diag ->
      eps_t = Nx.tensor(epsilon, type: fp, backend: Nx.BinaryBackend)

      try do
        raw.(q, p, grad, eps_t, inv_mass_diag, data)
      rescue
        ArithmeticError ->
          # Divide-by-zero / overflow inside value_and_grad. On EXLA this
          # produces IEEE inf/nan that flow through and the divergence
          # check at the tree level handles them. On BinaryBackend the
          # Complex.divide path raises instead, which would crash the
          # whole sampler if not caught here. Return a divergent 5-tuple
          # — q/p/grad unchanged, joint_logp very negative so the delta
          # against joint_logp_0 trips the divergence threshold cleanly.
          neg_huge = Nx.tensor(-1.0e300, type: fp, backend: Nx.BinaryBackend)

          zero_grad =
            Nx.broadcast(
              Nx.tensor(0.0, type: fp, backend: Nx.BinaryBackend),
              Nx.shape(grad)
            )

          {q, p, neg_huge, zero_grad, neg_huge}
      end
    end
  end

  # --- Private: term generation ---

  defp build_terms(%IR{} = ir, %PointMap{} = pm, ncp_info) do
    ir.nodes
    |> Map.values()
    |> Enum.flat_map(&node_term(&1, ir, pm, ncp_info))
  end

  defp build_pointwise_obs_terms(%IR{} = ir, %PointMap{} = pm, ncp_info) do
    ir.nodes
    |> Map.values()
    |> Enum.reduce(%{}, fn node, acc ->
      if obs_node?(node) do
        terms = node_term(node, ir, pm, ncp_info)
        if terms != [], do: Map.put(acc, node.id, terms), else: acc
      else
        acc
      end
    end)
  end

  defp obs_node?(%{op: {:obs, _, _}}), do: true
  defp obs_node?(%{op: {:obs, _, _, _}}), do: true
  defp obs_node?(%{op: {:meas_obs, _, _, _}}), do: true
  defp obs_node?(%{op: {:meas_obs, _, _, _, _}}), do: true
  defp obs_node?(_), do: false

  # Free RV without transform
  defp node_term(%{id: id, op: {:rv, dist, params}}, _ir, pm, ncp_info) do
    if PointMap.has_entry?(pm, id) do
      # Eagerly pre-compute derived constants (e.g. MvNormal prec from cov)
      params = eager_prepare_params(dist, params)

      [
        fn vm ->
          resolved = resolve_params_constrained(params, vm, pm, ncp_info)
          # Sum across data dims so multivariate priors collapse to a
          # scalar joint logp. No-op for scalar v. Without this,
          # Nx.Defn.Grad fails on the vector-shaped output ("cannot
          # reshape {n} to {}") for any RV with shape > {}.
          Nx.sum(dist.logpdf(Map.fetch!(vm, id), resolved))
        end
      ]
    else
      []
    end
  end

  # Free RV with transform
  defp node_term(%{id: id, op: {:rv, dist, params, transform}}, _ir, pm, ncp_info) do
    if PointMap.has_entry?(pm, id) do
      # Eagerly pre-compute derived constants
      params = eager_prepare_params(dist, params)

      [
        fn vm ->
          resolved = resolve_params_constrained(params, vm, pm, ncp_info)
          z = Map.fetch!(vm, id)
          x = Transform.apply(transform, z)
          logp = dist.logpdf(x, resolved)
          jac = Transform.log_abs_det_jacobian(transform, z)
          Nx.sum(Nx.add(logp, jac))
        end
      ]
    else
      []
    end
  end

  # Obs without meta -> delegate to obs with empty meta
  defp node_term(%{op: {:obs, target_id, value}} = node, ir, pm, ncp_info) do
    node_term(%{node | op: {:obs, target_id, value, %{}}}, ir, pm, ncp_info)
  end

  # Obs with meta — eagerly compute constant logp, or defer if target has param refs
  defp node_term(%{op: {:obs, target_id, value, meta}}, ir, pm, ncp_info) do
    if Map.get(meta, :likelihood, true) == false do
      []
    else
      target_node = IR.get_node!(ir, target_id)

      if has_param_refs?(target_node) do
        deferred_obs_term(target_node, value, meta, pm, ncp_info)
      else
        eager_obs_term(target_node, value, meta)
      end
    end
  end

  # Meas obs without meta
  defp node_term(%{op: {:meas_obs, rv_id, value, op_info}} = node, ir, pm, ncp_info) do
    node_term(%{node | op: {:meas_obs, rv_id, value, op_info, %{}}}, ir, pm, ncp_info)
  end

  # Meas obs with meta — eagerly compute
  defp node_term(%{op: {:meas_obs, rv_id, value, op_info, meta}}, ir, _pm, _ncp_info) do
    rv_node = IR.get_node!(ir, rv_id)
    eager_meas_obs_term(rv_node, value, op_info, meta)
  end

  # Det and other nodes contribute nothing
  defp node_term(_node, _ir, _pm, _ncp_info), do: []

  # --- Eager obs computation (constant w.r.t. free RVs) ---

  # Censored obs (eager): use CDF-based likelihood instead of logpdf
  defp eager_obs_term(%{op: {:rv, dist, params}}, value, %{censored: censor_type} = meta) do
    logp = Exmc.Dist.Censored.log_likelihood(censor_type, value, dist, params)
    [fn _vm -> apply_obs_meta(logp, Map.delete(meta, :censored)) end]
  end

  defp eager_obs_term(%{op: {:rv, dist, params}}, value, meta) do
    params = eager_prepare_params(dist, params)
    logp = dist.logpdf(value, params)
    [fn _vm -> apply_obs_meta(logp, meta) end]
  end

  defp eager_obs_term(%{op: {:rv, dist, params, transform}}, value, meta) do
    z = inverse_transform(transform, value)
    x = Transform.apply(transform, z)
    logp = dist.logpdf(x, params)
    jac = Transform.log_abs_det_jacobian(transform, z)
    combined = Nx.add(logp, jac)
    [fn _vm -> apply_obs_meta(combined, meta) end]
  end

  defp eager_obs_term(_target, _value, _meta), do: []

  # --- Deferred obs computation (depends on free parent RVs) ---

  # Censored obs (deferred): use CDF-based likelihood with resolved params
  defp deferred_obs_term(
         %{op: {:rv, dist, params}},
         value,
         %{censored: censor_type} = meta,
         pm,
         ncp_info
       ) do
    [
      fn vm ->
        resolved = resolve_params_constrained(params, vm, pm, ncp_info)
        logp = Exmc.Dist.Censored.log_likelihood(censor_type, value, dist, resolved)
        apply_obs_meta(logp, Map.delete(meta, :censored))
      end
    ]
  end

  defp deferred_obs_term(%{op: {:rv, dist, params}}, value, meta, pm, ncp_info) do
    [
      fn vm ->
        resolved = resolve_params_constrained(params, vm, pm, ncp_info)
        logp = dist.logpdf(value, resolved)
        apply_obs_meta(logp, meta)
      end
    ]
  end

  defp deferred_obs_term(%{op: {:rv, dist, params, transform}}, value, meta, pm, ncp_info) do
    [
      fn vm ->
        resolved = resolve_params_constrained(params, vm, pm, ncp_info)
        z = inverse_transform(transform, value)
        x = Transform.apply(transform, z)
        logp = dist.logpdf(x, resolved)
        jac = Transform.log_abs_det_jacobian(transform, z)
        apply_obs_meta(Nx.add(logp, jac), meta)
      end
    ]
  end

  defp deferred_obs_term(_target, _value, _meta, _pm, _ncp_info), do: []

  # --- Eager meas_obs computation ---

  defp eager_meas_obs_term(%{op: {:rv, dist, params}}, value, {:matmul, a}, meta) do
    x = jit_solve(a, value)
    logp = dist.logpdf(x, params)
    jac = Nx.negate(Nx.log(Nx.abs(jit_determinant(a))))
    combined = Nx.add(logp, jac)
    [fn _vm -> apply_obs_meta(combined, meta) end]
  end

  defp eager_meas_obs_term(%{op: {:rv, dist, params, transform}}, value, {:matmul, a}, meta) do
    x = jit_solve(a, value)
    z = inverse_transform(transform, x)
    x2 = Transform.apply(transform, z)
    logp = dist.logpdf(x2, params)
    jac = Transform.log_abs_det_jacobian(transform, z)
    meas_jac = Nx.negate(Nx.log(Nx.abs(jit_determinant(a))))
    combined = Nx.add(Nx.add(logp, jac), meas_jac)
    [fn _vm -> apply_obs_meta(combined, meta) end]
  end

  defp eager_meas_obs_term(%{op: {:rv, dist, params}}, value, {:affine, a, b}, meta) do
    a_t = to_tensor(a)
    b_t = to_tensor(b)
    x = Nx.divide(Nx.subtract(value, b_t), a_t)
    logp = dist.logpdf(x, params)
    jac = Nx.negate(Nx.log(Nx.abs(a_t)))
    combined = Nx.add(logp, jac)
    [fn _vm -> apply_obs_meta(combined, meta) end]
  end

  defp eager_meas_obs_term(%{op: {:rv, dist, params, transform}}, value, {:affine, a, b}, meta) do
    a_t = to_tensor(a)
    b_t = to_tensor(b)
    x = Nx.divide(Nx.subtract(value, b_t), a_t)
    z = inverse_transform(transform, x)
    x2 = Transform.apply(transform, z)
    logp = dist.logpdf(x2, params)
    jac = Transform.log_abs_det_jacobian(transform, z)
    meas_jac = Nx.negate(Nx.log(Nx.abs(a_t)))
    combined = Nx.add(Nx.add(logp, jac), meas_jac)
    [fn _vm -> apply_obs_meta(combined, meta) end]
  end

  defp eager_meas_obs_term(_rv_node, _value, _op_info, _meta), do: []

  # --- Term evaluation ---

  defp eval_terms([], _vm), do: Nx.tensor(0.0, backend: Nx.BinaryBackend)

  defp eval_terms(terms, vm) do
    terms
    |> Enum.map(fn term -> term.(vm) end)
    |> sum_logps()
  end

  defp sum_logps([one]), do: Nx.reshape(one, {})
  defp sum_logps(list), do: Enum.reduce(list, fn x, acc -> Nx.add(Nx.reshape(x, {}), acc) end)

  # --- Helpers ---

  defp apply_obs_meta(logp, meta) do
    weight = Map.get(meta, :weight, 1.0)
    weight_t = to_tensor(weight)
    logp = Nx.multiply(logp, weight_t)

    masked =
      case Map.get(meta, :mask) do
        nil -> logp
        mask -> Nx.select(to_tensor(mask), logp, Nx.tensor(0.0, backend: Nx.BinaryBackend))
      end

    case Map.get(meta, :reduce) do
      :sum -> Nx.sum(masked)
      :mean -> Nx.mean(masked)
      :logsumexp -> Nx.logsumexp(masked)
      _ -> masked
    end
  end

  defp inverse_transform(nil, x), do: x
  defp inverse_transform(:log, x), do: Nx.log(x)
  defp inverse_transform(:softplus, x) do
    # Numerically stable: log(expm1(x)) = x + log(1 - exp(-x)).
    # Avoids exp(x) overflow at large x (PATH_TO_FULL_PASS step 2a).
    Nx.add(x, Nx.log1p(Nx.negate(Nx.exp(Nx.negate(x)))))
  end
  defp inverse_transform(:logit, x), do: Nx.subtract(Nx.log(x), Nx.log1p(Nx.negate(x)))
  defp inverse_transform(:stick_breaking, x), do: Transform.inverse_stick_breaking(x)

  defp has_param_refs?(%{op: {:rv, _dist, params}}),
    do: Enum.any?(Map.values(params), &is_binary/1)

  defp has_param_refs?(%{op: {:rv, _dist, params, _transform}}),
    do: Enum.any?(Map.values(params), &is_binary/1)

  defp has_param_refs?(_), do: false

  # Resolve params with transforms and NCP reconstruction applied.
  # String refs return constrained values; NCP'd refs get reconstructed as mu + sigma * z.
  defp resolve_params_constrained(params, vm, pm, ncp_info) do
    Map.new(params, fn
      {k, v} when is_binary(v) ->
        {k, resolve_ref(v, vm, pm, ncp_info)}

      {k, v} ->
        {k, v}
    end)
  end

  defp resolve_ref(id, vm, pm, ncp_info) do
    case Map.get(ncp_info, id) do
      nil ->
        z = Map.fetch!(vm, id)
        transform = entry_transform(pm, id)
        Transform.apply(transform, z)

      %{mu: mu_src, sigma: sigma_src} ->
        z = Map.fetch!(vm, id)
        mu = resolve_value(mu_src, vm, pm, ncp_info)
        sigma = resolve_value(sigma_src, vm, pm, ncp_info)
        Nx.add(mu, Nx.multiply(sigma, z))
    end
  end

  defp resolve_value(v, vm, pm, ncp_info) when is_binary(v) do
    resolve_ref(v, vm, pm, ncp_info)
  end

  defp resolve_value(%Nx.Tensor{} = v, _vm, _pm, _ncp_info), do: v
  defp resolve_value(v, _vm, _pm, _ncp_info) when is_number(v), do: Nx.tensor(v)

  defp entry_transform(%PointMap{} = pm, id) do
    case Enum.find(pm.entries, fn e -> e.id == id end) do
      %{transform: t} -> t
      nil -> nil
    end
  end

  # Eagerly pre-compute derived params for distributions that need it
  # (e.g. MvNormal converts cov -> prec + log_det_cov to avoid LinAlg in traced code)
  defp eager_prepare_params(dist, params) do
    if function_exported?(dist, :prepare_params, 1) do
      dist.prepare_params(params)
    else
      params
    end
  end

  defp to_tensor(%Nx.Tensor{} = t), do: t
  defp to_tensor(v) when is_number(v) or is_boolean(v), do: Nx.tensor(v)

  # EXLA.jit tracing requires all captured tensors to be on BinaryBackend.
  # Tensors on EXLA.Backend cannot be traced — they're already device buffers.
  defp ensure_binary_backend(%IR{} = ir) do
    nodes =
      Map.new(ir.nodes, fn {id, node} ->
        {id, %{node | op: ensure_op_binary(node.op)}}
      end)

    %{ir | nodes: nodes}
  end

  defp ensure_op_binary({:rv, dist, params}) do
    {:rv, dist, copy_params(params)}
  end

  defp ensure_op_binary({:rv, dist, params, transform}) do
    {:rv, dist, copy_params(params), transform}
  end

  defp ensure_op_binary({:obs, target, value}) do
    {:obs, target, to_binary(value)}
  end

  defp ensure_op_binary({:obs, target, value, meta}) do
    {:obs, target, to_binary(value), copy_meta(meta)}
  end

  defp ensure_op_binary({:meas_obs, rv_id, value, op_info}) do
    {:meas_obs, rv_id, to_binary(value), copy_op_info(op_info)}
  end

  defp ensure_op_binary({:meas_obs, rv_id, value, op_info, meta}) do
    {:meas_obs, rv_id, to_binary(value), copy_op_info(op_info), copy_meta(meta)}
  end

  defp ensure_op_binary(other), do: other

  defp copy_params(params) do
    Map.new(params, fn
      {k, %Nx.Tensor{} = t} -> {k, to_binary(t)}
      {k, v} -> {k, v}
    end)
  end

  defp copy_meta(meta) do
    Map.new(meta, fn
      {k, %Nx.Tensor{} = t} -> {k, to_binary(t)}
      {k, v} -> {k, v}
    end)
  end

  defp copy_op_info({:matmul, a}), do: {:matmul, to_binary(a)}
  defp copy_op_info({:affine, a, b}), do: {:affine, to_binary(a), to_binary(b)}
  defp copy_op_info(other), do: other

  # JIT-wrap LinAlg ops to avoid Nx 0.10 BinaryBackend LU bug on small matrices
  defp jit_determinant(a) do
    Exmc.JIT.jit(fn x -> Nx.LinAlg.determinant(x) end).(a)
    |> Nx.backend_copy(Nx.BinaryBackend)
  end

  defp jit_solve(a, b) do
    Exmc.JIT.jit(fn {x, y} -> Nx.LinAlg.solve(x, y) end).({a, b})
    |> Nx.backend_copy(Nx.BinaryBackend)
  end

  defp to_binary(%Nx.Tensor{} = t) do
    t =
      case t.data do
        %Nx.BinaryBackend{} -> t
        _ -> Nx.backend_copy(t, Nx.BinaryBackend)
      end

    Exmc.JIT.ensure_precision(t)
  end

  defp to_binary(v), do: v
end
