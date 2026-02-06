defmodule Exmc.Compiler do
  @moduledoc """
  Compiles IR into a differentiable logp function.

  The compiler pre-dispatches at build time: it walks all IR nodes once,
  producing a list of term closures that are pure Nx ops at runtime. This
  means the returned `logp_fn` and `vag_fn` are ready for `Nx.Defn.grad`.
  """

  alias Exmc.{IR, Rewrite, PointMap, Transform}

  @doc """
  Compile an IR into a logp function and its PointMap.

  Returns `{logp_fn, point_map}` where `logp_fn :: flat_tensor -> scalar_logp`.
  """
  def compile(%IR{} = ir) do
    {logp_fn, pm, _ncp_info} = do_compile(ir)
    {logp_fn, pm}
  end

  @doc """
  Compile an IR into a value-and-grad function and its PointMap.

  Returns `{vag_fn, point_map}` where `vag_fn :: flat_tensor -> {logp, grad}`.
  """
  def value_and_grad(%IR{} = ir) do
    {logp_fn, pm} = compile(ir)
    {build_vag_fn(logp_fn), pm}
  end

  @doc """
  Compile an IR for sampling: returns `{vag_fn, step_fn, pm, ncp_info}`.

  `step_fn` is a fused leapfrog step: `(q, p, grad, epsilon, inv_mass_diag) -> {q, p, logp, grad}`.
  When EXLA is available, the entire leapfrog + gradient is one JIT-compiled call.
  `ncp_info` maps NCP'd variable ids to their original `%{mu:, sigma:}` sources.
  """
  def compile_for_sampling(%IR{} = ir) do
    {logp_fn, pm, ncp_info} = do_compile(ir)
    vag_fn = build_vag_fn(logp_fn)
    step_fn = build_step_fn(logp_fn, vag_fn)
    {vag_fn, step_fn, pm, ncp_info}
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
        evaluated = Map.new(obs_terms, fn {obs_id, terms} ->
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

  defp do_compile(%IR{} = ir) do
    ir = Rewrite.apply(ir)
    ir = ensure_binary_backend(ir)
    pm = PointMap.build(ir)
    ncp_info = ir.ncp_info || %{}
    terms = build_terms(ir, pm, ncp_info)

    logp_fn =
      if pm.size == 0 do
        constant = eval_terms(terms, %{})
        fn _flat -> constant end
      else
        fn flat ->
          vm = PointMap.unpack(flat, pm)
          eval_terms(terms, vm)
        end
      end

    {logp_fn, pm, ncp_info}
  end

  defp build_vag_fn(logp_fn) do
    if Code.ensure_loaded?(EXLA) do
      EXLA.jit(fn flat -> Nx.Defn.value_and_grad(flat, logp_fn) end)
    else
      fn flat -> Nx.Defn.value_and_grad(flat, logp_fn) end
    end
  end

  defp build_step_fn(logp_fn, vag_fn) do
    if Code.ensure_loaded?(EXLA) do
      jitted = EXLA.jit(fn q, p, grad, eps, inv_mass ->
        two = Nx.tensor(2.0, type: :f64, backend: Nx.BinaryBackend)
        half_eps = Nx.divide(eps, two)
        p_half = Nx.add(p, Nx.multiply(half_eps, grad))
        q_new = Nx.add(q, Nx.multiply(eps, Nx.multiply(inv_mass, p_half)))
        {logp_new, grad_new} = Nx.Defn.value_and_grad(q_new, logp_fn)
        p_new = Nx.add(p_half, Nx.multiply(half_eps, grad_new))
        {q_new, p_new, Nx.reshape(logp_new, {}), grad_new}
      end)

      fn q, p, grad, epsilon, inv_mass_diag ->
        eps_t = Nx.tensor(epsilon, type: :f64, backend: Nx.BinaryBackend)
        jitted.(q, p, grad, eps_t, inv_mass_diag)
      end
    else
      fn q, p, grad, epsilon, inv_mass_diag ->
        Exmc.NUTS.Leapfrog.step(vag_fn, q, p, grad, epsilon, inv_mass_diag)
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
      [fn vm ->
        resolved = resolve_params_constrained(params, vm, pm, ncp_info)
        dist.logpdf(Map.fetch!(vm, id), resolved)
      end]
    else
      []
    end
  end

  # Free RV with transform
  defp node_term(%{id: id, op: {:rv, dist, params, transform}}, _ir, pm, ncp_info) do
    if PointMap.has_entry?(pm, id) do
      [
        fn vm ->
          resolved = resolve_params_constrained(params, vm, pm, ncp_info)
          z = Map.fetch!(vm, id)
          x = Transform.apply(transform, z)
          logp = dist.logpdf(x, resolved)
          jac = Transform.log_abs_det_jacobian(transform, z)
          Nx.add(logp, jac)
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

  defp eager_obs_term(%{op: {:rv, dist, params}}, value, meta) do
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

  defp deferred_obs_term(%{op: {:rv, dist, params}}, value, meta, pm, ncp_info) do
    [fn vm ->
      resolved = resolve_params_constrained(params, vm, pm, ncp_info)
      logp = dist.logpdf(value, resolved)
      apply_obs_meta(logp, meta)
    end]
  end

  defp deferred_obs_term(%{op: {:rv, dist, params, transform}}, value, meta, pm, ncp_info) do
    [fn vm ->
      resolved = resolve_params_constrained(params, vm, pm, ncp_info)
      z = inverse_transform(transform, value)
      x = Transform.apply(transform, z)
      logp = dist.logpdf(x, resolved)
      jac = Transform.log_abs_det_jacobian(transform, z)
      apply_obs_meta(Nx.add(logp, jac), meta)
    end]
  end

  defp deferred_obs_term(_target, _value, _meta, _pm, _ncp_info), do: []

  # --- Eager meas_obs computation ---

  defp eager_meas_obs_term(%{op: {:rv, dist, params}}, value, {:matmul, a}, meta) do
    x = Nx.LinAlg.solve(a, value)
    logp = dist.logpdf(x, params)
    jac = Nx.negate(Nx.log(Nx.abs(Nx.LinAlg.determinant(a))))
    combined = Nx.add(logp, jac)
    [fn _vm -> apply_obs_meta(combined, meta) end]
  end

  defp eager_meas_obs_term(%{op: {:rv, dist, params, transform}}, value, {:matmul, a}, meta) do
    x = Nx.LinAlg.solve(a, value)
    z = inverse_transform(transform, x)
    x2 = Transform.apply(transform, z)
    logp = dist.logpdf(x2, params)
    jac = Transform.log_abs_det_jacobian(transform, z)
    meas_jac = Nx.negate(Nx.log(Nx.abs(Nx.LinAlg.determinant(a))))
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
  defp inverse_transform(:softplus, x), do: Nx.log(Nx.expm1(x))
  defp inverse_transform(:logit, x), do: Nx.subtract(Nx.log(x), Nx.log1p(Nx.negate(x)))

  defp has_param_refs?(%{op: {:rv, _dist, params}}), do: Enum.any?(Map.values(params), &is_binary/1)
  defp has_param_refs?(%{op: {:rv, _dist, params, _transform}}), do: Enum.any?(Map.values(params), &is_binary/1)
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

  defp to_tensor(%Nx.Tensor{} = t), do: t
  defp to_tensor(v) when is_number(v) or is_boolean(v), do: Nx.tensor(v)

  # Copy all tensors in IR nodes to BinaryBackend so they can be captured
  # by closures that EXLA.jit traces. EXLA tracing requires captured tensors
  # to be BinaryBackend (not EXLA.Backend).
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

  defp to_binary(%Nx.Tensor{} = t) do
    case t.data do
      %Nx.BinaryBackend{} -> t
      _ -> Nx.backend_copy(t, Nx.BinaryBackend)
    end
  end

  defp to_binary(v), do: v
end
