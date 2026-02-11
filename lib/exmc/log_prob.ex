defmodule Exmc.LogProb do
  @moduledoc """
  Build and evaluate logprob expressions from a probabilistic IR.

  Responsibilities:
  - Run the rewrite pipeline before evaluation
  - Evaluate RV/obs/measurable obs terms
  - Apply observation metadata (weight/mask/reduce)

  ## Examples

      iex> alias Exmc.{Builder, LogProb}
      iex> ir = Builder.new_ir()
      iex> ir = Builder.rv(ir, "x", Exmc.Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
      iex> LogProb.eval(ir, %{"x" => Nx.tensor(0.0)}) |> Nx.to_number() |> Float.round(6)
      -0.918939
  """

  alias Exmc.{IR, Node, Transform, Rewrite}

  @type value_map :: %{required(String.t()) => Nx.t()}

  @doc """
  Evaluate the model logprob for the given IR and value map.

  The `value_map` should contain values for free RVs by id.
  Observed values are read from obs nodes.
  """
  def eval(%IR{} = ir, value_map) when is_map(value_map) do
    ir = Rewrite.apply(ir)

    ir.nodes
    |> Map.values()
    |> Enum.flat_map(&node_logp(&1, ir, value_map))
    |> sum_logps()
  end

  defp node_logp(%Node{op: {:rv, dist, params}, id: id}, _ir, value_map) do
    case Map.fetch(value_map, id) do
      {:ok, v} ->
        resolved = resolve_params(params, value_map)
        [dist.logpdf(v, resolved)]

      :error ->
        []
    end
  end

  defp node_logp(%Node{op: {:rv, dist, params, transform}, id: id}, _ir, value_map) do
    case Map.fetch(value_map, id) do
      {:ok, z} ->
        resolved = resolve_params(params, value_map)
        x = Transform.apply(transform, z)
        logp = dist.logpdf(x, resolved)
        jac = Transform.log_abs_det_jacobian(transform, z)
        [Nx.add(logp, jac)]

      :error ->
        []
    end
  end

  defp node_logp(%Node{op: {:obs, target_id, value}}, %IR{} = ir, value_map) do
    node_logp(%Node{op: {:obs, target_id, value, %{}}}, ir, value_map)
  end

  defp node_logp(%Node{op: {:obs, target_id, value, meta}}, %IR{} = ir, value_map) do
    target_node = IR.get_node!(ir, target_id)

    if Map.get(meta, :likelihood, true) == false do
      []
    else
      case target_node.op do
        {:rv, dist, params} ->
          resolved = resolve_params(params, value_map)
          apply_obs_meta([dist.logpdf(value, resolved)], meta)

        {:rv, dist, params, transform} ->
          resolved = resolve_params(params, value_map)
          z = inverse_transform(transform, value)
          x = Transform.apply(transform, z)
          logp = dist.logpdf(x, resolved)
          jac = Transform.log_abs_det_jacobian(transform, z)
          apply_obs_meta([Nx.add(logp, jac)], meta)

        _ ->
          []
      end
    end
  end

  defp node_logp(
         %Node{op: {:meas_obs, rv_id, value, {:matmul, a}}},
         %IR{} = ir,
         value_map
       ) do
    node_logp(%Node{op: {:meas_obs, rv_id, value, {:matmul, a}, %{}}}, ir, value_map)
  end

  defp node_logp(
         %Node{op: {:meas_obs, rv_id, value, {:matmul, a}, meta}},
         %IR{} = ir,
         _value_map
       ) do
    rv_node = IR.get_node!(ir, rv_id)

    case rv_node.op do
      {:rv, dist, params} ->
        x = Nx.LinAlg.solve(a, value)
        logp = dist.logpdf(x, params)
        jac = Nx.negate(Nx.log(Nx.abs(Nx.LinAlg.determinant(a))))
        apply_obs_meta([Nx.add(logp, jac)], meta)

      {:rv, dist, params, transform} ->
        x = Nx.LinAlg.solve(a, value)
        z = inverse_transform(transform, x)
        x2 = Transform.apply(transform, z)
        logp = dist.logpdf(x2, params)
        jac = Transform.log_abs_det_jacobian(transform, z)
        meas_jac = Nx.negate(Nx.log(Nx.abs(Nx.LinAlg.determinant(a))))
        apply_obs_meta([Nx.add(Nx.add(logp, jac), meas_jac)], meta)

      _ ->
        []
    end
  end

  defp node_logp(
         %Node{op: {:meas_obs, rv_id, value, {:affine, a, b}, meta}},
         %IR{} = ir,
         _value_map
       ) do
    rv_node = IR.get_node!(ir, rv_id)

    case rv_node.op do
      {:rv, dist, params} ->
        a_t = to_tensor(a)
        b_t = to_tensor(b)
        x = Nx.divide(Nx.subtract(value, b_t), a_t)
        logp = dist.logpdf(x, params)
        jac = Nx.negate(Nx.log(Nx.abs(a_t)))
        apply_obs_meta([Nx.add(logp, jac)], meta)

      {:rv, dist, params, transform} ->
        a_t = to_tensor(a)
        b_t = to_tensor(b)
        x = Nx.divide(Nx.subtract(value, b_t), a_t)
        z = inverse_transform(transform, x)
        x2 = Transform.apply(transform, z)
        logp = dist.logpdf(x2, params)
        jac = Transform.log_abs_det_jacobian(transform, z)
        meas_jac = Nx.negate(Nx.log(Nx.abs(a_t)))
        apply_obs_meta([Nx.add(Nx.add(logp, jac), meas_jac)], meta)

      _ ->
        []
    end
  end

  defp node_logp(_node, _ir, _value_map), do: []

  defp sum_logps([]), do: Nx.tensor(0.0)
  defp sum_logps([one]), do: one
  defp sum_logps(list), do: Enum.reduce(list, fn x, acc -> Nx.add(acc, x) end)

  defp inverse_transform(nil, x), do: x
  defp inverse_transform(:log, x), do: Nx.log(x)
  defp inverse_transform(:softplus, x), do: Nx.log(Nx.expm1(x))
  defp inverse_transform(:logit, x), do: Nx.subtract(Nx.log(x), Nx.log1p(Nx.negate(x)))

  defp apply_obs_meta([logp], meta) do
    weight = Map.get(meta, :weight, 1.0)
    weight_t = to_tensor(weight)
    logp = Nx.multiply(logp, weight_t)

    masked =
      case Map.get(meta, :mask) do
        nil -> logp
        mask -> Nx.select(to_tensor(mask), logp, Nx.tensor(0.0))
      end

    case Map.get(meta, :reduce) do
      :sum -> [Nx.sum(masked)]
      :mean -> [Nx.mean(masked)]
      :logsumexp -> [Nx.logsumexp(masked)]
      _ -> [masked]
    end
  end

  defp resolve_params(params, value_map) do
    Map.new(params, fn
      {k, v} when is_binary(v) -> {k, Map.fetch!(value_map, v)}
      {k, v} -> {k, v}
    end)
  end

  defp to_tensor(%Nx.Tensor{} = t), do: t
  defp to_tensor(v) when is_number(v) or is_boolean(v), do: Nx.tensor(v)
end
