defmodule Exmc.Predictive do
  @moduledoc """
  Prior and posterior predictive sampling.

  Uses `Dist.sample/2` callbacks to draw forward samples from the model.
  """

  alias Exmc.{IR, Rewrite}

  @doc """
  Draw prior samples from all RVs in the model.

  Walks the IR in topological order, sampling each RV using its
  `Dist.sample/2` callback. String param references are resolved
  from already-sampled ancestors.

  Returns `%{var_name => Nx.t() of shape {n, ...var_shape}}`.
  """
  def prior_samples(ir, n \\ 100, opts \\ []) do
    ir = Rewrite.apply(ir)
    seed = Keyword.get(opts, :seed, 0)
    rng = :rand.seed_s(:exsss, seed)

    rv_nodes = all_rv_nodes(ir)
    sorted = topo_sort(rv_nodes)

    {draws, _rng} =
      Enum.map_reduce(1..n, rng, fn _i, rng ->
        sample_prior_once(sorted, rng)
      end)

    stack_draws(draws, Enum.map(sorted, &elem(&1, 0)))
  end

  @doc """
  Draw posterior predictive samples for observed variables.

  For each posterior sample in the trace, resolves the obs target's
  distribution params from the trace, then draws a new sample from
  the likelihood.

  Returns `%{obs_var_name => Nx.t() of shape {n, ...var_shape}}`.
  """
  def posterior_predictive(ir, trace, opts \\ []) do
    ir = Rewrite.apply(ir)
    seed = Keyword.get(opts, :seed, 0)
    rng = :rand.seed_s(:exsss, seed)

    obs_targets = find_obs_targets(ir)

    if obs_targets == [] do
      %{}
    else
      n = trace_length(trace)

      {draws, _rng} =
        Enum.map_reduce(0..(n - 1), rng, fn i, rng ->
          sample_predictive_once(obs_targets, trace, i, rng)
        end)

      stack_draws(draws, Enum.map(obs_targets, &elem(&1, 0)))
    end
  end

  # --- Private: prior sampling ---

  defp all_rv_nodes(ir) do
    for {id, node} <- ir.nodes,
        match?({:rv, _, _}, node.op) or match?({:rv, _, _, _}, node.op),
        do: {id, node}
  end

  defp sample_prior_once(sorted_rvs, rng) do
    {values, rng} =
      Enum.reduce(sorted_rvs, {%{}, rng}, fn {id, node}, {values, rng} ->
        {dist, params} = extract_dist_params(node)
        resolved = resolve_params(params, values)
        {value, rng} = dist.sample(resolved, rng)
        {Map.put(values, id, value), rng}
      end)

    {values, rng}
  end

  # --- Private: posterior predictive ---

  defp find_obs_targets(ir) do
    for {_id, node} <- ir.nodes,
        {obs_name, target_id} <- obs_target(node),
        rv_node = IR.get_node!(ir, target_id),
        do: {obs_name, rv_node}
  end

  defp obs_target(%{id: id, op: {:obs, target_id, _value}}), do: [{id, target_id}]
  defp obs_target(%{id: id, op: {:obs, target_id, _value, _meta}}), do: [{id, target_id}]
  defp obs_target(_), do: []

  defp sample_predictive_once(obs_targets, trace, i, rng) do
    {values, rng} =
      Enum.reduce(obs_targets, {%{}, rng}, fn {obs_name, rv_node}, {values, rng} ->
        {dist, params} = extract_dist_params(rv_node)
        resolved = resolve_params_from_trace(params, trace, i)
        {value, rng} = dist.sample(resolved, rng)
        {Map.put(values, obs_name, value), rng}
      end)

    {values, rng}
  end

  # --- Private: helpers ---

  defp extract_dist_params(%{op: {:rv, dist, params}}), do: {dist, params}
  defp extract_dist_params(%{op: {:rv, dist, params, _transform}}), do: {dist, params}

  defp resolve_params(params, values) do
    Map.new(params, fn
      {k, v} when is_binary(v) -> {k, Map.fetch!(values, v)}
      {k, v} -> {k, v}
    end)
  end

  defp resolve_params_from_trace(params, trace, i) do
    Map.new(params, fn
      {k, v} when is_binary(v) ->
        samples = Map.fetch!(trace, v)
        {k, Nx.reshape(samples[i], {})}
      {k, v} ->
        {k, v}
    end)
  end

  defp trace_length(trace) do
    {_key, first} = Enum.at(trace, 0)
    elem(Nx.shape(first), 0)
  end

  # --- Private: topological sort ---

  defp topo_sort(rv_nodes) do
    rv_ids = MapSet.new(Enum.map(rv_nodes, &elem(&1, 0)))
    node_map = Map.new(rv_nodes)

    deps =
      Map.new(rv_nodes, fn {id, node} ->
        {_dist, params} = extract_dist_params(node)

        parent_ids =
          params
          |> Map.values()
          |> Enum.filter(fn v -> is_binary(v) and MapSet.member?(rv_ids, v) end)

        {id, parent_ids}
      end)

    kahn_sort(deps, node_map)
  end

  defp kahn_sort(deps, node_map) do
    # Kahn's algorithm for topological sort
    in_degree =
      Map.new(deps, fn {id, parents} -> {id, length(parents)} end)

    queue =
      in_degree
      |> Enum.filter(fn {_id, deg} -> deg == 0 end)
      |> Enum.map(&elem(&1, 0))
      |> Enum.sort()

    do_kahn(queue, deps, in_degree, node_map, [])
  end

  defp do_kahn([], _deps, _in_degree, _node_map, acc), do: Enum.reverse(acc)

  defp do_kahn([id | rest], deps, in_degree, node_map, acc) do
    node = Map.fetch!(node_map, id)

    # Find nodes that depend on this id
    {new_ready, in_degree} =
      Enum.reduce(deps, {[], in_degree}, fn {child_id, parents}, {ready, deg} ->
        if id in parents do
          new_deg = Map.update!(deg, child_id, &(&1 - 1))

          if new_deg[child_id] == 0 do
            {[child_id | ready], new_deg}
          else
            {ready, new_deg}
          end
        else
          {ready, deg}
        end
      end)

    queue = Enum.sort(rest ++ new_ready)
    do_kahn(queue, deps, in_degree, node_map, [{id, node} | acc])
  end

  # --- Private: stacking draws ---

  defp stack_draws(draws, var_names) do
    Map.new(var_names, fn name ->
      tensors = Enum.map(draws, fn draw -> Map.fetch!(draw, name) end)
      {name, Nx.stack(tensors)}
    end)
  end
end
