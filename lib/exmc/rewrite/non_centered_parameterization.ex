defmodule Exmc.Rewrite.NonCenteredParameterization do
  @moduledoc """
  Non-centered parameterization for hierarchical Normal RVs.

  Transforms `x ~ N(mu_ref, sigma_ref)` (where both params are string
  references to parent RVs) into `x ~ N(0, 1)` with NCP metadata
  recording the original params. The compiler reconstructs
  `x_original = mu + sigma * z` when other nodes reference `x`.

  This eliminates funnel geometry in hierarchical models, improving
  NUTS sampling efficiency for weakly-identified parameters.

  Only applies to free (unobserved) Normal RVs without transforms.
  """

  @behaviour Exmc.Rewrite.Pass

  alias Exmc.{IR, Node}
  alias Exmc.Dist.Normal

  @impl true
  def name, do: "non_centered_parameterization"

  @impl true
  def run(%IR{} = ir) do
    observed = observed_target_ids(ir)

    {new_nodes, ncp_info} =
      Enum.reduce(ir.nodes, {ir.nodes, %{}}, fn {id, node}, {nodes, ncp} ->
        case should_ncp?(node, observed) do
          {:yes, mu_src, sigma_src} ->
            new_op =
              {:rv, Normal,
               %{
                 mu: Nx.tensor(0.0, backend: Nx.BinaryBackend),
                 sigma: Nx.tensor(1.0, backend: Nx.BinaryBackend)
               }}

            new_node = %Node{node | op: new_op, deps: []}
            {Map.put(nodes, id, new_node), Map.put(ncp, id, %{mu: mu_src, sigma: sigma_src})}

          :no ->
            {nodes, ncp}
        end
      end)

    # Merge, do not replace.
    #
    # This pass now runs twice on a model — once in Compiler.do_compile/2 and
    # once in CustomSynth.synthesise/2. On the second run should_ncp?/2 returns
    # :no for every node, because the params it guards on (`is_binary(mu) and
    # is_binary(sigma)`) are literal tensors by then. A plain replace therefore
    # wipes the first run's record, and every reference that needed
    # reconstructing as `mu + sigma * z` loses the sources to reconstruct from.
    %{ir | nodes: new_nodes, ncp_info: Map.merge(ir.ncp_info || %{}, ncp_info)}
  end

  defp should_ncp?(%Node{id: id, op: {:rv, Normal, %{mu: mu, sigma: sigma}}}, observed)
       when is_binary(mu) and is_binary(sigma) do
    if MapSet.member?(observed, id), do: :no, else: {:yes, mu, sigma}
  end

  defp should_ncp?(_node, _observed), do: :no

  defp observed_target_ids(%IR{} = ir) do
    ir.nodes
    |> Map.values()
    |> Enum.flat_map(fn node ->
      case node.op do
        {:obs, target_id, _value} -> [target_id]
        {:obs, target_id, _value, _meta} -> [target_id]
        {:meas_obs, rv_id, _value, _op_info} -> [rv_id]
        {:meas_obs, rv_id, _value, _op_info, _meta} -> [rv_id]
        _ -> []
      end
    end)
    |> MapSet.new()
  end

  @doc """
  Rebuild non-centred variables in a trace or a point: `x = mu + sigma * z`,
  in dependency order, for every id in `ncp_info` (the map `run/1` records).

  `values` maps ids to tensors. A **trace** carries a leading sample axis
  (`{n}` for a scalar RV, `{n, 8}` for a `shape: {8}` one); a **point**, as
  `sample_stream` builds per draw, does not. Both work.

  Values taken from `values` are padded with trailing singleton axes up to
  z's rank before broadcasting. Nx broadcasts right-aligned, so a scalar
  parent's draws `{n}` against a vector child's `{n, 8}` failed ("cannot
  broadcast tensor of dimensions {n} to {n, 8}") while the leading sample axis
  was what they shared. Found 2026-09-13, sampling the vectorised eight schools
  with NCP on; this function replaced two private copies, in `Exmc.NUTS.Sampler`
  and `Exmc.MCLMC` (the latter also serving `Exmc.MAMS`), that both had it.
  Constant parameters (tensors or numbers from the model, not from `values`)
  have no sample axis and are left to broadcast as they are.
  """
  @spec reconstruct(%{optional(String.t()) => Nx.Tensor.t()}, map()) :: map()
  def reconstruct(values, ncp_info) when map_size(ncp_info) == 0, do: values

  def reconstruct(values, ncp_info) do
    ncp_info
    |> dependency_order()
    |> Enum.reduce(values, fn id, acc ->
      %{mu: mu_src, sigma: sigma_src} = Map.fetch!(ncp_info, id)
      z = Map.fetch!(acc, id)
      mu = ncp_value(mu_src, acc, z)
      sigma = ncp_value(sigma_src, acc, z)
      Map.put(acc, id, Nx.add(mu, Nx.multiply(sigma, z)))
    end)
  end

  defp ncp_value(id, values, z) when is_binary(id) do
    v = Map.fetch!(values, id)
    pad = Nx.rank(z) - Nx.rank(v)

    if pad > 0,
      do:
        Nx.reshape(
          v,
          Tuple.to_list(Nx.shape(v)) |> Kernel.++(List.duplicate(1, pad)) |> List.to_tuple()
        ),
      else: v
  end

  defp ncp_value(%Nx.Tensor{} = v, _values, _z), do: v

  defp ncp_value(v, _values, z) when is_number(v),
    do: Nx.tensor(v, type: Nx.type(z), backend: Nx.BinaryBackend)

  # An entry whose mu or sigma names another NCP'd variable waits for it. Ready
  # ids are taken in sorted order, so the order is deterministic; the values do
  # not depend on it.
  defp dependency_order(ncp_info) do
    ids = MapSet.new(Map.keys(ncp_info))
    do_order(Map.keys(ncp_info), ncp_info, ids, MapSet.new(), [])
  end

  defp do_order([], _ncp, _ids, _done, acc), do: Enum.reverse(acc)

  defp do_order(remaining, ncp, ids, done, acc) do
    {ready, blocked} =
      Enum.split_with(remaining, fn id ->
        %{mu: mu, sigma: sigma} = ncp[id]
        dep_ready?(mu, ids, done) and dep_ready?(sigma, ids, done)
      end)

    case Enum.sort(ready) do
      # A cycle cannot come out of run/1; emit the rest rather than loop.
      [] ->
        Enum.reverse(acc) ++ blocked

      ready ->
        do_order(
          blocked,
          ncp,
          ids,
          MapSet.union(done, MapSet.new(ready)),
          Enum.reverse(ready) ++ acc
        )
    end
  end

  defp dep_ready?(src, ids, done) when is_binary(src),
    do: not MapSet.member?(ids, src) or MapSet.member?(done, src)

  defp dep_ready?(_src, _ids, _done), do: true
end
