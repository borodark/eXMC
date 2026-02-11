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

    %{ir | nodes: new_nodes, ncp_info: ncp_info}
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
end
