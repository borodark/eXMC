defmodule Exmc.IR do
  @moduledoc """
  Minimal probabilistic IR for logprob construction.
  """

  defstruct nodes: %{}, outputs: [], ncp_info: %{}

  alias Exmc.Node

  @type t :: %__MODULE__{
          nodes: %{required(String.t()) => Node.t()},
          outputs: [String.t()],
          ncp_info: %{optional(String.t()) => %{mu: term(), sigma: term()}}
        }

  @doc "Create an empty IR."
  def new, do: %__MODULE__{}

  @doc "Add a node to the IR, keyed by its id."
  def add_node(%__MODULE__{} = ir, %Node{id: id} = node) when is_binary(id) do
    %{ir | nodes: Map.put(ir.nodes, id, node)}
  end

  @doc "Fetch a node by id, raising if not found."
  def get_node!(%__MODULE__{} = ir, id) when is_binary(id) do
    case Map.fetch(ir.nodes, id) do
      {:ok, node} -> node
      :error -> raise ArgumentError, "unknown node id: #{inspect(id)}"
    end
  end
end
