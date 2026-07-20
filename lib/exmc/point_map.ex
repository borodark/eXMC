defmodule Exmc.PointMap do
  @moduledoc """
  Flat-vector <-> named value-map bijection for samplers.

  Maps between a flat f64 tensor (unconstrained space) and a named map
  of per-variable tensors. Free RVs are identified by exclusion: any RV
  node not targeted by an obs or meas_obs node.
  """

  alias Exmc.{IR, Transform}

  defstruct entries: [], size: 0

  @type entry :: %{
          id: String.t(),
          offset: non_neg_integer(),
          length: pos_integer(),
          shape: tuple(),
          transform: atom() | nil
        }

  @type t :: %__MODULE__{entries: [entry()], size: non_neg_integer()}

  @doc """
  Build a PointMap from rewritten IR.

  Free RVs are RV nodes not targeted by any obs/meas_obs. Sorted
  alphabetically by id for deterministic layout.
  """
  def build(%IR{} = ir) do
    observed_targets = observed_target_ids(ir)

    entries =
      ir.nodes
      |> Map.values()
      |> Enum.filter(&free_rv?(&1, observed_targets))
      |> Enum.sort_by(& &1.id)
      |> Enum.reduce({[], 0}, fn node, {acc, offset} ->
        shape = node_shape(node)
        transform = node_transform(node)
        length = unconstrained_length(transform, shape)
        unc_shape = unconstrained_shape(transform, shape)

        entry = %{
          id: node.id,
          offset: offset,
          length: length,
          shape: unc_shape,
          transform: transform
        }

        {[entry | acc], offset + length}
      end)
      |> then(fn {entries, size} ->
        {Enum.reverse(entries), size}
      end)

    {entries_list, total_size} = entries
    %__MODULE__{entries: entries_list, size: total_size}
  end

  @doc """
  Pack a named value map into a flat f64 tensor in entry order.
  """
  def pack(value_map, %__MODULE__{} = pm) do
    tensors =
      Enum.map(pm.entries, fn entry ->
        value_map
        |> Map.fetch!(entry.id)
        |> Nx.reshape({entry.length})
        |> Nx.as_type(Exmc.JIT.precision())
      end)

    case tensors do
      [] -> Nx.tensor([], type: Exmc.JIT.precision())
      _ -> Nx.concatenate(tensors)
    end
  end

  @doc """
  Unpack a flat tensor into a named value map via Nx.slice + Nx.reshape.

  This is differentiable — uses only Nx ops.
  """
  def unpack(flat, %__MODULE__{} = pm) do
    Map.new(pm.entries, fn entry ->
      slice = Nx.slice(flat, [entry.offset], [entry.length])
      value = Nx.reshape(slice, entry.shape)
      {entry.id, value}
    end)
  end

  @doc """
  Check if the PointMap has an entry for the given id.
  """
  def has_entry?(%__MODULE__{} = pm, id) when is_binary(id) do
    Enum.any?(pm.entries, fn entry -> entry.id == id end)
  end

  @doc """
  Apply forward transforms (unconstrained -> constrained) to all entries.
  """
  def to_constrained(value_map, %__MODULE__{} = pm) do
    Map.new(pm.entries, fn entry ->
      z = Map.fetch!(value_map, entry.id)
      x = Transform.apply(entry.transform, z)
      {entry.id, x}
    end)
  end

  @doc """
  Apply inverse transforms (constrained -> unconstrained) to all entries.
  """
  def to_unconstrained(value_map, %__MODULE__{} = pm) do
    Map.new(pm.entries, fn entry ->
      x = Map.fetch!(value_map, entry.id)
      z = inverse_transform(entry.transform, x)
      {entry.id, z}
    end)
  end

  # --- Private helpers ---

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

  defp free_rv?(node, observed_targets) do
    case node.op do
      {:rv, _dist, _params} -> not MapSet.member?(observed_targets, node.id)
      {:rv, _dist, _params, _transform} -> not MapSet.member?(observed_targets, node.id)
      _ -> false
    end
  end

  defp node_shape(node) do
    case node.shape do
      nil -> {}
      shape when is_tuple(shape) -> shape
    end
  end

  defp unconstrained_length(transform, shape) do
    Transform.unconstrained_length(transform, shape)
  end

  defp unconstrained_shape(transform, shape) do
    Transform.unconstrained_shape(transform, shape)
  end

  defp node_transform(node) do
    case node.op do
      {:rv, _dist, _params, transform} -> transform
      {:rv, _dist, _params} -> nil
      _ -> nil
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
end
