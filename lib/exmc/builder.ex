defmodule Exmc.Builder do
  @moduledoc """
  Helpers to build a minimal probabilistic IR.
  """

  alias Exmc.{IR, Node}

  @doc "Create a fresh empty IR."
  def new_ir, do: IR.new()

  @doc """
  Register observation data as a runtime JIT argument.

  The tensor is passed as an XLA parameter (not a constant), so EXLA
  caches by shape only — different data values with the same shape
  reuse the compiled executable. Reference in Custom dist params via
  the string `"__obs_data"`.
  """
  def data(%IR{} = ir, %Nx.Tensor{} = tensor) do
    # Force observation data onto BinaryBackend at ingress. Downstream,
    # Compiler.build_vag_fn/build_step_fn closes over this tensor inside
    # Nx.Defn.value_and_grad, and the defn tracer rejects captured tensors on
    # non-Defn-compatible backends (Nx.Vulkan.VulkanoBackend and friends) with
    # "two incompatible tensor implementations". Obs data is small and
    # conceptually host-resident, so the copy is cheap and it removes the whole
    # class of cross-backend errors at the root rather than at each call site.
    %{ir | data: Nx.backend_copy(tensor, Nx.BinaryBackend)}
  end

  @doc """
  Add a random variable node to the IR.

  ## Examples

      iex> alias Exmc.Builder
      iex> ir = Builder.new_ir()
      iex> ir = Builder.rv(ir, "x", Exmc.Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
      iex> map_size(ir.nodes) > 0
      true
  """
  def rv(%IR{} = ir, id, dist, params, opts \\ []) when is_binary(id) and is_map(params) do
    transform = Keyword.get(opts, :transform)
    shape = Keyword.get(opts, :shape)

    # Same coercion as data/2, for the same reason: the logp_fn closes over
    # these prior params inside Nx.Defn.value_and_grad. BinaryBackend constants
    # substitute cleanly into the Expr tree regardless of the global default.
    params =
      Map.new(params, fn
        {k, %Nx.Tensor{} = t} -> {k, Nx.backend_copy(t, Nx.BinaryBackend)}
        {k, v} -> {k, v}
      end)

    op =
      if is_nil(transform) do
        {:rv, dist, params}
      else
        {:rv, dist, params, transform}
      end

    deps = params |> Map.values() |> Enum.filter(&is_binary/1)
    node = %Node{id: id, op: op, deps: deps, shape: shape}
    IR.add_node(ir, node)
  end

  @doc """
  Add an observation node to the IR, with optional metadata.

  ## Examples

      iex> alias Exmc.Builder
      iex> ir = Builder.new_ir()
      iex> ir = Builder.rv(ir, "x", Exmc.Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
      iex> ir = Builder.obs(ir, "x_obs", "x", Nx.tensor(0.2), weight: 2.0)
      iex> map_size(ir.nodes) == 2
      true
  """
  def obs(%IR{} = ir, id, rv_id, value, opts \\ [])
      when is_binary(id) and is_binary(rv_id) do
    meta = build_obs_meta(opts, value)
    node = %Node{id: id, op: {:obs, rv_id, value, meta}, deps: [rv_id]}
    IR.add_node(ir, node)
  end

  @doc """
  Add a deterministic node to the IR.

  ## Examples

      iex> alias Exmc.Builder
      iex> ir = Builder.new_ir()
      iex> ir = Builder.det(ir, "d", :add, ["x", Nx.tensor(1.0)])
      iex> Map.has_key?(ir.nodes, "d")
      true
  """
  def det(%IR{} = ir, id, fun, args) when is_binary(id) and is_list(args) do
    node = %Node{id: id, op: {:det, fun, args}, deps: []}
    IR.add_node(ir, node)
  end

  defp build_obs_meta(opts, value) when is_list(opts) do
    meta = Keyword.get(opts, :meta, %{})
    meta = if is_map(meta), do: meta, else: %{}

    meta =
      meta
      |> put_if_present(:likelihood, Keyword.get(opts, :likelihood))
      |> put_if_present(:weight, Keyword.get(opts, :weight))
      |> put_if_present(:mask, Keyword.get(opts, :mask))
      |> put_if_present(:reduce, Keyword.get(opts, :reduce))
      |> put_if_present(:censored, Keyword.get(opts, :censored))

    # Auto-add reduce: :sum for non-scalar obs (rank > 0)
    if not Map.has_key?(meta, :reduce) and vector_obs?(value) do
      Map.put(meta, :reduce, :sum)
    else
      meta
    end
  end

  defp vector_obs?(%Nx.Tensor{} = t), do: tuple_size(Nx.shape(t)) > 0
  defp vector_obs?(_), do: false

  defp put_if_present(meta, _key, nil), do: meta
  defp put_if_present(meta, key, value), do: Map.put(meta, key, value)
end
