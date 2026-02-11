defmodule Exmc.Dist.Custom do
  @moduledoc """
  User-defined distribution wrapper.

  Allows using any function as a distribution in Exmc models.
  The user provides a `logpdf_fn` that takes `(x, params)` and returns log-density.
  The Custom struct is passed through the params map under the `:__dist__` key,
  so no changes to Builder or Compiler are needed.

  ## Example

      logpdf = fn x, params ->
        mu = params.mu
        Nx.negate(Nx.multiply(0.5, Nx.pow(Nx.subtract(x, mu), 2)))
      end

      dist = Exmc.Dist.Custom.new(logpdf, support: :real)
      ir = Builder.rv(ir, "x", Exmc.Dist.Custom, Map.put(%{mu: Nx.tensor(0.0)}, :__dist__, dist))

  ## Convenience

  Use `Exmc.Dist.Custom.rv/5` to avoid manually injecting `__dist__`:

      dist = Exmc.Dist.Custom.new(logpdf, support: :real)
      ir = Exmc.Dist.Custom.rv(ir, "x", dist, %{mu: Nx.tensor(0.0)})
  """

  @behaviour Exmc.Dist

  defstruct [:logpdf_fn, :support, :transform, :sample_fn]

  @doc """
  Create a new custom distribution.

  ## Options
  - `:support` - `:real` (default), `:positive`, or `:unit_interval`
  - `:transform` - `nil` (default), `:log`, `:softplus`, `:logit` (overrides auto)
  - `:sample_fn` - optional `fn(params, rng) -> {sample, rng}` for prior predictive
  """
  def new(logpdf_fn, opts \\ []) when is_function(logpdf_fn, 2) do
    support = Keyword.get(opts, :support, :real)

    transform =
      case Keyword.get(opts, :transform) do
        nil ->
          case support do
            :positive -> :log
            :unit_interval -> :logit
            _ -> nil
          end

        explicit ->
          explicit
      end

    %__MODULE__{
      logpdf_fn: logpdf_fn,
      support: support,
      transform: transform,
      sample_fn: Keyword.get(opts, :sample_fn)
    }
  end

  @doc """
  Convenience helper: add a custom dist RV to the IR.

  Injects the `:__dist__` key into params automatically.
  """
  def rv(ir, id, %__MODULE__{} = dist, params, opts \\ []) do
    full_params = Map.put(params, :__dist__, dist)
    Exmc.Builder.rv(ir, id, __MODULE__, full_params, opts)
  end

  # --- Behaviour callbacks ---
  # These are called by the compiler/rewrite as dist_mod.logpdf(x, params),
  # dist_mod.support(params), etc. The %Custom{} struct is inside params[:__dist__].

  @impl true
  def logpdf(x, params) do
    dist = Map.fetch!(params, :__dist__)
    user_params = Map.delete(params, :__dist__)
    dist.logpdf_fn.(x, user_params)
  end

  @impl true
  def support(params) do
    dist = Map.fetch!(params, :__dist__)
    dist.support
  end

  @impl true
  def transform(params) do
    dist = Map.fetch!(params, :__dist__)
    dist.transform
  end

  @impl true
  def sample(params, rng) do
    dist = Map.fetch!(params, :__dist__)
    user_params = Map.delete(params, :__dist__)

    if dist.sample_fn do
      dist.sample_fn.(user_params, rng)
    else
      raise "Custom distribution does not implement sample/2. " <>
              "Provide a :sample_fn option to Exmc.Dist.Custom.new/2."
    end
  end
end
