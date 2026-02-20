defmodule Exmc.Stan.DistMap do
  @moduledoc """
  Maps Stan distribution names to `Exmc.Dist` modules and their parameter names.

  ## Supported Distributions

  | Stan name | Exmc module | Parameters |
  |-----------|-------------|------------|
  | `normal` | `Exmc.Dist.Normal` | `mu`, `sigma` |
  | `gamma` | `Exmc.Dist.Gamma` | `alpha`, `beta` |
  | `exponential` | `Exmc.Dist.Exponential` | `lambda` |
  | `beta` | `Exmc.Dist.Beta` | `alpha`, `beta` |
  | `half_normal` | `Exmc.Dist.HalfNormal` | `sigma` |
  | `half_cauchy` | `Exmc.Dist.HalfCauchy` | `beta` |
  | `cauchy` | `Exmc.Dist.Cauchy` | `mu`, `gamma` |
  | `student_t` | `Exmc.Dist.StudentT` | `nu`, `mu`, `sigma` |
  | `bernoulli` | `Exmc.Dist.Bernoulli` | `p` |
  | `poisson` | `Exmc.Dist.Poisson` | `mu` |
  | `lognormal` | `Exmc.Dist.Lognormal` | `mu`, `sigma` |
  | `truncated_normal` | `Exmc.Dist.TruncatedNormal` | `mu`, `sigma`, `lower`, `upper` |
  | `laplace` | `Exmc.Dist.Laplace` | `mu`, `b` |
  | `dirichlet` | `Exmc.Dist.Dirichlet` | `alpha` |
  """

  @map %{
    "normal" => {Exmc.Dist.Normal, [:mu, :sigma]},
    "gamma" => {Exmc.Dist.Gamma, [:alpha, :beta]},
    "exponential" => {Exmc.Dist.Exponential, [:lambda]},
    "beta" => {Exmc.Dist.Beta, [:alpha, :beta]},
    "half_normal" => {Exmc.Dist.HalfNormal, [:sigma]},
    "half_cauchy" => {Exmc.Dist.HalfCauchy, [:beta]},
    "cauchy" => {Exmc.Dist.Cauchy, [:mu, :gamma]},
    "student_t" => {Exmc.Dist.StudentT, [:nu, :mu, :sigma]},
    "bernoulli" => {Exmc.Dist.Bernoulli, [:p]},
    "poisson" => {Exmc.Dist.Poisson, [:mu]},
    "lognormal" => {Exmc.Dist.Lognormal, [:mu, :sigma]},
    "truncated_normal" => {Exmc.Dist.TruncatedNormal, [:mu, :sigma, :lower, :upper]},
    "laplace" => {Exmc.Dist.Laplace, [:mu, :b]},
    "dirichlet" => {Exmc.Dist.Dirichlet, [:alpha]}
  }

  @doc "Look up a Stan distribution name. Returns `{:ok, {module, param_names}}` or `:error`."
  def lookup(name) when is_binary(name) do
    Map.fetch(@map, name)
  end

  @doc "Look up a Stan distribution name, raising on unknown with a list of supported distributions."
  def lookup!(name) when is_binary(name) do
    case Map.fetch(@map, name) do
      {:ok, val} ->
        val

      :error ->
        supported_list = @map |> Map.keys() |> Enum.sort() |> Enum.join(", ")

        raise ArgumentError,
              "unknown distribution '#{name}'. Supported distributions: #{supported_list}"
    end
  end

  @doc "List all supported distribution names."
  def supported, do: Map.keys(@map)
end
