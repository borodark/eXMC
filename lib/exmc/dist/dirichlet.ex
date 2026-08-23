defmodule Exmc.Dist.Dirichlet do
  import Exmc.Math, only: [c: 2]

  @moduledoc """
  Dirichlet distribution on the K-simplex.

  Parameterized by concentration vector `alpha` ({K}).
  The RV lives on the simplex: x_i > 0, sum(x) = 1.

  Uses stick-breaking transform for unconstrained sampling:
  unconstrained space is R^{K-1}, constrained space is the K-simplex.

  ## Examples

      iex> alpha = Nx.tensor([1.0, 1.0, 1.0])
      iex> x = Nx.tensor([1/3, 1/3, 1/3])
      iex> Exmc.Dist.Dirichlet.logpdf(x, %{alpha: alpha}) |> Nx.to_number() |> Float.round(4)
      0.6931
  """

  @behaviour Exmc.Dist

  @impl true
  def logpdf(x, %{alpha: alpha}) do
    # logpdf = sum((alpha_i - 1) * log(x_i)) + lgamma(sum(alpha)) - sum(lgamma(alpha_i))
    log_x = Nx.log(x)
    logp_kernel = Nx.sum(Nx.multiply(Nx.subtract(alpha, c(1.0, x)), log_x))

    # Log normalizing constant: lgamma(sum(alpha)) - sum(lgamma(alpha))
    log_norm =
      Nx.subtract(
        Exmc.Math.lgamma(Nx.sum(alpha)),
        Nx.sum(Exmc.Math.lgamma(alpha))
      )

    Nx.add(logp_kernel, log_norm)
  end

  @impl true
  def support(_params), do: :simplex

  @impl true
  def transform(_params), do: :stick_breaking

  @impl true
  def sample(%{alpha: alpha}, rng) do
    alpha_list = Nx.to_flat_list(alpha)

    {gammas, rng} =
      Enum.map_reduce(alpha_list, rng, fn a, r ->
        Exmc.Dist.Gamma.sample_gamma(a, 1.0, r)
      end)

    total = Enum.sum(gammas)
    x = Nx.tensor(Enum.map(gammas, &(&1 / total)), type: :f64)
    {x, rng}
  end
end
