defmodule Exmc.Dist.Beta do
  @moduledoc """
  Beta distribution parameterized by alpha and beta.

  ## Examples

      iex> x = Nx.tensor(0.4)
      iex> Exmc.Dist.Beta.logpdf(x, %{alpha: Nx.tensor(2.0), beta: Nx.tensor(3.0)}) |> Nx.to_number() |> Float.round(6)
      0.546966
  """

  @behaviour Exmc.Dist

  @impl true
  def logpdf(x, %{alpha: alpha, beta: beta}) do
    Nx.add(
      Nx.add(
        Nx.multiply(Nx.subtract(alpha, Nx.tensor(1.0)), Nx.log(x)),
        Nx.multiply(Nx.subtract(beta, Nx.tensor(1.0)), Nx.log(Nx.subtract(Nx.tensor(1.0), x)))
      ),
      Nx.negate(Exmc.Math.lbeta(alpha, beta))
    )
  end

  @impl true
  def support(_params), do: :unit

  @impl true
  def transform(_params), do: :logit

  @impl true
  def sample(%{alpha: alpha, beta: beta}, rng) do
    alpha_f = Nx.to_number(alpha)
    beta_f = Nx.to_number(beta)
    {x, rng} = Exmc.Dist.Gamma.sample_gamma(alpha_f, 1.0, rng)
    {y, rng} = Exmc.Dist.Gamma.sample_gamma(beta_f, 1.0, rng)
    {Nx.tensor(x / (x + y)), rng}
  end
end
