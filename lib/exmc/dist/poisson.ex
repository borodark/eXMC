defmodule Exmc.Dist.Poisson do
  import Exmc.Math, only: [c: 2]

  @moduledoc """
  Poisson distribution parameterized by rate mu.

  Primarily used as an observation likelihood with count data.

  ## Examples

      iex> y = Nx.tensor(3.0)
      iex> Exmc.Dist.Poisson.logpdf(y, %{mu: Nx.tensor(2.0)}) |> Nx.to_number() |> Float.round(6)
      -1.712059
  """

  @behaviour Exmc.Dist

  @impl true
  def logpdf(y, %{mu: mu}) do
    # y * log(mu) - mu - lgamma(y + 1)
    Nx.multiply(y, Nx.log(mu))
    |> Nx.subtract(mu)
    |> Nx.subtract(Exmc.Math.lgamma(Nx.add(y, c(1.0, y))))
  end

  @impl true
  def support(_params), do: :positive

  @impl true
  def transform(_params), do: :log

  @impl true
  def sample(%{mu: mu}, rng) do
    mu_f = Nx.to_number(mu)
    {value, rng} = knuth_poisson(mu_f, rng)
    {Nx.tensor(value * 1.0), rng}
  end

  # Knuth's algorithm for Poisson sampling (good for small mu)
  defp knuth_poisson(mu, rng) do
    l = :math.exp(-mu)
    knuth_loop(1.0, l, rng, 0)
  end

  defp knuth_loop(p, l, rng, k) do
    {u, rng} = :rand.uniform_s(rng)
    p = p * u

    if p > l do
      knuth_loop(p, l, rng, k + 1)
    else
      {k, rng}
    end
  end
end
