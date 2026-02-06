defmodule Exmc.Dist.Exponential do
  @moduledoc """
  Exponential distribution with rate lambda (> 0).

  ## Examples

      iex> x = Nx.tensor(0.5)
      iex> Exmc.Dist.Exponential.logpdf(x, %{lambda: Nx.tensor(2.0)}) |> Nx.to_number() |> Float.round(6)
      -0.306853
  """

  @behaviour Exmc.Dist

  @impl true
  def logpdf(x, %{lambda: lambda}) do
    Nx.subtract(Nx.log(lambda), Nx.multiply(lambda, x))
  end

  @impl true
  def support(_params), do: :positive

  @impl true
  def transform(_params), do: :log

  @impl true
  def sample(%{lambda: lambda}, rng) do
    lambda_f = Nx.to_number(lambda)
    {u, rng} = :rand.uniform_s(rng)
    value = -:math.log(u) / lambda_f
    {Nx.tensor(value), rng}
  end
end
