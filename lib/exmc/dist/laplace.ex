defmodule Exmc.Dist.Laplace do
  import Exmc.Math, only: [c: 2]

  @moduledoc """
  Laplace distribution parameterized by mu (location) and b (scale).

  ## Examples

      iex> x = Nx.tensor(0.0)
      iex> Exmc.Dist.Laplace.logpdf(x, %{mu: Nx.tensor(0.0), b: Nx.tensor(1.0)}) |> Nx.to_number() |> Float.round(6)
      -0.693147
  """

  @behaviour Exmc.Dist

  @impl true
  def logpdf(x, %{mu: mu, b: b}) do
    safe_b = Nx.max(b, c(1.0e-30, x))

    Nx.negate(Nx.log(Nx.multiply(c(2.0, x), safe_b)))
    |> Nx.subtract(Nx.divide(Nx.abs(Nx.subtract(x, mu)), safe_b))
  end

  @impl true
  def support(_params), do: :real

  @impl true
  def transform(_params), do: nil

  @impl true
  def sample(%{mu: mu, b: b}, rng) do
    mu_f = Nx.to_number(mu)
    b_f = Nx.to_number(b)
    {u, rng} = :rand.uniform_s(rng)
    value = mu_f - b_f * sign(u - 0.5) * :math.log(1.0 - 2.0 * abs(u - 0.5))
    {Nx.tensor(value), rng}
  end

  defp sign(x) when x > 0, do: 1.0
  defp sign(x) when x < 0, do: -1.0
  defp sign(_), do: 0.0
end
