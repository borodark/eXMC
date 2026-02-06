defmodule Exmc.Dist.HalfNormal do
  @moduledoc """
  Half-Normal distribution with sigma > 0.

  ## Examples

      iex> x = Nx.tensor(0.5)
      iex> Exmc.Dist.HalfNormal.logpdf(x, %{sigma: Nx.tensor(1.0)}) |> Nx.to_number() |> Float.round(6)
      -0.350791
  """

  @behaviour Exmc.Dist

  @impl true
  def logpdf(x, %{sigma: sigma}) do
    two_pi = Nx.tensor(2.0 * :math.pi())
    z = Nx.divide(x, sigma)
    z2 = Nx.multiply(z, z)
    base = Nx.multiply(Nx.tensor(-0.5), Nx.add(z2, Nx.log(two_pi)))
    Nx.add(base, Nx.subtract(Nx.log(Nx.tensor(2.0)), Nx.log(sigma)))
  end

  @impl true
  def support(_params), do: :positive

  @impl true
  def transform(_params), do: :softplus

  @impl true
  def sample(%{sigma: sigma}, rng) do
    sigma_f = Nx.to_number(sigma)
    {z, rng} = :rand.normal_s(rng)
    value = abs(z) * sigma_f
    {Nx.tensor(value), rng}
  end
end
