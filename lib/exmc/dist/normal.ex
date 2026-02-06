defmodule Exmc.Dist.Normal do
  @moduledoc """
  Univariate Normal distribution.

  ## Examples

      iex> x = Nx.tensor(0.0)
      iex> Exmc.Dist.Normal.logpdf(x, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)}) |> Nx.to_number() |> Float.round(6)
      -0.918939
  """

  @behaviour Exmc.Dist

  @impl true
  def logpdf(x, %{mu: mu, sigma: sigma}) do
    two_pi = Nx.tensor(2.0 * :math.pi())
    z = Nx.divide(Nx.subtract(x, mu), sigma)
    z2 = Nx.multiply(z, z)
    log_term = Nx.add(Nx.log(two_pi), Nx.multiply(Nx.tensor(2.0), Nx.log(sigma)))
    Nx.multiply(Nx.tensor(-0.5), Nx.add(z2, log_term))
  end

  @impl true
  def support(_params), do: :real

  @impl true
  def transform(_params), do: nil

  @impl true
  def sample(%{mu: mu, sigma: sigma}, rng) do
    mu_f = Nx.to_number(mu)
    sigma_f = Nx.to_number(sigma)
    {z, rng} = :rand.normal_s(rng)
    value = mu_f + sigma_f * z
    {Nx.tensor(value), rng}
  end
end
