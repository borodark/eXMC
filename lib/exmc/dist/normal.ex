defmodule Exmc.Dist.Normal do
  import Exmc.Math, only: [c: 2]

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
    # Guard sigma > 0 to prevent ArithmeticError on BinaryBackend (Erlang
    # arithmetic throws on divide-by-zero unlike GPU which returns NaN/Inf).
    safe_sigma = Nx.max(sigma, c(1.0e-30, x))
    two_pi = c(2.0 * :math.pi(), x)
    z = Nx.divide(Nx.subtract(x, mu), safe_sigma)
    z2 = Nx.multiply(z, z)
    log_term = Nx.add(Nx.log(two_pi), Nx.multiply(c(2.0, x), Nx.log(safe_sigma)))
    Nx.multiply(c(-0.5, x), Nx.add(z2, log_term))
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
