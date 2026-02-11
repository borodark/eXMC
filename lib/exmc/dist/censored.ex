defmodule Exmc.Dist.Censored do
  @moduledoc """
  Censored observation support.

  For right-censored at c: logp = log(1 - CDF(c)) = log(SF(c))
  For left-censored at c:  logp = log(CDF(c))
  For interval [a,b]:      logp = log(CDF(b) - CDF(a))

  Usage via Builder.obs with censoring metadata:

      Builder.obs(ir, "obs", "x", value, censored: :right)
      Builder.obs(ir, "obs", "x", value, censored: :left)
      Builder.obs(ir, "obs", "x", %{lower: a, upper: b}, censored: :interval)
  """

  @doc "Compute censored log-likelihood for Normal distribution."
  def log_likelihood(:right, x, Exmc.Dist.Normal, %{mu: mu, sigma: sigma}) do
    z = Nx.divide(Nx.subtract(x, mu), sigma)
    log_sf(z)
  end

  def log_likelihood(:left, x, Exmc.Dist.Normal, %{mu: mu, sigma: sigma}) do
    z = Nx.divide(Nx.subtract(x, mu), sigma)
    log_cdf(z)
  end

  def log_likelihood(:interval, %{lower: lower, upper: upper}, Exmc.Dist.Normal, %{
        mu: mu,
        sigma: sigma
      }) do
    z_lo = Nx.divide(Nx.subtract(lower, mu), sigma)
    z_hi = Nx.divide(Nx.subtract(upper, mu), sigma)
    Nx.log(Nx.subtract(normal_cdf(z_hi), normal_cdf(z_lo)))
  end

  @doc "Normal CDF: Phi(z) = 0.5 * erfc(-z / sqrt(2))"
  def normal_cdf(z) do
    Nx.multiply(Nx.tensor(0.5), erfc(Nx.negate(Nx.divide(z, Nx.tensor(:math.sqrt(2.0))))))
  end

  defp log_cdf(z), do: Nx.log(normal_cdf(z))

  defp log_sf(z) do
    # log(1 - Phi(z)) = log(Phi(-z)) for numerical stability
    log_cdf(Nx.negate(z))
  end

  @doc """
  Complementary error function approximation (Horner form).
  Max error ~1.5e-7 (Abramowitz & Stegun 7.1.26).
  """
  def erfc(x) do
    abs_x = Nx.abs(x)

    t =
      Nx.divide(Nx.tensor(1.0), Nx.add(Nx.tensor(1.0), Nx.multiply(Nx.tensor(0.3275911), abs_x)))

    coeffs = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429]

    poly =
      Enum.reduce(Enum.reverse(coeffs), Nx.tensor(0.0), fn c, acc ->
        Nx.add(Nx.tensor(c), Nx.multiply(t, acc))
      end)

    result = Nx.multiply(Nx.multiply(t, poly), Nx.exp(Nx.negate(Nx.multiply(abs_x, abs_x))))
    # erfc(-x) = 2 - erfc(x)
    Nx.select(Nx.less(x, Nx.tensor(0.0)), Nx.subtract(Nx.tensor(2.0), result), result)
  end
end
