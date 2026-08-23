defmodule Exmc.Dist.TruncatedNormal do
  import Exmc.Math, only: [c: 2]

  @moduledoc """
  Truncated Normal distribution with bounds [lower, upper].

  ## Examples

      iex> x = Nx.tensor(0.0)
      iex> params = %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0), lower: Nx.tensor(-1.0), upper: Nx.tensor(1.0)}
      iex> Exmc.Dist.TruncatedNormal.logpdf(x, params) |> Nx.to_number() |> Float.round(4)
      -0.2676
  """

  @behaviour Exmc.Dist

  @impl true
  def logpdf(x, %{mu: mu, sigma: sigma, lower: lower, upper: upper}) do
    safe_sigma = Nx.max(sigma, c(1.0e-30, x))
    # Normal logpdf
    z = Nx.divide(Nx.subtract(x, mu), safe_sigma)
    z2 = Nx.multiply(z, z)
    two_pi = c(2.0 * :math.pi(), x)
    log_term = Nx.add(Nx.log(two_pi), Nx.multiply(c(2.0, x), Nx.log(safe_sigma)))
    normal_logpdf = Nx.multiply(c(-0.5, x), Nx.add(z2, log_term))

    # Normalizing constant: log(Phi((upper-mu)/sigma) - Phi((lower-mu)/sigma))
    alpha = Nx.divide(Nx.subtract(lower, mu), safe_sigma)
    beta = Nx.divide(Nx.subtract(upper, mu), safe_sigma)
    log_norm = Nx.log(Nx.subtract(normal_cdf(beta), normal_cdf(alpha)))

    Nx.subtract(normal_logpdf, log_norm)
  end

  defp normal_cdf(z) do
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    Nx.multiply(
      Nx.tensor(0.5),
      Nx.add(Nx.tensor(1.0), Nx.erf(Nx.divide(z, Nx.tensor(:math.sqrt(2.0)))))
    )
  end

  @impl true
  def support(_params), do: :real

  @impl true
  def transform(_params), do: nil

  @impl true
  def sample(%{mu: mu, sigma: sigma, lower: lower, upper: upper}, rng) do
    mu_f = Nx.to_number(mu)
    sigma_f = Nx.to_number(sigma)
    lower_f = Nx.to_number(lower)
    upper_f = Nx.to_number(upper)

    # Rejection sampling from Normal
    {value, rng} = rejection_sample(mu_f, sigma_f, lower_f, upper_f, rng)
    {Nx.tensor(value), rng}
  end

  defp rejection_sample(mu, sigma, lower, upper, rng) do
    {z, rng} = :rand.normal_s(rng)
    value = mu + sigma * z

    if value >= lower and value <= upper do
      {value, rng}
    else
      rejection_sample(mu, sigma, lower, upper, rng)
    end
  end
end
