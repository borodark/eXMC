defmodule Exmc.Dist.Gamma do
  @moduledoc """
  Gamma distribution parameterized by shape (alpha) and rate (beta).

  ## Examples

      iex> x = Nx.tensor(1.0)
      iex> Exmc.Dist.Gamma.logpdf(x, %{alpha: Nx.tensor(2.0), beta: Nx.tensor(1.0)}) |> Nx.to_number() |> Float.round(6)
      -1.0
  """

  @behaviour Exmc.Dist

  @impl true
  def logpdf(x, %{alpha: alpha, beta: beta}) do
    Nx.subtract(
      Nx.add(
        Nx.multiply(Nx.subtract(alpha, Nx.tensor(1.0)), Nx.log(x)),
        Nx.multiply(alpha, Nx.log(beta))
      ),
      Nx.add(
        Nx.multiply(beta, x),
        Exmc.Math.lgamma(alpha)
      )
    )
  end

  @impl true
  def support(_params), do: :positive

  @impl true
  def transform(_params), do: :log

  @impl true
  def sample(%{alpha: alpha, beta: beta}, rng) do
    alpha_f = Nx.to_number(alpha)
    beta_f = Nx.to_number(beta)
    {value, rng} = sample_gamma(alpha_f, beta_f, rng)
    {Nx.tensor(value), rng}
  end

  @doc false
  def sample_gamma(alpha, beta, rng) when alpha >= 1.0 do
    d = alpha - 1.0 / 3.0
    c = 1.0 / :math.sqrt(9.0 * d)
    marsaglia_loop(d, c, beta, rng)
  end

  def sample_gamma(alpha, beta, rng) do
    # Boost: Gamma(alpha, beta) = Gamma(alpha+1, beta) * U^(1/alpha)
    {value, rng} = sample_gamma(alpha + 1.0, beta, rng)
    {u, rng} = :rand.uniform_s(rng)
    {value * :math.pow(u, 1.0 / alpha), rng}
  end

  defp marsaglia_loop(d, c, beta, rng) do
    {x, rng} = :rand.normal_s(rng)
    v = 1.0 + c * x

    if v <= 0.0 do
      marsaglia_loop(d, c, beta, rng)
    else
      v = v * v * v
      {u, rng} = :rand.uniform_s(rng)

      if :math.log(u) < 0.5 * x * x + d - d * v + d * :math.log(v) do
        {d * v / beta, rng}
      else
        marsaglia_loop(d, c, beta, rng)
      end
    end
  end
end
