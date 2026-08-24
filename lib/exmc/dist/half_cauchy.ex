defmodule Exmc.Dist.HalfCauchy do
  import Exmc.Math, only: [c: 2]

  @moduledoc """
  Half-Cauchy distribution with scale > 0 (location fixed at 0).

  PyMC's default weakly-informative prior for scale parameters.

  ## Examples

      iex> x = Nx.tensor(1.0)
      iex> Exmc.Dist.HalfCauchy.logpdf(x, %{scale: Nx.tensor(1.0)}) |> Nx.to_number() |> Float.round(6)
      -0.451583
  """

  @behaviour Exmc.Dist

  @impl true
  def logpdf(x, %{scale: scale}) do
    safe_scale = Nx.max(scale, c(1.0e-30, x))
    z = Nx.divide(x, safe_scale)
    z2 = Nx.multiply(z, z)

    c(:math.log(2.0 / :math.pi()), x)
    |> Nx.subtract(Nx.log(safe_scale))
    |> Nx.subtract(Nx.log(Nx.add(c(1.0, x), z2)))
  end

  @impl true
  def support(_params), do: :positive

  @impl true
  def transform(_params), do: :log

  @impl true
  def sample(%{scale: scale}, rng) do
    scale_f = Nx.to_number(scale)
    {u, rng} = :rand.uniform_s(rng)
    # U ~ Uniform(0, 0.5) for half-Cauchy (positive side only)
    value = scale_f * abs(:math.tan(:math.pi() * u * 0.5))
    {Nx.tensor(value), rng}
  end
end
