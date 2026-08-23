defmodule Exmc.Dist.Beta do
  import Exmc.Math, only: [c: 2]

  @moduledoc """
  Beta distribution parameterized by alpha and beta.

  ## Examples

      iex> x = Nx.tensor(0.4)
      iex> Exmc.Dist.Beta.logpdf(x, %{alpha: Nx.tensor(2.0), beta: Nx.tensor(3.0)}) |> Nx.to_number() |> Float.round(4)
      0.547

  Rounded to four places, not six, and that is calibration rather than
  sloppiness. The exact value is 0.546965599060..., which sits close enough to
  the 0.5469655 boundary that the sixth decimal flips on last-bit differences:
  it rounds to 0.546966 standalone and to 0.546965 inside a suite that has
  mutated the global default backend. `Exmc.Math.lgamma/1` is a nine-term
  Lanczos sum, so a one-ulp difference anywhere in it is enough.

  This doctest asserted six places and passed in this tree while failing in the
  applications tree against the identical code. A gate tighter than the
  computation is reproducible does not test precision, it tests which suite you
  are in.
  """

  @behaviour Exmc.Dist

  @impl true
  def logpdf(x, %{alpha: alpha, beta: beta}) do
    Nx.add(
      Nx.add(
        Nx.multiply(Nx.subtract(alpha, c(1.0, x)), Nx.log(x)),
        Nx.multiply(Nx.subtract(beta, c(1.0, x)), Nx.log(Nx.subtract(c(1.0, x), x)))
      ),
      Nx.negate(Exmc.Math.lbeta(alpha, beta))
    )
  end

  @impl true
  def support(_params), do: :unit

  @impl true
  def transform(_params), do: :logit

  @impl true
  def sample(%{alpha: alpha, beta: beta}, rng) do
    alpha_f = Nx.to_number(alpha)
    beta_f = Nx.to_number(beta)
    {x, rng} = Exmc.Dist.Gamma.sample_gamma(alpha_f, 1.0, rng)
    {y, rng} = Exmc.Dist.Gamma.sample_gamma(beta_f, 1.0, rng)
    {Nx.tensor(x / (x + y)), rng}
  end
end
