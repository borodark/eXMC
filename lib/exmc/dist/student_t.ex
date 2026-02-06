defmodule Exmc.Dist.StudentT do
  @moduledoc """
  Student's t-distribution parameterized by df (degrees of freedom), loc, and scale.

  ## Examples

      iex> x = Nx.tensor(0.0)
      iex> Exmc.Dist.StudentT.logpdf(x, %{df: Nx.tensor(3.0), loc: Nx.tensor(0.0), scale: Nx.tensor(1.0)}) |> Nx.to_number() |> Float.round(6)
      -1.000889
  """

  @behaviour Exmc.Dist

  @impl true
  def logpdf(x, %{df: df, loc: loc, scale: scale}) do
    z = Nx.divide(Nx.subtract(x, loc), scale)
    z2 = Nx.multiply(z, z)

    half_dfp1 = Nx.divide(Nx.add(df, Nx.tensor(1.0)), Nx.tensor(2.0))
    half_df = Nx.divide(df, Nx.tensor(2.0))

    Exmc.Math.lgamma(half_dfp1)
    |> Nx.subtract(Exmc.Math.lgamma(half_df))
    |> Nx.subtract(Nx.multiply(Nx.tensor(0.5), Nx.log(Nx.multiply(df, Nx.tensor(:math.pi())))))
    |> Nx.subtract(Nx.log(scale))
    |> Nx.subtract(Nx.multiply(half_dfp1, Nx.log(Nx.add(Nx.tensor(1.0), Nx.divide(z2, df)))))
  end

  @impl true
  def support(_params), do: :real

  @impl true
  def transform(_params), do: nil

  @impl true
  def sample(%{df: df, loc: loc, scale: scale}, rng) do
    df_f = Nx.to_number(df)
    loc_f = Nx.to_number(loc)
    scale_f = Nx.to_number(scale)
    {z, rng} = :rand.normal_s(rng)
    {chi2, rng} = Exmc.Dist.Gamma.sample_gamma(df_f / 2.0, 0.5, rng)
    value = loc_f + scale_f * z / :math.sqrt(chi2 / df_f)
    {Nx.tensor(value), rng}
  end
end
