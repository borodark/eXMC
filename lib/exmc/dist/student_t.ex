defmodule Exmc.Dist.StudentT do
  import Exmc.Math, only: [c: 2]

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
    safe_scale = Nx.max(scale, c(1.0e-30, x))
    safe_df = Nx.max(df, c(1.0e-30, x))
    z = Nx.divide(Nx.subtract(x, loc), safe_scale)
    z2 = Nx.multiply(z, z)

    half_dfp1 = Nx.divide(Nx.add(safe_df, c(1.0, x)), c(2.0, x))
    half_df = Nx.divide(safe_df, c(2.0, x))

    Exmc.Math.lgamma(half_dfp1)
    |> Nx.subtract(Exmc.Math.lgamma(half_df))
    |> Nx.subtract(Nx.multiply(c(0.5, x), Nx.log(Nx.multiply(safe_df, c(:math.pi(), x)))))
    |> Nx.subtract(Nx.log(safe_scale))
    |> Nx.subtract(Nx.multiply(half_dfp1, Nx.log(Nx.add(c(1.0, x), Nx.divide(z2, safe_df)))))
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
