defmodule Exmc.Dist.Cauchy do
  @moduledoc """
  Cauchy distribution parameterized by loc and scale.

  ## Examples

      iex> x = Nx.tensor(0.0)
      iex> Exmc.Dist.Cauchy.logpdf(x, %{loc: Nx.tensor(0.0), scale: Nx.tensor(1.0)}) |> Nx.to_number() |> Float.round(6)
      -1.14473
  """

  @behaviour Exmc.Dist

  @impl true
  def logpdf(x, %{loc: loc, scale: scale}) do
    z = Nx.divide(Nx.subtract(x, loc), scale)
    z2 = Nx.multiply(z, z)

    Nx.tensor(-:math.log(:math.pi()))
    |> Nx.subtract(Nx.log(scale))
    |> Nx.subtract(Nx.log(Nx.add(Nx.tensor(1.0), z2)))
  end

  @impl true
  def support(_params), do: :real

  @impl true
  def transform(_params), do: nil

  @impl true
  def sample(%{loc: loc, scale: scale}, rng) do
    loc_f = Nx.to_number(loc)
    scale_f = Nx.to_number(scale)
    {u, rng} = :rand.uniform_s(rng)
    value = loc_f + scale_f * :math.tan(:math.pi() * (u - 0.5))
    {Nx.tensor(value), rng}
  end
end
