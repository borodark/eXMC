defmodule Exmc.Dist.Uniform01 do
  @moduledoc """
  Uniform distribution on (0,1). Logpdf is constant 0 within support.

  ## Examples

      iex> x = Nx.tensor(0.3)
      iex> Exmc.Dist.Uniform01.logpdf(x, %{}) |> Nx.to_number() |> Float.round(6)
      0.0
  """

  @behaviour Exmc.Dist

  @impl true
  def logpdf(_x, _params) do
    Nx.tensor(0.0)
  end

  @impl true
  def support(_params), do: :unit

  @impl true
  def transform(_params), do: :logit

  @impl true
  def sample(_params, rng) do
    {u, rng} = :rand.uniform_s(rng)
    {Nx.tensor(u), rng}
  end
end
