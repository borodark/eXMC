defmodule Exmc.Dist.Bernoulli do
  import Exmc.Math, only: [c: 2]

  @moduledoc """
  Bernoulli distribution parameterized by probability p.

  Primarily used as an observation likelihood with binary data.

  ## Examples

      iex> y = Nx.tensor(1.0)
      iex> Exmc.Dist.Bernoulli.logpdf(y, %{p: Nx.tensor(0.7)}) |> Nx.to_number() |> Float.round(6)
      -0.356675
  """

  @behaviour Exmc.Dist

  @impl true
  def logpdf(y, %{p: p}) do
    # y * log(p) + (1 - y) * log(1 - p)
    # Clamp p to avoid log(0) = -inf when sigmoid saturates at 0/1
    eps = c(1.0e-7, y)
    p_safe = Nx.clip(p, eps, Nx.subtract(c(1.0, y), eps))

    Nx.add(
      Nx.multiply(y, Nx.log(p_safe)),
      Nx.multiply(Nx.subtract(c(1.0, y), y), Nx.log(Nx.subtract(c(1.0, y), p_safe)))
    )
  end

  @impl true
  def support(_params), do: :unit

  @impl true
  def transform(_params), do: :logit

  @impl true
  def sample(%{p: p}, rng) do
    p_f = Nx.to_number(p)
    {u, rng} = :rand.uniform_s(rng)
    value = if u < p_f, do: 1.0, else: 0.0
    {Nx.tensor(value), rng}
  end
end
