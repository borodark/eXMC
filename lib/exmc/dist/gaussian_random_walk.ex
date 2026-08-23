defmodule Exmc.Dist.GaussianRandomWalk do
  import Exmc.Math, only: [c: 2]

  @moduledoc """
  Gaussian Random Walk distribution.

  A vector-valued distribution where:
  - x[0] ~ Normal(0, sigma)
  - x[t] ~ Normal(x[t-1], sigma) for t = 1..T-1

  Parameterized by `sigma` (innovation standard deviation).

  ## Examples

      iex> x = Nx.tensor([0.1, 0.2, 0.15])
      iex> params = %{sigma: Nx.tensor(1.0)}
      iex> Exmc.Dist.GaussianRandomWalk.logpdf(x, params) |> Nx.to_number()
  """

  @behaviour Exmc.Dist

  @impl true
  def logpdf(x, %{sigma: sigma}) do
    safe_sigma = Nx.max(sigma, c(1.0e-30, x))
    t = elem(Nx.shape(x), 0)

    # First step: x[0] ~ Normal(0, sigma)
    x_init = Nx.slice(x, [0], [1]) |> Nx.reshape({})

    two_pi = c(2.0 * :math.pi(), x)
    log_sigma = Nx.log(safe_sigma)

    z_init = Nx.divide(x_init, safe_sigma)

    logp_init =
      Nx.multiply(
        c(-0.5, x),
        Nx.add(
          Nx.multiply(z_init, z_init),
          Nx.add(Nx.log(two_pi), Nx.multiply(c(2.0, x), log_sigma))
        )
      )

    if t == 1 do
      logp_init
    else
      # Subsequent steps: x[t] - x[t-1] ~ Normal(0, sigma)
      x_rest = Nx.slice(x, [1], [t - 1])
      x_prev = Nx.slice(x, [0], [t - 1])
      diffs = Nx.subtract(x_rest, x_prev)

      z_steps = Nx.divide(diffs, safe_sigma)

      logp_steps =
        Nx.sum(
          Nx.multiply(
            c(-0.5, x),
            Nx.add(
              Nx.multiply(z_steps, z_steps),
              Nx.add(Nx.log(two_pi), Nx.multiply(c(2.0, x), log_sigma))
            )
          )
        )

      Nx.add(logp_init, logp_steps)
    end
  end

  @impl true
  def support(_params), do: :real

  @impl true
  def transform(_params), do: nil

  @impl true
  def sample(%{sigma: sigma, steps: steps}, rng) do
    sigma_f = Nx.to_number(sigma)

    {values, rng} =
      Enum.map_reduce(1..steps, rng, fn _i, r ->
        :rand.normal_s(r)
      end)

    # Scale by sigma and compute cumulative sum
    increments = Enum.map(values, &(&1 * sigma_f))

    walk =
      Enum.scan(increments, fn inc, prev -> prev + inc end)

    # First element is just the first increment
    walk = [hd(increments) | tl(walk)]
    {Nx.tensor(walk, type: :f64), rng}
  end

  def sample(%{sigma: _sigma}, _rng) do
    raise ArgumentError, "GaussianRandomWalk.sample requires :steps in params"
  end
end
