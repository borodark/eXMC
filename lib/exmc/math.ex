defmodule Exmc.Math do
  @moduledoc """
  Special math functions via Lanczos approximation.

  All ops are pure Nx and differentiable through `Nx.Defn.grad`.
  """

  # Lanczos approximation coefficients (g=7, 9 terms)
  @lanczos_g 7.0
  @lanczos_coeffs [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
  ]

  @doc """
  Log-gamma function via Lanczos approximation.

  Accurate to ~15 digits for Re(x) > 0.5.
  """
  def lgamma(x) do
    # Lanczos: lgamma(x) = 0.5*ln(2*pi) + (x-0.5)*ln(t) - t + ln(Ag(x))
    # where t = x + g - 0.5
    half_log_2pi = Nx.tensor(0.5 * :math.log(2.0 * :math.pi()))

    # t = x + g - 0.5
    t = Nx.add(x, Nx.tensor(@lanczos_g - 0.5))

    # Build Ag(x) = c0 + c1/(x) + c2/(x+1) + ... + c8/(x+7)
    [c0 | rest] = @lanczos_coeffs
    ag = Nx.tensor(c0)

    ag =
      rest
      |> Enum.with_index()
      |> Enum.reduce(ag, fn {c, i}, acc ->
        term = Nx.divide(Nx.tensor(c), Nx.add(x, Nx.tensor(i * 1.0)))
        Nx.add(acc, term)
      end)

    # lgamma = 0.5*ln(2pi) + (x - 0.5)*ln(t) - t + ln(Ag)
    half_log_2pi
    |> Nx.add(Nx.multiply(Nx.subtract(x, Nx.tensor(0.5)), Nx.log(t)))
    |> Nx.subtract(t)
    |> Nx.add(Nx.log(ag))
  end

  @doc """
  Log-beta function: lbeta(a, b) = lgamma(a) + lgamma(b) - lgamma(a+b).
  """
  def lbeta(a, b) do
    lgamma(a)
    |> Nx.add(lgamma(b))
    |> Nx.subtract(lgamma(Nx.add(a, b)))
  end
end
