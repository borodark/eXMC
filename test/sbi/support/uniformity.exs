defmodule Exmc.SBITest.Uniformity do
  @moduledoc """
  Uniformity tests for a simulation-based-calibration rank histogram, plus the
  incomplete-gamma machinery they need.

  Kept as test support rather than library code on purpose: a chi-squared
  survival function is not part of `Exmc.SBI`'s job, and adding one to `lib/`
  to make a test read nicely is how a library acquires a statistics department.

  ## The test, and why chi-squared rather than KS

  SBC produces integer ranks in `0 … L`. Under a correctly calibrated posterior
  they are uniform on those `L + 1` values. The Kolmogorov–Smirnov test is
  built for continuous distributions and is *conservative* on discrete data —
  its p-values are too large, so it under-rejects, which is the wrong direction
  for a gate. Binning the ranks into `J` bins and running Pearson's chi-squared
  is the standard SBC test (Talts et al. 2018; the Stan user's guide gives the
  same recipe) and it is exact in its asymptotics as long as every bin has an
  expected count of at least about five.

  `L + 1` must be divisible by `J`, or the bins have unequal expected counts
  and the statistic is not chi-squared. `bin_ranks/3` raises rather than
  rounding, because a gate whose null distribution is slightly wrong is a gate
  whose reported false-positive rate is a guess.
  """

  @doc """
  Bin ranks in `0..l` into `bins` equal-width bins; returns the counts.
  """
  @spec bin_ranks([non_neg_integer()], pos_integer(), pos_integer()) :: [non_neg_integer()]
  def bin_ranks(ranks, l, bins) do
    width = div(l + 1, bins)

    if width * bins != l + 1 do
      raise ArgumentError,
            "Uniformity.bin_ranks/3: #{l + 1} rank values do not divide into #{bins} equal bins"
    end

    counts = :array.new(bins, default: 0)

    Enum.reduce(ranks, counts, fn r, acc ->
      if r < 0 or r > l do
        raise ArgumentError, "rank #{r} outside 0..#{l}"
      end

      i = min(div(r, width), bins - 1)
      :array.set(i, :array.get(i, acc) + 1, acc)
    end)
    |> :array.to_list()
  end

  @doc """
  Pearson chi-squared p-value for a rank histogram against uniformity.

  Returns `{statistic, df, p_value}`.
  """
  @spec chisq_uniform([non_neg_integer()], pos_integer(), pos_integer()) ::
          {float(), pos_integer(), float()}
  def chisq_uniform(ranks, l, bins) do
    counts = bin_ranks(ranks, l, bins)
    n = length(ranks)
    expected = n / bins

    if expected < 5.0 do
      raise ArgumentError,
            "Uniformity.chisq_uniform/3: expected count per bin is #{Float.round(expected, 2)}; " <>
              "the chi-squared approximation needs at least 5. Use more replicates or fewer bins."
    end

    stat =
      Enum.reduce(counts, 0.0, fn c, acc ->
        d = c - expected
        acc + d * d / expected
      end)

    {stat, bins - 1, chisq_sf(stat, bins - 1)}
  end

  @doc """
  Upper tail of the chi-squared distribution: `P(X > x)` with `df` degrees of
  freedom.
  """
  @spec chisq_sf(float(), pos_integer()) :: float()
  def chisq_sf(x, _df) when x <= 0.0, do: 1.0
  def chisq_sf(x, df), do: gammq(df / 2.0, x / 2.0)

  # Regularised upper incomplete gamma Q(a, x) = 1 - P(a, x).
  # Numerical Recipes: series below x < a + 1, continued fraction above.
  defp gammq(a, x) when x < a + 1.0, do: 1.0 - gser(a, x)
  defp gammq(a, x), do: gcf(a, x)

  defp gser(a, x) do
    gln = lgamma(a)
    do_gser(a, x, 1.0 / a, 1.0 / a, a, 0) * :math.exp(-x + a * :math.log(x) - gln)
  end

  defp do_gser(_a, _x, _del, sum, _ap, n) when n > 500, do: sum

  defp do_gser(a, x, del, sum, ap, n) do
    ap = ap + 1.0
    del = del * x / ap
    sum = sum + del

    if abs(del) < abs(sum) * 1.0e-15, do: sum, else: do_gser(a, x, del, sum, ap, n + 1)
  end

  defp gcf(a, x) do
    gln = lgamma(a)
    tiny = 1.0e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = do_gcf(a, x, b, c, d, d, 1)
    :math.exp(-x + a * :math.log(x) - gln) * h
  end

  defp do_gcf(_a, _x, _b, _c, _d, h, i) when i > 500, do: h

  defp do_gcf(a, x, b, c, d, h, i) do
    an = -i * (i - a)
    b = b + 2.0
    d = an * d + b
    d = if abs(d) < 1.0e-300, do: 1.0e-300, else: d
    c = b + an / c
    c = if abs(c) < 1.0e-300, do: 1.0e-300, else: c
    d = 1.0 / d
    del = d * c
    h = h * del

    if abs(del - 1.0) < 1.0e-15, do: h, else: do_gcf(a, x, b, c, d, h, i + 1)
  end

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

  @doc "Log-gamma, Lanczos g=7."
  @spec lgamma(float()) :: float()
  def lgamma(x) when x > 0.0 do
    [c0 | rest] = @lanczos_coeffs

    series =
      rest
      |> Enum.with_index(1)
      |> Enum.reduce(c0, fn {c, i}, acc -> acc + c / (x - 1.0 + i) end)

    t = x - 1.0 + @lanczos_g + 0.5
    0.5 * :math.log(2.0 * :math.pi()) + (x - 0.5) * :math.log(t) - t + :math.log(series)
  end
end
