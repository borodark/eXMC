defmodule Exmc.NUTS.Vulkan.Validator do
  @moduledoc """
  Statistical validation harness for GPU-node chain shaders.

  Runs the same NUTS sampler on the same prior model under both the
  EXLA reference path and the candidate Vulkan-fused-chain path, with
  the same random seed. Compares the resulting posterior samples via
  three layered tests:

    1. **Mean** within 3σ — first-moment agreement.
    2. **Variance** within 3σ — second-moment agreement (catches the
       NUTS capped-leaf-weight / balanced-outer-merge family of bugs
       that produce correct means but wrong variances).
    3. **Two-sample Kolmogorov–Smirnov** rejected at α = 0.001 —
       distribution-shape agreement (asymptotic critical value
       `c(α=0.001) ≈ 1.95` × `sqrt((n+m)/(n·m))`).

  Returns `:ok` only if all three pass. Otherwise `{:error, reason}`
  identifying which check rejected, with the observed and tolerated
  values.

  ## Cauchy

  Cauchy has no defined first or second moment, so for `:cauchy` meta
  the harness substitutes **median + IQR** for mean + variance. KS
  still applies (KS is distribution-free and well-defined for Cauchy).

  ## Usage

      iex> ir = Exmc.Builder.new_ir() |> Exmc.Builder.rv(
      ...>   "x", Exmc.Dist.Normal,
      ...>   %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
      iex> Exmc.NUTS.Vulkan.Validator.validate(ir, {:normal, 0.0, 1.0}, n_warmup: 200, n_samples: 500)
      :ok

  ## Options

    * `:n_warmup`  — warmup iterations per backend (default 500)
    * `:n_samples` — sampling iterations per backend (default 1000)
    * `:seed`      — base PRNG seed (default 42)

  Both backends use the *same* seed; this maximises the signal of any
  divergence between the two paths.
  """

  alias Exmc.{Builder, NUTS.Sampler}

  # 2-sample KS asymptotic critical value at α = 0.001:
  # c(α) ≈ sqrt(-0.5 * ln(α/2)) ≈ 1.9495
  @ks_c_001 1.9495

  # Multiplier on standard error for mean / variance checks (~3σ).
  @sigma_tol 3.0

  # Multiplier for the ANALYTIC checks. Wider than @sigma_tol because those
  # compare one arm against a fixed number rather than two arms against each
  # other, so there is no cancellation of shared sampler noise.
  @analytic_tol 4.0

  @default_opts [n_warmup: 500, n_samples: 1000, seed: 42]

  @doc """
  Validate a candidate Vulkan shader against the EXLA reference path.

  `ir` is a single-RV `Exmc.IR` (built with `Exmc.Builder.new_ir/0` +
  `Exmc.Builder.rv/4`). `vulkan_meta` is the tagged-tuple consumed by
  `Exmc.NUTS.Tree.do_dispatch/10` (e.g. `{:normal, 0.0, 1.0}`).

  Returns `:ok` or `{:error, reason}` where `reason` is a tagged map
  describing which check failed and the observed numbers.
  """
  @spec validate(Exmc.IR.t(), tuple(), keyword()) ::
          :ok | {:error, map()}
  def validate(ir, vulkan_meta, opts \\ []) do
    opts = Keyword.merge(@default_opts, opts)

    exla_samples = run_exla(ir, opts)

    case run_vulkan(ir, vulkan_meta, opts) do
      {:ok, vulkan_samples} ->
        compare(exla_samples, vulkan_samples, vulkan_meta)

      {:error, reason} ->
        {:error, %{check: :backend_unavailable, reason: reason}}
    end
  end

  @doc """
  Run the comparison pipeline on two pre-collected sample lists.
  Exposed so callers can wire in alternative sample sources (e.g.
  for the negative test we feed two different EXLA distributions).

  `meta` only matters for selecting the location-scale check
  (`:cauchy` → median/IQR; otherwise mean/variance).
  """
  @spec compare([number()], [number()], tuple()) :: :ok | {:error, map()}
  def compare(exla_samples, vulkan_samples, meta) do
    cauchy? = match?({:cauchy, _, _, _}, meta) or match?({:cauchy, _, _}, meta)

    location_check =
      if cauchy? do
        check_median(exla_samples, vulkan_samples)
      else
        check_mean(exla_samples, vulkan_samples)
      end

    scale_check =
      if cauchy? do
        check_iqr(exla_samples, vulkan_samples)
      else
        check_variance(exla_samples, vulkan_samples)
      end

    with :ok <- location_check,
         :ok <- scale_check,
         :ok <- check_ks(exla_samples, vulkan_samples) do
      :ok
    end
  end

  # --- Backends ---------------------------------------------------

  defp run_exla(ir, opts) do
    # Force EXLA path: clear any compiler / fused-meta overrides.
    prev_compiler = Application.get_env(:exmc, :compiler)
    prev_meta = Application.get_env(:exmc, :fused_leapfrog_meta)
    prev_norm_meta = Application.get_env(:exmc, :fused_leapfrog_normal_meta)
    prev_force_prec = Application.get_env(:exmc, :force_precision)

    Application.delete_env(:exmc, :compiler)
    Application.delete_env(:exmc, :fused_leapfrog_meta)
    Application.delete_env(:exmc, :fused_leapfrog_normal_meta)

    # When the caller passes `precision: :f32`, force the EXLA path
    # to f32 so it matches the chain shader's working precision.
    # Without this, the validator compares f32 Vulkan against f64
    # EXLA — a precision gap that masks shader correctness for
    # fat-tailed distributions (Cauchy especially). See
    # WORKSTREAM_W7 Stage 2 notes for the full diagnosis.
    case Keyword.get(opts, :precision) do
      :f32 -> Application.put_env(:exmc, :force_precision, :f32)
      :f64 -> Application.put_env(:exmc, :force_precision, :f64)
      _ -> :ok
    end

    try do
      sample_to_list(ir, opts)
    after
      restore(:compiler, prev_compiler)
      restore(:fused_leapfrog_meta, prev_meta)
      restore(:fused_leapfrog_normal_meta, prev_norm_meta)
      restore(:force_precision, prev_force_prec)
    end
  end

  defp run_vulkan(ir, meta, opts) do
    if Code.ensure_loaded?(Nx.Vulkan) do
      prev_compiler = Application.get_env(:exmc, :compiler)
      prev_meta = Application.get_env(:exmc, :fused_leapfrog_meta)

      Application.put_env(:exmc, :compiler, :vulkan)
      Application.put_env(:exmc, :fused_leapfrog_meta, meta)

      try do
        {:ok, sample_to_list(ir, opts)}
      after
        restore(:compiler, prev_compiler)
        restore(:fused_leapfrog_meta, prev_meta)
      end
    else
      {:error, :nx_vulkan_not_loaded}
    end
  end

  defp sample_to_list(ir, opts) do
    {trace, _stats} =
      Sampler.sample(ir, %{},
        num_warmup: Keyword.fetch!(opts, :n_warmup),
        num_samples: Keyword.fetch!(opts, :n_samples),
        seed: Keyword.fetch!(opts, :seed)
      )

    [{_id, samples}] = Enum.to_list(trace)
    samples |> Nx.to_flat_list() |> Enum.map(&to_finite/1)
  end

  defp to_finite(x) when is_number(x), do: x * 1.0
  # Atom returns from special IEEE values — propagate as NaN-ish so
  # downstream stats fail loudly instead of silently dropping points.
  defp to_finite(:nan), do: :nan
  defp to_finite(:infinity), do: :infinity
  defp to_finite(:neg_infinity), do: :neg_infinity

  defp restore(key, nil), do: Application.delete_env(:exmc, key)
  defp restore(key, v), do: Application.put_env(:exmc, key, v)

  # --- Checks -----------------------------------------------------

  @doc """
  Compare sample means; fail if they disagree by more than `@sigma_tol`
  combined standard errors of the mean.
  """
  @spec check_mean([number()], [number()]) :: :ok | {:error, map()}
  def check_mean(a, b) do
    {mean_a, var_a} = mean_var(a)
    {mean_b, var_b} = mean_var(b)

    se_a = :math.sqrt(var_a / length(a))
    se_b = :math.sqrt(var_b / length(b))
    se_combined = :math.sqrt(se_a * se_a + se_b * se_b)

    diff = abs(mean_a - mean_b)
    tol = @sigma_tol * max(se_combined, 1.0e-12)

    if diff <= tol do
      :ok
    else
      {:error,
       %{
         check: :mean,
         exla: mean_a,
         vulkan: mean_b,
         diff: diff,
         tol: tol,
         se_combined: se_combined
       }}
    end
  end

  @doc """
  Compare sample variances; fail if they disagree by more than
  `@sigma_tol` combined standard errors of the variance estimator
  (Gaussian-asymptotic SE = `var · sqrt(2/(n-1))`).
  """
  @spec check_variance([number()], [number()]) :: :ok | {:error, map()}
  def check_variance(a, b) do
    {_, var_a} = mean_var(a)
    {_, var_b} = mean_var(b)

    n_a = length(a)
    n_b = length(b)

    se_a = var_a * :math.sqrt(2.0 / max(n_a - 1, 1))
    se_b = var_b * :math.sqrt(2.0 / max(n_b - 1, 1))
    se_combined = :math.sqrt(se_a * se_a + se_b * se_b)

    diff = abs(var_a - var_b)
    tol = @sigma_tol * max(se_combined, 1.0e-12)

    if diff <= tol do
      :ok
    else
      {:error,
       %{
         check: :variance,
         exla: var_a,
         vulkan: var_b,
         diff: diff,
         tol: tol,
         se_combined: se_combined
       }}
    end
  end

  # --- Analytic checks --------------------------------------------
  #
  # Everything above this line is DIFFERENTIAL: it asks whether two arms
  # agree. That is structurally blind to any defect the two arms share — and
  # both arms run the same NUTS tree, so a defect in the tree moves them
  # identically and the comparison passes. Two real defects lived behind a
  # green mean/variance/KS suite for weeks because of exactly this. The checks
  # below compare ONE arm against the distribution's own moments, which is the
  # only thing that can see a shared defect.

  @doc """
  Effective sample size by Geyer's initial monotone positive sequence.

  Needed because NUTS output is autocorrelated: an iid standard error
  understates the true one, and a gate that is too tight rejects correct runs.
  A validator that cries wolf gets switched off, which is the same way a
  vacuous check survives — nobody reads a harness they do not trust.

  Returns a float in `[1, n]`. Capped at `n`: NUTS is often antithetic
  (negative odd-lag autocorrelation) which can push the true ESS above `n`,
  but exploiting that would *narrow* the tolerance, and this gate should err
  wide.
  """
  @spec ess([number()]) :: float()
  def ess(xs) do
    n = length(xs)
    {mean, var} = mean_var(xs)

    cond do
      n < 8 -> n * 1.0
      var <= 0.0 -> n * 1.0
      true -> ess_from_autocorr(List.to_tuple(xs), n, mean, var)
    end
  end

  defp ess_from_autocorr(arr, n, mean, var) do
    rho = fn lag ->
      sum =
        Enum.reduce(0..(n - lag - 1), 0.0, fn i, acc ->
          acc + (elem(arr, i) - mean) * (elem(arr, i + lag) - mean)
        end)

      sum / ((n - lag) * var)
    end

    # Geyer: sum successive PAIRS of autocorrelations, stop at the first
    # non-positive pair, and keep the sequence monotone decreasing. Pairing is
    # what makes this robust to the alternating-sign autocorrelation NUTS
    # produces — truncating on the first negative *single* lag would stop
    # almost immediately on an antithetic chain and report a wildly optimistic
    # ESS.
    max_lag = min(n - 2, 512)
    sum = geyer_pairs(rho, 0, max_lag, 0.0, :infinity)

    tau = max(1.0 + 2.0 * sum, 1.0)
    min(n / tau, n * 1.0)
  end

  defp geyer_pairs(rho, k, max_lag, acc, prev) do
    l1 = 2 * k + 1
    l2 = l1 + 1

    if l2 > max_lag do
      acc
    else
      pair = rho.(l1) + rho.(l2)
      pair = if prev == :infinity, do: pair, else: min(pair, prev)

      if pair <= 0.0,
        do: acc,
        else: geyer_pairs(rho, k + 1, max_lag, acc + pair, pair)
    end
  end

  @doc """
  Analytic moments implied by a chain-shader `meta`, or `:unknown`.

  Cauchy has no moments, so it reports median and IQR instead, which are
  defined: median `loc`, IQR `2 * scale`.
  """
  @spec analytic_moments(tuple()) ::
          {:moments, %{mean: float(), var: float()}}
          | {:quantiles, %{median: float(), iqr: float()}}
          | :unknown
  def analytic_moments({:normal, mu, sigma}),
    do: {:moments, %{mean: mu * 1.0, var: sigma * sigma * 1.0}}

  def analytic_moments({:exponential, lambda}) when lambda > 0,
    do: {:moments, %{mean: 1.0 / lambda, var: 1.0 / (lambda * lambda)}}

  def analytic_moments({:halfnormal, sigma, _log_const}), do: half_normal_moments(sigma)
  def analytic_moments({:half_normal, sigma}), do: half_normal_moments(sigma)

  def analytic_moments({:lognormal, mu, sigma}) do
    m = :math.exp(mu + sigma * sigma / 2.0)
    v = (:math.exp(sigma * sigma) - 1.0) * :math.exp(2.0 * mu + sigma * sigma)
    {:moments, %{mean: m, var: v}}
  end

  # StudentT's variance is nu/(nu-2) and only exists for nu > 2; below that the
  # moment check would be comparing against an undefined quantity.
  def analytic_moments({:studentt, mu, sigma, nu, _c}) when nu > 2,
    do: {:moments, %{mean: mu * 1.0, var: sigma * sigma * nu / (nu - 2.0)}}

  def analytic_moments({:cauchy, loc, scale, _log_pi_scale}),
    do: {:quantiles, %{median: loc * 1.0, iqr: 2.0 * scale}}

  def analytic_moments({:cauchy, loc, scale}),
    do: {:quantiles, %{median: loc * 1.0, iqr: 2.0 * scale}}

  # Weibull needs the gamma function for its moments; not worth pulling one in
  # for a check that reports :unknown gracefully.
  def analytic_moments(_meta), do: :unknown

  defp half_normal_moments(sigma) do
    {:moments,
     %{
       mean: sigma * :math.sqrt(2.0 / :math.pi()),
       var: sigma * sigma * (1.0 - 2.0 / :math.pi())
     }}
  end

  @doc """
  Check one arm's samples against the distribution's analytic moments.

  `label` names the arm in any error, because the whole point is knowing
  *which* arm is wrong — or, when both are, that the defect is shared and
  therefore not in the backend at all.

  Returns `:ok` when the meta has no closed-form moments.
  """
  @spec check_analytic([number()], atom(), tuple()) :: :ok | {:error, map()}
  def check_analytic(samples, label, meta) do
    case analytic_moments(meta) do
      :unknown ->
        # Visible, not silent. A check that quietly returns :ok for families it
        # cannot evaluate reports the same thing as a check that ran and
        # passed, which is how a vacuous verdict survives.
        require Logger
        Logger.info("[Validator] no analytic moments for #{inspect(meta)} — moment check SKIPPED")
        :ok

      {:moments, %{mean: tm, var: tv}} ->
        {m, v} = mean_var(samples)
        n = ess(samples)

        se_m = :math.sqrt(max(tv, 1.0e-30) / n)
        se_v = variance_se(samples, n)

        cond do
          abs(m - tm) > @analytic_tol * se_m ->
            {:error,
             %{
               check: :analytic_mean,
               arm: label,
               truth: tm,
               got: m,
               diff: abs(m - tm),
               tol: @analytic_tol * se_m,
               ess: n
             }}

          abs(v - tv) > @analytic_tol * se_v ->
            {:error,
             %{
               check: :analytic_variance,
               arm: label,
               truth: tv,
               got: v,
               diff: abs(v - tv),
               tol: @analytic_tol * se_v,
               ess: n
             }}

          true ->
            :ok
        end

      {:quantiles, %{median: tmed, iqr: tiqr}} ->
        med = median(samples)
        got_iqr = iqr(samples)
        n = ess(samples)

        se_med = tiqr / 1.349 / :math.sqrt(n)
        # The IQR's own SE has no clean closed form across distributions; the
        # 25% band is the same conservative proxy check_iqr/2 uses. Checking it
        # at all matters: without it the Cauchy branch verified location and
        # said nothing whatever about scale, so a sampler that got the centre
        # right and the spread wrong passed.
        se_iqr = 0.25 * tiqr / :math.sqrt(n)

        cond do
          abs(med - tmed) > @analytic_tol * se_med ->
            {:error,
             %{
               check: :analytic_median,
               arm: label,
               truth: tmed,
               got: med,
               tol: @analytic_tol * se_med,
               ess: n
             }}

          abs(got_iqr - tiqr) > @analytic_tol * se_iqr ->
            {:error,
             %{
               check: :analytic_iqr,
               arm: label,
               truth: tiqr,
               got: got_iqr,
               tol: @analytic_tol * se_iqr,
               ess: n
             }}

          true ->
            :ok
        end
    end
  end

  # Standard error of the sample variance, WITHOUT assuming normality.
  #
  # The Gaussian form `var * sqrt(2/(n-1))` used by check_variance/2 comes from
  # Var(s²) = 2σ⁴/n, which holds only when the fourth central moment is 3σ⁴.
  # The general result is Var(s²) = (μ₄ − σ⁴)/n, so for any heavier-tailed
  # distribution the Gaussian form UNDERSTATES the true standard error — by
  # exactly 2.00x for Exponential and StudentT(5), 1.99x for LogNormal(0, 0.5),
  # since all three have μ₄ = 9σ⁴ or near it.
  #
  # A gate two times too tight rejects correct runs. Measured at n=800 and 4σ
  # over 20,000 replications of *iid* draws: Exponential(1) false-rejected
  # 4.30% of the time against a nominal 0.0063%.
  #
  # μ₄ is estimated from the sample rather than added to the analytic table:
  # it generalises to every family, including the ones analytic_moments/1
  # reports as :unknown, and the estimate's own error is second-order here.
  defp variance_se(xs, ess) do
    {m, v} = mean_var(xs)
    n = length(xs)

    m4 =
      Enum.reduce(xs, 0.0, fn x, acc ->
        d = x - m
        acc + d * d * d * d
      end) / n

    max(:math.sqrt(max(m4 - v * v, 0.0) / max(ess, 1.0)), 1.0e-30)
  end

  @doc """
  Two-sample Kolmogorov–Smirnov test.

  Computes the maximum absolute difference between the two empirical
  CDFs, then compares against the asymptotic critical value at
  α = 0.001:

      D > c(α) · sqrt((n + m) / (n · m))     where c(0.001) ≈ 1.95

  Returns `:ok` if the test does *not* reject (i.e. the samples are
  statistically indistinguishable at this α), `{:error, ...}` if it
  rejects.
  """
  @spec check_ks([number()], [number()]) :: :ok | {:error, map()}
  def check_ks(a, b) do
    sa = Enum.sort(a)
    sb = Enum.sort(b)
    n = length(sa)
    m = length(sb)

    d = ks_statistic(sa, sb, n, m)
    crit = @ks_c_001 * :math.sqrt((n + m) / (n * m))

    if d <= crit do
      :ok
    else
      # Asymptotic p-value approximation (Kolmogorov 1933 series, first term).
      # Used purely for diagnostic reporting — the gate is `d <= crit`.
      lambda = d * :math.sqrt(n * m / (n + m))
      p = 2.0 * :math.exp(-2.0 * lambda * lambda)

      {:error,
       %{
         check: :ks,
         d: d,
         crit: crit,
         alpha: 0.001,
         approx_p: p,
         n: n,
         m: m
       }}
    end
  end

  # Median + IQR variants for Cauchy (no defined moments).

  defp check_median(a, b) do
    med_a = median(a)
    med_b = median(b)
    # Bootstrap-free SE proxy for the median of a continuous
    # distribution: scale-IQR / sqrt(n) (rough but fine as a 3σ gate).
    se_a = iqr(a) / 1.349 / :math.sqrt(length(a))
    se_b = iqr(b) / 1.349 / :math.sqrt(length(b))
    se_combined = :math.sqrt(se_a * se_a + se_b * se_b)

    diff = abs(med_a - med_b)
    tol = @sigma_tol * max(se_combined, 1.0e-12)

    if diff <= tol do
      :ok
    else
      {:error, %{check: :median, exla: med_a, vulkan: med_b, diff: diff, tol: tol}}
    end
  end

  defp check_iqr(a, b) do
    iqr_a = iqr(a)
    iqr_b = iqr(b)
    # SE of IQR has no closed form independent of distribution; use
    # 25% of the IQR as a conservative tolerance band per side.
    se = 0.25 * (iqr_a + iqr_b) / 2.0 / :math.sqrt(min(length(a), length(b)))
    diff = abs(iqr_a - iqr_b)
    tol = @sigma_tol * max(se, 1.0e-12)

    if diff <= tol do
      :ok
    else
      {:error, %{check: :iqr, exla: iqr_a, vulkan: iqr_b, diff: diff, tol: tol}}
    end
  end

  # --- Stat helpers -----------------------------------------------

  @doc false
  def mean_var(xs) do
    n = length(xs)
    sum = Enum.reduce(xs, 0.0, &(&1 + &2))
    mean = sum / n
    sq = Enum.reduce(xs, 0.0, fn x, acc -> acc + (x - mean) * (x - mean) end)
    var = sq / max(n - 1, 1)
    {mean, var}
  end

  defp median(xs) do
    sorted = Enum.sort(xs)
    n = length(sorted)
    mid = div(n, 2)

    if rem(n, 2) == 1 do
      Enum.at(sorted, mid)
    else
      (Enum.at(sorted, mid - 1) + Enum.at(sorted, mid)) / 2.0
    end
  end

  defp iqr(xs) do
    sorted = Enum.sort(xs)
    n = length(sorted)
    q1 = Enum.at(sorted, div(n, 4))
    q3 = Enum.at(sorted, div(3 * n, 4))
    q3 - q1
  end

  # KS statistic via merge-walk over the two sorted samples.
  #
  # Walk the union of the two sorted samples in order. After processing
  # each value, the empirical CDF on side A is rank_a/n_a and similarly
  # for B. The KS statistic is the supremum of |F_a - F_b|.
  defp ks_statistic(sa, sb, n, m) do
    ks_walk(sa, sb, 0, 0, n, m, 0.0)
  end

  defp ks_walk([], _, _, _, _, _, max_d), do: max_d

  defp ks_walk(_, [], _, _, _, _, max_d), do: max_d

  defp ks_walk([ha | ta] = a, [hb | tb] = b, ra, rb, n, m, max_d) do
    cond do
      ha < hb ->
        ra2 = ra + 1
        d = abs(ra2 / n - rb / m)
        ks_walk(ta, b, ra2, rb, n, m, max(max_d, d))

      ha > hb ->
        rb2 = rb + 1
        d = abs(ra / n - rb2 / m)
        ks_walk(a, tb, ra, rb2, n, m, max(max_d, d))

      true ->
        # Tie: advance both before evaluating the step diff (matches
        # the standard "step at end of tied block" definition).
        {a2, ra2} = consume_ties(a, ha, ra)
        {b2, rb2} = consume_ties(b, ha, rb)
        d = abs(ra2 / n - rb2 / m)
        ks_walk(a2, b2, ra2, rb2, n, m, max(max_d, d))
    end
  end

  defp consume_ties([h | t], v, r) when h == v, do: consume_ties(t, v, r + 1)
  defp consume_ties(rest, _v, r), do: {rest, r}
end
