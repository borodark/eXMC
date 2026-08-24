defmodule Exmc.NUTS.Vulkan.Validator do
  @moduledoc """
  Statistical validation harness for GPU-node chain shaders.

  Runs the same NUTS sampler on the same prior model under both an
  independent reference path and the candidate Vulkan-fused-chain path,
  with the same random seed. Compares the resulting posterior samples via
  three layered tests:

  The reference is `:exla` where EXLA is loadable and `:none` (pure-CPU
  `Nx.BinaryBackend`) otherwise — see `reference/0`. It is never the
  candidate backend. It used to be: the arm was selected by clearing
  `:exmc, :compiler` and letting auto-detection run, which resolves to
  `Nx.Vulkan` on any host without EXLA. On the FreeBSD fleet, where EXLA
  does not exist at all, this harness spent three weeks comparing Vulkan
  against Vulkan and reporting that it agreed.

    1. **Mean** within 3σ — first-moment agreement.
    2. **Variance** within 3σ — second-moment agreement (catches the
       NUTS capped-leaf-weight / balanced-outer-merge family of bugs
       that produce correct means but wrong variances).
    3. **Two-sample Kolmogorov–Smirnov** rejected at α = 0.001 —
       distribution-shape agreement (asymptotic critical value
       `c(α=0.001) ≈ 1.95` × `sqrt((n+m)/(n·m))`).
    4. **Analytic moments**, on *each* arm independently — the only check
       here that can fail for a defect the two arms SHARE. The first three
       all ask "do these agree?", so a bug in the host NUTS tree moves both
       arms identically and passes them. See `check_analytic/3`.

  All standard errors divide by effective sample size, not by `length/1` —
  these are Markov chains. See `ess/1` for what the iid assumption cost.

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

  # Multiplier for the ANALYTIC check. Wider than @sigma_tol because it is a
  # one-sample test against a fixed truth rather than a two-sample comparison,
  # and because ESS is itself estimated — a noisy denominator on an absolute
  # gate is how you get a flaky suite. 4σ still catches the defect this check
  # exists for by a wide margin: the invalid-doubling bug put Normal(0,1)'s
  # variance 45% high, which is ~14σ at n=800.
  @analytic_tol 4.0

  # The widest variance error this gate is allowed to call a pass.
  #
  # If `@analytic_tol * se` exceeds this fraction of the true variance, the
  # chain cannot resolve an error of that size and the gate is not measuring
  # anything — it is skipped and says so, rather than returning a verdict it
  # has no power to support. 50% is deliberately generous: the point is to
  # catch the cases where the band is most of the answer, not to be strict.
  @max_resolvable 0.5

  @default_opts [n_warmup: 500, n_samples: 1000, seed: 42]

  @doc """
  Validate a candidate Vulkan shader against an independent reference path
  (`reference/0` — `:exla`, or `:none` where EXLA is absent).

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

    ref_samples = run_reference(ir, opts)

    case run_vulkan(ir, vulkan_meta, opts) do
      {:ok, vulkan_samples} ->
        case compare(ref_samples, vulkan_samples, vulkan_meta) do
          :ok -> :ok
          {:error, reason} -> {:error, Map.put(reason, :reference, reference())}
        end

      {:error, reason} ->
        {:error, %{check: :backend_unavailable, reason: reason}}
    end
  end

  @doc """
  Run the comparison pipeline on two pre-collected sample lists.
  Exposed so callers can wire in alternative sample sources (e.g.
  for the negative test we feed two different reference distributions).

  `meta` only matters for selecting the location-scale check
  (`:cauchy` → median/IQR; otherwise mean/variance).
  """
  @spec compare([number()], [number()], tuple()) :: :ok | {:error, map()}
  def compare(reference_samples, vulkan_samples, meta) do
    cauchy? = match?({:cauchy, _, _, _}, meta) or match?({:cauchy, _, _}, meta)

    location_check =
      if cauchy? do
        check_median(reference_samples, vulkan_samples)
      else
        check_mean(reference_samples, vulkan_samples)
      end

    scale_check =
      if cauchy? do
        check_iqr(reference_samples, vulkan_samples, meta)
      else
        check_variance(reference_samples, vulkan_samples)
      end

    # The analytic check runs on BOTH arms, and it is the only one here that
    # can fail for a defect the two arms share. location/scale/KS all ask
    # "do these agree?", so a bug in the host NUTS tree — which moves both arms
    # identically — passes them unanimously. That is exactly what happened: an
    # invalid doubling merged post-U-turn states into the trajectory and put
    # Normal(0,1)'s variance at ~1.45 against a true 1.0 in both arms, under a
    # green comparison, until someone thought to check against the
    # distribution itself.
    #
    # Ordered last so a genuine backend divergence is still reported as such:
    # if the arms disagree, that is the more specific finding.
    with :ok <- location_check,
         :ok <- scale_check,
         :ok <- check_ks(reference_samples, vulkan_samples),
         :ok <- check_analytic(reference_samples, :reference, meta),
         :ok <- check_analytic(vulkan_samples, :vulkan, meta) do
      :ok
    end
  end

  # --- Backends ---------------------------------------------------

  @doc """
  The compiler this harness uses for the REFERENCE arm on this host.

  `:exla` where EXLA is loadable, `:none` (pure-CPU `Nx.BinaryBackend`)
  otherwise. Never `:vulkan` — see the comment in `run_reference/2`.

  Report this alongside any validation result: a verdict is only as good as
  the thing it was compared against.
  """
  @spec reference() :: :exla | :none
  def reference do
    if Code.ensure_loaded?(EXLA) and function_exported?(EXLA, :__info__, 1),
      do: :exla,
      else: :none
  end

  defp run_reference(ir, opts) do
    # Pin the reference compiler EXPLICITLY.
    #
    # This used to `Application.delete_env(:exmc, :compiler)` and rely on
    # `Exmc.JIT.detect_compiler/0` falling through to EXLA. But its private
    # `auto_detect/0` is `EXLA -> Nx.Vulkan -> nil`, so on any host
    # WITHOUT EXLA it resolved to Nx.Vulkan — the arm under test. Reference and
    # candidate were then the same backend, sampled with the same seed, and the
    # mean/variance/KS checks compared a run against itself.
    #
    # FreeBSD has no EXLA at all, so every validation run on the Kepler fleet
    # was that self-comparison: mac-247's "16 tests, 0 failures" was vacuous,
    # and super-io scored 8/16 only because it was the one host actually
    # performing the test. The fleet's standing verdict — "super-io is not
    # valid for numerical validation, the macs are the reference" — was
    # exactly backwards, and it stood for three weeks.
    #
    # `:none` is slower than EXLA but it is a genuine independent reference:
    # pure-Elixir BinaryBackend shares no shader, no NIF and no driver with the
    # candidate. A slow honest reference beats a fast vacuous one.
    ref = reference()

    prev_compiler = Application.get_env(:exmc, :compiler)
    prev_meta = Application.get_env(:exmc, :fused_leapfrog_meta)
    prev_norm_meta = Application.get_env(:exmc, :fused_leapfrog_normal_meta)
    prev_force_prec = Application.get_env(:exmc, :force_precision)

    Application.put_env(:exmc, :compiler, ref)
    Application.delete_env(:exmc, :fused_leapfrog_meta)
    Application.delete_env(:exmc, :fused_leapfrog_normal_meta)

    # Belt and braces. If a future change to detect_compiler/0 ever routes the
    # reference back onto the candidate, fail loudly rather than return a
    # comparison that cannot fail.
    if Code.ensure_loaded?(Nx.Vulkan) and Exmc.JIT.detect_compiler() == Nx.Vulkan do
      restore(:compiler, prev_compiler)

      raise """
      Validator reference arm resolved to Nx.Vulkan — the backend under test.
      Comparing a backend against itself passes unconditionally and validates
      nothing. Reference requested: #{inspect(ref)}.
      """
    end

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

    # ESS, not length: these are NUTS draws, not iid samples. See ess/1.
    se_a = :math.sqrt(var_a / ess(a))
    se_b = :math.sqrt(var_b / ess(b))
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

    n_a = ess(a)
    n_b = ess(b)

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
    se_a = iqr(a) / 1.349 / :math.sqrt(ess(a))
    se_b = iqr(b) / 1.349 / :math.sqrt(ess(b))
    se_combined = :math.sqrt(se_a * se_a + se_b * se_b)

    diff = abs(med_a - med_b)
    tol = @sigma_tol * max(se_combined, 1.0e-12)

    if diff <= tol do
      :ok
    else
      {:error, %{check: :median, exla: med_a, vulkan: med_b, diff: diff, tol: tol}}
    end
  end

  # The differential IQR gate, and the second half of a fix that was only half
  # done.
  #
  # This used `0.25 * IQR / sqrt(n)` as "a conservative tolerance band". It is
  # not conservative, it is about SIX TIMES TOO TIGHT, and the identical proxy
  # was removed from check_analytic/3 in the D92 merge while this copy — the
  # differential one — was left standing. Measured on the FreeBSD fleet
  # 2026-08-23, Cauchy(0,1) at ESS 131:
  #
  #     proxy SE per arm    0.0450        tol 0.1349
  #     correct SE per arm  0.2747        tol 1.1657      6.11x
  #     arms measured IQR 1.928 and 2.185, true value 2.0
  #     observed diff 0.2572 -> failed against the proxy, passes against the truth
  #
  # Both arms bracketed the true IQR and the gate called it a backend
  # disagreement. That is the failure direction that matters: a check which
  # rejects a correct sampler gets switched off, and then it is not a check.
  #
  # SE(IQR) = 0.5 / (f sqrt(n)) with f the density at the quartiles, which the
  # meta supplies via analytic_moments/1 for the families that know it. Without
  # one there is no distribution-free answer, so the old proxy remains as the
  # fallback — wrong, but no more wrong than it was, and now only where nothing
  # better exists.
  defp check_iqr(a, b, meta) do
    iqr_a = iqr(a)
    iqr_b = iqr(b)
    n = min(ess(a), ess(b))

    se_per_arm =
      case analytic_moments(meta) do
        {:quantiles, %{f_quartile: f}} when is_number(f) and f > 0.0 ->
          0.5 / (f * :math.sqrt(n))

        _ ->
          0.25 * (iqr_a + iqr_b) / 2.0 / :math.sqrt(n)
      end

    # Two independent arms, so the combined SE is sqrt(2) times one of them.
    se = :math.sqrt(2.0) * se_per_arm
    diff = abs(iqr_a - iqr_b)
    tol = @sigma_tol * max(se, 1.0e-12)

    if diff <= tol do
      :ok
    else
      {:error, %{check: :iqr, exla: iqr_a, vulkan: iqr_b, diff: diff, tol: tol, ess: n}}
    end
  end

  # --- Stat helpers -----------------------------------------------

  @doc """
  Effective sample size of a NUTS chain, by Geyer's initial monotone positive
  sequence — the estimator Stan uses.

  Every standard error in this module divides by this rather than by
  `length/1`. They used to divide by `length/1`, which assumes iid draws;
  NUTS output is a Markov chain, and these chains carry lag-1 autocorrelation
  around 0.33–0.38. That understates the true standard error by roughly
  1.5–2×, so a nominal 3σ gate was really operating near 1.7σ and rejecting
  perfectly good runs — one Exponential(2) verdict flipped between seeds while
  its pooled variance was *closer* to truth than the run that passed.

  A validator that cries wolf gets switched off, which is the same way the
  self-comparing reference arm survived: a harness nobody trusts is a harness
  nobody reads.

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

  This is the check the differential tests cannot perform. `compare/3` asks
  whether two arms agree; it is structurally blind to any defect they share,
  because a bug in the host NUTS tree moves both arms identically and the
  comparison passes.

  That is not hypothetical. A missing `if (!valid_subtree) break;` in
  `Tree.do_build/11` merged post-U-turn states into the trajectory, inflating
  Normal(0,1)'s posterior variance to ~1.45 against a true 1.0 — in BOTH arms,
  for weeks, under a green mean/variance/KS comparison. Only measuring against
  the distribution's own moments could see it.

  Cauchy has no moments, so it reports median and IQR instead, which are
  defined: median `loc`, IQR `2 * scale`.
  """
  @spec analytic_moments(tuple()) ::
          {:moments, %{mean: float(), var: float()}}
          | {:quantiles, %{median: float(), iqr: float(), f_quartile: float()}}
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

  # Gamma(alpha, beta) in the shape/RATE parameterisation, which is what
  # Exmc.Dist.Gamma uses: mean alpha/beta, var alpha/beta^2.
  def analytic_moments({:gamma, alpha, beta}) when alpha > 0 and beta > 0,
    do: {:moments, %{mean: alpha / beta, var: alpha / (beta * beta)}}

  def analytic_moments({:beta, a, b}) when a > 0 and b > 0 do
    s = a + b
    {:moments, %{mean: a / s, var: a * b / (s * s * (s + 1.0))}}
  end

  # Student-t needs nu > 4 here, not nu > 2.
  #
  # The mean and variance exist for nu > 2, but every gate built on them is a
  # multiple of the standard error of the SAMPLE variance, and that standard
  # error is sqrt((mu4 - sigma^4)/n) — see variance_se/2. For 2 < nu <= 4 the
  # fourth moment is INFINITE, so the sample variance has infinite variance and
  # no sigma-multiple gate on it means anything at all: it will compute a
  # number, and the number is noise. Returning :unknown makes callers say so
  # out loud instead. This clause must precede the nu > 4 one.
  def analytic_moments({:studentt, _mu, _sigma, nu, _c}) when nu > 2 and nu <= 4, do: :unknown

  def analytic_moments({:studentt, mu, sigma, nu, _c}) when nu > 4,
    do: {:moments, %{mean: mu * 1.0, var: sigma * sigma * nu / (nu - 2.0)}}

  # `f_quartile` is the density AT the quartiles, and it is what makes a
  # quantile gate a gate rather than a number — see the se_iqr comment in
  # check_analytic/3. Cauchy's quartiles are loc +/- scale, where
  # f = 1/(pi * scale * (1 + 1^2)) = 1/(2 pi scale).
  def analytic_moments({:cauchy, loc, scale, _log_pi_scale}), do: cauchy_quantiles(loc, scale)

  def analytic_moments({:cauchy, loc, scale}), do: cauchy_quantiles(loc, scale)

  defp cauchy_quantiles(loc, scale) do
    {:quantiles,
     %{
       median: loc * 1.0,
       iqr: 2.0 * scale,
       f_quartile: 1.0 / (2.0 * :math.pi() * scale)
     }}
  end

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
        # passed, which is how a vacuous verdict survives — see run_reference/2
        # for what that already cost once here.
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

          # The variance gate only counts when it can resolve something.
          #
          # `@analytic_tol * se_v` is what this chain can distinguish, and on a
          # heavy-tailed target it can be most of the truth. LogNormal(0,1) on
          # the FreeBSD fleet, 2026-08-23: truth 4.6708, measured 2.4796, ESS
          # 211, tolerance 2.0209 — the gate could resolve a 43% error and the
          # observed shortfall was 47%, a ratio of 1.08. That is a coin flip
          # reported as a verdict.
          #
          # And 2.4796 is not evidence of a broken sampler. The sample variance
          # of a lognormal is downward-biased until n is enormous: with
          # mu4 = e^4(e^6 - 4e^3 + 6e - 3), resolving 47% at 4 sigma needs ESS
          # ~61,000 (~174,000 draws at 0.35 effective per draw), and 20% needs
          # ESS ~336,000. At ESS 211 this gate cannot tell a correct sampler
          # from a badly broken one, so a failure from it says nothing.
          #
          # Same defect as the one D91 documented in the test suite, in the
          # module that suite borrows its statistics from: check_analytic/3
          # sizes its band from the chain's own ESS, so it can never be too
          # tight, and it degrades to nothing on a short chain while still
          # returning a verdict. assert_posterior!/3 grew a power gate for
          # exactly this; this is the same gate, here.
          #
          # Declining to run beats returning a number: the check is skipped
          # out loud, with what it would have taken to make it mean something.
          @analytic_tol * se_v > @max_resolvable * abs(tv) ->
            require Logger

            Logger.warning(
              "[Validator] variance check on #{inspect(meta)} arm #{inspect(label)} is " <>
                "UNDERPOWERED and was SKIPPED: at ESS #{Float.round(n, 1)} it resolves only " <>
                "#{Float.round(@analytic_tol * se_v / abs(tv) * 100, 1)}% of the true " <>
                "variance #{Float.round(tv, 4)} (measured #{Float.round(v, 4)}). " <>
                "Needs roughly #{round(n * :math.pow(@analytic_tol * se_v / (@max_resolvable * abs(tv)), 2))} " <>
                "effective draws to resolve #{round(@max_resolvable * 100)}%."
            )

            :ok

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

      {:quantiles, %{median: tmed, iqr: tiqr} = q} ->
        med = median(samples)
        got_iqr = iqr(samples)
        n = ess(samples)

        se_med = tiqr / 1.349 / :math.sqrt(n)

        # Checking the scale at all matters: without it the Cauchy branch
        # verified location and said nothing whatever about spread, so a
        # sampler that got the centre right and the spread wrong passed.
        #
        # But the IQR's standard error is NOT a fixed fraction of the IQR, and
        # the `0.25 * iqr / sqrt(n)` proxy that used to stand here was not
        # conservative — it was about **six times too tight** for Cauchy, so it
        # would have failed a correct sampler rather than passed a wrong one.
        #
        # For quantiles, Var(q_p) = p(1-p)/(n f(q_p)^2) and
        # Cov(q_25, q_75) = 0.25 * 0.25/(n f^2), so for a symmetric density
        #
        #     SE(IQR) = sqrt(0.25 / (n f^2)) = 0.5 / (f sqrt(n))
        #
        # with f the density at the quartiles. For Cauchy that is
        # pi * scale / sqrt(n), against the old proxy's 0.5 * scale / sqrt(n).
        # Checked against a t(4) chain, where the same proxy was 5x too tight
        # and the correct form put the measured IQR at 1.9 sigma.
        #
        # The fallback keeps the old proxy for a `meta` whose quantile map
        # carries no density — wrong, but no more wrong than it was.
        se_iqr =
          case q do
            %{f_quartile: f} when is_number(f) and f > 0.0 -> 0.5 / (f * :math.sqrt(n))
            _ -> 0.25 * tiqr / :math.sqrt(n)
          end

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
  # This was `var * sqrt(2/(n-1))`, which is the Gaussian special case: it comes
  # from Var(s²) = 2σ⁴/n, which holds only when the fourth central moment is
  # 3σ⁴. The general result is Var(s²) = (μ₄ − σ⁴)/n, so for any heavier-tailed
  # distribution the Gaussian form UNDERSTATES the true standard error — by
  # exactly 2.00x for Exponential and StudentT(5), 1.99x for LogNormal(0, 0.5),
  # since all three have μ₄ = 9σ⁴ or near it.
  #
  # A gate two times too tight rejects correct runs. Measured at n=800 and 4σ
  # over 20,000 replications of *iid* draws: Exponential(1) false-rejected
  # 4.30% of the time against a nominal 0.0063%. Widening @analytic_tol from
  # 3 to 4 masked that rather than fixing it, and it is the likely mechanism
  # behind an Exponential(2) verdict that flipped between seeds while its
  # pooled variance was closer to truth than the run that passed.
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
