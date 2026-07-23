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
