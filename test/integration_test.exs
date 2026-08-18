defmodule Exmc.IntegrationTest do
  use ExUnit.Case

  import Exmc.TestHelper, only: [assert_posterior!: 3]

  @moduletag :integration
  @moduletag timeout: 120_000

  alias Exmc.{Builder, Diagnostics, ModelComparison, Predictive}
  alias Exmc.NUTS.Sampler
  alias Exmc.Dist.{Normal, Exponential, Beta, StudentT}

  # ── 1. Conjugate Normal-Normal posterior recovery ───────────

  test "conjugate Normal-Normal: posterior mean and variance match analytic" do
    # mu ~ N(0, 10), x|mu ~ N(mu, 1), observe x=5.0
    # Analytic posterior: mean = (0/100 + 5/1)/(1/100 + 1) ≈ 4.950
    #                     var  = 1/(1/100 + 1)              ≈ 0.990
    ir =
      Builder.new_ir()
      |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(10.0)})
      |> Builder.rv("x", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
      |> Builder.obs("x_obs", "x", Nx.tensor(5.0))

    # 4000 draws, not 500. At 500 this chain reaches ESS ~180, where a 4-sigma
    # analytic gate can only detect a 38% variance error — which is the size of
    # the defect that shipped. 4000 draws reach ESS ~1280 and resolve ~15%,
    # verified by injecting the 0.3.0 defect (variance x1.378) into this exact
    # chain and confirming the gate reports `analytic_variance failed`.
    {trace, stats} = Sampler.sample(ir, %{}, num_warmup: 300, num_samples: 4000, seed: 42)

    samples = trace |> Map.fetch!("mu") |> Nx.to_flat_list()

    # The conjugate posterior in closed form, rather than the rounded 4.95 and
    # sqrt(0.99) this test used to compare against:
    #   precision = 1/100 + 1/1        = 1.01
    #   var       = 1/1.01             = 0.990099...
    #   mean      = (0/100 + 5/1)/1.01 = 4.950495...
    post_var = 1.0 / (1.0 / 100.0 + 1.0)
    post_mean = 5.0 / (1.0 / 100.0 + 1.0)

    # This used to be `assert_in_delta mu_stats.std, sqrt(0.99), 0.5`, which
    # accepts any std from 0.495 to 1.495 — a factor of 2.3 either way on the
    # standard deviation, so a factor of 5 on the variance. It passed
    # comfortably for months while the sampler returned a variance inflated by
    # 38% (see CHANGELOG 0.3.1). A tolerance that wide is not a correctness
    # test, it is a smoke test wearing one's clothes.
    #
    # 0.3.1 replaced it with a bare `check_analytic/3`, which sizes its
    # tolerance from the chain's own ESS and therefore cannot be too tight.
    # It can be far too loose, and it was: at 500 draws the gate admitted a
    # 37.8% variance error against a defect that was 37.8%. It would have been
    # a coin flip on the very bug it was written for.
    #
    # `assert_posterior!` adds the half that was missing — the chain must have
    # enough effective draws for the gate to see a 20% variance error, or the
    # test fails as INCONCLUSIVE rather than passing.
    assert_posterior!(samples, {:normal, post_mean, :math.sqrt(post_var)}, resolution: 0.20)

    assert stats.divergences < 20
  end

  # ── 2. Multi-chain convergence ──────────────────────────────

  test "multi-chain: R-hat near 1.0 and ESS reasonable" do
    ir =
      Builder.new_ir()
      |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(5.0)})
      |> Builder.rv("x", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
      |> Builder.obs("x_obs", "x", Nx.tensor(3.0))

    {traces, _stats_list} =
      Sampler.sample_chains(ir, 2, num_warmup: 200, num_samples: 300, seed: 7)

    chains = Enum.map(traces, fn t -> Nx.to_flat_list(t["mu"]) end)

    r = Diagnostics.rhat(chains)
    assert_in_delta r, 1.0, 0.2

    # ESS on combined samples
    combined = List.flatten(chains)
    ess = Diagnostics.ess(combined)
    assert ess > 50
  end

  # ── 3. Constrained distributions: samples respect support ──

  test "Gamma prior: all trace values positive" do
    # alpha ~ Gamma(2, 1) with log transform
    ir =
      Builder.new_ir()
      |> Builder.rv("alpha", Exmc.Dist.Gamma, %{alpha: Nx.tensor(2.0), beta: Nx.tensor(1.0)})

    # 4000 draws, not 200. `assert_in_delta mean, 2.0, 1.0` on 200 draws
    # asserted the mean to within 0.71 posterior sd and said NOTHING about the
    # spread, so it passed for a frozen chain, a doubled variance, or anything
    # in between. At 200 draws the best possible variance gate could only see
    # an 86% error (bench/tolerance_audit.exs).
    # 8500, not 200. Gamma(2,1)'s fourth moment is 5x its squared variance, so
    # a 20% gate needs ESS ~2000 — and the count is sized for ~15% rather than
    # 20% on purpose. ESS varies with the backend: the first version of this
    # sweep sized every count to just under 20% on EXLA and the Lognormal test
    # then failed under `EXMC_COMPILER=vulkan` at 21.3%. Headroom is not
    # padding here, it is what stops the gate being a coin flip.
    {trace, _stats} = Sampler.sample(ir, %{}, num_warmup: 500, num_samples: 8500, seed: 99)

    values = Nx.to_flat_list(trace["alpha"])
    assert Enum.all?(values, &(&1 > 0.0))

    # Gamma(2,1) in shape/rate: mean 2.0, variance 2.0.
    assert_posterior!(values, {:gamma, 2.0, 1.0}, resolution: 0.20)
  end

  # Beta/Gamma lgamma gradient triggers Complex.divide on BinaryBackend.
  # Tracked in INTEGRATION_TESTING.md as a known limitation.
  # Use Exponential (no lgamma) as constrained-support test instead.

  test "Exponential prior: all trace values positive, mean near 1/lambda" do
    # rate ~ Exp(2), prior mean = 0.5
    ir =
      Builder.new_ir()
      |> Builder.rv("rate", Exponential, %{lambda: Nx.tensor(2.0)})

    # Was 300 draws with `assert_in_delta mean, 0.5, 0.3` — 0.60 posterior sd on
    # the mean and no gate on the spread at all.
    # 12000 draws. Exponential is the most skewed target in this file — its
    # fourth moment is 9/lambda^4 against a squared variance of 1/lambda^4, so
    # (mu4 - sigma^4)/sigma^4 = 8 and a 20% variance gate needs ESS ~3200,
    # four times what the Normal target needs. Skew is expensive to verify; the
    # alternative on offer was not verifying it.
    {trace, _stats} = Sampler.sample(ir, %{}, num_warmup: 500, num_samples: 12000, seed: 77)

    values = Nx.to_flat_list(trace["rate"])
    assert Enum.all?(values, &(&1 > 0.0))

    # Exponential(2): mean 0.5, variance 0.25.
    assert_posterior!(values, {:exponential, 2.0}, resolution: 0.20)
  end

  # ── 4. Hierarchical model end-to-end with diagnostics ──────

  test "hierarchical: parent_mu ~ N(0,5), child ~ N(parent_mu, 2), obs -> posterior shift" do
    # Two-level Normal-Normal hierarchy (no constrained params, avoids lgamma/overflow)
    # parent_mu ~ N(0, 5), child ~ N(parent_mu, 2), observe child = 4.0
    ir =
      Builder.new_ir()
      |> Builder.rv("parent_mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(5.0)})
      |> Builder.rv("child", Normal, %{mu: "parent_mu", sigma: Nx.tensor(2.0)})
      |> Builder.obs("child_obs", "child", Nx.tensor(4.0))

    {trace, stats} = Sampler.sample(ir, %{}, num_warmup: 300, num_samples: 400, seed: 55)

    summary = Diagnostics.summary(trace)

    # parent_mu posterior should shift toward 4.0
    # Analytic: mean = (0/25 + 4/4) / (1/25 + 1/4) = 1.0 / 0.29 ≈ 3.45
    pm_stats = summary["parent_mu"]
    assert_in_delta pm_stats.mean, 3.45, 1.0

    # Quantiles should be ordered
    assert pm_stats.q5 < pm_stats.q25
    assert pm_stats.q25 < pm_stats.q50
    assert pm_stats.q50 < pm_stats.q75
    assert pm_stats.q75 < pm_stats.q95

    # ESS should be reasonable
    ess = Diagnostics.ess(trace["parent_mu"])
    assert ess > 30

    assert stats.divergences < 50
  end

  # ── 5. DSL → sample → diagnostics round-trip ───────────────

  test "DSL model through full sample + diagnostics pipeline" do
    use Exmc.DSL

    ir =
      Exmc.DSL.model do
        rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(5.0)})
        rv("x", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
        obs("x_obs", "x", Nx.tensor(2.0))
      end

    {trace, _stats} = Sampler.sample(ir, %{}, num_warmup: 200, num_samples: 300, seed: 11)

    summary = Diagnostics.summary(trace)
    assert Map.has_key?(summary, "mu")

    mu_stats = summary["mu"]
    assert is_float(mu_stats.mean)
    assert is_float(mu_stats.std)
    assert mu_stats.std > 0.0

    # ESS on the samples
    ess = Diagnostics.ess(trace["mu"])
    assert ess > 10

    # ACF lag 0 should be 1.0
    acf = Diagnostics.autocorrelation(trace["mu"], 5)
    assert_in_delta hd(acf), 1.0, 1.0e-10
  end

  # ── 6. Sample stats internal consistency ────────────────────

  test "sample_stats: lengths, bounds, divergence count" do
    ir =
      Builder.new_ir()
      |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

    num_samples = 100
    max_depth = 10

    {_trace, stats} =
      Sampler.sample(ir, %{},
        num_warmup: 100,
        num_samples: num_samples,
        max_tree_depth: max_depth,
        seed: 33
      )

    ss = stats.sample_stats
    assert length(ss) == num_samples

    # Tree depths within bounds
    assert Enum.all?(ss, fn s -> s.tree_depth >= 0 and s.tree_depth <= max_depth end)

    # n_steps positive
    assert Enum.all?(ss, fn s -> s.n_steps >= 1 end)

    # accept_prob in [0, 1]
    assert Enum.all?(ss, fn s ->
             p = s.accept_prob
             p >= 0.0 and p <= 1.0
           end)

    # divergent is boolean
    assert Enum.all?(ss, fn s -> is_boolean(s.divergent) end)

    # Sampling-phase divergences are a subset of total (which includes warmup)
    div_from_stats = Enum.count(ss, & &1.divergent)
    assert stats.divergences >= div_from_stats
  end

  # ── 7. Multiple observations strengthen posterior ───────────

  test "more observations narrow the posterior" do
    # Single observation
    ir1 =
      Builder.new_ir()
      |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(10.0)})
      |> Builder.rv("x1", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
      |> Builder.obs("x1_obs", "x1", Nx.tensor(4.0))

    {trace1, _} = Sampler.sample(ir1, %{}, num_warmup: 200, num_samples: 300, seed: 42)
    std1 = Diagnostics.summary(trace1)["mu"].std

    # Three observations all near 4.0
    ir3 =
      Builder.new_ir()
      |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(10.0)})
      |> Builder.rv("x1", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
      |> Builder.obs("x1_obs", "x1", Nx.tensor(4.0))
      |> Builder.rv("x2", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
      |> Builder.obs("x2_obs", "x2", Nx.tensor(3.8))
      |> Builder.rv("x3", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
      |> Builder.obs("x3_obs", "x3", Nx.tensor(4.2))

    {trace3, _} = Sampler.sample(ir3, %{}, num_warmup: 200, num_samples: 300, seed: 42)
    std3 = Diagnostics.summary(trace3)["mu"].std

    # More data should narrow the posterior
    assert std3 < std1
  end

  # ── 8. Beta prior: samples in (0,1), mean near a/(a+b) ───────
  # Previously blocked on BinaryBackend (lgamma gradient triggers Complex.divide).
  # Now works with EXLA backend.

  test "Beta prior: samples in (0,1), mean near a/(a+b)" do
    # p ~ Beta(2, 5), prior mean = 2/7 ≈ 0.286
    ir =
      Builder.new_ir()
      |> Builder.rv("p", Beta, %{alpha: Nx.tensor(2.0), beta: Nx.tensor(5.0)})

    # init near mode (0.2) in unconstrained space: logit(0.2) ≈ -1.39
    # Was 400 draws with `assert_in_delta mean, 2/7, 0.15` — 0.94 posterior sd
    # on the mean, i.e. a mean nearly a full sd wrong still passed, and no gate
    # on the spread.
    {trace, _stats} =
      Sampler.sample(ir, %{},
        num_warmup: 500,
        num_samples: 5000,
        seed: 88,
        init_values: %{"p" => Nx.tensor(0.2)}
      )

    values = Nx.to_flat_list(trace["p"])

    # All samples must be in (0, 1)
    assert Enum.all?(values, &(&1 > 0.0 and &1 < 1.0))

    # Beta(2,5): mean 2/7, variance ab/((a+b)^2 (a+b+1)) = 10/(49*8).
    assert_posterior!(values, {:beta, 2.0, 5.0}, resolution: 0.20)
  end

  # ── 9. StudentT prior: mean near loc ─────────────────────────
  # Previously blocked on BinaryBackend (lgamma gradient).

  test "StudentT prior: mean near loc" do
    # x ~ StudentT(df=4, loc=3.0, scale=1.0), prior mean = loc = 3.0
    ir =
      Builder.new_ir()
      |> Builder.rv("x", StudentT, %{
        df: Nx.tensor(4.0),
        loc: Nx.tensor(3.0),
        scale: Nx.tensor(1.0)
      })

    {trace, _stats} = Sampler.sample(ir, %{}, num_warmup: 500, num_samples: 4000, seed: 66)

    values = Nx.to_flat_list(trace["x"])
    n = length(values)
    ess = Exmc.NUTS.Vulkan.Validator.ess(values)

    # This one deliberately does NOT use assert_posterior!, and the reason is
    # the point of the test.
    #
    # Student-t with df = 4 has a finite mean and variance but an INFINITE
    # fourth moment. Every variance gate in this suite is a multiple of
    # sqrt((mu4 - sigma^4)/n_eff), so at df <= 4 that standard error does not
    # exist: the sample variance has infinite variance and any sigma-multiple
    # band around it is noise dressed as a tolerance. `analytic_moments/1`
    # therefore returns `:unknown` for 2 < df <= 4, and `assert_posterior!`
    # flunks rather than computing a number that means nothing.
    #
    # What CAN be checked at df = 4:
    #
    #   the mean   — Var(x_bar) = sigma^2 / n_eff needs only the second moment,
    #                and sigma^2 = scale^2 * df/(df-2) = 2.0 exactly
    #   the IQR    — a quantile, so it needs no moments at all
    #
    # The old assertion was `assert_in_delta mean, 3.0, 1.5` on 400 draws: 1.06
    # posterior sd of slack on the mean, and nothing whatsoever on the spread.
    # A frozen chain sitting at 3.0 passed it.
    se_mean = :math.sqrt(2.0 / ess)
    mean = Enum.sum(values) / n
    assert_in_delta mean, 3.0, 4.0 * se_mean

    # t(4)'s 75th percentile is 0.7406971, so IQR = 2 * scale * 0.7406971.
    #
    # The standard error of a sample IQR is NOT a fixed fraction of the IQR.
    # For quantiles q_p, Var(q_p) = p(1-p)/(n_eff f(q_p)^2) and
    # Cov(q_25, q_75) = 0.25*0.25/(n_eff f^2), so with a symmetric density
    #
    #     SE(IQR) = sqrt(0.25 / (n_eff f^2)) = 0.5 / (f sqrt(n_eff))
    #
    # where f is the density at the quartiles: 0.27190 for t(4) with scale 1.
    # The `0.25 * IQR / sqrt(n)` proxy that check_analytic/3 uses is about 5x
    # TOO TIGHT here and 6x too tight for Cauchy, i.e. it fails correct
    # samplers — see the note on the quantile branch in Validator.
    sorted = Enum.sort(values)
    q25 = Exmc.Diagnostics.quantile(sorted, n, 0.25)
    q75 = Exmc.Diagnostics.quantile(sorted, n, 0.75)
    analytic_iqr = 2.0 * 1.0 * 0.7406971
    f_quartile = 0.2719

    assert_in_delta q75 - q25, analytic_iqr, 4.0 * 0.5 / (f_quartile * :math.sqrt(ess))
  end

  # ── 10. Hierarchical with constrained parent ─────────────────
  # sigma ~ Exp(1), child ~ N(0, sigma), observe child = 2.0
  # Previously blocked on BinaryBackend (exp overflow in unconstrained space).

  test "hierarchical with constrained parent: sigma ~ Exp, child ~ N(0, sigma), obs" do
    ir =
      Builder.new_ir()
      |> Builder.rv("sigma", Exponential, %{lambda: Nx.tensor(1.0)})
      |> Builder.rv("child", Normal, %{mu: Nx.tensor(0.0), sigma: "sigma"})
      |> Builder.obs("child_obs", "child", Nx.tensor(2.0))

    {trace, _stats} =
      Sampler.sample(ir, %{},
        num_warmup: 500,
        num_samples: 500,
        seed: 44,
        init_values: %{"sigma" => Nx.tensor(2.0)}
      )

    sigma_values = Nx.to_flat_list(trace["sigma"])

    # All sigma samples must be positive (Exponential support)
    assert Enum.all?(sigma_values, &(&1 > 0.0))

    # sigma posterior should be centered somewhere reasonable (not extreme)
    sigma_mean = Enum.sum(sigma_values) / length(sigma_values)
    assert sigma_mean > 0.5
    assert sigma_mean < 10.0
  end

  # ── 11. Prior predictive sampling ─────────────────────────────

  test "prior_samples: shapes, support, and hierarchical resolution" do
    # mu ~ N(0, 5), x ~ N(mu, 1)
    ir =
      Builder.new_ir()
      |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(5.0)})
      |> Builder.rv("x", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})

    samples = Predictive.prior_samples(ir, 200, seed: 42)

    # Both variables present
    assert Map.has_key?(samples, "mu")
    assert Map.has_key?(samples, "x")

    # Shape is {n}
    assert Nx.shape(samples["mu"]) == {200}
    assert Nx.shape(samples["x"]) == {200}

    # mu ~ N(0,5): mean near 0, std near 5
    mu_mean = Nx.mean(samples["mu"]) |> Nx.to_number()
    assert_in_delta mu_mean, 0.0, 1.5

    # x ~ N(mu, 1): variance should be > 1 (prior variance of mu propagates)
    x_var = Nx.variance(samples["x"]) |> Nx.to_number()
    assert x_var > 1.0
  end

  # ── 12. Prior predictive with constrained distribution ────────

  test "prior_samples: constrained distributions respect support" do
    ir =
      Builder.new_ir()
      |> Builder.rv("rate", Exponential, %{lambda: Nx.tensor(2.0)})

    samples = Predictive.prior_samples(ir, 500, seed: 7)

    values = Nx.to_flat_list(samples["rate"])
    assert Enum.all?(values, &(&1 > 0.0))

    mean = Enum.sum(values) / length(values)
    assert_in_delta mean, 0.5, 0.15
  end

  # ── 13. Posterior predictive sampling ──────────────────────────

  test "posterior_predictive: draws from likelihood with posterior params" do
    ir =
      Builder.new_ir()
      |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(10.0)})
      |> Builder.rv("x", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
      |> Builder.obs("x_obs", "x", Nx.tensor(5.0))

    {trace, _stats} = Sampler.sample(ir, %{}, num_warmup: 200, num_samples: 300, seed: 42)

    ppc = Predictive.posterior_predictive(ir, trace, seed: 99)

    assert Map.has_key?(ppc, "x_obs")
    assert Nx.shape(ppc["x_obs"]) == {300}

    # PPC mean should be near the observation (5.0) since posterior concentrates near 5
    ppc_mean = Nx.mean(ppc["x_obs"]) |> Nx.to_number()
    assert_in_delta ppc_mean, 5.0, 1.5
  end

  # ── 14. Large model stress test (5+ free RVs) ─────────────────

  test "large model: 5-parameter hierarchical with multiple observations" do
    # 5 free params, all connected to data:
    # mu_global ~ N(0, 10)          -- global location
    # sigma_global ~ Exp(1)         -- global scale
    # alpha ~ N(mu_global, sigma_global) -- group mean A
    # beta ~ N(mu_global, sigma_global)  -- group mean B
    # sigma_obs ~ Exp(2)            -- observation noise
    # y1 ~ N(alpha, sigma_obs), obs=4
    # y2 ~ N(alpha, sigma_obs), obs=5
    # y3 ~ N(beta, sigma_obs), obs=8

    ir =
      Builder.new_ir()
      |> Builder.rv("mu_global", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(10.0)})
      |> Builder.rv("sigma_global", Exponential, %{lambda: Nx.tensor(1.0)})
      |> Builder.rv("alpha", Normal, %{mu: "mu_global", sigma: "sigma_global"})
      |> Builder.rv("beta", Normal, %{mu: "mu_global", sigma: "sigma_global"})
      |> Builder.rv("sigma_obs", Exponential, %{lambda: Nx.tensor(2.0)})
      |> Builder.rv("y1", Normal, %{mu: "alpha", sigma: "sigma_obs"})
      |> Builder.obs("y1_obs", "y1", Nx.tensor(4.0))
      |> Builder.rv("y2", Normal, %{mu: "alpha", sigma: "sigma_obs"})
      |> Builder.obs("y2_obs", "y2", Nx.tensor(5.0))
      |> Builder.rv("y3", Normal, %{mu: "beta", sigma: "sigma_obs"})
      |> Builder.obs("y3_obs", "y3", Nx.tensor(8.0))

    init = %{
      "mu_global" => Nx.tensor(5.0),
      "sigma_global" => Nx.tensor(2.0),
      "alpha" => Nx.tensor(4.5),
      "beta" => Nx.tensor(8.0),
      "sigma_obs" => Nx.tensor(1.0)
    }

    {trace, stats} =
      Sampler.sample(ir, init,
        num_warmup: 500,
        num_samples: 500,
        seed: 42
      )

    # Should have 5 free parameters
    assert map_size(trace) == 5

    # All constrained params positive
    assert Enum.all?(Nx.to_flat_list(trace["sigma_global"]), &(&1 > 0.0))
    assert Enum.all?(Nx.to_flat_list(trace["sigma_obs"]), &(&1 > 0.0))

    # alpha posterior should be near ~4.5 (average of obs 4 and 5)
    alpha_mean = Nx.mean(trace["alpha"]) |> Nx.to_number()
    assert alpha_mean > 1.0, "alpha_mean=#{alpha_mean}, expected > 1.0"
    assert alpha_mean < 9.0, "alpha_mean=#{alpha_mean}, expected < 9.0"

    # beta posterior should be near ~8 (obs=8)
    beta_mean = Nx.mean(trace["beta"]) |> Nx.to_number()
    assert beta_mean > 2.0, "beta_mean=#{beta_mean}, expected > 2.0"
    assert beta_mean < 14.0, "beta_mean=#{beta_mean}, expected < 14.0"

    # Diagnostics should work on all variables
    summary = Diagnostics.summary(trace)
    assert map_size(summary) == 5

    for {_var, s} <- summary do
      assert is_float(s.mean)
      assert is_float(s.std)
      assert s.std > 0.0
    end
  end

  # ── 15. NCP: hierarchical model auto-reparameterized ────────────

  test "NCP: hierarchical Normal-Normal uses non-centered parameterization" do
    # mu ~ N(0, 5), sigma ~ Exp(1), alpha ~ N(mu, sigma), obs alpha = 3.0
    # The NCP pass should transform alpha ~ N(mu, sigma) -> alpha ~ N(0,1)
    # and reconstruct alpha = mu + sigma * z in the trace.
    ir =
      Builder.new_ir()
      |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(5.0)})
      |> Builder.rv("sigma", Exponential, %{lambda: Nx.tensor(1.0)})
      |> Builder.rv("alpha", Normal, %{mu: "mu", sigma: "sigma"})
      |> Builder.rv("y", Normal, %{mu: "alpha", sigma: Nx.tensor(1.0)})
      |> Builder.obs("y_obs", "y", Nx.tensor(3.0))

    # Verify NCP is applied
    rewritten = Exmc.Rewrite.apply(ir)
    assert Map.has_key?(rewritten.ncp_info, "alpha")
    assert rewritten.ncp_info["alpha"] == %{mu: "mu", sigma: "sigma"}

    # Sample and check posterior makes sense
    {trace, stats} =
      Sampler.sample(ir, %{},
        num_warmup: 400,
        num_samples: 400,
        seed: 42,
        init_values: %{
          "mu" => Nx.tensor(3.0),
          "sigma" => Nx.tensor(1.0),
          "alpha" => Nx.tensor(3.0)
        }
      )

    # alpha should be reconstructed (not raw z)
    assert Map.has_key?(trace, "alpha")
    alpha_mean = Nx.mean(trace["alpha"]) |> Nx.to_number()

    # Posterior for alpha should be near observation (3.0)
    assert alpha_mean > 0.0, "alpha_mean=#{alpha_mean}, expected > 0"
    assert alpha_mean < 8.0, "alpha_mean=#{alpha_mean}, expected < 8"

    # sigma should be positive
    assert Enum.all?(Nx.to_flat_list(trace["sigma"]), &(&1 > 0.0))

    assert stats.divergences < 100
  end

  # ── 16. NCP: verify logp equivalence (centered vs non-centered) ─

  test "NCP: logp at same point is equivalent to centered parameterization" do
    # Build a simple hierarchical model
    ir =
      Builder.new_ir()
      |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(5.0)})
      |> Builder.rv("sigma", Exponential, %{lambda: Nx.tensor(1.0)})
      |> Builder.rv("x", Normal, %{mu: "mu", sigma: "sigma"})

    # Compile with NCP (default)
    {logp_fn_ncp, _pm_ncp} = Exmc.Compiler.compile(ir)

    # The NCP-compiled logp should be finite at a reasonable point
    # mu=1.0, sigma stored as log(sigma)=0.0, x stored as z=0.5
    flat = Nx.tensor([1.0, 0.0, 0.5], type: :f64)
    logp = logp_fn_ncp.(flat) |> Nx.to_number()

    assert is_number(logp), "logp should be a number, got #{inspect(logp)}"
    assert logp < 0.0, "logp should be negative for a log-density"
    assert logp > -100.0, "logp should be reasonable, got #{logp}"
  end

  # ── 17. WAIC: basic computation on conjugate model ──────────────

  test "WAIC: pointwise log-likelihood and WAIC on Normal-Normal model" do
    ir =
      Builder.new_ir()
      |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(10.0)})
      |> Builder.rv("x", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
      |> Builder.obs("x_obs", "x", Nx.tensor(5.0))

    {trace, _stats} = Sampler.sample(ir, %{}, num_warmup: 300, num_samples: 300, seed: 42)

    # Compute pointwise log-likelihood
    pw_ll = ModelComparison.pointwise_log_likelihood(ir, trace)

    assert Map.has_key?(pw_ll, "x_obs")
    assert length(pw_ll["x_obs"]) == 300

    # All log-likelihoods should be finite and negative
    assert Enum.all?(pw_ll["x_obs"], &(is_number(&1) and &1 < 0.0))

    # Compute WAIC
    result = ModelComparison.waic(pw_ll)

    assert is_float(result.waic)
    assert is_float(result.elpd_waic)
    assert is_float(result.p_waic)
    assert is_float(result.se)
    assert result.n_obs == 1

    # WAIC should be finite and positive
    assert result.waic > 0.0
    # p_waic should be small (roughly 1 effective parameter)
    assert result.p_waic > 0.0
    assert result.p_waic < 5.0
  end

  # ── 18. WAIC model comparison: better model has lower WAIC ──────

  test "WAIC: better-fitting model has lower WAIC" do
    # Model A: mu ~ N(5, 1) with tight prior near data
    ir_good =
      Builder.new_ir()
      |> Builder.rv("mu", Normal, %{mu: Nx.tensor(5.0), sigma: Nx.tensor(1.0)})
      |> Builder.rv("x", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
      |> Builder.obs("x_obs", "x", Nx.tensor(5.0))

    # Model B: mu ~ N(0, 1) with prior far from data
    ir_bad =
      Builder.new_ir()
      |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
      |> Builder.rv("x", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
      |> Builder.obs("x_obs", "x", Nx.tensor(5.0))

    {trace_good, _} = Sampler.sample(ir_good, %{}, num_warmup: 300, num_samples: 300, seed: 42)
    {trace_bad, _} = Sampler.sample(ir_bad, %{}, num_warmup: 300, num_samples: 300, seed: 42)

    waic_good =
      ModelComparison.waic(ModelComparison.pointwise_log_likelihood(ir_good, trace_good))

    waic_bad = ModelComparison.waic(ModelComparison.pointwise_log_likelihood(ir_bad, trace_bad))

    # Better model should have higher elpd (less negative) and lower WAIC
    assert waic_good.elpd_waic > waic_bad.elpd_waic,
           "good model elpd=#{waic_good.elpd_waic} should > bad model elpd=#{waic_bad.elpd_waic}"

    # Model comparison
    compared = ModelComparison.compare([{"good", waic_good}, {"bad", waic_bad}])
    assert hd(compared).label == "good"
  end

  # ── 19. LOO: basic computation ──────────────────────────────────

  test "LOO: basic LOO-CV computation" do
    ir =
      Builder.new_ir()
      |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(10.0)})
      |> Builder.rv("x1", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
      |> Builder.obs("x1_obs", "x1", Nx.tensor(4.0))
      |> Builder.rv("x2", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
      |> Builder.obs("x2_obs", "x2", Nx.tensor(5.0))

    {trace, _stats} = Sampler.sample(ir, %{}, num_warmup: 300, num_samples: 300, seed: 42)

    pw_ll = ModelComparison.pointwise_log_likelihood(ir, trace)
    assert map_size(pw_ll) == 2

    result = ModelComparison.loo(pw_ll)

    assert is_float(result.loo)
    assert is_float(result.elpd_loo)
    assert is_float(result.p_loo)
    assert is_float(result.se)
    assert result.n_obs == 2
    assert result.loo > 0.0
  end

  # ── 21. Vector obs narrows posterior same as scalar obs ──────────

  # REGRESSION GUARD for the observed-model defect, fixed in 6c1589a. This test
  # was red under `compiler: :vulkan` for weeks and was deliberately left red
  # rather than skipped: `compose_logp_defn/1` gave every observed node the
  # WHOLE observation buffer, so the scalar arm (3 obs nodes) counted the
  # likelihood 3x and returned a completely frozen chain — 1 distinct value in
  # 500 draws — while the vector arm (1 obs node) landed on the analytic
  # answer. Green on both arms since 6c1589a; confirmed across nine model
  # shapes and four seeds by bench/observed_model_evidence.exs. See
  # docs/OPEN_VULKAN_OBSERVED_MODEL.md.
  #
  # Do not weaken the deltas below and do not skip this under Vulkan. Hiding a
  # real failure behind a skip is the exact habit that let two posterior
  # defects ship — see CHANGELOG 0.3.1.
  test "vector obs produces same posterior as equivalent scalar obs" do
    # Scalar version: 3 separate obs nodes
    ir_scalar =
      Builder.new_ir()
      |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(10.0)})
      |> Builder.rv("x1", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
      |> Builder.obs("x1_obs", "x1", Nx.tensor(4.0))
      |> Builder.rv("x2", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
      |> Builder.obs("x2_obs", "x2", Nx.tensor(3.8))
      |> Builder.rv("x3", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
      |> Builder.obs("x3_obs", "x3", Nx.tensor(4.2))

    {trace_scalar, _} =
      Sampler.sample(ir_scalar, %{}, num_warmup: 300, num_samples: 500, seed: 42)

    scalar_summary = Diagnostics.summary(trace_scalar)

    # Vector version: single obs node with vector data
    ir_vector =
      Builder.new_ir()
      |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(10.0)})
      |> Builder.rv("x", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
      |> Builder.obs("x_obs", "x", Nx.tensor([4.0, 3.8, 4.2]))

    {trace_vector, _} =
      Sampler.sample(ir_vector, %{}, num_warmup: 300, num_samples: 500, seed: 42)

    vector_summary = Diagnostics.summary(trace_vector)

    # Both should recover similar posterior for mu
    # Analytic: mean ≈ (0/100 + 12/1) / (1/100 + 3) ≈ 3.99, std ≈ sqrt(1/3.01) ≈ 0.577
    assert_in_delta scalar_summary["mu"].mean, vector_summary["mu"].mean, 0.5
    assert_in_delta scalar_summary["mu"].std, vector_summary["mu"].std, 0.3
  end

  # ── 22. Vector obs with hierarchical model ──────────────────────

  test "vector obs with hierarchical model" do
    # mu ~ N(0, 5), x ~ N(mu, 1), obs x = [1, 2, 3, 4, 5]
    ir =
      Builder.new_ir()
      |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(5.0)})
      |> Builder.rv("x", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
      |> Builder.obs("x_obs", "x", Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))

    {trace, stats} = Sampler.sample(ir, %{}, num_warmup: 300, num_samples: 500, seed: 42)

    summary = Diagnostics.summary(trace)
    mu_stats = summary["mu"]

    # Analytic: mean = (0/25 + 15/1) / (1/25 + 5) ≈ 2.98
    #           std  = sqrt(1/(1/25 + 5)) ≈ 0.445
    assert_in_delta mu_stats.mean, 3.0, 0.5
    assert mu_stats.std < 1.0, "std=#{mu_stats.std}, expected < 1.0 (5 obs should narrow)"

    assert stats.divergences < 50
  end

  # ── 23. WAIC with vector obs ────────────────────────────────────

  test "WAIC with vector obs returns per-element pointwise keys" do
    ir =
      Builder.new_ir()
      |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(10.0)})
      |> Builder.rv("x", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
      |> Builder.obs("x_obs", "x", Nx.tensor([4.0, 3.8, 4.2]))

    {trace, _stats} = Sampler.sample(ir, %{}, num_warmup: 300, num_samples: 300, seed: 42)

    pw_ll = ModelComparison.pointwise_log_likelihood(ir, trace)

    # Should have 3 per-element keys: {"x_obs", 0}, {"x_obs", 1}, {"x_obs", 2}
    assert map_size(pw_ll) == 3
    assert Map.has_key?(pw_ll, {"x_obs", 0})
    assert Map.has_key?(pw_ll, {"x_obs", 1})
    assert Map.has_key?(pw_ll, {"x_obs", 2})

    # Each key should have 300 samples
    for i <- 0..2 do
      ll = pw_ll[{"x_obs", i}]
      assert length(ll) == 300
      assert Enum.all?(ll, &(is_number(&1) and &1 < 0.0))
    end

    # WAIC should work with tuple keys
    result = ModelComparison.waic(pw_ll)

    assert is_float(result.waic)
    assert result.waic > 0.0
    assert result.n_obs == 3

    # LOO should also work
    loo_result = ModelComparison.loo(pw_ll)
    assert is_float(loo_result.loo)
    assert loo_result.n_obs == 3
  end

  # ── 24. Parallel chains: faster than sequential ─────────────────

  test "vectorized chains: faster than old parallel for 4 chains" do
    ir =
      Builder.new_ir()
      |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(5.0)})
      |> Builder.rv("x", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
      |> Builder.obs("x_obs", "x", Nx.tensor(3.0))

    shared_opts = [num_warmup: 200, num_samples: 200, seed: 42]

    # Old parallel path (independent warmup per chain, Task.async_stream)
    t0 = System.monotonic_time(:millisecond)
    {traces_par, _} = Sampler.sample_chains(ir, 4, [vectorized: false] ++ shared_opts)
    t_par = System.monotonic_time(:millisecond) - t0

    # Vectorized path (shared warmup, sequential in one process)
    t0 = System.monotonic_time(:millisecond)
    {traces_vec, _} = Sampler.sample_chains(ir, 4, [vectorized: true] ++ shared_opts)
    t_vec = System.monotonic_time(:millisecond) - t0

    # Both should produce 4 chains
    assert length(traces_par) == 4
    assert length(traces_vec) == 4

    # Vectorized should be faster (shared warmup + no XLA contention)
    assert t_vec < t_par, "vectorized=#{t_vec}ms should be < parallel=#{t_par}ms"
  end

  # ── 25. Parallel chains with init_values ────────────────────────

  test "parallel chains: init_values propagated to all chains" do
    ir =
      Builder.new_ir()
      |> Builder.rv("sigma", Exponential, %{lambda: Nx.tensor(1.0)})
      |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: "sigma"})
      |> Builder.obs("x_obs", "x", Nx.tensor(2.0))

    {traces, stats_list} =
      Sampler.sample_chains(ir, 2,
        num_warmup: 300,
        num_samples: 300,
        seed: 44,
        init_values: %{"sigma" => Nx.tensor(2.0)}
      )

    assert length(traces) == 2

    # Both chains should have positive sigma (init helped avoid overflow)
    for trace <- traces do
      assert Enum.all?(Nx.to_flat_list(trace["sigma"]), &(&1 > 0.0))
    end

    # R-hat should be reasonable
    chains = Enum.map(traces, fn t -> Nx.to_flat_list(t["sigma"]) end)
    r = Diagnostics.rhat(chains)
    assert_in_delta r, 1.0, 0.3

    # Both chains should have some stats
    for stats <- stats_list do
      assert is_float(stats.step_size)
      assert stats.step_size > 0.0
    end
  end

  # ── 26. Vectorized chains: posterior recovery with shared warmup ──

  test "vectorized chains: 4 chains recover posterior with shared warmup" do
    ir =
      Builder.new_ir()
      |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(10.0)})
      |> Builder.rv("x", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
      |> Builder.obs("x_obs", "x", Nx.tensor(5.0))

    {traces, stats_list} =
      Sampler.sample_chains_vectorized(ir, 4,
        num_warmup: 300,
        num_samples: 300,
        seed: 42
      )

    assert length(traces) == 4
    assert length(stats_list) == 4

    # All chains should recover posterior mean near 5.0
    for trace <- traces do
      mu_mean = Nx.mean(trace["mu"]) |> Nx.to_number()
      assert_in_delta mu_mean, 4.95, 1.0
    end

    # Shared warmup: all chains share same step_size and inv_mass_diag
    step_sizes = Enum.map(stats_list, & &1.step_size)
    assert Enum.uniq(step_sizes) |> length() == 1

    # R-hat across chains should be reasonable
    chains = Enum.map(traces, fn t -> Nx.to_flat_list(t["mu"]) end)
    r = Diagnostics.rhat(chains)
    assert_in_delta r, 1.0, 0.2

    # Total divergences across chains
    total_div = stats_list |> Enum.map(& &1.divergences) |> Enum.sum()
    assert total_div < 50
  end
end
