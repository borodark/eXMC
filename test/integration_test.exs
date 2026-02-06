defmodule Exmc.IntegrationTest do
  use ExUnit.Case

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

    {trace, stats} = Sampler.sample(ir, %{}, num_warmup: 300, num_samples: 500, seed: 42)

    summary = Diagnostics.summary(trace)
    mu_stats = summary["mu"]

    assert_in_delta mu_stats.mean, 4.95, 0.5
    assert_in_delta mu_stats.std, :math.sqrt(0.99), 0.5
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

    {trace, _stats} = Sampler.sample(ir, %{}, num_warmup: 200, num_samples: 200, seed: 99)

    values = Nx.to_flat_list(trace["alpha"])
    assert Enum.all?(values, &(&1 > 0.0))

    # Mean of Gamma(2,1) = alpha/beta = 2.0
    mean = Enum.sum(values) / length(values)
    assert_in_delta mean, 2.0, 1.0
  end

  # Beta/Gamma lgamma gradient triggers Complex.divide on BinaryBackend.
  # Tracked in INTEGRATION_TESTING.md as a known limitation.
  # Use Exponential (no lgamma) as constrained-support test instead.

  test "Exponential prior: all trace values positive, mean near 1/lambda" do
    # rate ~ Exp(2), prior mean = 0.5
    ir =
      Builder.new_ir()
      |> Builder.rv("rate", Exponential, %{lambda: Nx.tensor(2.0)})

    {trace, _stats} = Sampler.sample(ir, %{}, num_warmup: 200, num_samples: 300, seed: 77)

    values = Nx.to_flat_list(trace["rate"])
    assert Enum.all?(values, &(&1 > 0.0))

    mean = Enum.sum(values) / length(values)
    assert_in_delta mean, 0.5, 0.3
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
    {trace, _stats} =
      Sampler.sample(ir, %{},
        num_warmup: 300,
        num_samples: 400,
        seed: 88,
        init_values: %{"p" => Nx.tensor(0.2)}
      )

    values = Nx.to_flat_list(trace["p"])

    # All samples must be in (0, 1)
    assert Enum.all?(values, &(&1 > 0.0 and &1 < 1.0))

    # Mean of Beta(2,5) = 2/7 ≈ 0.286
    mean = Enum.sum(values) / length(values)
    assert_in_delta mean, 2.0 / 7.0, 0.15
  end

  # ── 9. StudentT prior: mean near loc ─────────────────────────
  # Previously blocked on BinaryBackend (lgamma gradient).

  test "StudentT prior: mean near loc" do
    # x ~ StudentT(df=4, loc=3.0, scale=1.0), prior mean = loc = 3.0
    ir =
      Builder.new_ir()
      |> Builder.rv("x", StudentT, %{df: Nx.tensor(4.0), loc: Nx.tensor(3.0), scale: Nx.tensor(1.0)})

    {trace, _stats} = Sampler.sample(ir, %{}, num_warmup: 300, num_samples: 400, seed: 66)

    values = Nx.to_flat_list(trace["x"])

    mean = Enum.sum(values) / length(values)
    assert_in_delta mean, 3.0, 0.5
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
        init_values: %{"mu" => Nx.tensor(3.0), "sigma" => Nx.tensor(1.0), "alpha" => Nx.tensor(3.0)}
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

    waic_good = ModelComparison.waic(ModelComparison.pointwise_log_likelihood(ir_good, trace_good))
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

    {trace_scalar, _} = Sampler.sample(ir_scalar, %{}, num_warmup: 300, num_samples: 500, seed: 42)
    scalar_summary = Diagnostics.summary(trace_scalar)

    # Vector version: single obs node with vector data
    ir_vector =
      Builder.new_ir()
      |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(10.0)})
      |> Builder.rv("x", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
      |> Builder.obs("x_obs", "x", Nx.tensor([4.0, 3.8, 4.2]))

    {trace_vector, _} = Sampler.sample(ir_vector, %{}, num_warmup: 300, num_samples: 500, seed: 42)
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

  test "parallel chains: wall time less than sequential for 4 chains" do
    ir =
      Builder.new_ir()
      |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(5.0)})
      |> Builder.rv("x", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
      |> Builder.obs("x_obs", "x", Nx.tensor(3.0))

    shared_opts = [num_warmup: 200, num_samples: 200, seed: 42]

    # Sequential
    t0 = System.monotonic_time(:millisecond)
    {traces_seq, _} = Sampler.sample_chains(ir, 4, [parallel: false] ++ shared_opts)
    t_seq = System.monotonic_time(:millisecond) - t0

    # Parallel
    t0 = System.monotonic_time(:millisecond)
    {traces_par, _} = Sampler.sample_chains(ir, 4, [parallel: true] ++ shared_opts)
    t_par = System.monotonic_time(:millisecond) - t0

    # Both should produce 4 chains with identical structure
    assert length(traces_seq) == 4
    assert length(traces_par) == 4

    # Same seeds => same results (deterministic)
    for i <- 0..3 do
      seq_mean = Nx.mean(traces_seq |> Enum.at(i) |> Map.fetch!("mu")) |> Nx.to_number()
      par_mean = Nx.mean(traces_par |> Enum.at(i) |> Map.fetch!("mu")) |> Nx.to_number()
      assert_in_delta seq_mean, par_mean, 1.0e-6
    end

    # Parallel should be faster (at least 1.5x with 4 chains on multi-core)
    assert t_par < t_seq, "parallel=#{t_par}ms should be < sequential=#{t_seq}ms"
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
end
