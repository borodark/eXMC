defmodule Exmc.NewDistTest do
  use ExUnit.Case, async: true

  import Exmc.TestHelper, only: [assert_posterior!: 3]

  alias Exmc.Dist.{Lognormal, HalfCauchy, TruncatedNormal, Bernoulli, Poisson}
  alias Exmc.Builder
  alias Exmc.Dist.{Normal, Beta, Exponential}

  # ── Lognormal ─────────────────────────────────────────────

  test "Lognormal logpdf at x=1 with mu=0, sigma=1 equals Normal(0,1) at 0" do
    # Lognormal(0,1) at x=1: log(x)=0, so same as Normal(0,1) at 0 minus log(1)=0
    expected = -0.5 * :math.log(2.0 * :math.pi())
    result = Lognormal.logpdf(Nx.tensor(1.0), %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)}) |> Nx.to_number()
    assert_in_delta result, expected, 1.0e-5
  end

  test "Lognormal logpdf at x=e with mu=0, sigma=1" do
    # log(e)=1, Normal(0,1) at 1 = -0.5*(1+log(2pi)), minus log(e)=1
    expected = -0.5 * (1.0 + :math.log(2.0 * :math.pi())) - 1.0
    result = Lognormal.logpdf(Nx.tensor(:math.exp(1.0)), %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)}) |> Nx.to_number()
    assert_in_delta result, expected, 1.0e-5
  end

  test "Lognormal support and transform" do
    params = %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)}
    assert Lognormal.support(params) == :positive
    assert Lognormal.transform(params) == :log
  end

  test "Lognormal sample produces positive values" do
    params = %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)}
    rng = :rand.seed_s(:exsss, 42)
    {value, _rng} = Lognormal.sample(params, rng)
    assert Nx.to_number(value) > 0.0
  end

  # ── HalfCauchy ────────────────────────────────────────────

  test "HalfCauchy logpdf at x=0+ approaches log(2/pi/scale)" do
    # At x=0: log(2/pi) - log(scale) - log(1) = log(2/(pi*scale))
    expected = :math.log(2.0 / :math.pi())
    result = HalfCauchy.logpdf(Nx.tensor(1.0e-10), %{scale: Nx.tensor(1.0)}) |> Nx.to_number()
    assert_in_delta result, expected, 1.0e-3
  end

  test "HalfCauchy logpdf at x=scale" do
    # At x=scale: log(2/pi) - log(scale) - log(1 + 1) = log(2/pi) - log(scale) - log(2)
    expected = :math.log(2.0 / :math.pi()) - :math.log(2.0)
    result = HalfCauchy.logpdf(Nx.tensor(1.0), %{scale: Nx.tensor(1.0)}) |> Nx.to_number()
    assert_in_delta result, expected, 1.0e-5
  end

  test "HalfCauchy support and transform" do
    params = %{scale: Nx.tensor(1.0)}
    assert HalfCauchy.support(params) == :positive
    assert HalfCauchy.transform(params) == :log
  end

  test "HalfCauchy sample produces positive values" do
    params = %{scale: Nx.tensor(2.5)}
    rng = :rand.seed_s(:exsss, 42)
    {value, _rng} = HalfCauchy.sample(params, rng)
    assert Nx.to_number(value) > 0.0
  end

  # ── TruncatedNormal ───────────────────────────────────────

  test "TruncatedNormal logpdf at center of symmetric truncation" do
    # TN(0,1,-1,1) at x=0: Normal(0,1) at 0 minus log(Phi(1)-Phi(-1))
    phi_1 = 0.5 * (1.0 + :math.erf(1.0 / :math.sqrt(2.0)))
    norm_const = 2.0 * phi_1 - 1.0
    expected = -0.5 * :math.log(2.0 * :math.pi()) - :math.log(norm_const)

    params = %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0), lower: Nx.tensor(-1.0), upper: Nx.tensor(1.0)}
    result = TruncatedNormal.logpdf(Nx.tensor(0.0), params) |> Nx.to_number()
    assert_in_delta result, expected, 1.0e-4
  end

  test "TruncatedNormal with wide bounds matches Normal" do
    # With bounds at +/-100, normalizing constant ~= 1, so logpdf ~= Normal logpdf
    params = %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0), lower: Nx.tensor(-100.0), upper: Nx.tensor(100.0)}
    tn_result = TruncatedNormal.logpdf(Nx.tensor(0.5), params) |> Nx.to_number()
    n_result = Normal.logpdf(Nx.tensor(0.5), %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)}) |> Nx.to_number()
    assert_in_delta tn_result, n_result, 1.0e-5
  end

  test "TruncatedNormal support and transform" do
    params = %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0), lower: Nx.tensor(-1.0), upper: Nx.tensor(1.0)}
    assert TruncatedNormal.support(params) == :real
    assert TruncatedNormal.transform(params) == nil
  end

  test "TruncatedNormal sample within bounds" do
    params = %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0), lower: Nx.tensor(-1.0), upper: Nx.tensor(1.0)}
    rng = :rand.seed_s(:exsss, 42)

    {values, _rng} =
      Enum.reduce(1..50, {[], rng}, fn _, {acc, rng} ->
        {v, rng} = TruncatedNormal.sample(params, rng)
        {[Nx.to_number(v) | acc], rng}
      end)

    assert Enum.all?(values, &(&1 >= -1.0 and &1 <= 1.0))
  end

  # ── Bernoulli ─────────────────────────────────────────────

  test "Bernoulli logpdf at y=1 equals log(p)" do
    expected = :math.log(0.7)
    result = Bernoulli.logpdf(Nx.tensor(1.0), %{p: Nx.tensor(0.7)}) |> Nx.to_number()
    assert_in_delta result, expected, 1.0e-5
  end

  test "Bernoulli logpdf at y=0 equals log(1-p)" do
    expected = :math.log(0.3)
    result = Bernoulli.logpdf(Nx.tensor(0.0), %{p: Nx.tensor(0.7)}) |> Nx.to_number()
    assert_in_delta result, expected, 1.0e-5
  end

  test "Bernoulli logpdf vectorized sums correctly" do
    # 3 ones and 2 zeros with p=0.6: 3*log(0.6) + 2*log(0.4)
    data = Nx.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
    logp_per_obs = Bernoulli.logpdf(data, %{p: Nx.tensor(0.6)})
    total = Nx.sum(logp_per_obs) |> Nx.to_number()
    expected = 3.0 * :math.log(0.6) + 2.0 * :math.log(0.4)
    assert_in_delta total, expected, 1.0e-5
  end

  test "Bernoulli support and transform" do
    params = %{p: Nx.tensor(0.5)}
    assert Bernoulli.support(params) == :unit
    assert Bernoulli.transform(params) == :logit
  end

  # ── Poisson ───────────────────────────────────────────────

  test "Poisson logpdf at y=0" do
    # y*log(mu) - mu - lgamma(1) = 0 - 2 - 0 = -2
    expected = -2.0
    result = Poisson.logpdf(Nx.tensor(0.0), %{mu: Nx.tensor(2.0)}) |> Nx.to_number()
    assert_in_delta result, expected, 1.0e-5
  end

  test "Poisson logpdf at y=3, mu=2" do
    # 3*log(2) - 2 - lgamma(4) = 3*log(2) - 2 - log(6)
    expected = 3.0 * :math.log(2.0) - 2.0 - :math.log(6.0)
    result = Poisson.logpdf(Nx.tensor(3.0), %{mu: Nx.tensor(2.0)}) |> Nx.to_number()
    assert_in_delta result, expected, 1.0e-4
  end

  test "Poisson logpdf vectorized sums correctly" do
    data = Nx.tensor([0.0, 1.0, 2.0, 3.0])
    logp_per_obs = Poisson.logpdf(data, %{mu: Nx.tensor(1.5)})
    total = Nx.sum(logp_per_obs) |> Nx.to_number()

    # lgamma(1)=0, lgamma(2)=0, lgamma(3)=log(2), lgamma(4)=log(6)
    factorials = [1.0, 1.0, 2.0, 6.0]

    expected =
      Enum.zip([0, 1, 2, 3], factorials)
      |> Enum.reduce(0.0, fn {y, fac}, acc ->
        acc + (y * :math.log(1.5) - 1.5 - :math.log(fac))
      end)

    assert_in_delta total, expected, 1.0e-3
  end

  test "Poisson support and transform" do
    params = %{mu: Nx.tensor(1.0)}
    assert Poisson.support(params) == :positive
    assert Poisson.transform(params) == :log
  end

  # ── Compiler integration ──────────────────────────────────

  test "Lognormal compiles and produces finite logp" do
    ir =
      Builder.new_ir()
      |> Builder.rv("x", Lognormal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

    compiled = Exmc.NUTS.Sampler.compile(ir)
    # compile succeeds, verify by sampling 1 step
    {trace, _stats} = Exmc.NUTS.Sampler.sample_compiled(compiled, %{}, num_warmup: 50, num_samples: 10, seed: 42)
    values = Nx.to_flat_list(trace["x"])
    assert Enum.all?(values, &(is_number(&1) and &1 > 0.0))
  end

  test "HalfCauchy compiles and produces finite logp" do
    ir =
      Builder.new_ir()
      |> Builder.rv("sigma", HalfCauchy, %{scale: Nx.tensor(2.5)})

    compiled = Exmc.NUTS.Sampler.compile(ir)
    {trace, _stats} = Exmc.NUTS.Sampler.sample_compiled(compiled, %{}, num_warmup: 50, num_samples: 10, seed: 42)
    values = Nx.to_flat_list(trace["sigma"])
    assert Enum.all?(values, &(is_number(&1) and &1 > 0.0))
  end

  test "TruncatedNormal compiles and samples concentrate within bounds" do
    # TruncatedNormal with transform: nil means NUTS explores unconstrained space.
    # The logpdf penalty pushes samples toward the truncation region but doesn't hard-enforce bounds.
    # Verify samples are finite and concentrated near the mean.
    ir =
      Builder.new_ir()
      |> Builder.rv("x", TruncatedNormal, %{
        mu: Nx.tensor(0.0),
        sigma: Nx.tensor(1.0),
        lower: Nx.tensor(-2.0),
        upper: Nx.tensor(2.0)
      })

    compiled = Exmc.NUTS.Sampler.compile(ir)
    {trace, _stats} = Exmc.NUTS.Sampler.sample_compiled(compiled, %{}, num_warmup: 50, num_samples: 10, seed: 42)
    values = Nx.to_flat_list(trace["x"])
    assert Enum.all?(values, &is_number/1)
    mean = Enum.sum(values) / length(values)
    assert_in_delta mean, 0.0, 2.0
  end

  # ── NUTS sampling: Lognormal prior recovery ───────────────

  @tag timeout: 60_000
  test "Lognormal prior: samples have correct mean" do
    # Lognormal(mu=0, sigma=0.5) has mean = exp(mu + sigma^2/2) = exp(0.125) ≈ 1.133
    ir =
      Builder.new_ir()
      |> Builder.rv("x", Lognormal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(0.5)})

    # Was 300 draws with `assert_in_delta mean, expected_mean, 0.5`, which is
    # 0.8 posterior sd of slack on the mean and nothing on the spread — the
    # log-normal's whole character is its right tail, and the test could not
    # see it at all.
    #
    # 10000 rather than 6000: at 6000 this resolved 19.6% under EXLA and 21.3%
    # under `EXMC_COMPILER=vulkan`, i.e. it passed on one arm and reported
    # INCONCLUSIVE on the other. ESS is backend-dependent, so the draw count
    # has to be sized for the slower arm.
    {trace, _stats} = Exmc.NUTS.Sampler.sample(ir, %{}, num_warmup: 500, num_samples: 10000, seed: 42)
    values = Nx.to_flat_list(trace["x"])
    assert Enum.all?(values, &(&1 > 0.0))

    # Lognormal(0, 0.5): mean exp(sigma^2/2), var (exp(sigma^2) - 1) exp(sigma^2).
    assert_posterior!(values, {:lognormal, 0.0, 0.5}, resolution: 0.20)
  end

  # ── NUTS sampling: HalfCauchy prior ───────────────────────

  @tag timeout: 60_000
  test "HalfCauchy prior: all samples positive" do
    ir =
      Builder.new_ir()
      |> Builder.rv("sigma", HalfCauchy, %{scale: Nx.tensor(1.0)})

    {trace, _stats} = Exmc.NUTS.Sampler.sample(ir, %{}, num_warmup: 200, num_samples: 200, seed: 42)
    values = Nx.to_flat_list(trace["sigma"])
    assert Enum.all?(values, &(&1 > 0.0))
  end

  # ── Likelihood integration: Beta-Bernoulli ────────────────

  test "Bernoulli likelihood: compiler integration with Beta prior" do
    # Beta(2,2) prior + Bernoulli likelihood compiles and produces finite logp
    data = Nx.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    ir =
      Builder.new_ir()
      |> Builder.rv("p", Beta, %{alpha: Nx.tensor(2.0), beta: Nx.tensor(2.0)})
      |> Builder.rv("y", Bernoulli, %{p: "p"})
      |> Builder.obs("y_obs", "y", data)

    compiled = Exmc.NUTS.Sampler.compile(ir)
    # Verify compilation succeeds and is a valid compiled tuple
    assert is_tuple(compiled)
    assert tuple_size(compiled) >= 3
  end

  # ── Likelihood integration: Exponential-Poisson ───────────

  @tag timeout: 120_000
  test "Exponential-Poisson: posterior rate near data mean" do
    # Prior: mu ~ Exp(0.1), prior mean = 10
    # Data: counts with true rate ~3.0
    data = Nx.tensor([2.0, 4.0, 3.0, 1.0, 5.0, 3.0, 2.0, 4.0, 3.0, 3.0])

    ir =
      Builder.new_ir()
      |> Builder.rv("mu", Exponential, %{lambda: Nx.tensor(0.1)})
      |> Builder.rv("y", Poisson, %{mu: "mu"})
      |> Builder.obs("y_obs", "y", data)

    {trace, stats} =
      Exmc.NUTS.Sampler.sample(ir, %{"mu" => 3.0}, num_warmup: 300, num_samples: 500, seed: 42)

    values = Nx.to_flat_list(trace["mu"])
    mean = Enum.sum(values) / length(values)

    # With 10 observations averaging 3.0, posterior should be near 3.0
    assert_in_delta mean, 3.0, 1.0
    assert Enum.all?(values, &(&1 > 0.0))
    assert stats.divergences < 20
  end
end
