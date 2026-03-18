defmodule Exmc.WeibullTest do
  use ExUnit.Case, async: true

  alias Exmc.Dist.Weibull
  alias Exmc.Dist.Censored
  alias Exmc.{Builder, NUTS.Sampler}
  alias Exmc.Dist.{Normal, Exponential}

  # ── Weibull logpdf ──────────────────────────────────────

  test "Weibull logpdf at t=1, k=2, lambda=1" do
    # f(1; 2, 1) = 2 * 1 * exp(-1) = 2*exp(-1)
    # logpdf = log(2) + 0 - 1 = log(2) - 1
    expected = :math.log(2.0) - 1.0
    result = Weibull.logpdf(Nx.tensor(1.0), %{k: Nx.tensor(2.0), lambda: Nx.tensor(1.0)}) |> Nx.to_number()
    assert_in_delta result, expected, 1.0e-6
  end

  test "Weibull logpdf at t=2, k=1, lambda=1 matches Exponential" do
    # Weibull(k=1, lambda=1) = Exponential(rate=1)
    # logpdf = log(1) - 1*log(1) + 0*log(2) - 2 = -2
    expected = Exponential.logpdf(Nx.tensor(2.0), %{lambda: Nx.tensor(1.0)}) |> Nx.to_number()
    result = Weibull.logpdf(Nx.tensor(2.0), %{k: Nx.tensor(1.0), lambda: Nx.tensor(1.0)}) |> Nx.to_number()
    assert_in_delta result, expected, 1.0e-6
  end

  test "Weibull logpdf at t=0.5, k=3, lambda=2" do
    # Analytic: log(3) - 3*log(2) + 2*log(0.5) - (0.5/2)^3
    # = log(3) - 3*log(2) + 2*log(0.5) - 0.015625
    t = 0.5
    k = 3.0
    lam = 2.0
    expected = :math.log(k) - k * :math.log(lam) + (k - 1) * :math.log(t) - :math.pow(t / lam, k)
    result = Weibull.logpdf(Nx.tensor(t), %{k: Nx.tensor(k), lambda: Nx.tensor(lam)}) |> Nx.to_number()
    assert_in_delta result, expected, 1.0e-6
  end

  # ── Support and transform ───────────────────────────────

  test "Weibull support and transform" do
    params = %{k: Nx.tensor(2.0), lambda: Nx.tensor(1.0)}
    assert Weibull.support(params) == :positive
    assert Weibull.transform(params) == :log
  end

  # ── Sampling ────────────────────────────────────────────

  test "Weibull sample produces positive values" do
    params = %{k: Nx.tensor(2.0), lambda: Nx.tensor(3.0)}
    rng = :rand.seed_s(:exsss, 42)
    {value, _rng} = Weibull.sample(params, rng)
    assert Nx.to_number(value) > 0.0
  end

  test "Weibull sample mean approximates lambda * Gamma(1 + 1/k)" do
    k = 2.0
    lambda = 3.0
    # E[X] = lambda * Gamma(1 + 1/k) = 3 * Gamma(1.5) = 3 * sqrt(pi)/2
    expected_mean = lambda * :math.sqrt(:math.pi()) / 2.0

    params = %{k: Nx.tensor(k), lambda: Nx.tensor(lambda)}
    rng = :rand.seed_s(:exsss, 42)

    {samples, _rng} =
      Enum.map_reduce(1..2000, rng, fn _, rng ->
        {val, rng} = Weibull.sample(params, rng)
        {Nx.to_number(val), rng}
      end)

    sample_mean = Enum.sum(samples) / length(samples)
    assert_in_delta sample_mean, expected_mean, 0.15
  end

  # ── Survival function ───────────────────────────────────

  test "Weibull log_survival at t=0+ is ~0" do
    result = Weibull.log_survival(Nx.tensor(0.001), %{k: Nx.tensor(2.0), lambda: Nx.tensor(1.0)}) |> Nx.to_number()
    assert_in_delta result, 0.0, 1.0e-4
  end

  test "Weibull log_survival at t=lambda, k=1 is -1" do
    # SF(lambda; 1, lambda) = exp(-1), so log(SF) = -1
    result = Weibull.log_survival(Nx.tensor(2.0), %{k: Nx.tensor(1.0), lambda: Nx.tensor(2.0)}) |> Nx.to_number()
    assert_in_delta result, -1.0, 1.0e-6
  end

  # ── Censored integration ────────────────────────────────

  test "Censored right-censored Weibull log-likelihood" do
    # Right-censored at t=1 with k=2, lambda=1: log(SF) = -(1/1)^2 = -1
    result = Censored.log_likelihood(:right, Nx.tensor(1.0), Exmc.Dist.Weibull,
      %{k: Nx.tensor(2.0), lambda: Nx.tensor(1.0)}) |> Nx.to_number()
    assert_in_delta result, -1.0, 1.0e-6
  end

  # ── Compile + sample integration ────────────────────────

  test "Weibull RV compiles and samples via NUTS" do
    ir =
      Builder.new_ir()
      |> Builder.rv("k", Exponential, %{lambda: Nx.tensor(1.0)})
      |> Builder.rv("t", Weibull, %{k: "k", lambda: Nx.tensor(2.0)})
      |> Builder.obs("t_obs", "t", Nx.tensor(1.5))

    init = %{"k" => Nx.tensor(1.5)}

    {trace, stats} =
      Sampler.sample(ir, init, num_warmup: 200, num_samples: 200, seed: 42, ncp: false)

    k_mean = Nx.to_number(Nx.mean(trace["k"]))
    assert k_mean > 0.0
    assert stats.divergences < 50
  end
end
