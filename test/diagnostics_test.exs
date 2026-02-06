defmodule Exmc.DiagnosticsTest do
  use ExUnit.Case, async: true

  alias Exmc.Diagnostics

  # ── Summary ─────────────────────────────────────────────────

  test "summary: known trace, verify mean/std/quantiles" do
    # 1..100
    samples = Nx.tensor(Enum.to_list(1..100), type: :f64)
    trace = %{"x" => samples}
    result = Diagnostics.summary(trace)

    x = result["x"]
    assert_in_delta x.mean, 50.5, 0.01
    # std of uniform 1..100: sqrt((100^2-1)/12) ≈ 28.87
    assert_in_delta x.std, 28.87, 0.1
    assert_in_delta x.q50, 50.5, 1.0
    assert x.q5 < x.q25
    assert x.q25 < x.q50
    assert x.q50 < x.q75
    assert x.q75 < x.q95
  end

  # ── ESS ─────────────────────────────────────────────────────

  test "ess: independent samples -> ESS near N" do
    n = 1000
    rng = :rand.seed_s(:exsss, 42)

    {values, _rng} =
      Enum.map_reduce(1..n, rng, fn _i, rng ->
        {v, rng} = :rand.normal_s(rng)
        {v, rng}
      end)

    ess = Diagnostics.ess(values)
    # ESS should be close to N for iid samples
    assert ess > n * 0.5
  end

  test "ess: AR(1) rho=0.99 -> ESS << N" do
    n = 1000
    rho = 0.99
    rng = :rand.seed_s(:exsss, 42)

    {values, _rng} =
      Enum.map_reduce(1..n, {0.0, rng}, fn _i, {prev, rng} ->
        {noise, rng} = :rand.normal_s(rng)
        x = rho * prev + :math.sqrt(1.0 - rho * rho) * noise
        {x, {x, rng}}
      end)

    ess = Diagnostics.ess(values)
    # ESS should be much less than N for highly correlated samples
    assert ess < n * 0.2
  end

  # ── R-hat ───────────────────────────────────────────────────

  test "rhat: identical chains -> near 1.0" do
    rng = :rand.seed_s(:exsss, 42)

    {chain1, rng} =
      Enum.map_reduce(1..500, rng, fn _i, rng ->
        {v, rng} = :rand.normal_s(rng)
        {v, rng}
      end)

    {chain2, _rng} =
      Enum.map_reduce(1..500, rng, fn _i, rng ->
        {v, rng} = :rand.normal_s(rng)
        {v, rng}
      end)

    r = Diagnostics.rhat([chain1, chain2])
    assert_in_delta r, 1.0, 0.1
  end

  test "rhat: different-mean chains -> >>1.0" do
    rng = :rand.seed_s(:exsss, 42)

    {chain1, rng} =
      Enum.map_reduce(1..500, rng, fn _i, rng ->
        {v, rng} = :rand.normal_s(rng)
        {v, rng}
      end)

    {chain2, _rng} =
      Enum.map_reduce(1..500, rng, fn _i, rng ->
        {v, rng} = :rand.normal_s(rng)
        {v + 10.0, rng}
      end)

    r = Diagnostics.rhat([chain1, chain2])
    assert r > 1.5
  end

  # ── Autocorrelation ─────────────────────────────────────────

  test "autocorrelation: white noise -> near-zero lags" do
    n = 1000
    rng = :rand.seed_s(:exsss, 42)

    {values, _rng} =
      Enum.map_reduce(1..n, rng, fn _i, rng ->
        {v, rng} = :rand.normal_s(rng)
        {v, rng}
      end)

    acf = Diagnostics.autocorrelation(values, 10)
    # Lag 0 should be 1.0
    assert_in_delta hd(acf), 1.0, 1.0e-10
    # Lags 1+ should be near zero for white noise
    Enum.drop(acf, 1)
    |> Enum.each(fn v -> assert abs(v) < 0.1 end)
  end

  test "autocorrelation: AR(1) rho=0.8 -> geometric decay" do
    n = 5000
    rho = 0.8
    rng = :rand.seed_s(:exsss, 42)

    {values, _rng} =
      Enum.map_reduce(1..n, {0.0, rng}, fn _i, {prev, rng} ->
        {noise, rng} = :rand.normal_s(rng)
        x = rho * prev + :math.sqrt(1.0 - rho * rho) * noise
        {x, {x, rng}}
      end)

    acf = Diagnostics.autocorrelation(values, 5)
    # Lag 0 = 1.0, lag 1 ≈ 0.8, lag 2 ≈ 0.64, etc.
    assert_in_delta Enum.at(acf, 0), 1.0, 1.0e-10
    assert_in_delta Enum.at(acf, 1), 0.8, 0.1
    assert_in_delta Enum.at(acf, 2), 0.64, 0.15

    # ACF should be monotonically decreasing
    pairs = Enum.zip(acf, tl(acf))
    Enum.each(pairs, fn {a, b} -> assert a > b end)
  end

  # ── Sample stats in sampler output ──────────────────────────

  test "sample_stats present in sampler output with correct keys" do
    ir =
      Exmc.Builder.new_ir()
      |> Exmc.Builder.rv("x", Exmc.Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

    {_trace, stats} =
      Exmc.NUTS.Sampler.sample(ir, %{}, num_warmup: 50, num_samples: 20, seed: 42)

    assert is_list(stats.sample_stats)
    assert length(stats.sample_stats) == 20

    first = hd(stats.sample_stats)
    assert Map.has_key?(first, :tree_depth)
    assert Map.has_key?(first, :n_steps)
    assert Map.has_key?(first, :divergent)
    assert Map.has_key?(first, :accept_prob)
    assert is_integer(first.n_steps)
    assert is_boolean(first.divergent)
    assert is_float(first.accept_prob) or is_integer(first.accept_prob)
  end
end
