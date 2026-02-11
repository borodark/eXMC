defmodule Exmc.HierarchicalTest do
  use ExUnit.Case, async: true

  alias Exmc.{Builder, LogProb, Compiler}

  # ── LogProb.eval with param refs ────────────────────────────

  test "LogProb.eval with hierarchical param refs matches manual" do
    # mu ~ N(0, 10), x ~ N(mu, 1)
    # logp(mu=2, x=3) = logpdf_N(2; 0,10) + logpdf_N(3; 2,1)
    ir =
      Builder.new_ir()
      |> Builder.rv("mu", Exmc.Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(10.0)})
      |> Builder.rv("x", Exmc.Dist.Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})

    value_map = %{"mu" => Nx.tensor(2.0), "x" => Nx.tensor(3.0)}
    result = LogProb.eval(ir, value_map) |> Nx.to_number()

    # Manual: N(2; 0, 10) + N(3; 2, 1)
    logp_mu = -0.5 * (:math.log(2 * :math.pi()) + 2 * :math.log(10.0) + (2.0 / 10.0) ** 2)
    logp_x = -0.5 * (:math.log(2 * :math.pi()) + (3.0 - 2.0) ** 2)
    expected = logp_mu + logp_x

    assert_in_delta result, expected, 1.0e-6
  end

  # ── Compiler logp matches LogProb.eval ──────────────────────

  test "Compiler logp matches LogProb.eval for hierarchical model" do
    ir =
      Builder.new_ir()
      |> Builder.rv("mu", Exmc.Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(10.0)})
      |> Builder.rv("x", Exmc.Dist.Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})

    value_map = %{"mu" => Nx.tensor(2.0), "x" => Nx.tensor(3.0)}
    logprob_result = LogProb.eval(ir, value_map) |> Nx.to_number()

    {logp_fn, pm} = Compiler.compile(ir)
    flat = Exmc.PointMap.pack(value_map, pm)
    compiler_result = logp_fn.(flat) |> Nx.to_number()

    assert_in_delta compiler_result, logprob_result, 1.0e-6
  end

  # ── Compiler value_and_grad ─────────────────────────────────

  test "Compiler value_and_grad: gradient matches finite differences" do
    ir =
      Builder.new_ir()
      |> Builder.rv("mu", Exmc.Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(10.0)})
      |> Builder.rv("x", Exmc.Dist.Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})

    {vag_fn, pm} = Compiler.value_and_grad(ir)
    value_map = %{"mu" => Nx.tensor(2.0), "x" => Nx.tensor(3.0)}
    flat = Exmc.PointMap.pack(value_map, pm)

    {logp, grad} = vag_fn.(flat)
    grad_list = Nx.to_flat_list(grad)

    # Finite differences
    eps = 1.0e-5

    fd_grad =
      Enum.map(0..(pm.size - 1), fn i ->
        flat_plus =
          Nx.indexed_put(flat, Nx.tensor([[i]]), Nx.tensor([Nx.to_number(flat[i]) + eps]))

        flat_minus =
          Nx.indexed_put(flat, Nx.tensor([[i]]), Nx.tensor([Nx.to_number(flat[i]) - eps]))

        {lp_plus, _} = vag_fn.(flat_plus)
        {lp_minus, _} = vag_fn.(flat_minus)
        (Nx.to_number(lp_plus) - Nx.to_number(lp_minus)) / (2 * eps)
      end)

    Enum.zip(grad_list, fd_grad)
    |> Enum.each(fn {g, fd} ->
      assert_in_delta g, fd, 0.01
    end)

    assert is_float(Nx.to_number(logp))
  end

  # ── Two-level hierarchical model ────────────────────────────

  test "two-level: sigma ~ Exp(1); mu ~ N(0, sigma); obs(mu, data)" do
    # sigma is free (Exp with log transform), mu is free, obs ties mu to data
    ir =
      Builder.new_ir()
      |> Builder.rv("sigma", Exmc.Dist.Exponential, %{lambda: Nx.tensor(1.0)}, transform: :log)
      |> Builder.rv("mu", Exmc.Dist.Normal, %{mu: Nx.tensor(0.0), sigma: "sigma"})
      |> Builder.obs("mu_obs", "mu", Nx.tensor(5.0))

    # sigma=2.0 (unconstrained = log(2)), mu is observed at 5.0
    # The obs node targets "mu" which has a param ref to "sigma"
    # mu is observed -> not free. sigma is free.
    {logp_fn, pm} = Compiler.compile(ir)

    assert pm.size == 1
    entries = pm.entries
    assert hd(entries).id == "sigma"

    # Evaluate at sigma=2 (unconstrained z=log(2))
    z = Nx.tensor([:math.log(2.0)], type: :f64)
    logp = logp_fn.(z) |> Nx.to_number()
    assert is_float(logp)
  end

  # ── Posterior shift test ────────────────────────────────────

  test "posterior shift: obs pulls mu toward data" do
    ir =
      Builder.new_ir()
      |> Builder.rv("mu", Exmc.Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(10.0)})
      |> Builder.rv("x", Exmc.Dist.Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
      |> Builder.obs("x_obs", "x", Nx.tensor(5.0))

    # x is observed -> not free. mu is free.
    {logp_fn, pm} = Compiler.compile(ir)
    assert pm.size == 1

    logp_at_5 = logp_fn.(Nx.tensor([5.0], type: :f64)) |> Nx.to_number()
    logp_at_0 = logp_fn.(Nx.tensor([0.0], type: :f64)) |> Nx.to_number()

    # logp should be higher at mu=5 (near data) than mu=0
    assert logp_at_5 > logp_at_0
  end

  # ── Sampler e2e ─────────────────────────────────────────────

  test "sampler e2e: hierarchical posterior mean shifts toward data" do
    ir =
      Builder.new_ir()
      |> Builder.rv("mu", Exmc.Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(10.0)})
      |> Builder.rv("x", Exmc.Dist.Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
      |> Builder.obs("x_obs", "x", Nx.tensor(5.0))

    {trace, stats} =
      Exmc.NUTS.Sampler.sample(ir, %{}, num_warmup: 200, num_samples: 300, seed: 42)

    mu_samples = trace["mu"]
    mu_mean = Nx.mean(mu_samples) |> Nx.to_number()

    # Posterior mean of mu should be near 5 (data pulls strongly with sigma_x=1 vs sigma_prior=10)
    # Analytic posterior mean = (0/100 + 5/1) / (1/100 + 1/1) = 5/1.01 ≈ 4.95
    assert_in_delta mu_mean, 4.95, 1.0
    assert stats.divergences < 50
  end
end
