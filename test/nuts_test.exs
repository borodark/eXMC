defmodule Exmc.NUTSTest do
  use ExUnit.Case

  @moduledoc false

  alias Exmc.{Builder, Rewrite}
  alias Exmc.Dist.{Normal, Exponential}
  alias Exmc.NUTS.{Leapfrog, MassMatrix, StepSize, Tree, Sampler}
  import Exmc.TestHelper

  # =============================================
  # Helper: build a simple vag_fn for testing
  # =============================================

  defp standard_normal_vag do
    ir =
      Builder.new_ir()
      |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
      |> Rewrite.apply()

    Exmc.Compiler.value_and_grad(ir)
  end

  defp standard_normal_step_fn do
    ir =
      Builder.new_ir()
      |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
      |> Rewrite.apply()

    {vag_fn, step_fn, pm, _ncp_info} = Exmc.Compiler.compile_for_sampling(ir)
    {vag_fn, step_fn, pm}
  end

  # =============================================
  # Leapfrog tests (1-4)
  # =============================================

  describe "Leapfrog" do
    test "1. energy conservation with small epsilon" do
      {vag_fn, _pm} = standard_normal_vag()
      inv_mass = Nx.tensor([1.0], type: :f64)
      q = Nx.tensor([0.5], type: :f64)
      {logp, grad} = vag_fn.(q)
      key = Nx.Random.key(42)
      {p, _key} = Leapfrog.sample_momentum(key, inv_mass)

      h0 = Leapfrog.joint_logp(logp, p, inv_mass) |> Nx.to_number()

      # Take 100 small steps
      epsilon = 0.01

      {_q_final, p_final, logp_final, _grad_final} =
        Enum.reduce(1..100, {q, p, logp, grad}, fn _i, {q, p, _logp, grad} ->
          Leapfrog.step(vag_fn, q, p, grad, epsilon, inv_mass)
        end)

      h1 = Leapfrog.joint_logp(logp_final, p_final, inv_mass) |> Nx.to_number()

      # Energy should be nearly conserved
      assert abs(h1 - h0) < 0.01
    end

    test "2. time reversibility" do
      {vag_fn, _pm} = standard_normal_vag()
      inv_mass = Nx.tensor([1.0], type: :f64)
      q0 = Nx.tensor([0.3], type: :f64)
      {_logp, grad0} = vag_fn.(q0)
      key = Nx.Random.key(7)
      {p0, _key} = Leapfrog.sample_momentum(key, inv_mass)
      epsilon = 0.1

      # Step forward
      {q1, p1, _logp1, grad1} = Leapfrog.step(vag_fn, q0, p0, grad0, epsilon, inv_mass)

      # Negate momentum and step forward again
      p1_neg = Nx.negate(p1)
      {q2, _p2, _logp2, _grad2} = Leapfrog.step(vag_fn, q1, p1_neg, grad1, epsilon, inv_mass)

      # Should return to start
      assert_close(q2, q0, 1.0e-10)
    end

    test "3. kinetic energy correctness" do
      p = Nx.tensor([2.0, 3.0], type: :f64)
      inv_mass = Nx.tensor([0.5, 2.0], type: :f64)

      ke = Leapfrog.kinetic_energy(p, inv_mass) |> Nx.to_number()
      # 0.5 * (4*0.5 + 9*2.0) = 0.5 * (2 + 18) = 10.0
      assert_close(ke, 10.0)
    end

    test "4. momentum sampling variance matches mass matrix" do
      key = Nx.Random.key(123)
      inv_mass = Nx.tensor([0.25, 4.0], type: :f64)

      # Sample many momenta and check variance
      # For p = z / sqrt(inv_mass), Var(p) = 1 / inv_mass
      {samples, _key} =
        Enum.reduce(1..5000, {[], key}, fn _i, {acc, key} ->
          {p, key} = Leapfrog.sample_momentum(key, inv_mass)
          {[p | acc], key}
        end)

      stacked = Nx.stack(samples)
      var = Nx.variance(stacked, axes: [0])
      expected_var = Nx.divide(Nx.tensor(1.0, type: :f64), inv_mass)

      # Generous tolerance for statistical test
      assert_close(var, expected_var, 0.15)
    end
  end

  # =============================================
  # MassMatrix tests (5-7)
  # =============================================

  describe "MassMatrix" do
    test "5. Welford mean and variance match Nx" do
      samples = [
        Nx.tensor([1.0, 2.0], type: :f64),
        Nx.tensor([3.0, 4.0], type: :f64),
        Nx.tensor([5.0, 6.0], type: :f64),
        Nx.tensor([2.0, 8.0], type: :f64),
        Nx.tensor([4.0, 0.0], type: :f64)
      ]

      state =
        Enum.reduce(samples, MassMatrix.init(2), fn s, state ->
          MassMatrix.update(state, s)
        end)

      stacked = Nx.stack(samples)
      expected_mean = Nx.mean(stacked, axes: [0])
      expected_var = Nx.variance(stacked, axes: [0])

      assert_close(state.mean, expected_mean, 1.0e-10)

      # Welford uses n-1 denominator (sample variance), Nx uses n (population variance)
      inv_mass = MassMatrix.finalize(state)
      n = length(samples)
      sample_var = Nx.multiply(expected_var, Nx.tensor(n / (n - 1), type: :f64))
      assert_close(inv_mass, sample_var, 1.0e-10)
    end

    test "6. finalize with n<3 returns identity" do
      state = MassMatrix.init(3)
      inv_mass = MassMatrix.finalize(state)
      assert_close(inv_mass, Nx.tensor([1.0, 1.0, 1.0], type: :f64))

      # Also with 2 samples
      state =
        state
        |> MassMatrix.update(Nx.tensor([1.0, 2.0, 3.0], type: :f64))
        |> MassMatrix.update(Nx.tensor([4.0, 5.0, 6.0], type: :f64))

      inv_mass = MassMatrix.finalize(state)
      assert_close(inv_mass, Nx.tensor([1.0, 1.0, 1.0], type: :f64))
    end

    test "7. variance floor respected" do
      # Identical samples -> zero variance -> clamped to 1e-3
      samples = List.duplicate(Nx.tensor([5.0, 5.0], type: :f64), 10)

      state =
        Enum.reduce(samples, MassMatrix.init(2), fn s, state ->
          MassMatrix.update(state, s)
        end)

      inv_mass = MassMatrix.finalize(state)
      assert_close(inv_mass, Nx.tensor([1.0e-3, 1.0e-3], type: :f64))
    end
  end

  # =============================================
  # StepSize tests (8-10)
  # =============================================

  describe "StepSize" do
    test "8. DA converges toward target" do
      state = StepSize.init(1.0, 0.8)

      # Feed low accept stats (below 0.8 target)
      state_low =
        Enum.reduce(1..50, state, fn _i, state ->
          StepSize.update(state, 0.3)
        end)

      eps_low = :math.exp(state_low.log_epsilon)

      # Feed high accept stats (above 0.8 target)
      state_high =
        Enum.reduce(1..50, StepSize.init(1.0, 0.8), fn _i, state ->
          StepSize.update(state, 0.95)
        end)

      eps_high = :math.exp(state_high.log_epsilon)

      # Low accept -> smaller step size, high accept -> larger step size
      assert eps_low < eps_high
    end

    test "9. find_reasonable_epsilon returns positive finite" do
      {vag_fn, _pm} = standard_normal_vag()
      q = Nx.tensor([0.0], type: :f64)
      {logp, grad} = vag_fn.(q)
      inv_mass = Nx.tensor([1.0], type: :f64)
      key = Nx.Random.key(0)

      {epsilon, _key} = StepSize.find_reasonable_epsilon(vag_fn, q, logp, grad, inv_mass, key)

      assert epsilon > 0
      assert epsilon < 1_000
      assert is_float(epsilon)
    end

    test "10. finalize returns smoothed epsilon" do
      state = StepSize.init(0.5, 0.8)

      state =
        Enum.reduce(1..100, state, fn _i, state ->
          StepSize.update(state, 0.8)
        end)

      eps = StepSize.finalize(state)
      assert eps > 0
      assert is_float(eps)
    end
  end

  # =============================================
  # Tree tests (11-13)
  # =============================================

  describe "Tree" do
    test "11. single-depth tree: n_steps=1" do
      {vag_fn, step_fn, _pm} = standard_normal_step_fn()
      q = Nx.tensor([0.0], type: :f64)
      {logp, grad} = vag_fn.(q)
      inv_mass = Nx.tensor([1.0], type: :f64)
      key = Nx.Random.key(42)
      {p, _key} = Leapfrog.sample_momentum(key, inv_mass)
      joint_logp_0 = Leapfrog.joint_logp(logp, p, inv_mass)

      rng = :rand.seed_s(:exsss, 42)
      result = Tree.build(step_fn, q, p, logp, grad, 0.1, inv_mass, 1, rng, joint_logp_0)

      # With depth 1, we get at least 1 step
      assert result.n_steps >= 1
      assert is_boolean(result.divergent)
    end

    test "12. divergence detection with extreme step size" do
      {vag_fn, step_fn, _pm} = standard_normal_step_fn()
      q = Nx.tensor([0.0], type: :f64)
      {logp, grad} = vag_fn.(q)
      inv_mass = Nx.tensor([1.0], type: :f64)
      key = Nx.Random.key(7)
      {p, _key} = Leapfrog.sample_momentum(key, inv_mass)
      joint_logp_0 = Leapfrog.joint_logp(logp, p, inv_mass)

      rng = :rand.seed_s(:exsss, 7)
      # Extremely large step size -> divergence
      result = Tree.build(step_fn, q, p, logp, grad, 1000.0, inv_mass, 10, rng, joint_logp_0)

      assert result.divergent == true
    end

    test "13. U-turn detection keeps depth small for narrow Normal" do
      {vag_fn, step_fn, _pm} = standard_normal_step_fn()
      q = Nx.tensor([0.0], type: :f64)
      {logp, grad} = vag_fn.(q)
      inv_mass = Nx.tensor([1.0], type: :f64)
      key = Nx.Random.key(99)
      {p, _key} = Leapfrog.sample_momentum(key, inv_mass)
      joint_logp_0 = Leapfrog.joint_logp(logp, p, inv_mass)

      rng = :rand.seed_s(:exsss, 99)
      result = Tree.build(step_fn, q, p, logp, grad, 0.1, inv_mass, 10, rng, joint_logp_0)

      # For standard Normal, tree should U-turn well before max depth 10
      assert result.depth < 10
      assert result.divergent == false
    end
  end

  # =============================================
  # Sampler end-to-end tests (14-19)
  # =============================================

  describe "Sampler" do
    @tag timeout: 120_000
    test "14. standard Normal: E[mu] ~ 0, Var[mu] ~ 1" do
      ir =
        Builder.new_ir()
        |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

      {trace, stats} = Sampler.sample(ir, %{}, num_warmup: 500, num_samples: 500, seed: 42)

      samples = trace["mu"]
      mean = Nx.mean(samples) |> Nx.to_number()
      var = Nx.variance(samples) |> Nx.to_number()

      assert abs(mean) < 0.3, "E[mu] = #{mean}, expected near 0"
      assert abs(var - 1.0) < 0.5, "Var[mu] = #{var}, expected near 1"
      assert stats.divergences < 10
    end

    @tag timeout: 120_000
    test "15. two-parameter model: prior means recovered" do
      # Two independent normals with known means
      # mu1 ~ N(2.0, 0.5), mu2 ~ N(-1.0, 0.5)
      # Verify the sampler recovers the prior means and variances
      ir =
        Builder.new_ir()
        |> Builder.rv("mu1", Normal, %{mu: Nx.tensor(2.0), sigma: Nx.tensor(0.5)})
        |> Builder.rv("mu2", Normal, %{mu: Nx.tensor(-1.0), sigma: Nx.tensor(0.5)})

      {trace, stats} = Sampler.sample(ir, %{}, num_warmup: 300, num_samples: 300, seed: 123)

      mu1_samples = trace["mu1"]
      mu2_samples = trace["mu2"]

      mu1_mean = Nx.mean(mu1_samples) |> Nx.to_number()
      mu2_mean = Nx.mean(mu2_samples) |> Nx.to_number()
      mu1_var = Nx.variance(mu1_samples) |> Nx.to_number()
      mu2_var = Nx.variance(mu2_samples) |> Nx.to_number()

      assert abs(mu1_mean - 2.0) < 0.2,
             "E[mu1] = #{mu1_mean}, expected 2.0"

      assert abs(mu2_mean - (-1.0)) < 0.2,
             "E[mu2] = #{mu2_mean}, expected -1.0"

      # Var should be ~0.25 (0.5^2)
      assert abs(mu1_var - 0.25) < 0.2,
             "Var[mu1] = #{mu1_var}, expected 0.25"

      assert abs(mu2_var - 0.25) < 0.2,
             "Var[mu2] = #{mu2_var}, expected 0.25"

      assert stats.divergences < 10
    end

    @tag timeout: 120_000
    test "16. constrained parameter: Exponential trace values all positive" do
      ir =
        Builder.new_ir()
        |> Builder.rv("rate", Exponential, %{lambda: Nx.tensor(1.0)})

      {trace, _stats} = Sampler.sample(ir, %{}, num_warmup: 200, num_samples: 200, seed: 77)

      samples = trace["rate"]
      min_val = Nx.reduce_min(samples) |> Nx.to_number()

      assert min_val > 0.0, "All Exponential samples should be positive, got min=#{min_val}"
    end

    @tag timeout: 120_000
    test "17. no divergences for simple well-conditioned model" do
      ir =
        Builder.new_ir()
        |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

      {_trace, stats} = Sampler.sample(ir, %{}, num_warmup: 200, num_samples: 200, seed: 11)

      assert stats.divergences < 5, "Expected few divergences, got #{stats.divergences}"
    end

    @tag timeout: 120_000
    test "18. seed reproducibility" do
      ir =
        Builder.new_ir()
        |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

      {trace1, _} = Sampler.sample(ir, %{}, num_warmup: 200, num_samples: 100, seed: 999)
      {trace2, _} = Sampler.sample(ir, %{}, num_warmup: 200, num_samples: 100, seed: 999)

      assert_close(trace1["x"], trace2["x"], 1.0e-10)
    end

    @tag timeout: 120_000
    test "19. stats structure" do
      ir =
        Builder.new_ir()
        |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

      {_trace, stats} = Sampler.sample(ir, %{}, num_warmup: 50, num_samples: 50, seed: 0)

      assert is_float(stats.step_size) or is_number(stats.step_size)
      assert %Nx.Tensor{} = stats.inv_mass_diag
      assert is_integer(stats.divergences)
      assert stats.num_warmup == 50
      assert stats.num_samples == 50
    end
  end
end
