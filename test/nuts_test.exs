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

    {vag_fn, step_fn, pm, _ncp_info, _multi_step_fn} = Exmc.Compiler.compile_for_sampling(ir)
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
    test "5. Welford mean and variance with Stan-style regularization" do
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

      # Stan-style regularization: α = 5/(n+5) = 0.5, target = 1e-3
      # result = (1-α)*sample_var + α*1e-3
      alpha = 5.0 / (n + 5.0)

      expected_inv_mass =
        Nx.add(
          Nx.multiply(Nx.tensor(1.0 - alpha, type: :f64), sample_var),
          Nx.multiply(Nx.tensor(alpha, type: :f64), Nx.tensor(1.0e-3, type: :f64))
        )

      assert_close(inv_mass, expected_inv_mass, 1.0e-10)
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

    test "7. variance floor + Stan-style regularization" do
      # Identical samples -> zero variance -> clamped to 1e-6, then regularization
      # α = 5/(10+5) = 1/3, target = 1e-3
      # result = (1-1/3)*1e-6 + (1/3)*1e-3 ≈ 3.34e-4
      samples = List.duplicate(Nx.tensor([5.0, 5.0], type: :f64), 10)

      state =
        Enum.reduce(samples, MassMatrix.init(2), fn s, state ->
          MassMatrix.update(state, s)
        end)

      inv_mass = MassMatrix.finalize(state)
      expected = 10.0 / 15.0 * 1.0e-6 + 5.0 / 15.0 * 1.0e-3
      assert_close(inv_mass, Nx.tensor([expected, expected], type: :f64))
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
      assert abs(var - 1.0) < 1.0, "Var[mu] = #{var}, expected near 1"
      assert stats.divergences <= 15
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

      assert abs(mu2_mean - -1.0) < 0.2,
             "E[mu2] = #{mu2_mean}, expected -1.0"

      # Var should be ~0.25 (0.5^2)
      assert abs(mu1_var - 0.25) < 0.2,
             "Var[mu1] = #{mu1_var}, expected 0.25"

      assert abs(mu2_var - 0.25) < 0.2,
             "Var[mu2] = #{mu2_var}, expected 0.25"

      assert stats.divergences <= 15
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

      assert stats.divergences < 20, "Expected few divergences, got #{stats.divergences}"
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

  # =============================================
  # Speculative pre-computation tests (20-22)
  # =============================================

  describe "Speculative pre-computation" do
    @tag timeout: 120_000
    test "20. chain continuity: multi_step_fn(N) prefix matches multi_step_fn(N/2)" do
      ir =
        Builder.new_ir()
        |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
        |> Builder.rv("y", Normal, %{mu: Nx.tensor(2.0), sigma: Nx.tensor(0.5)})
        |> Rewrite.apply()

      {_vag_fn, _step_fn, pm, _ncp_info, multi_step_fn} =
        Exmc.Compiler.compile_for_sampling(ir)

      assert multi_step_fn != nil, "multi_step_fn should be compiled"

      q = Nx.tensor([0.0, 0.0], type: :f64)
      inv_mass = Nx.tensor([1.0, 1.0], type: :f64)
      {_logp, grad} = Exmc.Compiler.value_and_grad(ir) |> elem(0) |> then(& &1.(q))
      key = Nx.Random.key(42)
      {p, _key} = Leapfrog.sample_momentum(key, inv_mass)

      epsilon = 0.1
      eps_t = Nx.tensor(epsilon, type: :f64, backend: Nx.BinaryBackend)

      # Compute 32 steps
      n32 = Nx.tensor(32, type: :s64)
      {q32, p32, logp32, grad32} = multi_step_fn.(q, p, grad, eps_t, inv_mass, n32)

      # Compute 16 steps
      n16 = Nx.tensor(16, type: :s64)
      {q16, p16, logp16, grad16} = multi_step_fn.(q, p, grad, eps_t, inv_mass, n16)

      # Copy to BinaryBackend for comparison
      q32 = Nx.backend_copy(q32, Nx.BinaryBackend)
      p32 = Nx.backend_copy(p32, Nx.BinaryBackend)
      logp32 = Nx.backend_copy(logp32, Nx.BinaryBackend)
      grad32 = Nx.backend_copy(grad32, Nx.BinaryBackend)
      q16 = Nx.backend_copy(q16, Nx.BinaryBackend)
      p16 = Nx.backend_copy(p16, Nx.BinaryBackend)
      logp16 = Nx.backend_copy(logp16, Nx.BinaryBackend)
      grad16 = Nx.backend_copy(grad16, Nx.BinaryBackend)

      d = pm.size

      # multi_step_fn returns {max_steps, d} tensors; only first N rows valid.
      # First 16 rows of 32-step run should match first 16 rows of 16-step run.
      q32_prefix = Nx.slice(q32, [0, 0], [16, d])
      p32_prefix = Nx.slice(p32, [0, 0], [16, d])
      logp32_prefix = Nx.slice(logp32, [0], [16])
      grad32_prefix = Nx.slice(grad32, [0, 0], [16, d])

      q16_valid = Nx.slice(q16, [0, 0], [16, d])
      p16_valid = Nx.slice(p16, [0, 0], [16, d])
      logp16_valid = Nx.slice(logp16, [0], [16])
      grad16_valid = Nx.slice(grad16, [0, 0], [16, d])

      assert_close(q32_prefix, q16_valid, 1.0e-10)
      assert_close(p32_prefix, p16_valid, 1.0e-10)
      assert_close(logp32_prefix, logp16_valid, 1.0e-10)
      assert_close(grad32_prefix, grad16_valid, 1.0e-10)
    end

    @tag timeout: 120_000
    test "21. speculative vs non-speculative: same tree structure" do
      ir =
        Builder.new_ir()
        |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
        |> Rewrite.apply()

      {vag_fn, step_fn, _pm, _ncp_info, multi_step_fn} =
        Exmc.Compiler.compile_for_sampling(ir)

      q = Nx.tensor([0.0], type: :f64)
      {logp, grad} = vag_fn.(q)
      inv_mass = Nx.tensor([1.0], type: :f64)
      inv_mass_list = Nx.to_flat_list(inv_mass)
      key = Nx.Random.key(42)
      {p, _key} = Leapfrog.sample_momentum(key, inv_mass)
      joint_logp_0 = Leapfrog.joint_logp(logp, p, inv_mass)

      rng = :rand.seed_s(:exsss, 42)

      # Run with speculative enabled (default), but disable full-tree NIF
      # so both paths use the same Erlang PRNG (Rust Xoshiro diverges)
      Application.put_env(:exmc, :speculative_precompute, true)
      Application.put_env(:exmc, :use_nif, false)
      Application.put_env(:exmc, :full_tree_nif, false)

      result_spec =
        Tree.build(
          step_fn,
          q,
          p,
          logp,
          grad,
          0.1,
          inv_mass,
          5,
          rng,
          joint_logp_0,
          multi_step_fn,
          inv_mass_list
        )

      # Run with speculative disabled
      Application.put_env(:exmc, :speculative_precompute, false)

      result_no_spec =
        Tree.build(
          step_fn,
          q,
          p,
          logp,
          grad,
          0.1,
          inv_mass,
          5,
          rng,
          joint_logp_0,
          multi_step_fn,
          inv_mass_list
        )

      # Reset
      Application.put_env(:exmc, :speculative_precompute, true)
      Application.put_env(:exmc, :full_tree_nif, true)

      # Same number of steps and divergence status
      assert result_spec.n_steps == result_no_spec.n_steps,
             "n_steps: spec=#{result_spec.n_steps} vs no_spec=#{result_no_spec.n_steps}"

      assert result_spec.divergent == result_no_spec.divergent
      assert result_spec.depth == result_no_spec.depth

      # Accept sums should be close (same states, same RNG path)
      assert abs(result_spec.accept_sum - result_no_spec.accept_sum) < 0.01,
             "accept_sum: spec=#{result_spec.accept_sum} vs no_spec=#{result_no_spec.accept_sum}"
    end

    @tag timeout: 120_000
    test "22. sampling quality with speculative pre-computation" do
      ir =
        Builder.new_ir()
        |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

      Application.put_env(:exmc, :speculative_precompute, true)
      {trace, stats} = Sampler.sample(ir, %{}, num_warmup: 300, num_samples: 300, seed: 42)

      samples = trace["mu"]
      mean = Nx.mean(samples) |> Nx.to_number()
      var = Nx.variance(samples) |> Nx.to_number()

      assert abs(mean) < 0.3, "E[mu] = #{mean}, expected near 0"
      assert abs(var - 1.0) < 1.0, "Var[mu] = #{var}, expected near 1"
      assert stats.divergences <= 15
    end
  end

  describe "Full-tree NIF" do
    @tag timeout: 120_000
    test "23. produces valid posterior" do
      ir =
        Builder.new_ir()
        |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

      Application.put_env(:exmc, :full_tree_nif, true)
      {trace, stats} = Sampler.sample(ir, %{}, num_warmup: 300, num_samples: 300, seed: 42)

      samples = trace["mu"]
      mean = Nx.mean(samples) |> Nx.to_number()
      var = Nx.variance(samples) |> Nx.to_number()

      assert abs(mean) < 0.3, "E[mu] = #{mean}, expected near 0"
      assert abs(var - 1.0) < 1.0, "Var[mu] = #{var}, expected near 1"
      assert stats.divergences <= 15
    end

    @tag timeout: 120_000
    test "24. comparable to speculative path" do
      ir =
        Builder.new_ir()
        |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(5.0)})
        |> Builder.rv("x", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
        |> Builder.obs("x_obs", "x", Nx.tensor([2.5, 3.0, 3.5]))

      # Full-tree NIF path
      Application.put_env(:exmc, :full_tree_nif, true)

      {trace_full, _} =
        Sampler.sample(ir, %{}, num_warmup: 300, num_samples: 500, seed: 42, ncp: false)

      mu_full = Nx.mean(trace_full["mu"]) |> Nx.to_number()

      # Speculative path (disable full-tree NIF)
      Application.put_env(:exmc, :full_tree_nif, false)

      {trace_spec, _} =
        Sampler.sample(ir, %{}, num_warmup: 300, num_samples: 500, seed: 42, ncp: false)

      mu_spec = Nx.mean(trace_spec["mu"]) |> Nx.to_number()

      # Reset
      Application.put_env(:exmc, :full_tree_nif, true)

      # Both should be near the true posterior mean (~3.0)
      assert abs(mu_full - 3.0) < 0.5, "Full-tree mean #{mu_full} too far from 3.0"
      assert abs(mu_spec - 3.0) < 0.5, "Speculative mean #{mu_spec} too far from 3.0"

      assert abs(mu_full - mu_spec) < 1.0,
             "Full-tree #{mu_full} vs Speculative #{mu_spec} too different"
    end
  end
end
