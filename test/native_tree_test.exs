defmodule Exmc.NativeTreeTest do
  use ExUnit.Case, async: true

  alias Exmc.NUTS.NativeTree
  alias Exmc.NUTS.Tree

  describe "NIF availability" do
    test "NativeTree module loads and NIF functions are available" do
      assert Tree.nif_available?()
    end
  end

  describe "init_trajectory + get_result" do
    test "creates trajectory and extracts initial state" do
      q = [1.0, 2.0, 3.0]
      p = [0.1, 0.2, 0.3]
      grad = [-1.0, -2.0, -3.0]
      logp = -5.0

      ref = NativeTree.init_trajectory(q, p, grad, logp)
      assert is_reference(ref)

      result = NativeTree.get_result(ref)
      assert is_map(result)
      assert result.q == q
      assert result.logp == logp
      assert result.grad == grad
      assert result.n_steps == 0
      assert result.divergent == false
      assert result.accept_sum == 0.0
      assert result.depth == 0
    end
  end

  describe "is_terminated" do
    test "new trajectory is not terminated" do
      ref = NativeTree.init_trajectory([1.0], [0.5], [-1.0], -2.0)
      assert NativeTree.is_terminated(ref) == false
    end
  end

  describe "get_endpoint" do
    test "returns left/right endpoints" do
      q = [1.0, 2.0]
      p = [0.1, 0.2]
      grad = [-1.0, -2.0]

      ref = NativeTree.init_trajectory(q, p, grad, -5.0)

      {q_r, p_r, grad_r} = NativeTree.get_endpoint(ref, true)
      assert q_r == q
      assert p_r == p
      assert grad_r == grad

      {q_l, p_l, grad_l} = NativeTree.get_endpoint(ref, false)
      assert q_l == q
      assert p_l == p
      assert grad_l == grad
    end
  end

  describe "build_and_merge" do
    test "depth 0 subtree with single pre-computed state" do
      d = 2
      # Starting position
      q = [0.0, 0.0]
      p = [1.0, 1.0]
      grad = [0.5, 0.5]
      logp = -1.0

      ref = NativeTree.init_trajectory(q, p, grad, logp)
      inv_mass = [1.0, 1.0]

      # joint_logp_0 = logp - 0.5 * sum(p^2 * inv_mass) = -1.0 - 0.5*2 = -2.0
      joint_logp_0 = -2.0

      # Pre-computed state (one leapfrog step)
      all_q = [0.5, 0.6]
      all_p = [0.9, 0.8]
      all_logp = [-0.8]
      all_grad = [0.3, 0.4]

      result =
        NativeTree.build_and_merge(
          ref,
          all_q,
          all_p,
          all_logp,
          all_grad,
          inv_mass,
          joint_logp_0,
          0,
          d,
          true,
          12345
        )

      assert result == :ok

      final = NativeTree.get_result(ref)
      assert final.n_steps == 1
      assert final.depth == 1
      assert final.divergent == false
    end

    test "depth 1 subtree with two pre-computed states" do
      d = 1
      q = [0.0]
      p = [1.0]
      grad = [0.5]
      logp = -1.0

      ref = NativeTree.init_trajectory(q, p, grad, logp)
      inv_mass = [1.0]
      joint_logp_0 = -1.5

      # Two pre-computed states (for depth 1 = 2^1 = 2 leaves)
      all_q = [0.5, 1.0]
      all_p = [0.9, 0.8]
      all_logp = [-0.8, -1.2]
      all_grad = [0.3, 0.1]

      :ok =
        NativeTree.build_and_merge(
          ref,
          all_q,
          all_p,
          all_logp,
          all_grad,
          inv_mass,
          joint_logp_0,
          1,
          d,
          true,
          54321
        )

      final = NativeTree.get_result(ref)
      assert final.n_steps == 2
      assert final.depth == 1
    end

    test "divergent state detected" do
      d = 1
      q = [0.0]
      p = [1.0]
      grad = [0.5]
      logp = -1.0

      ref = NativeTree.init_trajectory(q, p, grad, logp)
      inv_mass = [1.0]
      joint_logp_0 = -1.5

      # State with very low logp => divergent (joint_logp << joint_logp_0 - 1000)
      all_q = [100.0]
      all_p = [0.1]
      all_logp = [-1.0e10]
      all_grad = [0.0]

      :ok =
        NativeTree.build_and_merge(
          ref,
          all_q,
          all_p,
          all_logp,
          all_grad,
          inv_mass,
          joint_logp_0,
          0,
          d,
          true,
          99999
        )

      final = NativeTree.get_result(ref)
      assert final.divergent == true
      assert NativeTree.is_terminated(ref)
    end
  end

  describe "build_full_tree_bin" do
    test "1D full tree produces valid result" do
      d = 1
      # Starting position: q=0, p=1, grad=0, logp=-0.5 (standard normal at 0)
      q0_bin = <<0.0::float-64-native>>
      p0_bin = <<1.0::float-64-native>>
      grad0_bin = <<0.0::float-64-native>>
      # = 0.0 for N(0,1) unnormalized
      logp0 = -0.5 * 0.0 * 0.0

      inv_mass_bin = <<1.0::float-64-native>>

      # joint_logp_0 = logp - 0.5 * p^2 * inv_mass = 0.0 - 0.5 * 1.0 = -0.5
      joint_logp_0 = -0.5

      # Forward chain: 7 states (enough for max_depth=3, budget=7)
      # Simulate simple harmonic oscillator: q(t) ≈ sin(t), p(t) ≈ cos(t)
      fwd_qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
      fwd_ps = [0.99, 0.98, 0.955, 0.92, 0.878, 0.825, 0.765]
      fwd_logps = [-0.005, -0.02, -0.045, -0.08, -0.125, -0.18, -0.245]
      fwd_grads = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7]

      # Backward chain: 7 states
      bwd_qs = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7]
      bwd_ps = [0.99, 0.98, 0.955, 0.92, 0.878, 0.825, 0.765]
      bwd_logps = [-0.005, -0.02, -0.045, -0.08, -0.125, -0.18, -0.245]
      bwd_grads = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

      to_bin = fn list ->
        list |> Enum.map(&<<&1::float-64-native>>) |> IO.iodata_to_binary()
      end

      result =
        NativeTree.build_full_tree_bin(
          q0_bin,
          p0_bin,
          grad0_bin,
          logp0,
          to_bin.(fwd_qs),
          to_bin.(fwd_ps),
          to_bin.(fwd_logps),
          to_bin.(fwd_grads),
          to_bin.(bwd_qs),
          to_bin.(bwd_ps),
          to_bin.(bwd_logps),
          to_bin.(bwd_grads),
          inv_mass_bin,
          joint_logp_0,
          3,
          d,
          12345
        )

      assert is_map(result)
      assert is_binary(result.q_bin)
      assert byte_size(result.q_bin) == d * 8
      assert is_float(result.logp)
      assert is_binary(result.grad_bin)
      assert byte_size(result.grad_bin) == d * 8
      assert result.n_steps > 0
      assert is_boolean(result.divergent)
      assert is_float(result.accept_sum)
      assert result.accept_sum > 0.0
      assert result.depth > 0
      assert result.depth <= 3
    end

    test "depth 0 early termination with divergent state" do
      d = 1
      q0_bin = <<0.0::float-64-native>>
      p0_bin = <<1.0::float-64-native>>
      grad0_bin = <<0.0::float-64-native>>
      logp0 = 0.0
      inv_mass_bin = <<1.0::float-64-native>>
      joint_logp_0 = -0.5

      # All states have extremely low logp => divergent
      to_bin = fn list -> list |> Enum.map(&<<&1::float-64-native>>) |> IO.iodata_to_binary() end

      fwd_logps = List.duplicate(-1.0e10, 7)
      bwd_logps = List.duplicate(-1.0e10, 7)
      fwd_qs = List.duplicate(100.0, 7)
      bwd_qs = List.duplicate(-100.0, 7)
      ps = List.duplicate(0.1, 7)
      grads = List.duplicate(0.0, 7)

      result =
        NativeTree.build_full_tree_bin(
          q0_bin,
          p0_bin,
          grad0_bin,
          logp0,
          to_bin.(fwd_qs),
          to_bin.(ps),
          to_bin.(fwd_logps),
          to_bin.(grads),
          to_bin.(bwd_qs),
          to_bin.(ps),
          to_bin.(bwd_logps),
          to_bin.(grads),
          inv_mass_bin,
          joint_logp_0,
          3,
          d,
          54321
        )

      assert result.divergent == true
      # Should terminate early — not use all 7 steps per direction
      assert result.n_steps <= 2
    end
  end

  describe "NIF vs Elixir equivalence" do
    test "simple normal model produces valid posterior with NIF" do
      alias Exmc.{Builder, Dist.Normal, NUTS.Sampler}

      ir =
        Builder.new_ir()
        |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(10.0)})
        |> Builder.rv("x", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
        |> Builder.obs("x_obs", "x", Nx.tensor([2.5, 3.0, 3.5]))

      # With NIF (default)
      {trace_nif, stats_nif} =
        Sampler.sample(ir, %{}, num_warmup: 200, num_samples: 300, seed: 42, ncp: false)

      mu_mean_nif = Nx.mean(trace_nif["mu"]) |> Nx.to_number()

      # Posterior mean should be near 3.0 (data mean with weak prior)
      assert abs(mu_mean_nif - 3.0) < 0.5, "NIF posterior mean #{mu_mean_nif} too far from 3.0"
      assert stats_nif.divergences < 20, "Too many divergences: #{stats_nif.divergences}"
    end

    test "NIF and Elixir paths produce similar posteriors" do
      alias Exmc.{Builder, Dist.Normal, NUTS.Sampler}

      ir =
        Builder.new_ir()
        |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(5.0)})
        |> Builder.rv("x", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
        |> Builder.obs("x_obs", "x", Nx.tensor([2.0, 2.5, 3.0]))

      # NIF path (default)
      {trace_nif, _} =
        Sampler.sample(ir, %{}, num_warmup: 300, num_samples: 500, seed: 42, ncp: false)

      mu_nif = Nx.mean(trace_nif["mu"]) |> Nx.to_number()

      # Elixir path (disable NIF)
      Application.put_env(:exmc, :use_nif, false)

      {trace_ex, _} =
        Sampler.sample(ir, %{}, num_warmup: 300, num_samples: 500, seed: 42, ncp: false)

      mu_ex = Nx.mean(trace_ex["mu"]) |> Nx.to_number()
      Application.put_env(:exmc, :use_nif, true)

      # Both should be near the true posterior mean (~2.5)
      # They won't be identical (different RNG) but should be statistically equivalent
      assert abs(mu_nif - 2.5) < 0.5, "NIF mean #{mu_nif} too far from 2.5"
      assert abs(mu_ex - 2.5) < 0.5, "Elixir mean #{mu_ex} too far from 2.5"
      assert abs(mu_nif - mu_ex) < 1.0, "NIF #{mu_nif} vs Elixir #{mu_ex} too different"
    end
  end
end
