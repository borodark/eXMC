defmodule Exmc.FaultTolerantTest do
  use ExUnit.Case

  import Exmc.TestHelper, only: [put_env_scoped: 2]

  @moduledoc false

  alias Exmc.{Builder, Rewrite}
  alias Exmc.Dist.Normal
  alias Exmc.NUTS.{FaultInjector, Leapfrog, Tree, Sampler}

  # =============================================
  # Helper: build a simple standard normal model
  # =============================================

  defp standard_normal_step_fn do
    ir =
      Builder.new_ir()
      |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
      |> Rewrite.apply()

    {vag_fn, step_fn, pm, _ncp_info, _multi_step_fn, _chain_meta} =
      Exmc.Compiler.compile_for_sampling(ir)

    {vag_fn, step_fn, pm}
  end

  defp standard_normal_ir do
    Builder.new_ir()
    |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
  end

  # =============================================
  # 1. FaultInjector basics
  # =============================================

  describe "FaultInjector" do
    test "activate and deactivate" do
      assert FaultInjector.activate(%{step_count: 5, error: :crash}) == :ok
      assert Process.get(:exmc_fault_inject) != nil
      assert FaultInjector.deactivate() == :ok
      assert Process.get(:exmc_fault_inject) == nil
    end

    test "raises at step_count" do
      FaultInjector.activate(%{step_count: 3, error: :crash})

      # First two calls should be fine
      assert FaultInjector.maybe_fault!(0) == :ok
      assert FaultInjector.maybe_fault!(0) == :ok

      # Third call triggers
      assert_raise RuntimeError, ~r/Injected crash/, fn ->
        FaultInjector.maybe_fault!(0)
      end

      FaultInjector.deactivate()
    end

    test "raises at specific depth" do
      FaultInjector.activate(%{depth: 2, error: :arithmetic})

      # Wrong depth — no-op
      assert FaultInjector.maybe_fault!(0) == :ok
      assert FaultInjector.maybe_fault!(1) == :ok

      # Matching depth triggers
      assert_raise ArithmeticError, fn ->
        FaultInjector.maybe_fault!(2)
      end

      FaultInjector.deactivate()
    end

    test "no-op when not activated" do
      assert FaultInjector.maybe_fault!(0) == :ok
      assert FaultInjector.maybe_fault!(5) == :ok
    end

    test "supports OOM error type" do
      FaultInjector.activate(%{step_count: 1, error: :oom})

      assert_raise ErlangError, fn ->
        FaultInjector.maybe_fault!(0)
      end

      FaultInjector.deactivate()
    end

    test "supports EXLA error type" do
      FaultInjector.activate(%{step_count: 1, error: :exla})

      assert_raise ErlangError, fn ->
        FaultInjector.maybe_fault!(0)
      end

      FaultInjector.deactivate()
    end
  end

  # =============================================
  # 2. Tree crash recovery (supervised: true)
  # =============================================

  describe "Tree crash recovery" do
    test "supervised=true: crash produces divergent result with valid structure" do
      {vag_fn, step_fn, _pm} = standard_normal_step_fn()
      q = Nx.tensor([0.0], type: :f64)
      {logp, grad} = vag_fn.(q)
      inv_mass = Nx.tensor([1.0], type: :f64)
      key = Nx.Random.key(42)
      {p, _key} = Leapfrog.sample_momentum(key, inv_mass)
      joint_logp_0 = Leapfrog.joint_logp(logp, p, inv_mass)
      rng = :rand.seed_s(:exsss, 42)

      # Inject crash at step 3 — some leaves will evaluate, then crash
      FaultInjector.activate(%{step_count: 3, error: :crash})
      Process.put(:exmc_supervised, true)

      result = Tree.build(step_fn, q, p, logp, grad, 0.1, inv_mass, 5, rng, joint_logp_0)

      Process.delete(:exmc_supervised)
      FaultInjector.deactivate()

      # Result must have valid structure
      assert is_map(result)
      assert Map.has_key?(result, :q)
      assert Map.has_key?(result, :logp)
      assert Map.has_key?(result, :grad)
      assert Map.has_key?(result, :n_steps)
      assert Map.has_key?(result, :divergent)
      assert result.n_steps >= 1

      # The crash was recovered, so either:
      # - the overall result is divergent due to the crashed subtree, OR
      # - a non-crashed subtree was selected as the proposal
      # Either way, the tree completed without raising
      assert is_boolean(result.divergent) or result.divergent == true
    end

    test "supervised=true with depth-based fault: recovered flag propagated" do
      {vag_fn, step_fn, _pm} = standard_normal_step_fn()
      q = Nx.tensor([0.0], type: :f64)
      {logp, grad} = vag_fn.(q)
      inv_mass = Nx.tensor([1.0], type: :f64)
      key = Nx.Random.key(7)
      {p, _key} = Leapfrog.sample_momentum(key, inv_mass)
      joint_logp_0 = Leapfrog.joint_logp(logp, p, inv_mass)
      rng = :rand.seed_s(:exsss, 7)

      # Crash at depth 2 — first doubling (depth 0,1) works, depth 2 crashes
      FaultInjector.activate(%{depth: 2, error: :oom})
      Process.put(:exmc_supervised, true)

      result = Tree.build(step_fn, q, p, logp, grad, 0.1, inv_mass, 5, rng, joint_logp_0)

      Process.delete(:exmc_supervised)
      FaultInjector.deactivate()

      # Recovery should be flagged
      assert result.recovered == true
    end
  end

  # =============================================
  # 3. Unsupervised crash propagates
  # =============================================

  describe "Unsupervised crash propagation" do
    test "crash propagates as exception without supervision" do
      {vag_fn, step_fn, _pm} = standard_normal_step_fn()
      q = Nx.tensor([0.0], type: :f64)
      {logp, grad} = vag_fn.(q)
      inv_mass = Nx.tensor([1.0], type: :f64)
      key = Nx.Random.key(42)
      {p, _key} = Leapfrog.sample_momentum(key, inv_mass)
      joint_logp_0 = Leapfrog.joint_logp(logp, p, inv_mass)
      rng = :rand.seed_s(:exsss, 42)

      # Inject crash but do NOT enable supervision
      FaultInjector.activate(%{step_count: 2, error: :crash})
      Process.delete(:exmc_supervised)

      assert_raise RuntimeError, ~r/Injected crash/, fn ->
        Tree.build(step_fn, q, p, logp, grad, 0.1, inv_mass, 5, rng, joint_logp_0)
      end

      FaultInjector.deactivate()
    end
  end

  # =============================================
  # 4. End-to-end: no-failure parity
  # =============================================

  describe "No-failure parity" do
    @tag timeout: 120_000
    test "supervised=true with no faults produces same trace as supervised=false" do
      ir = standard_normal_ir()

      # Disable full-tree NIF so both paths use same Erlang PRNG
      # (full-tree NIF uses Rust Xoshiro256** which produces different sequences)
      # Was `prev = Application.get_env(:exmc, :full_tree_nif, true)` followed by
      # a restore to `prev` — but the real default is `false` (tree.ex:85), so an
      # unset key came back set to `true` and flipped the Rust full-tree path on
      # for everything after. The restore was also skipped whenever an assertion
      # below raised first.
      put_env_scoped(:full_tree_nif, false)

      {trace_unsup, stats_unsup} =
        Sampler.sample(ir, %{}, num_warmup: 200, num_samples: 100, seed: 42, supervised: false)

      {trace_sup, stats_sup} =
        Sampler.sample(ir, %{}, num_warmup: 200, num_samples: 100, seed: 42, supervised: true)

      # Same number of samples
      assert Nx.shape(trace_unsup["x"]) == Nx.shape(trace_sup["x"])

      # Traces should be identical (same seed, same code path when no failures)
      unsup_vals = Nx.to_flat_list(trace_unsup["x"])
      sup_vals = Nx.to_flat_list(trace_sup["x"])

      diffs = Enum.zip(unsup_vals, sup_vals) |> Enum.map(fn {a, b} -> abs(a - b) end)
      max_diff = Enum.max(diffs)
      assert max_diff < 1.0e-10, "Traces diverged: max diff #{max_diff}"

      # Both should have zero recoveries
      assert Map.get(stats_unsup, :recoveries, 0) == 0
      assert Map.get(stats_sup, :recoveries, 0) == 0
    end
  end

  # =============================================
  # 5. End-to-end recovery: posterior still valid
  # =============================================

  describe "End-to-end recovery" do
    @tag timeout: 120_000
    test "sampling completes with injected crash, posterior reasonable" do
      ir = standard_normal_ir()

      # Depth 1, not depth 3. At depth 3 this test was VACUOUS on the host
      # path: measured 2026-08-17, `compiler: :none` consulted the injector
      # 1463 times and hit `divergent_placeholder` **zero** times, because a
      # well-adapted host sampler on N(0,1) never builds a subtree that deep.
      # It passed for two backends without recovery ever running. Depth 1
      # fires on every backend — 357 recoveries on `:none`, 29 on `:vulkan`.
      #
      # The `recoveries > 0` assertion below is what keeps it honest: if a
      # future change stops the injector reaching this path, the test fails
      # instead of quietly passing.
      FaultInjector.activate(%{depth: 1, error: :crash})

      {trace, stats} =
        Sampler.sample(ir, %{}, num_warmup: 200, num_samples: 300, seed: 42, supervised: true)

      FaultInjector.deactivate()

      # Sampling completed
      assert Nx.shape(trace["x"]) == {300}

      # NON-VACUITY: recovery must actually have run.
      assert Map.get(stats, :recoveries, 0) > 0,
             "no crash was recovered — the injector never reached a supervised subtree, " <>
               "so this test proves nothing about recovery"

      # Posterior should still be vaguely reasonable (N(0,1) target)
      vals = Nx.to_flat_list(trace["x"])
      mean = Enum.sum(vals) / length(vals)
      var = Enum.sum(Enum.map(vals, fn v -> (v - mean) * (v - mean) end)) / length(vals)

      # Generous tolerance — crashes may degrade quality
      assert abs(mean) < 1.5, "Mean too far from 0: #{mean}"
      assert var > 0.1, "Variance collapsed: #{var}"
      assert var < 5.0, "Variance exploded: #{var}"

      # Stats should report divergences/recoveries
      assert is_number(stats.divergences)
    end

    # Regression guard for the crash/step-size feedback loop.
    #
    # A crashed subtree reports acceptance 0.0. Fed to dual averaging that
    # reads as "eps is catastrophically too large", so eps shrinks, so
    # trajectories need more steps to reach a U-turn, so trees get deeper, so
    # they hit the failing region more often — a loop. Under `compiler:
    # :vulkan` it drove eps from 1.053 to 2.41e-11 and collapsed the posterior
    # to variance 1.45e-15 while reporting success.
    #
    # The variance assertion above would catch a full collapse. This one
    # catches the mechanism directly, and catches it earlier: eps is the thing
    # that moves eleven orders of magnitude.
    @tag timeout: 120_000
    test "crash recovery does not destroy the adapted step size" do
      ir = standard_normal_ir()

      FaultInjector.activate(%{depth: 1, error: :crash})

      {_trace, stats} =
        Sampler.sample(ir, %{}, num_warmup: 200, num_samples: 300, seed: 42, supervised: true)

      FaultInjector.deactivate()

      assert Map.get(stats, :recoveries, 0) > 0, "vacuous: nothing was recovered"

      eps =
        case stats.step_size do
          e when is_number(e) -> e
          t -> Nx.to_number(t)
        end

      # Uninjected, this model adapts to eps ~1.0 on every backend. Anything
      # below 0.01 means the crashes drove the adaptation, not the geometry.
      assert eps > 0.01,
             "step size collapsed to #{eps} under crash recovery — crashed subtrees are " <>
               "being fed to dual averaging as if they measured something"

      assert eps < 100.0, "step size exploded to #{eps}"

      # Placeholders are not integrator divergences. Before the fix this
      # reported 178 divergences for 176 placeholders.
      assert stats.divergences < Map.get(stats, :recoveries, 0),
             "divergences (#{stats.divergences}) not clearly below recoveries " <>
               "(#{Map.get(stats, :recoveries, 0)}) — crash placeholders are probably " <>
               "still being counted as divergences"
    end
  end

  # =============================================
  # 6. Recovery stats tracked
  # =============================================

  describe "Recovery stats" do
    @tag timeout: 120_000
    test "recoveries counter incremented on crash recovery" do
      # Test at the Tree level directly to avoid NIF/cached path bypassing FaultInjector.
      # Build multiple trees with supervision and step_count injection.
      {vag_fn, step_fn, _pm} = standard_normal_step_fn()
      q = Nx.tensor([0.0], type: :f64)
      {logp, grad} = vag_fn.(q)
      inv_mass = Nx.tensor([1.0], type: :f64)
      key = Nx.Random.key(42)
      {p, _key} = Leapfrog.sample_momentum(key, inv_mass)
      joint_logp_0 = Leapfrog.joint_logp(logp, p, inv_mass)

      Process.put(:exmc_supervised, true)

      # Run several trees, injecting crash at step 2 each time
      recoveries =
        Enum.reduce(1..10, 0, fn i, acc ->
          rng = :rand.seed_s(:exsss, i)
          FaultInjector.activate(%{step_count: 2, error: :crash})

          result = Tree.build(step_fn, q, p, logp, grad, 0.1, inv_mass, 5, rng, joint_logp_0)

          FaultInjector.deactivate()

          if Map.get(result, :recovered, false), do: acc + 1, else: acc
        end)

      Process.delete(:exmc_supervised)

      assert recoveries >= 1,
             "Expected at least 1 recovery out of 10 trees, got #{recoveries}"
    end

    @tag timeout: 120_000
    test "sample_stats include recovered field" do
      ir = standard_normal_ir()

      # Run without fault injection — just verify the field exists
      {_trace, stats} =
        Sampler.sample(ir, %{}, num_warmup: 100, num_samples: 50, seed: 42, supervised: true)

      # sample_stats should exist and have recovered fields
      assert is_list(stats.sample_stats)
      assert length(stats.sample_stats) == 50

      Enum.each(stats.sample_stats, fn ss ->
        assert Map.has_key?(ss, :recovered)
      end)

      # Without faults, all should be false
      assert Enum.all?(stats.sample_stats, fn ss -> ss.recovered == false end)
      assert stats.recoveries == 0
    end
  end

  # =============================================
  # 7. Task supervision timeout
  # =============================================

  describe "Task supervision" do
    test "timeout produces divergent placeholder" do
      {vag_fn, step_fn, _pm} = standard_normal_step_fn()
      q = Nx.tensor([0.0], type: :f64)
      {logp, grad} = vag_fn.(q)
      inv_mass = Nx.tensor([1.0], type: :f64)
      key = Nx.Random.key(42)
      {p, _key} = Leapfrog.sample_momentum(key, inv_mass)
      joint_logp_0 = Leapfrog.joint_logp(logp, p, inv_mass)
      rng = :rand.seed_s(:exsss, 42)

      # Use task supervision with very short timeout
      Process.put(:exmc_supervised, :task)
      # 1ms — will timeout
      Process.put(:exmc_supervised_timeout, 1)

      # Inject a sleep into the step_fn to guarantee timeout
      slow_step_fn = fn q, p, grad, epsilon, inv_mass_diag ->
        Process.sleep(100)
        step_fn.(q, p, grad, epsilon, inv_mass_diag)
      end

      result = Tree.build(slow_step_fn, q, p, logp, grad, 0.1, inv_mass, 3, rng, joint_logp_0)

      Process.delete(:exmc_supervised)
      Process.delete(:exmc_supervised_timeout)

      # Should complete (not hang or crash)
      assert is_map(result)
      assert Map.has_key?(result, :q)
      # Timeout causes divergent placeholder
      assert result.recovered == true
    end
  end

  # =============================================
  # 8. Overhead benchmark
  # =============================================

  describe "Overhead" do
    @tag :benchmark
    @tag timeout: 300_000
    test "supervised=true overhead < 10% with no faults" do
      ir = standard_normal_ir()

      # Warmup JIT cache
      Sampler.sample(ir, %{}, num_warmup: 50, num_samples: 50, seed: 0, supervised: false)

      # Unsupervised timing
      {time_unsup, _} =
        :timer.tc(fn ->
          Sampler.sample(ir, %{}, num_warmup: 200, num_samples: 500, seed: 42, supervised: false)
        end)

      # Supervised timing
      {time_sup, _} =
        :timer.tc(fn ->
          Sampler.sample(ir, %{}, num_warmup: 200, num_samples: 500, seed: 42, supervised: true)
        end)

      overhead = (time_sup - time_unsup) / time_unsup * 100

      IO.puts("\n  Unsupervised: #{div(time_unsup, 1000)}ms")
      IO.puts("  Supervised:   #{div(time_sup, 1000)}ms")
      IO.puts("  Overhead:     #{Float.round(overhead, 1)}%")

      # Allow 10% overhead (try/rescue is free on BEAM when no exception)
      assert overhead < 10.0,
             "Supervised overhead #{Float.round(overhead, 1)}% exceeds 10% threshold"
    end
  end
end
