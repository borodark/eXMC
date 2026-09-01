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
    # What supervision costs, measured in BEAM reductions rather than wall clock.
    #
    # The old form of this test timed one unsupervised run against one
    # supervised run and asserted `(t_sup - t_unsup) / t_unsup < 10%`. That
    # assertion is on a ratio whose denominator is the useful work, and the
    # useful work is the part that hardware makes faster. Supervision cost is
    # not: `with_supervision/6` is a try/rescue plus a process-dictionary read
    # per subtree build, a fixed BEAM cost per operation. Speed the box up and
    # the denominator shrinks faster than the numerator, so the ratio rises —
    # the test got HARDER to pass on better hardware. It failed on a Jetson
    # Nano at 10.7% only after that box moved from nvpmodel 5W (2 cores, 918
    # MHz) to MAXN (4 cores, 1479 MHz). It passed on the slow configuration.
    #
    # The threshold was also below the measurement noise, which means the old
    # test never measured what it claimed on any host. Seven alternating pairs
    # of the exact runs it timed, on one idle-ish 88-core box, 2026-09-01:
    #
    #     pairwise overhead: -5.7%, -1.5%, +19.3%, -11.8%, +21.6%, +3.4%, +0.3%
    #
    # A 10% gate against a ±20% instrument is a coin flip. Its passing
    # everywhere else was not evidence that supervision is cheap.
    #
    # Reductions are the BEAM's own count of work done. They do not change when
    # the host gets faster, which removes the defect above, and the same seeded
    # run does the same work on every machine. Same seven pairs:
    #
    #     reduction overhead: 0.08%, 0.30%, 0.32%, 0.82%, 0.27%, 0.32%, 0.18%
    #
    # — an instrument two orders of magnitude tighter than the wall clock, on
    # the same runs.
    #
    # The `:task` arm is a control, and it is what makes the threshold a
    # measurement instead of a guess. `supervised: :task` is a real supervision
    # implementation in this codebase (tree.ex:1273) that spawns a process per
    # subtree — precisely the regression `supervised: true` must not become. It
    # is measured in the same run, so the gate can be expressed as a fraction of
    # it rather than as an absolute percentage:
    #
    #     (reductions added by `true`) / (reductions added by `:task`) < 1/3
    #
    # Both quantities are per-subtree costs, so the total-work denominator
    # cancels and the gate does not move with the backend. It has to not move:
    # under `EXMC_COMPILER=none` the same run costs 168M reductions instead of
    # 8.5M, and `:task` shows up as +1.1% rather than +10%. A fixed 5% gate
    # would have been unfalsifiable there — it could not even resolve a
    # process-per-subtree regression — while this ratio reads 0.03 on EXLA and
    # -0.1 on `:none`, i.e. two orders of magnitude clear of the gate on both.
    #
    # Four things are asserted, in descending order of strength:
    #
    #   1. Supervision changes no work at all: the per-sample `n_steps` lists
    #      are equal element-for-element across all three modes. Exact, integer,
    #      host-independent, and it fails naming the cause. (The trace equality
    #      that goes with it is the "No-failure parity" test above.)
    #   2. Non-vacuity: the `:task` control must cost measurably more than
    #      nothing, or the ratio below is a quotient of two noise figures.
    #   3. The ratio gate above, plus a coarse absolute gate at 5% of total
    #      reductions for the case where the control is somehow inflated.
    #   4. A wall-clock backstop at 2x, for the one regression class reductions
    #      cannot see: one that blocks rather than computes (a sleep, a remote
    #      call, lock contention). 2x is ten times the worst noise observed
    #      above; it will not catch a small blocking regression, and nothing
    #      measurable on a shared host would.
    #
    # Sensitivity, measured 2026-09-01 on EXLA by loading a patched copy of Tree
    # into the VM with extra work injected into the `true ->` branch only. This
    # run makes roughly 1400 supervised subtree builds, so:
    #
    #   +290 reductions/build  → +4.9% reductions, ratio 0.48 → FAILS
    #   +190 reductions/build  → +4.3% reductions, ratio 0.44 → FAILS
    #    +60 reductions/build  → +1.5% reductions, ratio 0.14 → passes
    #   Process.sleep(1)/build → +1.4% to +4.9% reductions (gate 3 is borderline
    #                            on it) and +268% to +418% wall → FAILS on gate 4
    #
    # So the floor is somewhere near +100 reductions per subtree build, about
    # 3% of the run: a regression that makes supervision cost a third of what
    # spawning a process per subtree costs. `supervised: true` today sits at
    # 0.0-0.5% of the run and ratio 0.00-0.07 — five to a hundred times inside
    # the gate. What this test can no longer do is resolve a 10% change in
    # *wall clock*; nothing on a shared host could, which is the whole point.
    @tag :benchmark
    @tag timeout: 300_000
    test "supervision adds no work: identical steps, a fraction of :task's cost" do
      ir = standard_normal_ir()

      # With the full-tree NIF on, the unsupervised arm runs the Rust whole-tree
      # path and the supervised arm cannot (tree.ex:88 disables it under
      # supervision), so the comparison would measure NIF-vs-Elixir and not
      # supervision. The real default is false; pin it so this does not depend
      # on that staying true.
      put_env_scoped(:full_tree_nif, false)

      opts = [num_warmup: 200, num_samples: 500, seed: 42]

      run = fn mode ->
        :erlang.statistics(:reductions)

        {micros, {_trace, stats}} =
          :timer.tc(fn -> Sampler.sample(ir, %{}, [supervised: mode] ++ opts) end)

        {_total, reductions} = :erlang.statistics(:reductions)

        %{
          micros: micros,
          reductions: reductions,
          steps: Enum.map(stats.sample_stats, & &1.n_steps),
          recoveries: Map.get(stats, :recoveries, 0)
        }
      end

      # Warm the JIT/executable caches. Discarded.
      Sampler.sample(ir, %{}, num_warmup: 50, num_samples: 50, seed: 0, supervised: false)

      # Three rounds, each round running the arms adjacently. The noise on this
      # measurement drifts on a scale of seconds, so adjacent pairs cancel most
      # of it; the median of the three then discards a single bad round.
      rounds =
        for _ <- 1..3 do
          %{off: run.(false), sup: run.(true), task: run.(:task)}
        end

      median = fn xs -> xs |> Enum.sort() |> Enum.at(1) end
      ratio = fn rounds, arm, field -> Enum.map(rounds, &(&1[arm][field] / &1.off[field])) end

      red_sup = median.(ratio.(rounds, :sup, :reductions))
      red_task = median.(ratio.(rounds, :task, :reductions))
      wall_sup = median.(ratio.(rounds, :sup, :micros))

      pct = fn r -> Float.round((r - 1.0) * 100, 2) end

      # Supervision's cost as a fraction of the process-per-subtree control's.
      # Both are per-subtree costs over the same denominator, which therefore
      # cancels — see the header.
      cost_vs_task = (red_sup - 1.0) / (red_task - 1.0)

      IO.puts("\n  reductions: unsupervised #{hd(rounds).off.reductions}")
      IO.puts("  supervised=true  #{pct.(red_sup)}% reductions, #{pct.(wall_sup)}% wall")
      IO.puts("  supervised=:task #{pct.(red_task)}% reductions (control)")
      IO.puts("  true costs #{Float.round(cost_vs_task, 3)}x what :task costs (gate: 0.333)")

      # 1. No mode changed the work. Exact integer equality, per sample.
      first = hd(rounds)

      assert first.sup.steps == first.off.steps,
             "supervised=true did not take the same trajectory as unsupervised: " <>
               "#{length(Enum.reject(Enum.zip(first.sup.steps, first.off.steps), fn {a, b} -> a == b end))} " <>
               "of #{length(first.off.steps)} samples used a different number of leapfrog steps"

      assert first.task.steps == first.off.steps,
             "supervised=:task did not take the same trajectory as unsupervised"

      assert first.sup.recoveries == 0 and first.task.recoveries == 0,
             "no fault was injected, so nothing should have been recovered"

      assert Enum.sum(first.off.steps) > 0, "vacuous: the sampler took no leapfrog steps"

      # 2. Non-vacuity: the control must cost something the counter can resolve,
      #    or the ratio below is one noise figure divided by another. 0.5% of
      #    total reductions is comfortably above the pairwise noise measured on
      #    both backends (worst case 0.82%, typically 0.3%) — but note that if
      #    the run-to-run noise on some future host exceeds this, the ratio gate
      #    goes soft rather than wrong, and this assertion is what will say so.
      assert red_task > 1.005,
             "the process-per-subtree supervision path (:task) added only " <>
               "#{pct.(red_task)}% reductions — the control is not resolvable on " <>
               "this host, so the ratio below is noise over noise"

      # 3. The claim. Ratio first (backend-independent), absolute second.
      assert cost_vs_task < 1 / 3,
             "supervised=true costs #{Float.round(cost_vs_task, 3)}x what " <>
               "supervised=:task costs (gate: 0.333). true is a try/rescue and a " <>
               "process-dictionary read per subtree; :task spawns a process per " <>
               "subtree. In reductions: #{pct.(red_sup)}% against #{pct.(red_task)}%."

      assert red_sup < 1.05,
             "supervised=true costs #{pct.(red_sup)}% extra BEAM reductions " <>
               "(absolute gate: 5%; :task, one process per subtree, costs " <>
               "#{pct.(red_task)}%)"

      # 4. Blocking backstop. Coarse on purpose — see the header.
      assert wall_sup < 2.0,
             "supervised=true took #{pct.(wall_sup)}% longer in wall clock while adding " <>
               "only #{pct.(red_sup)}% reductions — supervision is blocking, not computing"
    end
  end
end
