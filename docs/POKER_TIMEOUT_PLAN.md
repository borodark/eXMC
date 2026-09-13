# Poker integration test timeout: investigation plan

**Written 2026-09-13 (night), at exmc `eedb9a3ec` (nx_vulkan lock `f450e0c`).**
This is a handover. Nothing below has been investigated beyond the facts in
*What is known*. It is for a session that did not see the work that found it.

## The test

`test/poker_test.exs:324`, `Exmc.PokerTest` "integration parameter recovery on
synthetic data":
- **Setup.** Tagged `:poker_integration`, `@tag timeout: 300_000`. It simulates
  100 hands for one player (`Exmc.Poker.Simulator`) and builds
  `Exmc.Poker.OpponentModel`: Normal and HalfCauchy priors plus a grouped
  `Dist.Custom` likelihood (`lib/exmc/poker/opponent_model.ex:79-150`).
- **Sampling.** `Exmc.Sampler.sample/3` with 300 warmup, 200 draws, seed 42,
  `ncp: false`.
- **Assertions.** Posterior means within generous deltas of the true vpip, pfr,
  agg and bluff.

## What is known (measured)

**On super-io, one test alone, at `eedb9a3ec`:**

| arm | how | wall |
|---|---|---|
| EXLA (host client) | `mix test test/poker_test.exs:324 --include poker_integration --slowest 1` | **6.8 s** |
| Vulkan (RTX 3060 Ti) | the same with `EXMC_COMPILER=vulkan` | **232.7 s** |

The Vulkan arm is **34x slower** and uses 78% of the 300 s budget on the
fastest GPU host in the fleet. `scripts/fleet_verify.sh` runs the suite with
`EXMC_COMPILER=vulkan`, so every fleet host runs the slow path.

**Fleet results for this test:**

| host | GPU | Rustler 0.38 run (`85f289306`..`138eb518a`, before the going_right fix) | `fb7733ea0` (after the fix and the nx_vulkan bump) |
|---|---|---|---|
| super-io | RTX 3060 Ti | pass | pass (Vulkan suite 741/0) |
| NUC | HD 520 (iGPU, Mesa) | pass | pass (741/0) |
| mac-248 | GT 750M | pass | pass (741/0) |
| mac-247 | GT 650M | **timeout** (its only failure, 740/1) | **timeout** (its only failure, 741/1) |
| asus | GTX 1660 Ti | **timeout** (among 11 failures during the start-up hang) | **timeout** (its only failure, 741/1, suite 1822 s) |
| Jetson | Tegra X1 | 740/3, two timeouts "expected"; **which tests was not recorded** | **timeout**, plus `IntegrationTest` "large model: 5-parameter hierarchical" at 120 s (741/2) |

So:
- **Not new.** It timed out on mac-247 and asus before `509e22b26` (the
  going_right fix).
- **Not the asus start-up hang.** That hang is fixed at nx_vulkan `bd17793`,
  and asus still times out.
- **Not ordered by GPU speed.** The NUC's iGPU and the GT 750M pass while the
  GT 650M and the GTX 1660 Ti fail. Something other than GPU throughput decides
  it: CPU single-thread speed, driver, host load, or which code path runs.

## What is NOT known

1. **Where the 232 s goes on Vulkan.** Chain-shader dispatches, the per-op
   `Nx.Defn.Evaluator` path, host fallback, warmup or sampling.
2. **Whether the model synthesises a chain shader at all.** It has a
   `Dist.Custom` likelihood. `LevelSetIntegrationTest` (docs/VULKAN_KNOWN_ISSUES.md
   #4) is the precedent for a model that cannot reach the fused shader and runs
   op by op.
3. **Whether the going_right fix changed the duration.** Longer, correct trees
   cost more gradient evaluations; SV's reference run went from 881 s to 1305 s.
   The fix did not cause the timeout, but it may have narrowed the margin.
4. **Per-host durations.** fleet_verify logs record no per-test wall time, so
   how far each passing host is from 300 s is unknown.
5. **Which Jetson timeouts were "expected"** in the Rustler 0.38 run.

## Hypotheses, to be tested, not assumed

- **H1: warmup is per-op on Vulkan.** `Exmc.NUTS.Tree.warmup_downgrade?/0`
  (`lib/exmc/nuts/tree.ex:729`) sends every Vulkan warmup subtree down the
  plain per-leaf `step_fn` path, bypassing the fused, cached and NIF paths (the
  D91 Option C comment at `dispatch_subtree`). 300 warmup iterations of
  per-leaf evaluation through the Evaluator could be most of the 232 s. **Test:**
  time warmup and sampling separately (e.g. `num_warmup: 300, num_samples: 0`
  against `num_warmup: 0` with a fixed tuning via `sample_compiled_tuned`),
  on Vulkan and EXLA.
- **H2: no chain shader.** The Custom likelihood fails synthesis and the whole
  run is per-op. **Test:** inspect `ChainShaderCodegen.detect_meta` /
  `try_synthesise` for this IR (a `{:unsupported, reason}` result, or the
  logged "synthesis raised" warning), and count `chain_dispatches` as
  `scripts/vulkan_smoke.exs` does.
- **H3: host CPU speed decides it.** If H1 or H2 holds, the cost is BEAM-side
  interpretation, which tracks single-thread CPU speed and not the GPU. **Test:**
  per-host single-test wall times (step 1 below) against CPU model.
- **H4: tree length after `509e22b26`.** **Test:** super-io Vulkan wall time and
  mean `n_steps` at `05944d18a` against `eedb9a3ec`, from a worktree. It needs
  its own `_build`, `deps` and NIF build.

## Steps

1. **Measure before changing anything.** On every host run
   `mix test test/poker_test.exs:324 --include poker_integration --slowest 1`
   (Vulkan arm, device pinned as fleet_verify pins it), and also
   `test/integration_test.exs:468` on the Jetson. Record wall time, CPU model
   and host load. Run each host only when idle (see *Fleet etiquette*).
2. **Split the 232 s on super-io**: warmup against sampling; chain-shader
   dispatches against Evaluator ops (H1, H2). An `:eprof` or `:fprof` profile of
   one warmup iteration names the hot path.
3. **H4** only if 2 leaves the tree length as a candidate.
4. **Decide, with the numbers.** In order of preference:
   - **(a) Fix the slow path** if it is a defect: a missed synthesis, or a
     downgrade that is broader than it needs to be.
   - **(b) Make the test cheaper without weakening what it checks.** It checks
     parameter recovery, which needs a correct posterior, not 300 warmup
     iterations on a slow path.
   - **(c) Size the timeout for the slowest host**, as
     `test/reproducibility_contract_test.exs` did (a comment with the host,
     the date and the measured time).
   - **Not acceptable:** an exclusion tag without the mechanism written down.
     This codebase treats "a test that passes by not running" with suspicion;
     `docs/VULKAN_KNOWN_ISSUES.md` #4 is the format if an exclusion turns out to
     be the honest answer.
5. **Verify** with `scripts/fleet_verify.sh` on every host: counts and failure
   blocks, not the count alone.

## Fleet etiquette (read before touching a host)

**Hosts:**
- super-io is local.
- NUC `io@192.168.0.245` and mac-247 `io@192.168.0.247`, FreeBSD, `~/exmc_oss`.
- mac-248 `io@192.168.0.248`, FreeBSD, `~/exmc_oss`.
- Jetson `io@192.168.0.250`, aarch64. **Never build its Rust NIF on the board.**
  Cross-build on super-io with nx_vulkan's `scripts/deploy_jetson_nif.sh`
  (`REF=<lock sha> DEST_DIR='$HOME/exmc_oss/deps/nx_vulkan'`).
- asus `io@192.168.0.246`, FreeBSD, **checkout `~/exmc_race`** (run with
  `EXMC_DIR=$HOME/exmc_race`). `~/exmc_oss` there belongs to other work.

**Sharing:**
- asus is shared with the nx_vulkan session and ex-pathmc-39 (`~/ex_pathmc`
  runs sampling tests there). Run only when no `beam.smp` is running, checked
  with `pgrep -x beam.smp`, never `pgrep -f` (it matches its own ssh command
  line).
- **Never run `nvidia-smi -pm 1` on asus**: it is host-fatal.
- The Macs run long-lived `zedweb` BEAM services. Those `beam.smp` processes are
  not tests; identify a process before waiting on it.
- The PyMC race (docs/PYMC_RACE_PLAN.md) needs super-io and asus idle for its
  runs. Coordinate before a long profile on either.

**Git:** push `main` to `origin` only, never to `upstream`. Other sessions edit
sibling repos, so stage by hunk.
