# Crash recovery under supervision — before and after

`bench/crash_recovery.exs`. Prior-only `N(0,1)`, 200 warmup + 300 samples, seed
42, `supervised: true`, crashes injected with `Exmc.NUTS.FaultInjector` at a
fixed subtree depth. Analytic truth: mean 0, variance 1.

    COMPILER=none   INJECT=0        mix run --no-deps-check bench/crash_recovery.exs   # control
    COMPILER=none   DEPTH=1         mix run --no-deps-check bench/crash_recovery.exs
    COMPILER=vulkan DEPTH=3         mix run --no-deps-check bench/crash_recovery.exs

Host: super-io — Intel Xeon E5-2699 v4 (88 threads), NVIDIA GeForce RTX 3060 Ti
(DiscreteGpu), Linux 6.8.0-137-generic x86_64, Erlang/OTP 27 (erts 15.2.7.2),
Elixir 1.18.3. Measured 2026-08-17/18 against `82db4f8`.

"before" is that tree with `lib/exmc/nuts/{tree,sampler}.ex` stashed; nothing
else differs between the two columns.

## Summary

| arm | injection | before | after |
|---|---|---|---|
| `:none` | none (control) | var 1.0212, eps 0.9281 | **bit-identical** |
| `:none` | depth 1 | **run dies** — `(RuntimeError) Injected crash`, unhandled | var 0.8768, eps 2.0408, 357 recoveries |
| `:vulkan` | none (control) | var 0.9551, eps 1.0534 | **bit-identical** |
| `:vulkan` | depth 1 | **run dies** — unhandled, in the sampling phase | var 0.8768, eps 2.0408, 357 recoveries |
| `:vulkan` | depth 3 | var **1.45e-15**, eps **2.41e-11**, 176 placeholders, **0 recoveries reported**, 178 "divergences" | var 0.8807, eps 0.8102, 29 placeholders, 29 recoveries, 3 divergences |

Three distinct defects show up in that table.

**1. The step-size feedback loop.** `:vulkan` / depth 3, the only row where the
old code survived to produce output: eps 2.41e-11 against 1.0534 uninjected,
eleven orders of magnitude, and a posterior collapsed to variance 1.45e-15 —
300 distinct draws inside a neighbourhood of ~4e-8. **The run reported
success.** A crashed subtree contributed acceptance 0.0, dual averaging read
that as "eps far too large", smaller eps meant longer trajectories to a U-turn,
deeper trees hit the injected depth more often, and round it went: 3405
injector consultations against 2018 uninjected. Crash-recovered iterations are
now excluded from the dual-averaging update entirely.

**2. Supervision did nothing wherever the speculative buffer was live.** The
two "run dies" rows are `supervised: true` runs that crashed anyway.
`safe_build_subtree` guarded only the non-speculative branch of `do_build/11`,
and `speculative_precompute` defaults to `true` — so on the control run,
`:none` built 861 subtrees, all speculative, and entered the supervision
wrapper **zero** times. The guard now wraps whichever dispatch runs. That is
also why the depth-3 row is the *only* one that produced numbers: under Vulkan
warmup the downgrade path disables speculation, so supervision happened to be
reachable there and nowhere else.

**3. Warmup recoveries were invisible.** The depth-3 row built 176 placeholders
and reported `recoveries: 0`. `nuts_step_warmup` never read the `:recovered`
flag — only `nuts_step_with_stats` did, and that runs in the sampling phase,
where the Vulkan hot path never consults the injector. Every crash happened
during warmup, so the one counter that would have told an operator anything was
structurally zero. `placeholders` and `recoveries` now agree on every row.

Related, and visible in the same row: 176 placeholders were reported as **178
divergences**. Nearly every "divergence" was a placeholder marked
`divergent: true` to stop the doubling, not an integrator divergence. Those are
now counted as recoveries instead; the depth-3 row reports 3 divergences.

## What recovery still costs

It is not free, and the numbers should not be read as if it were. At depth 1
— where 357 of 500 iterations crash — eps adapts to 2.04 against 0.93
uninjected, the posterior variance lands at 0.877 against 1.0, and only 245 of
300 draws are distinct. That is a degraded chain. It is a *chain*, which is the
point: the alternative in that row was a dead process or a variance of 1e-15
reported as success.

---

## Raw output — after

```
=== crash recovery — none, no injection ===
warmup/samples    : 200/300   seed 42   supervised: true
  injector calls  : 1463
  placeholders    : 0          <- 0 means recovery never ran
  with_supervision: 861
  spec dispatch   : 861
  plain dispatch  : 0
  elixir subtree  : 1463
  recoveries      : 0   (reported in stats)
  divergences     : 6
  adapted eps     : 0.9280998272940233
  mean / var      : -0.0707 / 1.0211975361484702    (truth 0.0 / 1.0)
  distinct draws  : 290/300
=== crash recovery — none, crash at depth 1 ===
warmup/samples    : 200/300   seed 42   supervised: true
  injector calls  : 857
  placeholders    : 357          <- 0 means recovery never ran
  with_supervision: 857
  spec dispatch   : 857
  plain dispatch  : 0
  elixir subtree  : 857
  recoveries      : 357   (reported in stats)
  divergences     : 2
  adapted eps     : 2.040818026778171
  mean / var      : 0.0843 / 0.8768445641129604    (truth 0.0 / 1.0)
  distinct draws  : 245/300
=== crash recovery — vulkan, no injection ===
warmup/samples    : 200/300   seed 42   supervised: true
  injector calls  : 2018
  placeholders    : 0          <- 0 means recovery never ran
  with_supervision: 904
  spec dispatch   : 520
  plain dispatch  : 384
  elixir subtree  : 2018
  recoveries      : 0   (reported in stats)
  divergences     : 7
  adapted eps     : 1.0534214769334085
  mean / var      : -0.0844 / 0.9550596983027236    (truth 0.0 / 1.0)
  distinct draws  : 290/300
=== crash recovery — vulkan, crash at depth 1 ===
warmup/samples    : 200/300   seed 42   supervised: true
  injector calls  : 857
  placeholders    : 357          <- 0 means recovery never ran
  with_supervision: 857
  spec dispatch   : 485
  plain dispatch  : 372
  elixir subtree  : 857
  recoveries      : 357   (reported in stats)
  divergences     : 2
  adapted eps     : 2.040818026778171
  mean / var      : 0.0843 / 0.8768445641129604    (truth 0.0 / 1.0)
  distinct draws  : 245/300
=== crash recovery — vulkan, crash at depth 3 ===
warmup/samples    : 200/300   seed 42   supervised: true
  injector calls  : 1973
  placeholders    : 29          <- 0 means recovery never ran
  with_supervision: 1006
  spec dispatch   : 595
  plain dispatch  : 411
  elixir subtree  : 1973
  recoveries      : 29   (reported in stats)
  divergences     : 3
  adapted eps     : 0.8102187165959089
  mean / var      : 0.0373 / 0.8806843661792749    (truth 0.0 / 1.0)
  distinct draws  : 298/300
```

## Raw output — before (lib stashed)

```
### COMPILER=none INJECT=0 DEPTH=1
=== crash recovery — none, no injection ===
warmup/samples    : 200/300   seed 42   supervised: true
  injector calls  : 1463
  placeholders    : 0          <- 0 means recovery never ran
  with_supervision: 0
  spec dispatch   : 861
  plain dispatch  : 0
  elixir subtree  : 1463
  recoveries      : 0   (reported in stats)
  divergences     : 6
  adapted eps     : 0.9280998272940233
  mean / var      : -0.0707 / 1.0211975361484702    (truth 0.0 / 1.0)
  distinct draws  : 290/300
### COMPILER=none INJECT=1 DEPTH=1
** (RuntimeError) Injected crash for fault tolerance testing
    (exmc 0.3.1) lib/exmc/nuts/fault_injector.ex:62: Exmc.NUTS.FaultInjector.raise_error/1
    (exmc 0.3.1) lib/exmc/nuts/tree.ex:1501: Exmc.NUTS.Tree.build_subtree/10
    (exmc 0.3.1) lib/exmc/nuts/tree.ex:430: Exmc.NUTS.Tree.do_build/11
    (exmc 0.3.1) lib/exmc/nuts/tree.ex:128: Exmc.NUTS.Tree.build/12
    (exmc 0.3.1) lib/exmc/nuts/sampler.ex:885: Exmc.NUTS.Sampler.nuts_step_warmup/8
    (exmc 0.3.1) lib/exmc/nuts/sampler.ex:721: anonymous fn/8 in Exmc.NUTS.Sampler.run_phase/10
    (elixir 1.18.3) lib/enum.ex:4507: Enum.reduce_range/5
    (exmc 0.3.1) lib/exmc/nuts/sampler.ex:636: Exmc.NUTS.Sampler.run_warmup/10
### COMPILER=vulkan INJECT=0 DEPTH=1
=== crash recovery — vulkan, no injection ===
warmup/samples    : 200/300   seed 42   supervised: true
  injector calls  : 2018
  placeholders    : 0          <- 0 means recovery never ran
  with_supervision: 384
  spec dispatch   : 520
  plain dispatch  : 384
  elixir subtree  : 2018
  recoveries      : 0   (reported in stats)
  divergences     : 7
  adapted eps     : 1.0534214769334085
  mean / var      : -0.0844 / 0.9550596983027236    (truth 0.0 / 1.0)
  distinct draws  : 290/300
### COMPILER=vulkan INJECT=1 DEPTH=1
** (RuntimeError) Injected crash for fault tolerance testing
    (exmc 0.3.1) lib/exmc/nuts/fault_injector.ex:62: Exmc.NUTS.FaultInjector.raise_error/1
    (exmc 0.3.1) lib/exmc/nuts/tree.ex:1501: Exmc.NUTS.Tree.build_subtree/10
    (exmc 0.3.1) lib/exmc/nuts/tree.ex:430: Exmc.NUTS.Tree.do_build/11
    (exmc 0.3.1) lib/exmc/nuts/tree.ex:128: Exmc.NUTS.Tree.build/12
    (exmc 0.3.1) lib/exmc/nuts/sampler.ex:946: Exmc.NUTS.Sampler.nuts_step_with_stats/8
    (exmc 0.3.1) lib/exmc/nuts/sampler.ex:1020: anonymous fn/9 in Exmc.NUTS.Sampler.run_sampling/8
    (elixir 1.18.3) lib/enum.ex:4507: Enum.reduce/3
    (exmc 0.3.1) lib/exmc/nuts/sampler.ex:1018: Exmc.NUTS.Sampler.run_sampling/8
### COMPILER=vulkan INJECT=1 DEPTH=3
=== crash recovery — vulkan, crash at depth 3 ===
warmup/samples    : 200/300   seed 42   supervised: true
  injector calls  : 3405
  placeholders    : 176          <- 0 means recovery never ran
  with_supervision: 746
  spec dispatch   : 3000
  plain dispatch  : 746
  elixir subtree  : 3405
  recoveries      : 0   (reported in stats)
  divergences     : 178
  adapted eps     : 2.4076557940102123e-11
  mean / var      : -0.5225 / 1.4501178438056725e-15    (truth 0.0 / 1.0)
  distinct draws  : 300/300
MISREPORTED: 176 placeholders built but stats.recoveries = 0. Warmup-phase recoveries used to vanish here.
```
