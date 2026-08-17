# NEXT — eXMC (open source)

**Written:** 2026-08-16, against `main` @ `6d6ae4f` (0.3.1, the P0 correctness
backport, merged).
**Read `MISSION.md` first** — this file assumes it and does not repeat it. This
one is only *what to do next and in what order*, with the state as it actually
stands rather than as the mission planned it.

---

## Status — 2026-08-16, end of the exla/verification pass

Items 3, 5 and 6 are closed and committed. `mix test` runs the whole repo on
both paths, and **the Vulkan sweep is reproducible for the first time**: two
consecutive `EXMC_COMPILER=vulkan mix test` runs returned identical failure
sets, differing only in ordering.

| run | result |
|---|---|
| `mix test` (default → EXLA) | 472 tests, **0 failures** |
| `EXMC_COMPILER=vulkan mix test` ×2 | 472 tests, **5 failures**, identical both times |

Item 1 remains open and is now the top of the queue.

---

## 0. Two things to know before you touch anything

### `origin` is private. `upstream` publishes.

```
origin    git@localhost:/home/git/repos/exmc.git      # private server — working remote
upstream  git@github.com:borodark/exmc.git            # PUBLIC — pushing here is a release
```

The naming inverts the usual fork convention. `main` tracks `origin/main`.
**Never push to `upstream` as the last step of a task.** It is a separate,
outward-facing decision. See §1.

### `rm -rf _build/` — do it early, do not agonise

`_build/` regenerates from source and the lockfile. Nothing is lost. The `test`
env goes stale *independently* of `dev`, and a stale `_build/test/lib/<dep>` is
a first-class time sink: on 2026-08-16 it cost a long detour before anyone
noticed `_build/test/lib/nx_vulkan` was version **0.1.0** while `mix.lock`
pinned `7067499`, with a NIF missing `device_supports_f64/0` and
`leapfrog_chain_synth_f64/6`. Symptom was 20 integration failures that looked
like anything but a build artifact.

Suspect `_build/test/lib/` **first** on: `UndefinedFunctionError` for a NIF, a
`:bad_lib` on_load warning, a loaded version disagreeing with `mix.lock`, or
"suddenly every test in the repo fails."

```sh
rm -rf _build/                                     # the blunt instrument, and it is fine
MIX_ENV=test mix deps.compile nx_vulkan --force    # the surgical one
```

**The one exception was `exla`, and it is now resolved** — see §2 item 3. One
thing about it is still worth knowing here: neither the built `libexla.so`
(cached in `~/.cache/xla/exla/`, keyed by elixir/erts/xla/exla versions and
**not** by target) nor exla's C++ objects are reached by `rm -rf _build/`, so a
stale CUDA build survives every `_build` deletion you can think of.
[`docs/EXLA_CPU_BUILD.md`](docs/EXLA_CPU_BUILD.md) has the clearing recipe.

---

## 1. Decide: publish 0.3.1, or fix the known issue first

**This is the only item that is genuinely blocked on a human.**

`origin/main` is at 0.3.1. `upstream/main` is at `2de6c60` (0.3.0) — nine
commits behind. The whole correctness release is unpublished, and 0.3.0 is
still what a `mix deps.get` gets.

The case for publishing now: 0.3.0 ships a sampler whose posteriors are
over-dispersed — `Normal(0,1)` variance 1.378 against a true 1.0, worse under
`compiler: :vulkan`. Every day it stays up, someone can draw samples from it.
The CHANGELOG says so plainly and that is the right instinct.

The case for waiting: Vulkan is the default *for the users who most need this
library*, and it is still not correct for models with observations (§2).
Publishing a release whose headline is "correctness" while that backend returns
a frozen chain for a whole model class is a second credibility problem, not a
fix for the first.

> **Correction, 2026-08-16.** This file, `MISSION.md`, and the annotation on
> the red test all say `compiler: :vulkan` is "the default". That is not what
> the code does. `config/config.exs` sets no compiler at all, so the default is
> `Exmc.JIT.auto_detect/0`, which prefers **EXLA when it is available** and only
> falls through to Vulkan when it is not. The belief came from this host, where
> EXLA could not load and auto-detect therefore always landed on Vulkan (§2
> item 3, now fixed). Verified directly on one checkout, no config changed:
> while `libexla.so` was unloadable `Exmc.JIT.detect_compiler/0` returned
> `Nx.Vulkan`; once the CPU EXLA loaded, the same call returned `EXLA`.
>
> This narrows the §2 item 1 exposure rather than removing it: the users who get
> Vulkan by default are exactly the ones with no working EXLA — the FreeBSD and
> non-CUDA GPU hosts the backend exists for. They are still the ones who cannot
> use the alternative. But "the default compiler is silently wrong" is not an
> accurate description of what a hex user gets, and the §1 decision should not
> be argued on it. The middle path below still stands on its own terms; it is
> just a smaller change than "flip the default" makes it sound, since for most
> users the default is already EXLA.

**A middle path worth considering, and probably the right one:** publish 0.3.1
with the default compiler changed to `:none`, and `:vulkan` opt-in until §2 is
resolved. That makes the release honest end to end — correct posteriors *and* a
default that delivers them — and it costs users nothing they were actually
getting, since MODEL_SCALING says the Vulkan path is slower than
`BinaryBackend` at the widths eXMC runs anyway (MISSION §1). It is a one-line
change plus a CHANGELOG note.

Whichever way it goes, `mix hex.publish` needs an interactive password and is
the operator's to run.

---

## 2. P0 continued — what 0.3.1 did not close

Ranked. Item 1 is the only one that blocks calling the default backend correct.

| # | item | effort | why it ranks here |
|---:|---|---|---|
| 1 | **The vulkan observed-model defect.** `compiler: :vulkan` returns a frozen chain (1 distinct value in 500 draws, `accept_prob` ≈ 0.002) for models with observations. Full write-up, evidence, and the next experiment in [`docs/OPEN_VULKAN_OBSERVED_MODEL.md`](docs/OPEN_VULKAN_OBSERVED_MODEL.md). | 1–3 days | it is the **default** compiler. Until this is fixed, the default can silently return a degenerate posterior. |
| 2 | **The `assert_in_delta` sweep.** 0.3.1 tightened exactly one assertion (`integration_test.exs:29`, now checking the closed-form conjugate posterior via `Validator.check_analytic/3`). The rest of the suite is unswept. Find every tolerance that would accept a 20% variance error. | half a day | this is VERIFICATION_METHODS' rank-1 item and the reason two defects shipped |
| 3 | ~~**The EXLA build.**~~ **Done — both halves.** The library bug is fixed (`exla` is `runtime: false`, `Exmc.JIT` starts it lazily and treats a failed start as "backend unavailable", covered by `test/optional_deps_test.exs`), *and* this host now has a working CPU EXLA. Recipe and its two traps in [`docs/EXLA_CPU_BUILD.md`](docs/EXLA_CPU_BUILD.md); the short version is `EXLA_CPU_ONLY=1 XLA_TARGET=cpu`, not `XLA_TARGET=cpu`. | — | — |
| 4 | **The wall-clock test.** `mix test` is now **0 failures** on the default (EXLA) path — 375 tests on `main`, 472 on `gate1/reconcile-core` with the MCLMC/MAMS/SBI suites. The only default-path failure left is `integration_test.exs:738` — `assert t_vec < t_par` — and it is **timing-flaky**, not consistently red: it failed at `1034ms < 659ms` on one run and passed on the next with no code change. Move it to `bench/`. The other three failures NEXT.md originally listed were artefacts of Vulkan-by-default and are green under EXLA — **not fixed, not exercised**. | 1 hour | a flaky red trains people to ignore red faster than a stable one |
| 6 | ~~**Tests leak `:exmc` application env into each other.**~~ **Done, and verified by the check that matters: two consecutive Vulkan sweeps now return identical failure sets.** Three separate instances, all restoring wrongly or not at all: `p0_correctness_test.exs` leaked `compiler: :none`; `nuts_test.exs:618` "reset" `full_tree_nif` to `true` when its default is `false`; `fault_tolerant_test.exs` did the same via `get_env(..., true)` and skipped its restore entirely on a raised assertion. `native_tree_test.exs` was `async: true` while setting `use_nif` globally, so it raced concurrent tests rather than merely later ones. Fixed with `Exmc.TestHelper.put_env_scoped/3` (reads the previous value instead of assuming a default — the assumption is what went wrong three times) plus an `ExUnit.after_suite` tripwire over **all twelve** `:exmc` keys that gate behaviour, since an ordinary assertion only sees leaks from files that ran *before* it. | — | — |
| 5 | ~~**`config/test.exs` had never been loaded.**~~ **Fixed.** `config/config.exs` was one line, `import Config`, with no `import_config` — and Mix auto-loads only `config/config.exs`, so every setting in `config/test.exs` was dead: the `EXMC_COMPILER` switch, `config :exla, default_client: :host`, `allow_vulkan_perop_sampling`. **Every `EXMC_COMPILER=vulkan mix test` ever run sampled with whatever auto-detect picked and reported a pass for it.** Now imported, with `test/config_test.exs` as the tripwire. | — | — |

### Do not skip the red test — and mind which backend it is running

The annotated test is `integration_test.exs:639`, "vector obs produces same
posterior as equivalent scalar obs" (this file previously said 611). It fails
for a real reason and is annotated to say so. Skipping a test that fails for a
real reason is precisely the habit that let both 0.3.1 defects ship.

**It is red again under Vulkan, and that is the correct state.** Since item 5
was fixed, `EXMC_COMPILER=vulkan mix test test/integration_test.exs:639` fails
on exactly the documented assertion:

```
code: assert_in_delta scalar_summary["mu"].std, vector_summary["mu"].std, 0.3
```

Item 1 is confirmed alive, and reproduces the numbers in
`docs/OPEN_VULKAN_OBSERVED_MODEL.md` to the digit — scalar arm mean **3.6503**,
sd **3.29e-14**, **1 distinct draw in 500**; vector arm mean 3.9716, sd 0.5516,
472/500. Under `EXMC_COMPILER=none` both arms are correct.
`allow_vulkan_perop_sampling` makes no difference to it either way, which rules
that out as the route around the chain shader.

It is **green under a bare `mix test`**, and that is not a fix — auto-detect
picks EXLA on this host and the Vulkan path is never entered (see the
correction in §1). A bare `mix test` passing says nothing about item 1. Use
`EXMC_COMPILER=vulkan`, and note that it only means anything now that item 5 is
fixed.

### What the Vulkan sweep actually says

The first honest `EXMC_COMPILER=vulkan mix test` this repo has run:
**375 tests, 4 failures**, against 375/0 on the default path.

| test | failure |
|---|---|
| `new_dist_test.exs:271` | `{:error, :dispatch_failed, "read spv: No such file or directory"}` — a missing SPIR-V file, not a numerical defect |
| `level_set_integration_test.exs:11` | timed out at 300s |
| `fault_tolerant_test.exs:234` | `Variance collapsed: 1.45e-15` — **the frozen-chain signature of item 1**, in a second test |
| `integration_test.exs:738` | the wall-clock assertion (item 4), fails on both paths |

Both cautions that used to sit here are resolved. The table **is** now
reproducible: two consecutive sweeps produced exactly these five, differing only
in ordering. `integration_test.exs:639` now fails in the full run as it always
did in isolation — that was item 6, and it is what made the difference.

Two entries deserve reading as item-1 evidence rather than as separate bugs:
`fault_tolerant_test.exs:239`'s `Variance collapsed: 1.45e-15` is the same
frozen-chain signature as `:639`, so the observed-model defect shows up in more
than the one test this file has been tracking.

`new_dist_test.exs:271`'s `read spv: No such file or directory` is **gone from
both sweeps** and is no longer expected: it was a race in the shader cache, now
fixed — see below.

### The shader cache was racy, and it served empty shaders

`Exmc.NUTS.CustomSynth.Compile.compile_fresh/2` pointed `glslangValidator -o`
straight at the content-addressed cache path, and derived its temp GLSL path
the same way. Concurrent callers synthesising the same shader therefore shared
both paths — and ten-plus test modules are `async: true` and sample.

Measured, 24 concurrent compiles of one shader over 40 rounds:
**50 of 960 callers received `{:ok, spv_path}` for a zero-byte file.**
`glslangValidator` creates its output before writing it, so the `File.exists?/1`
fast path returned a module that had no contents yet. After compiling to
per-caller temp paths and `File.rename/2`-ing into place: **960/960 clean.**

Note what the repro corrected. The predicted mechanism was the shared `.comp`
path letting one caller delete another's source mid-compile, surfacing as
ENOENT. That is real and is also fixed, but it is *not* what dominates — the
existence check racing the validator's file creation is, and it fails in a
worse way, because an empty shader is a successful return rather than an error.
Fixing on the hypothesis alone would have left the common case in place.

The ENOENT interleaving was never directly reproduced (it needs a validator
failure, and none occurred in 960 runs); it is eliminated by construction
rather than by observation.

**Related, and still open:** `exmc` and `nx_vulkan` hardcode the *same*
`~/.exmc/gpu_node/spv` directory in two separate codebases, and `nx_vulkan`
ships `Nx.Vulkan.Synthesis.clear_cache/0` — an `File.rm_rf` of it. Nothing
calls it today. Any future caller silently deletes exmc's synthesised shaders
mid-run.

### Two documented safety features never run

Found while chasing the above, and worth knowing before item 1:

- **`Exmc.NUTS.Vulkan.SuspectTracker` is never started.** No `start_link`
  anywhere in `lib/` or `test/`, so `alive?/0` is always false and the
  per-shader eviction policy in its moduledoc — three consecutive timeouts
  evicts a shader and routes around the GPU — has never executed.
- **`:gpu_node` is read but never set** (`tree.ex:885`), so the watchdog path
  through `Nx.Vulkan.Node.with_node` is unreachable and `route_chain_direct`
  always takes the bare-dispatch branch.

Net: there is currently **no timeout containment on the Vulkan path**. The two
300s timeouts in the sweep hang until ExUnit kills them.

---

## 3. P1 — verification as a deliverable

Unchanged from `MISSION.md` §7 P1, except that its foundation now exists: 0.3.1
landed `Validator.ess/1` (Geyer), `analytic_moments/1`, `check_analytic/3`, a
variance SE that does not assume normality, and `bench/nuts_truth.exs`. Build on
those rather than starting over.

The ranked plan is in
`/home/io/projects/learn_erl/pymc/exmc/docs/VERIFICATION_METHODS.md`
(cross-repo, 1,641 lines). Its rank-1 item is §2 item 2 above. Its next is
**Geweke's joint distribution test**, which is the one check that would have
caught the tree defect at the point of introduction rather than months later.

**One addition to that plan, from what 0.3.1 found:** a **leaf-level
differential between the chain shader and the host**. Fix `q0`, `p0`, `eps`,
`inv_mass`, `K`; dispatch `leapfrog_chain_synth_f64`; read back all four
arrays; run the same K leapfrog steps through `Exmc.Compiler`'s `vag_fn`;
compare element-wise. No existing test does this, and it would have caught the
`logp_chain` off-by-one immediately. It is also the next experiment §2 item 1
needs.

### A methodological note worth keeping

Every statistical check in this repo before 0.3.1 was **differential** — run the
model two ways, assert agreement. That is structurally blind to any defect the
two arms share, and both arms share the NUTS tree. Two real defects lived behind
a green suite for months because of it.

When you add a check, ask: *what would this see that comparing two arms would
not?* If the answer is nothing, it is not buying much.

---

## 4. Beyond correctness

Only after §2 items 1 and 2. From `MISSION.md` §7 P2 onward, unchanged, with
the ordering intact — graph shape (16–20×) before shaders, and the README
honesty fix (§1 of the mission: reach, not speed) before any new performance
claim.

One item that moved up as a result of 0.3.1: **`bench_results/` still does not
exist here.** `bench/` now has two files (`cse_race.exs`, `nuts_truth.exs`).
Every performance claim in the README should point at a file containing raw
output and the host it ran on, the way `nx_vulkan/bench_results/` does.

---

## 5. Where the numbers came from

So the next person can re-run rather than trust:

```sh
# posterior moments vs analytic truth — the check that can see a shared defect
mix run --no-deps-check bench/nuts_truth.exs
COMPILER=vulkan SEEDS=1,2,3 mix run --no-deps-check bench/nuts_truth.exs
USE_NIF=0      mix run --no-deps-check bench/nuts_truth.exs   # pure Elixir tree
FULL_TREE_NIF=1 mix run --no-deps-check bench/nuts_truth.exs  # Rust build_full_tree

# the regression tests (move _build/test/lib/exla aside first — §0)
mix test --no-deps-check test/nuts/p0_correctness_test.exs
```

**There are three tree implementations**, selected by `use_nif` and
`full_tree_nif`, and each carries its own copy of the doubling logic. A guard
added to one and not the others is a defect that only appears under whichever
flag nobody set. `bench/nuts_truth.exs` sweeps all three for this reason.

To prove a code path actually ran — which matters more than it sounds, since a
vacuous check reads exactly like a passing one — use `:call_count` tracing, and
**force-load the modules first** or `trace_pattern` silently matches nothing:

```elixir
Code.ensure_loaded!(Exmc.NUTS.NativeTree)
:erlang.trace_pattern({Exmc.NUTS.NativeTree, :build_subtree_bin, 10}, true, [:call_count])
# ... run ...
:erlang.trace_info({Exmc.NUTS.NativeTree, :build_subtree_bin, 10}, :call_count)
```

---

## 6. B1 (MCLMC / MAMS) — landed, with one measurement unfinished

**Written 2026-08-16.** Roadmap item B1 from
`/home/io/projects/learn_erl/pymc/exmc/docs/PLAN_SAMPLER_ROADMAP.md` §3.
Stages B1.1–B1.3 are complete and gated. **B1.4, the bias measurement, is
partially run and needs finishing — a host reboot interrupted it.** B1.5 (the
GLSL arm) is deferred to Gate 5 and was deliberately not started.

### What landed

| file | what |
|---|---|
| `lib/exmc/mclmc/integrator.ex` | the isokinetic step — minimal-norm `V T V T V` splitting, λ = 0.1931833275037836, plus leapfrog |
| `lib/exmc/mclmc/tuning.ex` | EEVPD step-size adaptation, the two `L` estimators, the moment accumulator |
| `lib/exmc/mclmc.ex` | `sample/3`, `sample_compiled/3` — biased, one draw per integrator step |
| `lib/exmc/mams.ex` | the same dynamics with a Metropolis accept — asymptotically unbiased |
| `test/mclmc/{integrator,tuning,mclmc,mams}_test.exs` | 42 tests, all green |
| `bench/mclmc_bias.exs` | the B1.4 sweep |
| `bench_results/MCLMC_BIAS.md` | **partial** — see below |

Nothing in the model layer changed. Both samplers take
`Exmc.Compiler.compile_for_sampling/1`'s tuple and use only its `vag_fn` and
`PointMap` slots.

### To continue after the reboot: finish the bias sweep

**This is the one unfinished thing. Start here.**

`bench_results/MCLMC_BIAS.md` carries a `PARTIAL RUN` banner. What it actually
contains, and what it does not:

| block | state |
|---|---|
| `Normal(0,1)`, `HalfNormal(1)`, `Exponential(2)` at `d = 2` | **complete** — all ten rows each |
| `Normal(0,1)` at `d = 8` | **partial** — MCLMC at six step sizes, MCLMC (tuned) and MAMS are there; **the NUTS row is missing** |
| `HalfNormal`/`Exponential` at `d = 8` | missing |
| everything at `d = 32` | missing |

The missing NUTS row at `d = 8` is not a cosmetic gap: at `d = 8` MAMS reaches
**1.6964 ESS/gradient** and MCLMC (tuned) **1.5258**, against 0.4169 and 0.2283
for the same two at `d = 2`. Whether that beats NUTS is the entire B1 case and
**the number that would answer it is the one that did not finish.** `d = 32` is
what the roadmap's "abandon if" is really about.

**Do not quote this file as evidence about high dimensions until the sweep is
re-run.** The command, unchanged:

```sh
DIMS=2,8,32 SEEDS=1,2,3 WARMUP=1000 SAMPLES=3000 \
  EPS=0.1,0.25,0.5,1.0,2.0,4.0 OUT=bench_results/MCLMC_BIAS.md \
  mix run --no-deps-check bench/mclmc_bias.exs 2>&1 | tee /tmp/mclmc_bias.log
```

Budget roughly **3–4 hours** on `super-io` under load — the three `d = 2`
blocks alone took about an hour of process time with a second agent on the box.
Run it on an otherwise idle host if one is available, and `tee` it: three
things worth knowing before starting.

- **Fix the incremental write first — it is ten lines and it already cost one
  run.** The script builds the whole document in memory and writes `OUT` once,
  at the very end, so a kill loses the file. Every row *is* printed to stdout as
  it is produced, which is how the current partial file was recovered, but that
  recovery should not have been necessary. Append each block to `OUT` as it
  completes.

- **NUTS is the slow arm by a wide margin, and not because of tree depth.**
  Measured directly on this host, `compiler: :none`, two `HalfNormal(1)` RVs:
  600 NUTS iterations took 18.8 s at a mean tree depth of 2.1 and 1106 total
  gradient evaluations — about **8 ms per gradient**. MCLMC on the same model
  and backend runs ~2150 gradients/second, i.e. **~0.5 ms per gradient**. That
  is a 15× per-gradient overhead in the tree machinery, not an algorithmic
  difference, and it is not visible in any ESS-per-gradient table. It deserves
  a profile of its own.

- **`mix test` is green before you start.** The full suite was **461 tests, 1
  failure** with all of this in place, and the one failure is the known
  pre-existing wall-clock assertion at `integration_test.exs:762` (§2 item 4).
  The four new files add 42 tests and take about 5 minutes of the run. If the
  suite is not in that state after the reboot, fix that before trusting any
  benchmark number.

### What is not done, and should be

- **B1.4 at `d = 8` and `d = 32`.** Above. This is the item that decides
  whether B1 was worth doing.
- **A Geweke joint-distribution run against MAMS.** The roadmap asks for it and
  it is the right check for a novel accept step. It needs `simulate_from_prior/2`
  and a public single transition — §3 / `MISSION.md` §7 P1 item 9. MAMS's
  accept step is currently gated by an exact involution test
  (`test/mclmc/mams_test.exs`) plus the analytic-moment battery, which is
  strong but is not a joint-distribution test.
- **`init_values` under NCP.** Both samplers raise rather than guess;
  `Exmc.NUTS.Sampler.invert_ncp_init/2` is private and was not duplicated.

### One general finding, and it is not confined to B1

**A bare Elixir float in an `Nx` binary op silently computes at f32, even
against an f64 tensor.** `Nx.divide(f64_tensor, 0.9695359714832659)` returns an
f32-accurate result: the scalar becomes a default-typed `{:f, 32}` tensor and
the promotion widens *after* the arithmetic. Measured on `Nx.BinaryBackend`.

This cost real time here — the `‖u‖ = 1` invariant failed at **3e-8**, which is
f32 epsilon, and read exactly like an algebra error in a new integrator.
`Exmc.MCLMC.Integrator` now routes every scalar through a `c/2` helper that
builds it at the tensor's own type, and the note is in that module's source.

**The rest of the repository has not been audited for this.** `sampler.ex`,
`leapfrog.ex`, `tree.ex` and `mass_matrix.ex` all mix Elixir floats with
tensors. Any place that does is computing at f32 while believing it is at f64,
and `MISSION.md` §4's "the f64 default is not costing anything and it is a
correctness asset" is only true where the default actually applies. Worth a
grep before the next precision question is diagnosed as a backend problem.
