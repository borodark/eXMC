# eXMC — mission, goals, and a working plan

**Written:** 2026-08-16, against `main` @ `2de6c60` (release 0.3.0, *"A Posterior
on Any GPU"*).
**Audience:** whoever picks this repository up next, with no memory of the
conversation that produced this file.
**Status:** planning document. It supersedes nothing in the codebase; it decides
what the next six months of work are for and, more importantly, what they are
*not* for.

Every number below is either **measured** — with the file that measured it — or
**read from source**, with `file:line`. Where the two disagree with something
this repository currently claims in public, that is said plainly rather than
smoothed over. Two of those disagreements are correctness bugs shipping in
0.3.0 today.

---

## 0. Orientation — three repositories, and which one this is

You will find three trees on this machine with `exmc` in the path. They are not
the same thing and confusing them wastes a day.

| path | what it is | your relationship to it |
|---|---|---|
| `/home/io/projects/learn_erl/_exmc-things/exmc/` | **this repo.** The open-source release. Remote `git@github.com:borodark/exmc.git`, `main` in sync with `origin/main`. | yours |
| `/home/io/projects/learn_erl/pymc/exmc/` | the private trader. Upstream of this one; 261 commits ahead of its own private remote. Contains `lib/exmc/trading/` (48 files), `lib/exmc/license/`, and `research/` (patent drafts, business material). | **read-only.** Backport *from*; never modify, never copy wholesale. |
| `/home/io/projects/learn_erl/nx_vulkan/` | the Vulkan GPU backend for Nx, a separate library with its own plan. | read-only; file bugs against it, do not edit it from here |

The OSS repo is a **backport target**, not a fork. Code flows private → public,
filtered. One module has gone the other way
(`lib/exmc/nuts/vulkan/scheduler.ex`, 383 lines, OSS-only) and that is a
forward-port candidate, not a divergence to reconcile.

---

## 1. The mission

> **eXMC is a probabilistic programming environment for the BEAM: PyMC's model
> semantics and Stan's sampler discipline, executed on a runtime whose native
> units are isolated processes, message passing and supervision — so that
> multi-chain sampling, per-sample streaming and distribution across
> heterogeneous nodes are language primitives instead of infrastructure.**
>
> **Its GPU story is reach, not speed: the Vulkan path exists so that a
> posterior can be computed on hardware where EXLA does not exist — FreeBSD,
> non-NVIDIA, Apple via MoltenVK — and wherever EXLA *does* exist, eXMC should
> use it and say so.**

Two sentences, and the second one is the one that costs something to say.

### Why "on the BEAM" and not "on any GPU"

The operator's stated goal for this repository is *"open source eXMC with
appropriate GPU compute (it has deeper models in terms of parameters and can
benefit from GPU)."* The premise is half right, and the half that is wrong is
load-bearing.

**True:** there is a real, reproducible crossover where GPU compute wins. The
per-op Vulkan path overtakes `Nx.BinaryBackend` at roughly **10³ f64 elements**
in the likelihood and reaches **410×** by 4×10⁵ elements
([`nx_vulkan/bench_results/MODEL_SCALING.md`](../../nx_vulkan/bench_results/MODEL_SCALING.md),
Result 1). Depth helps. The GPU arm is essentially flat in model size — d=4 to
d=1024, 256× more arithmetic, 12.4 ms → 17.6 ms — because it pays for
dispatches, not work. And the shipped models really are deep enough: seven of
them exceed 10⁴ likelihood elements (§5.1).

**False, and this is the part that decides the mission:** that crossover is
against a pure-BEAM tree-walking interpreter. Against a CPU that JIT-compiles,
**EXLA on the same host's CPU beats the Vulkan per-op path at every size
measured — 20× at the small end, 215× at 6×10⁶ elements, gap widening**
(MODEL_SCALING Result 4). `exla_cuda` is indistinguishable from `exla_host` on
this workload even at 61 million elements. There is no reachable model width on
a machine with EXLA where the Vulkan path is the right choice.

So "MCMC on any GPU" as a *performance* proposition is not supported by any
measurement we have. As a *portability* proposition it is fully supported: the
FreeBSD Kepler hosts (mac-247, mac-248) have no EXLA, `BinaryBackend` genuinely
is the alternative there, and Result 1 is then the whole story — one to two
orders of magnitude at n_obs in the thousands.

`nx_vulkan/README.md` already carries this warning in its own benchmark section
("This project's case is reach, not speed… Where CUDA exists, use EXLA").
**This repository should carry the same one**, for the same reason: an OSS
README that publishes speedup-against-an-interpreter tables next to the word
"GPU" is telling users something that will not survive their first
`mix deps.get exla`. That is a credibility cost, and it is avoidable by writing
one paragraph.

---

## 2. Goals

Ordered. Each has a test that says whether it is met.

1. **Ship a sampler whose posteriors are right, and prove it.**
   *Met when:* a Geweke joint-distribution test and a set of analytic-moment
   checks run in CI, and both of the correctness defects in §6.1 are fixed with
   regression tests that would fail without the fix.

2. **Be the obvious choice on the BEAM.** Model semantics a PyMC user
   recognises; diagnostics an ArviZ user recognises; concurrency, streaming and
   distribution that no Python PPL can offer without external infrastructure.
   *Met when:* the README's first three sections are about those, and the
   performance section compares against a compiler rather than an interpreter.

3. **Run everywhere the BEAM runs.** EXLA where available; Vulkan where it is
   not; `BinaryBackend` as the correctness reference and the last resort.
   *Met when:* backend selection is documented as a reach hierarchy, and the
   0.3.0 Vulkan work is described in the README at all (it currently is not —
   §6.2).

4. **Make deep models cheap to write and cheap to run — on every backend.**
   The single largest measured performance factor available to this repository
   is not a shader, it is graph shape: 16–20× on the GPU, and it helps EXLA and
   `BinaryBackend` too (§4, finding 3).
   *Met when:* the shipped models and the posteriordb harness are written with
   vector RVs, and a standing width benchmark records the effect.

5. **Keep the public/private boundary honest.** Everything general goes public;
   trading, licensing and business material never does.
   *Met when:* the backport queue in §7 is empty of P0/P1 items.

---

## 3. Non-goals

- **Beating EXLA.** Not a goal at any size. Nothing measured suggests it is
  reachable, and pursuing it would misdirect the whole project.
- **Being a general-purpose GPU compute library.** That is `nx_vulkan`'s job,
  and it has its own plan.
- **Feature parity with PyMC's distribution catalogue.** 21 distributions
  cover the shipped notebooks. More distributions is the cheapest-looking and
  lowest-value work available (§8).
- **A terminal progress UI.** `sample_stream/4` composes with LiveView, Scenic
  or a GenServer; that is the BEAM answer and it is better than a built-in.

---

## 4. The evidence this plan rests on

Six findings. The first five are measured; the sixth is a literature review.
Cite these, not intuition, when a plan item is questioned.

**1 — The crossover is real and it is on total elements.**
`nx_vulkan/bench_results/MODEL_SCALING.md` Result 1. Per-op Vulkan overtakes
`BinaryBackend` at ~10³ f64 elements; 410× at 4×10⁵. Neither `d` nor `n_obs` is
special — equal `d × n_obs` gives equal times on both arms regardless of which
axis supplied them. The practical asymmetry is that `n_obs` is a data decision
and `d` is a modelling decision, so `n_obs` moves further in practice; the
physics does not care.

**2 — EXLA on the host CPU beats the Vulkan path everywhere.**
MODEL_SCALING Result 4: 22× at 480 elements, 215× at 6×10⁶, gap widening.
`exla_cuda ≈ exla_host` at every cell. Whole-graph fusion
(`Nx.Vulkan.Compiler`) is within noise of per-op on **13 of 13 cells** on
exactly the elementwise-heavy graph it was designed for (Result 2) — so the one
mechanism that could close the gap closes none of it, for reasons still
undiagnosed.

**3 — Graph shape is worth ~20×, and it is backend-agnostic.**
MODEL_SCALING Result 5. The same posterior with the same FLOP count, written as
`d` scalar RVs (Model S) instead of one `shape: {d}` RV (Model V), costs
**~15 ms per additional RV** on the per-op GPU path — linear in `d`, undisguised
dispatch cost — and moves the crossover from ~10³ elements to ~2×10⁴. At d=64
the two forms are **16× apart on the GPU with identical arithmetic**.

The mechanism is on the eXMC side, not the backend's, and it is three lines
long. `Exmc.PointMap.unpack/2` (`lib/exmc/point_map.ex:85-91`) emits one
`Nx.slice` plus one `Nx.reshape` **per map entry**:

```elixir
def unpack(flat, %__MODULE__{} = pm) do
  Map.new(pm.entries, fn entry ->
    slice = Nx.slice(flat, [entry.offset], [entry.length])
    value = Nx.reshape(slice, entry.shape)
    {entry.id, value}
  end)
end
```

A model with `d` scalar RVs therefore emits `2d` ops before any likelihood
arithmetic happens; a model with one `shape: {d}` RV emits two. Everything
downstream inherits that op count, on every backend.

**The good news is that the compiler path already handles vector RVs
correctly.** `PointMap.build/1` gives a `shape: {n}` RV one entry of length `n`
(`point_map.ex:38-53`, via `Transform.unconstrained_length/2`), `unpack/2`
reshapes it back to a single tensor, and `Compiler` calls the logpdf once on the
whole thing — `Nx.sum(dist.logpdf(...))` at `lib/exmc/compiler.ex:294`, with the
reason spelled out at `:290-293`. **A vector RV stays one tensor end to end; it
is not split into `d` scalars.** So this is a modelling-and-rewriting job, not
new compiler machinery. Two paths do *not* handle it (§5.2b, §5.2f) and need
guards.

**4 — The chain-shader synthesis path cannot reach the widths where GPU compute
starts to win.** MODEL_SCALING Result 6, confirmed here from source.
`Exmc.NUTS.CustomSynth.Push.pack/1`
(`lib/exmc/nuts/custom_synth/push.ex:73-97`) writes a 24-byte header
(`K`, `n_obs`, `d`, pad, f64 `eps`) plus 8 bytes per prior scalar into a
`@max_bytes 128` block (`push.ex:45`). That leaves room for exactly **13 prior
floats** — `d ≤ 13` for one-parameter priors, **`d ≤ 6` for `Normal`**. The
per-op crossover sits at d ≈ 15–30 at n_obs = 60. *The two envelopes do not
overlap.* Synthesis also costs up to two minutes at d = 12 (~1.8× per unit of
`d`), runs **3.2× slower than `BinaryBackend`** at d = 8 / n_obs = 60, and
**panics the NIF (`:nif_panicked`) at n_obs = 600**.

The documented cap is wrong by a factor of 20 and the wrong number is in this
repository in twelve places: `lib/exmc/nuts/vulkan/dispatch.ex:86`,
`lib/exmc/compiler.ex:106`, and `lib/exmc/nuts/tree.ex:630,681,690,700,710,720,
730,740,751,761,771,784` all say or guard on `d <= 256`.

**5 — Two correctness defects, both fixed upstream today, both shipping here.**
See §6.1. Measured effect: `Normal(0,1)` posterior variance **1.45** against a
true 1.0 for the tree defect; **8.55** for the chain-shader defect.

**6 — Verification methodology exists and is ranked.**
`/home/io/projects/learn_erl/pymc/exmc/docs/VERIFICATION_METHODS.md` (1,641
lines, 2026-08-16) is a literature review that ranks what to build by
`defect classes caught / (effort + flakiness)` for a two-person project:

| rank | item | effort | flakiness |
|---:|---|---|---|
| 1 | repair the gates that already exist | one day | reduces it |
| 2 | **Geweke joint distribution test** (exact-invariance form) | ~10⁴ chain iterations of plumbing | low |
| 3 | integrator invariants on shipped code | half a day | **zero** — exact assertions |
| 4 | conjugate exact-posterior tests | a day | low |
| 5 | Simulation-Based Calibration | 2–3 days + a lot of compute | controllable |
| 6 | close the backend-residency gap | an hour | zero |
| 7 | finite-difference check of the log-density gradient | two hours | low |

Note the nuance: Geweke is ranked **second**, not first, and only because rank 1
is its prerequisite — *"every other recommendation here is worthless in a
harness people have learned to ignore."* On cost, the document computes that
Geweke reaches 100% power at 10⁴ iterations of one chain where SBC needs 400
complete fits (~4×10⁵ iterations) for 76.5% power: **roughly 40× cheaper in
sampler work**, and far cheaper in orchestration.

That document also says, about the tree defect that had just been found:
*"the defect was in `Tree.do_build/11`, not in the integrator, so none of the
above would have found it — and the tree-level property tests that would be the
analogue already exist and did not find it either."* Take that seriously. It is
the argument for Geweke over more unit tests.

---

## 5. What actually ships, and does it justify the GPU claim

**Yes — against `BinaryBackend`, and with three qualifications that change what
you should do about it.** Eight runnable model instances exceed 10⁴ likelihood
elements against a ~10³ crossover, and 21 of the 32 posteriordb models exceed
10³. Six of the eight are already vectorised — and those six are precisely the
ones the fused GPU sampling path mis-handles.

### 5.1 The inventory

`d` = free RV scalars (an RV declared `shape: {n}` counts `n`). `n_obs` = the
observation count the likelihood is evaluated over. `d × n_obs` is the variable
MODEL_SCALING found the crossover in (~10³). Graph shape: **S** = `N` scalar RVs
with the likelihood unrolled in Elixir; **V** = `shape: {d}` vector RVs with the
likelihood as tensor ops.

| model | file | d | n_obs | d × n_obs | shape | ≥10³? |
|---|---|---:|---:|---:|---|---|
| the batch — 4-level hierarchical Bernoulli | `notebooks/14_the_batch.livemd:162,367-373` | 34 | **2,000** | **68,000** | **V** (3 vector RVs) | **yes** |
| stochastic volatility state space | `notebooks/17_state_space.livemd:78,197-198` | 302 | 150 | **45,300** | **V** (2 × `GaussianRandomWalk shape: {150}`) | **yes** |
| hierarchical Weibull, 20% censored | `benchmark/reliability_model.exs:37-48`, `reliability_data.exs:12-13` | 44 | 1,000 | **44,000** | **S** | **yes** |
| bivariate unobserved components | `notebooks/trend_cycle_demo.livemd:713,742-743` | 244 | 120×2 | **29,280** | **V** | **yes** |
| UC stochastic volatility | `notebooks/trend_cycle_demo.livemd:264,497-498` | 242 | 120 | 29,040 | **V** | yes |
| local-level state space | `notebooks/17_state_space.livemd:131` | 152 | 150 | 22,800 | **V** | yes |
| local-level UC | `notebooks/trend_cycle_demo.livemd:294` | 122 | 120 | 14,640 | **V** | yes |
| posteriordb `nes1992-nes` (largest of 32) | `benchmark/posteriordb/validate_posteriordb.exs:151` | 10 | 1,350 | 13,500 | **S** | yes |
| posteriordb, median of 32 | idem | 3–10 | 46–1,350 | **1,736** | **S** | **21 of 32** |
| poker opponent model | `lib/exmc/poker/opponent_model.ex:58-64,77-104` | 20 | 240 | 4,800 | **S** | yes |
| capacity / CPU-pinning benches | `benchmark/capacity_bench.exs:24-37`, `cpu_pinning_bench.exs:35-42` | 8 | 200 | 1,600 | **S** | borderline |
| beta-binomial (placenta previa) | `notebooks/bda/stan_translations.livemd:106,118` | 1 | 980 | 980 | **S** | no |
| level set, 8×8 as shipped | `notebooks/08_level_set.livemd:72-73,245,265`; `lib/exmc/physics/level_set.ex:127,153-155` | 65 | 8 sensors | 520 | **V** | no — but see (b) |
| level set, 16×16 / 32×32 — *described, not run* | `notebooks/08_level_set.livemd:510-511` | 257 / 1,025 | 8 | — | **V** | see (b) |
| Bayesian SPC (Nile changepoint) | `notebooks/13_bayesian_spc.livemd:78,497-502` | 4 | 100 | 400 | **S** | no |
| the remaining ~30 notebook models | `notebooks/`, `notebooks/bda/`, `notebooks/bda-cyber/` | 1–10 | 5–200 | 5–400 | **S** | no |

Vector RVs appear in exactly **six** shipped (non-test) call sites, and only
**one** of them is in `lib/`: `lib/exmc/physics/level_set.ex:155`. The others
are `notebooks/08:265`, `notebooks/14:367,370,373`, `notebooks/17:131,197,198`,
`notebooks/trend_cycle_demo:294,497,498,742,743`.

Two notebooks are **broken as shipped**: `notebooks/09_radon_bhm.livemd`
(nominally d=90 / ~919 obs) and `notebooks/11_insurance_claims.livemd` both
`Code.require_file` builders in `benchmark/` — `radon_{data,model}.exs`,
`insurance_{data,model}.exs` — that are absent from the working tree *and* from
git history.

### 5.2 What the inventory says

**a. The premise holds against the interpreter, and the models are already deep
enough.** Eight runnable model instances sit between 1.35×10⁴ and 6.8×10⁴
elements, and 21 of the 32 posteriordb models exceed 10³. That is in the region
MODEL_SCALING Result 1 says the per-op Vulkan path beats `BinaryBackend` by
1–2 orders of magnitude. **On a host without EXLA — the FreeBSD Keplers — the
GPU is the right answer for these models today.** That is a real, defensible
claim and it is the one worth making.

It is not the right answer on a host *with* EXLA, at any of these sizes, by
20–215× (Result 4). Both statements are true; the README has to carry both.

**b. The six deepest vectorised models are the ones the fused GPU path
mis-handles.** `CustomSynth.extract_components/1`
(`lib/exmc/nuts/custom_synth.ex:326`) builds `layout` with **one slot per RV
id**, while `PointMap` gives a `shape: {150}` RV **150 slots**.
`standard_rv_node?/1` (`:334-338`) has **no `node.shape` guard**, so nothing
rejects a vector RV. Downstream, the mismatch is *silently absorbed* by
`pad_or_truncate(list, d, 1.0)` in `lib/exmc/nuts/sampler.ex:1531` — comment:
*"If layout length didn't match d for any reason"* — so the RV gets one real
per-RV inverse-mass value and `n−1` identity fillers. Related:
`multi_rv_custom_spec.ex:951-960` scalarises vector prior params by taking
element 0.

**A silently wrong mass matrix is the worst failure mode available**: no
exception, no `:unsupported`, no divergence, just a slower-mixing chain that
looks fine. MODEL_SCALING Result 6 read this from source and noted it never
fires *in that benchmark* because the vectorised likelihood raises in the
emitter first. That is a benchmark-specific accident, not a guarantee. This
needs a guard before anything else vectorises.

**c. The Model-S models are paying 20× for nothing, and the validation corpus is
the cleanest demonstration.**
`benchmark/posteriordb/validate_posteriordb.exs:150-169` declares `n_beta`
separate `beta_#{j}` scalar RVs, and `:200-205` builds the linear predictor as
an `Enum.reduce` accumulating `beta_j * x_col_j` — `X·β` written as `k`
broadcast-multiply-adds instead of one `Nx.dot`. The reliability model
(`benchmark/reliability_model.exs:44-48,72-116`, 44,000 elements) and the poker
model (`lib/exmc/poker/opponent_model.ex:58-64,77-104`) do the same thing with
`Enum.reduce` in *both* the RV block and the likelihood. **The corpus that
produces this project's headline correctness claim is also the best available
before/after demonstration of the graph-shape penalty, on every backend.**

**d. The level-set model is the one with genuinely GPU-shaped *compute*, and it
does not run on the GPU.** Its `d × n_obs` is small (520) because it has only
8 sensors, but that undersells it: the forward model is **50 Jacobi iterations
over the full 2-D grid** (`lib/exmc/physics/heat_2d.ex:42`), so the tensor
traffic is `n × 50` with a dispatch count constant in `n` — the exact profile
the crossover rewards, and the notebook itself proposes 16×16 and 32×32 (lines
510–511) as the interesting scales. **It is issue #1 in
`docs/VULKAN_KNOWN_ISSUES.md`**: three `Heat2D.solve` tests fail under
`EXMC_COMPILER=vulkan` with a `reduce_scalar` `ArgumentError` on a cross-module
`ResourceArc`. The best GPU demo this repository owns is currently a skipped
test.

**e. Depth in parameters and depth in data are the same knob.** `d` and `n_obs`
multiply; equal products give equal times on both arms (Result 1). `n_obs` moves
further in practice only because it is a data decision rather than a modelling
one. A user who wants eXMC to benefit from GPU compute should be told: write it
vectorised (worth 20× on every backend), bring more data, and use EXLA if you
have it.

**f. One reference path does not support vector RVs at all.**
`Exmc.LogProb.eval/2` (`lib/exmc/log_prob.ex:40-47`) returns `dist.logpdf/2`
unsummed and `sum_logps/1` (`:164`) is a bare `Nx.add`, so two vector RVs of
different sizes broadcast-fail. The JIT path is correct — `lib/exmc/compiler.ex:294`
sums across data dims, with the reason documented at `:290-293` — and is the one
the samplers use. Worth a guard or a docstring before someone reaches for
`LogProb` as the "simple" reference.

---

## 6. Four places the shipped story and the evidence disagree

### 6.1 Two correctness defects are shipping in 0.3.0 — this is the emergency

**Defect A — invalid doublings merged into the trajectory.**
Fixed upstream in `ce5775430` (2026-08-15, *"fix(nuts): an invalid doubling was
still merged into the trajectory"*). Stan's `base_nuts::transition` does
`if (!valid_subtree) break;` **before** the progressive-sampling step;
`Tree.do_build/11` calls `merge_trajectories/6` unconditionally, so states
beyond a U-turn still enter `combined_log_weight` and can still be drawn as the
proposal. `build_subtree/10` has the same hole one level down: it guards the
*left* half and merges the right half whatever it is.

Present in this repository in all four places:

| site | file:line | state |
|---|---|---|
| Elixir `do_build/11` | `lib/exmc/nuts/tree.ex:485-501` | `merge_trajectories/6` called unconditionally — no guard |
| Elixir `build_subtree/10` | `lib/exmc/nuts/tree.ex:1469` (left guarded), `:1494` (right merged unguarded) | defect present |
| Rust `build_subtree` | `native/exmc_tree/src/tree.rs:34-40` | left guarded, right merged unguarded |
| Rust `build_full_tree` | `native/exmc_tree/src/tree.rs:322` | `merge_into_trajectory` called unconditionally |

Measured upstream, 500 warmup + 2000 samples, seeds 1–6 pooled, against
analytic moments:

| | truth | before | after |
|---|---:|---|---|
| `Normal(0,1)` variance | 1.0 | **1.2200 (+22.0%)** | 0.9914 (−0.9%) |
| `HalfNormal(1)` variance | 0.363380 | 0.4091 (+12.6%) | 0.3572 (−1.7%) |
| `Exponential(2)` variance | 0.25 | 0.2754 (+10.2%) | 0.2531 (+1.2%) |

**Defect B — `logp_chain[k]` described the state *before* step k.** Fixed
upstream in `2a1b6b4eb` (2026-08-15). The synthesised chain shader's template
emitted the log-density body *above* the position update, so `logp_chain[k]`
carried `log p(q_chain[k-1])` while `q_chain[k]`, `p_chain[k]` and
`grad_chain[k]` were all post-step. `Tree.synth_chain_subtree/10` pairs the four
by index, so every NUTS leaf carried a stale, systematically-too-high density;
the multinomial then over-weighted the far end of each trajectory. Mis-scaled
rather than mis-signed, so it produced **no divergences** and adaptation simply
pushed `eps` up.

Present here: `lib/exmc/nuts/custom_synth/multi_rv_custom_spec.ex:103` puts
`{{prior_logp_body_q}}` in the pre-update block, with substitutions at `:407`,
`:482` and `:1191`. The upstream fix renames it to `{{prior_logp_body_qn}}` and
moves it below the position update, in both `@template` and `@batched_template`.

Upstream's own comment on this one is worth reading before you plan anything
around the Vulkan path:

> *This read as "Ampere over-dispersion" for three weeks (Normal(0,1) posterior
> variance 8.55 against a CPU reference's 1.45) and was blamed on the GPU. It is
> not hardware: both Keplers and the Ampere produce bit-identical q/p/grad from
> this shader.*

**Defect B is in the feature 0.3.0 is named after.** The release headline is
*"A Posterior on Any GPU"*; the synthesised chain shader is the mechanism that
makes GPU sampling fast; and its posterior variance was 8.55 against a true 1.0.
That is not a performance caveat, it is a wrong answer, and it is on
`origin/main` and in the CHANGELOG right now.

**Do not backport Defect B by overwriting the file.** OSS `2de6c60` added a
`config :exmc, :glsl_cse` toggle (`cse_loop_body/1` → `do_cse_loop_body/1`) that
the private repo does not have. Merge by hand.

### 6.2 The README does not describe the software

`README.md` contains **zero occurrences of "Vulkan"** and was last touched in
`f1bacf6` (2026-04-02). Its backends table lists EXLA, EMLX ("Planned") and
BinaryBackend. Release 0.3.0 (2026-07-23) *added* Vulkan and *dropped* EMLX.
So the front page of an OSS project simultaneously omits its headline feature
and advertises one that was removed.

This is a gift disguised as a bug: **the README does not currently oversell GPU
speedup, because it does not mention GPU at all.** The overselling risk is
entirely in front of us, in whatever paragraph gets written next. Write it with
the `nx_vulkan` banner already in hand, not after someone benchmarks us.

The performance section that *is* there ("Beats PyMC on 4 of 7 benchmarks",
repeated in `mix.exs:16` as the hex package description) is a fair
compiled-against-compiled comparison and should stay. Do not put a
Vulkan-vs-`BinaryBackend` multiplier next to it; the two are not the same kind
of number and printing them adjacently is how a project loses an argument on
Hacker News.

### 6.3 "33/33 posteriordb" cannot detect the defects in §6.1

`benchmark/posteriordb/validate_posteriordb.exs:364` defines the pass criterion:

```elixir
pass = mean_err < 0.5 and sd_ratio > 0.5 and sd_ratio < 2.0
```

A factor of 2 in SD is a **factor of 4 in variance**. Defect A produces a factor
of ~1.22 in variance and Defect B a factor of ~8.55 in a case where the CPU
reference was itself at 1.45 (SD ratio 2.9). Only the second would trip this
gate, and only if it were run on the Vulkan arm — the recorded run
(`benchmark/posteriordb/validation_results.md`, 2026-02-19) is EXLA-GPU per the
0.3.0 CHANGELOG entry. **The chain-shader path has never been validated against
posteriordb at all.**

The recorded numbers are also worth a second look. All 33 models report
`Max SD Ratio` between **1.038 and 1.490, and not one below 1.0**. Be careful
with that: the column is a *maximum over parameters*, so it is biased above 1 by
construction and this is **not** proof of systematic over-dispersion. What it is
proof of is that **the harness does not report the statistic that would settle
the question.** Re-run recording the signed distribution of `sd_ratio` — min,
median, max, and the sign test across all parameters of all models — and the
answer becomes readable in one table.

The unit tests are no better. `test/integration_test.exs:29` asserts

```elixir
assert_in_delta mu_stats.std, :math.sqrt(0.99), 0.5
```

Defect A's `Normal(0,1)` gives σ = √1.22 = 1.10 against an expected 0.995. The
assertion passes with 0.39 to spare. **A tolerance wide enough to accept a 22%
variance error is not a correctness test**, and it is exactly the class of thing
VERIFICATION_METHODS ranks as job #1.

### 6.4 The documented `d ≤ 256` cap is wrong, and it has already misled planning

Twelve sites (listed in §4, finding 4) claim or guard on `d <= 256`. The real
limit is 13 — six with `Normal` priors. The guards are unreachable, the docs are
wrong, and at least one task brief has already been written from the wrong
number (`nx_vulkan/docs/TODO_CHAIN_SHADER_BUGS.md`, Bug 2: *"it has already
misled planning… including a task brief written from it this week"*).
It is a one-line-per-site fix and should be done in the same commit as
the P0 backport, because until it is, every capacity discussion about the chain
shader starts from a false premise.

---

## 7. The plan, ranked by value over effort

Ranking is by defects prevented or wall-clock recovered per day of work.
"Value" and "effort" are stated so a future reader can disagree with the
ranking rather than the conclusion.

### P0 — correctness. Nothing else ships until these do. (~2 days)

| # | item | effort | value |
|---:|---|---|---|
| 1 | **Backport `ce5775430`** — all four `!valid_subtree` guards (`tree.ex` `do_build/11` + `build_subtree/10`; `tree.rs` `build_subtree` + `build_full_tree`). The OSS regions are byte-identical to the private pre-fix state, so this is a clean cherry-pick. | half a day | removes a detailed-balance violation from a released library |
| 2 | **Backport `2a1b6b4eb` by hand** — `multi_rv_custom_spec.ex` `logp_chain` ordering, both `@template` and `@batched_template`. Preserve the OSS-only `glsl_cse` toggle. | half a day | removes an 8.55× variance error from the feature 0.3.0 is named after |
| 3 | **Regression tests that fail without 1 and 2.** Analytic-moment checks on `Normal(0,1)`, `HalfNormal(1)`, `Exponential(2)` with tolerances derived from the sampler's own ESS, not from a round number. Port `bench/nuts_truth.exs` (35 lines) and `bench/validator_three.exs` (19 lines) from private. | half a day | the defects came back once already; they will come back again |
| 4 | **Fix `test/integration_test.exs:29`** and audit every `assert_in_delta` in the suite for tolerances that would accept a 20% variance error. This is VERIFICATION_METHODS' rank-1 item ("repair the gates that already exist"). | half a day | the difference between having tests and having a habit |
| 5 | **Correct `d ≤ 256` → the real cap** at all twelve sites, next to the `push_too_large` handling that explains it. | one hour | stops the next plan being written from a false number |
| 6 | **Release 0.3.1** with 1–5 and a CHANGELOG entry that says plainly what was wrong. | — | an OSS project that quietly fixes a posterior bug has spent its credibility for nothing; one that announces it has bought some |

### P1 — verification as a deliverable. (~1 week)

An OSS project has a stronger claim on correctness infrastructure than a private
one does: it is the thing users evaluate, and it is the thing a PPL is *for*.
This is the section that differentiates eXMC from a hobby sampler.

| # | item | effort | value |
|---:|---|---|---|
| 7 | **Backport `Exmc.NUTS.Vulkan.Validator`'s analytic machinery** — `check_analytic/3`, `analytic_moments/1` (normal, exponential, halfnormal, lognormal, studentt, cauchy), `ess/1` (Geyer), `variance_se/2` (private `validator.ex` is 744 lines to OSS's 410; grep for any of these in OSS returns zero). Plus `test/exmc/nuts/vulkan/validator_test.exs` (349 lines), which contains explicit inflated-variance and shifted-mean negative controls. | 1–2 days | this is the machinery that *found* both P0 defects |
| 8 | **Integrator invariants** (VERIFICATION_METHODS rank 3). Copy Stan's `expl_leapfrog2_test.cpp` symplecticness test: a circle of 1000 points in (q,p) of radius 1.5, assert the mapped area is πr² to 1e-2. ~40 lines, **exact assertions, zero flakiness**. | half a day | best value/effort ratio in the entire document |
| 9 | **Geweke joint distribution test** (rank 2), Blang variant — fresh prior draw per replicate, apply the kernel K times, two-sample KS. Needs three additions: `simulate_from_prior/2`; a `with_data/2` mirroring `with_multi_step_fn/2` so replicates do not recompile; a public single transition (`Sampler.nuts_step_with_stats/8` is currently `defp`). **The kernel under test must have adaptation frozen** — dual averaging and mass-matrix estimation are not Markov kernels, and a Geweke run with adaptation on will fail for a correct sampler. Prototype cheaply first: `Sampler.sample_compiled/3` accepts a plain tuple and never touches the IR, so a hand-written conjugate `vag_fn` gets you a first result without the API work. | 2–3 days | 100% power at 10⁴ iterations; 40× cheaper than SBC for the same job |
| 10 | **Conjugate exact-posterior tests** (rank 4) and a finite-difference gradient check (rank 7). | 1 day | cheap, and they close the two classes Geweke does not |
| 11 | **Backport `docs/VERIFICATION_METHODS.md`** with a scrub pass. It names private bench files, internal host names (super-io, mac-247/248) and private docs. Scrubbed, it is a genuinely good public artefact — a ranked literature review of how to check an MCMC implementation is the kind of document an OSS PPL should be known for. | half a day | positioning, and it documents the plan for whoever comes after |

**Deliberately not in P1: SBC.** Rank 5, 2–3 days plus 400 complete sampler
fits for 76.5% power where Geweke gets 100% from 10⁴ iterations of one chain.
Revisit only after Geweke is green.

### P2 — the highest-value performance work, and it is not GPU work. (~1 week)

| # | item | effort | value |
|---:|---|---|---|
| 12 | **Guard `CustomSynth` against vector RVs — do this first.** `extract_components/1` (`lib/exmc/nuts/custom_synth.ex:326`) builds `layout` one slot per RV *id* while `PointMap` gives a `shape: {n}` RV `n` slots; `standard_rv_node?/1` (`:334-338`) has no shape check; and `pad_or_truncate(list, d, 1.0)` (`lib/exmc/nuts/sampler.ex:1531`) **silently absorbs** the mismatch into a wrong inverse-mass vector. Add the shape guard so the path returns `{:unsupported, _}` instead. | half a day | closes a silent-wrong-answer path that every other P2 item makes more reachable |
| 13 | **Vectorise the posteriordb harness.** Rewrite `validate_posteriordb.exs:150-205` to declare one `shape: {n_beta}` β and compute `X·β` with `Nx.dot`. Re-run all 33 and record wall time before/after alongside the correctness columns. | half a day | the cleanest measurement of the 20× graph-shape effect on a real corpus, on every backend |
| 14 | **Vectorise the two `Enum.reduce` models in shipped code** — `Exmc.Poker.OpponentModel` (`lib/exmc/poker/opponent_model.ex:58-64,77-104`, `shape: {num_players}`) and `benchmark/reliability_model.exs:44-48,72-116` (`shape: {n_types}`; 44,000 elements, the largest Model-S workload here). Both unroll in *both* the RV block and the likelihood. | 1–2 days | worked examples users can copy, and the second is above the crossover |
| 15 | **Port `model_scaling.exs` into `bench/`** as the standing width benchmark. The harness at `nx_vulkan/bench_results/model_scaling/model_scaling.exs` is fully synthetic — `Exmc.Builder`, `HalfCauchy`, `Custom` only, no trading code — so it drops in unchanged. **Require an EXLA arm.** Every eXMC benchmark in these repos so far has compared against `BinaryBackend`, which flatters the GPU by one to two orders of magnitude. | half a day | makes future performance claims falsifiable by construction |
| 16 | **Document vector RVs as the recommended form.** A `docs/VECTORISING_MODELS.md` with the S-vs-V numbers, the `PointMap.unpack` mechanism (§4 finding 3), a before/after rewrite, and the note that the JIT path handles vector RVs end to end while `Exmc.LogProb.eval/2` does not (§5.2f). Add to `mix.exs` `docs: extras`. | half a day | the highest-leverage thing a user can learn from us |

### P3 — reach, honestly stated. (~3 days)

| # | item | effort | value |
|---:|---|---|---|
| 17 | **Rewrite the README for 0.3.0.** Vulkan in the backends table; EMLX removed; a backend-selection section framed as a *reach hierarchy* (EXLA where it exists → Vulkan where it does not → BinaryBackend as reference); and the `nx_vulkan` honesty banner adapted: *most GPU multipliers in these repos are against a pure-Elixir interpreter; on a host with EXLA, use EXLA.* Cite MODEL_SCALING. | 1 day | §6.2; this is a credibility item, not a documentation item |
| 18 | **Make the FreeBSD case properly, with numbers from FreeBSD.** Every measurement in MODEL_SCALING is Ampere. The Keplers have different f64 rates, different bandwidth and no EXLA, and the report explicitly says its conclusions are *not* transferable there without measurement. Run the width sweep on mac-247, publish it, and *that* is the Vulkan case — a portability argument backed by the hardware it applies to. | 1 day + fleet time | turns the strongest available claim from an assertion into a measurement |
| 19 | **Fix `VULKAN_KNOWN_ISSUES.md` #1** — the `reduce_scalar` cross-module `ResourceArc` failure that takes out `Heat2D.solve` and the level-set tests — then run the level set at 16×16 and 32×32 as the GPU demo it was always going to be. | 1 day | unblocks the best GPU case this repo owns (§5.2d) |
| 20 | **Fix or delete the two broken notebooks.** `notebooks/09_radon_bhm.livemd` and `notebooks/11_insurance_claims.livemd` `Code.require_file` builders in `benchmark/` that exist neither in the working tree nor in git history. `test/notebooks_test.exs` does not catch this. A notebook that cannot run is worse than an absent one in a repo whose notebooks *are* the documentation. | half a day | the first thing a new user runs |

### P4 — decide the chain shader's future. (~2 days, or zero)

| # | item | effort | value |
|---:|---|---|---|
| 21 | **File and fix or retire `:nif_panicked` at n_obs = 600.** Documented in `nx_vulkan/docs/TODO_CHAIN_SHADER_BUGS.md` Bug 1; the panic is Rust-side, so it is a bounds or size computation — candidates in order: push-constant `n_obs` versus the observation SSBO length; dispatch geometry (`d` workgroups × 256 threads with a `q_shared[256]` tile against a loop bounded by `pc.n_obs`); descriptor-set buffer sizing. **Done when it either dispatches correctly at n_obs in the thousands or returns `{:unsupported, _}` the way `push_too_large` already does.** A graceful refusal is acceptable; a panic in a NIF takes down more than the caller and is not. | 1–2 days | shipping a sampler that crashes at 10× the observations is worse than not shipping it |
| 22 | **Then decide.** After P0 item 2 and P2 item 15, re-measure the chain shader against per-op *and* EXLA. If it is still 3.2× slower than `BinaryBackend` at the only width it is allowed, retire it behind a config flag and say so in the CHANGELOG. Keeping a synthesis path that costs two minutes to compile, caps models at d ≤ 6 with `Normal` priors, and loses to the interpreter is a maintenance liability with no measured upside. | half a day | the decision is cheap; carrying it undecided is not |

### P5 — general code the private repo has and this one does not

Only after P0–P2. Both are genuinely general (no trading content) and both have
tests upstream.

- `lib/exmc/smc/` — `particle_filter.ex`, `pmcmc.ex`, `online_smc2.ex`, plus
  `test/particle_filter_test.exs` and `test/online_smc2_test.exs`. Note the OSS
  repo already ships `docs/SMC2_OPTIMIZATION.md` and `test/smc_test.exs` but no
  `lib/exmc/smc/`, which is an inconsistency worth closing either way. Consider
  whether this belongs here at all given `smc_ex` exists as a sibling library —
  the README's "Three Comrades" framing says SMC lives there.
- `lib/exmc/mesh/` — `worker.ex`, `capability.ex`, `supervisor.ex`,
  `dispatcher.ex`, `pool.ex`, plus `test/mesh/`. Distributed inference mesh;
  fits goal 2 directly.

**Forward-port (OSS → private), not backport:**
`lib/exmc/nuts/vulkan/scheduler.ex` + `docs/SCHEDULER_PINNING.md`.

**Never backport:** `lib/exmc/trading/` (48 files: `ibkr.ex`, `alpaca.ex`,
`oanda.ex`, `risk_manager.ex`, `regime_model.ex`, `news_signal/*`, …),
`lib/exmc/trading.ex`, `lib/exmc/license*`, `lib/exmc/application.ex` as-is
(it pulls in trading and licensing; this repo is deliberately library-shaped
with no supervision tree), all of `test/trading*` and `test/license*`, and all
of `research/`.

**A note on `research/D90_BACKLOG_FIX_PLAN.md`:** its "OSS backport status"
paragraph (lines 64–68) lists `80ec0bb`, `9381119`, `97b19f2` and `323c753` as
committed locally but *"not yet pushed to public GitHub."* **That paragraph is
stale** — all four are ancestors of `origin/main` here, merged via `fa012a1` and
shipped in `e733598`. The real queue is everything dated 2026-08-15/16, which
that document predates. Do not plan from it.

---

## 8. What not to do, and why

- **Do not write another shader before P2 lands.** Model S vs Model V is 16× at
  d=64 with identical arithmetic (MODEL_SCALING Result 5). Any kernel you write
  is optimising a graph that has 20× too many ops in it. Vectorise first; the
  kernel may turn out to be unnecessary.

- **Do not invest in `Nx.Vulkan.Compiler` fusion on eXMC's behalf.** It is
  within noise of per-op on 13 of 13 cells, on exactly the elementwise-heavy
  graph it was designed for, and the previous explanation (137 host fallbacks
  below the compiler) is dead — there are zero fallbacks now and the result is
  unchanged (Result 2). The cause is unidentified. Until `nx_vulkan` diagnoses
  it, "fusion will fix the per-dispatch cost" is not a claim this repository can
  make, and building on it is building on a negative result.

- **Do not chase f32.** Worth nothing below 10⁵ elements on either arm and ~2×
  above it (Result 3). There is no crossover at f32 that does not already exist
  at f64. The f64 default is not costing anything and it is a correctness asset.

- **Do not add distributions.** 21 cover every shipped notebook. Each new one
  adds a `logpdf`, a transform, a chain-shader prior encoder, a push-constant
  budget entry and a test. VERIFICATION_METHODS lists "add more distributions to
  the differential harness" in its explicit *do-not* section. If a user needs
  one, `Exmc.Dist.Custom` takes a closure.

- **Do not implement SBC yet, and do not implement Cook–Gelman–Rubin at all.**
  Ranked 5 and explicitly excluded respectively. Geweke gets 100% power on the
  known defect class for ~1/40th of the sampler work.

- **Do not publish a Vulkan-vs-`BinaryBackend` speedup table.** Not in the
  README, not in a blog post, not in a conference talk. It is a real number
  measuring a real thing (reach on hardware with no compiler) and it will be
  read as a claim about GPUs versus CPUs, which it is not. Publish
  Vulkan-vs-EXLA-vs-`BinaryBackend` as three columns, or publish nothing.

- **Do not widen a tolerance to make a test pass.** That is how §6.3 happened.
  If a test is flaky, fix the statistic — divide by ESS, use the sampler's own
  variance estimate — not the threshold.

- **Do not touch `/home/io/projects/learn_erl/pymc/exmc/` or
  `/home/io/projects/learn_erl/nx_vulkan/` from this repository.** Read them;
  cherry-pick from the first; file issues against the second.

---

## 9. Reproducing every number in this document

```sh
# The width sweep this plan is built on (run from an eXMC tree with nx_vulkan)
cp /home/io/projects/learn_erl/nx_vulkan/bench_results/model_scaling/model_scaling.exs bench/
SWEEP_MODE=grad MODELS=S,V DIMS=8,16,32,64,128,256 NOBS=60 \
  ARMS=cpu,perop,fused REPS=5 mix run bench/model_scaling.exs

# The n_obs axis, where the crossover reproduces cleanly
SWEEP_MODE=grad MODELS=V DIMS=8 NOBS=60,600,6000,60000 ARMS=cpu,perop \
  REPS=5 mix run bench/model_scaling.exs

# The chain-shader envelope (slow: ~2 min per cell at d=12)
SWEEP_MODE=synth MODELS=S DIMS=2,4,6,8,10,12,13,14 mix run bench/model_scaling.exs

# The n_obs=600 NIF panic
SWEEP_MODE=nuts MODELS=S DIMS=8 NOBS=60,600 ARMS=cpu,chain \
  WARMUP=25 SAMPLES=25 mix run bench/model_scaling.exs

# The EXLA reference — needs no eXMC and no nx_vulkan, only a working EXLA NIF
mix run /home/io/projects/learn_erl/nx_vulkan/bench_results/model_scaling/exla_ref.exs

# posteriordb, and the element counts in §5.1
sh benchmark/posteriordb/run_validation.sh
```

Source citations in §4–§6 are `file:line` against `main` @ `2de6c60` and will
drift. If a line number does not match, grep for the quoted code; the claims are
about the code, not the coordinates.

Primary references:

- `nx_vulkan/bench_results/MODEL_SCALING.md` — the width sweep. Read Results 4,
  5 and 6 before proposing any GPU work.
- `nx_vulkan/bench_results/EXMC_PEROP_RACE.md` — the earlier race MODEL_SCALING
  supersedes. Carries a superseding banner; do not quote it standalone.
- `nx_vulkan/docs/TODO_CHAIN_SHADER_BUGS.md` — the two chain-shader bugs, filed.
- `nx_vulkan/README.md` § Benchmarks — the honesty banner this repo should mirror.
- `pymc/exmc/docs/VERIFICATION_METHODS.md` — the ranked verification plan (P1).
- `DECISIONS.md` — 87 architectural decisions with rationale. **Append to it.**

---

## 10. Working agreements

- **New decisions go in `DECISIONS.md`**, numbered, with rationale and
  implications. It is at 87 entries and it is the reason this repository can be
  picked up cold. Items 1, 2, 12, 17 and 22 above all warrant entries.
- **`main` tracks `origin/main`.** Work on a branch; the operator pushes.
- **`bench_results/` should exist here.** It does not yet; `bench/` has one
  file. Every performance claim in the README should point at a file in it that
  contains the raw output and the host it ran on.
- **Contamination gets reported.** MODEL_SCALING documents a foreign `mix test`
  holding the GPU for 17 minutes mid-run and reports the affected cells as
  ranges rather than points. That is the standard.
- **The reference arm is always `BinaryBackend` for correctness and EXLA for
  speed.** They are different jobs and using one for the other is how §6.2
  becomes a problem.

---

## Appendix — the one-paragraph version

eXMC is a PPL for the BEAM whose distinguishing features are concurrency,
streaming, fault isolation and distribution, and whose GPU support is about
reaching hardware EXLA cannot rather than beating hardware it can. Its models
*are* deep enough for the GPU to pay against the interpreter — seven runnable
models sit between 1.4×10⁴ and 6.8×10⁴ likelihood elements against a ~10³
crossover — but the three deepest vectorised ones are silently mis-handled by
the fused GPU sampling path, the deepest Model-S ones are paying 20× for their
graph shape on every backend, and the model with the most GPU-shaped compute is
a skipped test. Meanwhile two correctness defects fixed upstream on 2026-08-15
are shipping here in 0.3.0 — one of them in the feature the release is named
after, where the `Normal(0,1)` posterior variance was 8.55 against a true 1.0 —
and no test in this repository is tight enough to catch either. **So: fix the
posteriors, build the verification harness that would have caught them,
vectorise the models, and describe the GPU story as portability. Then, if there
is still a shader worth writing, write it.**
