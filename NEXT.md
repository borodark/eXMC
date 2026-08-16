# NEXT — eXMC (open source)

**Written:** 2026-08-16, against `main` @ `6d6ae4f` (0.3.1, the P0 correctness
backport, merged).
**Read `MISSION.md` first** — this file assumes it and does not repeat it. This
one is only *what to do next and in what order*, with the state as it actually
stands rather than as the mission planned it.

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
| 4 | **`test/integration_test.exs`.** With EXLA working, `mix test` is now **372 tests, 1 failure** for the whole repo. The remaining one is the wall-clock assertion (`integration_test.exs:762`, now `vectorized=1078ms should be < parallel=575ms`) and belongs in `bench/`. The other three from the old count were artefacts of Vulkan-by-default and are green under EXLA — **not fixed, not exercised**; see item 5. | 1 hour | a permanently-red suite trains people to ignore red |
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

Two traps worth carrying forward, both of which produce a confident green:

- **ExUnit's line filter snaps backwards.** `mix test path:638` on a comment
  line runs the *previous* test, reports `1 test, 0 failures`, and looks like
  the test you meant. Give it the `test do` line — 639, not 638.
- The test is **differential** (scalar arm vs vector arm), so it can only see
  item 1 while the two arms disagree. It is the same structural blindness §3
  describes; it happens to work here only because the defect hits one arm.

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
