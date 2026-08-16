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

**The one exception:** `exla` here is a CUDA build whose NIF cannot load
(`libnvshmem_host.so.3` is absent machine-wide) and it does not rebuild from
source either (`runtime_callback_cuda.o` fails against the installed g++).
Deleting `_build/` does not fix that; it just re-fetches the same broken thing.
See §2, item 1.

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

The case for waiting: `compiler: :vulkan` is the **default**, and it is still
not correct for models with observations (§2). Publishing a release whose
headline is "correctness" while the default backend returns a frozen chain for
a whole model class is a second credibility problem, not a fix for the first.

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
| 3 | **The EXLA build.** Either install a CPU EXLA (`XLA_TARGET=cpu mix deps.compile exla --force`) or stop the test env requiring an optional dep to be startable. The latter is arguably a real bug in this library: advertising `optional: true` and then failing the whole suite when the optional dep is present-but-broken. | 1–2 hours | you cannot run `mix test` at all without working around it |
| 4 | **`test/integration_test.exs`: 26 tests, 4 failures.** One is item 1 and is **red on purpose**. Two raise `SynthUnsupportedError` because the default compiler refuses non-synthesisable models — likely the *tests* should pick a compiler, but check whether the guard is too aggressive first. One asserts wall-clock (`6198ms should be < 1757ms`) and belongs in `bench/`. | half a day | a permanently-red suite trains people to ignore red |

### Do not skip the red test

`integration_test.exs:611` fails for a real reason and is annotated to say so.
Skipping a test that fails for a real reason is precisely the habit that let
both 0.3.1 defects ship. If you fix item 1, it goes green on its own.

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
