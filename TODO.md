# TODO — handed over from the applications-tree session

**Written 2026-08-20.** Two sessions have been working in this checkout at
once. On 2026-08-17 at 23:39 I found `lib/exmc/nuts/tree.ex` mid-edit —
`with_supervision/6` called at `:437` and `:466` and defined nowhere — while I
was committing B1 and C2. That edit has since landed as `bd86e96` and the tree
compiles, so nothing is broken. This file is the other half of that collision:
what I know that is not in `NEXT.md`, and what I touched.

`NEXT.md` remains the plan of record for the OSS repo. This is a list of loose
ends, ordered by how much it costs to leave them.

---

## 0. ~~There is one uncommitted file and it is mine~~ — closed

**Decided and committed 2026-08-20 as `92afdee`.** Nothing is outstanding here;
the section is kept because the reasoning is the durable part. The working tree
is clean.

`test/optional_deps_test.exs`.

`b536a40` dropped `exla` from the dependency list on FreeBSD, because the `xla`
archive ships darwin and linux-gnu targets only and `mix compile` dies inside
the dependency before reaching a module of this library. That broke the
existing assertion at `:30`, `assert Code.ensure_loaded?(EXLA)`, which is true
on Linux and cannot be true on the fleet.

The uncommitted diff splits it on `:os.type()`: off FreeBSD it keeps asserting
exla is loadable; on FreeBSD it asserts the inverse — that exla is **absent** —
so that the next person to "fix" that conditional in `mix.exs` finds out from a
red test rather than from a Kepler.

One change on the way in: the failure message originally pointed at
`docs/GATE1_RECONCILIATION.md F4`, which lives in the applications tree and is
untracked in either repo (§3). An OSS test must not cite a document nobody
outside that checkout can open, so it now names the `@freebsd?` conditional in
`mix.exs` and `b536a40`, both of which are here. 4 tests, 0 failures on Linux;
the FreeBSD arm is exercised on the fleet by construction.

---

## 1. A live defect, reproduced: `log1p` and `expm1` raise

`lib/exmc/nuts/custom_synth/eval.ex:124-125`

```elixir
log1p: &:math.log1p/1,
expm1: &:math.expm1/1,
```

**Neither function exists.** Erlang's `:math` has no `log1p` and no `expm1` —
checked directly on this OTP, both `function_exported/3` return `false`. The
capture itself is legal, so this is a compile *warning*, not an error; the
`UndefinedFunctionError` waits until something calls it.

Reproduced end to end:

```
--- log1p ---        {:raised, UndefinedFunctionError, ...}
--- reference Nx --- 0.4054651081081644
--- expm1 ---        {:raised, UndefinedFunctionError, ...}
--- control: log --- {:ok, -0.6931471805599453}
```

This matters more than a dead branch in a validation walker would, because
`Nx.log1p/1` and `Nx.expm1/1` are what the softplus and logit inverse
transforms are built from — `transform.ex:295`, `point_map.ex:175,177`,
`compiler.ex:517,519`, `log_prob.ex:168`, `model_comparison.ex:247,248`. Any
model with a positive-constrained or bounded RV emits those ops. `Eval` is the
faithfulness check that `CustomSynth`'s GLSL walker is structurally correct
(see its `@moduledoc`), so this is a hole in a verification tool, on exactly
the node types a constrained model produces.

The fix is two expressions, `log1p(x) = log(1 + x)` and `expm1(x) = exp(x) - 1`
— but write the numerically stable forms, not those, since stability at small
`x` is the entire reason the ops exist. Then add a test: this survived because
nothing ever evaluated one.

While you are in there, two more compile warnings that are not noise:

- `Nx.Vulkan.NativeV.leapfrog_chain_synth_batch_f64/6 is undefined` at
  `lib/exmc/nuts/vulkan/dispatch.ex:222`. The pinned `nx_vulkan` exports
  `_synth/6`, `_batch/6` and `_f64/6` but not batch-and-f64. `Dispatch.chain_batch/5`
  raises if it is ever reached on the f64 batched path.
- Two clauses of `defp stream_from_compiled/4` at `sampler.ex:1273` and `:1282`
  are unreachable. Either the arity-6 compile tuple made them dead, in which
  case delete them, or the dispatch above them is wrong.

---

## 2. B1 and C2 are committed; C2 has no `NEXT.md` section

```
6dcd59e  feat(mclmc): microcanonical Langevin (MCLMC) and its Metropolised form (MAMS)
82db4f8  feat(sbi): likelihood-free inference — rejection ABC and ABC-SMC
```

6,724 insertions, 26 files. Verified green before committing: `mix test
test/sbi test/mclmc` → **97 tests, 0 failures, 2 excluded**, 226s.

Twelve of those files were not formatted; I ran `mix format` on them, and that
green run was taken *before* the format pass. **Re-run 2026-08-20, after it,
and both are still green:** `mix test test/sbi` → 55 tests, 0 failures, 2
excluded, 33.7s; `mix test test/mclmc` → 42 tests, 0 failures, 267.4s. The
caveat is closed.

The 2 excluded are `sbc_test.exs:183` and `:213`, both `@tag :slow` — see
`NEXT.md` §7, they are the tests that measure the SBC gate's own error rates.

~~There is no §7 for C2.~~ **Written 2026-08-20** — `NEXT.md` §7 now covers it.
Three things in it are new findings rather than restatement, and are the reason
it was worth writing:

- **C2.4 was never started**, and it is the stage the roadmap item is named
  after — "simulation-based inference *on `sim_ex`*". There is no bridge, no
  notebook, and no reference to `sim_ex` anywhere under `lib/exmc/sbi/` or
  `test/sbi/`. What landed is a correct, tested ABC library with no consumer,
  and the roadmap's "abandon if" — does the simulator call dominate so badly
  that useful particle counts are unreachable — is therefore still unanswered.
- **C2.3 is half done.** `Task.async_stream` landed; `Mesh.Pool` did not,
  because `lib/exmc/mesh/` does not exist in this repository. It is one of the
  private-only subtrees. The distributed arm is blocked on the split, not on
  ABC.
- **The two tests that establish the SBC gate's error rates are excluded from
  the default run** (`@tag :slow`, `sbc_test.exs:183` and `:213`). So `mix test`
  re-checks the gate but never re-checks the gate's credibility, and nothing in
  CI would notice those numbers drifting.

---

## 3. The OSS/applications split, and a reconciliation item that is wrong

The three plan documents governing this repo live **in the other tree and are
uncommitted**:

```
/home/io/projects/learn_erl/pymc/exmc/docs/PLAN_CORE_AND_APPLICATIONS.md
/home/io/projects/learn_erl/pymc/exmc/docs/PLAN_SAMPLER_ROADMAP.md
/home/io/projects/learn_erl/pymc/exmc/docs/GATE1_RECONCILIATION.md
```

Untracked, in either repo, since 2026-08-16. `NEXT.md` §6 already cites the
second one by absolute path. That is a bad arrangement and it should be fixed
in whichever direction the operator prefers, but it should not stay as it is.

Gate 1's actual reconciliation — 20 differing core files, ~950 diff lines —
is **surveyed, not applied**. Only `b536a40` landed from it.

### Batch D is wrong, and it is inverted

`GATE1_RECONCILIATION.md` Batch D says OSS `nuts/vulkan/validator.ex` is 410
lines against private's 744, and that it lacks `check_analytic/3`,
`analytic_moments/1`, `ess/1` and `variance_se/2` — repeating `MISSION.md` §7
P1 item 7's claim that "grep for any of these in OSS returns zero". Measured
today:

| | OSS | private |
|---|---:|---:|
| lines | **708** | 744 |
| `def` + `defp` heads | 26 | 26 |
| function names in common | 24 | 24 |

The whole delta is two names each way. Private has `reference/_` and
`run_reference/_`; OSS has `run_exla/_` and `cauchy_quantiles/_`.

### …and that reading was wrong too — resolved 2026-08-20

**Counting names is what produced both errors.** The delta was never in the
function list, it was in the **bodies**, and `diff -u` on the two files is 624
lines. It runs in both directions and three of the private→OSS items are live
defects, not style:

1. OSS defined `check_analytic/3` and **never called it from `compare/3`**. So
   `validate/3` ran mean, variance and KS — all differential — and was
   structurally blind to any defect the two arms share, which is the whole
   reason that function exists. Its moduledoc said so while the code did not do
   it.
2. OSS's `check_mean/2`, `check_variance/2`, `check_median/2` and `check_iqr/2`
   divided by `length/1`, not `ess/1`.
3. OSS's `run_exla/2` cleared `:exmc, :compiler` and let auto-detect run —
   which resolves to `Nx.Vulkan` on any host without EXLA. **That is the
   self-comparison bug private already fixed**, still live in OSS, and the
   FreeBSD fleet is exactly the host class it bites.

The OSS→private direction was right as far as it went, and it was one item
short: private also still carried the `0.25 · IQR / sqrt(n)` Cauchy gate, which
is ~6× too tight and would fail a correct sampler.

**Both directions are now merged.** The trees are byte-identical on
`lib/exmc/nuts/vulkan/validator.ex` and on
`test/exmc/nuts/vulkan/validator_test.exs` — OSS had no
`test/exmc/nuts/vulkan/` directory at all, which is why its copy was free to
drift, and that is the part of Batch D that was accurate. `GATE1_RECONCILIATION.md`
Batch D and `MISSION.md` §7 P1 item 7 are corrected in place rather than
deleted, and the reasoning is filed as **D92**.

Still open from this section: the four other test files in private's
`test/exmc/nuts/vulkan/` (`batch_coordinator`, `bulkhead`, `chaos`, `server`)
have no OSS counterpart, and the plan-document arrangement above still needs an
operator decision.

---

## 4. B1.4 — the measurement that decides whether B1 was worth doing

`bench_results/MCLMC_BIAS.md` carries a `PARTIAL RUN` banner and it is honest.
Complete: all three targets at `d = 2`. Partial: `Normal(0,1)` at `d = 8`, with
every row except NUTS. Missing: `HalfNormal`/`Exponential` at `d = 8`, and
**all of `d = 32`** — which is the dimension the roadmap's "abandon if" is
actually about.

At `d = 8` MAMS reaches **1.6964** ESS/gradient and MCLMC (tuned) **1.5258**,
against 0.4169 and 0.2283 for the same two at `d = 2`. Whether that beats NUTS
is the entire case for B1, and the NUTS row is the one that did not finish.
**Do not quote that file as evidence about high dimensions.**

```sh
DIMS=2,8,32 SEEDS=1,2,3 WARMUP=1000 SAMPLES=3000 \
  EPS=0.1,0.25,0.5,1.0,2.0,4.0 OUT=bench_results/MCLMC_BIAS.md \
  mix run --no-deps-check bench/mclmc_bias.exs 2>&1 | tee /tmp/mclmc_bias.log
```

Budget 3–4 hours on `super-io` under load. **Fix the incremental write first**
— it is ten lines and it has already cost one run. `bench/mclmc_bias.exs`
builds the whole document in memory and writes `OUT` once at the very end, so a
kill loses the file; the current partial was recovered from stdout. Append each
block as it completes.

A related thing worth a profile of its own: NUTS costs **~8 ms per gradient**
on this host against MCLMC's **~0.5 ms** on the same model and backend —
measured at `compiler: :none`, 600 iterations, mean tree depth 2.1, 1106
gradient evaluations. A 15× per-gradient overhead in the tree machinery, not an
algorithmic difference, and invisible in any ESS-per-gradient table.

---

## 5. The f32 audit nobody has done

**A bare Elixir float in an `Nx` binary op computes at f32 even against an f64
tensor.** The scalar becomes a default-typed `{:f, 32}` and the promotion
widens *after* the arithmetic, so the result is f64-*typed* and f32-*accurate*.
Measured on `Nx.BinaryBackend`:

```
f64 tensor / bare elixir float:  {{:f, 64}, 1.0314463831230813}
f64 tensor / f64 tensor:         {{:f, 64}, 1.031446380705704}
difference:                      2.4173774093583233e-9
```

This cost real time in B1: the `‖u‖ = 1` invariant failed at 3e-8, which is f32
epsilon, and read exactly like an algebra error in a new integrator.
`Exmc.MCLMC.Integrator` now routes every scalar through a `c/2` helper that
builds it at the tensor's own type, and the note is in that module's source.

**The rest of the repository has not been audited.** `sampler.ex`,
`leapfrog.ex`, `tree.ex` and `mass_matrix.ex` all mix Elixir floats with
tensors. A one-line grep finds six sites and is a lower bound — it misses
multi-line calls and any literal bound to a variable first:

```sh
grep -nE 'Nx\.(add|subtract|multiply|divide|pow|remainder|max|min)\([^)]*[0-9]+\.[0-9]' \
  lib/exmc/nuts/{sampler,leapfrog,tree,mass_matrix}.ex
```

`MISSION.md` §4 says the f64 default "is not costing anything and it is a
correctness asset". That is only true where the default actually applies. Worth
doing before the next precision question gets diagnosed as a backend problem —
which is what happened here.

---

## 6. The tag exclusions key off the wrong thing

`test/test_helper.exs:131`

```elixir
case Application.get_env(:exmc, :compiler) do
  :vulkan -> ExUnit.configure(exclude: base ++ f64 ++ [:vulkan_known_failure])
  _       -> ExUnit.configure(exclude: base ++ [:requires_vulkan])
end
```

The exclusions read the **configured** compiler; the backend actually in use
comes from `Exmc.JIT.auto_detect/0`. On the FreeBSD fleet, where nothing sets
`:exmc, :compiler` and exla is not installed, those disagree: the run printed
`Excluding tags: [:diag, :slow, :requires_vulkan]` while `detect_compiler/0`
returned `Nx.Vulkan`. So `:requires_vulkan` tests were skipped *while on
Vulkan*, and `:vulkan_known_failure` tests ran *while on Vulkan*.

Key it off the detected backend, not the config.

---

## 7. Two notes on history, neither worth rewriting

- **`b536a40` carries more than its message says.** Its subject is
  `deps: FreeBSD is Vulkan-only` and its body describes only the dependency
  change, but its diff also registers `Exmc.MCLMC`, `Exmc.MAMS` and the
  "Inference internals" group in `mix.exs`'s ex_doc groups. B1's agent edited
  `mix.exs` and I swept it in without noticing. It is already on `origin`, so
  it stays; this is the note that says so. `82db4f8` adds the `Exmc.SBI`
  groups the same way, and does say so.
- **`gate1/reconcile-core` is pushed to `origin` through `b2aefca`.**
  *(Corrected 2026-08-20 by the other session. This bullet originally read
  "only through `b536a40`. Everything after it is local." That was measured
  before the P0 run was pushed and it is four commits stale — `6c1589a`,
  `765d86f`, `bd86e96` and `b2aefca` are all on `origin`. Acting on the old
  sentence, by force-pushing back to `b536a40` or by trying to "rescue" work
  believed to be local only, would discard the whole item 1 / 7 / 2 run. Check
  `git rev-parse origin/gate1/reconcile-core` rather than this file.)*
  Of the two local commits named there, only `0fe59d2` (this file) is the
  applications-tree session's — `9c74bb0 notebooks(bda-cyber)` is the other
  session's own, and its files were already untracked in the checkout before
  B1 and C2 were staged. The list has since grown either way; take
  `git rev-parse` over any sentence in here. `origin` is the private server;
  `upstream` is GitHub and pushing there is a release — see `NEXT.md` §0.

---

## Working in a shared checkout

If both of us are in here again: stage explicit paths, never `git add -A` or
`git commit -a`; check `git diff <file>` immediately before staging anything
you did not create; and do not `git checkout`, `stash` or `clean` the tree
without looking at what is modified first. Both of my commits are clean of the
crash-recovery work because I staged file lists, and that was the only thing
that made it safe.
