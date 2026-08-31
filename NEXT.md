# NEXT — eXMC (open source)

**Written:** 2026-08-16, against `main` @ `6d6ae4f` (0.3.1, the P0 correctness
backport, merged).
**Read `MISSION.md` first** — this file assumes it and does not repeat it. This
one is only *what to do next and in what order*, with the state as it actually
stands rather than as the mission planned it.

---

## Status — 2026-08-31, before the reboot

Written against `gate1/reconcile-core` @ `a178a0833`, **pushed to `origin`**.
Working tree clean. Nothing is in flight and nothing is half-applied — the
reboot can happen without losing state.

The session ran `nx_vulkan` `501fa08 -> 2617e5e` and then closed six defects.
Five were silent. The sixth had been on this file's own known-failures list
since 2026-08-16 as "item 4, the wall-clock one", carried for two weeks as a
scheduling artifact. It was not one.

| commit | what |
|---|---|
| `7d40ce49c` | `nx_vulkan` `501fa08 -> 2617e5e` |
| `0ff2d866c` | D3 + D1 — coordinator died on a raise; batched path packed 5 of 12 priors |
| `3e58fb70f` | Validator pinned an `:exla` reference arm it could not run, and leaked it |
| `237c0c3f8` | Vectorized (DEFAULT) multi-chain path never reached the chain shader |
| `edbeb2713` | D2 — batched push had no 128-byte cap |
| `a178a0833` | Distributed and streaming paths dropped `chain_meta` too |

### Verification as it actually stands

| run | result |
|---|---|
| `mix test` (EXLA), @ `a178a0833` | 652 tests, **0 failures**, 449s |
| `EXMC_COMPILER=vulkan mix test`, @ `a178a0833` | 652 tests, **2 failures**, 1217s |
| mac-247, @ `3e58fb70f` | 640 tests, **4 failures** |
| mac-248, @ `3e58fb70f` | 640 tests, **5 failures** |
| the Jetson, @ `ad464bce5` | 632 tests, **9 failures** — 7 of them ExUnit timeouts |

**The EXLA arm is fully green for the first time.** The Vulkan arm's two are
Poker and LevelSet, both 300s `ExUnit.TimeoutError` on this host, both
confirmed pre-existing: Poker was re-run alone against HEAD with the session's
changes reverted and timed out identically at 300.4s.

**No host has been verified at `a178a0833`.** Both Keplers were launched and
then killed partway when another session started
`examples/unified_vs_discrete_race.exs` on mac-247 — a GPU suite would have
corrupted a timing benchmark in both directions. The partial logs were
discarded rather than read. Both boxes are already ON `a178a0833`, clean, so
relaunching is just the suite. The race began at 22:08, two minutes after the
suite died at 22:05:56, so **that benchmark is not contaminated** — checked,
not assumed.

### Do this first

1. **Fleet-verify `a178a0833` on mac-247/248, once the race is done.**
   Expect 646 tests (six added) and the wall-clock failure gone — 4 -> 3 on
   247. If it does not drop, `237c0c3f8` did not do on FreeBSD/MoltenVK what it
   did on Linux/NVIDIA, and that is the next thing to chase. This is also the
   first run of the new dispatch-count guards on Vulkan-ONLY hosts; super-io
   auto-detects EXLA, so its Vulkan arm is a forced override.
2. **The Jetson has not been verified since `ad464bce5`.** Three commits behind
   the Keplers. It needs `PATH=/home/io/.asdf/shims:/home/io/.cargo/bin:/home/io/.local/bin`
   and `CXX=g++-13 CC=gcc-13`, and takes ~1.8h.

### What is open

* **D4 — the batched chain path still cannot run.** Nothing sets
  `:exmc_chain_coord`; `synthesise_batched/1` has no callers; the f64 batch NIF
  does not exist in `nx_vulkan`. D1/D2/D3 are closed, so this is the last of the
  four in `docs/BATCHED_CHAIN_DISPATCH.md`, and it is a decision — wire it or
  retire it — not a fix. See that file's Option 1 vs Option 2.
* **Poker on super-io is borderline.** Passes sometimes, times out at 300s
  otherwise, times out on the Jetson, and fails FAST with
  `SynthUnsupportedError` on both Keplers. That last difference is unexplained
  and is the interesting part. It is the only thing between this branch and a
  clean Vulkan arm here.
* **`lib/exmc/nuts/sampler.ex` is not `mix format`-clean at HEAD** (~60 lines).
  Every edit to it this session was hand-formatted to avoid churning unrelated
  lines into a behavioural diff. Wants its own commit.
* **`sample_from_compiled/3` deletes `:exmc_chain_meta` on the success path
  only** — a raise mid-sampling leaks it into the process. The three sites
  fixed this session all use `try/after`; this one was deliberately not
  changed, only not copied.
* **`Chain 0 on :worker_1_45511@... failed (:erpc, :noconnection), retrying on
  coordinator`** appears in every fleet log and predates all of this work.
  Nobody has looked at it.
* **`PlanBPrimeGuardTest` does run here** — an earlier note in this session
  claimed it never executes anywhere. It does, 6 tests. The claim was wrong.

### Two host facts that cost an hour each, so they are written down

**EXLA on super-io needs `LD_LIBRARY_PATH`.** `libexla.so` is a CUDA 12 build
needing `libnvshmem_host.so.3` and `libnvrtc-builtins.so.12.9`, and the wheels
are under **python3.12** site-packages — not the python3.10 tree that holds the
other `nvidia/*` wheels, so searching 3.10 finds nothing and it looks like the
libraries are absent:

    NV=/home/io/.local/lib/python3.12/site-packages/nvidia
    export LD_LIBRARY_PATH=$NV/nvshmem/lib:$NV/cuda_nvrtc/lib

Interactive shells inherit it; **agent, `nohup` and cron shells do not**, so a
suite launched that way silently runs a different backend. `jit.ex:84` has
documented this failure mode since 2026-08-23 without recording the path.
Without it the run is not merely EXLA-less — until `3e58fb70f` the Validator
turned it into ~100 failures across 12 unrelated modules.

**Fetch-by-sha now works.** `uploadpack.allowReachableSHA1InWant` is enabled on
`nx_vulkan.git` at 192.168.0.249, so `mix deps.get` no longer needs a per-host
seeding fetch after a bump. First unattended bump confirmed it: `GET_EXIT=0` on
both Keplers.

### The methodological point, since it is the reusable part

Three separate entry points — the default multi-chain path, the distributed
path, and the streaming path — each destructured `_chain_meta` and threw it
away, disabling the fused chain shader everywhere except single-chain
`Sampler.sample/3`. Measured: **12.9x**, **5.8x**, **2.6x**.

All three survived because the only test watching was a wall-clock inequality
against a concurrent path, which is host-dependent, so its failure was
dismissed as noise for two weeks. **A test that fails for a reason nobody
believes is worse than no test.** The replacements in
`test/exmc/nuts/vulkan/chain_meta_routing_test.exs` count dispatches: exact,
host-independent, and each was run against its own reverted fix to confirm it
fails saying `0 chain dispatches` rather than passing vacuously.

Keep doing that. Every fix this session has a negative control, and two of them
changed conclusions that code-reading alone had gotten wrong.

---

## Status — 2026-08-18, after the reboot

**All seven §2 items are closed.**
**Item 1 was confirmed on 2026-08-17** by the sweep it asked for — see below and
[`bench_results/OBSERVED_MODEL_EVIDENCE.md`](bench_results/OBSERVED_MODEL_EVIDENCE.md).
`6c1589a fix(synth): every observed node summed the WHOLE obs buffer` closed
it. **Item 7 was fixed on 2026-08-18** — three defects, not one; see
[`bench_results/CRASH_RECOVERY.md`](bench_results/CRASH_RECOVERY.md).

| run | result |
|---|---|
| `mix test` (default → EXLA), after items 7 + 2 | 476 tests, **1 failure** (the wall-clock one, item 4), 585s |
| `EXMC_COMPILER=vulkan mix test`, after items 7 + 2 | 476 tests, **3 failures**, 1301s |
| `EXMC_COMPILER=vulkan mix test`, after item 7 only | 473 tests, **2 failures**, 1047s |
| `EXMC_COMPILER=vulkan mix test`, after `6c1589a` | 472 tests, **4 failures** |
| `EXMC_COMPILER=vulkan mix test` ×2, before `6c1589a` | 472 tests, **5 failures**, identical both times |

Start `epmd -daemon` before any sweep — without it `distributed_test.exs`
contributes two failures that have nothing to do with the code.

`integration_test.exs:646` went green in the last one, and it is a real fix
rather than a tolerance passing — the scalar arm measures **mean 3.9864, sd
0.5536, 479/500 distinct** against analytic truth of mean 3.99, sd 0.577. It
had been frozen at 1 distinct draw in 500 with sd 3.29e-14 for the whole of the
preceding session.

That was one seed on one model, so it was not enough on its own. The evidence
table has now been re-run as a sweep — `bench/observed_model_evidence.exs`,
nine model shapes x four seeds x both arms, each row scored against the
closed-form conjugate posterior rather than against the other arm.
**72 of 72 rows within tolerance**; worst mean error 0.091, worst sd error
8.1%, fewest distinct draws 447/500. The frozen-chain signature appears
nowhere. `docs/OPEN_VULKAN_OBSERVED_MODEL.md` carries the table and the three
things about the sweep worth keeping; the raw output and host are in
`bench_results/OBSERVED_MODEL_EVIDENCE.md`.

Items 7 and 2 followed it and are also closed (§2), so **the whole P0 list is
now done**. Item 2 is the one worth reading the write-up for: the sweep found
that no statistical assertion in the suite could see a 20% variance error, the
0.3.1 "fix" included, and that the reason was sample size rather than tolerance
— which means a pure tolerance rewrite would have changed nothing.
[`docs/TOLERANCE_AUDIT.md`](docs/TOLERANCE_AUDIT.md).

### ✅ The crash-recovery defect, in flight when the reboot came — closed

Diagnosed before the reboot, fixed and verified 2026-08-18. It turned out to be
three defects rather than one, and two of them were vacuities that made the
first invisible. `fault_tolerant_test.exs:239` is green on both arms. See §2
item 7 and [`bench_results/CRASH_RECOVERY.md`](bench_results/CRASH_RECOVERY.md).

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

The case for waiting **was item 1** — that Vulkan is the default for the users
who most need this library and was not correct for models with observations.
**That argument is gone as of 2026-08-17:** item 1 is fixed and confirmed across
nine variants and four seeds (§2). What is left against publishing is item 7 —
under `supervised: true`, a crash-recovered Vulkan run returns a destroyed
posterior and reports success — and the unswept tolerances of item 2. Item 7 is
narrower than item 1 was: it needs crashes to trigger, and it is a silent wrong
answer rather than a whole broken model class.

**Updated 2026-08-18: items 7 and 2 are closed too, so P0 is done.** Nothing on
that list blocks publishing on correctness grounds any more.

One thing the release notes should not claim, though. The tolerance sweep
(§2 item 2) makes the suite able to see a 20% variance error **in the marginals
of models with closed forms**. It says nothing about correlations, about the
joint distribution, or about any model without a closed form — which is most
real models. "No known correctness defect, and the suite can now see the defect
class that caused the last two" is the honest sentence. "Verified correct" is
not, and will not be until §3's Geweke and SBC items land.

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

Ranked. **All seven are closed.** P0 is done; §3 (verification as a
deliverable) is what follows, and `docs/TOLERANCE_AUDIT.md` names the two
things there that item 2 could not substitute for — Geweke and SBC.

| # | item | effort | why it ranks here |
|---:|---|---|---|
| 1 | ~~**The vulkan observed-model defect.**~~ **Closed 2026-08-17, confirmed by sweep.** Was: `compiler: :vulkan` returned a frozen chain (1 distinct value in 500 draws, `accept_prob` ≈ 0.002) for models with observations. Fixed in `6c1589a`; confirmed by `bench/observed_model_evidence.exs` — 9 variants x 4 seeds x both arms, **72/72 rows** within tolerance of the closed form, worst sd error 8.1%, fewest distinct draws 447/500. The distinct-sigma variants are the load-bearing ones (equal sigmas make a mis-assigned span bit-identical), and a per-row GPU dispatch count rules out rows that silently fell back to the host. [`docs/OPEN_VULKAN_OBSERVED_MODEL.md`](docs/OPEN_VULKAN_OBSERVED_MODEL.md), raw in [`bench_results/OBSERVED_MODEL_EVIDENCE.md`](bench_results/OBSERVED_MODEL_EVIDENCE.md). | — | — |
| 2 | ~~**The `assert_in_delta` sweep.**~~ **Done 2026-08-18.** The answer was that **no** statistical assertion in the suite could see a 20% variance error — including the one 0.3.1 added as the fix, which resolved 37.8% against a defect that was 37.8%. Three findings: 89 of 99 sampling tests asserted nothing about dispersion at all; where a gate existed the sample size made it decorative (ESS ~0.35/draw, so the resolution limits ran 36-86%); and two gates were wrong in the *other* direction (Student-t at df<=4 has no valid variance gate, Cauchy's IQR gate was 6x too tight). Fixed with `TestHelper.assert_posterior!/3`, which fails as INCONCLUSIVE when the chain cannot resolve the error it claims to check. [`docs/TOLERANCE_AUDIT.md`](docs/TOLERANCE_AUDIT.md). | — | — |
| 3 | ~~**The EXLA build.**~~ **Done — both halves.** The library bug is fixed (`exla` is `runtime: false`, `Exmc.JIT` starts it lazily and treats a failed start as "backend unavailable", covered by `test/optional_deps_test.exs`), *and* this host now has a working CPU EXLA. Recipe and its two traps in [`docs/EXLA_CPU_BUILD.md`](docs/EXLA_CPU_BUILD.md); the short version is `EXLA_CPU_ONLY=1 XLA_TARGET=cpu`, not `XLA_TARGET=cpu`. | — | — |
| 4 | **The wall-clock test.** `mix test` is now **0 failures** on the default (EXLA) path — 375 tests on `main`, 472 on `gate1/reconcile-core` with the MCLMC/MAMS/SBI suites. The only default-path failure left is `integration_test.exs:738` — `assert t_vec < t_par` — and it is **timing-flaky**, not consistently red: it failed at `1034ms < 659ms` on one run and passed on the next with no code change. Move it to `bench/`. The other three failures NEXT.md originally listed were artefacts of Vulkan-by-default and are green under EXLA — **not fixed, not exercised**. | 1 hour | a flaky red trains people to ignore red faster than a stable one |
| 5 | ~~**`config/test.exs` had never been loaded.**~~ **Fixed.** `config/config.exs` was one line, `import Config`, with no `import_config` — and Mix auto-loads only `config/config.exs`, so every setting in `config/test.exs` was dead: the `EXMC_COMPILER` switch, `config :exla, default_client: :host`, `allow_vulkan_perop_sampling`. **Every `EXMC_COMPILER=vulkan mix test` ever run sampled with whatever auto-detect picked and reported a pass for it.** Now imported, with `test/config_test.exs` as the tripwire. | — | — |
| 6 | ~~**Tests leak `:exmc` application env into each other.**~~ **Done, and verified by the check that matters: two consecutive Vulkan sweeps now return identical failure sets.** Three separate instances, all restoring wrongly or not at all: `p0_correctness_test.exs` leaked `compiler: :none`; `nuts_test.exs:618` "reset" `full_tree_nif` to `true` when its default is `false`; `fault_tolerant_test.exs` did the same via `get_env(..., true)` and skipped its restore entirely on a raised assertion. `native_tree_test.exs` was `async: true` while setting `use_nif` globally, so it raced concurrent tests rather than merely later ones. Fixed with `Exmc.TestHelper.put_env_scoped/3` (reads the previous value instead of assuming a default — the assumption is what went wrong three times) plus an `ExUnit.after_suite` tripwire over **all twelve** `:exmc` keys that gate behaviour, since an ordinary assertion only sees leaks from files that ran *before* it. | — | — |
| 7 | ~~**Vulkan crash recovery destroys the posterior, and reports success.**~~ **Closed 2026-08-18.** Three defects, all fixed and all measured: crash-recovered iterations were fed to dual averaging as acceptance 0.0 (a feedback loop that drove eps to 2.41e-11 and the posterior to variance 1.45e-15); `supervised: true` did **nothing at all** wherever the speculative buffer was live, i.e. the default path; and warmup recoveries were never counted, so 176 placeholders reported `recoveries: 0`. Before/after matrix in [`bench_results/CRASH_RECOVERY.md`](bench_results/CRASH_RECOVERY.md), reproducible via `bench/crash_recovery.exs`. | — | — |

### Item 7 in full — Vulkan crash recovery, and how it closed

Diagnosed 2026-08-17, fixed and verified 2026-08-18. Every link below was
measured, not inferred; the before/after matrix is in
[`bench_results/CRASH_RECOVERY.md`](bench_results/CRASH_RECOVERY.md) and
re-runnable with `bench/crash_recovery.exs`.

`fault_tolerant_test.exs:239` samples a **prior-only** `N(0,1)` — no
observations at all, so item 1's model class was never involved — with
`FaultInjector` set to crash at depth 3 and `supervised: true`.

| backend | injector consulted | placeholders | result | adapted eps |
|---|---:|---:|---|---|
| `:none` | 1463 | **0** | mean −0.0707, var 1.0212, 290/300 distinct, 6 div | 0.928 |
| `:exla` | 1463 | — | identical to `:none`, bit for bit | — |
| `:vulkan`, before | 3405 | **176** | mean −0.5225, **var 1.45e-15**, 300/300 distinct, **178 div** | **2.41e-11** |
| `:vulkan`, after | 1973 | 29 | mean 0.0373, **var 0.8807**, 298/300 distinct, 3 div, **29 recoveries** | **0.810** |

Uninjected, Vulkan samples this model at var 0.955 / eps 1.053. The collapse
needed the crashes; the fixed path degrades to 0.881 instead of collapsing.

**It was never item 1.** Item 1's frozen chain is literally 1 distinct draw in
500. This was 300 distinct draws in 300, inside a neighbourhood of ~4e-8. Both
read as "variance collapsed"; they were different failures.

#### Defect 1 — a crashed subtree was treated as evidence about the step size

ε fell from 1.053 to 2.41e-11 — eleven orders of magnitude:

1. A subtree crashes; the supervision wrapper catches it and substitutes
   `divergent_placeholder/4`.
2. The placeholder carried `accept_sum: 0.0`, so `nuts_step_warmup` computed
   `accept_stat = 0.0` and fed it to dual averaging.
3. Acceptance 0.0 reads as "step size catastrophically too large", so ε shrinks.
4. Smaller ε means longer trajectories to a U-turn, so trees get **deeper**.
5. Deeper trees hit the depth-3 injection point more often — 3405 consultations
   against 2018 uninjected.
6. Back to 1.

A crashed subtree measured nothing. The iteration is now **excluded from the
dual-averaging update** (`Sampler.maybe_adapt/3`) rather than fed a zero. The
placeholder also stopped fabricating `n_steps: 2 ** depth` for leaves that were
never integrated — those phantom leaves went into the denominator of
`accept_sum / n_steps` and into per-draw `sample_stats`.

**`n_steps` alone was not the lever**, despite how `divergent_placeholder`
reads: `sampler.ex` already guarded `n_steps > 0` and yielded `0.0` either way,
so zeroing the leaf count fixes the reported statistic but not the loop.
Checked before concluding; both changes are in.

#### Defect 2 — `supervised: true` did nothing on the default path

`safe_build_subtree` was reachable only from the *non-speculative* branch of
`do_build/11`, and `speculative_precompute` defaults to `true`. Measured: on
`:none` with `supervised: true`, 861 subtree builds, all speculative, the
supervision wrapper entered **zero** times. Injecting a crash at depth 1 there
killed the run outright — an unhandled `RuntimeError` straight through
`run_phase/10` — on a run that had explicitly asked to be supervised.

That is also the only reason the depth-3 row above produced numbers at all:
Vulkan's warmup downgrade disables speculation, so supervision happened to be
reachable during Vulkan warmup and nowhere else.

Fixed by hoisting the guard to wrap **whichever** dispatch runs
(`Tree.with_supervision/7`), speculative included, with `ensure_available`
inside the guarded region since the bulk pre-compute can fail the same ways.

#### Defect 3 — warmup recoveries were invisible

The before row built 176 placeholders and reported `recoveries: 0`.
`nuts_step_warmup` never read the `:recovered` flag; only `nuts_step_with_stats`
did, and that runs in the sampling phase, where the Vulkan hot path never
consults the injector. Every crash was a warmup crash, so the one counter an
operator could have noticed was structurally zero. `placeholders` and
`recoveries` now agree on every row of the matrix.

**The divergence count was fabricated too:** 176 placeholders, 178 reported
divergences. Nearly every "divergence" was a placeholder marked
`divergent: true` — which it must be, to stop the doubling — not an integrator
divergence. Crash-recovered iterations now count as recoveries and not as
divergences. That attribution is exact rather than approximate: `do_build/11`
breaks on `subtree.divergent or subtree.turning`, so a genuine divergence in an
earlier subtree would have stopped the loop before the crashing one was built,
and the two cannot co-occur in one iteration.

#### The test that proved nothing

`fault_tolerant_test.exs`'s end-to-end recovery test injected at **depth 3**,
and `:none` hit `divergent_placeholder` **zero** times while consulting the
injector 1463 times: a well-adapted host sampler never builds trees that deep.
It passed on `:none` and `:exla` for months without recovery ever running.

It now injects at depth 1, which fires on every backend (357 recoveries on
`:none`, 29 at depth 3 on `:vulkan`), and asserts `recoveries > 0` so the
vacuity cannot come back silently. A second test guards the mechanism directly
— `assert eps > 0.01` catches the feedback loop eleven orders of magnitude
before the variance assertion would. Both fail against the unfixed tree.

#### What recovery still costs

It is not free. At depth 1, where 357 of 500 iterations crash, ε adapts to 2.04
against 0.93 uninjected, variance lands at 0.877 against 1.0, and 245 of 300
draws are distinct. That is a degraded chain — and it is a chain, which is the
point. Both no-injection controls are **bit-identical** before and after, so
nothing on the non-crash path moved.

### Do not skip the red test — and mind which backend it is running

The annotated test is `integration_test.exs:646`, "vector obs produces same
posterior as equivalent scalar obs" (this file previously said 611, then 639 —
the annotation was rewritten on 2026-08-17 when the defect closed, which moved
it again; grep for the test name rather than trusting the number). It fails
for a real reason and is annotated to say so. Skipping a test that fails for a
real reason is precisely the habit that let both 0.3.1 defects ship.

**It is green under Vulkan as of 2026-08-17**, verified directly:
`EXMC_COMPILER=vulkan mix test test/integration_test.exs:646` passes, and the
sweep in item 1 says the same thing across nine variants and four seeds. The
annotation on the test still describes it as known-failing and should be
rewritten to describe the closed defect instead — do not simply delete it, the
test is the regression guard for `6c1589a`.

What follows is the state it was in before the fix, kept because it is what the
signature looks like. Since item 5 was fixed,
`EXMC_COMPILER=vulkan mix test test/integration_test.exs:646` used to fail on
exactly the documented assertion:

```
code: assert_in_delta scalar_summary["mu"].std, vector_summary["mu"].std, 0.3
```

Item 1 was confirmed alive at that point, and reproduced the numbers in
`docs/OPEN_VULKAN_OBSERVED_MODEL.md` to the digit — scalar arm mean **3.6503**,
sd **3.29e-14**, **1 distinct draw in 500**; vector arm mean 3.9716, sd 0.5516,
472/500. Under `EXMC_COMPILER=none` both arms are correct.
`allow_vulkan_perop_sampling` made no difference to it either way, which ruled
that out as the route around the chain shader.

The standing lesson: it was **green under a bare `mix test`** the whole time it
was broken, and that was not a fix — auto-detect picks EXLA on this host and the
Vulkan path is never entered (see the correction in §1). A bare `mix test`
passing says nothing about the Vulkan path. Use `EXMC_COMPILER=vulkan`, and
note that it only means anything now that item 5 is fixed.

### What the Vulkan sweep actually says

Latest, 2026-08-18, after items 7 and 2 — **476 tests, 3 failures** (was 4,
was 5; item 2 added three tests and made several much slower). The default
(EXLA) path is **476 tests, 1 failure** — the wall-clock one — on the same tree.

| test | failure |
|---|---|
| `level_set_integration_test.exs:11` | timed out at 300s |
| `poker_test.exs:228` | timed out at 300s — **back**, see below |
| `integration_test.exs:745` | the wall-clock assertion (item 4), fails on both paths |

**The poker timeout came back, and item 2 is the likely reason.** It dropped
off the list on the sweep immediately after item 7, which was run on an
otherwise idle box; item 2 raised several tests from a few hundred draws to
five figures and the Vulkan sweep went from 1047s to **1301s**. A test that
hangs until ExUnit kills it at 300s is exactly the kind that a busier box tips
over. It is a timeout with no containment behind it (see below), not a
correctness failure, and it should be treated as load-sensitive rather than
fixed or broken.

Dropped off since the last sweep:

* `fault_tolerant_test.exs:239` — **item 7, fixed.** Was `Variance collapsed:
  1.45e-15`.
* `poker_test.exs:228` — dropped off after item 7 and came back after item 2,
  both times a 300s timeout with no change that should affect it. Load-flaky;
  see the note above.
* `integration_test.exs:646` dropped off at `6c1589a` — see item 1.
* `new_dist_test.exs:271`'s `read spv: No such file or directory` dropped off
  earlier and is not expected back: it was a shader-cache race, fixed in
  `e167734`.

**Run `epmd -daemon` before a sweep.** Two `distributed_test.exs` failures on
2026-08-17 were nothing but a missing epmd after the reboot — `Cannot start
distribution ... econnrefused`. With epmd up the file is 5 tests, 0 failures.
It costs a minute to chase and looks like a real distributed-sampling defect.

The remaining timeout is unexplained and is **not** contained: per the findings
below there is no timeout containment on the Vulkan path at all, so it hangs
until ExUnit kills it at 300s.

The sweep is reproducible as of `1e735bc` — two consecutive runs before
`6c1589a` produced identical failure sets, differing only in ordering.

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
(cross-repo, 1,641 lines). Its rank-1 item was §2 item 2 above, now closed —
though only partly in the form that document expected. Item 2 did its own
items 1 and 2 (the variance SE, and refusing a variance gate for
`2 < nu <= 4` Student-t), fixed a third gate it had not spotted (Cauchy's IQR
band, six times too tight), and added the thing that was missing from all of
them: a **power** check, so a gate that cannot see the defect it claims to
check fails rather than passes. See `docs/TOLERANCE_AUDIT.md`.

**Still open from that rank-1 list**, and cheap: thin to independence before
the KS gate (or drop it), make `:unknown` visible at suite level, and add
split-R-hat with an `ESS >= 100` precondition. Items 7-9 there — un-tag
`nuts_test.exs`, parametrise the reversibility test over all five leapfrog
implementations, ungate `native_tree_test.exs`'s `:vulkan_known_failure` — are
untouched and are the ones that make existing tests run against shipping code.
(`nuts_test.exs` no longer carries `@moduletag :gpu_state`, so item 7 there may
already be moot; check before doing it.)

Its next rank is **Geweke's joint distribution test**, which is the one check
that would have caught the tree defect at the point of introduction rather than
months later. Nothing in item 2 substitutes for it: the tolerance sweep can see
a wrong marginal in a model with a closed form, and says nothing about the
joint or about models without one.

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

One item that moved up as a result of 0.3.1: **every claim should point at raw
output.** `bench_results/` now exists and holds four files — `MCLMC_BIAS.md` (partial,
§6), `OBSERVED_MODEL_EVIDENCE.md`, `CRASH_RECOVERY.md` and `TOLERANCE_AUDIT.md`
(complete). `bench/` has seven scripts. Every performance claim in the README should point at a file
containing raw output and the host it ran on, the way `nx_vulkan/bench_results/`
does; none of them do yet.

---

## 5. Where the numbers came from

So the next person can re-run rather than trust:

```sh
# posterior moments vs analytic truth — the check that can see a shared defect
mix run --no-deps-check bench/nuts_truth.exs
COMPILER=vulkan SEEDS=1,2,3 mix run --no-deps-check bench/nuts_truth.exs
USE_NIF=0      mix run --no-deps-check bench/nuts_truth.exs   # pure Elixir tree
FULL_TREE_NIF=1 mix run --no-deps-check bench/nuts_truth.exs  # Rust build_full_tree

# the observed-model evidence sweep — the check that closed §2 item 1
COMPILER=none   SEEDS=42,1,2,3 mix run --no-deps-check bench/observed_model_evidence.exs
COMPILER=vulkan SEEDS=42,1,2,3 mix run --no-deps-check bench/observed_model_evidence.exs

# what the suite's statistical tolerances admit — the check that closed §2
# item 2. Each test appears twice, `was` and `now`; the `4sd floor` column is
# the resolution limit no tolerance rewrite can beat.
mix run --no-deps-check bench/tolerance_audit.exs

# crash recovery under supervision — the check that closed §2 item 7.
# INJECT=0 is the control; `placeholders: 0` with injection on means the run
# never exercised recovery, however green it looks.
COMPILER=none   DEPTH=1  mix run --no-deps-check bench/crash_recovery.exs
COMPILER=vulkan DEPTH=3  mix run --no-deps-check bench/crash_recovery.exs
COMPILER=vulkan INJECT=0 mix run --no-deps-check bench/crash_recovery.exs

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

---

## 7. C2 (SBI / ABC) — landed, and the thing it was built for is not

**Written 2026-08-20.** Roadmap item C2 from
`/home/io/projects/learn_erl/pymc/exmc/docs/PLAN_SAMPLER_ROADMAP.md` §5, whose
title is "simulation-based inference **on `sim_ex`**". Four of its five stages
are complete and gated. **C2.4 — the `sim_ex` bridge and the calibration
notebook — was not started**, and since that is the stage the item is named
after, read the rest of this section with that in mind: what landed is a
correct, tested ABC library with no consumer.

Committed as `82db4f8`.

### What landed

| file | what |
|---|---|
| `lib/exmc/sbi/simulator.ex` | the behaviour — `simulate(params, rng) :: {summary, rng}`, a parameter map of plain floats and a functional `:rand` state in, a summary vector and the advanced state out |
| `lib/exmc/sbi/prior.ex` | independent scalar priors, drawable and evaluable. Deliberately standalone rather than a route into `Exmc.IR`: ABC needs draw and `logpdf` and nothing else — no gradient, no transform |
| `lib/exmc/sbi/abc.ex` | rejection ABC — the reference arm |
| `lib/exmc/sbi/abc_smc.ex` | Toni et al. (2009), Del Moral's adaptive tolerance, the Beaumont et al. (2009) perturbation kernel |
| `lib/exmc/sbi/engine.ex` | option resolution, the distance, RNG splitting, batched evaluation |
| `lib/exmc/sbi.ex` | `run/3`, `run!/3`, and the posterior accessors — `posterior_mean/1`, `credible_interval/3`, `weight_ess/1`, `resample/3`, `rank/5`, `prior_predictive_scale/3` |
| `test/sbi/*` | 55 tests, all green |

Nothing outside `lib/exmc/sbi/` changed except `mix.exs`, which gained the
ex_doc groups. There is no dependency on the model layer at all: `Exmc.SBI`
never calls `Exmc.Compiler`, never builds an `Exmc.IR`, and never
differentiates anything. It is the only inference path in the repository that
does not.

### The stages, and which one is missing

| # | work | state |
|---|---|---|
| C2.1 | `Simulator` behaviour + rejection ABC | ✅ |
| C2.2 | ABC-SMC — schedule, kernel, weights, resampling | ✅ |
| C2.3 | parallel evaluation over `Task.async_stream`, **then `Mesh.Pool`** | ✅ / **✗** |
| C2.4 | the `sim_ex_exmc` bridge + the M/M/1 calibration notebook | **✗ not started** |
| C2.5 | the SBC gate | ✅ |

C2.3 is half done and the half that is missing is the one that matters for the
BEAM claim. `Task.async_stream` fans a population out across schedulers on one
node; `Mesh.Pool` is what would fan it across the cluster. **`lib/exmc/mesh/`
does not exist in this repository** — it is one of the private-only subtrees the
Gate 1 survey found. So the distributed arm of C2.3 is blocked on the
core/applications split, not on ABC.

### The gate, and the two tests the default run does not execute

`test/sbi/sbc_test.exs` is the primary gate, and it is the one place in the
whole roadmap where SBC is the right tool: Geweke is ~40× cheaper but needs an
exact-invariance argument about a Markov kernel, and a likelihood-free
posterior has none. It is affordable here because the target is a
Normal–Normal conjugate model whose simulator is ten normal draws — 800
complete ABC-SMC fits cost about ten seconds, not the hours a NUTS SBC would.

The target is conjugate for a second reason worth keeping: its summary, the
sample mean, is **sufficient**. Run SBC on the M/M/1 queue instead and a red
gate is ambiguous between "the sampler is wrong" and "mean waiting time is not
sufficient for (λ, μ)" — and an ambiguous gate teaches people to ignore it.

The module documents its own error rates rather than asserting them. Null, 300
experiments of 800 replicates: rejection at α = 0.01 measured **0.0033**
against a nominal 0.01, median p-value **0.485**. Power at α = 0.01 against a
posterior whose standard deviation is wrong by a fixed factor: **0.913** at
sd × 0.85, **0.427** at sd × 0.90, **0.307** at sd × 1.10. 800 replicates and
10 bins were both chosen off that table rather than picked. And the gate is
then measured *in situ* against the real sampler by throwing the importance
weights away — the single most likely way for an ABC-SMC implementation to be
wrong — which gives **p = 7.3e-8** against **0.299** for the unmodified
posteriors.

**But the two tests that produce those error rates are `@tag :slow` and are
excluded from the default run** — `sbc_test.exs:183` (false-positive rate under
the null) and `:213` (power against a known scale error). That is a defensible
call, since between them they are 300 complete SBC experiments. It does mean
the default `mix test` re-checks the gate but never re-checks the gate's
credibility. The numbers above are as measured on the day and nothing in CI
would notice them drifting.

```sh
mix test --include slow test/sbi/sbc_test.exs   # runtime not measured; budget generously
```

The M/M/1 arm (`test/sbi/mm1_test.exs`) validates the fixture against
`ρ = λ/μ`, `Wq = ρ/(μ−λ)` and `Lq = ρ²/(1−ρ)` **before** using it to gate the
inference, which is what makes it a gate rather than a comparison against
another piece of our own code. Every tolerance in it is a standard error
computed from the run's own replications or a binomial bound on a coverage
count.

### What is not done, and should be

- **C2.4, the whole of it.** No `sim_ex_exmc` bridge — `sim_ex` is not
  referenced anywhere under `lib/exmc/sbi/` or `test/sbi/`. No calibration
  notebook; `notebooks/` has nothing on ABC. The M/M/1 that exists is a test
  fixture in `test/sbi/support/mm1.exs`, not a document anyone would read.
  This is the deliverable the roadmap calls P-3 and describes as "the only item
  on this roadmap that is plausibly publishable on its own", and it is the item
  the "abandon if" condition is written about — whether the simulator call
  dominates so completely that useful particle counts are out of reach. **That
  question is currently unanswered**, because the only simulator ABC has been
  run against is a fixture designed to be fast.
- **`Mesh.Pool` evaluation.** Above. Blocked on the split.
- **No `bench/` script and no `bench_results/` file.** §4 of this document says
  every claim should point at raw output; C2 is the newest subsystem in the
  repo and is the one with no such file. The SBC error-rate table lives in a
  `@moduledoc` and the M/M/1 coverage numbers live in assertions. Both should
  be re-derivable by running one script.
- **Neither arm has been run on Vulkan or EXLA**, because neither touches Nx.
  That is correct — a simulator is a process, not a tensor lane — but it means
  C2 contributes nothing to the backend sweep in §2 and should not be counted
  toward it.

### One design decision worth not undoing

Proposals are generated **sequentially** from the parent `:rand` state and
evaluated in a **batch whose size is fixed before any simulator runs**, so
nothing about the result depends on the concurrency. `parallel: false` and
`parallel: true` over the same seed produce bit-identical particles, weights
and simulation counts — which turns "is the parallel path the same algorithm?"
from a hopeful assertion into an equality test, and
`test/sbi/determinism_test.exs` asserts exactly that, including across worker
counts.

The cost is real and is not hidden: a batch may simulate more proposals than
the population needs, because it cannot stop early. `n_simulations` in every
result counts what actually ran. Anyone tempted to reclaim those simulations by
generating proposals inside the workers should understand they are trading the
equality test for them.

### The caveat that belongs on the front page

`Exmc.SBI`'s `@moduledoc` carries it as a warning block and it should stay
there: ABC targets `p(θ | S(y*))`, not `p(θ | y*)`, and those are the same
distribution only when `S` is sufficient. Driving `ε → 0` does not repair it.
There are two approximations in every ABC posterior and only `ε` is under the
user's control — insufficiency does not appear in any diagnostic, does not
shrink with budget, and does not show up as a divergence or a low ESS. The
roadmap made this an explicit gate requirement ("in the docs and not only in
the tests") because the characteristic failure of ABC software is that the
sentence exists somewhere and nobody reads it. If the README ever grows an SBI
section, it goes there too.
