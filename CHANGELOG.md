# Changelog

## Unreleased

Everything since 0.4.0, from `git log v0.4.0..HEAD`; short shas are the
commits to read.

**Fixed**

- `stats.divergences` counted warmup, and every consumer divided it by the
  number of kept samples. It now counts kept draws only (`994305de4`). A
  downstream notebook that asserted the old behaviour (pathmc_ex `guide/02`)
  broke on the next pin bump, as the commit predicted.
- Vectorized chains sampled from a state warmup never saw their own warmup
  state — the frozen-chain bug pathmc_ex reproduced (`cbc858ee7`).
- `Nx.pow` in a likelihood killed the shader; our own docs had taught the idiom
  (`da6ea6887`).
- The synthesiser's refusal reason was discarded on the way out and a
  placeholder was over-refused; multi-equation models now come back as
  `{:unsupported, :multiple_custom_nodes}` rather than a bare atom
  (`147305261`).
- Two posteriordb bench fixes: one slow posterior no longer kills the other 32,
  and the crashing posterior is named (`4c658c0d7`, `45c7b7566`).

**Added**

- Vector-valued RVs and `Nx.dot` over a captured design matrix reach the fused
  shader (`f04d080a5`, `371785ff5`).
- The observation buffer is populated from a Custom RV's own observations,
  with two new refusals for the shapes that cannot be (`d299f4fc4`,
  `d410b183a`).
- Observation-axis loops that share a trip count are fused; measured 1.83×,
  against a plan that predicted otherwise (`f2ae139d7`, `e10dfee35`).
- The leapfrog leaf-diff harness is a test that can fail:
  `test/nuts/leapfrog_leaf_diff_test.exs`, `:requires_vulkan`, with
  fixture-calibrated tolerances measured on four hosts (`671150a0d`,
  `acccf8348`, `267634cb0`).
- `scripts/fleet_verify.sh` drives the fleet suite with a non-vacuity gate; the
  posteriordb fixtures are committed (`699f3a870`, `702fb780f`).
- The env tripwire snapshots the whole application env rather than a list
  somebody wrote (`5f2545b47`).

**Changed**

- nx_vulkan pin: `36c6805` → `bc54f34` → `9a8427c` (`745376bb5`, `f1e9b2207`,
  `144d441db`); nx pinned to three components, `~> 0.13.1` (`745376bb5`).
- Bench scripts no longer carry the `EXMC_COMPILER` workarounds that
  `config/runtime.exs` made redundant (`3eb9a6ec1`).

**Known**

- Multi-equation models (more than one Custom likelihood node) are refused
  with `:multiple_custom_nodes`; single-equation models synthesise.

## 0.4.0 (2026-09-06) — A Backend That Says What It Ran

The headline is not a feature. It is that the fused Vulkan chain path was
returning wrong posteriors, silently, and now does not — and that the harness
which should have caught it can now state what produced its numbers.

**Three defects on the Vulkan path, all silent, all fixed.**

1. *Shader size scaled with the DATA.* Closure-captured rank-1 tensors were
   emitted as `const double[]` literals, so SPIR-V grew with `n_obs * n_beta`.
   Synthesised shaders reached 2.15 MB against ~8 KB for a hand-written one, and
   past ~1300 inlined elements the driver refused to create the compute
   pipeline. 21 of 33 posteriordb models could not run at all. Captures now live
   in the extras SSBO; SPIR-V is byte-identical across a 20x change in `n_obs`.

2. *A non-finite trajectory crashed instead of diverging.* The recursive Elixir
   path had always routed non-finite values to the divergent fallback; the NIF
   path handed the chain binaries to `build_subtree_bin/9` unchecked, where a
   non-finite f64 is `badarg`. A routine "reject this trajectory" became a hard
   crash depending only on tree depth.

3. *The reduce loop was bounded by an empty observation axis.* This is the one
   that mattered. Loops were emitted as `j < pc.n_obs`, but a likelihood whose
   data arrives as closure captures does not populate that buffer — `n_obs` was
   0, the loop ran ZERO times, the likelihood evaluated to nothing, and the
   sampler returned the PRIOR while reporting it as a posterior. No crash, no
   error, no divergence. 32 of 33 posteriordb models. On `sblrc-blr`: max mean
   error **55.085 before, 0.12 after**, against EXLA's 0.13 — with an identical
   7/300 divergence count.

**The benchmark harness could not have caught any of it, and now can.**
`benchmark/posteriordb` had no compiler selection at all: it ran under `mix run`,
where `EXMC_COMPILER` was inert, so its "33/33 PASS" was an EXLA result that
never said so. It now names its compiler, records full provenance (both shas, a
dirty flag, resolved backend, precision) into a versioned JSON artifact, and
gates on statistics rather than two fixed constants — R-hat, ESS, MCSE and a
divergence rate calibrated against the healthy baseline instead of convention.
The old criteria had certified a model with **R-hat 1.845** as PASS.

**And the banner stopped lying.** `Exmc.JIT.describe/0` printed a backend
*derived* from the detected compiler while never asking Nx anything. It now
prints derived and observed side by side; on the Vulkan arm they disagree, which
is how we learned the per-op path is an interpreter on the CPU.

### Fixed

- **The shader cache handed out empty SPIR-V modules under concurrency.**
  `CustomSynth.Compile.compile_fresh/2` pointed `glslangValidator -o` straight
  at the content-addressed cache path and derived its temp GLSL path the same
  way, so concurrent callers synthesising the same shader shared both.
  `glslangValidator` creates its output file before writing to it, so the
  `File.exists?/1` fast path returned `{:ok, spv_path}` for a module with no
  contents yet. Measured at 24 concurrent compiles over 40 rounds: **50 of 960
  callers got a zero-byte file**. Ten-plus test modules are `async: true` and
  sample, so this was reachable from an ordinary `mix test`.

  Compilation now goes to per-caller temp paths and `File.rename/2`s into
  place, which is atomic within a filesystem: a concurrent existence check sees
  either no file or a complete one. Same repro after the fix: 960/960 clean.

  The shared temp *source* path was the same bug one level down — one caller's
  cleanup could delete the source out from under another's running validator,
  and a validator that fails partway can take its `-o` target with it. That is
  the likely origin of the intermittent
  `read spv: No such file or directory` seen under `EXMC_COMPILER=vulkan`,
  though that interleaving was never directly reproduced and is eliminated by
  construction rather than by observation.

- **Tests leaked `:exmc` application env into each other**, which made the
  backend sweep report passes for runs it never performed. `put_env/3` is
  global and VM-lifetime and ExUnit orders files by a random seed, so which
  backend — and which of the three tree implementations — a test exercised
  depended on file order and varied run to run. Three instances, each restoring
  wrongly or not at all: `p0_correctness_test.exs` leaked `compiler: :none`;
  `nuts_test.exs` "reset" `full_tree_nif` to `true` when its default is
  `false`; `fault_tolerant_test.exs` did the same through a wrong default and
  skipped its restore entirely when an assertion raised first.
  `native_tree_test.exs` was `async: true` while setting `use_nif` globally.

  `Exmc.TestHelper.put_env_scoped/3` reads the previous value rather than
  assuming a default — assuming is what went wrong all three times — and an
  `ExUnit.after_suite` tripwire now watches all twelve `:exmc` keys that gate
  behaviour and fails the run on a leak. An ordinary assertion cannot catch
  this: it only sees leaks from files that happened to run before it.

  Effect: two consecutive `EXMC_COMPILER=vulkan mix test` runs now return
  identical failure sets, and `integration_test.exs:639` fails in the full
  suite as it always did in isolation.

- **A broken optional dependency could make the whole test suite unrunnable.**
  `exla` was declared `optional: true` but still landed in `exmc`'s
  `applications` list, so the BEAM started it at boot. On a host where the
  installed exla is present but cannot load — a CUDA build missing
  `libnvshmem_host.so.3`, a stale `_build`, a mismatched `XLA_TARGET` —
  `EXLA.Application.start/2` failed, took the VM with it, and `mix test` never
  reached ExUnit. A dependency this project advertises as optional could stop
  every test in the repo from running.

  `exla` is now `runtime: false`: still on the code path, no longer on the boot
  path. `Exmc.JIT` starts it on first use and treats a failed start as "backend
  unavailable", falling through to the next one. Consumers are unaffected —
  `optional: true` already means they declare exla themselves, which puts it in
  their own application list and starts it at boot as before.

- **`Exmc.JIT.detect_compiler/0` selected backends it could not run.** The
  availability check was `Code.ensure_loaded?/1`, which a present-but-broken
  exla passes: every module is there, only the NIF and the application are not.
  Detection returned `EXLA` and the first `jit/2` call raised. It now requires
  the backend's application to actually start, memoised in `:persistent_term`.

  `test/optional_deps_test.exs` covers both: `:exla` must stay out of `exmc`'s
  `applications`, and a selected compiler's application must be running.

- **`config/test.exs` was never loaded, so the backend sweep was vacuous.**
  `config/config.exs` was a single `import Config` with no `import_config`, and
  Mix auto-loads only `config/config.exs`. Every setting in `config/test.exs`
  was dead: the `EXMC_COMPILER` switch, `config :exla, default_client: :host`,
  and `allow_vulkan_perop_sampling`. **Every `EXMC_COMPILER=vulkan mix test`
  ever run sampled with whatever `Exmc.JIT.auto_detect/0` happened to pick, and
  reported a pass for it.** A dead config file fails silently in both
  directions — nothing warns that it was skipped, and every test still passes.

  `config/config.exs` now imports it, and `test/config_test.exs` is the
  tripwire: it asserts the import is live and that a configured compiler is the
  compiler `detect_compiler/0` actually returns.

- **A test leaked `compiler: :none` into every test that ran after it.**
  `p0_correctness_test.exs`'s `draw/3` helper pinned the compiler to `:none`
  for its host-path checks and never restored it. `Application.put_env/3` is
  global and lives for the rest of the VM, and ExUnit orders files by a random
  seed, so an arbitrary and run-varying fraction of the suite silently sampled
  on the pure-Elixir path regardless of `EXMC_COMPILER`. It now saves and
  restores.

  This is the same vacuity as the config defect above and survived the fix for
  it. The same pattern remains for `use_nif`, `full_tree_nif` and
  `speculative_precompute` across four other test files — the keys that select
  which of the three tree implementations runs — and is tracked in NEXT.md.

  With the sweep working, `integration_test.exs:639` is red again under
  `EXMC_COMPILER=vulkan`, which is the correct state — the open defect in
  `docs/OPEN_VULKAN_OBSERVED_MODEL.md` reproduces to the digit (scalar arm: 1
  distinct draw in 500, sd 3.29e-14). It is not a new defect and not a
  regression; it is the first time the check that was supposed to see it
  actually ran.

**If you drew posterior samples with 0.1.0–0.3.0, they were over-dispersed.**
Upgrade and re-run anything you are relying on. This release fixes two defects
that made every posterior this sampler produced too wide and — on any
distribution with a bounded support — shifted its mean away from the boundary.

### Fixed

- **An invalid doubling was still merged into the trajectory.** Stan's
  `base_nuts::transition` does `if (!valid_subtree) break;` *before* the
  progressive-sampling step. `Tree.do_build/11` called `merge_trajectories/6`
  unconditionally: `subtree.turning` and `subtree.divergent` only ever reached
  the loop condition, so states beyond the U-turn still entered
  `combined_log_weight` and could still be drawn as the proposal.
  `Tree.build_subtree/10` had the same hole one level down — it guarded the
  left half and merged the right half whatever it was.
  `native/exmc_tree/src/tree.rs` is a 1:1 translation and carried both.
  Present since 0.1.0.

  Those states are not in the reversible set the transition samples from.
  Including them breaks detailed balance in one direction only — outward,
  toward the far end of the trajectory.

- **The chain shader's `logp_chain[k]` described the state *before* step k.**
  The prior log-density was evaluated in the pre-update block and then stored
  alongside the post-update `q_chain[k]`/`p_chain[k]`/`grad_chain[k]`. The host
  fed that one-step-lagged density straight into the Metropolis ratio and the
  U-turn test. Affects `compiler: :vulkan`, so 0.3.0 only.

### Measured

Host path, 6 seeds × 2000 draws, against analytic truth — not against another
backend:

| | truth | 0.3.0 | 0.3.1 |
|---|---|---|---|
| `Normal(0,1)` variance | 1.0 | **1.378** | 1.018 |
| `HalfNormal(1)` mean | 0.7979 | **0.8631** | 0.8009 |
| `HalfNormal(1)` variance | 0.3634 | **0.4631** | 0.3505 |
| `Exponential(2)` mean | 0.5 | **0.5749** | 0.4955 |
| `Exponential(2)` variance | 0.25 | **0.3376** | 0.2569 |

Under `compiler: :vulkan`, where both defects compounded, `Normal(0,1)`'s
variance was **23.68** against a true 1.0 and `HalfNormal(1)`'s mean was
**3.27** against 0.798.

### Expect more divergences

A doubling that diverges now terminates the trajectory instead of being absorbed
into it, so reported divergence counts rise. That is the fix working, not a
regression. Leaf counts and accept probabilities still accumulate across the
discarded subtree, because dual averaging is defined over every leaf the
integrator actually visited — Stan updates `n_leapfrog_` and `sum_metro_prob`
per leaf, before the validity check.

### Why the test suite did not catch this

Every statistical check in the repo was **differential**: run the model under two
backends, assert the posteriors agree. Both backends share the NUTS tree, so a
defect there moves both arms identically and the comparison passes. The
posteriordb gate could not see it either — its window is a factor of 2 on the
standard deviation, which is a factor of 4 on the variance, and the worst of
these was 1.38.

So this release also adds the check that *can* see a defect two arms share:

- `Exmc.NUTS.Vulkan.Validator.check_analytic/3` — compares one arm against the
  distribution's own moments, with tolerances derived from the sampler's
  effective sample size (`ess/1`, by Geyer's initial monotone positive sequence)
  rather than from a round number. Its variance standard error uses the general
  `Var(s²) = (μ₄ − σ⁴)/n` rather than the Gaussian `2σ⁴/n`, which understates
  the true error by 2× for exponential-tailed families.
- `bench/nuts_truth.exs` — posterior moments against analytic truth, sweeping
  all three tree implementations (`use_nif`, `full_tree_nif`), since each
  carries its own copy of the doubling logic.
- `test/nuts/p0_correctness_test.exs` — regression tests verified by reverting
  the fixes and watching 4 of them fail.

### Also

- **The documented chain-shader width cap was wrong by a factor of 20.**
  `Push`'s moduledoc claimed a 16-byte header leaving room for 28 prior-param
  floats. The header is **24** bytes and the remaining 104 hold **13** f64
  values: `d ≤ 13` for one-parameter priors, `d ≤ 6` for `Normal`, `d ≤ 4` for
  `StudentT`, `d ≤ 3` for `TruncatedNormal`. The `d <= 256` in the dispatch
  guards is the thread-tile size and is never the binding constraint.

### Known issue

`compiler: :vulkan` — the default — is **not correct for models with
observations**, independently of the above. See
[`docs/OPEN_VULKAN_OBSERVED_MODEL.md`](docs/OPEN_VULKAN_OBSERVED_MODEL.md).
Use `compiler: :none` or `compiler: :exla` for observed models until that is
resolved.

## 0.3.0 (2026-07-23) — A Posterior on Any GPU

Vulkan f64 GPU backend + Nx 0.13. NUTS now runs on the GPU via nx_vulkan's
`VulkanoBackend` (fused f64 leapfrog chain shaders) in addition to EXLA — select
with `EXMC_COMPILER=vulkan`. Runs anywhere Vulkan does, including FreeBSD and MoltenVK.

- Vulkan f64 chain-shader sampling: multi-RV models synthesize a fused f64 leapfrog
  chain shader and dispatch on the GPU; non-synthesizable models fall back to per-op.
- Nx/EXLA 0.13; EMLX dropped; default precision is **f64** end to end.
- Gamma/Beta priors route through the synth chain path (added `Push.prior_param_floats/1`
  encoders).
- Measurable-matmul under Vulkan: `jit_solve`/`jit_determinant` pinned to `BinaryBackend`
  so the LU host-fallback's index tensors don't leak into `Nx.BinaryBackend.slice` and
  crash under nx 0.13.
- `push_too_large`: models whose priors exceed the 128-byte f64 push-constants block now
  degrade to per-op sampling (with a warning) instead of crashing at dispatch.
- f64 `@data_sentinel` (matches the post-EMLX default precision).
- Validation: posteriordb **33/33** on EXLA; Vulkan fallback verified on the FreeBSD
  GT 650M (mac-247); clean-room `mix deps.get` resolves nx 0.13 + nx_vulkan.
  (2026-09-06 correction: this said "EXLA-GPU". The device is not recorded
  anywhere in the artifact, and the scripted path — `run_validation.sh:81` —
  forces `CUDA_VISIBLE_DEVICES=""`, i.e. CPU. So the arm was EXLA and the
  device is unknown. The suite gained a compiler switch and provenance only on
  2026-09-05; every figure predating that names no backend.)
- Requires nx_vulkan `main` (Nx-0.13-compatible).

Story / release notes: [*A Posterior on Any GPU*](https://www.dataalienist.com/blog-a-posterior-on-any-gpu.html).

## 0.2.0 (2026-03-30)

- Warm-start NUTS: reuse previous mass matrix + step size (5.8x speedup)
- 21 distributions (Lognormal, HalfCauchy, TruncatedNormal, Bernoulli, Poisson added)
- Builder.data/2 API for JIT-safe observation data (fixes 256GB memory leak)
- 4 new notebooks (Bayesian SPC, Bearing Degradation, Turbofan Fleet, State-Space)
- 4 new docs (Warm Start, State Space Models, Scheduler Pinning, Forest Tracker)
- Les Trois Chambrées cross-references (smc_ex, StochTree-Ex)
- Beats PyMC on 4 of 7 benchmarks (medium 1.90x, stress 1.16x, eight_schools 2.55x, sv 1.20x)

## 0.1.0 (2026-01-15)

Initial release.

- NUTS sampler with Stan-style three-phase warmup
- ADVI (mean-field variational inference)
- SMC (likelihood tempering)
- Pathfinder (L-BFGS initialization)
- 16 distributions with automatic constraint transforms
- Streaming inference via sample_stream/4
- Distributed MCMC across Erlang nodes
- 337 tests, 33/33 posteriordb validation
