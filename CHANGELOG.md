# Changelog

## 0.3.1 (2026-08-16) — Correctness

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
- Validation: posteriordb **33/33** on EXLA-GPU; Vulkan fallback verified on the FreeBSD
  GT 650M (mac-247); clean-room `mix deps.get` resolves nx 0.13 + nx_vulkan.
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
