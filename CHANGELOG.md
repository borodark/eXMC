# Changelog

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
