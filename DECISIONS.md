# Decisions

This document records key architectural and design decisions for the eXMC prototype.
Each entry includes the assumption that must hold for the decision to remain valid.

## 1. Nx as the numeric backend
- Decision: Use Nx for tensor operations, broadcasting, and autodiff primitives.
- Rationale: Nx provides a pure Elixir API with backends for CPU/GPU, and integrates with Defn/EXLA later.
- Assumption: Nx BinaryBackend is sufficient for prototype-scale models (< ~10 RVs, < ~1000 samples). **[EVOLVED]** D24 added EXLA for gradient computation, D53 added EMLX for macOS Metal GPU. Nx remains the API; backends are runtime-selectable via `Exmc.JIT`.
- Implication: All arithmetic in the codebase must use `Nx.*` operators, not `Kernel` arithmetic.

## 2. Minimal probabilistic IR
- Decision: Represent models as a small IR (`Exmc.IR` + `Exmc.Node`) with RV, obs, and det nodes.
- Rationale: Keeps logprob construction explicit and composable before introducing a full graph compiler.
- Assumption: Three node types (RV, obs, det) are sufficient to express the target model class.
- Implication: No implicit inference; logprob is derived from explicit nodes.

## 3. Rewrite pipeline with named passes
- Decision: Implement a rewrite pipeline with named passes and a `Pass` behavior.
- Rationale: Mirrors PyMC/PyTensor rewrite systems, keeps transforms/measurable ops modular.
- Assumption: Structural changes to the IR can be decomposed into independent, composable passes.
- Implication: All structural changes to the IR should be modeled as passes.

## 4. Default transforms by distribution metadata
- Decision: Distributions declare their default transform (`:log`, `:softplus`, `:logit`).
- Rationale: Centralizes constraints per distribution and mirrors PyMC transforms.
- Assumption: Distributions are univariate with scalar transforms.
- Implication: Transform handling is automatic unless overridden.

## 5. Observations carry metadata
- Decision: Observations carry metadata (likelihood, weight, mask, reduce).
- Rationale: Enables weighted likelihoods, masking, and aggregation without extra graph nodes.
- Assumption: Metadata is sufficient for the target use cases (no need for per-element custom logp).
- Implication: Logprob application must honor metadata.

## 6. Measurable ops lifted by rewrite
- Decision: Measurable `matmul` and `affine` are rewritten into `:meas_obs` nodes.
- Rationale: Keeps measurable logic out of core obs handling and mirrors PyMC logprob rewrites.
- Assumption: Only matmul and affine are needed as measurable ops for now.
- Implication: Future measurable ops should be implemented as passes.

## 7. Deterministic nodes do not contribute to logp
- Decision: Deterministic nodes only feed obs or other dets, they add no logprob directly.
- Rationale: Matches probabilistic semantics and PyMC behavior.
- Assumption: No deterministic transforms require Jacobian corrections outside of measurable ops.

## 8. Tests prioritize numeric correctness
- Decision: All tests use explicit Nx expressions and compare numerically with tolerance.
- Rationale: Avoids Kernel arithmetic and ensures numeric parity for logprob terms.
- Assumption: BinaryBackend float precision is sufficient for test comparisons with tolerances ~1e-5. **[NOTE]** D53 adds f32 support via EMLX; f32 test tolerances may need to be wider (~1e-3).

## 9. Free RVs identified by exclusion
- Decision: Free RVs are RV nodes not targeted by any obs/meas_obs node.
- Rationale: Simple, robust identification — no extra metadata needed on nodes.
- Assumption: An RV is either fully observed or fully free (no partial observation).

## 10. Flat vector holds unconstrained values
- Decision: The sampler's flat vector stores unconstrained values; transforms are applied inside the logp function.
- Rationale: Samplers (HMC, NUTS) operate in unconstrained space. Transforms inside logp keep the interface clean.
- Assumption: All constrained distributions have a known bijective transform to unconstrained space.

## 11. Compiler pre-dispatches at build time
- Decision: The compiler walks IR nodes once at build time, producing closures that are pure Nx ops at runtime.
- Rationale: Elixir-level dispatch (pattern matching, map lookups) happens once; the returned logp_fn is a chain of Nx ops traceable by Nx.Defn.grad.
- Assumption: `Nx.Defn.grad` can trace through all ops used in the closures.

## 12. Obs terms computed eagerly
- Decision: Observation logprob terms are computed eagerly at compile time as constant tensors.
- Rationale: Observed values are constant w.r.t. free RVs in the current IR, so their logprob contribution is fixed.
- Assumption: The target RV's distribution params are all constants (not references to other free RVs).
- Note: Partially superseded by D22 — obs terms with param refs now produce deferred closures.

## 13. Free RVs sorted alphabetically for deterministic layout
- Decision: Free RVs are sorted alphabetically by id in the PointMap.
- Rationale: Ensures deterministic flat-vector layout across runs, independent of map insertion order.
- Assumption: Number of free RVs is small enough that alphabetical sort has negligible cost.

## 14. NUTS uses plain Elixir + Nx, no defn
- Decision: The NUTS sampler is implemented with plain Elixir functions and Nx tensor ops, not `defn`.
- Rationale: Only BinaryBackend is available; `defn` adds tracing complexity without JIT benefit.
- Assumption: BinaryBackend is the only backend in use. If EXLA is added, revisit for JIT benefit. **[EVOLVED]** EXLA (D24) and EMLX (D53) are now available. The sampler loop remains plain Elixir; only `value_and_grad` is JIT-compiled. Decision still holds in spirit.

## 15. Sampler operates in unconstrained space, returns constrained trace
- Decision: The sampler operates entirely in unconstrained (flat f64) space and applies forward transforms when building the trace.
- Rationale: Consistent with D10. HMC/NUTS require unconstrained geometry; users expect constrained outputs.
- Assumption: Forward transforms are cheap to apply post-hoc.

## 16. PRNG via Erlang `:rand` with deterministic seeding
- Decision: Sampler uses `:rand.seed_s(:exsss, seed)` for all random decisions (direction, proposal, momentum).
- Rationale: `Nx.Random.split/uniform` are prohibitively slow with BinaryBackend due to defn tracing overhead.
- Assumption: BinaryBackend is the only backend. With EXLA, Nx.Random would be fast and preferable. **[EVOLVED]** EXLA is available but `:rand` is still preferred — Nx.Random is used only for momentum sampling (D27 step_fn). The tree builder's scalar decisions remain on `:rand` because they execute outside the JIT boundary.

## 17. Scalar math for adaptation, Nx for geometry
- Decision: Dual averaging and Welford use Erlang `float` arithmetic; leapfrog/KE/momentum use `Nx.t()`.
- Rationale: DA is simple scalar arithmetic that doesn't benefit from tensors. Leapfrog must compose with autodiff.
- Assumption: Adaptation parameters are scalars; model dimension is small enough that element-wise Erlang ops are fast.

## 18. Multinomial NUTS (not slice-based)
- Decision: Tree building uses multinomial sampling for proposal selection, not the original slice-based method.
- Rationale: Modern standard per Betancourt 2017. Better exploration and simpler implementation.
- Assumption: Log-sum-exp arithmetic is numerically stable in f64.

## 19. Diagonal mass matrix by default
- Decision: Mass matrix adaptation uses diagonal (element-wise variance) by default, with opt-in dense mode via `dense_mass: true`.
- Rationale: Diagonal is sufficient for most models. Dense captures off-diagonal correlations but requires O(d^2) samples for stable covariance estimation.
- Assumption: Target models have weak posterior correlations or few enough dimensions that diagonal is adequate. Dense mode available for expert users.
- Note: Superseded by D37 — dense mass matrix now available as opt-in.

## 20. Stan-style three-phase warmup with doubling windows
- Decision: Warmup uses three phases (step size only, step size + mass matrix with doubling windows, step size only).
- Rationale: Proven effective schedule from Stan. Doubling windows allow the mass matrix to stabilize progressively.
- Assumption: Warmup budget is >= ~100 iterations (enough for at least one mass matrix window).

## 21. Distributions use Lanczos lgamma (pure Nx ops, differentiable)
- Decision: `Exmc.Math.lgamma` uses the Lanczos approximation (g=7, 9 coefficients) implemented entirely with Nx ops.
- Rationale: No external C dependencies; the implementation is differentiable through `Nx.Defn.grad`. Accurate to ~15 digits for Re(x) > 0.5.
- Assumption: The gradient of lgamma via Lanczos is numerically stable on BinaryBackend. **Known violation on BinaryBackend only:** gradient triggers `Complex.divide` at extreme values. **Resolved by D24:** when EXLA is available, `Compiler.value_and_grad` uses `EXLA.jit` which handles lgamma gradient correctly.
- Implication: Gamma, Beta, and StudentT work as sampled priors when EXLA is available. On pure BinaryBackend, they remain limited to observation-only use. The distinction: when a distribution is a **free RV** (sampled prior), the NUTS sampler must differentiate its logpdf with respect to the variable's own value — this requires `d/dx lgamma(x)`, which crashes on BinaryBackend. When a distribution is an **observation likelihood** with fixed (constant) shape/rate/nu parameters, `lgamma(alpha)` evaluates to a scalar constant that vanishes from the gradient entirely — no differentiation through lgamma occurs, so BinaryBackend handles it correctly.

## 22. Hierarchical params use string references resolved at eval time
- Decision: String values in a distribution's params map (e.g., `%{mu: "parent_mu"}`) reference other RVs and are resolved at evaluation time.
- Rationale: Simple convention that avoids graph-level transformations. Both parent and child RVs remain free in PointMap unless explicitly observed.
- Assumption: String param refs are sufficient for the target hierarchical models (no need for arbitrary expressions).
- Implication: `LogProb.eval` and `Compiler` must resolve string params from the value map before computing logpdf. Obs terms with param refs produce deferred closures (partially supersedes D12).

## 23. Diagnostics use direct ACF computation in Erlang floats
- Decision: `Exmc.Diagnostics.autocorrelation` uses direct summation in Erlang floats. ESS via initial positive sequence estimator (Geyer 1992). Split R-hat (Vehtari et al. 2021).
- Rationale: Direct computation in Erlang floats avoids Nx BinaryBackend overhead for the many small operations in ACF computation. Standard diagnostic methods from the literature.
- Assumption: Sample counts are small enough (< ~10k) that O(n*max_lag) direct ACF is fast. If sample counts grow, switch to FFT-based ACF.

## 24. EXLA used for gradient computation when available
- Decision: `Compiler.value_and_grad` auto-detects EXLA via `Code.ensure_loaded?(EXLA)` and wraps the logp closure in `EXLA.jit`. All IR tensor parameters are copied to BinaryBackend before closure capture to avoid backend conflicts during EXLA tracing.
- Rationale: EXLA handles lgamma gradient correctly (no `Complex.divide` crash), enabling Beta/StudentT/Gamma as sampled priors. The Evaluator (BinaryBackend) is used as fallback when EXLA is not available.
- Assumption: All ops in the logp closures (distribution logpdfs, transforms, Jacobians) are EXLA-traceable. IR parameters captured in closures must be BinaryBackend tensors.
- Implication: D14 (no defn in sampler) still holds — the sampler loop remains plain Elixir. Only the inner `value_and_grad` call is JIT-compiled. D16 (`:rand` PRNG) and D17 (scalar adaptation) are unaffected.

## 25. Numerically stable transforms via softplus identity
- Decision: Transforms (logit, softplus) and their Jacobians use `softplus(x) = max(x, 0) + log1p(exp(-|x|))` instead of `log1p(exp(x))` or `Nx.sigmoid(z)`.
- Rationale: The naive `exp(x)` overflows on BinaryBackend for large x. The rewritten form never overflows. Log-Jacobian uses `-softplus(-z)` instead of `log(sigmoid(z))`.
- Assumption: `Nx.max` and `Nx.abs` subgradients at 0 are handled correctly by Nx's autodiff.

## 26. NaN guards in sampler arithmetic
- Decision: Tree building and step-size search guard against non-numeric values from `Nx.to_number` (`:nan`, `:neg_infinity`, `:infinity`). Non-finite joint_logp is treated as a divergent step.
- Rationale: When the logp function returns `-inf` (e.g., Beta logpdf at sigmoid(z)=1.0), the gradient is NaN. `Nx.to_number` converts these to atoms that crash Erlang arithmetic. Guards allow the sampler to gracefully reject such proposals.
- Assumption: Non-finite logp indicates a region the sampler should avoid, not a bug in the model.

## 27. Fused JIT leapfrog step
- Decision: `Compiler.compile_for_sampling` returns `{vag_fn, step_fn, pm}` where `step_fn` fuses the entire leapfrog step (half-step momentum, position update, value_and_grad, half-step momentum) into a single `EXLA.jit` call. Falls back to `Leapfrog.step(vag_fn, ...)` without EXLA.
- Rationale: Reduces JIT call overhead per leapfrog step from 1 (vag_fn) to 1 (step_fn), but the step_fn compiles the entire leapfrog as one XLA computation. The epsilon scalar is wrapped in `Nx.tensor` with BinaryBackend before each call.
- Assumption: The leapfrog arithmetic (add, multiply, divide) is EXLA-traceable alongside the logp closure.
- Implication: `Tree.build` now takes `step_fn` (arity 5) instead of `vag_fn` (arity 1). Unit tests updated accordingly.

## 28. String param refs resolved to constrained values
- Decision: `resolve_params_constrained` applies the forward transform when resolving string param references from the unconstrained value map. E.g., if sigma has a `:log` transform, `vm["sigma"]` is `log(sigma)`, so the resolver applies `exp()` to get the constrained value before passing it to the distribution's logpdf.
- Rationale: Distribution logpdfs expect params in natural (constrained) space. Without this fix, hierarchical models with constrained parents (e.g., `sigma ~ Exp(1), child ~ N(0, sigma)`) get `sigma=log(sigma)` instead of `sigma=sigma`, causing NaN gradients and sampler failure.
- Assumption: All string param references target free RVs whose transforms are known via PointMap entries.
- Implication: Partially supersedes D22 — string refs now require transform-aware resolution, not just map lookup.

## 29. Distribution sample callbacks for forward sampling
- Decision: All 9 distributions implement `sample(params, rng) :: {Nx.t(), rng}` using Erlang `:rand` for scalar random draws. The callback is optional (`@optional_callbacks [sample: 2]`).
- Rationale: Enables prior predictive and posterior predictive sampling via `Exmc.Predictive` without running MCMC. Pure Erlang arithmetic avoids Nx overhead for scalar samples.
- Assumption: Scalar sampling is sufficient (no vectorized/batched sampling needed).

## 30. Predictive module uses topological walk
- Decision: `Exmc.Predictive.prior_samples` topologically sorts RV nodes (Kahn's algorithm) and samples each in order, resolving string param refs from already-sampled ancestors. `posterior_predictive` resolves obs target params from the trace and samples from the likelihood.
- Rationale: Topological sort handles hierarchical models where child RVs depend on parent RVs. The walk is O(V+E) and happens once per call.
- Assumption: The model DAG is acyclic (guaranteed by construction).

## 31. Feature parallelism: NCP and WAIC first, Vectorized Obs after
- Decision: Develop Non-Centered Parameterization and WAIC/LOO in parallel (Phase I), then Vectorized Observations (Phase II).
- Rationale: NCP and WAIC touch orthogonal layers (rewrite vs diagnostics). Vectorized Obs restructures core data flow in compiler.ex and PointMap, which both NCP and WAIC depend on.
- Assumption: Vectorized Obs will extend, not replace, the scalar obs API.

## 32. Automatic non-centered parameterization via rewrite pass
- Decision: The `NonCenteredParameterization` rewrite pass automatically transforms free Normal RVs where both `mu` and `sigma` are string references into standard Normal `N(0,1)` with NCP metadata. The compiler reconstructs `x = mu + sigma * z` when resolving param references. The sampler reconstructs constrained values in `build_trace`.
- Rationale: Eliminates funnel geometry in hierarchical Normal models. The standard normal prior `N(z|0,1)` has uniform curvature, making NUTS sampling more efficient for weakly-identified parameters.
- Assumption: Auto-NCP is always beneficial for hierarchical Normals with both parent params as string refs. Models where centered is better (highly informative data) may need future opt-out support.
- Implication: `resolve_ref` in the compiler handles recursive NCP reconstruction for nested hierarchies. `IR.ncp_info` stores the original `%{mu:, sigma:}` sources. `compile_for_sampling` returns a 4-tuple `{vag_fn, step_fn, pm, ncp_info}`.

## 33. WAIC/LOO via pointwise log-likelihood on constrained trace
- Decision: `ModelComparison.pointwise_log_likelihood` evaluates per-observation log-likelihood directly from the constrained trace (not via the compiler's flat-vector interface). WAIC uses `lppd - p_waic` (Vehtari et al. 2017). LOO uses basic importance sampling.
- Rationale: Working with constrained trace values avoids the unconstrained ↔ constrained round-trip. For NCP'd models, the trace already contains reconstructed values, so obs logp evaluation works transparently.
- Assumption: Vector obs are supported via per-element pointwise evaluation (D34).
- Implication: `Compiler.compile_pointwise` provides an alternative compilation path for tagged per-obs closures. `ModelComparison.compare` ranks models by IC.

## 34. Vectorized observations auto-reduce with :sum
- Decision: `Builder.obs` auto-adds `reduce: :sum` to obs metadata when the observation value is a non-scalar tensor (rank > 0) and no explicit `reduce` option is set. `ModelComparison` strips `:reduce` from metadata for pointwise evaluation to get per-element log-likelihoods.
- Rationale: Distribution logpdfs already broadcast (e.g., `Normal.logpdf(Nx.tensor([1,2,3]), %{mu: scalar, sigma: scalar})` returns shape `{3}`). The compiler's `apply_obs_meta` already handles `:sum`/`:mean`/`:logsumexp` reduction. The only missing piece was auto-detecting vector obs and stripping reduce for WAIC pointwise evaluation.
- Assumption: Vector obs are i.i.d. given the distribution params. The sum-reduction produces the same logp as N separate scalar obs nodes.
- Implication: Users can pass `Nx.tensor([...])` to `Builder.obs` instead of creating one RV+obs pair per data point. `ModelComparison.pointwise_log_likelihood` returns `{obs_id, index}` tuple keys for vector obs.

## 35. Parallel chains via Task.async_stream
- Decision: `Sampler.sample_chains` compiles the model once and dispatches chains in parallel via `Task.async_stream` with `ordered: true`. Parallel is the default (`parallel: true`). `max_concurrency:` controls the number of concurrent chains.
- Rationale: Each chain is fully independent — `:rand` uses explicit-state `_s` functions (no process dictionary), Nx tensors are immutable, EXLA JIT closures are thread-safe. Compiling once avoids redundant IR rewriting and EXLA JIT compilation per chain. On multi-core machines this gives near-linear speedup.
- Assumption: No shared mutable state between chains. EXLA's XLA compilation cache is internally thread-safe.
- Implication: `sample/3` delegates to `sample_from_compiled/3` (private). `sample_chains` accepts `:init_values`, `:parallel`, and `:max_concurrency` options.

## 36. NCP init values must be inverted to raw space
- Decision: `init_position` calls `invert_ncp_init/2` to convert user-provided constrained init values to NCP raw space before packing into the unconstrained vector. For NCP'd variable `x = mu + sigma * z`, the inversion is `z = (x - mu) / sigma`.
- Rationale: After NCP rewrite, the free variable is `z ~ N(0,1)`, not `x`. Without inversion, user's constrained value (e.g., `alpha=4.0`) is placed directly as `z=4.0` — 4 standard deviations from the prior mode. This poisons the mass matrix during warmup and produces tiny step sizes.
- Assumption: NCP info (`%{id => %{mu:, sigma:}}`) is available from compilation. Init values for NCP parents (mu, sigma sources) must also be provided.
- Implication: `compile_for_sampling` returns a 4-tuple `{vag_fn, step_fn, pm, ncp_info}`. All call sites of `init_position` updated to pass `ncp_info`. Benchmark showed 2x ESS/s improvement for hierarchical models.

## 37. Dense mass matrix as opt-in via `dense_mass: true`
- Decision: Dense (full covariance) mass matrix available via `dense_mass: true` option, defaulting to diagonal. Dense mode tracks the full covariance matrix via Welford, computes Cholesky decomposition at window boundaries, and samples momentum via `p = L^{-T} @ z`.
- Rationale: Dense mass captures off-diagonal posterior correlations (mu-sigma funnel) that diagonal cannot. However, empirical testing showed noisy covariance estimates from ~25-75 samples for d=5 matrices *increased* divergences (49 vs 9) and *decreased* ESS/s (0.2 vs 2.1) for the medium benchmark model. The regularization (shrink toward sample diagonal) helped but did not overcome the fundamental sample-size limitation.
- Assumption: Dense mode may help for larger models (d>10) where cumulative warmup samples >> d^2, or with longer warmup budgets. Currently, the number of effective warmup samples per window is too small relative to d^2 for reliable covariance estimation.
- Implication: `MassMatrix.init_dense/1`, `finalize_dense/1` with Cholesky. Sampler threads `chol_cov` through warmup/sampling. `build_generic_step_fn` dispatches on mass matrix shape.

## 38. Shape-dispatched mass matrix in leapfrog and tree builder
- Decision: `Leapfrog.mass_times_p/2` dispatches on `Nx.rank(inv_mass)`: rank 1 (diagonal) uses element-wise multiply, rank 2 (dense) uses matrix-vector dot. All call sites (position update, kinetic energy, U-turn check) use this single dispatcher.
- Rationale: Avoids separate code paths for diagonal vs dense throughout the sampler. The rank check is a cheap Elixir pattern match that happens once per call.
- Assumption: Only diagonal `{d}` and dense `{d,d}` mass matrices are needed. No block-diagonal or sparse variants.
- Implication: `kinetic_energy` and `step` work transparently for both modes. Note: U-turn check no longer uses `Leapfrog.mass_times_p` — see D43 (rho-based criterion uses direct list arithmetic).

## 39. Speculative pre-computation for NUTS tree builder
- Decision: Before entering the NUTS tree doubling loop, speculatively pre-compute a budget of leapfrog states for both forward (+epsilon) and backward (-epsilon) directions using `multi_step_fn` (a JIT-compiled XLA while-loop). The tree builder then reads from this pre-computed buffer instead of calling `step_fn` per leaf.
- Rationale: The original tree builder interleaved JIT calls (leapfrog via EXLA) with Elixir tree logic (merges, U-turn checks). Each JIT call has ~100-200us overhead for kernel launch + PCIe transfer. By batching all leapfrog steps into 2 JIT calls (one per direction), this overhead is amortized. For d=5 medium model, this reduced per-tree JIT overhead from ~4ms (10+ JIT calls) to ~500us (2 JIT calls).
- Assumption: The XLA while-loop in `multi_step_fn` is numerically equivalent to sequential `step_fn` calls. Budget can be bounded by `2^max_depth - 1` per direction. Pre-computed states fit in memory.
- Implication: `Compiler.compile_for_sampling` returns a 5-tuple `{vag_fn, step_fn, pm, ncp_info, multi_step_fn}`. `Tree.build_speculative` pre-computes states, slices into subtrees. States are `backend_copy`'d to BinaryBackend for fast scalar ops in the tree builder.

## 40. Full-tree NIF: entire NUTS tree in a single Rust NIF call
- Decision: When speculative pre-computation provides both forward and backward chains, the entire tree build (direction choices, subtree construction, merges, U-turn checks, termination) can be performed in a single Rust NIF call (`NativeTree.build_full_tree_bin`). Architecture: 2 JIT calls (pre-compute) + 1 NIF call (tree build) per NUTS iteration.
- Rationale: Eliminates ALL Elixir merge overhead (~200us/merge × 4 merges = 800us per tree for medium model). For the simple model (d=2, tree depth ~2), this gives 1.93x PyMC ESS/s (785 vs 406). The NIF handles direction sampling via Rust Xoshiro256**, subtree construction via existing `build_subtree`, and multinomial trajectory merging — all in contiguous memory without Elixir map allocation.
- Assumption: Pre-computed budget is sufficient for the tree (bounded by adaptive budget mechanism, D41). Rust PRNG produces valid MCMC trajectories (different from Erlang PRNG, but statistically equivalent). NIF scheduling on DirtyCpu doesn't starve the BEAM scheduler.
- Implication: `Tree.build` checks `Application.get_env(:exmc, :full_tree_nif, true)` + infrastructure availability. Falls back to speculative+inner-NIF path on failure via try/rescue. `exmc_tree` Rust crate grows to ~500 lines with `build_full_tree`, `PrecomputedStates::slice`, and `PrecomputedStates::len` methods.

## 41. Adaptive budget for full-tree NIF pre-computation
- Decision: Instead of pre-computing `2^max_depth - 1` states (up to 1023 per direction), track the maximum observed tree depth via process dictionary (`Process.get(:exmc_max_tree_depth_seen)`) and set budget to `2^(max_seen + 1) - 1`. Default budget for first tree: `2^4 - 1 = 15`. Reset tracker at start of each sampling run.
- Rationale: Fixed budget of 1023 wastes ~97% of JIT compute for shallow trees (simple model, depth ~2). Adaptive budget tracks actual tree depth and grows organically. For simple (depth 2, budget 15): 4us × 30 = 120us wasted, negligible. For medium (depth 5, budget 63): 4us × 126 = 504us, acceptable. Bounds check in Rust gracefully terminates the tree if budget is exhausted.
- Assumption: Tree depth is stable within a sampling run after warmup converges. Process dictionary access is effectively free.
- Implication: `build_full_tree_nif` reads/writes `Process.get/put(:exmc_max_tree_depth_seen)`. `sample_from_compiled` resets tracker to 0 at start. Rust `build_full_tree` checks `fwd_states.len()` / `bwd_states.len()` before each doubling to prevent out-of-range panics.

## 42. No hybrid dispatch: full-tree NIF unconditionally when eligible
- Decision: Do NOT gate full-tree NIF on observed tree depth or model dimension. When NIF infrastructure is available, always use full-tree NIF. Users can opt out via `Application.put_env(:exmc, :full_tree_nif, false)`.
- Rationale: Attempted hybrid dispatch (gate on `observed_max_depth <= 3`) failed because early warmup iterations produce temporarily deep trees (garbage mass matrix), permanently disabling the NIF for the rest of the run. Dimension-based gating (d <= 3) is too simplistic. The adaptive budget (D41) already mitigates over-speculation for deeper trees. The remaining overhead for medium/stress is the JIT pre-computation cost, which is bounded by the adaptive budget.
- Assumption: Users running models where speculative path is faster (medium, stress) can toggle `:full_tree_nif` to false. Default-on favors the common case (simple/medium models).
- Implication: `Tree.build` dispatch logic is simple: `use_full_tree = nif_eligible`. No depth tracking for dispatch decisions. Depth tracking only used for adaptive budget sizing.

## 43. Generalized (rho-based) U-turn criterion
- Decision: Replace endpoint-displacement U-turn criterion `(q⁺-q⁻) · (M⁻¹p)` with the generalized criterion from Betancourt 2017: `ρ · (M⁻¹p±) < 0` where `ρ = Σ pᵢ` is the cumulative momentum sum over all trajectory points. Track `rho_list` through tree nodes: leaves initialize `rho = p_new`, merges accumulate `rho = rho_first + rho_second`, trajectory-level merges accumulate `rho = rho_traj + rho_subtree`.
- Rationale: The endpoint criterion computes `Σⱼ (q⁺ⱼ - q⁻ⱼ) × inv_massⱼ × pⱼ` — the extra `inv_mass` factor per dimension makes the dot product dominated by high-variance components. For the stress model (3-group hierarchical), inv_mass ranges 200x (0.028 for mu_pop to 5.6 for noise params), causing premature tree termination driven by mu_pop alone. The rho criterion `Σⱼ ρⱼ × inv_massⱼ × pⱼ` avoids this bias because `ρ` naturally scales with trajectory length rather than endpoint separation. Measured impact: stress model ESS/s improved from ~67 to 89 (+33%), avg tree depth 2.2→2.9, step size 0.46→0.60. Warmup divergences increased (5→31) because the adapted trajectory is different, but sampling divergences dropped to ~1/1000.
- Assumption: The momentum sum `ρ` is a sufficient statistic for the U-turn criterion. This holds for the standard NUTS multinomial variant (Betancourt 2017, §4.1). Dense mass matrix U-turn checks must also use `ρ` (currently only diagonal is implemented). Supersedes D38's mention of `Leapfrog.mass_times_p` for U-turn — that function is no longer used; U-turn checks use direct `rho × inv_mass` list arithmetic.
- Implication: Both Elixir (`check_uturn_rho` + `zip_reduce_rho` in tree.ex) and Rust (`check_uturn` in uturn.rs) paths updated. NIF subtree returns `rho_bin` for trajectory-level accumulation in Elixir. Test divergence tolerances relaxed from <5 to <20 for simple models (warmup path sensitivity). Gap vs PyMC: simple 1.93x, medium 1.02x, stress 0.57x (was 0.41x).

## 44. Skip divergent samples in Welford mass matrix update
- Decision: During Phase II warmup, skip the Welford mass matrix update when a divergent transition occurred. Check `state.divergences` before and after `nuts_step_warmup`; if incremented, do not call `MassMatrix.update`.
- Rationale: Stan documents that "samples generating a divergent transition are excluded from the estimation of the covariance matrix." Divergent trees are truncated early, and the multinomial-selected proposal from a truncated tree is biased toward the starting point. Including these biased samples inflates variance estimates in the mass matrix, degrading adaptation quality — especially for models with many warmup divergences (stress: ~30-40).
- Assumption: Step size adaptation (`StepSize.update`) should still use the divergent iteration's accept_stat, matching Stan behavior where DA uses all iterations.
- Implication: Phase II inner loop in `do_warmup` compares `state.divergences` before/after the step. Combined with term_buffer=50 (D45), measured +109% medium ESS/s, +180% stress ESS/s (1ch 5-seed median).

## 45. term_buffer=50 (matching Stan)
- Decision: Set `term_buffer = 50` unconditionally, matching Stan's default. Previously used `min(200, max(50, div(num_warmup, 5)))`.
- Rationale: After the log_epsilon_bar initialization fix (D26/lesson #26), dual averaging converges in ~50 iterations since it starts from the correct step size rather than biasing toward eps=1.0. The old term_buffer=200 allocated 200 iterations for Phase III (final step size adaptation) and only 725 for Phase II (mass matrix windows). With term_buffer=50, Phase II gets 875 iterations — the final mass matrix window grows from ~350 to ~500 samples (+43%), directly improving adaptation quality for d>=5.
- Assumption: DA converges in 50 iterations when initialized correctly. This holds after the log_epsilon_bar fix; without it, DA needed ~200 iterations due to bias toward eps=1.0.
- Implication: Window schedule with num_warmup=1000: Phase I 0-74 (75), Phase II 75-949 (875), Phase III 950-999 (50). Tests with small num_warmup (200-500) have slightly different RNG trajectories due to changed window boundaries; test tolerances widened accordingly.

## 46. Sub-trajectory U-turn checks (PyMC-style)
- Decision: Perform 3 U-turn checks at each tree merge instead of 1. Check 1 (existing): full trajectory `ρ · (M⁻¹p±) < 0`. Check 2 (new): `(ρ_left + p_right_first) · (M⁻¹p)` — left sub-trajectory plus first momentum of right. Check 3 (new): `(p_left_last + ρ_right) · (M⁻¹p)` — last momentum of left plus right sub-trajectory. Gated on `first.depth > 0` in `merge_subtrees` (leaf merges skip). Always-on in `merge_trajectories` and `merge_into_trajectory`.
- Rationale: PyMC and Stan perform these additional checks to detect U-turns at the junction between merged sub-trajectories, not just across the full trajectory. Without checks 2-3, the sampler can miss U-turns that form at the merge boundary, producing longer trajectories that waste compute without improving ESS. Pure Elixir path: medium +46% ESS/s (82→120), warmup divergences -41% (34→20). The sub-trajectory checks are undocumented in published papers — discovered by reading Stan source code (`nuts.hpp`).
- Assumption: Tree nodes track `p_left` and `p_right` (endpoint momenta) and `rho` (cumulative momentum sum). Direction-aware assignment is critical: in `merge_subtrees`, `left_sub` and `right_sub` must be assigned based on `going_right` to correctly orient the sub-trajectories.
- Implication: Both Elixir (`merge_subtrees`, `merge_trajectories`, `merge_into_trajectory` in tree.ex) and Rust (`merge_subtrees`, `merge_into_trajectory` in tree.rs) paths updated. More early termination means the full-tree NIF wastes more pre-computed states, making the speculative path preferable for all models.

## 47. NIF direction fix for sub-trajectory U-turn checks
- Decision: In `build_full_tree` (Rust), pass the actual `go_right` direction to `build_subtree` for backward chains instead of hardcoded `true`.
- Rationale: The full-trajectory U-turn check is symmetric — swapping `p_left`/`p_right` produces the same result — so passing `going_right=true` for backward chains was harmless before D46. Sub-trajectory checks break this symmetry: they assign `left_sub`/`right_sub` based on direction, so backward chains with `going_right=true` got reversed endpoint assignments, producing incorrect U-turn decisions. This caused severe ESS regression (simple 785→160, medium 115→65, stress 89→51) when sub-trajectory checks were first added to the NIF.
- Assumption: All tree-building code paths must propagate the correct direction. This is a class of bug that symmetric criteria mask — any future asymmetric criterion must audit direction propagation.
- Implication: One-line fix in `tree.rs` `build_full_tree`: backward chain calls `build_subtree(..., go_right, ...)` instead of `build_subtree(..., true, ...)`.

## 48. Full-tree NIF disabled by default
- Decision: Set `Application.get_env(:exmc, :full_tree_nif, false)` — default to speculative path. Users can opt in via `Application.put_env(:exmc, :full_tree_nif, true)`.
- Rationale: Sub-trajectory checks (D46) cause more early tree termination, increasing the fraction of wasted pre-computed leapfrog states in the full-tree NIF. After the direction fix (D47), benchmarks showed the NIF is slower than the speculative path for all models: the extra JIT pre-computation cost exceeds the NIF merge savings when trees terminate early. 5-seed medians (speculative): simple 233, medium 116, stress 107 ESS/s. The full-tree NIF remains available for models where it wins (e.g., very shallow trees).
- Assumption: The speculative+inner-NIF path is the best default for general models. Full-tree NIF may become competitive again if pre-computation can be made lazier (compute-on-demand instead of batch-ahead).
- Implication: Reverses D40's default-on for full-tree NIF. `Tree.build` checks `Application.get_env(:exmc, :full_tree_nif, false)`. Simple model benchmark drops from 785 to 233 ESS/s (speculative path), but medium and stress improve.

## 49. Uncap multinomial log_weight at tree leaves
- Decision: In `build_subtree` depth 0, change `log_weight = min(0.0, d)` to `log_weight = d` where `d = joint_logp_new - joint_logp_0`.
- Rationale: The `min(0.0, d)` cap was capping multinomial weights at `exp(0) = 1`, so trajectory points with *better* energy than the starting point (d > 0) were underweighted. This biased multinomial selection toward q_0, inflating the duplicate rate. The `accept_prob = min(1, exp(min(d, 0)))` cap for DA feedback is correct (standard MH ratio), but the log_weight for multinomial proposal selection must be uncapped. Discovered via diagnostic: eXMC had 37.7% duplicate samples vs PyMC's 7.8%.
- Assumption: Trajectory points can have d > 0 or d < 0, and the multinomial should weight them accordingly.
- Implication: Combined with D50, reduced duplicate rate from 37.7% to 6.5%, ESS improvement 2-3x across all models.

## 50. Biased progressive sampling for outer tree merge
- Decision: In `merge_trajectories`, change acceptance probability from `exp(subtree.lsw - combined_lsw)` (balanced multinomial) to `min(1, exp(subtree.lsw - traj.lsw))` (biased progressive, matching Stan/PyMC).
- Rationale: Stan and PyMC use biased progressive sampling (Betancourt 2017, Appendix A.3.2) for the outer merge (trajectory extension), while using balanced multinomial for the inner merge (within `build_subtree`). The balanced formula gives `P(accept subtree) = w_subtree / (w_traj + w_subtree)`, while biased progressive gives `P = min(1, w_subtree / w_traj)`. When the subtree outweighs the existing trajectory, biased progressive always accepts (P=1), while balanced gives P < 1. This makes the balanced formula "sticky" on q_0 — the starting point survives merges more often than it should. PyMC's `index_in_trajectory` analysis showed only 3.7% q_0 selection rate at depth 2 (vs balanced prediction of 25%). Inner merges (`merge_subtrees`) correctly use balanced multinomial in all implementations.
- Assumption: Biased progressive sampling is a valid MCMC proposal mechanism that preserves the target distribution (proven in Betancourt 2017).
- Implication: Implemented in both Elixir `merge_trajectories` and Rust NIF `merge_into_trajectory`. Uses `log(U) < (subtree.lsw - traj.lsw)` to avoid overflow. 1-chain 5-seed race: simple 469 (0.81x PyMC), medium 298 (1.90x PyMC), stress 215 (1.16x PyMC). eXMC now beats PyMC on medium and stress models.

## 53. Precision portability via `Exmc.JIT.precision()`
- Decision: Replace all hardcoded `:f64` type annotations with `Exmc.JIT.precision()`, which returns `:f32` for EMLX and `:f64` for EXLA. Applied across 8 files: `tree.ex`, `sampler.ex`, `mass_matrix.ex`, `leapfrog.ex`, `batched_leapfrog.ex`, `distributed.ex`, `point_map.ex`, `transform.ex`.
- Rationale: EMLX (Apple Metal GPU backend) is f32-only. Hardcoded `:f64` annotations cause silent type mismatches or crashes when tensors flow between EMLX (f32) and code expecting f64. A single dispatch point (`JIT.precision()`) makes the entire codebase precision-portable.
- Assumption: f32 precision is sufficient for MCMC sampling on models where EMLX is used. Numerical range is narrower (exp overflow at ~88 vs ~709 for f64), requiring tighter clamping (D54).
- Implication: The same `Sampler.sample` call works on both EXLA (f64) and EMLX (f32) with zero code changes. Benchmark results are backend-dependent but posteriors match within f32 tolerance.

## 54. `Nx.clip` broken gradient workaround — use `Nx.max`/`Nx.min`
- Decision: Replace `Nx.clip(z, lo, hi)` with `Nx.max(Nx.tensor(lo), Nx.min(z, Nx.tensor(hi)))` in `Transform.apply(:log, z)` and `Transform.log_abs_det_jacobian(:log, z)`.
- Rationale: `Nx.clip` has broken gradient in Evaluator autodiff — when composed with `Nx.exp()` inside `value_and_grad` closures, the gradient is incorrect (measured 1.22 vs correct 1.105). `Nx.max` and `Nx.min` produce correct gradients in the same composition. The bug is specific to the Evaluator backend; EXLA's `Nx.clip` gradient is correct. Since EMLX falls back to Evaluator for `value_and_grad`, this is a blocking issue for macOS support.
- Assumption: `Nx.max`/`Nx.min` subgradients are correct in all Nx backends (confirmed by test). The Evaluator `Nx.clip` gradient bug may be fixed in a future Nx release.
- Implication: Transform clamping now uses precision-dependent ranges via `exp_safe_range/0`: `{-20, 20}` for f32 (sigma ∈ [2e-9, 5e8]), `{-200, 200}` for f64 (sigma ∈ [1e-87, 7e86]).

## 55. BinaryBackend numerical safety — distribution scale guards
- Decision: All distributions that divide by a scale parameter now floor it at `1.0e-30`: `safe_sigma = Nx.max(sigma, Nx.tensor(1.0e-30))`. Applied to 10 distributions: Normal, HalfNormal, Laplace, Cauchy, HalfCauchy, Lognormal, StudentT, TruncatedNormal, GaussianRandomWalk, Censored.
- Rationale: On BinaryBackend (used by EMLX/Evaluator), Erlang arithmetic throws `ArithmeticError` on divide-by-zero, unlike GPU backends which silently return NaN/Inf per IEEE 754. During NUTS tree building, divergent trajectories can produce extreme q values that, after `exp()` transform, yield sigma ≈ 0 — triggering a divide-by-zero crash in the logpdf. The `1.0e-30` floor prevents the crash while producing a very large negative logpdf that the tree builder correctly treats as divergent.
- Assumption: A scale of `1.0e-30` is small enough to never affect valid posterior regions. The resulting logpdf is finite (not NaN/Inf) and correctly signals low probability.
- Implication: Combined with D54 (transform clamping) and D56 (tree crash resilience), this forms a three-layer defense against BinaryBackend arithmetic crashes during NUTS sampling.

## 56. Tree builder crash resilience — try/rescue and NaN-safe divergent leaves
- Decision: Two changes to `tree.ex` build_subtree base case: (A) Wrap the `step_fn` call in `try/rescue` — on `ArithmeticError`, return a divergent leaf at the starting position instead of crashing the tree. (B) When divergent is detected (NaN `joint_logp` via `is_number/1` guard), fall back to original `q`/`p` for all flat lists instead of using `q_new`/`p_new` which may contain NaN values.
- Rationale: Even with scale guards (D55), step_fn can produce NaN in momentum without throwing. `Nx.to_flat_list` converts NaN tensors to `:nan` atoms, and Erlang `+` on `:nan` atoms crashes with `ArithmeticError` in `zip_add`. Falling back to original (pre-step) position/momentum for flat lists ensures only valid floats reach Erlang arithmetic. The try/rescue catches any remaining arithmetic failures from the BinaryBackend/Evaluator path.
- Assumption: Divergent leaves with original q/p are valid NUTS proposals (they have zero acceptance probability and are almost never selected by multinomial sampling). The try/rescue has zero cost on the non-faulting path (BEAM exception registration).
- Implication: Extends the fault tolerance hierarchy from Chapter 3 to cover BinaryBackend-specific failure modes. The existing NaN detection (leaf level) now has a fallback path that prevents atom-arithmetic crashes.

## 57. Batched leapfrog type cast for Evaluator fallback
- Decision: After `value_and_grad` inside the `Nx.Defn.while` loop in `batched_leapfrog.ex`, cast results back to target precision: `logp_new = Nx.as_type(logp_new, fp)` and `grad_new = Nx.as_type(grad_new, fp)`.
- Rationale: The Evaluator/BinaryBackend returns f64 tensors from `value_and_grad` (BinaryBackend's default precision) even when the while-loop accumulator tensors are f32. This causes a while-loop type mismatch (`CompileError`: body returns f64 but initial arguments are f32). The explicit cast ensures type consistency regardless of which backend executes `value_and_grad`.
- Assumption: The f64→f32 downcast does not lose information relevant to MCMC sampling (the f32 precision is already the working precision for the entire model on EMLX).
- Implication: Batched leapfrog now works on both EXLA (f64, cast is a no-op) and EMLX/Evaluator (f64→f32 cast). This was a blocking issue for EMLX support — without it, all Livebook notebooks crashed.

## 58. Purpose-built MLX NIF — architecture and negative result
- Decision: Built a C++ NIF (`exmc_mlx_nif.cpp`, ~850 lines) that serializes Exmc's model IR as Erlang terms, reconstructs the logp computation in C++ using MLX ops, and calls `mlx::core::value_and_grad()` with Metal kernel fusion. Bypasses EMLX's Evaluator delegation entirely. Implemented 4 NIF functions: `compile_model/1`, `value_and_grad/2`, `leapfrog_step/6`, `batched_leapfrog/7`. Added serializer (`Exmc.MLX.Serializer`), compiler (`Exmc.MLX.Compiler`), and NIF bindings (`Exmc.MLX.Native`).
- Rationale: EMLX delegates `__jit__`/`__compile__` to `Nx.Defn.Evaluator`, causing every Nx op to dispatch individually through the NIF boundary (15-71x slower than EXLA). MLX provides native `compile()` and `value_and_grad()` which EMLX doesn't use. The hypothesis was that fusing the logp computation in C++ and using native autodiff would recover 10-30x of the gap.
- Result: **Negative**. The MLX NIF is 0.4-0.8x *slower* than EMLX+Evaluator (three-way race 2026-02-14). Root cause: the NIF boundary crossing overhead dominates — each leapfrog step requires `Nx.to_binary()` → NIF call (dirty CPU) → `Nx.from_binary()` round-trip, plus recreating the `value_and_grad` function on each call. For the simple model (2 params), NIF adds 3s overhead. For eight_schools (10 params), 2.5x slower.
- Assumption: MLX native autodiff produces correct gradients (verified: logp and grad match Evaluator within 1e-6 for Normal/Exponential model).
- Implication: The bottleneck for EMLX is NOT the lack of kernel fusion or native autodiff — it's the granularity of NIF dispatch. The Evaluator path dispatches ~50 Nx ops per logp evaluation through EMLX's NIF boundary. The MLX NIF replaces those ~50 calls with 1 NIF call, but adds binary serialization overhead that's worse. The real fix is upstream: EMLX needs to implement `__jit__`/`__compile__` to batch operations, or Nx needs a "compiled function" protocol. Code is retained as `config :exmc, :mlx_nif, true/false` for future optimization.

## 52. Distributed 4-node sampling with compile options fix
- Decision: (1) Thread `[:ncp, :device]` compile options through `Distributed.sample_chains` to `Compiler.compile_for_sampling` in `run_coordinator_warmup`, `run_chain_local`, and `run_chain_remote`. (2) Created `benchmark/dist_bench.exs` for 4-node `:peer` distributed benchmarking.
- Rationale: `distributed.ex` called `Compiler.compile_for_sampling(ir)` without passing `ncp: false` or `device` options, so distributed sampling always used NCP (wrong for medium/stress models where `ncp: false` is 9x better). Additionally, local 4-chain via `sample_chains_compiled` runs chains sequentially on a single BEAM (Elixir NIF scheduling prevents true CPU parallelism), while PyMC uses 4 OS processes. Using 4 `:peer` nodes gives same parallelism model.
- Assumption: `:peer` nodes on the same machine provide true OS-process parallelism equivalent to PyMC's multiprocessing.
- Implication: 4-chain distributed results (seed=42): simple 1665 ESS/s (3.4x local), medium 812 ESS/s (3.5x local), stress 628 ESS/s (3.7x local). Near-linear scaling from 5 nodes (coordinator + 4 peers, each running 1 chain; coordinator chain runs concurrently with peers). Distributed overhead is minimal — dominated by per-node JIT compilation (cold start).

## 62. Generation counter for async fault isolation
- Decision: Each sampling client maintains a monotonic `sample_gen` counter. Incremented on every submission. The gen is passed through the compute pool and returned in result casts. Client ignores results where `gen != current_gen`.
- Rationale: If a client process crashes and restarts while an async sampling task is running, the task finishes and casts its result to the new process (re-registered under the same name). Without gen checking, the new process would accept a stale posterior from the old process's job.
- Assumption: Registry name re-registration is the mechanism for process identity. Monotonic counter is sufficient (no wraparound concern for u64 range).
- Implication: Stale results are silently dropped. The pattern generalizes to any system where async jobs may outlive their requestor.

## 68. BEAM scheduler pinning (+sbt tnnps) for NUTS throughput
- Decision: Launch BEAM node with `+sbt tnnps` (thread no-node processor spread). NUMA-aware core pinning for all schedulers.
- Rationale: Benchmark: unbound 0.73 j/s → tnnps 0.98 j/s at 44 concurrent (+34%). Sequential: 4.9s → 3.8s (+23%). Prevents OS from migrating scheduler threads between cores, keeping L1/L2 cache warm for JIT'd EXLA code. Thread-spread (`ts`) is best for sequential but tnnps wins at high concurrency due to NUMA-aware distribution.
- Assumption: Machine has NUMA topology (multi-socket or chiplet design like Ryzen). Single-socket machines may see `ts` perform equally. The 88-scheduler Ryzen has 2 CCDs with 4 CCXes each — tnnps distributes across them optimally.
- Implication: Set via `--erl "+sbt tnnps"`. Benchmark script: `benchmark/cpu_pinning_bench.exs`.

## 69. EXLA JIT cache leak fix: obs data as JIT argument
- Decision: Observation data tensors are registered via `Builder.data(ir, tensor)` and passed as JIT function arguments (XLA parameters), not captured in closures (XLA constants). `Compiler`, `BatchedLeapfrog` carry a `data` argument through all JIT'd functions. `@data_sentinel` (scalar 0.0) used when model has no data.
- Rationale: EXLA caches compiled executables by graph hash. Tensors captured in closures become XLA constants — each unique VALUE produces a new cached executable in native C++ memory, never freed. With many clients updating frequently: cache entries grow unbounded, consuming gigabytes of native memory. Passing as JIT argument means EXLA caches by SHAPE only — same-shape data reuses one executable.
- Assumption: All observation data for a model has the same shape across updates (rolling window capped at `@max_history 200`). If shape changes, a new compilation occurs (but only one per unique shape, not per unique value).
- Implication: IR has `data: nil` field. `Builder.data/2` API. Compiler `logp_fn` is 2-arg `(flat, data)`. Public `compile/2` wraps back to 1-arg. Ported to OSS repo.

## 71. Warm-start NUTS for sequential model updates
- Decision: `Sampler.sample/3` accepts `warm_start: previous_stats` option. When provided, reuses the previous run's mass matrix (`inv_mass_diag`) and step size (`step_size`), running only 50 warmup iterations for fine-tuning instead of full warmup (200-500). Callers pass `state.stats` from the previous posterior update.
- Rationale: Sequential model updates (e.g., streaming data arriving periodically) re-sample the same model with new observations. Each cold-start wastes 200+ warmup iterations re-discovering the same mass matrix. With warm-start, the adapted mass matrix carries over — only the step size needs minor adjustment. Measured speedup: 5.8x (1979ms → 339ms per sampling round).
- Assumption: The model structure (number of free RVs, their transforms) doesn't change between sampling rounds. If the model changes (new RVs added/removed), warm-start must be skipped (cold start). The `PointMap` size must match.
- Implication: First sampling round: full warmup. All subsequent rounds: warm-start with 50-iteration fine-tune. No changes to the public `Sampler.sample/3` API — `warm_start` is an optional kwarg. Backward compatible.

## 72. Bootstrap Particle Filter for discrete state models
- Decision: New module `Exmc.SMC.ParticleFilter` implements the Bootstrap Particle Filter (BPF) from Gordon et al. (1993). Supports state-space models where the likelihood is intractable (discrete transitions like Binomial SEIR) but simulation is feasible. Uses stratified resampling with ESS-based trigger.
- Rationale: NUTS requires a differentiable logp. Models with discrete state transitions (Binomial SEIR, HMMs with discrete hidden states) cannot use NUTS directly. The particle filter provides likelihood estimates via sequential importance sampling, enabling Bayesian inference on this class of models. Also required as the inner loop for SMC² (D73).
- Assumption: The model can be specified as `{init, transition, observation_logp}` where transition produces new states given old states + RNG, and observation_logp evaluates the observation density. States are plain Elixir terms (maps), not Nx tensors.
- Implication: `ParticleFilter.filter/3` for full filtering, `ParticleFilter.filter_window/4` for windowed likelihood (used by O-SMC²). No Nx/EXLA dependency — pure Elixir. Model spec: `%{init: fn(rng), transition: fn(state, t, rng), observation_logp: fn(state, y_t)}`.

## 73. O-SMC² for online joint parameter and state inference
- Decision: New modules `Exmc.SMC.OnlineSMC2` and `Exmc.SMC.PMCMC` implement the Online SMC² algorithm from Chopin et al. (2013) / Vieira (2018) / Temfack & Wyse (2025). Maintains Nθ parameter particles, each with Nx state particles (via BPF). Windowed PMCMC rejuvenation keeps cost O(tk × Nθ × Nx) — constant regardless of time series length.
- Rationale: Standard MCMC (NUTS) re-processes all data from scratch on each update. SMC² evolves the parameter posterior continuously as new observations arrive — no warmup waste, proper sequential uncertainty quantification. The O-SMC² variant (windowed PMCMC) makes this feasible for real-time applications by bounding computation per step. The BEAM's concurrency model maps perfectly: each θ-particle's BPF step and PMCMC rejuvenation are independent → `Task.async_stream` with `max_concurrency: System.schedulers_online()`.
- Assumption: Parameters either remain constant or evolve slowly (geometric Brownian motion for time-varying parameters). The window size tk must be large enough to capture temporal dependencies. Particle degeneracy is managed by ESS-triggered rejuvenation. The model is specified with θ-parameterized init/transition/observation_logp functions.
- Implication: Three-module SMC stack: `ParticleFilter` (D72) → `PMCMC` → `OnlineSMC2`. Public API: `OnlineSMC2.run(model, prior, observations, opts)`. Applications: epidemic surveillance (SEIR), regime detection with discrete states, any state-space model with intractable likelihood. Parallel rejuvenation on 88 cores: 400 θ-particles / 88 cores ≈ 5 waves per observation.

## 75. StochTree-Ex: BART as a separate Elixir library
- Decision: Bayesian Additive Regression Trees (BART) implemented as a separate Mix project (`stochtree_ex/`) with a pure Rust NIF. Not inside eXMC — BART has no dependency on NUTS, IR, or Compiler. Hybrid sampling: GFR (grow-from-root) for exploration every 3rd iteration, proper MH grow/prune for refinement on other iterations.
- Rationale: BART is complementary to NUTS, not a replacement. NUTS handles parametric models with known structure; BART handles nonparametric regression where the functional form is unknown. Different sampling algorithm (MH on tree topology, not HMC on continuous space), different build chain (Rust NIF, no C++/EXLA), different use cases (feature discovery, causal inference, high-dimensional prediction). Clean separation as `{:stochtree_ex, "~> 0.1"}`.
- Assumption: The Rust BART implementation is sufficient for the target use cases. If full StochTree C++ feature parity is needed (BCF, variance forests, ordinal outcomes), Phase 2 adds C++ FFI via the `cc` crate. The current pure-Rust implementation achieves 0.96-1.57x of StochTree Python RMSE on Friedman benchmarks.
- Implication: 14 tests, 0 failures. Validated against StochTree Python 0.4.0 on Friedman #1, #2, and simple linear benchmarks. API: `StochTree.BART.fit/3`, `StochTree.predict/2`, `StochTree.variable_importance/1`. Roadmap: Phase 2 = proper MH MCMC (done), Phase 3 = BCF causal inference, Phase 4 = StochTree C++ FFI.

## 76. smc_ex extraction: O-SMC² as a standalone library
- Decision: Extract the SMC stack (`ParticleFilter`, `PMCMC`, `OnlineSMC2`) from eXMC into a standalone library `smc_ex/` with zero dependencies. Module rename: `Exmc.SMC.*` → `SMC.*`. Public API: `SMC.filter/3`, `SMC.run/4`.
- Rationale: Same logic as StochTree-Ex (D75) — the SMC stack has zero dependency on eXMC internals (no IR, no Compiler, no NUTS, no Nx, no EXLA). A particle filter user should not need to install a probabilistic programming framework. Different algorithm, different use cases, different users. Pure Elixir with `:rand` for PRNG.
- Assumption: The SMC stack remains pure Elixir with no tensor library dependency. If Nx-based vectorization is added (D78), it becomes an optional dependency. The public API (`SMC.run`, `SMC.filter`) is stable.
- Implication: 7 tests, 0 failures. Standalone `mix test` passes without eXMC. Livebook notebook `01_epidemic_tracking.livemd` demonstrates SEIR epidemic tracking. eXMC can depend on `{:smc_ex, path: "../smc_ex"}` with backward-compat aliases.

## 77. O-SMC² Sprint 1+2: data structure + redundant PF elimination
- Decision: Four optimizations applied to smc_ex. Sprint 1: (a) list append → prepend + reverse in all accumulation loops, (b) tuple-based resampling (`List.to_tuple` + `elem` for O(1) access instead of `Enum.at` O(N)), (c) bounded observation window (keep only last `window` observations, not the full history). Sprint 2: (d) eliminate redundant post-rejuvenation PF re-run by having `PMCMC.rejuvenate` return the accepted PF state, (e) adaptive n_moves via `Enum.reduce_while` — halt on first PMCMC acceptance.
- Rationale: Endurance benchmark showed the full-scale test (Nθ=400, Nx=200, T=200) took 92 minutes. Three sources of waste: (1) O(N²) resampling via `Enum.at` on lists, called millions of times; (2) growing `obs_hist` list with O(T) appends; (3) 400 redundant PF runs per rejuvenation to reconstruct particle states that PMCMC already had. Sprint 1 fixes structural inefficiencies. Sprint 2 eliminates ~600 unnecessary PF runs per rejuvenation (round 3 elimination + early PMCMC termination).
- Assumption: Early PMCMC termination (halt on first acceptance) provides sufficient particle diversity. This is a pragmatic optimization, not a theoretical guarantee — the kernel is no longer fixed-scan. For applications where strict theoretical PMCMC properties are needed, use `adaptive_moves: false`. Tuple-based resampling requires particles to fit in memory as a tuple (fine for N_x ≤ 10,000).
- Implication: Measured 1.8x speedup on smoke benchmark (warm). Projected 2-3x on full-scale. All 7 tests still pass. API unchanged — optimizations are internal. Sprint 3 (adaptive ESS threshold, incremental evidence caching) planned for the next round.

## 78. StochTree-Ex ForestTracker: sorted indices for BART split evaluation
- Decision: New Rust module `forest_tracker.rs` implements pre-sorted column indices and per-leaf membership tracking. Split evaluation uses a single O(n) sorted scan per feature (with rayon parallelism across features) instead of the previous O(n_leaf × C × p) brute-force enumeration. New structs: `SortedColumnIndex` (sorted once at initialization), `ForestTracker` (maintained through split/prune/reset operations).
- Rationale: Endurance benchmark showed extreme-p test (n=1K, p=500) took 4.7 hours. The bottleneck was `grow_recursive`'s nested loop: for each leaf, enumerate all p features × all C cutpoints × scan all n_leaf observations = 250M operations per tree at root level. The sorted-index approach discovers all cutpoints in a single pass through pre-sorted indices, accumulating running sufficient statistics. `SuffStats::sub` enables O(1) right_stats = total - left_stats. Rayon parallelizes across features (up to 16 threads, scoped pool).
- Assumption: Memory for sorted indices is acceptable: p × n × 4 bytes (u32 indices). For p=500, n=50K: 100MB. The `leaf_assignment` array (per tree × per observation) adds num_trees × n × 4 bytes. For 200 trees, n=50K: 40MB. Total overhead ~140MB for the largest benchmarked case. If memory-constrained, subsample features per split (random forest style) as a future optimization.
- Implication: Measured **142x speedup** on extreme-p (4.7h → 2 min) and **3.4x** on smoke (p=10). RMSE improved from 0.49 to 0.05 on extreme-p because sorted scan evaluates every possible cutpoint rather than a sampled subset. 14 tests pass, RMSE quality within 1.6x of StochTree Python across all benchmarks. Elixir API unchanged — optimization is entirely within the Rust NIF. rayon added to Cargo.toml.

## 79. Head-to-head benchmark evidence: Elixir vs Python incumbents
- Decision: Ran full endurance suites sequentially (each gets all 88 cores) against Python reference implementations. StochTree Python 0.4.0 (C++ backend, OMP_NUM_THREADS=88) for BART. Nicolas Chopin's `particles` 0.4 library for O-SMC². Results filed in `stochtree_ex/benchmark/RESULTS.md` and `smc_ex/benchmark/RESULTS.md`.
- Rationale: Fair comparison requires: (a) same test configs (data size, particle counts, iterations), (b) same hardware, (c) sequential execution so each gets full machine, (d) Python with proper thread settings (OMP_NUM_THREADS=88). Previous benchmarks ran concurrently, splitting cores unfairly.
- Evidence (BART): Python 7x faster on wall time (193s vs 1,327s total). Elixir wins RMSE on 6/7 tests. Gap narrows with dimensionality: p=100 at 0.94x, p=500 at 0.58x. Per-tree-per-iteration: Elixir 0.88ms vs Python ~0.09ms (10x constant factor gap, not algorithmic).
- Evidence (O-SMC²): Elixir wins 5/7 tests and 1.3x overall (346s vs 458s). Python wins full-scale (Nθ=400): 3.0 min vs 5.1 min. Root cause: `particles` implements waste-free SMC² (Dau & Chopin 2022) which keeps all MCMC intermediate states — zero wasted computation. Our approach discards all but the final accepted state.
- Implication: Sprint 3 priorities identified. BART: close per-iteration gap via dense Vec leaf_stats, per-leaf sorted indices (32x fewer loop iterations), fused O(n) passes. O-SMC²: implement waste-free MCMC rejuvenation (Dau & Chopin 2022), incremental evidence caching (eliminate Round 1), adaptive ESS threshold. Plans filed in `stochtree_ex/SPRINT3_PLAN.md` and `smc_ex/SPRINT3_PLAN.md`.

## 80. Sprint 3 regression: per-leaf sorted indices and adaptive threshold reverted
- Decision: BART per-leaf sorted indices (Sprint 3d) regressed 5/7 endurance tests (extreme-p: 97s → 452s, california: 680s → 765s). Reverted by raising `LEAF_SORTED_MIN_OBS` to 100K (effectively disabled). O-SMC² adaptive threshold regressed full-scale from 307s to 569s. Defaulted to `adaptive_threshold: false`. Both Sprint 3 features remain in code as opt-in for future tuning.
- Rationale: Per-leaf index maintenance cost (O(n_leaf × p) per split/prune, O(n × p × log n_leaf) per refresh) exceeds skip-elimination savings at n < 50K. The overhead of building and maintaining sorted indices through 200 trees × 200 iterations swamps the benefit of avoiding 97% skip rate in the global scan. Adaptive ESS threshold allowed too many lightweight (n_moves=1) rejuvenations that failed to diversify particles, triggering more frequent rejuvenation overall.
- Assumption: The global sorted scan (Sprint 2) remains the right approach for n < 50K. Per-leaf indices may help at n > 100K where the skip rate exceeds 99%. Adaptive threshold needs per-particle acceptance tracking rather than population-level heuristics.
- Implication: Sprint 2 remains the stable baseline for both libraries. BART: 133x on extreme-p, 3.4x on smoke. O-SMC²: 17.5x on full-scale. Sprint 3 incremental evidence caching (3b) is kept on by default in O-SMC² — it eliminates Round 1 without regression. Waste-free MCMC (3a) is implemented and opt-in (`waste_free: true`).

## 81. Milestone: three-library ecosystem merge to main
- Decision: Merge the `fun/serialize-compute` branch to `main` with the complete three-library ecosystem: eXMC (NUTS/HMC), smc_ex (O-SMC²/particle filters), stochtree_ex (BART). Each library is standalone with its own tests, benchmarks, docs, and notebooks. Head-to-head benchmark evidence vs Python incumbents filed.
- Rationale: The three libraries cover the three major inference families: gradient-based (NUTS for continuous parameters), particle-based (SMC² for discrete states), and tree-based (BART for unknown functional forms). Each has demonstrated correctness (tests pass, posteriors converge) and performance (measured vs Python). The ecosystem is presentation-ready for ElixirConf 2026.
- Assumption: Core MCMC, SMC, and BART modules are stable on the current branch. The merge adds the extracted smc_ex/stochtree_ex libraries and supporting docs/notebooks.
- Implication: `main` becomes the reference for the PhD thesis and dataalienist.com blog. Future work (BART applications, O-SMC² extensions, waste-free SMC² tuning) branches from this baseline.

## 82. sim_ex: BEAM-native discrete-event simulation engine
- Decision: New standalone library `sim_ex/` — a discrete-event simulation engine built on OTP patterns, informed by Averill Law's methodology and InterSCSimulator's architecture. Zero dependencies. Two execution modes: Engine (tight loop, single process, no message passing — default) and GenServer (interactive/distributed). Core modules: `Sim.Entity` behaviour, `Sim.Calendar` (:gb_trees priority queue with FIFO tie-breaking), `Sim.Clock` (next-event time advance), `Sim.EntityManager` (registry + dispatch), `Sim.Resource` (M/M/c queues), `Sim.Source` (arrival generators), `Sim.Topology` (ETS shared state — InterSCSimulator pattern), `Sim.Statistics` (Welford streaming + batch means CI), `Sim.Experiment` (replications, CRN, paired comparison), `Sim.PHOLD` (standard DES benchmark).
- Rationale: Every DES concept maps structurally to OTP: entity = process, event calendar = priority queue, simulation clock = GenServer, entity failure = supervisor restart, parallel replications = distributed Erlang. Sim-Diasca (EDF, 2010) proved millions of Erlang actors for DES. InterSCSimulator proved millions of traffic agents. What's missing is the statistical layer — and that's what Les Trois Chambrées provide. Previous sidecar approach wraps external engines; `sim_ex` IS the engine. Combined with eXMC (input modeling), smc_ex (self-calibrating twins), StochTree-Ex (metamodeling), this creates "the simulation that learns" — no existing engine offers posterior-propagated uncertainty + online calibration + automatic metamodeling.
- Assumption: The tight-loop Engine mode is sufficient for single-node simulation. For distributed simulation (multi-node, millions of entities), the GenServer mode provides the foundation but will need Sim-Diasca-style barrier synchronization (tick-diasca model). The current centralized Calendar is the bottleneck at scale — PHOLD shows ~124K events/sec at 10K LPs in Engine mode, degrading with entity count due to Map lookup and :gb_trees O(log n). Future: ETS-backed entity state, sharded calendars.
- Evidence: PHOLD benchmark on 88-core Xeon: Engine mode achieves **539K events/sec** peak (100 LPs) and **124K events/sec** sustained (10K LPs). Engine vs GenServer speedup: **2.4-7.8x** across all configurations. 11 tests, 0 failures. M/M/1 queue converges to theoretical utilization (ρ=0.5). Deterministic replay confirmed (same seed = same trajectory).
- Implication: sim_ex completes the quartet: eXMC (NUTS), smc_ex (SMC²), StochTree-Ex (BART), sim_ex (DES). Optional deps for Les Trois Chambrées integration. PHOLD provides standard benchmark for ongoing optimization. Next: ETS entity storage, sharded calendar, process-per-entity mode for fault tolerance, Livebook notebooks (M/M/1, job shop, self-calibrating twin).

## 83. sim_ex v0.1.2: property tests prove invariants, Rust NIF gets all verbs, preemptive resources
- Decision: Three features in parallel. (1) Property-based tests verify Little's Law (L = λW), flow conservation, determinism, and edge cases across random parameter sweeps — not just seed=42. (2) Rust NIF engine extended from 4 to 12 verbs (decide, batch, split, combine, route, label, assign, decide_multi) via NIF signature upgrade from `(String, f64, f64)` to `(String, Vec<f64>)` and two-pass label resolution. (3) Preemptive resources: `seize :machine, priority: :priority, preemptive: true` with generation-counter hold cancellation, `:gb_trees` priority queue, remaining_hold for service resume.
- Rationale: Property tests move from "it works for one seed" to "the invariants hold." Rust NIF all-verbs means the DSL is engine-agnostic — same model runs on Elixir or Rust. Preemptive resources close the last gap vs Arena (SimPy still lacks preemption). Together: correctness (properties) × performance (Rust) × expressiveness (preemption).
- Assumption: Property test tolerances (15-20% relative for stochastic invariants) are tight enough to catch real bugs without producing false negatives from Monte Carlo noise. Rust NIF `Payload::Continue` for batch/split/combine adds one event per buffered job — acceptable for batch sizes < 100. Generation counter approach for hold cancellation (ignore stale events instead of deleting them) keeps the engine's no-mutation invariant.
- Evidence: **112 tests, 0 failures.** Property tests: Little's Law within 20% across 30 random M/M/1 configurations, flow conservation exact across 50 random M/M/c runs, cross-mode determinism (engine vs ETS) exact for 20 configs. Rust NIF: rework model converges to 14.9% (target 15%) across 1.4M events. Preemptive: rush orders (priority 1) have lower mean_wait than normal (priority 5), preemption count > 0, backward compat preserved (non-preemptive resources unchanged). Bug found and fixed: `Resource.busy_time` never incremented → utilization always 0.0.
- Implication: sim_ex now has 12 DSL verbs across both engines, property-verified correctness, and Arena-equivalent resource preemption. The Rust NIF's `(String, Vec<f64>)` encoding is forward-compatible for future verbs. The property test infrastructure (hand-rolled `PropertyHelper.check/3`) can be upgraded to PropEr when desired.

## 84. Honest benchmarks: SimPy race re-run reveals measurement variance
- Decision: Re-ran all benchmarks under production load (system load 3.9 vs 1.0 original). SimPy batch replications improved from 8.3ms to 6.8ms per rep between runs. Revised all published claims: Elixir speedup 1.8-2.9x (was 2.8-3.7x), Rust batch 14x (was 24x). Updated blog title from "Twenty-Four to One" to "Fourteen to One", all tables in README, RESULTS.md, and three HTML files.
- Rationale: Publishing inflated numbers destroys credibility. The original numbers were real measurements, but publishing best-case without variance disclosure is misleading. The conservative numbers (14x, not 24x) are reproducible under load. The margin is still decisive — 473ms vs 6.8 seconds for 1000 replications.
- Assumption: SimPy's per-rep variance (6.8-8.3ms) is driven by Python 3.12 JIT warmup effects and OS cache state, not by code changes. sim_ex's Rust NIF variance (345-473ms) is driven by system load. On a quiet system, both would be closer to the original measurements.
- Evidence: Barbershop 200K: SimPy 168ms (was 195ms), Elixir 89ms (was 53ms), Rust 16ms (was 12ms). Job Shop 200K: SimPy 3298ms (was 3383ms), Elixir 1879ms (was 1205ms). Batch 1K reps: SimPy 6783ms (was 8310ms), Rust 473ms (was 345ms). M/M/1 theoretical accuracy: 0.0-2.6% error (Erlang-verified, stable across runs).
- Implication: Benchmark claims must include system load context. For future publications: report median of 5 runs on quiet system, or report range with load average. The 14x claim is defensible. The 24x claim requires "on a quiet 88-core system" qualifier.

## 85. eXMC NUTS tests: all 24 pass (stale memory corrected)
- Decision: Investigation confirmed all three root causes of "11/24 NUTS test failures" were already fixed: (1) stale tolerances updated in commit 46bc05f78, (2) Nx 0.10 LU 1x1 bug worked around via `jit_determinant`/`jit_solve` in compiler.ex and log_prob.ex, (3) EXLA tensor fixtures handled by `ensure_binary_backend` pass. The memory note "11/24 NUTS tests fail" was 2+ months stale.
- Rationale: Recorded because stale assumptions in project memory can misdirect effort. The eXMC test fixes were listed as a 2-3 day task blocking book chapters 3-6. In reality, the work was already done — the memory hadn't been updated.
- Assumption: The GPU OOM transient failure in compiler_test.exs:236 (`jit_solve` via EXLA to GPU when CUDA cannot allocate) is not a NUTS bug but a resource contention issue when other GPU workloads are running. Fix: add try/rescue fallback to CPU host client in `jit_determinant`/`jit_solve`.
- Evidence: NUTS: 24/24 pass. Compiler: 15/15 pass (CPU mode). Book chapters 3-6 are unblocked.
- Implication: Memory hygiene matters. Stale "X is broken" notes create phantom work items. The fix: verify before planning. `mix test` is the source of truth, not memory.
