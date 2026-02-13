# Decisions

This document records key architectural and design decisions for the eXMC prototype.
Each entry includes the assumption that must hold for the decision to remain valid.

## 1. Nx as the numeric backend
- Decision: Use Nx for tensor operations, broadcasting, and autodiff primitives.
- Rationale: Nx provides a pure Elixir API with backends for CPU/GPU, and integrates with Defn/EXLA later.
- Assumption: Nx BinaryBackend is sufficient for prototype-scale models (< ~10 RVs, < ~1000 samples).
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
- Assumption: BinaryBackend float precision is sufficient for test comparisons with tolerances ~1e-5.

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
- Assumption: BinaryBackend is the only backend in use. If EXLA is added, revisit for JIT benefit.

## 15. Sampler operates in unconstrained space, returns constrained trace
- Decision: The sampler operates entirely in unconstrained (flat f64) space and applies forward transforms when building the trace.
- Rationale: Consistent with D10. HMC/NUTS require unconstrained geometry; users expect constrained outputs.
- Assumption: Forward transforms are cheap to apply post-hoc.

## 16. PRNG via Erlang `:rand` with deterministic seeding
- Decision: Sampler uses `:rand.seed_s(:exsss, seed)` for all random decisions (direction, proposal, momentum).
- Rationale: `Nx.Random.split/uniform` are prohibitively slow with BinaryBackend due to defn tracing overhead.
- Assumption: BinaryBackend is the only backend. With EXLA, Nx.Random would be fast and preferable.

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

## 52. Distributed 4-node sampling with compile options fix
- Decision: (1) Thread `[:ncp, :device]` compile options through `Distributed.sample_chains` to `Compiler.compile_for_sampling` in `run_coordinator_warmup`, `run_chain_local`, and `run_chain_remote`. (2) Created `benchmark/dist_bench.exs` for 4-node `:peer` distributed benchmarking.
- Rationale: `distributed.ex` called `Compiler.compile_for_sampling(ir)` without passing `ncp: false` or `device` options, so distributed sampling always used NCP (wrong for medium/stress models where `ncp: false` is 9x better). Additionally, local 4-chain via `sample_chains_compiled` runs chains sequentially on a single BEAM (Elixir NIF scheduling prevents true CPU parallelism), while PyMC uses 4 OS processes. Using 4 `:peer` nodes gives same parallelism model.
- Assumption: `:peer` nodes on the same machine provide true OS-process parallelism equivalent to PyMC's multiprocessing.
- Implication: 4-chain distributed results (seed=42): simple 1665 ESS/s (3.4x local), medium 812 ESS/s (3.5x local), stress 628 ESS/s (3.7x local). Near-linear scaling from 5 nodes (coordinator + 4 peers, each running 1 chain; coordinator chain runs concurrently with peers). Distributed overhead is minimal — dominated by per-node JIT compilation (cold start).
