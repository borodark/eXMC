# Decisions

This document records key architectural and design decisions for the Exmc prototype.
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

## 19. Diagonal mass matrix only
- Decision: Mass matrix adaptation uses diagonal (element-wise variance) only.
- Rationale: Sufficient for prototype; dense mass matrix can be added later without API changes.
- Assumption: Target models have weak posterior correlations or few enough dimensions that diagonal is adequate.

## 20. Stan-style three-phase warmup with doubling windows
- Decision: Warmup uses three phases (step size only, step size + mass matrix with doubling windows, step size only).
- Rationale: Proven effective schedule from Stan. Doubling windows allow the mass matrix to stabilize progressively.
- Assumption: Warmup budget is >= ~100 iterations (enough for at least one mass matrix window).

## 21. Distributions use Lanczos lgamma (pure Nx ops, differentiable)
- Decision: `Exmc.Math.lgamma` uses the Lanczos approximation (g=7, 9 coefficients) implemented entirely with Nx ops.
- Rationale: No external C dependencies; the implementation is differentiable through `Nx.Defn.grad`. Accurate to ~15 digits for Re(x) > 0.5.
- Assumption: The gradient of lgamma via Lanczos is numerically stable on BinaryBackend. **Known violation on BinaryBackend only:** gradient triggers `Complex.divide` at extreme values. **Resolved by D24:** when EXLA is available, `Compiler.value_and_grad` uses `EXLA.jit` which handles lgamma gradient correctly.
- Implication: Gamma, Beta, and StudentT work as sampled priors when EXLA is available. On pure BinaryBackend, they remain limited to observation-only use.

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
