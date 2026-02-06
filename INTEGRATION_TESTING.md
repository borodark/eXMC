# Integration Testing Tracker

Tracks what end-to-end integration tests exist, what's possible but not yet written,
and what's blocked by known limitations.

## Current Tests (test/integration_test.exs)

| # | Test | Pipeline | Status |
|---|------|----------|--------|
| 1 | Conjugate Normal-Normal posterior recovery | Builder -> Sampler -> Diagnostics.summary | PASS |
| 2 | Multi-chain R-hat + ESS | Builder -> sample_chains -> Diagnostics.rhat/ess | PASS |
| 3 | Gamma prior: samples positive, mean correct | Builder(Gamma+log) -> Sampler -> constraint check | PASS |
| 4 | Exponential prior: constrained support | Builder(Exp+log) -> Sampler -> constraint check | PASS |
| 5 | Hierarchical Normal-Normal with diagnostics | Builder(string refs) -> Sampler -> summary + ESS | PASS |
| 6 | DSL -> sample -> diagnostics round-trip | DSL.model -> Sampler -> summary + ESS + ACF | PASS |
| 7 | Sample stats internal consistency | Sampler -> sample_stats field validation | PASS |
| 8 | More observations narrow posterior | Sampler(1 obs) vs Sampler(3 obs) -> std comparison | PASS |
| 9 | Beta prior: samples in (0,1), mean near a/(a+b) | Builder(Beta+logit) -> Sampler -> constraint + mean check | PASS |
| 10 | StudentT prior: mean near loc | Builder(StudentT) -> Sampler -> mean check | PASS |
| 11 | Hierarchical with constrained parent (sigma~Exp, child~N) | Builder(string refs + log transform) -> Sampler -> positivity + mean check | PASS |
| 12 | Prior predictive sampling | Predictive.prior_samples -> shape + support + hierarchical check | PASS |
| 13 | Prior predictive with constrained distribution | Predictive.prior_samples(Exp) -> positivity + mean check | PASS |
| 14 | Posterior predictive sampling | Sampler -> Predictive.posterior_predictive -> mean check | PASS |
| 15 | Large model: 5-param hierarchical with obs | Builder(5 free RVs + 3 obs) -> Sampler -> Diagnostics | PASS |
| 16 | NCP: hierarchical auto-reparameterized | Rewrite(NCP) -> Compiler -> Sampler -> reconstruction check | PASS |
| 17 | NCP: logp equivalence at a point | Compiler(NCP) -> logp finite and reasonable | PASS |
| 18 | WAIC: pointwise log-likelihood and WAIC | Sampler -> ModelComparison.pointwise_log_likelihood -> WAIC | PASS |
| 19 | WAIC: better model has lower WAIC | Two models -> WAIC comparison -> ranking | PASS |
| 20 | LOO: basic LOO-CV computation | Sampler -> ModelComparison.loo -> finite results | PASS |
| 21 | Vector obs same posterior as scalar obs | Builder(vector obs) -> Sampler -> mean/std match scalar version | PASS |
| 22 | Vector obs with hierarchical model | Builder(vector obs + string refs) -> Sampler -> posterior check | PASS |
| 23 | WAIC with vector obs per-element keys | Builder(vector obs) -> Sampler -> ModelComparison -> tuple keys | PASS |
| 24 | Parallel chains faster than sequential | sample_chains(4, parallel: true) vs sequential -> timing + determinism | PASS |
| 25 | Parallel chains with init_values | sample_chains(2, init_values: ...) -> positivity + R-hat | PASS |

## Possible But Not Yet Written

| Scenario | What it tests | Difficulty | Notes |
|----------|--------------|------------|-------|
| Obs metadata (weight, mask, reduce) e2e | Builder metadata -> Compiler -> shifted posterior | Medium | Verify weighted obs shifts posterior differently |
| Affine/matmul measurable obs e2e | Rewrite pipeline -> Compiler -> Sampler | Medium | Existing unit tests cover rewrite; need sampler e2e |
| DSL with det nodes | DSL.det + DSL.matmul -> full pipeline | Easy | |
| Autocorrelation on real trace | Sampler -> Diagnostics.autocorrelation -> geometric decay check | Easy | |

## Blocked / Known Limitations

| Scenario | Blocker | Root Cause |
|----------|---------|------------|
| Prior predictive for observed variables | Predictive.prior_samples only samples free RVs | Could be extended |
| NCP opt-out per variable | Auto-NCP always applies to hierarchical Normals | May need opt-out for data-rich models |
| PSIS-LOO (Pareto smoothing) | LOO uses basic importance sampling | Pareto tail fitting not yet implemented |

### Previously Blocked, Now Resolved

| Scenario | Resolution |
|----------|-----------|
| Beta distribution in sampler | EXLA JIT for `value_and_grad` handles lgamma gradient correctly (test 9) |
| StudentT as sampled prior | Same EXLA JIT fix (test 10) |
| Hierarchical with constrained parent (sigma~Exp, child~N) | Numerically stable softplus in transforms + NaN guards in sampler + EXLA JIT (test 11) |
| Posterior predictive checks | Implemented `Exmc.Predictive` module (test 14) |
| Prior predictive sampling | Implemented `Exmc.Predictive.prior_samples` (tests 12-13) |
| Large model (5+ RVs) | 5-param hierarchical model now works with proper init values (test 15) |
| Model comparison (WAIC/LOO) | Implemented `Exmc.ModelComparison` module (tests 18-20) |
| String param refs to transformed RVs | Fixed `resolve_params_constrained` to apply forward transform (D28) |

## Patterns Learned

1. **Init values help constrained models.** Without good starting points, the step-size search can explore extreme unconstrained regions causing overflow. Provide `init_values` near the prior mode.

2. **lgamma gradient is fragile on BinaryBackend.** The Lanczos approximation's `c / (x + i)` terms produce `1/(x+i)^2` gradients that can trigger complex arithmetic in Nx's BinaryBackend. **Resolved:** `EXLA.jit` wraps `value_and_grad` when EXLA is available, handling lgamma gradient correctly. IR tensors are copied to BinaryBackend before closure capture to avoid backend conflicts during EXLA tracing.

3. **Tolerances must be generous.** With 200-500 samples on BinaryBackend, expect posterior mean within ~0.5 and std within ~0.5 of analytic values.

4. **`stats.divergences` includes warmup.** The sampler accumulates divergence count across warmup and sampling phases. `sample_stats` only covers the sampling phase. So `sampling_divergences <= stats.divergences`.

5. **`Nx.default_backend(EXLA.Backend)` does NOT affect `Nx.Defn.value_and_grad`.** Setting EXLA as default backend only affects tensor creation. Anonymous closures passed to `value_and_grad` always go through the Evaluator (BinaryBackend) unless explicitly wrapped with `EXLA.jit`. The Compiler now auto-detects EXLA and wraps `value_and_grad` in `EXLA.jit`.

6. **Numerically stable softplus prevents transform overflow.** `softplus(x) = max(x, 0) + log1p(exp(-|x|))` never overflows. Used for logit forward/jacobian and softplus forward/jacobian. Avoids `Nx.sigmoid` overflow on BinaryBackend.

7. **NaN guards needed in sampler arithmetic.** `Nx.to_number` returns atoms (`:nan`, `:neg_infinity`) for special values. Erlang arithmetic on these atoms crashes. Tree building and step-size search now guard against non-numeric `joint_logp` values.

8. **String param refs must resolve to constrained values.** When a distribution param references another RV via string (e.g., `sigma: "sigma_global"`), the resolved value must be in constrained space. The unconstrained value map stores `log(sigma)`, not `sigma`. Without applying the forward transform, logpdf gets wrong params (e.g., sigma=0 instead of sigma=1), causing NaN gradients and sampler failure.

9. **Init values must cover ALL free RVs.** `PointMap.to_unconstrained` requires entries for every free RV. Partial init maps cause `KeyError`. Either provide all or none.
