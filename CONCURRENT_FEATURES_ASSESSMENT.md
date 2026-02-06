# On the Mixing Properties of Parallel Feature Chains

*An assessment of whether three proposed features can be developed concurrently,
presented in the spirit of the samplers that inspired them.*

---

## The Proposal

Three features stand before the acceptance criterion:

1. **Vectorized Observations** — batch logpdf, shape-aware PointMap
2. **Non-Centered Parameterization** — a rewrite pass to escape the funnel
3. **WAIC/LOO Model Comparison** — pointwise log-likelihood diagnostics

The question: can we run these as parallel chains, or must they be sequential?

## The State Space

Every Markov chain needs a well-defined state space. Ours is the codebase, and the
"state" of each feature chain is the set of files it modifies. When two chains
propose transitions to the same file, we get a merge conflict — the developer's
divergence.

Here is the transition kernel — which files each feature touches:

| File | Vectorized Obs | NCP | WAIC/LOO |
|------|:-:|:-:|:-:|
| `compiler.ex` | **HEAVY** | light | medium |
| `point_map.ex` | **HEAVY** | - | - |
| `builder.ex` | medium | - | - |
| `dist/*.ex` | medium | - | - |
| `rewrite.ex` | - | medium | - |
| `diagnostics.ex` | - | - | medium |
| `integration_test.exs` | adds tests | adds tests | adds tests |

The diagonal is clear: NCP lives in the rewrite layer, WAIC lives in diagnostics,
and Vectorized Obs lives in the compiler and data structures. They occupy nearly
orthogonal subspaces.

Nearly.

## Detailed Balance: Where the Chains Collide

**`compiler.ex` is the funnel.** All three features need to touch it:

- *Vectorized Obs* rewrites `node_term`, `eager_obs_term`, `deferred_obs_term`,
  `eval_terms`, and `apply_obs_meta` to handle tensor-shaped observations. This is
  a seismic change — the compiler's core loop gets reshaped.

- *NCP* needs the compiler to evaluate deterministic nodes (`det` nodes) that
  participate in the logp graph. Currently `det` nodes contribute nothing (D7).
  NCP would need them to define `x = mu + sigma * z` and thread that through
  param resolution.

- *WAIC/LOO* needs a new compile mode — `compile_for_pointwise_logp` — that returns
  per-observation log-likelihood instead of a scalar sum. This means `build_terms`
  must tag which terms belong to which obs node.

Three chains trying to sample from the same narrow region of `compiler.ex`. Classic
funnel geometry. We know what happens in funnels: divergences.

**`integration_test.exs` is shared** but benign — each feature appends independent
test blocks. No conflict, just a file that grows. Like a well-mixing chain
exploring new territory.

## The Acceptance Probabilities

### NCP + WAIC (parallel): **ACCEPT** (p ≈ 0.95)

These two chains explore completely different regions:

- NCP lives in `lib/exmc/rewrite/` (new pass) and touches `rewrite.ex` (add to
  pass list). Its compiler interaction is limited to ensuring `det` nodes evaluate
  correctly — a small, isolated change.

- WAIC lives in a new `lib/exmc/model_comparison.ex` module and adds functions to
  `diagnostics.ex`. Its compiler interaction is a new `compile_for_pointwise_logp`
  function — additive, not modifying existing code.

Their compiler.ex changes don't overlap: NCP modifies `node_term` for `det` nodes,
WAIC adds a new public function. Different regions of the file, different
concerns. **These chains mix well.**

### Vectorized Obs + NCP (parallel): **MARGINAL** (p ≈ 0.5)

Vectorized Obs restructures how shapes propagate through the compiler. NCP
introduces new node patterns (replacing an RV with a det + a standard-Normal RV).
If the compiler's shape logic changes while NCP is being written against the old
API, the NCP pass may generate nodes that the new compiler doesn't handle correctly.

However: NCP's output is just standard IR nodes (`rv` + `det`). If Vectorized Obs
preserves the existing node API and only extends it with shape handling, NCP
wouldn't break. The risk is **moderate** — it depends on whether Vectorized Obs
changes the `node_term` dispatch or just the internals.

### Vectorized Obs + WAIC (parallel): **REJECT** (p ≈ 0.2)

WAIC needs to compute per-observation log-likelihood. Its design depends intimately
on how observations are represented in the compiler:

- With scalar obs (current): each obs node → one logp term → easy to attribute
- With vector obs (proposed): one obs node → vector logp → needs `reduce: :none`
  mode to get pointwise values

If WAIC is built against the scalar-obs API, then Vectorized Obs lands and changes
how obs terms work, WAIC breaks. The pointwise logp extraction logic would need
to be rewritten. **These chains are anti-correlated** — progress in one undoes
assumptions in the other.

## The Recommended Schedule

Like a proper Stan warmup with doubling windows:

```
Phase I:  NCP + WAIC in parallel     (independent chains)
Phase II: Vectorized Obs             (after Phase I stabilizes)
Phase III: Reconciliation            (update WAIC for vector obs if needed)
```

**Why this order?**

1. **NCP is pure upstream.** It's a rewrite pass — it transforms the IR before
   the compiler ever sees it. Zero risk of breaking existing tests. The compiler
   change (evaluating det nodes in logp) is small and additive. And NCP directly
   improves sampling quality for existing hierarchical models — immediate payoff
   from the D28 fix.

2. **WAIC is pure downstream.** It consumes traces that already exist. The new
   `compile_for_pointwise_logp` is additive. Building it against the current
   scalar-obs API is safe because Vectorized Obs won't remove scalar obs support
   — it will extend it.

3. **Vectorized Obs is foundational.** It changes PointMap, Compiler, Builder,
   and distributions. Doing it last means NCP and WAIC are stable, and any
   shape-related issues only need to be reconciled once.

This schedule has the best **effective sample size** — maximum useful work per
unit of developer time, minimum divergences from merge conflicts.

## Decision Record

**D31. Feature parallelism: NCP and WAIC first, Vectorized Obs after.**
- Decision: Develop Non-Centered Parameterization and WAIC/LOO in parallel
  (Phase I), then Vectorized Observations (Phase II).
- Rationale: NCP and WAIC touch orthogonal layers (rewrite vs diagnostics).
  Vectorized Obs restructures core data flow in compiler.ex and PointMap,
  which both NCP and WAIC depend on. Building downstream features against
  a stable API avoids rework.
- Assumption: Vectorized Obs will extend, not replace, the scalar obs API.
  If this assumption breaks, WAIC's pointwise logp extraction needs updating
  in Phase III.
- Implication: NCP benefits immediately from the D28 fix (correct constrained
  param resolution in hierarchical models). WAIC benefits from the existing
  15 integration tests as validation targets.

---

*Like any good sampler, we aim for detailed balance: the probability of proposing*
*"do everything at once" times the probability of accepting that proposal should*
*equal the probability of proposing "do it in phases" times its acceptance. The*
*math works out — sequential where it must be, parallel where it can be. The chain*
*converges either way, but one path has fewer divergences.*
