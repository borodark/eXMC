# The `assert_in_delta` sweep — what the suite's tolerances actually admitted

**Done 2026-08-18.** NEXT.md §2 item 2: *"find every tolerance that would
accept a 20% variance error"*. Reproduce with `bench/tolerance_audit.exs`; raw
output in [`bench_results/TOLERANCE_AUDIT.md`](../bench_results/TOLERANCE_AUDIT.md).

## The answer, up front

Not one statistical assertion in the suite could see a 20% variance error.
Including the one 0.3.1 added as the fix for exactly this problem.

The sweep found three separate things, and only the first is the one the item
was written to find.

### 1. 89 of 99 sampling tests asserted nothing about dispersion at all

*(Counted by scanning every `test`/`property` block for a call to a sampler and
then for any assertion mentioning `std`, `sd`, `var`, `variance`,
`posterior_sd` or a quantile. It is a heuristic over assertion text, so treat
89 as "the overwhelming majority" rather than an exact census.)*

A test that never checks the spread does not "accept a 20% variance error" — it
accepts **any** variance error, a doubled posterior and a frozen chain alike.
That is not a loose tolerance, it is an absent one, and it is the majority of
the suite.

The clearest case was `nuts_test.exs:308`, the suite's own standard-normal
test:

```elixir
assert abs(var - 1.0) < 1.0, "Var[mu] = #{var}, expected near 1"
```

Any variance in `[0, 2]`. A frozen chain has variance 0 and passed.

### 2. Where a gate existed, the sample size made it decorative

This is the finding that mattered, because it means rewriting the tolerances
would have changed nothing. A 4-sigma analytic gate resolves a variance error
of `4 * sqrt((mu4 - sigma^4)/n_eff) / sigma^2`. NUTS on these targets yields
**ESS ~= 0.35 draws per draw** (Geyer and `Diagnostics.ess` agree to 1%), so at
the draw counts the suite used:

| test | draws | ESS | smallest variance error a 4-sigma gate could see |
|---|---:|---:|---:|
| `integration_test.exs:15` conjugate N-N | 500 | 178 | **37.8%** |
| `integration_test.exs:88` Gamma(2,1) | 200 | 83 | **86.2%** |
| `integration_test.exs:113` Exponential(2) | 300 | 69 | **53.3%** |
| `integration_test.exs:270` Beta(2,5) | 400 | 159 | **40.6%** |
| `new_dist_test.exs:230` Lognormal(0,0.5) | 300 | 98 | **63.7%** |
| `nuts_test.exs:308` Normal(0,1) | 500 | 172 | **36.4%** |

**The defect that shipped in 0.3.0 was a 37.8% variance inflation.** The
conjugate test 0.3.1 rewrote to use `check_analytic/3` resolves 37.8%. It was a
coin flip on the exact bug it was written for, and it read as a tightened
assertion.

`check_analytic/3` cannot be too tight — it sizes its band from the chain's own
ESS. That is the property that makes it safe, and it is also why it silently
degrades to nothing when the chain is short. Nothing in it says "this chain was
too small for the answer to mean anything."

### 3. Two gates were wrong in the other direction

* **Student-t at `df <= 4` has no valid variance gate at all.** The mean and
  variance exist for `df > 2`, but every gate here is a multiple of the standard
  error of the *sample* variance, `sqrt((mu4 - sigma^4)/n_eff)` — and the fourth
  moment is infinite for `2 < df <= 4`. `analytic_moments/1` returned finite
  moments for `df > 2` and the resulting band was noise with a number on it.
  It now returns `:unknown` there, and `assert_posterior!` refuses rather than
  computing something meaningless.

* **The Cauchy IQR gate was about six times too tight**, so it would have failed
  a correct sampler. `check_analytic/3` used `0.25 * IQR / sqrt(n)` as the
  standard error of a sample IQR. The IQR's SE is not a fixed fraction of the
  IQR: for quantiles, `Var(q_p) = p(1-p)/(n f(q_p)^2)`, so for a symmetric
  density `SE(IQR) = 0.5 / (f sqrt(n))` with `f` the density at the quartiles.
  For Cauchy that is `pi * scale / sqrt(n)` against the old `0.5 * scale /
  sqrt(n)`. Measured on a t(4) chain the old proxy was 5x too tight; the
  corrected form put the measured IQR at 1.9 sigma.

## The fix: a gate that fails when it cannot see

`Exmc.TestHelper.assert_posterior!/3` asserts the closed-form moments **and**
that the chain had enough effective draws for that assertion to mean something.
If it did not, the test fails as `INCONCLUSIVE` and says how many draws it
would take:

```
assert_posterior!: INCONCLUSIVE, not passing.

This chain resolves a variance error of 37.8% at best.
The test asks to see 20.0%.

  draws              500
  effective (Geyer)  178

Raise num_samples to about 1788, or pool seeds, until the resolution is met.
Lowering :resolution instead is allowed only with a comment saying what
defect size the test is knowingly blind to.
```

**Order matters, and getting it wrong was caught by probing the helper.** The
resolution figure is computed from the sample's own fourth moment, so a chain
with too much spread inflates its own estimate and can push itself over the
threshold — reporting a genuine variance defect as "not enough draws". The
moments are therefore checked *first*; the power gate only governs whether a
*pass* is allowed to stand. `test/tolerance_gate_test.exs` pins all three
behaviours, including that one.

## What changed, per test

| test | was | now |
|---|---|---|
| `integration_test.exs:15` conjugate N-N | `check_analytic` at 500 draws | `assert_posterior!` at 4000, resolves 14.5% |
| `integration_test.exs:88` Gamma(2,1) | `assert_in_delta mean, 2.0, 1.0` | `assert_posterior!` at 8500 |
| `integration_test.exs:113` Exponential(2) | `assert_in_delta mean, 0.5, 0.3` | `assert_posterior!` at 12000 |
| `integration_test.exs:270` Beta(2,5) | `assert_in_delta mean, 2/7, 0.15` | `assert_posterior!` at 5000 |
| `integration_test.exs:307` StudentT(4,3,1) | `assert_in_delta mean, 3.0, 1.5` | ESS-sized mean gate + analytic IQR gate; no variance gate, and the comment says why |
| `new_dist_test.exs:230` Lognormal(0,0.5) | `assert_in_delta mean, 1.133, 0.5` | `assert_posterior!` at 10000 |
| `stan_test.exs:8` conjugate N-N | `assert_in_delta mean, 4.95, 0.5` | `assert_posterior!` at 4000, resolves 14.5% |
| `nuts_test.exs:308` Normal(0,1) | `abs(var - 1.0) < 1.0` | `assert_posterior!` at 4000, resolves 14.1% |

Two supporting changes in `Exmc.NUTS.Vulkan.Validator`: `analytic_moments/1`
gained `{:gamma, alpha, beta}` and `{:beta, a, b}` (the suite asserts on both
and neither had closed-form moments available), and the Cauchy branch now
carries `f_quartile` so the IQR gate can be computed correctly.

## Two things the draw counts had to learn

**Headroom is not padding.** The first version of this sweep sized every count
to land just under 20% on the default (EXLA) arm — 17-19% across the board.
The Lognormal test then failed under `EXMC_COMPILER=vulkan` at **21.3%**: same
model, same seed, same draw count, different effective sample size. A gate
sized to its own arm's ESS is a coin flip on any other. The counts now target
roughly 15%, on the slower arm.

**Skew is what it costs.**

The draw counts are not uniform and should not be. The gate's width scales with
`(mu4 - sigma^4)/sigma^4`, which is 2 for a Normal and **8 for an Exponential**
— so the Exponential test needs four times the effective draws of the Normal
one for the same resolution. That is why it went to 9000 draws and the Beta to
3000. Verifying a skewed posterior is genuinely more expensive; the alternative
on offer was not verifying it.

Cost, measured by stashing the four touched test files and running them both
ways on the default (EXLA) arm: **54.4s to 149.4s**, same 80 tests. Add the new
15s `test/tolerance_gate_test.exs`, and the full suite lands at 585s. Under
`EXMC_COMPILER=vulkan` these files are slower again — the five touched files
take 259s there.

That is roughly 110 seconds, and it buys assertions that can fail. The previous
54 seconds bought assertions that could not.

## What was deliberately left loose, and why

Not every loose tolerance is a defect. These were reviewed and kept:

* **`advi_test.exs`, `pathfinder_test.exs`, `smc_test.exs`** — ADVI, Pathfinder
  and SMC are *approximations*. A mean-field Gaussian approximation to a
  posterior is not supposed to match its moments, so an analytic gate on them
  would be asserting something false. Their wide deltas are smoke tests and
  should say so; what they need is a comparison against their own reference
  implementations, not a tighter number.
* **`test/sbi/*`** — already the best-calibrated statistical assertions in the
  repo. They derive standard errors and assert at `4.0 * se`, and
  `conjugate_test.exs` carries an explicit `abc_bias` term for the tolerance's
  own approximation error. Nothing to fix.
* **`poker_test.exs:228`, `hierarchical_test.exs:131`, `integration_test.exs`
  hierarchical cases** — hierarchical posteriors with no closed form. A
  moment gate needs a truth to compare against. These want either a conjugate
  reduction or SBC (`VERIFICATION_METHODS.md` rank 5), not a smaller delta.
* **`diagnostics_test.exs:8`** — asserts against a *fixed* known trace, not a
  sampled one. `assert_in_delta x.std, 28.87, 0.1` is a numeric identity check
  and is correctly tight.
* **The ~148 `assert_in_delta`s in non-sampling tests** — logpdf values,
  gradients, transforms, quantile arithmetic. These are numeric identities at
  1e-6 to 1e-12 and are not in scope.

## What this does not do

The gate proves a chain *could* have seen a 20% variance error in the marginal
it checks. It says nothing about correlations, about the joint distribution, or
about any model without a closed form — which is most real models. That is
`VERIFICATION_METHODS.md`'s rank 2 (Geweke) and rank 5 (SBC), and this work
does not substitute for either.
