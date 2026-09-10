# Two gates that do not gate

**Status:** plans, nothing started. Written 2026-09-10 against `699f3a870`.

Both items are the same species. One is a harness that cannot fail; the other
is a test that fails on exactly one machine and has been read as noise. Neither
is a bug in the sampler, and both stand between here and the obs-axis
parallelism work in `docs/OBS_LOOP_FUSION.md` §6 — the first because that
change is *not* bit-identical and needs a numerical gate under it, the second
because a permanently-red test trains everyone to skim the suite output.

---

## Part 1 — `bench/leapfrog_leaf_diff.exs`

### What it is, and why it is the one that matters

It is the only harness in this repository that dispatches a synthesised chain
shader and compares `q`, `p`, `grad` and `logp` **element-wise along a
trajectory** against the host's own leapfrog. Everything else numerical here is
either differential between two backends — which a defect in the shared NUTS
tree moves identically in both arms — or end-to-end statistical, which sees
only what survives 800 draws.

Its own header records what that cost: it destructured the meta 6-tuple for a
day after captures moved to the extras SSBO, so **the one instrument that could
have caught the zero-likelihood reduce-bound defect was itself un-runnable, on
the very commit that made that defect reachable.** `mix test` never ran it, so
nothing said so.

### It has TWO defects, and the second is the one people miss

**(a) The booleans are computed and discarded.** `run/3` ends by returning
`{ok_q, ok_p, ok_g, ok_lp}` into nothing. There is no assertion anywhere.

**(b) Its threshold is nine orders of magnitude too loose.** `cmp` flags a
divergence at relative Δ > `1.0e-6`. MEASURED on super-io, 2026-09-10, all
three parameter sets it already runs:

| arm | q | p | grad | logp | logp offset spread |
|---|---|---|---|---|---|
| eps=0.05, q0=0.5 | 0.0 | 0.0 | 3.6e-16 | 2.2e-16 | 2.7e-15 |
| eps=1.139, q0=0.5 | 1.9e-15 | 2.2e-15 | 3.0e-15 | 9.2e-16 | 1.8e-14 |
| eps=1.139, q0=3.99 | 5.7e-16 | 2.1e-15 | 3.1e-15 | 1.1e-16 | 8.9e-16 |

The true agreement is **~3e-15**. A gate at 1e-6 would pass a shader wrong in
the ninth significant figure. Asserting the booleans as they stand produces a
test that is green and nearly meaningless — which is worse than the current
state, because it *looks* like coverage.

So the promotion is two changes, and (b) is not optional.

### Proposed

1. **Move to `test/nuts/leapfrog_leaf_diff_test.exs`**, tagged
   `@moduletag :requires_vulkan`. It needs a real device; the whole point is
   the dispatched shader.
2. **Assert, at 1e-13 relative**, for `q`, `p`, `grad` and `logp`
   element-wise. That is ~30x the worst observed value — loose enough not to
   flake on a different GPU's rounding, tight enough that a real defect cannot
   hide. Do **not** pick 1e-15: the measurements above are one machine, and
   §2 of this document is about a test tuned on one machine.
3. **Assert the logp offset is CONSTANT** along the trajectory, at 1e-12
   spread. This is the sharpest assertion in the file and is currently only
   printed. A constant offset is ratio-equivalent and harmless; a varying one
   is exactly the stale-`logp_chain[k]` defect that read as "Ampere
   over-dispersion" for three weeks.
4. **Keep the distinct sigmas (1.0/2.0/3.0) and say why in the test.** The
   header already explains it: with identical sigmas a permuted marker-to-node
   attribution is bit-for-bit the correct answer, and the harness would pass
   while the code was wrong. This property is easy to "simplify" away.
5. **Add the model shapes the original cannot reach.** It covers one
   three-observed-node model at three parameter settings. Add: a vector RV with
   `Nx.dot` over a design matrix (new since `371785ff5`), a Custom likelihood
   reading its observations (new since `d299f4fc4`), and a capture-only Custom.
   Those are the paths with the least element-wise coverage and the most recent
   churn.

### Verification of the gate itself

A gate not shown to fail is not evidence — and this file is the standing proof
of that, having been un-runnable without anyone noticing.

Mutation candidates, each of which must turn it red:

* perturb one leapfrog half-step coefficient in the host reference (proves the
  comparison is live at all);
* swap two observed nodes' sigmas in the fixture but not in the expected values
  (proves the permutation argument in point 4);
* re-emit `logp_chain[k]` from the pre-update block (the historical
  over-dispersion defect — must be caught by the offset-constancy assertion,
  point 3, and by nothing else).

The third is the important one: if the offset assertion does not catch it, the
assertion is misspecified.

### Cost

Three dispatches of K=32 at d=1 plus a host leapfrog; the current script runs
in seconds. Adding four model shapes keeps it well under a minute. It is not a
candidate for `:slow`.

---

## Part 2 — the Cauchy KS failure

### The finding that reframes it

`Exmc.NUTS.Vulkan.ValidatorTest`, "Cauchy(0, 1) — synthesized
leapfrog_chain_synth (median + IQR)":

```
{:error, %{check: :ks, m: 800, n: 800, reference: :none,
           d: 0.09999999999999998, crit: 0.097475,
           alpha: 0.001, approx_p: 6.709252558050261e-4}}
```

It has been carried as "a marginal statistical gate" for days. It is more
specific than that:

**It fails on super-io and passes on mac-247, mac-248 and the Jetson.** Zero
occurrences across all three fleet logs at `d410b183a`, `144d441db` and
`371785ff5`; one on super-io in every run.

The test is seeded (`seed: 42`), the reference is `:none` — the KS is against
the **analytic** Cauchy CDF, so there is no Monte Carlo error on that side —
and the code and `nx_vulkan` pin are identical across hosts. Identical draws
would give an identical `d` and it would fail everywhere.

**So the draws differ by host.** That is the finding, and it sits in tension
with a claim recorded in `NEXT.md`: *"both Keplers and the Ampere produce
bit-identical q/p/grad from this shader"*. Either that claim does not extend to
this model, or something outside the shader diverges.

### First measurement, before any tuning

Do not touch the tolerance until this is answered, because the two outcomes
call for opposite responses.

Dump, on super-io and on one Kepler, for this exact model and seed:

1. a SHA-256 of the Cauchy draws vector;
2. the first dispatched `q_chain`/`p_chain`/`grad_chain`/`logp_chain` buffers
   at a fixed `q0/p0/eps/K` — the `golden.exs` shape already used for the
   18-digest shader goldens;
3. the adapted step size and inverse mass after warmup;
4. `Exmc.JIT.describe/0` and the dispatch count, to confirm both hosts are
   actually on the shader rather than one silently on the per-op path.

Item 4 first. super-io reports `perop_fallback=true` under an explicit
`EXMC_COMPILER=vulkan` while the FreeBSD hosts have no EXLA at all, and a
model refused on one host and accepted on the other would explain everything
without any numerical divergence.

### The branches

* **Buffers identical, draws differ** → the divergence is in the host half:
  RNG, adaptation, or the acceptance test. Nothing to do with the shader.
* **Buffers differ** → the `NEXT.md` bit-identity claim is narrower than
  stated, and *that* is the finding; the KS failure is a symptom and the
  tolerance is the wrong thing to touch.
* **Different code paths (item 4)** → not a numerics question at all, and the
  fix is the routing.

### Only then, the statistic

`d = 0.09999999999999998` against `crit = 0.097475` is **2.6% over** the
critical value, with `approx_p = 6.7e-4` against `alpha = 1e-3`. That is a hair,
and Cauchy is the worst case for a KS test — undefined mean and variance, so
the harness already substitutes median + IQR for the moment check, and the tail
mass that drives `d` is exactly where a sampler struggles.

Three responses, in preference order:

1. **A real numerical difference is found and fixed.** Then nothing about the
   test changes, which is the outcome worth spending on.
2. **The draws are correct and the gate is simply too tight for Cauchy at
   n=800.** Then raise `n_samples` for this case rather than loosening `alpha`
   — more draws sharpen the test, a wider alpha blinds it — and record the
   power calculation, the way `assert_posterior!/3` already does elsewhere.
3. **Tag `:vulkan_known_failure` with the mechanism written down**, as
   `LevelSetIntegrationTest` was in `c233e97bc`. Only if 1 and 2 are exhausted;
   a tag is a decision to stop looking and should read like one.

**Not acceptable: leaving it red.** One permanently-failing test on the
development machine is how a second one gets ignored. It has already cost this
project a re-run — a fleet suite was re-run to recover failure names because
`707 tests, 1 failure` had stopped carrying information.
