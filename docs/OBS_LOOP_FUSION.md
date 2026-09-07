# Fusing the observation-axis loops

**Status:** IMPLEMENTED in `f2ae139d7`. Plan written 2026-09-07 against
`95b3a06b1`; §4 below replaced with the measured result, which refuted the
prediction it originally contained.
**Measured on:** mac-248 (GT 750M, headless, idle), nx_vulkan `7af37b3`.

The fused f64 chain shader re-walks the observation axis once per `Nx.sum`
node in the autodiffed log-density, instead of once per dispatch. A
three-parameter Student-t likelihood walks it **165 times**. This document
plans the fix, and argues for doing it *before* the obs-axis parallelism that
the module's own docstring has had queued as "R2.2.1 (next)" since the
beginning.

---

## 1. What was measured

`bench/chain_trace_split.exs` reported the observed model spending 92.8% of a
sampling run inside `Dispatch.chain/8` at 33.8 ms per dispatch, against 188 us
for a scalar-observation model. Sweeping `n_obs` at K=8, d=2:

| n_obs | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| us/dispatch | 348 | 484 | 750 | 1288 | 2312 | 4407 | 8588 | 17309 | 33685 |
| SPIR-V bytes | 41992 | 41992 | 41992 | 41992 | 41992 | 41992 | 41992 | 41992 | 41992 |

`cost ~= 215 + 131*n_obs` us, to within 2% over eight doublings. Byte-identical
SPIR-V confirms this is not the pipeline-ceiling defect returning; `21700c04a`
holds and the data lives in the extras SSBO.

Sweeping the model instead, n_obs fixed, slope taken over n_obs in {16,64,256}:

| model | obs loops emitted | ops inside them | slope us/obs | us per op-obs |
|---|---|---|---|---|
| `y ~ N(mu, 1)`, d=1 | **3** | 16 | 2.4 | 0.15 |
| `y ~ N(mu, sigma)`, d=2 | **15** | 117 | 130.7 | 1.12 |
| `y ~ T(df, mu, sigma)`, d=3 | **165** | 671 | 716.5 | 1.07 |

The per-op-per-observation cost is flat across the two larger models. **The
number of emitted loops is the driver, and it is not bounded by anything in
the model's size.**

### It is not what it first looks like

Two hypotheses were tested and both are refuted, which is why they are recorded
here rather than left as intuitions for someone to re-derive:

* **"d threads walk the axis in parallel."** `build_grad_body_with_loops/4`
  emits `if (tid == i) { <loops> }` per RV and its comment says these "run in
  parallel". On NVIDIA hardware the `tid == i` branches for i < 32 live in one
  warp and serialise. But a d-sweep does not show the linear-in-d scaling that
  predicts either — 2.4 -> 130.7 -> 716.5 us/obs is 54x then 5.5x. Those three
  models differ in likelihood complexity as well as in d, so the sweep
  **isolates nothing** and warp divergence remains unquantified. Do not repeat
  it in that form.
* **"the reduce is one serial loop and needs 256-way parallelism."** True as
  far as it goes, and it is what the original writeup in `NEXT.md` claims. But
  it is the second-order term: with ~4 ops per loop body, the loop *count* and
  its per-iteration global load dominate the arithmetic by an order of
  magnitude.

---

## 2. The defect

`Exmc.NUTS.CustomSynth.Glsl` emits `/*REDUCE_SUM*/(<inner>)` for each `sum`
node it walks (`glsl.ex:572`). `do_transform_rs/7`
(`multi_rv_custom_spec.ex:1160`) rewrites each marker into its own complete
`for` loop with its own accumulator:

```glsl
double _gacc0_0 = 0.0lf;
for (uint j = 0u; j < pc.n_obs; j++) {
    double obs_j = obs_inv_mass[j];
    _gacc0_0 += ((-((-1.0 * (obs_j - q_shared[0])))));
}
```

Reverse-mode autodiff of a log-density with several parameters produces many
`sum` nodes — one per parameter per additive term, multiplied again because the
same body is emitted three times (`{{prior_grad_body_q}}`,
`{{prior_grad_body_qn}}`, `{{prior_logp_body_qn}}`). Nothing merges them. So a
model whose gradient decomposes into L sums walks the observation axis L times
per leapfrog step, and **`obs_inv_mass[j]` is loaded from global memory L times
per observation** — 165 loads of one `double` for the Student-t, unhidden,
by a single active invocation, on a card with ~400-cycle global latency.

Verified for the d=2 model: all 15 loops carry **identical bounds**
(`uint j = 0u; j < pc.n_obs`) and all 15 read `obs_j`.

The CSE pass makes this worse than it looks. `cse_loop_body/1`
(`multi_rv_custom_spec.ex:1225`) hoists repeated subexpressions *within one
loop body*, with `@cse_min_len 18`. Across 165 bodies of ~4 ops each there is
nothing long enough to hoist, so the pass that exists to remove exactly this
redundancy never fires — the redundancy is between the loops, where it cannot
see it.

---

## 3. The fix: fuse loops that share bounds

One loop, L accumulators, one `obs_j` load per observation.

```glsl
double _gacc0_0 = 0.0lf;
double _gacc0_1 = 0.0lf;
/* ... */
for (uint j = 0u; j < pc.n_obs; j++) {
    double obs_j = obs_inv_mass[j];
    <shared CSE bindings, now spanning all L bodies>
    _gacc0_0 += (...);
    _gacc0_1 += (...);
}
```

### Where

`do_transform_rs/7` already returns `{loops, glsl}` with `loops` a list of
rendered blocks, and both `build_grad_body_with_loops/4` and
`build_logp_body_with_loops/3` do nothing but `Enum.join(loops, "\n")`. So the
change is local: keep collecting per-marker `{bounds, accum, cse_inner}`
tuples instead of pre-rendered strings, then render one loop per distinct
bounds pair at join time.

Do the fusion **before** CSE, not after. Running `cse_loop_body/1` once over
the fused body is most of the win beyond the load: it finally sees the
repetition across bodies that `@cse_min_len` was calibrated for. This reorders
the pipeline, so it is a real change to `do_transform_rs/7`, not a wrapper
around it.

Both templates need it — the single-instance one at `:56` and the batched one
at `:1532`, which shares `transform_reduce_sum/5` with `obs_index` of
`"extras_off + j"`. The bounds stay in per-instance coordinates, so grouping by
bounds composes with batching unchanged.

### Preconditions that make it legal — check them, do not assume them

1. **Identical bounds.** Group by the `{lo, hi}` pair from `reduce_bounds/4`.
   Loops with different bounds (per-node spans, or a capture-derived length)
   are fused only within their group. Confirmed identical for the d=2 model;
   a model with several observed nodes will have several groups and that is
   correct.
2. **No accumulator is read by another loop's body.** `do_transform_rs/7`
   rewrites each marker to its accumulator name in the *surrounding* text, so
   sibling markers are independent — but a `sum` nested inside another `sum`'s
   inner expression would break this, and fusing would then read a partial sum.
   Add an explicit check: if any collected `inner` mentions another group
   member's accumulator name, do not fuse that pair.
3. **No marker survives into a loop body.** Related to (2) and worth its own
   assertion: `find_matching_paren` moves a nested marker's text wholesale into
   the outer loop block, where the recursion never revisits it. That would emit
   a literal `/*REDUCE_SUM*/(` into the shader. It is not known whether the
   emitter can produce nested sums; assert `not String.contains?(body,
   "/*REDUCE_SUM*/")` after the rewrite and find out from a raise rather than
   from a GLSL compile error.

### Why this is the safe half of the work

**Fusion is bit-identical.** Each accumulator visits the same `j` in the same
order over the same values; only the loop nesting changes. There is no
reassociation, so the f64 result is unchanged to the last bit and the
equivalence gates that assert agreement at 1e-15 keep their current tolerances.

That is the whole reason it goes first. The obs-axis parallelism in §6 is a
tree reduction: it *does* change summation order, it *does* need the
tolerances re-derived, and it needs `barrier()` hoisted out of the
`if (in_bounds) { if (tid == i) { ... } }` nest that GLSL's uniform-control-flow
rule forbids it inside. Two hard problems. Neither is required to collect most
of this.

---

## 4. Measured effect — and the prediction it refuted

This section predicted: "If the cost is load-dominated, the slope falls by
close to L (15x for the d=2 model, 165x for the Student-t)."

**Wrong, by an order of magnitude.** Paired on mac-248, same box, same
session, only the exmc commit differing — `745376bb5` (unfused) against
`f2ae139d7` (fused), slope in us per observation, taken over n_obs in
{16, 64, 256}:

| model | loops | before | after | speedup |
|---|---|---|---|---|
| `y ~ N(mu, 1)`, d=1 | 3 -> 3 | 2.3 | 2.4 | 1.0x (control) |
| `y ~ N(mu, sigma)`, d=2 | 15 -> 5 | 130.8 | 72.5 | **1.80x** |
| `y ~ T(df, mu, sigma)`, d=3 | 165 -> 7 | 715.2 | 391.1 | **1.83x** |

23.6x fewer traversals bought 1.83x. The d=1 arm is a genuine control: its
three loops live in three different template holes and cannot merge, and it
did not move.

### The 2x2 that says why, instead of leaving it to inference

`config :exmc, glsl_cse: false` toggles the CSE pass, so fusion and CSE can be
crossed rather than attributed by reading static op counts. Slopes, us/obs:

| d=3 | CSE on | CSE off |
|---|---|---|
| unfused (165 loops) | 715.1 | 728.8 |
| fused (7 loops) | 389.9 | 488.7 |

| d=2 | CSE on | CSE off |
|---|---|---|
| unfused (15 loops) | 130.6 | 130.8 |
| fused (5 loops) | 72.6 | 73.2 |

Reading the cells:

* **CSE on the unfused shader is worth nothing** — 1.9% at d=3, 0.2% at d=2.
  That is the `@cse_min_len 18` diagnosis in §2 confirmed directly: with ~4-op
  bodies there is nothing long enough to hoist.
* **Fusion alone** (CSE off, both arms) is **1.49x** at d=3 and **1.79x** at
  d=2. This is the loop overhead and the L-1 redundant `obs_j` loads, and it
  is the larger half.
* **CSE after fusion** adds **1.25x** at d=3 and **nothing** (0.8%) at d=2 —
  even though it removes 31% of d=2's static ops (117 -> 81). The SPIR-V
  compiler was evidently already eliminating those; at d=3's 671 ops it is
  not, and source-level hoisting starts to pay.

So the honest decomposition is: most of the win is fusion itself, a further
quarter at large op counts is CSE that fusion unlocks, and **the arithmetic
that remains is the floor** — fusion removes traversals, not operations, and
after it the per-op cost is ~0.92 us against ~1.09 before.

An intermediate inference of mine, that CSE was the dominant mechanism because
static op counts fell 31-39%, was also wrong and is recorded here for the same
reason as §1's two: the op count moved and the clock did not.

### What this means for the queue

The remaining cost is arithmetic executed by one invocation while 255 idle, so
§6's obs-axis parallelism is now the whole of the remaining opportunity rather
than a second-order term. It was worth doing fusion first anyway — it is
bit-identical and it took ~1.8x off every observed model on the way — but it
does not substitute for it.

## 5. Verification

The failure mode for this class is a finite, plausible, wrong log-density, so
"the tests pass" is not evidence. In order:

**Done for this change:** step 2 (bit-identity, 18 goldens, all matching),
step 3 (the sweep, above), step 4 (SPIR-V fell: d=2 41992 -> 35392, d=3
568456 -> 492552) and the local half of step 5 (super-io Vulkan arm 688/1,
the same Cauchy KS failure that predates the change; an `ess/1` flake on an
unseeded `:rand.normal()` passed 3/3 on re-run). **Step 1 was NOT done** —
the goldens are a one-shot external check, not a gate in the suite, and
promoting `leapfrog_leaf_diff.exs` remains open. Step 5's fleet and
posteriordb arms remain open.

1. **A gate that can fail, first.** `bench/leapfrog_leaf_diff.exs` is the only
   shader-vs-host numerical harness and it computes `ok_q/ok_p/ok_g/ok_lp` and
   *returns* them — nothing asserts. Promote it into `test/` with those
   booleans asserted **before** touching the emitter, and confirm it fails when
   deliberately broken. `NEXT.md` already carries this as Wave 2 work; it is a
   prerequisite here, not a follow-up.
2. **Bit-identity, which is the whole claim.** For each of the three models
   above, render the GLSL before and after, compile both, dispatch both with
   the same q/p/eps/K, and assert the four output buffers are **byte-identical**
   — not close, identical. Fusion that changes any bit has reassociated
   something and the design argument in §3 is wrong.
3. **The sweep, unchanged.** Re-run the `n_obs` sweep at K=8 for all three
   models and report the slope beside the numbers in §1. Same host, nothing
   else running.
4. **SPIR-V size.** Expect it to *fall*; assert it does not grow. A fusion that
   inflates the shader has duplicated bodies rather than merged them.
5. **End to end.** `EXMC_COMPILER=vulkan mix test` on super-io and mac-248, and
   a posteriordb validate arm, which is where a wrong likelihood shows up as
   R-hat rather than as a diff.

---

## 6. Not in scope, and what it would take

**Obs-axis parallelism.** After fusion there is one loop of `n_obs` iterations
run by one invocation while 255 idle. Spreading it needs: a thread mapping of
`tid -> (rv, obs_lane)` with `T = 256/d` lanes per RV, baked as a literal
because `d` is fixed per synthesised shader (the SPV is content-addressed per
model, so this does not cost cache reuse); a segmented reduction in the
existing `shared double partial[256]`; and the `barrier()` calls hoisted out of
`if (in_bounds) { if (tid == i) ... }`, since GLSL requires uniform control
flow at a barrier and the current nest does not provide it. Cost model
`ceil(n_obs/T) + log2(T)` against `n_obs`, degrading to today's behaviour at
d=256. It changes summation order; see §3.

**Warp divergence across the `tid == i` chain.** Unquantified, see §1. Measure
it with two models of *identical* likelihood complexity and different d before
designing anything.

**The transcendental helpers.** `exp_d`/`log_d` cast to f32 and back (~1 ULP of
f32, ~6e-8). Not a performance item, but it is inside these loops and any
tolerance argument in §5 has to account for it.

**Whether 165 sums is itself avoidable.** This document takes the emitter's
output as given and fixes the GLSL. The upstream question — whether
`CustomSynth.Glsl` could emit one `sum` where autodiff produced many — is
larger, and fusing is worth doing even if the answer turns out to be yes.

---

## 7. Incidental

`do_transform_rs/7` calls `reduce_bounds/4` twice per marker to build one loop
header, and `reduce_bounds/4` runs a regex scan over the body via
`capture_reduce_len/2`. At 165 markers that is 330 scans per render. Synthesis
is cached by SHA so it is not a runtime cost, but the fusion change touches
this line anyway and it should compute the bounds once.
