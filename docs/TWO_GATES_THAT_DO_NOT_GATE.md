# Two gates that do not gate

**Status:** plans. Written 2026-09-10 against `699f3a870`; **substantially
revised 2026-09-11 after measuring on the fleet**, which merged the two items
into one investigation and falsified two claims in the first draft. See
"What the fleet measurement changed" below before reading the rest.

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

The test is seeded (`seed: 42`) and the code and `nx_vulkan` pin are identical
across hosts, so identical draws would give an identical `d` and it would fail
everywhere.

**CORRECTION (2026-09-11).** The first draft said `reference: :none` meant the
KS was against the analytic Cauchy CDF, with no Monte Carlo error on that side.
That is wrong. It is a TWO-SAMPLE KS — `m: 800, n: 800` — between a reference
arm run under compiler `:none` and the Vulkan candidate, and `:none` names the
reference COMPILER, not the absence of one. Both arms carry sampling error.
`validator.ex:189` pins it explicitly and records why: it used to fall through
to whatever `auto_detect/0` found, which on a host without EXLA was Nx.Vulkan
— the arm under test — so the Kepler fleet spent three weeks comparing a run
against itself while the fleet's standing verdict was that super-io was the
unreliable host. That verdict was exactly backwards.

**So the draws differ by host.** That is the finding, and it sits in tension
with a claim recorded in `NEXT.md`: *"both Keplers and the Ampere produce
bit-identical q/p/grad from this shader"*. Either that claim does not extend to
this model, or something outside the shader diverges.

### What the fleet measurement changed

Run on 2026-09-11 at `147305261`. The Cauchy case, verbatim from the test, on
three hosts:

| host | SPIR-V sha | dispatches | result |
|---|---|---|---|
| super-io (Ampere) | `a12cfb9f…` | **1251** | `{:error, d=0.0999…}` |
| mac-248 (GT 750M) | `a12cfb9f…` | **1172** | `:ok` |
| mac-247 (GT 650M) | `a12cfb9f…` | **1172** | `:ok` |

**The same shader, and both arms genuinely dispatching on every host** — so the
"it passes vacuously on the fleet because the candidate falls back to CPU"
hypothesis is refuted. What differs is the DISPATCH COUNT: the two Keplers
agree exactly and the Ampere does not. Same seed and same SPIR-V producing a
different number of dispatches means NUTS grew a different tree, which means
the leapfrog outputs differ.

Confirmed directly against the 18 shader goldens, mac-248 vs super-io:

| model | differing buffers |
|---|---|
| d1, Normal | `logp` in 3 of 6 cases |
| d2, Normal + HalfNormal | **none** — all 6 identical |
| d3, StudentT | `grad` x4, `p` x3, `q` x1; at n_obs=64/k=8 all four |

**Nine of eighteen cases differ between Ampere and Kepler from byte-identical
SPIR-V.**

This falsifies a claim this project has carried in `NEXT.md`: *"both Keplers
and the Ampere produce bit-identical q/p/grad from this shader."* It is true
for the model that claim was measured on and false in general, and the failure
mode of a precisely-worded, precisely-incomplete claim is that nobody re-tests
it.

**The transcendental hypothesis was tested on 2026-09-11. It is not confirmed,
it is partly REFUTED, and the lever is not usable.**

The hypothesis was that `exp_d`/`log_d` being `double(exp(float(x)))` — the
GPU's f32 transcendental unit, whose last-ULP behaviour is explicitly allowed
to differ across architectures — explained the cross-host divergence, and that
`:polynomial` would show it.

| model | SPIR-V under `:f32_cast` → `:polynomial` | cross-host divergence under `:f32_cast` |
|---|---|---|
| d1, Normal | 8968 → **8968, digests identical** | `logp`, 3 of 6 cases |
| d2, Normal + HalfNormal | 35392 → 39356 | none |
| d3, StudentT | 492552 → 496516 | heavy |

**d1 refutes it.** Its SPIR-V and every digest are unchanged by the switch, so
it uses no transcendentals at all — and it is one of the models that diverges
across hosts. Whatever makes an Ampere and a GT 750M disagree on d1's `logp`,
it is not `exp`/`log`.

d2 is uninformative: it never diverged under `:f32_cast`, so there is nothing
for the switch to remove.

**d3 cannot be tested, because `:polynomial` SEGFAULTS the Kepler.** `mix run`
exits 139 with a core, on both mac-247 and mac-248, at DISPATCH — not at
synthesis:

```
STEP 1 OK  spv=496516 bytes     <- compiles
STEP 2 dispatch
[nx_vulkan_vulkano] device: NVIDIA GeForce GT 750M (DiscreteGpu)
EXIT 139                        <- core dumped
```

The same module dispatches correctly on super-io and returns
`[0.2932, 0.1029, 0.2010]`. Both variants pass `Nx.Vulkan.Spirv.validate_file/1`
— 123138 and 124129 words — so this is not the 16-bit word-count wrap class
that DECISION 93 describes. It is valid SPIR-V that one driver runs and another
dies on.

### It is NOT the hardware, and the heading above is a misnomer

Two further hosts, both suggested by the nx_vulkan session, which pointed out
that super-io confounds more than one variable:

| host | GPU | arch | driver | OS | result |
|---|---|---|---|---|---|
| super-io | RTX 3060 Ti | Ampere | 580.178.04 | Linux | **OK** |
| Jetson | Tegra X1 | Maxwell | L4T / Tegra | Linux | **OK** |
| asus | GTX 1660 Ti | Turing | 470.256.02 | FreeBSD | **EXIT 139** |
| mac-248 | GT 750M | Kepler | 470.256.02 | FreeBSD | **EXIT 139** |
| mac-247 | GT 650M | Kepler | 470.256.02 | FreeBSD | **EXIT 139** |

Same module throughout: SPV `86970c0212aadda4…`, 496516 bytes, `validate_file`
`:ok`. The Jetson returns bit-identical numbers to super-io.

**"Old hardware" is eliminated from both directions.** A Turing part five
generations newer than the GT 650M crashes; a Maxwell-era Tegra older than that
Turing runs it. No cutoff year fits either way, so calling this "the Kepler
segfault" is wrong.

**But architecture as an UNORDERED category is not eliminated, and saying it was
is an error this document made.** Corrected 2026-09-11 after the pathmc_ex
session modelled the table: every host sharing an architecture shares an
outcome, and three of the four architectures have exactly one host, so a lookup
table — Ampere runs, Maxwell runs, Turing crashes, Kepler crashes — fits
perfectly. It survives VACUOUSLY. A hypothesis that cannot fail explains
nothing, and "architecture does not predict the outcome" was simply false: it
predicts it exactly, on four data points and four free parameters.

So three hypotheses survive, not two — architecture-as-lookup, driver branch,
and OS — and they are mutually confounded.

**One hypothesis IS eliminated outright, and it is the one nobody proposed.**
super-io and asus share the GLSL cache key `d915a7f8…` and have OPPOSITE
outcomes, so the source-text split cannot explain the crash. That is a real
elimination rather than a vacuous survival, and it is the only one in the set.

**Driver branch and OS remain perfectly confounded.** Every host that works is
Linux; every host that crashes is FreeBSD. The Jetson adds that a second,
unrelated Linux stack handles the module — so it is not something specific to
nvidia-580 — but it does not separate the two, because it is also Linux.

No further host of the kinds available changes this. Separating them requires
an INTERVENTION: nvidia-580 on a FreeBSD box, or nvidia-470 on a Linux one.
Both are operator decisions with real cost — super-io has a history of coming
up on llvmpipe after a driver/module mismatch, where it returns wrong u8
answers while the suite still largely passes — so the cheap experiment is the
FreeBSD ports side, not downgrading the development machine.

Until then the supportable claim is exactly: *nvidia-470 on FreeBSD crashes;
two unrelated Linux stacks do not.* Three crashing hosts are not three
independent observations when they share an OS.

**On "byte-identical", and how it was nearly wrong.** The first version of this
section asserted byte-identity from matching FILE SIZES. That is not evidence:
super-io's cache holds two distinct 492552-byte modules, so size is not a
unique key. Verified properly afterwards — SHA-256 of the module contents is
`86970c0212aadda471911bdd00ed7ef5` on both hosts, so the claim holds. It held
by luck rather than by method, on the same day this document was revised for
exactly that failure.

**And checking it turned up a separate finding.** The two hosts produced
DIFFERENT GLSL for the same model — cache keys `d915a7f8…` on super-io and
`fe41408a…` on mac-248 — which glslang compiled to byte-identical SPIR-V. Our
GLSL text is not host-deterministic while the SPIR-V is.

Not chased. The likely cause is ours: `shortest_repeat/1` in the CSE pass picks
via `Enum.min_by` over a map's enumeration order, and equal-length candidates
break ties by that order, which can differ between OTP builds.

Measured across five hosts, the split follows neither OS nor architecture:

* `d915a7f8…` — super-io (Linux/x86), asus (FreeBSD/x86)
* `fe41408a…` — mac-247, mac-248 (FreeBSD/x86), Jetson (Linux/aarch64)

Two groups whose membership crosses both axes, which rules out the platform-
shaped explanations and leaves the OTP build or term hashing. It matters
because the SPV cache is content-addressed on the source text: "same model
implies same shader SHA" is false across hosts. It can never surface as a wrong
answer — the SPIR-V is identical — only as a cache miss, or as a provenance
claim that quietly is not comparing what it says it is.

Cores preserved at `~/cores/beam.smp.248.polynomial.*.core` and
`~/cores/beam.smp.248.poly_d3.*.core`, moved out of the checkout because
`kern.corefile` is `%N.core` and they land in the working directory.

**Consequences for this document.** The `:polynomial` option is documented as
"the clean fix... available if a downstream model surfaces where the f32
precision loss provably matters". It is not available on half the fleet, and
nothing said so — no test exercises it. That is a third gate that does not
gate, and it belongs on this list rather than in a footnote.

The divergence itself remains unexplained. The next candidates, none tested:
FMA contraction differing by architecture; the `partial[]` workgroup tree
reduction over 256 lanes; or driver-level fast-math. The leaf-diff run in Part 1
is still the instrument — it localises which buffer and which trajectory step
first disagrees, which none of the above guesswork can.

### Consequence: these are one investigation, not two

The Cauchy failure IS cross-host numerical divergence, and the leaf-diff
harness is the instrument that measures exactly that divergence
element-wise along a trajectory. So:

* **The leaf-diff tolerance cannot be set from one host.** The first draft
  proposed 1e-13 from super-io numbers. Those numbers describe super-io's
  agreement with its own CPU, not two GPUs' agreement with each other. The
  fleet run must produce the tolerance, not assume it.
* **Cauchy triage step 1 is the leaf-diff run.** There is no separate
  measurement to design: dispatch the same fixture on three hosts and compare
  element-wise, which is what the harness already does against the host.

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
