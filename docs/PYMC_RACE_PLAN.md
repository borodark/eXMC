# The PyMC race — plan

**Written 2026-09-13**, at exmc `7db9ffcab`. Status: **PLAN, nothing run.**
**Operator decisions, 2026-09-13: all five below are YES**, and the race also
runs on two FreeBSD hosts (asus, then the NUC) — see *Hosts*.
**Operator decision, 2026-09-13 (night): PyMC's JAX path is added as a
baseline** (numpyro on JAX, CPU and CUDA, super-io only). See *Arms* and
*The JAX baseline*.
It replaces the February 2026 comparison the README used to publish
(`STANDARD_BENCHMARKS.md`, now bannered historical).

## The question, and what counts as an answer

*On the same machine, with each framework used the way its users would use it,
how many effective samples per second does eXMC deliver against the latest
PyMC, on the standard models, with both samplers verifiably correct?*

The answer is evidence only if a stranger can rerun it. So the harness, the
data, the pinned environment and the raw per-run results are all committed; the
README table is generated from the results file, losses included.

## What was wrong with the February race, and the fix for each

| February | Why it is not evidence | This plan |
|---|---|---|
| exmc sampler had two NUTS defects (fixed 2026-08, MISSION §6.1) | ESS was measured on draws a correct sampler would not produce | run at a commit after the fixes, recorded |
| harness outside this repo (phd.git `benchmark/`) | not reproducible from the public code | `bench/pymc_race/`, committed |
| ESS from each framework's own estimator | the ratio mixes two estimators | ArviZ computes ESS for **both**, from exported draws |
| ProcessPool saturating 88 cores, jobs in parallel | timings measure contention | runs sequential, pinned to one fixed core set, host otherwise idle |
| wall time including model compilation | a compile-cost race, not a sampling race | compile and first-run cost reported separately; the sampling wall is timed after an untimed warm-up run |
| no check that the two models are the same model | a Jacobian or a parameterisation slip is a different posterior | **log-density parity gate** before any timing |
| no correctness gate; PyMC seed 256 counted with 999 divergences | a broken run can post the best ESS/s | a run's ESS/s counts only if it passes the gate below |
| 5 seeds | too noisy on the funnel and SV | 10 seeds |
| no commit, host, versions or arm | unattributable | `stats.provenance` for exmc, `pip freeze` for PyMC, CPU model, both recorded per run |
| PyMC 5.x at the time; 5.27.1 installed on super-io today | not the latest release | **PyMC 6.3.2**, latest on PyPI at 2026-09-13, pinned in a dedicated venv |

## Environment

- **Host:** super-io (Tier 1, Linux x86_64). Fleet runs, the width race and the
  nx_vulkan session's work kept off it for the duration; the asus session is
  told before and after.
- **PyMC side:** a venv at `bench/pymc_race/.venv` (never the system
  site-packages, which hold 5.27.1): `pymc==6.3.2` with the `pytensor`,
  `arviz` and `numpy` it resolves, plus `nutpie==0.16.11`. From PyPI metadata:
  PyMC 6.3.2 needs **Python >= 3.12**, `pytensor>=3.2.2,<3.4` and
  **`arviz>=1.1,<2`**. ArviZ 1.x is a new major version, so step 1 confirms
  the ESS call (`ess_bulk`/`ess_tail`) under 1.x before anything depends on it.
  `requirements.lock` is committed from `pip freeze`. The PyTensor compile
  cache is warmed by the untimed run, then left alone.
- **PyMC on JAX:** a SECOND venv, `bench/pymc_race/.venv-jax`, with the same
  `pymc==6.3.2` and `pytensor==3.3.1` pins plus `jax[cuda13]` 0.11.1 (jaxlib
  0.11.1, the CUDA 13 plugin; super-io's driver is 580.178.04) and `numpyro`
  0.21.0 (latest on PyPI at 2026-09-13). It is a separate venv so JAX's own
  numpy and CUDA wheels cannot move the pins under the default and nutpie
  arms. Its lock is `requirements-jax.lock`. PyMC declares no JAX extra, so
  these pins are the harness's choice, recorded.
- **eXMC side:** `MIX_ENV=prod`-shaped run via `mix run`, the consumer path.
  EXLA arm (host client), which is what a Linux user runs. The commit and
  `mix.lock` are recorded; `stats.provenance` goes into every result.
- **Threads and cores:** both frameworks pinned with `taskset` to the same core
  set (1 core for the single-chain table, 4 for the 4-chain table),
  `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` set to the
  core count for both, so BLAS cannot quietly borrow cores the other side is
  denied. The CPU model and `nproc` inside the pin are recorded.

## Hosts

Three machines, one protocol. The references (gate 2) are computed once, on
super-io, and reused: the data and the posterior are identical everywhere.

| host | OS, CPU | GPU | exmc arms | PyMC arms | tables |
|---|---|---|---|---|---|
| **super-io** | Linux x86_64 | RTX 3060 Ti | **EXLA** (headline), Vulkan (information) | default NUTS, nutpie, **numpyro on JAX: CPU and CUDA** | single-chain, 4-chain |
| **asus** | FreeBSD 15, Xeon E5-2699 v3 (18 cores / 36 threads), 64 GiB | **GTX 1660 Ti** (uuid `cd6c2df3`), driver 580.178.04; the Quadro M4000 is not raced (a slower class) | **Vulkan**, pinned by uuid; CPU host tree | default NUTS; nutpie if it builds; no JAX (below) | single-chain, 4-chain |
| **NUC** | FreeBSD 15, i3-6100U (2 cores / 4 threads), 8 GB | HD 520, Mesa ANV | **Vulkan**; CPU host tree | default NUTS; nutpie if it builds; no JAX (below) | single-chain only |

**No JAX on the FreeBSD hosts.** jaxlib 0.11.1 publishes wheels for manylinux
x86_64 and aarch64, macOS arm64 and Windows only (PyPI, checked 2026-09-13).
A source build means Bazel plus XLA on FreeBSD, which this race does not
attempt. The JAX columns on asus and the NUC read "no jaxlib for FreeBSD". That
is part of the reach question those hosts answer, not a gap in the protocol.

**Why the FreeBSD hosts matter.** FreeBSD has no EXLA and no CUDA. There the
race is the reach thesis itself, against a compiled competitor rather than
the interpreter: *on a machine with a GPU but no CUDA, does exmc on Vulkan
deliver more effective samples per second than PyMC on the CPU?* asus
answers it on a discrete NVIDIA card, the NUC on a commodity iGPU.

**Vulkan arm coverage.** A model that does not synthesise a chain shader is
refused at compile time (the Plan-B' guard), and is recorded as a refused row,
not silently run per-op. Logistic (Custom Bernoulli likelihood) and SV
(GaussianRandomWalk with a StudentT likelihood on exp(s)) are the ones to
check first.

**FreeBSD Python stack.** PyMC is pure Python. FreeBSD packages provide
`python3.12`, `py312-numpy` 2.4.6, `py312-scipy` 1.17.1, `py312-xarray` and
`py312-arviz` 1.1.0; packaged `py312-pytensor` is 3.1.2, below PyMC's floor, so
PyTensor is built by pip against packaged Cython 3.2.8. **nutpie publishes no
FreeBSD wheel** (0.16.11: manylinux, macOS and Windows only); a source build
needs cargo plus `pyarrow` and `obstore`. It is attempted; if it does not build,
the nutpie column on that host is recorded as "not buildable on FreeBSD", with
the error.

**Installed on asus, 2026-09-13.**
- **From pkg**, additive only: `py312-numpy` 2.4.6, `py312-scipy` 1.17.1,
  `py312-numba` 0.67.0, `py312-llvmlite` 0.49.0, `py312-pandas` 2.3.3,
  `py312-pyarrow` 24.0.0, `py312-xarray` 2026.4.0, `py312-maturin`,
  `py312-cython`. The dry run listed 84 new packages and nothing upgraded or
  removed.
- **The venv**: `~/pymc_race_venv` with `--system-site-packages`, and pip
  constrained to `requirements.lock` for everything pkg did not provide:
  pymc 6.3.2, pytensor 3.3.1, arviz 1.3.0, and **nutpie 0.16.11 built from
  source** with cargo 1.94.0 in about 13 minutes at nice 15.
- **Lock**: `requirements-freebsd.lock`. numba and llvmlite match super-io
  exactly. numpy, scipy, pandas, pyarrow and xarray are a minor version behind,
  and every run records its versions.
- **Smoke test on `simple`** (1000 + 1000, one chain): PyMC NUTS gave mu 3.1396,
  sigma 1.1653; nutpie gave mu 3.1253, sigma 1.1676. Both in float64.

**asus** is shared (the nx_vulkan two-GPU work, the ex_pathmc session). Its
`~/exmc_oss` belongs to the two-GPU work, so the race uses a separate checkout
(`~/exmc_race`) and a venv under it (agreed by the nx_vulkan session;
`~/exmc_oss` and `~/nx_vulkan` stay untouched). Toolchain there: python3.12
from pkg with bundled pip via venv, clang 19.1.7 (gcc13 also present), cargo
1.94.0. **The window is the operator's to schedule**: the box is shared with
the nx_vulkan session (background cargo builds and suites on the M4000) and
ex-pathmc-39 (sampling tests on the 1660 Ti). Never `nvidia-smi -pm 1` there;
it is host-fatal (nx_vulkan NEXT.md §0a).

**The NUC** goes last and overnight. The 4-chain table is skipped there (four
chains would saturate its two cores and turn the table into a contention
measurement), and the venv plus any nutpie build must fit in the ~4 GB free on
its 11 GB pool. The width race on the same box (2026-09-13) spent 729 s
sampling one d=32 Vulkan cell, so budget several times super-io's hours.

## Models

The seven from February, specs taken verbatim from phd.git
`benchmark/benchmark_pymc.py` (commit recorded when copied), data from its
`data/benchmark_data.json`, frozen into `bench/pymc_race/data.json` with its
sha256 in the results.

| model | d | shape | notes |
|---|---|---|---|
| simple | 2 | Normal mean, Exponential sigma, ~observations | throughput-bound |
| medium | 5 | two groups sharing a hierarchical prior | |
| stress | 8 | three groups, per-group noise | |
| eight_schools | 10 | **centered**, Rubin (1981) data | divergences expected; that is the point |
| funnel | 10 | Neal's funnel, no data | analytic marginals: y ~ N(0, 3), Var(x_i) = exp(4.5) |
| logistic | 21 | n=500, p=20, `Bernoulli(logit_p = alpha + X beta)` | exmc side via `Dist.Custom`; compute-bound |
| sv | 102 | `GaussianRandomWalk` log-volatility, StudentT returns, T=100 | the hard one |

**Vectorisation is part of the comparison, and it is recorded.** PyMC's
builders use `shape=` vectors (eight schools' theta, logistic's beta, SV's s).
The eXMC side is written the idiomatic eXMC way *today* — vector RVs where
exmc supports them — and the results file says which form each side used.
A scalar-RV exmc variant is also run, to publish the graph-shape cost
MISSION §5.2c describes.

## Gate 1 — the two sides are the same model

Before any timing, for every model: draw 200 points in the unconstrained space,
map them to each framework's parameter ordering and transforms, and compare
PyMC's `model.compile_logp(jacobian=True)` against exmc's compiled log-density
(Jacobians included). **Pass:** max relative difference < 1e-9 (a constant
offset from normalising terms is removed first and reported). A model that
fails does not race.

## Sampler settings, both sides

1000 tune / 1000 draws, `target_accept=0.8`, max tree depth 10, diagonal mass
matrix adaptation, one chain per run for the headline table.

**Initialisation — decided.** (a) Each framework's defaults (PyMC's
`jitter+adapt_diag`, exmc's own), which is what a user gets and is the
headline; and (b) both from the same explicit point (February's init
dicts), which isolates the sampler from the initialiser. (a) is the
headline and (b) a one-seed control, reported beside it.

## Arms

| arm | what | status |
|---|---|---|
| **PyMC 6.3.2, default NUTS** | `pm.sample(nuts_sampler="pymc")` | always |
| **PyMC + nutpie 0.16.11** | `pm.sample(nuts_sampler="nutpie")`, PyMC's fastest supported path | yes — leaving it out would race a weaker PyMC than users can install |
| **eXMC, EXLA arm** | `Sampler.sample/3` | always |
| eXMC, Vulkan arm on super-io | information only, not in the headline | optional |
| **PyMC 6.3.2 + numpyro 0.21.0 on JAX 0.11.1, CPU** | `pm.sample(nuts_sampler="numpyro")`, JAX on the same pinned cores | super-io |
| **PyMC 6.3.2 + numpyro 0.21.0 on JAX 0.11.1, CUDA** | the same on the RTX 3060 Ti | super-io |
| PyMC + blackjax 1.5 on JAX | `nuts_sampler="blackjax"` | one-seed control, not scored |

A second table, same arms, **4 chains in parallel** (`chains=4, cores=4`
against `sample_chains/3`), total ESS/s. That is where the BEAM claim lives.

## The JAX baseline

**Why it is in.** It is the fastest path PyMC offers on a GPU, and on CPU it is
what many PyMC users switch to for speed. A race without it would leave out the
comparison a sceptical reader asks for first.

**What it isolates.** exmc's EXLA arm and JAX both compile through XLA.
- **exmc** compiles the gradient and leapfrog steps and runs the NUTS tree on
  the BEAM host (with a Rust NIF for subtrees).
- **numpyro** traces the whole NUTS transition, tree included, into one XLA
  program.

So the CPU pair holds the compiler fixed and measures the two framework
designs. The CUDA arm pairs with exmc's Vulkan arm on the same card: GPU
against GPU. That table is still information on super-io, as decision 5 says,
but it is printed side by side. No outcome is predicted here; losses are
published like wins.

**What changes in the protocol:**
- **Gate 1b.** PyTensor's JAX backend compiles the same graph, but not the
  same arithmetic. Gate 1's points are re-evaluated with the model's logp
  compiled for JAX (`jax_enable_x64` on), against the C backend: max relative
  difference < 1e-9. A model that fails races on the other arms only, and is
  recorded as such.
- **Precision.** f64 on every arm, recorded; a float32 JAX run is not raced.
- **Compile cost.** JAX compiles the sampler as well as the model. It is large,
  and it is reported separately, as the protocol already does for every arm.
- **Chains.** For the 4-chain table: `chain_method="parallel"` with
  `XLA_FLAGS=--xla_force_host_platform_device_count=4` on CPU. On one GPU,
  `"vectorized"` is the only multi-chain method, recorded as such.
- **GPU hygiene.** `XLA_PYTHON_CLIENT_PREALLOCATE=false`. The card is otherwise
  idle: no nx_vulkan runs, no fleet runs, and exmc's EXLA arm is the host
  client. `nvidia-smi` is read before and after each block.
- **Gate 2** unchanged: the same references, the same thresholds.

## Measurements, per run

- `min_ess_bulk` and `min_ess_tail` over every scalar parameter, from
  `arviz.ess` on the draws (exmc's exported to the same InferenceData layout);
- sampling wall time, and separately the compile and first-run time;
- **ESS/s** = `min_ess_bulk / sampling_wall_s`;
- divergences (post-warmup), final step size, mean tree depth, and gradient
  evaluations per draw where the framework reports them. That separates
  "better adaptation" from "faster gradients", the two things February's
  write-up had to guess between;
- provenance: exmc `stats.provenance`, the PyMC venv's `pip freeze`, CPU model,
  core pin, data sha256, harness commit.

## Gate 2 — a run's ESS/s counts only if the run is correct

Per model, across the 10 seeds of one arm:

1. R-hat (rank-normalised, ArviZ) < 1.01 for every parameter, treating the
   seeds as chains;
2. every parameter's pooled posterior mean within 4 Monte-Carlo standard errors
   of the reference;
3. post-warmup divergences below 1% of draws. **Above it, the run is reported
   but marked "not ESS-comparable"** and kept out of the ratio. Eight schools
   (centered) and the funnel will trip this on both sides, and that is the
   honest result for those two models.

**References:** funnel, analytic; eight schools (centered), the posteriordb
reference for `eight_schools-eight_schools_noncentered` (in
`benchmark/posteriordb/posteriordb_processed/`). The two parameterisations
describe the same posterior over (mu, tau, theta), and the non-centered one is
the one a sampler can actually get right, so its reference is the better
yardstick for the hard centered run. The theta draws are mapped back
(theta = mu + tau * theta_tilde) before comparing. The other five: a
long run of each framework (4 chains × 10,000 draws). The two must agree
within MCSE before either is used. If they disagree, that is a finding, and
the race waits for it to be resolved.

## Outputs

- `bench/pymc_race/`: `models.py`, `models.exs`, `data.json`,
  `requirements.lock`, `parity.py` + `parity.exs` (gate 1), `run_pymc.py`,
  `run_exmc.exs`, `score.py` (ArviZ ESS, gate 2, the tables);
- `bench_results/PYMC_RACE_<date>.md`: both tables, gates per model, every
  run's raw record, provenance; losses and "not ESS-comparable" rows included;
- the README Performance section rebuilt from that file, dated, linking to it.

## Budget — ESTIMATED

February's per-run times (from before the fixes, so treat them as a scale):
simple a few seconds; logistic 4 s (PyMC) to 16 s (exmc); SV 35–54 s (PyMC)
to 83–95 s (exmc). Ten seeds × seven models × three arms ≈ 45–60 min for the
single-chain table; the 4-chain table about the same; reference runs ≈ 30 min;
gate 1 and warm-ups ≈ 10 min. **About 2.5–3 hours of super-io**, plus the
harness work before it. **The two JAX arms add about 1–1.5 hours**
(ESTIMATED: JAX's per-model compile is unmeasured here, and SV's may dominate),
so **about 4–4.5 hours** in all. The JAX venv and gate 1b take about 30 min
before the pilot. asus: similar or somewhat longer (no EXLA; the CPU
host-tree arm is slow on logistic and SV). The NUC: several times that,
overnight, single-chain table only.

## Step 1 — DONE 2026-09-13, super-io

`bench/pymc_race/.venv`, lock in `bench/pymc_race/requirements.lock`:
Python 3.12.3, **pymc 6.3.2, pytensor 3.3.1, arviz 1.3.0, nutpie 0.16.11**,
numpy 2.5.3. Both samplers sample a Normal/Exponential model. `pm.sample` now
returns a `DataTree` (ArviZ 1.x), and `arviz.ess(idata, method="bulk")` works on
it; that is the call score.py uses for both frameworks.

## Steps 2–3 — DONE 2026-09-13, super-io: the models, and gate 1 passes

`bench/pymc_race/`: `data.json` (the seven models' data from phd.git
`benchmark/data/benchmark_data.json` @ `c579dba`, sha256 `5cc2601a58508af3`),
`models.py` (PyMC 6.3.2), `models.exs` (exmc, `:vector` and `:scalar`
variants), `parity.py` + `parity.exs`. The points file is regenerated from
its seed and not committed.

Gate 1 compares log densities at 200 PyMC points per model in CONSTRAINED
space (PyMC `compile_logp(jacobian=False)` against exmc's compiled density
minus exmc's own Jacobian), on exmc's CPU arm with NCP off:

| model | exmc variant | d | result | constant offset | worst relative residual |
|---|---|---|---|---|---|
| simple | (one form) | 2 | PASS | 3.2e-08 | 6.6e-16 |
| medium | (one form) | 5 | PASS | 3.0e-08 | 7.0e-16 |
| stress | (one form) | 8 | PASS | 3.4e-08 | 6.6e-16 |
| eight_schools | vector / scalar | 10 | PASS | 6.0e-08 | 3.3e-16 / 4.1e-16 |
| funnel | vector / scalar | 10 | PASS | 8.270447 (exmc's Custom drops the normaliser) | 4.2e-16 |
| logistic | vector / scalar | 21 | PASS | 6.7e-07 | 4.9e-16 / 7.7e-16 |
| sv | vector / scalar | 102 | PASS | 5.4e-06 | **6.9e-10** |

**Three findings the gate produced, and what the race does about each:**

1. **February's SV race compared two different models.** PyMC's
   `GaussianRandomWalk` without `init_dist` defaults the first step to
   Normal(0, 100); exmc's has x[0] ~ Normal(0, sigma). MEASURED as a negative
   control: with February's PyMC spec, both SV variants FAIL at a worst
   relative residual of 5.2e-2 (offset about -95), every other model still
   PASSes. The race's PyMC SV passes `init_dist=pm.Normal.dist(0, sigma)`.
2. **SV's residual is near the tolerance and not constant**: 6.9e-10 relative,
   about 5e-4 absolute on log densities near 7e5. INFERRED cause: exmc's
   `Exmc.Math.lgamma` is a Lanczos approximation and PyMC uses exact `gammaln`,
   and the StudentT likelihood's `lgamma((nu+1)/2) - lgamma(nu/2)` varies with
   `nu`. The model is the same; the arithmetic differs in the last digits. It
   is recorded beside SV's results.
3. **The parameter spaces differ where the models do not.** exmc samples
   HalfNormal through softplus (eight schools' tau), PyMC through log. The
   same posterior, sampled in different coordinates, which affects geometry
   and so ESS. That is a real framework difference, reported as such, not
   normalised away.

Also measured: February's exmc funnel clamped y/2 to [-20, 20]; the race's
does not.

## Step 5 — DONE 2026-09-13 (night), super-io: the JAX venv, and gate 1b passes

**The venv.** `bench/pymc_race/.venv-jax`, lock `requirements-jax.lock`:
pymc 6.3.2, pytensor 3.3.1, arviz 1.3.0, numpy 2.5.3 and scipy 1.18.1, the same
as the main venv. Also jax and jaxlib 0.11.1 with the CUDA 13 plugin,
numpyro 0.21.0 and **blackjax 1.5**.
- **Why not blackjax 1.6.x.** 1.6, 1.6.1 and 1.6.2 all fail inside PyMC 6.3.2's
  blackjax path: `TypeError: build_kernel.<locals>.kernel() got an unexpected
  keyword argument 'progress_bar'`. 1.5 and 1.4 sample, so 1.5 is the newest
  that works with the latest PyMC.
- **Smoke test on `simple`, 1000 + 1000, one chain.** numpyro gave mu 3.144,
  sigma 1.164 on CPU and mu 3.140, sigma 1.165 on CUDA (reference mu 3.136,
  sigma 1.169). PyMC switches on `jax_enable_x64` itself, and the draws are
  float64.
- **Device check.** `jax.devices()` read CudaDevice(id=0) with the GPU visible
  and CpuDevice with `CUDA_VISIBLE_DEVICES=""`.

**Gate 1b** (`parity_jax.py`, tolerance 1e-9). Points: 200 around the initial
point and 200 from the reference posteriors. Log density and gradient with the
Jacobian, JAX backend against the C backend:

| model | CPU logp | CPU dlogp | CUDA logp | CUDA dlogp | result |
|---|---|---|---|---|---|
| simple | 1.3e-15 | 1.5e-14 | 8.3e-16 | 1.1e-14 | PASS |
| medium | 9.2e-16 | 1.5e-14 | 7.1e-16 | 1.3e-14 | PASS |
| stress | 7.1e-16 | 1.8e-14 | 5.0e-16 | 1.4e-14 | PASS |
| eight_schools | 5.3e-16 | 3.9e-15 | 3.0e-16 | 2.3e-15 | PASS |
| funnel | 1.9e-15 | 2.1e-15 | 1.9e-15 | 2.6e-15 | PASS |
| logistic | 1.6e-15 | 1.7e-14 | 1.6e-15 | 1.0e-14 | PASS |
| sv | 2.0e-14 | 1.9e-12 | 2.9e-14 | 1.9e-12 | PASS |

## Order of work

1. ~~venv, pins, `requirements.lock`; confirm PyMC 6.3.2 and nutpie import and
   sample `simple`.~~ DONE.
2. ~~Port the seven models to `bench/pymc_race/models.{py,exs}`; freeze data.~~ DONE.
3. ~~**Gate 1** on all seven.~~ DONE, all PASS.
4. ~~Reference runs; check PyMC and exmc references agree.~~ DONE, after
   exmc `509e22b26`: all seven AGREE (NEXT.md has the table). The references
   found an exmc sampler defect first: the speculative subtree NIF built
   backward subtrees with swapped endpoints.
5. ~~**JAX baseline setup, super-io**~~ DONE: `.venv-jax`, numpyro on CPU and
   CUDA, gate 1b PASS on all seven (Step 5 above).
6. **exmc tree-health gate.** ex-pathmc-39's mediation model went from 1.95
   leapfrog steps per draw and 94 of 150 divergent at `d85630ab6` to 314.5
   steps and 19 at `509e22b26`, while its test checked only names.
   - **Not yet attributed.** That range also holds the vector-RV NCP
     reconstruction fix (`134f9d3aa`) and the Rustler 0.38 bump.
   - **A probe that did not discriminate.** exmc's own weakly identified
     regression (a ridge, 300 + 300, 3 seeds, EXLA and CPU arms) gave the same
     tree statistics with the old `going_right=true` as with the fix: about
     90–140 steps per draw, depth about 6.2, no divergences.
   - **So:** the gate needs a model shown to fail on a real regression. The
     candidate is pathmc's mediation model, once a one-commit comparison
     (`05944d18a` against `509e22b26`) says which change it measured.
   - **Until then**, nuts_test 21b (speculative and direct NIF trees identical)
     is the regression test for the direction defect.
7. **Harness:** `run_pymc.py` (arms: default, nutpie, numpyro-cpu,
   numpyro-cuda; blackjax control), `run_exmc.exs`, `score.py`.
8. Pilot: one seed, all arms, all models, end to end through `score.py`.
9. The full run, super-io idle.
10. Results file, then README.
11. asus, in its agreed window: `~/exmc_race` checkout, venv (PyMC stack:
    installed 2026-09-13, see *FreeBSD Python stack*), gate 1 re-run on that
    host, then the same run with the Vulkan and CPU arms. The **reciprocal
    reference run** comes first: exmc's CPU arm on FreeBSD, scored against
    super-io's references (`REF_COMPILER=none`, `REF_OUT`, `EXMC_REF`), which
    tests the cross-host statistical promise in docs/REPRODUCIBILITY.md.
12. The NUC, overnight: single-chain table only.

## Decisions — resolved 2026-09-13, all YES

1. **nutpie** is an arm (on FreeBSD, where it builds).
2. **Initialisation:** each framework's defaults are the headline; a shared
   explicit init is a one-seed control beside it.
3. **Core pin:** 1 core for the single-chain table, 4 cores for the 4-chain
   table (not run on the NUC).
4. A **scalar-RV exmc variant** runs too, and the graph-shape cost is published.
5. The **Vulkan arm** runs on super-io as information; on asus and the NUC it
   is the headline exmc arm.
6. **PyMC on JAX** (numpyro, CPU and CUDA) is a baseline arm on super-io
   (decided 2026-09-13, night); blackjax is a one-seed control.

Still open: the asus window and toolchain (asked of the nx_vulkan session).
