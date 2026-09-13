# The PyMC race — plan

**Written 2026-09-13**, at exmc `7db9ffcab`. Status: **PLAN, nothing run.**
**Operator decisions, 2026-09-13: all five below are YES**, and the race also
runs on two FreeBSD hosts (asus, then the NUC) — see *Hosts*.
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
| **super-io** | Linux x86_64 | RTX 3060 Ti | **EXLA** (headline), Vulkan (information) | default NUTS, nutpie | single-chain, 4-chain |
| **asus** | FreeBSD 15, Xeon E5-2699 v3 (18 cores / 36 threads), 64 GiB | **GTX 1660 Ti** (uuid `cd6c2df3`), driver 580.178.04; the Quadro M4000 is not raced (a slower class) | **Vulkan**, pinned by uuid; CPU host tree | default NUTS; nutpie if it builds | single-chain, 4-chain |
| **NUC** | FreeBSD 15, i3-6100U (2 cores / 4 threads), 8 GB | HD 520, Mesa ANV | **Vulkan**; CPU host tree | default NUTS; nutpie if it builds | single-chain only |

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

A second table, same arms, **4 chains in parallel** (`chains=4, cores=4`
against `sample_chains/3`), total ESS/s. That is where the BEAM claim lives.

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
harness work before it. asus: similar or somewhat longer (no EXLA; the CPU
host-tree arm is slow on logistic and SV). The NUC: several times that,
overnight, single-chain table only.

## Step 1 — DONE 2026-09-13, super-io

`bench/pymc_race/.venv`, lock in `bench/pymc_race/requirements.lock`:
Python 3.12.3, **pymc 6.3.2, pytensor 3.3.1, arviz 1.3.0, nutpie 0.16.11**,
numpy 2.5.3. Both samplers sample a Normal/Exponential model. `pm.sample` now
returns a `DataTree` (ArviZ 1.x), and `arviz.ess(idata, method="bulk")` works on
it; that is the call score.py uses for both frameworks.

## Order of work

1. venv, pins, `requirements.lock`; confirm PyMC 6.3.2 and nutpie import and
   sample `simple`.
2. Port the seven models to `bench/pymc_race/models.{py,exs}`; freeze data.
3. **Gate 1** on all seven. Fix or drop any model that fails it.
4. Reference runs; check PyMC and exmc references agree.
5. Pilot: one seed, all arms, all models, end to end through `score.py`.
6. The full run, super-io idle.
7. Results file, then README.
8. asus, in its agreed window: `~/exmc_race` checkout, venv, gate 1 re-run on
   that host, then the same run with the Vulkan and CPU arms.
9. The NUC, overnight: single-chain table only.

## Decisions — resolved 2026-09-13, all YES

1. **nutpie** is an arm (on FreeBSD, where it builds).
2. **Initialisation:** each framework's defaults are the headline; a shared
   explicit init is a one-seed control beside it.
3. **Core pin:** 1 core for the single-chain table, 4 cores for the 4-chain
   table (not run on the NUC).
4. A **scalar-RV exmc variant** runs too, and the graph-shape cost is published.
5. The **Vulkan arm** runs on super-io as information; on asus and the NUC it
   is the headline exmc arm.

Still open: the asus window and toolchain (asked of the nx_vulkan session).
