# Reproducibility — what eXMC promises, and the measurements it rests on

**Written 2026-09-13**, at exmc `098a802f0`–`e0d72e6f4`, nx_vulkan lock
`8116a19`. The README's [Reproducibility](../README.md#reproducibility) section
is the short form; this is the evidence, and the place to update when a
measurement moves.

## The contract

| claim | promised? | enforced by |
|---|---|---|
| Same seed, same arm, same build, same host → **bit-identical draws**, one chain or many | **yes** | `test/reproducibility_contract_test.exs`, on the CPU arm, the detected arm, and the Vulkan arm where present |
| A different seed → different draws | yes | the same test (it is what stops the first check passing vacuously) |
| Every result says what produced it | yes | `stats.provenance`, asserted by the same test |
| Different host, or different arm → **the same posterior, statistically** | **yes** | ESS-sized statistical checks in the suite; `bench/nuts_truth.exs` and `bench/nuts_width_race.exs` against closed-form posteriors, per host |
| Different host, or different arm → identical draws | **no** | — and the reasons are measured below |

"The same build" means the same exmc commit, the same `mix.lock` (which pins
nx_vulkan, nx and exla), the same OTP, and on the Vulkan arm the same driver.

## `stats.provenance`

Every stats map a sampler returns carries it (`Exmc.NUTS.Sampler`):

```elixir
%{
  seed: 7,                      # this chain's seed (sample_chains derives one per chain)
  arm: "compiler=Nx.Vulkan (configured: :vulkan) backend=... precision=:f64 ...",
  device: %{name: "NVIDIA GeForce RTX 3060 Ti", uuid: "f7e146ef...", pci: "0000:81:00.0",
            driver: "580.178.04", selected_by: "NXV_DEVICE=uuid:f7e146ef"},   # nil off Vulkan
  exmc: "0.4.0", nx: "0.13.1", nx_vulkan: "0.4.0", exla: "0.13.1",
  elixir: "1.18.4", otp: "27",
  os: "unix/linux", os_version: "6.8.0", arch: "x86_64-pc-linux-gnu", host: "super-io"
}
```

`arm` is `Exmc.JIT.describe/0`, the line `test/test_helper.exs` prints.
`device` is the device actually open, because on a two-GPU host the arm alone
does not say which card ran. The nx_vulkan **commit** is not recoverable at
runtime (a git dependency's version is its `mix.exs` version); `mix.lock`
records it. `host` is the machine's name: strip it before publishing a trace if
that matters to you.

## Same host: bit-identical — MEASURED

super-io, 2026-09-13, a Normal + HalfNormal (log-transformed) model, raw bytes
compared:

| arm | one chain, 500 warmup / 1000 draws | 4 chains, 300 / 500 | different seed differs |
|---|---|---|---|
| CPU (`:none`) | identical | identical | yes |
| Vulkan (chain shader) | identical | identical | yes |
| EXLA | identical | identical | yes |

The contract test repeats this (smaller) on every suite run. Parallel chains
are identical run to run because each chain's RNG state is derived from the
seed and threaded explicitly; scheduling order does not reach the draws.

## Across hosts: not bit-identical — MEASURED

### The host `libm` differs, even glibc to glibc

BEAM `:math` over the same 1,000,000 inputs (`:rand` exsss seed `{1,2,3}`),
sha256 of the raw f64 results, first 8 hex digits:

| function | super-io, glibc 2.39 x86_64 | mac-247, FreeBSD 15 msun amd64 | Jetson, glibc 2.27 aarch64 |
|---|---|---|---|
| inputs | `E065C4AD` | `E065C4AD` | `E065C4AD` |
| `sqrt` | `40AC5570` | `40AC5570` | `40AC5570` |
| `log` | `063CEF77` | `22414E99` | `F809B779` |
| `exp` | `ADEE036E` | `A7C66FD5` | `6329BC3A` |
| `log(1+x)` | `725CEE53` | `9CDCB469` | `AFDFF8BB` |
| `pow(x, 1.5)` | `C252E680` | `57890578` | `36140D92` |

Only `sqrt`, which IEEE 754 requires to be correctly rounded, agrees. The NUTS
tree calls `log` and `exp` on the host (energies, log-sum-exp in multinomial
sampling) on every arm.

### GPU vendors differ in the last bits

From nx_vulkan `scripts/arch_float_divergence.exs` at `eddb973` (their
NEXT_SESSION.md has the digests): NVIDIA Ampere, Turing, Maxwell and Kepler
agree bit-for-bit on `log`, `exp`, the Cauchy kernel and f64 division; the
Keplers alone differ on `sqrt`; Intel HD 520 on Mesa ANV differs on the f32-cast
`log`/`exp` the chain shader uses by default (and is more accurate there) and
on f64 division (`x/3` one f64 step off where NVIDIA is exact).

### What that does to posteriors

`bench/nuts_truth.exs`, seeds 1–6 pooled, 500 / 2000, same exmc commit on every
host (`docs/ARMS.md` has the full table):

| run | Normal(0,1) mean | var |
|---|---|---|
| super-io, CPU arm | −0.016021 | 1.011326 |
| NUC and mac-248 (FreeBSD), CPU arm | −0.015276 | 1.011511 |
| super-io, Vulkan (Ampere) | −0.003061 | 0.998776 |
| Jetson, Vulkan (Tegra, glibc aarch64) | −0.003061 | 0.998776 |
| NUC and mac-248, Vulkan (ANV, Kepler) | −0.000347 | 0.999222 |

All within tolerance of the truth (0, 1); none identical across the OS split.
A short run (1 seed, 100 / 200) was identical on super-io, mac-247 and the
Jetson for both tree implementations. The differences need a long chain to
flip an accept/reject decision, and then the chains decorrelate.

**Open:** the Jetson's Normal run matched super-io's to the digit despite a
different libm, while FreeBSD's did not. A libm difference is necessary for
divergence but not sufficient; why glibc-to-glibc did not flip a decision on
this model and Linux-to-FreeBSD did is unmeasured. The candidate, INFERRED, is
that msun differs from glibc on more inputs, or by more ULPs. Counting the
differing fraction and the ULP distance per function would decide it.

## How to compare runs from different machines

As two independent samplers: each posterior mean within its Monte-Carlo
standard error, `sd / sqrt(ESS)`, of the other's, not equal to the digit.
`Exmc.NUTS.Vulkan.Validator.ess/1` is the estimator the suite uses. For an exact
replay, match every field of `stats.provenance` except `host`, and use the same
`mix.lock`.

## What would change this

- A correctly rounded host libm on every platform (CORE-MATH, crlibm) would make
  the CPU arm bit-portable. Not planned: it is a native dependency on every
  host, to buy a property the statistical contract does not need.
- `config :exmc, :chain_shader_transcendentals, :polynomial` computes `log`/`exp`
  in f64 inside the chain shader, which would remove the GPU vendor term. It is
  not recommended today: REVIEW_PLAN Track 2 records it segfaulting on every
  box but one.
