# Arms — which backend ran, how it was chosen, and what each host should report

`docs/REVIEW_PLAN.md` Track 1 item 2 and Track 3. Every fleet status section in
`NEXT.md` used to re-derive the expected suite result per host; this file is
where that lives now. **Update the table when a count moves, and name the
commit it was measured at.** A row without a commit is a guess.

## The three arms

| arm | `:exmc, :compiler` | what computes | where it is the deploy |
|---|---|---|---|
| **CPU** | `:none` | `Nx.Defn.Evaluator` on BinaryBackend | everywhere; the reference |
| **EXLA** | `:exla` | XLA; tests pin `default_client: :host` (`config/test.exs`) | Linux |
| **Vulkan** | `:vulkan` | nx_vulkan (`Nx.Vulkan.Compiler`, the chain shaders) | FreeBSD, the only GPU option there |

Unset (or `:auto`) means auto-detect, in the order **EXLA > Vulkan > Evaluator**
(`Exmc.JIT.auto_detect/0`). "Available" means the application starts, not that
its modules load, so a CUDA `exla` that cannot find its libraries falls through
to Vulkan instead of aborting the VM.

That order has one consequence to keep in mind: **on a Linux box with a
working EXLA, the Vulkan arm never runs unless you ask for it.** On super-io a
bare `mix check` is the EXLA arm (since the loader fix in
`docs/EXLA_CPU_BUILD.md`), and every Vulkan count from super-io must say
`EXMC_COMPILER=vulkan`.

An explicit request never falls back: `compiler: :exla` on a host where EXLA
cannot start raises, naming the auto-detected alternative (`Exmc.JIT`
`demand/2`). Only unset/`:auto` degrades.

## Choosing one

- **`mix` in this repo** (`mix check`, `mix test`, `mix run`):
  `EXMC_COMPILER=vulkan | exla | none | auto`. Read by `config/runtime.exs`,
  so it reaches `:dev` as well as `:test`; a typo raises there.
- **A consumer** (pathmc_ex, a release): `EXMC_COMPILER` does nothing, since
  a dependency's `runtime.exs` is never loaded. Set
  `config :exmc, :compiler, :vulkan` in the consumer's config, or
  `Application.put_env(:exmc, :compiler, :vulkan)` before the first sample.
- **Precision:** `config :exmc, :force_precision, :f32 | :f64`. Unset means
  f64 wherever the device supports it.

Plain `mix test` on a Vulkan-only host and `EXMC_COMPILER=vulkan mix test` are
now the same arm. Until 2026-09-12 they were not: `runtime.exs` turned on
`:allow_vulkan_perop_sampling` only for the explicit form. The one test that
needed it sets it itself now (`put_env_scoped/2`).

## Reading which arm ran

`test/test_helper.exs` prints two lines before ExUnit starts. Check them
before quoting a count:

```
exmc: compiler=Nx.Vulkan (configured: :vulkan) backend=Nx.Vulkan.VulkanoBackend precision=:f64 perop_fallback=false ...
exmc: excluding [:diag, :slow, :vulkan_known_failure]
```

`configured: nil` means auto-detected. On the Vulkan arm nx_vulkan adds the
device it picked:

```
[nx_vulkan_vulkano] device [0] of 2: uuid ... pci 0000:01:00.0 driver 470.256.02 api 1.2.175; selected by default (...)
```

The reason in parentheses is a fixed string in nx_vulkan (`lib.rs:759`) and
reads "first DiscreteGpu" even when an integrated GPU won. The ranking behind
it is discrete < integrated < virtual < CPU. **Check the `pci` and `driver`
fields, not the reason.** The failure to watch for is on hosts whose only
other device is llvmpipe (the NUC and super-io, both measured): if the real
driver fails to start, llvmpipe becomes the only device and is selected
silently.

**`scripts/fleet_verify.sh` pins it** (Track 5 item 2). `expected_device()`
maps `hostname -s` to a uuid prefix and a name substring, and the script
exports `NXV_DEVICE=uuid:<prefix>`. A probe in its own `mix run` then
refuses to start the suite (exit 2) in four cases: the selector matches
nothing, the device is `kind=Cpu`, the name does not match, or the host has
no row. A selector that matches nothing is an error from the NIF that lists
the devices, not a fallback. The log line to read is `### DEVICE
DEVICEINFO kind=... uuid=... pci=... driver=... selected_by=NXV_DEVICE=...
name=...`. `NXV_SKIP_DEVICE_PIN=1` runs a host unpinned on purpose, and the
log says so. A new host needs a row before its first run.

| `hostname -s` | host | uuid prefix | name |
|---|---|---|---|
| `super-io` | super-io | `f7e146ef` | RTX 3060 Ti |
| `mac` | mac-247 | `c3fcb5dd` | GT 650M |
| `free-macpro-nvidia` | mac-248 | `91f659e1` | GT 750M |
| `nuc` | NUC | `86801619` | HD Graphics 520 |
| `jake-desktop` | Jetson | `a220528a` | Tegra X1 |

## What each arm excludes

`test_helper.exs` chooses excludes from the **detected** backend, so the arm
cannot be misread from the tags:

| arm | excluded tags | excluded count at `6eafcc29c` |
|---|---|---|
| EXLA / CPU | `:diag :slow :requires_vulkan` | 67 |
| Vulkan, f64 device | `:diag :slow :vulkan_known_failure` | 6 |
| Vulkan, f32-only device | the above plus `:requires_f64` | not measured |

The total is 723 tests on every arm; only the excluded set differs.
**From the synthesis-fallback fix on (2026-09-13) the total is 728** (five tests
in `test/exmc/nuts/synthesis_fallback_test.exs`, four of them
`:requires_vulkan`): MEASURED on super-io, Vulkan 728 / 0 with 6 excluded,
EXLA 728 / 0 with 71 excluded. Fleet rows below predate it; expect 728 with the
same failures on the next run. **From the reproducibility contract on, 731**
(`test/reproducibility_contract_test.exs`, one of its three tests
`:requires_vulkan`): MEASURED on super-io, Vulkan 731 / 0 with 6 excluded, EXLA
731 / 0 with 72 excluded.

## Fleet, measured

Suite results at **exmc `6eafcc29c`, nx_vulkan lock `8116a19`**, 2026-09-12/13.
Vulkan rows are `scripts/fleet_verify.sh` (which sets `EXMC_COMPILER=vulkan`
and starts epmd); super-io rows are `mix check` by hand. Time is ExUnit's
"Finished in".

| host | OS | device (driver) | arm | result | time | failures |
|---|---|---|---|---|---|---|
| super-io | Linux x86_64 | RTX 3060 Ti (580.178.04) | Vulkan f64 | **723 / 0** | 950 s | — |
| super-io | Linux x86_64 | CUDA `exla` build, host client | EXLA | 723 / 2 → **0** | 443 s | 2× `DistributedTest`, `:nodistribution` (epmd was not running); the file re-run with epmd: 5 / 0 |
| mac-248 | FreeBSD 15.0 | GT 750M (470.256.02) | Vulkan f64 | **723 / 0** | 795 s | — |
| mac-247 | FreeBSD 15.0 | GT 650M (470.256.02) | Vulkan f64 | 723 / 1 | 1240 s | `PokerTest` "parameter recovery", 300 s timeout |
| NUC | FreeBSD 15.0 | HD 520, Mesa ANV 26.1.3 | Vulkan f64 | 723 / 1 | 993 s | `ValidatorTest` "iid samples report ESS ~ n": a flaky test, fixed in `d68b86ccd` |
| Jetson | Linux aarch64 | Tegra X1 (32.7.6) | Vulkan f64 | 723 / 2 | 3622 s | `PokerTest` "parameter recovery", 300 s timeout; `IntegrationTest` "large model: 5-parameter hierarchical", 120 s timeout |

**Expected result per host**, which is what a new run is checked against:

| host | expected | why not 0 |
|---|---|---|
| super-io, both arms | 723 / 0 | — (the EXLA arm needs epmd running, as every arm does) |
| mac-248 | 723 / 0 | — |
| mac-247 | 723 / 1 | `PokerTest` times out. The same failure, alone, at `144d441db`, `d410b183a` and `6eafcc29c` |
| NUC | 723 / 0 | MEASURED at `d4eeba367` (983 s), after the `d68b86ccd` fix |
| Jetson | 723 / 2 | `PokerTest` and `IntegrationTest` time out (at `144d441db`, `d410b183a` and `6eafcc29c`) |

A timeout on a slow host is recorded here as expected so that a *new* failure
stands out. It is not a pass.

**Not measured:** the CPU build of `exla` on super-io (Track 3 item 1's second
row). Building it replaces the CUDA library both arms share, so it needs its
own window. The CPU arm (`EXMC_COMPILER=none`) as a full suite on any host.

### NUTS end-to-end race, NUC vs mac-248

`bench/nuts_truth.exs` at `7aae323a6` (lock `8116a19`), seeds 1–6 pooled,
500 warmup / 2000 samples, `MIX_ENV=test`, device pinned by uuid, both boxes
otherwise idle except mac-248's standing `zedweb`/`MeshWorker` load. Wall
time per arm includes `mix run` startup and, for `vulkan`, first-client
Vulkan startup. All arms were within tolerance of analytic truth.

| host | CPU | `vulkan` | `none` (host tree only) | none ÷ vulkan |
|---|---|---|---|---|
| NUC, HD 520 (ANV) | i3-6100U, 4 threads | 43 s | 402 s | 9.3× |
| mac-248, GT 750M | i7-4870HQ, 8 threads | 30 s | 318 s | 10.6× |

mac-248 is 1.43× faster on `vulkan` and 1.26× on `none`, so most of the gap is
the CPU. What is left for the GPU is roughly 1.13×, and startup is inside it.

**Agreement, and what it is not.** Both hosts' `vulkan` pooled moments are
identical to every printed digit (Normal mean −0.000347, var 0.999222, ess
4809), and so are their `none` moments. That is NOT a GPU grouping. Measured
the same day at the same settings (`667573f16`):

| run | Normal mean | HalfNormal mean | Exponential mean |
|---|---|---|---|
| super-io `none` | −0.016021 | 0.792226 | 0.495538 |
| NUC = mac-248 `none` | −0.015276 | 0.795172 | 0.496193 |
| super-io `vulkan` (Ampere) | −0.003061 | 0.795981 | 0.491954 |
| Jetson `vulkan` (Tegra, glibc aarch64) | −0.003061 | 0.790088 | 0.496679 |
| NUC = mac-248 `vulkan` (ANV, Kepler) | −0.000347 | 0.784924 | 0.489146 |

The host-only arm already differs between super-io and the FreeBSD boxes, so
part of the split is on the host. On Normal, whose chain shader takes no
`log`/`exp`, the grouping is by OS on both arms: Linux {super-io, Jetson}
vs FreeBSD {NUC, mac-248}. A short run (seed 1, 100/200) is identical on
super-io, mac-247 and the Jetson for both tree implementations, so the
difference needs a long chain to surface.

**The host libm is not bit-portable, even glibc to glibc.** BEAM `:math` over
1M seeded inputs, sha256 of the raw f64s: inputs and `sqrt` identical on
super-io (glibc 2.39 x86_64), mac-247 (FreeBSD msun) and the Jetson (glibc
2.27 aarch64); `log`, `exp`, `log(1+x)` and `pow` differ on all three. Yet the
Jetson's Normal `vulkan` run matched super-io's exactly, so a libm difference
has to flip an accept/reject decision before it reaches the moments. Why that
happened Linux-to-FreeBSD and not glibc-to-glibc is **open**. INFERRED
candidate: msun differs from glibc on more inputs or by more ULPs.

**Consequence:** exact posterior reproducibility across hosts is not a
property exmc has, on either arm. Tolerance to analytic truth is the gate.
Moments quoted across hosts are comparable statistically, not bitwise.

**Not a general bit-identity claim across vendors.** nx_vulkan's
`scripts/arch_float_divergence.exs` at `eddb973` measured the chain shaders'
default f32-cast transcendentals (`log`, `log(1+z²)`, `exp`) on ANV: more
accurate than NVIDIA (log relative error 1.0e-7 vs 1.5e-6), and not
bit-identical to it. Models whose synthesised body takes `log` or `exp` of a
sampled value should not be expected to reproduce NVIDIA draws on the NUC;
`:chain_shader_transcendentals, :polynomial` would. The same run found ANV
f64 division not always correctly rounded (`x/3` one f64 step off, where
NVIDIA is exact; `1.0/x` identical). exmc never emits GLSL `pow`, so ANV's
negative-base `pow` (nx_vulkan's 4 NUC failures) cannot reach a chain shader.

### Host notes

- **epmd.** Two `DistributedTest` cases fail with `:nodistribution` whenever
  epmd is not running, on every arm. `fleet_verify.sh` starts it. A hand run
  from a non-interactive shell (an agent, `nohup`) does not, which is what
  happened to the EXLA row above.
- **super-io, EXLA.** Loads through the system loader, with no
  `LD_LIBRARY_PATH` (`docs/EXLA_CPU_BUILD.md`). `DistributedTest`'s `:peer`
  nodes start `:exla` as well; with the loader fix they load it and log CUDA
  out-of-memory lines from preallocation, which are harmless.
- **NUC.** FreeBSD 15.0, i3-6100U, 8 GB, joined 2026-09-13 and set up the way
  mac-247 is (Elixir 1.18.4 on the `login.conf` PATH, `erlang-runtime27`,
  `rust`, `glslang`). Where mac-247 has `nvidia-driver-470`, the NUC has
  `drm-kmod` with `i915kms` in `kld_list` and `mesa-dri`. The HD 520 reports
  `shaderFloat64 = true`, so it runs the f64 arm. Its second Vulkan device is
  llvmpipe, so check that the banner reads `pci 0000:00:02.0` with a Mesa
  driver. The ZFS pool is 11 GB; about 4.4 GB was free after the first build.
- **mac-247, mac-248** share their boxes with long-running `zedweb` releases
  and other jobs; timings there are not benchmarks.
- **Jetson.** One Vulkan device, no llvmpipe (`device [0] of 1`, `pci none`).
  Its native NIF build takes ~12 min incremental, ~47 clean; a super-io
  cross-build path for exmc's lock is agreed with nx_vulkan and not yet built
  (`NEXT.md`, 2026-09-13).
- **Budget the caller's timeout from the slowest host.** The Jetson took
  60 min of suite time at `6eafcc29c`, plus its build; see
  `fleet_verify.sh`'s header.
