# Handover — asus takes the FreeBSD/multi-GPU work

**Written:** 2026-09-12, against `624ba5d1a`, from measurements taken on the box
at 02:05–02:10 that morning. Read this before touching asus. Everything below
that says MEASURED was run; everything that says REPORTED came from another
session's notes and is a thing for you to confirm, not to cite.

asus is now the most informative host in the fleet, and it changed under us:
the driver was swapped from 470 to 580 while nobody was looking at eXMC. That
one change answers a question this repository has been carrying for a week and
invalidates a section of `docs/TWO_GATES_THAT_DO_NOT_GATE.md`.

---

## 0. Before anything: you are not alone on this box

Another session — the nx_vulkan one — is **actively working on asus right now**
and owns these directories:

    ~/nxv_branch_multigpu     branch design/two-gpus-per-host   (touched 01:34)
    ~/nxv_device_selector     clone, main @ bae9221             (touched 23:42)
    ~/nx_vulkan               clone                             (touched 19:04)
    ~/pkg580                  the 580 packages
    ~/cores/, ~/state.470, ~/1, ~/crash_dtrace, ~/erts_flags

**Do not write in any of them.** Read freely; their `docs/FLEET.md` and
`NEXT.md` are the authority on the GPU-level facts and this document cites them
rather than keeping a third copy.

**Their measurements are timing-sensitive in a way yours are not.** They are
quantifying cross-process GPU contention — a warm card losing ~52% of its
throughput while a neighbour cold-starts. An eXMC suite running on the other
card *is* that neighbour. Before starting anything that dispatches, check the
box is idle and say so in what you report:

    ssh io@192.168.0.246 'uptime; nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv'

eXMC's own work lives in `~/exmc_oss` and nowhere else.

---

## 1. The box, MEASURED 2026-09-12 02:05

| | |
|---|---|
| OS | FreeBSD 15.0-RELEASE amd64 (`releng/15.0-n280995-7aedc8de6446`) |
| driver | **580.178.04** (`hw.nvidia.version`, module built 2026-07-07) |
| GPU 0 | GeForce GTX 1660 Ti, **Turing**, 6 GB, PCI `0000:02:00.0`, uuid `cd6c2df3…` |
| GPU 1 | Quadro M4000, **discrete Maxwell**, 8 GB, PCI `0000:04:00.0`, uuid `c8727fb7…` |
| Vulkan | loader 1.4.336; both cards `apiVersion 1.4.312`, `driverVersion 580.178.4.0`; `llvmpipe` present as GPU2 |
| toolchain | elixir 1.18.4 (`/usr/local/elixir-1.18.4/bin`), erlang27, `glslangValidator`, cargo — all on PATH |
| `~/exmc_oss` | `git@192.168.0.249:/home/git/repos/exmc.git`, at **`c1f7416ae`**, clean |
| disk | `/home` 102 G free; each BEAM core is ~800 MB and five are already parked |

`~/exmc_oss` is **behind by everything since `c1f7416ae`**, including the new
leaf-diff gate. First command:

    cd ~/exmc_oss && git fetch origin && git merge --ff-only origin/gate1/reconcile-core

Verify the checkout by its **remote URL, not its directory name** — that rule
exists because another host carries a second checkout with tracked credentials.
`scripts/fleet_verify.sh` does this check for you and is the supported way to
run the suite here.

### The swap was live, and that is what makes it usable

Uptime at 02:05 was 8h53m, so the box booted ≈17:12 on the 11th. The cores
labelled `_470` are timestamped 17:45 — *after* that boot. So 470 → 580 was a
`kldunload`/`kldload`, not a reboot: **same kernel, same userland, same
binaries, one variable changed.** That is a far better experiment than a
reinstall would have been, and it is why the conclusion below is as strong as
it is. Do not reboot the box for eXMC reasons without saying so — it would end
the unbroken window.

---

## 2. The question that just got answered

`docs/TWO_GATES_THAT_DO_NOT_GATE.md` ends its Part 2 with:

> Separating them requires an INTERVENTION: nvidia-580 on a FreeBSD box, or
> nvidia-470 on a Linux one.

**That intervention has happened.** REPORTED by the nx_vulkan session in
`~/nxv_branch_multigpu/NEXT.md`:

> eXMC's crash module segfaults on both cards under 470 and runs under 580.
> Main has this at `43e7700`.

If that holds, the confound resolves cleanly, and it kills more than the OS
hypothesis:

| hypothesis | status after the swap |
|---|---|
| GLSL source text | already eliminated — super-io and asus share cache key `d915a7f8…` and disagree |
| "old hardware" | already eliminated both ways |
| **architecture as a lookup table** | **falsified, non-vacuously** — the M4000 is Maxwell and crashed under 470, while the Jetson's Maxwell Tegra runs it; and one Turing part both crashes and runs depending only on the driver |
| **OS (FreeBSD)** | **eliminated** — FreeBSD held constant across the flip |
| **driver branch 470** | **the survivor** |

The causal design is the strongest we have had: same host, same card, same
kernel, same SPIR-V (`86970c0212aadda4…`, 496516 bytes, `validate_file :ok`),
driver toggled. Two cards of different architectures flipping together on one
driver change is a dose of evidence three FreeBSD hosts sharing a driver never
was — those were never three independent observations.

**But eXMC has not verified it in eXMC's own harness**, and the claim currently
rests on one line of another project's notes. That is task 1.

---

## 3. Tasks, in order

### Task 1 — confirm the crash module runs under 580, in our harness

Staged at `~/cores/crash_module/`: `crash_repro.exs` and the
`polynomial_studentt_d3.spv` it expects. It prints the SPV path, size, and the
first 32 hex of its sha256 before dispatching, so it self-checks that it is
exercising the same module the 470 runs did.

**Run it from `~/cores`, never from a git checkout.** `kern.corefile` is
`%N.core`, so a core lands in the *current directory*; a core inside
`~/exmc_oss` is 800 MB of untracked garbage sitting in a tree you are about to
`git status`.

    cd ~/cores
    EXMC_COMPILER=vulkan mix run --no-mix-exs ~/cores/crash_module/crash_repro.exs 2>&1 | tee poly_580_gtx1660ti.log

(if `--no-mix-exs` fights the deps, run it as
`cd ~/exmc_oss && EXMC_COMPILER=vulkan mix run ~/cores/crash_module/crash_repro.exs`
**after** `cd`-ing your shell's core-dropping directory elsewhere — or accept
the core and move it out immediately.)

Expected on a working device:

    [0.2932001238570881, 0.10290175721866154, 0.20103640819021756]

which is what super-io returns. Anything else — including a *finite, plausible,
wrong* triple — is a new finding and more interesting than the crash was.

**Selecting the M4000.** The script's header suggests `NXV_DEVICE_INDEX=1`.
**That variable does not exist** in the nx_vulkan eXMC pins (`9a8427c`):
`build_ctx()` calls `min_by_key` on device *type* only, and both cards are
`DiscreteGpu`, so it silently takes whichever the loader enumerates first — the
1660 Ti, always. The nx_vulkan session has since built a real `NXV_DEVICE`
selector (`name:` / `uuid:` / `pci:` / index, refusing rather than falling back
on no match), but it is on `feat/device-selector` @ `befb91b` and **not in the
rev eXMC pins**.

Use the loader instead. MEASURED on asus, and it works:

    $ VK_LOADER_DEVICE_SELECT=0x10de:0x13f1 vulkaninfo --summary | grep deviceName
        deviceName = Quadro M4000            <- first
        deviceName = NVIDIA GeForce GTX 1660 Ti
        deviceName = llvmpipe (LLVM 19.1.7, 256 bits)

Since `min_by_key` returns the *first* minimum on a tie, reordering the
enumeration is enough to select the card with no code change:

    VK_LOADER_DEVICE_SELECT=0x10de:0x13f1 ...   # Quadro M4000
    VK_LOADER_DEVICE_SELECT=0x10de:0x2182 ...   # GTX 1660 Ti

**Confirm from the `[nx_vulkan_vulkano] device:` banner every time.** A run that
does not print the card you meant is not evidence about that card. This
mechanism is a workaround with a shelf life: when eXMC's nx_vulkan pin moves
past `befb91b`, switch to `NXV_DEVICE=name:M4000` and delete this paragraph.

### Task 2 — the first FreeBSD-580 suite row, on both cards

    cd ~/exmc_oss
    ssh ... 'bash -s' < scripts/fleet_verify.sh        # from the driver host, or run it locally

Then the M4000 arm with `VK_LOADER_DEVICE_SELECT` set. This gives the fleet its
first discrete-Maxwell row and its first FreeBSD-580 row at once. Read the
failure blocks, not the count — the script's header explains at length why, and
it has a non-vacuity gate for the case where a suite dies without a summary
line.

### Task 3 — the leaf-diff gate, which asus has never run

`test/nuts/leapfrog_leaf_diff_test.exs` landed at `acccf8348`. It is the only
harness that compares `q`, `p`, `grad` and `logp` element-wise against the
host's leapfrog along a trajectory.

    EXMC_COMPILER=vulkan mix test test/nuts/leapfrog_leaf_diff_test.exs --include requires_vulkan

Its `@tol 1.0e-13` came from four hosts — super-io, mac-247, mac-248, jetson —
and **none of them is Turing or discrete Maxwell**. asus is new data on both
counts.

If it fails, **report the number; do not loosen the bound.** The mutation work
showed the headroom is not generous and the bound is fixture-calibrated:
identical sigmas reach 1.212e-13 on trajectory geometry alone. A small overshoot
on a new *architecture* is a finding about that architecture and belongs in
NEXT.md before it belongs in a constant.

### Task 4 — does the golden split follow the driver too?

The standing anomaly: 9 of 18 shader goldens differ between Ampere (super-io)
and Kepler (the macs), with the same GLSL. Every crashing host was also a
470 host, so that split has the same confound the segfault had — and asus can
now break it the same way. If asus under 580 matches **super-io**, the golden
divergence is the driver as well, and "Ampere vs Kepler" was never the axis. If
it matches the **Keplers**, the driver explains the crash but not the numbers,
and those are two separate findings.

Note there is no committed goldens harness — it was a one-shot script, which is
exactly why this is awkward to re-run. Reconstructing it as something
`mix test`-shaped would be worth more than the answer.

---

## 4. Hazards, all of them learned the expensive way

* **Cores land in the current directory** (`kern.corefile: %N.core`), are
  ~800 MB, and `~/cores` already holds five (4.2 GB). Run crash work from
  `~/cores`, and move any stray core out of a git checkout immediately.
* **`ERL_CRASH_DUMP_SECONDS=0` is a Jetson rule, not an asus one** — it exists
  because a 2.8 GB dump on a 4 GB shared-memory board is fatal. asus has 102 G
  free. Do not transplant the rule; do not assume the Jetson's constraints here.
* **Pull, never scp.** A file copied onto a host produces a result attributable
  to no commit. `scripts/fleet_verify.sh` prints `### BEFORE` / `### HEAD` for
  this reason.
* **Never `tail`, `head` or `grep` the suite output.** ExUnit prints failure
  blocks *before* the summary, so a tail drops precisely the part naming what
  failed. This cost a re-run four times in one week.
* **Budget timeouts from the slowest host.** An outer `timeout` that fires
  mid-suite produces a log with failure blocks and no summary — which reads as
  "some failures" and means "cut off".
* `~/exmc_oss` is the only eXMC tree on this box. The remote is
  `git@192.168.0.249:/home/git/repos/exmc.git`. Check the URL, not the name.

---

## 5. What to report back

Into `NEXT.md` as a dated Status section, and into
`docs/TWO_GATES_THAT_DO_NOT_GATE.md` Part 2 as a resolution note:

1. crash module under 580, per card, with the banner line and the returned
   triple;
2. suite counts per card, with the failure names, not the counts alone;
3. leaf-diff worst relative Δ per column, per card — those become fleet rows in
   the tolerance table;
4. whichever way task 4 lands.

If task 1 confirms, `docs/TWO_GATES_THAT_DO_NOT_GATE.md` Part 2 needs rewriting
from "three hypotheses survive, mutually confounded" to "the driver branch, and
here is the intervention that showed it". Leave the superseded reasoning in
place with the correction beside it, as Part 1 does — the record of what was
believed and why is the part that is hard to reconstruct later.
