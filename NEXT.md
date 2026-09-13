# NEXT — eXMC (open source)

**Written:** 2026-08-16, against `main` @ `6d6ae4f` (0.3.1, the P0 correctness
backport, merged).
**Read `MISSION.md` first** — this file assumes it and does not repeat it. This
one is only *what to do next and in what order*, with the state as it actually
stands rather than as the mission planned it.

---

## Status — 2026-09-13 (evening): Rustler 0.38 on main; two exmc defects fixed

**On main**
- `134f9d3aa`: automatic NCP on vector RVs rebuilt traces with a broadcast
  error. There is now one `NonCenteredParameterization.reconstruct/2` in place
  of two private copies, in Sampler and MCLMC (which MAMS shares). Also
  `shape: {1}` vector RVs now synthesise: the forward `param_vec` and the
  one-column gradient contraction.
- `85f289306`: Rustler 0.38 in lockstep with nx_vulkan `ca1e0c8`. The details
  and every host's count are in the commit message and `docs/ARMS.md`. What
  to remember:
  - the NIFs load as `priv/native/<crate>.so`, with no `lib` prefix;
    `fleet_verify.sh` deletes lib-prefixed NIFs, markers and `.so.prev`
    backups before hashing;
  - rustc >= 1.91 (`native/exmc_tree/rust-toolchain.toml`; super-io's rustup
    default is still 1.90);
  - the Jetson's prebuilt NIF comes from nx_vulkan's `deploy_jetson_nif.sh` at
    `ca1e0c8` or later, which defaults to the `nxv-jetson-cross:1.91.0` image.
- The bump was fleet-verified on a temporary branch through `FLEET_BRANCH`,
  since deleted. mac-247, mac-248, the NUC and the Jetson are still checked out
  on it locally; their next `fleet_verify.sh` moves them to `main` by itself.
- **Landed before the Jetson's suite finished**, on the operator's call. It had
  already used the prebuilt (marker `ca1e0c8`, hash_match yes, NIF `8647de56`,
  the same bytes the nx_vulkan session built) and passed the smoke gate.
  **Its count: 740 / 3**: the expected PokerTest and IntegrationTest timeouts,
  plus `ReproducibilityContractTest` "CPU arm", which timed out at its 300 s
  limit before reaching the comparison (not a bits mismatch). The contract
  tests' timeouts are now 1,200 s; the Jetson's expected count stays 740 / 2.
  NIF hashes read by exmc's gates: Keplers `5cb1e506`, NUC `11fdca96`, Jetson
  `8647de56`. They differ from the nx_vulkan session's `671acbd8` / `6f52785b`
  for native builds because a native build embeds its build path (deps/ against
  ~/nx_vulkan); the cross-built Jetson NIF is byte-identical.

**Not on main yet**
- **The shader-size fix**, worktree `../exmc-shader`, branch `wip/shader-size`
  (based on the Rustler branch; rebase onto main).
  - The cause: the synthesised chain shader grew as d^2. Each coordinate's
    gradient branch repeated the full residual, and the gradient was emitted
    twice. SPIR-V was 9.0 MB at d=128. It is why the NUC was OOM-killed at d=128
    in the width race: the driver's pipeline compile, on 8 GB shared with the GPU.
  - The fix: the gradient becomes one function `exmc_grad(tid)` with loops
    fused across coordinates, so the existing CSE hoists the shared residual.
  - MEASURED: SPIR-V 204 KB at d=128 (44x smaller); a d=128 10+10 sample
    takes 4.9 s against 72 s; peak RSS 685 MB against 1.44 GB.
  - Tests: shader correctness files on the Vulkan arm 243 / 1, the 1 a batched
    test's normalisation, since fixed; a new `shader_size_test.exs` fails on the
    old emitter (ratio 10.77).
  - Still to do: full suites on both arms, then the fleet including the NUC and
    the Jetson.
  - Note: the text CSE now takes 7.4 s at d=128, which is worth optimising.
- **The PyMC race** (`docs/PYMC_RACE_PLAN.md`), in order:
  1. reference posteriors: `bench/pymc_race/reference_{pymc.py,exmc.exs}`,
     not committed; the PyMC run was started on super-io, the exmc run was not;
  2. the harness and a pilot;
  3. super-io, then asus (a window for the operator to schedule), then the NUC.
  PyMC 6.3.2 picks **nutpie** by default when it is installed, so the "PyMC
  default NUTS" arm must pass `nuts_sampler="pymc"`.

## Branch — 2026-09-13: `gate1/reconcile-core` is retired; work on `main`

Merged into `main` by fast-forward at `325c48e56` (operator's decision). From
here: commit on `main`, push `main` to `origin`. `origin/gate1/reconcile-core`
is left in place, frozen, not deleted. `scripts/fleet_verify.sh` follows
`origin/main` and moves any host still on the old branch onto `main` itself,
so no host needs a manual checkout. Every "push gate1/reconcile-core and main
together" instruction below this section is superseded. `upstream` (GitHub)
is still not pushed; that remains a release, and the operator's.

## Status — 2026-09-13, both arms on super-io, the NUC joins, the whole fleet at one commit

Supersedes the section below's items 1 and 2. Its item 3 (the "(last)"
section's items 3–7) stands and follows these. **`docs/ARMS.md` is new and is
now where per-host expected results live**; this section does not repeat its
tables.

**Done and MEASURED** (exmc `6eafcc29c`, nx_vulkan lock `8116a19`)

- **Track 3 item 1, the EXLA arm at HEAD, CUDA build.** `EXMC_COMPILER=exla
  mix check` on super-io with no `LD_LIBRARY_PATH` set: 723 tests, 2 failures,
  443 s. Both failures were `DistributedTest` `:nodistribution`, because the
  shell was `nohup`-style and epmd was not running. With epmd started, that
  file is 5 / 0, and its `:peer` nodes load EXLA through the loader fix. **No
  real failures on the EXLA arm**, first measured since `a178a0833` (652 tests).
- **The Vulkan arm after the loader change**, as the previous section asked:
  `EXMC_COMPILER=vulkan mix check`, 723 / 0, 950 s, RTX 3060 Ti.
- **Whole fleet via `scripts/fleet_verify.sh`**, all at one commit, for the
  first time since `d410b183a`: mac-248 723 / 0; mac-247 723 / 1 (`PokerTest`
  timeout, as before); Jetson 723 / 2 (`PokerTest` + `IntegrationTest`
  timeouts, as before, 60 min); NUC 723 / 1 (below). The script was `scp`ed to
  `~/fleet_verify.sh` and run from there under `nohup`, not from the checkout.
  It fast-forwards the checkout, and bash reads its script incrementally, so
  running the in-tree copy across a merge that changes it is unsafe.
- **The NUC joined the fleet** (FreeBSD 15.0, i3-6100U, HD 520). It was set
  up the way mac-247 is. The differences are `drm-kmod` + `i915kms` +
  `mesa-dri` for the nvidia driver, and ANV reports `shaderFloat64 = true`, so
  it runs f64. The non-interactive PATH comes from `/etc/login.conf`'s default
  class, as on mac-247 and mac-248 (original saved as `login.conf.orig`).
  Setup details are in `docs/ARMS.md` host notes. mac-248 needed only
  `pkgconf` and the `.shrc` line.
- **A flaky test the NUC found, fixed and pushed:** `d68b86ccd`. `ValidatorTest`
  "iid samples report ESS ~ n" used unseeded `:rand` with a ±0.15 bound. It
  failed on 3.55% of 2000 seeds, and the NUC's seed reproduced on super-io to
  the last digit. It is now seeded, with a measured bound of 0.70 ≤ ESS/n ≤ 1.0.

**Observed, not explained:** the Jetson log has one `SuspectTracker`
"emergency brake — 5 timeouts in 60000 ms across 5 shader(s)" at 23:36,
mid-suite. Earlier status sections record only counts, so whether this is new
is unknown. Check the next Jetson log for it before calling it noise.

**Agreed with nx_vulkan (asus session), not built:** a super-io cross-build of
the Jetson NIF for **exmc's lock**. `deploy_jetson_nif.sh` gains `REF=<sha>`
(sources via `git archive`) and `DEST_DIR` (refuses unless the destination
checkout is at the built sha), and writes the same three-line `.provenance`.
Their side waits on their user. Ours is exmc's `fleet_verify.sh` mirroring
their prebuilt logic: use the prebuilt only when the marker's sha equals the
lock sha AND its hash equals the file on disk, with the same mode-switch wipe
of `_build/test/lib/nx_vulkan/{ebin,.mix}`. They TESTED that `mix deps.get`
on a lock bump leaves `priv/native` and the marker in place, so the sha check
is what forces the rebuild.

**Next, in order**

1. **Track 5 item 2, now with a concrete hazard:** pin the Vulkan device per
   host in `fleet_verify.sh`. The NUC and super-io both have llvmpipe as
   device [1]. If the real driver fails, llvmpipe is selected silently, and
   nx_vulkan's banner reason reads "first DiscreteGpu" whatever wins (a label
   bug on their side, queued). Until then, check `pci`/`driver` in every
   banner. **Done later the same day:** `fleet_verify.sh` pins by uuid per
   `hostname -s` and refuses before the suite on an unresolved selector, a
   `kind=Cpu` device, a name mismatch or an unknown host. Six cases were
   exercised on super-io (pass, unpinned, and four refusals incl. llvmpipe); `docs/ARMS.md` has the table.
   Each host's row is proved on its next real run.
2. ~~Rerun the NUC~~ **DONE:** 723 / 0 at `d4eeba367`, 983 s.
   **Also found the same day, by racing the NUC against mac-248** (`docs/ARMS.md`
   has the table): `bench/nuts_truth.exs COMPILER=vulkan` died after 1 s on every
   FreeBSD host, with FunctionClauseError in `Dispatch.do_chain/8`. The cause was
   exmc not declaring `:crypto`. CustomSynth hashes every shader with it; on
   Linux `xla` brings it in; FreeBSD drops exla, and `mix test` loads crypto
   anyway, so the suite never saw it. Fixed in `7aae323a6` and verified on
   mac-247 and in the race. **Still open:** a synthesis failure under Vulkan at
   f64 falls back to a family meta that no dispatch clause accepts. It should
   reach the Plan-B' refusal with the real reason; `try_synthesise` swallowing
   the exception is what made this cost an investigation. **DONE the same
   day:** under Vulkan, single-RV detect_meta synthesises or refuses (never a
   family meta), a raise is logged and becomes `{:unsupported,
   :synthesis_raised}`, and the Plan-B' guard's message says so. Tests in
   `synthesis_fallback_test.exs` fail 3 of 5 against the old code. super-io
   728 / 0 on both arms.
3. **The exmc half of the Jetson prebuilt**, once nx_vulkan says the
   interface has landed.
4. **Track 3 leftovers:** the CPU `exla` build row, which replaces the CUDA
   library both arms share, so it needs its own window; `test/run_all.sh`
   (fold or delete); the present-but-unloadable EXLA test (item 3).
5. Then the "(last)" section's items 3–7: seed 46 and the leaf-diff outlier,
   Track 5 item 2 (see 1), Track 4, Track 6 leftovers, phd.git.

**Do not:** start a hand-run suite from a non-interactive shell without
`epmd -daemon` first.

---

## Status — 2026-09-12 (after the lock bump), what to do next after the reboot

Supersedes the "(last)" section's items 1 and 2 below; its items 3–7 stand
unchanged and follow these.

**Done and MEASURED**

- **nx_vulkan lock `16d13f3` → `8116a19`** (server `main`, unchanged since the
  last section). Not docs only, as that section expected: the range carries the
  negative-base `pow` shader fix (five `elementwise_binary_*` shaders) and
  increment 3's `vulkano_backend.ex` device routing. `EXMC_COMPILER=vulkan mix
  check`, super-io: **723 tests, 0 failures**, 6 excluded, 2 skipped, exit 0,
  18m52s. No loader segfault. Banner: RTX 3060 Ti, "device [0] of 2".
- **Why EXLA kept being forgotten, answered.** Nothing on this host ever set
  `LD_LIBRARY_PATH`: no hit in `~/.bashrc`, `~/.profile`, `/etc/environment`,
  `/etc/profile.d`, `environment.d`. "Interactive shells inherit it" was never
  true; only a terminal where someone had exported it did. And
  `test/distributed_test.exs:227` starts `:exla` on `:peer` nodes, so even a
  Vulkan-arm suite logs "Failed to load NIF library" from them. Harmless to
  the count, since the coordinator retries, but it is the same miss.
- **The fix that sticks is the system loader, not an export.**
  `docs/EXLA_CPU_BUILD.md` now opens with the recipe: an
  `/etc/ld.so.conf.d/zz-nvidia-pip-wheels.conf` naming the python3.12
  `nvidia/nvshmem/lib` and `nvidia/cuda_nvrtc/lib`, then `ldconfig`. `zz-`
  keeps `/usr/local/cuda`'s `libnvrtc.so.12` (12.6) first for the rest of the
  box. Both orders were measured: `client: :cuda` gives the identical f64
  result either way. `Exmc.JIT`'s "not usable" error now points at the doc.
- **`bench/nuts_truth.exs` on both arms at `8116a19`**, super-io, seeds 1–6
  pooled, warmup/samples 500/2000, `use_nif=true full_tree_nif=false`. This is
  the first recorded run for any pin since `9a8427c`, so it is the baseline:

  | model | stat | truth | `none` | err | `vulkan` | err |
  |---|---|---|---|---|---|---|
  | Normal(0,1) | mean | 0 | −0.016021 | −0.0160 | −0.003061 | −0.0031 |
  | | var | 1 | 1.011326 | 1.13% (ess 4615) | 0.998776 | −0.12% (ess 4672) |
  | HalfNormal(1) | mean | 0.797885 | 0.792226 | −0.71% | 0.795981 | −0.24% |
  | | var | 0.363380 | 0.349842 | −3.73% (ess 4516) | 0.351352 | −3.31% (ess 4392) |
  | Exponential(2) | mean | 0.5 | 0.495538 | −0.89% | 0.491954 | −1.61% |
  | | var | 0.25 | 0.256864 | 2.75% (ess 4707) | 0.254845 | 1.94% (ess 4547) |

  Both arms: "all 3 models within tolerance of analytic truth", exit 0.
  Times: `none` 5m42s, `vulkan` 1m06s.
- **A bump trap the bench found, not the suite.** The first `COMPILER=vulkan`
  run died in 2 s: `{:bad_lib, 'Function not found
  Nx.Vulkan.NativeV:buf_alloc_zeroed_on/2'}`. `mix check` rebuilds nx_vulkan
  for `_build/test` only. The Rust `.so` lives in `deps/nx_vulkan/priv`,
  which every environment shares, so `_build/dev` ran a pre-bump
  `NativeV.beam` against the new library. `mix run --no-deps-check` is what
  lets that through. **After any nx_vulkan lock move, run `mix deps.compile
  nx_vulkan` (in `:dev`) before any bench.** §5's bench commands all use
  `--no-deps-check`.

**Next, in order**

1. ~~Install the loader entry~~ **DONE 20:00, and it took two steps, not
   one.** The `ld.so.conf.d` entry alone left `nvshmem_bootstrap_uid.so.3`
   and `nvshmem_transport_ibrc.so.3` unresolved, because `ldconfig` skips
   names without a `lib` prefix. Two symlinks in `/usr/lib/x86_64-linux-gnu`
   finished it (`docs/EXLA_CPU_BUILD.md` has both steps and why;
   `bench_results/EXLA_LOADER_FIX.md` records exactly what changed, and how to undo it). Verified
   from a shell with no `LD_LIBRARY_PATH`: `ldd` reports nothing missing,
   `client: :cuda` computes on the 3060 Ti, and `Exmc.JIT.describe/0` says
   `compiler=EXLA (configured: nil)`. **A bare `mix check` here is now the
   EXLA arm.**
2. **Track 3 — the EXLA arm at HEAD**, unblocked by 1:
   `EXMC_COMPILER=exla mix check`, no exports. Last measured at `a178a0833`
   (652 tests). Note that **a bare `mix check` on super-io becomes the EXLA
   arm** once EXLA loads, because auto-detect prefers it. Every Vulkan count
   from now on must name `EXMC_COMPILER=vulkan`. Rerun the Vulkan arm after
   the loader change too: its peer nodes will now actually start EXLA.
   Both counts go into a new `docs/ARMS.md` (Track 1 item 2).
3. Then the "(last)" section's items 3–7: seed 46 and the leaf-diff outlier,
   Track 5 item 2, Track 4, Track 6 leftovers, phd.git.

Pushed: `gate1/reconcile-core` and `main` together to `origin`, as before.
Not to `upstream`.

---

## Status — 2026-09-12 (last), what to do next after the restart

Supersedes the "evening" handoff below; everything there that is not repeated
here is done. Read `docs/REVIEW_PLAN.md` for the why.

**Where the trees stand**

| tree | state | pushed |
|---|---|---|
| exmc (this) | `mix check` **723 / 0** on super-io at `f79fdc27b` (auto-detected Vulkan, nx_vulkan lock `16d13f3`) — first fully green run here | yes, `gate1/reconcile-core` and `main` together |
| nx_vulkan | server `main` at `8116a19` ("increment 3 is landed; and what to check when the box comes back" — asus may be down, read that commit first): the divergence script, `docs/MULTI_DEVICE.md`, increments 1–3, the loader fix; their NEXT.md §1c.2 records the KS resolution | theirs; local checkout behind — `git fetch` first |
| pathmc_ex | `60d6ed0` (docs) pushed; the working tree is **owned by session `pathmc-ex-17`** — its uncommitted work includes the pin move to exmc `5b660310a` on the LAN address, `check.full`, an arm field on the posterior, and this session's guide/02 fix. They were asked to commit it as one piece. Do not stage anything there. | partly |
| phd.git | rewritten onto our root, `02469a6` on `o` with its own `exmc/NEXT.md` | `origin` (192.168.0.33) still has the old history — force-push when reachable |

**Next, in order**

1. **Bump this tree's nx_vulkan lock past `16d13f3`** to their current `main`
   (`mix deps.update nx_vulkan`), run `EXMC_COMPILER=vulkan mix check` (16 min
   here) and `bench/nuts_truth.exs` on both arms — the bump checklist their
   NEXT.md asks for and no pin since `9a8427c` has had. Everything past
   `16d13f3` is docs, tests and per-device capability; expect 723 / 0.
2. **Track 3 — the EXLA arm at HEAD.** Unmeasured since 2026-08
   (`a178a0833`, 652 tests). First deliverable: the `LD_LIBRARY_PATH` recipe
   from the agent memory `exla-ld-library-path-super-io` into
   `docs/EXLA_CPU_BUILD.md`; then `EXMC_COMPILER=exla mix check`; the count
   into a new `docs/ARMS.md` (Track 1 item 2) with the Vulkan rows.
3. **The two things the KS hunt left open** (Track 2 item 3, both this repo's):
   - seed 46 in `bench/validator_ks_seeds.exs`: d = 0.30 even ESS-sized, a
     4× ESS gap between arms (host 308, GPU 80) on one seed — look at the
     GPU chain's trace, step size and divergences before calling it MCMC luck;
   - the leaf-diff outlier: super-io is *closer* to the host than mac-247,
     mac-248 and the Jetson, which agree to the digit. nx_vulkan measured f64
     division exactly rounded on Turing and discrete Maxwell, so the grouping
     is Kepler + Tegra or it is `inversesqrt` (`:rsqrt`, glsl.ex:602). A
     three-constant `inversesqrt` inside a synthesised body, dispatched here
     and on a Kepler, is the test; nx_vulkan cannot run it for us because
     `Nx.rsqrt` at f64 falls back to the host there.
4. **Track 5 item 2** — `Exmc.NUTS.Vulkan.Dispatch` threads a device slot to
   `ChainTrace.dispatch_f64/7`, after reading nx_vulkan `docs/MULTI_DEVICE.md`
   (13 s first-client startup; `:cross_device` raises on Nx-level ops, returns
   on the chain path).
5. **Track 4** — `docs/PUBLIC_API.md`, a reason on every bare `:unsupported`
   (seven sites), the `detect_meta/2` spec. pathmc-ex-17 is building the arm
   field on their side; Track 1 item 4 (the arm in this sampler's stats map,
   from `Exmc.JIT.describe/0`) should land so both projects report one string.
6. **Track 6 leftovers** — 16 compile warnings before `--warnings-as-errors`
   joins `mix check`; `:requires_f64` excluded-never-tagged; the seven
   tagged-never-read tags.
7. **phd.git** when 192.168.0.33 answers: `git push origin --force` the 17
   branches and the tag, `git gc --prune=now` on both bare repos; then its
   `exmc/NEXT.md` items 1–3 (lock agreement, the app's own suite, the
   per-device application supervisor).

**Do not**: push exmc to `upstream` (GitHub) — a release, the operator's;
move the nx_vulkan lock without the suite; stage anything in pathmc_ex while
`pathmc-ex-17` owns the tree; fetch phd's `origin` into a rewritten branch.

---

## Status — 2026-09-12 (night), the Cauchy KS failure was the test's own statistic

The one failure this suite has carried on super-io since 2026-09-07 —
`ValidatorTest` "Cauchy(0, 1) — synthesized leapfrog_chain_synth", host-specific,
passing on both Keplers and the Jetson — is closed, and it was never the
shader.

**How it was found.** Three hypotheses, in order, each killed by measurement
with the nx_vulkan session (their `scripts/arch_float_divergence.exs`, run on
asus and here): per-architecture f32 transcendentals — **dead**, sha256
digests of `log`, `exp` and the Cauchy kernel identical on Ampere, Turing and
Maxwell at f32 and f64; FMA contraction in the driver — **dead**, the
fused-shape dot reads "NOT contracted" on all three; then the test itself.
`Validator.check_ks/2` sized its critical value with raw lengths (n = m = 800)
while D92 had moved every other check onto ESS. The inputs are NUTS chains on a
1-D Cauchy with ESS ≈ 65–545, so the KS was anti-conservative by ~sqrt(8).

**MEASURED**, `bench/validator_ks_seeds.exs`, super-io, both arms, eight seeds:

| seed | d | crit raw n | verdict | ESS host / gpu | crit ESS | verdict |
|---|---|---|---|---|---|---|
| 42 | 0.1000 | 0.0975 | **FAIL** | 122 / 95 | 0.267 | pass |
| 43 | 0.0875 | 0.0975 | pass | 160 / 365 | 0.185 | pass |
| 44 | 0.0475 | 0.0975 | pass | 96 / 144 | 0.257 | pass |
| 45 | 0.0687 | 0.0975 | pass | 173 / 145 | 0.220 | pass |
| 46 | 0.3013 | 0.0975 | **FAIL** | 308 / 80 | 0.245 | **FAIL** |
| 47 | 0.0550 | 0.0975 | pass | 289 / 150 | 0.196 | pass |
| 48 | 0.0700 | 0.0975 | pass | 83 / 65 | 0.324 | pass |
| 49 | 0.1175 | 0.0975 | **FAIL** | 544 / 337 | 0.135 | pass |

Seed 42 is the suite's failure to four digits. 3 of 8 rejections at a nominal
α = 0.001 is the statistic. Why host-specific: with a fixed seed the two arms'
trajectories decorrelate chaotically after a few hundred steps, and where the
seed lands relative to a too-tight bound is decided by 1e-15 bit differences
between GPUs — the leaf-diff table's, not the transcendentals'.

**Fixed:** `check_ks/2` sizes its critical value and p-value by `ess/1` (D92's
rule, applied to the check it had missed); the error map carries `n_eff` and
`m_eff`; the known-case unit test gets i.i.d.-shaped fixtures (its ramps had
ESS ≈ 2.5, which under the rule correctly widened the bound past 1.0).
Validator file: 28 tests, 0 failures on the Vulkan arm. Full `mix check` at
`f79fdc27b`, auto-detected Vulkan, nx_vulkan `16d13f3`: **723 tests, 0
failures**, exit 0, 820 s — the first fully green run of this suite on
super-io. MEASURED.

**Two things left open by this.**
- **Seed 46** rejects even ESS-sized (d = 0.30) with a 4× ESS gap between
  arms on the same seed (host 308, GPU 80). A heavy-tail MCMC story — a stuck
  or wandering chain on one arm — not a shader one, and it deserves its own
  look rather than being filed with the seven.
- **The leaf-diff outlier is still unexplained**, and it is the interesting
  one: super-io is *closer* to the host reference (q 1.92e-15) than mac-247,
  mac-248 and the Jetson (4.77e-15, identical to each other). nx_vulkan's rows
  now show f64 division exactly rounded on Turing and discrete Maxwell, so
  "older parts" is not the grouping — it is Kepler + Tegra specifically, or it
  is `inversesqrt`, which the emitter calls (`:rsqrt`, glsl.ex:602) and which
  nx_vulkan cannot probe for us because `Nx.rsqrt` at f64 host-falls-back
  there. A three-constant `inversesqrt` inside a synthesised body is the one
  remaining op-class test, and it belongs in this repo.

---

## Status — 2026-09-12 (evening), handoff before a restart: what to do next, in order

Everything below the line is committed and pushed unless it says otherwise.
Read `docs/REVIEW_PLAN.md` for the why; this is only the what.

**Where the four trees stand**

| tree | HEAD | pushed | notes |
|---|---|---|---|
| exmc (this) | `b53be8c74` + this section | yes, `gate1/reconcile-core` and `main` together | nx_vulkan lock at `16d13f3`; `mix check` green (723 tests, the Cauchy KS failure only) |
| nx_vulkan | server `main` past `3c5ae2d` (`docs/MULTI_DEVICE.md`); local checkout behind | theirs | the other session owns increment 3; `git fetch` before reading anything there |
| pathmc_ex | `60d6ed0` local, **unpushed** | no | pin bump to exmc `dc671b62c` + `guide/02` fix **uncommitted**; another session's edits in `lib/path_mc/{do,compile/exmc,backend/conjugate}.ex`, three tests, `TODO.md`, one `NEXT.md` hunk — stage by hunk, never by file |
| phd.git (`../../pymc`) | `a561f16`, rewritten onto our own root | yes to `o` (localhost); **not** to `origin` (192.168.0.33, no route today) | pre-rewrite bundle: `/home/io/backups/phd_before_rewrite_2026-09-12.bundle` |

**Next, in order**

1. **pathmc_ex, first, because it is the only tree with uncommitted verified
   work.** Coordinate with whichever session is editing its `lib/` (ListAgents;
   the name changes per restart). Then: re-bump its exmc pin from `dc671b62c`
   to this HEAD, `MIX_INSTALL_FORCE=1 mix guide` and `mix check` (the notebook
   is what breaks on a pin bump, not the gate — 2026-09-12 proved it), commit
   pin + lock + `guide/02` together, push `60d6ed0` and that.
2. **exmc `docs/REVIEW_PLAN.md` Track 3** — the EXLA arm at HEAD, unmeasured
   since 2026-08 (`a178a0833`, 652 tests). `LD_LIBRARY_PATH` for the CUDA
   build is in the agent memory `exla-ld-library-path-super-io`, nowhere in a
   repo; the first deliverable is putting it in `docs/EXLA_CPU_BUILD.md`.
   Then `EXMC_COMPILER=exla mix check`, and the count goes into a new
   `docs/ARMS.md` (Track 1 item 2) beside the Vulkan rows above.
3. **Track 5 item 2** — `Exmc.NUTS.Vulkan.Dispatch` threads a device slot to
   `ChainTrace.dispatch_f64/7`. One line, but read nx_vulkan
   `docs/MULTI_DEVICE.md` first; the startup budget is 13 s for the first
   Vulkan client in a BEAM, and `:cross_device` raises on Nx-level ops.
4. **Track 2** (cross-arm parity test) and **Track 4** (`docs/PUBLIC_API.md`,
   a reason on every bare `:unsupported`, the `detect_meta/2` spec).
5. **phd.git** when 192.168.0.33 is reachable: `git push origin --force` for
   the 17 branches and the tag (the list is the server's own `ls-remote`),
   then a server-side `git gc` on both to drop the old objects. The audit
   decisions (mesh to core?, `smc/` → `smc_ex`, the stranded core tests and
   benches, the `LICENSE` file that still says Apache for the trader) are in
   the 2026-09-12 handoff message and still open.
6. **exmc Track 6 leftovers**: 16 compile warnings before
   `--warnings-as-errors` can join `mix check`; the dead tags `:requires_f64`
   and the seven tagged-never-read ones.

**Do not**: push exmc to `upstream` (GitHub) — that is a release and the
operator's; move the nx_vulkan lock past a rev this suite has not run on;
reset the server's `main` without deciding whether it should lag
`gate1/reconcile-core` (they have been in lockstep since `dc671b62c`).

---

## Status — 2026-09-12 (later), both pins moved and the three-repo sweep

A cross-repo review of nx_vulkan, eXMC and pathmc_ex, with the pins moved
forward first so the review saw today's code rather than last week's. The doc
edits and `docs/REVIEW_PLAN.md` are committed; the `mix.lock` move is not
(see the segfault below — the lock in the tree is back at `bae9221`).

**nx_vulkan `5f65398` segfaults this suite; the lock stays at `bae9221`.**
nx_vulkan's `main` moved again the same afternoon (16 commits: the two-GPU
design branch merged, increment 2 "the context travels with the tensor",
`77bb61f`). Moved the lock to it, rebuilt, ran `EXMC_COMPILER=vulkan mix
test` twice: **SIGSEGV both times within 10 s**, before the device banner,
exit 139. `coredumpctl` backtrace, MEASURED:

    #0  0x0000000000000000
    #1  libvulkan.so.1 + 0x2a192
    #2  vkEnumerateInstanceExtensionProperties (libvulkan.so.1)
    #3  libnx_vulkan_vulkano.so + 0x1eeb96

`test/nuts/leapfrog_leaf_diff_test.exs` alone, `test/exmc/jit_vulkan_test.exs`
alone, and a three-op `mix run` at the same rev all pass, so it is the
suite's concurrency at first touch, not an op. The mechanism, INFERRED from
`native/nx_vulkan_vulkano/src/lib.rs` at `5f65398`: `ctx_for/1` now goes
through `slots()` → `enumerate_devices()`, and `default_slot()` →
`resolve_default()`, and when no context is open yet `enumerate_devices()`
builds a throwaway `VulkanLibrary::new()` + `Instance::new()` **outside the
init mutex** that serialises `build_ctx()`. Under `max_cases: 176` every
dirty-scheduler thread arriving first does that at once, and each temporary
instance is dropped on return. The comment above the mutex in the same file
describes this exact window — "a SIGSEGV was observed on mac-247 on
2026-09-07 inside the Vulkan loader under
`vkEnumerateInstanceExtensionProperties`" — as the thing the single-context
design had closed. Increment 2 reopened it one call earlier. The fix is
theirs: take the init mutex in `slots()`/`resolve_default()` when no context
is open, or make the library handle `'static`. Reported in nx_vulkan
`NEXT.md`.

**Fixed the same evening, `d657387`** (their session confirmed the mechanism
frame for frame: `enumerate_devices()` now takes `CTX_INIT` when no instance
is open). MEASURED here at `f8fa9e9`, `mix test`, `max_cases: 176`: **723
tests, 1 failure** (Cauchy KS), no segfault, 792.7 s. Their `16d13f3` then
added the consumer surface this repo asked for — `Nx.Vulkan.Device.resolve/1`
(selector or slot → `{:ok, slot, info}`), `ChainTrace.dispatch_f64/7` with a
defaulted device argument, and one `Node` per device — so the pin to move
to is `16d13f3`, and `Exmc.NUTS.Vulkan.Dispatch` gains one line
(`docs/REVIEW_PLAN.md` Track 5 item 2). Their warm-up measurement on asus:
the first Vulkan client in a BEAM pays **13.0 s** (loader, instance,
allocators, pipeline cache), the second device 204 ms; size any startup
budget against 13 s, not the 2.6 s the standalone probe measured.

**Pins.** `mix.lock` here follows nx_vulkan `bae9221` (from `9a8427c`, 52
commits; `mix deps.update nx_vulkan`; not `5f65398`, see above). pathmc_ex follows eXMC `dc671b62c`
(from `147305261`, 12 commits). The 2026-09-12 status below this one still
says `9a8427c` and "selector on `feat/device-selector`"; both were true when
written and are not now — the selector (`befb91b`) was merged to nx_vulkan
`main` in `70c96e9` on 2026-09-11 23:42, before that status was written, so
the "unpinned here" half was right and the "sits on a branch" half was not.
`docs/HANDOVER_ASUS.md` now says `NXV_DEVICE=name:M4000`, as it asked to on the
pin move.

**MEASURED on super-io, RTX 3060 Ti, driver 580.178.04, nx_vulkan `bae9221`:**

| run | result |
|---|---|
| `mix test`, auto-detected Vulkan (`EXMC_COMPILER` unset) | 723 tests, **2** failures, 957 s |
| `EXMC_COMPILER=vulkan mix test`, the fleet's arm | 723 tests, **1** failure (the Cauchy KS check) |
| `mix test test/custom_dist_test.exs:190`, same arm | 1 failure |
| same test, `EXMC_COMPILER=vulkan` | 0 failures |
| pathmc_ex `mix check` on `dc671b62c` | green (584 tests, 9 sampling, dialyzer) |
| pathmc_ex shader probe on `dc671b62c` | unchanged: single-equation synthesises, multi `:multiple_custom_nodes` |
| pathmc_ex `guide/02` on `dc671b62c`, as committed | **fails**, line 246, exactly as `994305de4` said it would |
| pathmc_ex `guide/02` after the fix | passes |

The two failures: the Cauchy KS check in `ValidatorTest`, host-specific to
super-io and already recorded above; and `CustomDistTest` "custom dist works
with NUTS sampler", which is **not a regression from the bump** but an
arm difference. The model is a single Custom RV with no priors, and
`CustomSynth.extract_components/1` refuses it as
`:no_free_rvs_in_custom_only_model`; whether that refusal falls back to the
per-op path or raises `SynthUnsupportedError` depends on
`:allow_vulkan_perop_sampling`, which `config/runtime.exs` sets **only when
`EXMC_COMPILER=vulkan` is in the environment**. That asymmetry is deliberate
and documented — the comment above it in `runtime.exs` and `Exmc.JIT.describe/0`'s
moduledoc both record the same 16/1 vs 16/0 measurement on mac-248 and say
"the fleet convention for Vulkan-only hosts is the explicit form". So this is
not a regression and not undocumented; it is a convention that `test_helper.exs`
(which keys its excludes off the *detected* backend so the arm cannot be
misread) does not share. What was open was the shape: one test needs the
per-op fallback, and the whole arm carried the flag for it. **Done the same
day** (`docs/REVIEW_PLAN.md` Track 1 item 1): `CustomDistTest` sets the flag
for itself with `put_env_scoped/2`, the block is out of `runtime.exs`, and
plain `mix test` on this host is now **723 tests, 1 failure** (the Cauchy KS
check) — identical to the explicit form, MEASURED.

**The nx_vulkan bump itself is clean.** The pin→HEAD diff in nx_vulkan `lib/`
touches `device.ex` (new), `native_v.ex` (additive: device NIFs and
`NXV_NIF_PROFILE`), `shader.ex` (docstring); the Rust diff is additive device
selection; no leapfrog NIF, no precision default, no release profile changed.
Every nx_vulkan function this repo calls has the same arity and semantics at
both revs.

**The deeper review is planned in `docs/REVIEW_PLAN.md`** (arms explicit →
format → EXLA arm re-measured → cross-arm parity → public API → nx_vulkan
contract), with sibling plans in nx_vulkan and pathmc_ex that share its
CPU / EXLA / Vulkan vocabulary.

**What the sweep found that is ours to fix** (the cross-repo list is in the
handoff to the user, 2026-09-12):

- `CHANGELOG.md` Unreleased was empty across 40 commits including
  `994305de4`, which broke a downstream notebook. Filled in this tree.
- Four docstrings cited nx_vulkan state that no longer exists
  (`leapfrog_chain_*` shaders, Stage 1.5.4 "expected to fail",
  `Spirv.validate_file/1` "lands in f2c0c69", `248_TODO.md`). Fixed.
- `@moduletag :vulkan` on four modules is excluded by nothing;
  `jit_vulkan_test.exs` says `mix test` skips it. Bulkhead and server tests
  fail rather than skip off-GPU. Open.
- `mix format --check-formatted` fails on **26 files** at `dc671b62c`, before
  any edit here — `tree.ex`, `sampler.ex`, `custom_synth.ex`, `leapfrog.ex`,
  the poker modules, seven test files among them. There is no format gate in
  this repo and nothing has run the formatter for some time. Open; a
  whole-tree `mix format` is its own commit.
- `mix.exs` admits rustler 0.37 (`~> 0.36`); nx_vulkan pins `~> 0.36.0` and
  says 0.37 is broken. exmc's own NIF builds on rustc 1.90, which nx_vulkan's
  toolchain pin says rustler 0.36 cannot do — one of those two claims is
  stale. Open.
- README describes neither the dependency wiring nor the Vulkan path at all.
  Open.

---

## Status — 2026-09-12, asus moved to 580 and the confound may already be broken

`docs/HANDOVER_ASUS.md` is the handover; this is the one-paragraph version.

asus is now **FreeBSD 15.0 + driver 580.178.04**, swapped live by the nx_vulkan
session with `kldunload`/`kldload` — no reboot, same kernel and userland, one
variable. Both cards enumerate under Vulkan 1.4.312: GTX 1660 Ti (Turing) and
Quadro M4000 (discrete Maxwell). That is exactly the intervention
`docs/TWO_GATES_THAT_DO_NOT_GATE.md` Part 2 says is required to separate driver
branch from OS, and they REPORT the outcome already: the crash module
"segfaults on both cards under 470 and runs under 580". Unconfirmed in our own
harness — that is task 1 of the handover, and until it is run this stays a
citation, not a result.

If it holds, three things follow at once. FreeBSD is eliminated.
Architecture-as-lookup is falsified **non-vacuously** — the M4000 is Maxwell and
crashed under 470 while the Jetson's Maxwell Tegra runs it, and one Turing part
does both depending only on the driver. And the 470 branch is the survivor, on
a within-host, within-card, single-variable design rather than on three FreeBSD
hosts that were never three independent observations.

Two operational notes worth having before anyone runs there. **asus is shared:**
the nx_vulkan session is measuring cross-process GPU contention on it, and an
eXMC suite on the other card is the neighbour that perturbs their numbers —
check the box is idle first. **There is no device knob in the nx_vulkan rev we
pin** (`9a8427c`): `build_ctx()` sorts on device *type* only and both cards are
`DiscreteGpu`, so it always takes the 1660 Ti. Their `NXV_DEVICE` selector is
real but sits on `feat/device-selector` @ `befb91b`, unpinned here. Until the
pin moves, `VK_LOADER_DEVICE_SELECT=0x10de:0x13f1` reorders the loader's
enumeration and selects the M4000 with no code change — MEASURED against
`vulkaninfo` on the box, and it works because `min_by_key` returns the first
minimum on a tie. Confirm the card from the banner on every run.

---

## Status — 2026-09-11 (later), the leaf-diff harness is a gate that can fail

`bench/leapfrog_leaf_diff.exs` is now `test/nuts/leapfrog_leaf_diff_test.exs`
(`acccf8348`), tagged `:requires_vulkan`. The bench copy is deleted rather than
left beside it — two copies drifting is the failure that started this. **The
prerequisite named in `docs/OBS_LOOP_FUSION.md` §6 and in the 2026-09-07 status
below is met**; the obs-axis parallelism it gates is not, and is now the whole
of the remaining performance opportunity.

### The threshold came from four machines, not from this one

The bench flagged at `1e-6`. That was not slack when written — the GLSL then
carried its distribution constants at f32 and packed the obs buffer from f32
tensors, so ~`1e-7` was the floor the path could reach. The f64 migration
removed the floor and nobody revisited the number.

Measured at `994305de4`, worst relative Δ over three (eps, q0, p0) settings:

| host | q | p | grad | logp | offset spread |
|---|---|---|---|---|---|
| super-io | 1.92e-15 | 2.22e-15 | 3.07e-15 | 9.20e-16 | 1.78e-14 |
| mac-247 | 4.77e-15 | 4.22e-15 | 6.72e-15 | 1.53e-15 | 2.84e-14 |
| mac-248 | 4.77e-15 | 4.22e-15 | 6.72e-15 | 1.53e-15 | 2.84e-14 |
| jetson | 4.77e-15 | 4.22e-15 | 6.72e-15 | 1.53e-15 | 2.84e-14 |

`@tol 1.0e-13` is ~15x the fleet worst, `@offset_tol 1.0e-12` ~35x the worst
spread. Three of those hosts agree to the last digit and the fourth does not,
which is why the bound is not taken from whichever machine you happen to be on.

### Three mutations, and two of them corrected the plan

1. Perturb the host reference's step by 1e-7 → all five tests fail. Live.
2. **All three sigmas at 1.0 → `grad` 1.212e-13, over the bound.** Trajectory
   geometry on a legitimate model, not a defect: a tighter posterior at
   eps=1.139 travels further per step. So the bounds are **fixture-calibrated**
   and the headroom is not generous — measure a new fixture before adding it.
   Not loosened: a bound widened for a model that is not in the file buys
   nothing and costs sharpness for the models that are.
3. `logp_chain[k]` lagging its position by one — the historical defect that
   read as "Ampere over-dispersion" for three weeks → all five fail.

Mutation 3 also falsified the plan's claim that the offset-constancy check is
the sharpest assertion in the file. It never fires: the element-wise `logp`
assertion catches the lag first, and as written the offset check is *implied*
by it. It is kept as a standby — it becomes load-bearing only if the
element-wise bound is ever relaxed to permit a constant normaliser, which
today's measurements say is unnecessary (the normaliser is 0.0 on every host
and arm).

Two fixtures were added for the paths with the least element-wise coverage and
the most recent churn: a vector RV with `Nx.dot` over a captured design matrix
(`371785ff5`) and a Custom likelihood that reads its observations
(`d299f4fc4`).

**One process note worth keeping.** `671150a0d`, the commit before it, shipped
only the deletion: its `git add` listed the already-`git rm`'d bench path
beside the new files, git aborts the whole invocation on an unmatched pathspec,
and the commit took what was already staged while the message described a
promotion the diff did not contain. Fixed forward rather than amended, because
the fleet fast-forwards from origin and a rewrite is a manual repair on four
machines. Stage explicit paths, then read `git show --stat` before pushing.

---

## Status — 2026-09-11, the multi-equation constraint is `:multiple_custom_nodes`

MEASURED by the pathmc_ex session at `147305261`, on real `Compile.Exmc`
output:

| model | result |
|---|---|
| single equation `Y ~ X` | **synthesised**, `["beta_Y[0]", "beta_Y[1]", "sigma_Y"]` |
| single, sigma pinned (their oracle) | **synthesised** |
| multi-equation `M~X; Y~M+X` | `{:unsupported, :multiple_custom_nodes}` |

The reason propagates now where a bare atom came back before, so the discard
site fixed in `147305261` was on their path. But the reason is not the one this
document has been carrying.

### Two of our records were wrong, and this corrects them

**`capture_guard/3`'s per-node-span refusal is NOT what blocks multi-equation
models.** I asserted that from reading the guard, never ran it, and it reached
their `PLAN.md`, `EXMC_NOTES.md` and a handoff doc as though measured. It is
also listed in this document's 2026-09-07 status as the thing that "the
consumer would notice if the fused shader ever grew per-node observation
spans". **That item would not unblock them.** Do not spend on it for that
reason.

Verified here while checking: build the condition `capture_guard/3` guards --
captures alongside per-node spans -- and it SYNTHESISES, because `obs_spans/1`
returns `:full` whenever a Custom likelihood is present. The combination may be
unreachable.

**The actual constraint is a scope limit.** `extract_components/1` returns
`{:error, :multiple_custom_nodes}` for more than one `Dist.Custom` node, with
the comment *"out of scope for R1. A real multi-likelihood model needs separate
handling."* Whether lifting it is hard, easy or deliberate is not established
and is not guessed at here.

It is worth knowing that this is structural on the consumer's side, not
incidental: `PathMC.Compile.Exmc` emits one Custom likelihood per outcome, and
a path model is a set of regressions by definition. Every multi-equation model
that library can express has several Custom nodes.

Their models also never use the scalar-placeholder idiom -- every equation gets
`Builder.obs(ir, "#{outcome}_obs", outcome, response)` with a real tensor -- so
each carries a non-empty observation region. Relevant to whatever lifting
`:multiple_custom_nodes` would involve.

### The process point, fourth instance, and the receiving half

I treated a reading of the code as an execution of it, having applied exactly
that correction to my own emitter two days earlier after two guessed axis
signatures matched nothing.

The new part is the receiving side. The claim was specific enough -- a named
tuple carrying two counts -- to read as a measurement, and it propagated into
three of their files across several commits while the probe that answers it sat
in their repo root. Their formulation, which is better than mine: *a claim about
what another system does is measurable or it is a guess, and which one it is
does not depend on how confident the sender sounded.*

---

## Status — 2026-09-10 (later), fleet at d410b183a — obs buffer populated

First fleet run through `scripts/fleet_verify.sh`, which landed with the
obs-buffer work. 713 tests, up from 707 with the six obs-buffer guard tests.

| host | GPU | result | failures |
|---|---|---|---|
| mac-248 | GT 750M | **713 / 0** | — |
| mac-247 | GT 650M | 713 / **1** | `PokerTest`, timeout |
| Jetson | Tegra X1 | 713 / **2** | `PokerTest` + `IntegrationTest`, timeouts |

Identical to the `371785ff5` and `144d441db` runs. Populating the observation
buffer from the Custom RV's own observed value, and the two refusals it made
necessary, changed nothing on any host — which is the intended result, since
every fleet fixture either captures its data or has no Custom likelihood at
all. The six new tests pass everywhere.

`SUITE EXIT` was 2, 0 and 2, and each run printed its `### SUMMARY` line — so
the non-vacuity gate ran rather than merely existing, which is the thing the
previous two runs could not say.

### The script did its job, including the part that had been guesswork

`epmd` was found on both layouts without a per-host special case: FreeBSD ports
put it on PATH at `/usr/local/lib/erlang27/bin/epmd`, the Jetson needs the
search under `~/.asdf/installs/erlang`. Two DistributedTest failures that were
twice mistaken for a code regression have now not recurred on a run where
nothing was done by hand.

The timeout was budgeted from the Jetson (~62 min) rather than from the
Keplers (~20). The previous attempt used 50 min and produced a log with
failure blocks and no summary — the exact shape the non-vacuity check flags,
which an outer kill prevents it from ever reporting. That asymmetry is now in
the script's header.

### Both remaining gates are now planned

`docs/TWO_GATES_THAT_DO_NOT_GATE.md` covers `bench/leapfrog_leaf_diff.exs` and
the Cauchy KS failure. Two findings from writing it, both measured:

* the leaf-diff harness has **two** defects, not one. Its booleans are
  discarded, which was known — and its threshold is `1e-6` while the measured
  agreement is **~3e-15**. Asserting the booleans as they stand yields a test
  that is green and nearly meaningless, which is worse than none because it
  reads as coverage.
* **the Cauchy KS failure is host-specific.** It fails on super-io and passes
  on mac-247, mac-248 and the Jetson — zero occurrences in the fleet logs of
  three separate runs. The test is seeded and its reference is the analytic
  CDF, so identical draws would fail everywhere. The draws differ by host,
  which sits in tension with this document's own claim that both Keplers and
  the Ampere produce bit-identical q/p/grad. Measure that before touching the
  tolerance.

### Not run: posteriordb

Unchanged NIF, byte-identical goldens, and the 2026-09-09 fleet run established
that every non-completion there is a 30-minute timeout rather than an accuracy
failure. `scripts/fleet_verify.sh --pdb` runs it whenever it is wanted, and
gates on the fixture count so a host missing them cannot pass by doing nothing.

---

## Status — 2026-09-10, fleet re-verified at 144d441db (nx_vulkan 9a8427c)

| host | GPU | result | failures |
|---|---|---|---|
| mac-248 | GT 750M | **707 / 0** | — |
| mac-247 | GT 650M | 707 / **1** | `PokerTest`, timeout |
| Jetson | Tegra X1 | 707 / **2** | `PokerTest` + `IntegrationTest`, timeouts |

Identical to the `371785ff5` run. nx_vulkan `9a8427c` is thirteen commits of
benchmark and documentation work with nothing under `native/`, and the 18
shader goldens are byte-identical across the bump, so this is the expected
outcome rather than a surprise — but it is now measured rather than assumed.

### The Jetson read 707/4 first, and both extras were the harness

Worth writing down because the same trap has now cost time twice in three days.

* **`epmd` is not on a non-interactive PATH.** Two `DistributedTest` failures
  with `econnrefused` and `:nodistribution`. `epmd` lives in the erlang install
  (`~/.asdf/installs/erlang/27.2.4/erts-15.2.2/bin/epmd`) and is not an asdf
  shim, so a `bash -s` session cannot find it and the BEAM cannot auto-start
  it. With it running, all 5 distributed tests pass — so the Jetson's real
  result is 707/2. The same failure hit super-io on 2026-09-07 for the same
  reason. **Any fleet script must put the erts bin directory on PATH and start
  epmd**, or two tests fail for reasons that have nothing to do with the code.

* **`set -e` plus a failing `mix test` swallowed the completion marker.** The
  runner script ends with `echo "### SUITE EXIT: $?"`, and `mix test` exits
  non-zero whenever anything fails, so `set -e` aborted the script first. The
  waiting loop then blocked on a marker that could never arrive while both
  suites had in fact finished. A completion marker guarded by `set -e` is not
  a completion marker; it needs `|| true` or the trap has to be explicit.

Neither is a code defect and neither changes a number, but both make a run
report something other than what happened, which is the failure this document
keeps returning to.

### posteriordb not re-run

Deliberate. The NIF is unchanged, the goldens are byte-identical, and the
2026-09-09 run established that every non-completion is a 30-minute timeout
rather than an accuracy failure. Re-running would cost hours per host to
reproduce a result that cannot have moved. The fixtures now arrive with
`git pull` (`702fb780f`), so it is one command whenever it is wanted.

---

## Status — 2026-09-09, fleet + posteriordb at 371785ff5 (vector RVs)

super-io excluded by request; the three remote hosts only.

### Fleet, `EXMC_COMPILER=vulkan mix test`

707 tests, up from 699 — the eight new slot-layout and vector-RV tests.

| host | GPU | result | failures |
|---|---|---|---|
| mac-248 | GT 750M | **707 / 0** | — |
| mac-247 | GT 650M | 707 / **1** | `PokerTest`, 300 s timeout |
| Jetson | Tegra X1 | 707 / **2** | `PokerTest` + `IntegrationTest`, timeouts |

**Identical failures to the `f1e9b2207` run**, so the vector-RV work introduced
no fleet regressions, and the new tests pass on all three. The two timeout
classes are unchanged and still distinct: mac-247's is inside
`chain_synth_vulkano`, the Jetson's are in the host interpreter.

### posteriordb, fast tier, Vulkan arm

| host | PASS | CRASH | completed |
|---|---|---|---|
| Jetson | 2 / 6 | 4 | mesquite, eight_schools |
| mac-248 | 1 / 6 | 5 | mesquite |
| mac-247 | 1 / 6 | 5 | mesquite |
| (super-io, 2026-09-07) | 4 / 6 | 2 | + sblri, kidiq |

**Every crash is `:timeout`. Zero accuracy failures on any host.** Where a
model completes the numbers are good — R-hat <= 1.001, error <= 0.05,
identical `ess`/`lf`/`div` across hosts for the same model (mesquite: ess 2018,
lf 25582, div 36/4000 on both Keplers).

So the pass rate is a clock, and it is monotone in GPU capability: 4/6 on the
Ampere, 2/6 on the Tegra, 1/6 on both Keplers. Only the smallest observation
axis (mesquite, n_obs=46) survives everywhere. This is the serial-reduce
ceiling in `docs/OBS_LOOP_FUSION.md` §6 seen at fleet scale — fusion bought
1.8x and the remaining cost is arithmetic run by one invocation while 255 idle.

Two harness fixes from 2026-09-07 earned their place here:

* every crashed posterior is NAMED. The previous run could only report
  `unknown (crash)`, which is true and useless; `ordered: true` plus a zip
  (`45c7b7566`) means the table above can say WHICH five timed out.
* `on_timeout: :kill_task` (`4c658c0d7`) is why there is a table at all — under
  the old default the first timeout would have killed the whole stream and
  reported one PASS.

### The fixtures are not in git, and that is deliberate

`benchmark/posteriordb/posteriordb_processed/` is gitignored — 28 MB of
regenerable reference draws — so no fleet host had it and the harness found
nothing to run. Staged as a tarball to each host. That is data, not code: the
code under test still comes from `origin`, so the provenance the harness
records is unaffected. Anyone re-running this on a fresh host has to do the
same thing or the run silently has nothing to do.

### A correction, and a process note

I told this project's user, and told the pathmc_ex session, that pathmc models
would now reach the shader because "they capture their data". **Wrong.** They
capture the design matrix X; the response y arrives through `Builder.obs` as
the Custom closure's first argument, and those are different things.

Their real models get bare `:unsupported`, refused BEFORE the new guard,
because `PathMC.Compile.Exmc` never emits `Nx.dot` — it sums over slots, and
that is load-bearing: a design-matrix column for a transformed term does not
exist until its parameter is drawn, so `adstock(tv, decay=theta)` cannot be
folded into a matrix at all. The `dot` clause covers an idiom that library
structurally cannot use for its interesting models.

The realistic ceiling on that line of work is therefore narrower than claimed:
single-equation, no-transform models only, and only after the observation
buffer is populated from the Custom RV's own observed value.

The process point, because it is now the third instance in one week: I reasoned
about a plausible reconstruction instead of running the real artifact. The
other two were emitter clauses written from an assumed axis signature that
matched nothing, fixed only by enumerating the actual `dot` nodes. Probe the
artifact, not a lookalike — and when the artifact belongs to another project,
ask that project to run it.

---

## Status — 2026-09-07 (later), fleet + posteriordb at the fused shader

Verification of `f1e9b2207` (fusion + nx_vulkan `bc54f34`) across all four
hosts and posteriordb. **No wrong answers anywhere. Every failure is a clock.**

### Fleet, `EXMC_COMPILER=vulkan mix test`

| host | GPU | result | failures |
|---|---|---|---|
| super-io | RTX 3060 Ti, Linux | 688 / **1** | Cauchy KS (pre-dates today) |
| mac-247 | GT 650M, FreeBSD | 688 / **1** | `PokerTest`, 300 s timeout |
| mac-248 | GT 750M, FreeBSD | 688 / **0** | — |
| Jetson | Tegra X1, aarch64 | 688 / **2** | `PokerTest` + `IntegrationTest`, timeouts |

Every failure is a `TimeoutError` or a marginal statistical gate. None is a
wrong number.

**The two timeout classes are different mechanisms and the stack traces say
so** — this is the distinction to keep, because the fix is different for each:

* **mac-247's `PokerTest`** times out *inside* `Dispatch.chain_synth_vulkano/8`.
  That is the serial-reduce cost on the fleet's weakest card, and it is what
  `docs/OBS_LOOP_FUSION.md` §6 is about.
* **Both Jetson failures** time out in `Nx.Defn.Evaluator` over
  `Nx.BinaryBackend` and `Nx.Defn.Grad` — the per-op HOST interpreter. Those
  models are not reaching the fused shader on that box at all. Per DECISION
  94 the per-op Vulkan arm is an interpreter on the CPU, so this is the
  Jetson's weak CPU, not its GPU.

Open, and not chased today: **why `PokerTest` synthesises to a shader on
mac-247 and falls back to the interpreter on the Jetson.** Same tree, same
commit, same lock. That difference is a finding waiting to happen.

`mac-247` ran 688/2 once and 688/1 on a clean re-run, so one of its failures
is flaky. The Jetson build needed `~/.asdf/shims` and `~/.cargo/bin` on the
PATH — a non-interactive ssh gets neither, and `mix`/`cargo` "not found" is
what a fleet script sees first.

### posteriordb, fast tier, super-io

| model | n_obs | n_beta | Vulkan | EXLA |
|---|---|---|---|---|
| eight_schools_noncentered | hier. | — | PASS 480 s | PASS 283 s |
| mesquite-logmesquite_logvolume | 46 | 2 | PASS 297 s | PASS 269 s |
| sblri-blr | 100 | 5 | PASS 768 s | PASS 285 s |
| kidiq-kidscore_momhs | 434 | 2 | PASS 994 s | PASS 287 s |
| nes2000-nes | 476 | 9 | **timeout** | PASS 328 s |
| earnings-earn_height | 1192 | 2 | **timeout** | PASS 338 s |

**Vulkan 4/6, EXLA 6/6 — and the failures are wall-clock, not accuracy.**
Where both arms finish, Vulkan matches or beats EXLA on error (0.04/0.02/0.03/
0.03 against 0.03/0.05/0.01/0.03) with R-hat <= 1.002 everywhere.

The shape is the point. **EXLA is flat in `n_obs`** — 269 to 338 s as the
observation axis goes 46 -> 1192, because it vectorises the reduction.
**Vulkan is monotone in it** and runs out of clock. That is the cleanest
demonstration of the serial reduce this project has, and it is a better
argument for the parallelism work than the microbenchmarks were: two real
posteriors are currently unsamplable on the GPU arm for no reason but the
loop.

The full tier under Vulkan is therefore not runnable end to end today. That is
not new breakage — it is the same defect, seen at the scale it actually bites.

### Two harness defects, both mine, both found by this run

* `Task.async_stream` defaulted to `on_timeout: :exit`, so ONE slow posterior
  killed the whole stream and the `{:exit, reason} -> %{status: :crash}` clause
  written to record exactly that **could never fire**. A full-tier run reported
  one PASS and then died. Fixed to `:kill_task` in `4c658c0d7`.
* `ordered: false` meant a crashed element could not be named — both timeouts
  logged as `unknown (crash)`. Fixed with `ordered: true` + a zip in
  `45c7b7566`.
* `exmc_dirty` counted UNTRACKED files, so a stray `?? .claude/` stamped a run
  `(DIRTY)` and invalidated the comparability the field exists to certify.
  `-uno` in `4c658c0d7`.

The first of those is the project's recurring failure mode — a check written,
and nothing letting it fire.

---

## Status — 2026-09-07, the host half MEASURED on Kepler, and the serial reduce

Two results, both from mac-248 (GT 750M, headless, 8 cores, idle). The first
closes an outstanding gap in this document. The second is a defect the first
one found by accident and is much the more important of the two.

`bench/chain_trace_split.exs`, at `5deaf60da`, nx_vulkan `7af37b3`.

### 1. The Jetson decomposition was a subtraction. This one is not.

This file has carried, since 2026-09-01, a split of one dispatch into ~29%
GPU / ~25% CPU-in-NIF / ~46% CPU-outside, obtained by subtracting a benchmark
median from a wall-clock average — and its own admission that **neither Kepler
had ever had the two halves separated at all**. Both are now addressed.

Three instruments, none computed from the others:

| | what it is |
|---|---|
| `wall` | a real `sample_compiled` run's wall clock |
| `in chain/8` | `Dispatch.dispatch_micros/0` from that same run — a timer around the call |
| `host` | `ChainTrace.replay/3`'s wall clock, every dispatch served from recorded bytes |

Because none is derived from another, `host + in-chain ~= wall` is evidence
rather than arithmetic. Measured, single chain, 2000 draws + 500 warmup,
medians of 7, three independent passes:

| model | dispatches | in chain/8 | host | us/dispatch | Closure A |
|---|---|---|---|---|---|
| scalar obs, d=1 | 2700 | **42.6%** | **55.8%** | 188.1 | -1.6% |
| scalar obs, d=1 | 2700 | **42.6%** | **55.9%** | 189.3 | -1.5% |
| scalar obs, d=1 | 2700 | 42.9% | 61.0% | 187.8 | +3.9% |
| n_obs=64, d=2 | 622 | **92.8%** | **5.1%** | 33813.5 | -2.1% |

Every closure is inside that run's own wall spread (8-12% for the scalar
model, 0.8% for the observed one), and the busy-wait reconstruction agrees
to 0.0-5.4%. Pass 3 is the noisy one — its replay spread was 5.1% against
2.1-2.2% for the others; read passes 1 and 2.

**Read `in chain/8` correctly.** It is marshalling, allocation, submit, fence,
readback *and* GPU compute. It is not a GPU-utilisation figure and must never
be quoted as one. Separating that pair needs an instrument inside the NIF,
which we do not have.

The scalar result lands near the Jetson's subtraction-derived one but leans
the other way: 43/56 in-call/outside here against the Jetson's implied 54/46.
That is one more reason not to have trusted the subtraction.

### The instrument has a self-inflicted bias, and it is 25%

The replay runs later, hotter, and in a process holding the whole trace live.
A control arm repeats the timed run under exactly those conditions, and it
found something the arm was not built for: **holding a 2.64 MiB trace slows
the NIF CALL down by 25%** (507.8 -> 634.5 ms in-chain), not just the tree
logic. Allocator pressure from a large live refc-binary set reaches inside
the dispatch.

So the control's tax must be taken on the *host portion*, not the wall — the
in-chain part is already inside `in_chain_ms`, and subtracting the wall delta
counts it twice. That error drove Closure A to -11% across two passes, which
is how it was caught: the correction overshot by more than the raw error it
was correcting.

Consistency check the numbers pass on their own: the control is 25-27% for a
188 us dispatch and **-0.0%** for a 33.8 ms one, where the same absolute cost
is invisible. Anyone recording a long trace should expect the recorded run to
be slower than the run they meant to measure.

### 2. THE FINDING — the fused chain shader does not parallelise over observations

The observed model spends 93% of its wall inside `chain/8` at 33.8 ms per
dispatch, against 188 us for the scalar one. 180x. That is not a plausible
cost for 64 observations on any GPU, so: sweep `n_obs`, K=8, d=2, median of 3
x 200 dispatches, same host.

| n_obs | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| us/dispatch | 348 | 484 | 750 | 1288 | 2312 | 4407 | 8588 | 17309 | 33685 |
| SPIR-V bytes | 41992 | 41992 | 41992 | 41992 | 41992 | 41992 | 41992 | 41992 | 41992 |

**`cost ~= 215 + 131 * n_obs` us, to within 2% over eight doublings.** Every
doubling of the data doubles the dispatch. Per leapfrog step that is
~16 us *per observation*.

The SPIR-V is byte-identical across all nine sizes, so this is not the
pipeline-ceiling defect returning — `21700c04a` holds, the data is in the
extras SSBO and the shader does not grow. It is the `/*REDUCE_SUM*/` marker
expanding to a **serial GLSL for-loop**: one invocation walks the whole
observation axis while the other 639 cores idle. The chain shader parallelises
over the K leapfrog steps and over d, and not at all over n_obs.

This is the same shape as the per-op finding already in this file — "the fix
is fusion or batching, not widening a gate" — one level further in. The fusion
happened; the reduce stayed scalar.

**Consequences.**

* Every posteriordb model with a real likelihood is on this curve. The
  sblrc-class fixes made the answers right; they did not make this fast.
* A "GPU-bound, 93% in the dispatch" reading of the observed model is
  technically true and completely misleading. The card is idle.
* It reframes the optimisation queue. Readback batching and fence costs are
  ~200 us of fixed overhead; at n_obs=256 there is 33.5 ms of serial loop
  sitting next to them.

**The mechanism is loop fission, not just a serial loop** — established after
the above was written, and it changes what to fix first. Sweeping the model
instead of the data, slope over n_obs in {16,64,256}:

| model | obs loops emitted | ops in them | us/obs | us per op-obs |
|---|---|---|---|---|
| `y ~ N(mu, 1)`, d=1 | 3 | 16 | 2.4 | 0.15 |
| `y ~ N(mu, sigma)`, d=2 | 15 | 117 | 130.7 | 1.12 |
| `y ~ T(df, mu, sigma)`, d=3 | 165 | 671 | 716.5 | 1.07 |

`CustomSynth.Glsl` emits one `/*REDUCE_SUM*/` per `sum` node in the autodiffed
log-density and `do_transform_rs/7` gives each its own complete `for` loop, so
a three-parameter Student-t walks the observation axis **165 times** and loads
`obs_inv_mass[j]` 165 times per observation. Bodies average ~4 ops, all 15
loops of the d=2 model carry identical bounds, and `cse_loop_body/1` cannot
help because its `@cse_min_len 18` sees each tiny body alone — the redundancy
is *between* the loops, where the pass does not look.

So the first fix is fusion, which is **bit-identical** (same j, same order, per
accumulator) and needs neither new tolerances nor barriers in divergent control
flow. Obs-axis parallelism is second, and it is the half that changes summation
order. Planned in `docs/OBS_LOOP_FUSION.md`.

**LANDED in `f2ae139d7`**, and the plan's own prediction was wrong. Loops fell
3->3 (d=1, its floor), 15->5 and 165->7; SPIR-V fell 41992->35392 and
568456->492552; all 18 golden digests matched **exactly**, so the bit-identity
claim holds. But 23.6x fewer traversals bought **1.83x**, not the "close to L"
the plan predicted. A 2x2 against `config :exmc, glsl_cse: false` says why:
CSE on the unfused shader is worth 0.2-1.9% (the `@cse_min_len` diagnosis,
confirmed), fusion alone is 1.49-1.79x, and CSE *after* fusion adds 1.25x at
d=3 and nothing at d=2. The arithmetic is the floor — fusion removes
traversals, not operations.

That makes obs-axis parallelism the whole of the remaining opportunity rather
than a second-order term: what is left is arithmetic run by one invocation
while 255 idle. Promoting `bench/leapfrog_leaf_diff.exs` into a test that can
fail is still open, and is a prerequisite for THAT change, which is not
bit-identical.

Two hypotheses were tested and refuted along the way; §1 of that document
records them so they are not re-derived.

---

## Status 2026-09-02 — the f64 batched chain shader is written and verified

The blocker is gone. `MultiRvCustomSpec.render_batched/1` now emits f64, every
model class `render/1` accepts renders through it, and a batched instance is
**bit-identical** to the same chain dispatched alone on all four output
buffers. Measured on super-io (RTX 3060 Ti), 17 tests in
`test/exmc/nuts/custom_synth/batched_shader_test.exs`.

### What was actually wrong

Four defects, three of which could not have been found by reading the shader.

1. **The template was f32 and could not have compiled.** `float eps`, `float
   q_init[]` — while the batched path's own reduce-sum rewriter emitted
   `double` accumulators into it. GLSL has no implicit double-to-float
   conversion, so every model with a vector observation failed in
   `glslangValidator`. Invisible because there was no f64 batch NIF to
   dispatch the result to, so nothing had ever rendered it in anger.

2. **The batched renderer was a drifted copy of `render_with_custom/1`.** It
   had lost the obs spans (a model with several observed nodes would have
   counted its whole likelihood once per node — the defect
   `docs/OPEN_VULKAN_OBSERVED_MODEL.md` describes, ~sqrt(n) too narrow a
   posterior and a frozen chain), the common-subexpression pass, and the f64
   transcendental rewrite.

3. **`chain_batch/5` appended the unread prior tail.** The same defect that
   cost the single-instance path 13.1x, still live on the batched one: 8
   Normal priors made a 152-byte push block against the NIF's 128-byte bound,
   16 made 280. So batching was **capped at about six free RVs** and every
   wider model raised and fell back. Nothing read the tail — the batched
   shader bakes priors as SPIR-V literals and the NIF forwards
   `sizeof(PushBlockBatchF64) = 24`.

4. **`synthesise_batched/1` sized `n_obs` from `ir.data` alone**, defaulting to
   1 for every `Builder.obs` model, and skipped `Exmc.Rewrite.apply/2`. `n_obs`
   is the per-instance extras stride (`inst * (n_obs + d)`), so a wrong value
   does not fail — it points instance 1 into the middle of instance 0's slice.

Also: `render_batched(%{custom: nil})` returned
`{:error, :prior_only_batched_not_supported}`, which excluded every conjugate
model — the entire class the coordinator exists to serve, since `Builder.obs`
models have no Custom term.

### The design that makes bit-identity a claim rather than a hope

Both variants now render from **one emitter**. `render_prior_only/3` and
`render_with_custom/3` take the template and the observation index expression
as arguments; `obs_inv_mass[j]` becomes `obs_inv_mass[extras_off + j]` and
nothing else differs. A batched instance therefore performs the same
arithmetic in the same order as the lone chain, which is why byte equality is
a reasonable thing to demand instead of a tolerance. A tolerance passes for a
shader that reads a neighbour's inverse mass or sums an observation slice one
element off, and those are exactly the failures this needed to catch.

Pinned directly: `emitted_lines/1` compares the two renderers' emitted bodies
line for line, for four model shapes.

### Verified, and verified to be non-vacuous

Bit-identity across prior-only (d=2), 1 RV, 3 RVs, vector obs (n_obs=4), 8 RVs
and 16 RVs, at 1, 3, 4, 5 and 6 instances, both `dir_sign` values. Plus an
instance-bleed test with observations at 3.0, -50.0 and 1000.0, each instance
matched against its own single-instance dispatch, and an inverse-mass bleed
test — inv_mass sits after the observations in each instance's extras slice,
so it is the half of the stride the observation test cannot reach.

Every one of these was mutation-tested. Five mutations, all caught:

| mutation | failures of 17 |
|---|---|
| `extras_off = inst * d` (drops `n_obs` from the stride) | 8 |
| batched obs index back to `"j"` | 2 |
| `chain_off = 0u` | 8 |
| `n_obs` back to defaulting to 1 | 9 |
| push prior tail restored in `chain_batch/5` | 2 |

The stride mutation has the signature you would expect and would not have
noticed by eye: instance 0 still matches (its offset is 0 either way), 1 and 2
break on all four buffers.

**The single-instance path is untouched, checked rather than assumed.** The
GLSL `render/1` produces is byte-identical to `92392d76e`'s for all five model
shapes (SHA-256 compared across a `git stash`), so no cached SPV is
invalidated and no sampling behaviour moves.

Full suite: **683 tests, 2 failures**, both pre-existing and both confirmed
identical at `92392d76e` by re-running them with these changes stashed —
`validator_test.exs:97` fails a KS check at d = 0.0999 against crit 0.0975
(alpha 0.001), and `level_set_integration_test.exs:11` times out at 300 s,
alone as well as under load.

### Two things left behind on purpose

`Push.prior_param_floats/1` and `Push.ensure_fits!/2` now have **no caller**.
Kept — public API, correct encoding — with docs saying plainly that no dispatch
path calls them and that their existence is not a reason to reinstate a tail.
`push_prior_param_floats_test.exs` was reframed to match; it was pinning "the
batched path's 128-byte cap", which no longer exists.

`chain_batch/5` accepts `obs: nil`, because a prior-only model has no
observations and Nx has no zero-size tensor to pass instead.

---

### Fleet-verified at `475bf73aa`

All four hosts, `~/exmc_oss` on each (mac-248's `~/exmc` stays excluded — it
holds `trial/accounts.config` on an unpushed branch).

| host | GPU | memory | batched shader test | full suite |
|---|---|---|---|---|
| super-io (.249) | RTX 3060 Ti | discrete | 17/17 | 683, **2** |
| mac-247 (.247) | GT 650M | discrete | 17/17 | 683, **1** |
| mac-248 (.248) | GT 750M | discrete | 17/17 | 683, **1** |
| Jetson (.250) | Tegra X1 | **unified** | 17/17 | 683, **3** |

**Every failure is pre-existing and was confirmed so, not assumed.**
`LevelSetIntegrationTest` times out at 300 s on all four. The Jetson adds two
more, `PokerTest` at 300 s and `IntegrationTest` at 120 s — both
`ExUnit.TimeoutError`, both in host-side `Nx.Defn.Evaluator` /
`BinaryBackend` stacks, and both reproduced identically at `92392d76e` by
checking that commit out on the Jetson and re-running the two files.
super-io's extra failure is the marginal KS check
(`validator_test.exs:97`, d = 0.0999 vs crit 0.0975 at alpha 0.001); it passes
on both headless Macs, which is what a noisy desktop looks like.

**The unified-memory arm is the one worth naming.** The Jetson reports
`unified memory: true (staging path: OFF)` — a different path through the
batch NIF, no staging buffers — and the batched instances are still
bit-identical to their lone dispatches there. Both memory regimes are covered.

`leapfrog_chain_synth_batch_f64/6` exported on all three remotes after the
rebuild.

### The `.so` hazard, measured across the fleet

Every host got `mix deps.clean nx_vulkan --build && mix deps.compile
nx_vulkan`, with `cksum deps/nx_vulkan/priv/native/*` recorded either side.

| host | before | after | |
|---|---|---|---|
| mac-247 | `426809503 3286600` | `164127991 3292760` | **replaced** (was the `d210601` build) |
| mac-248 | `164127991 3292760` | `164127991 3292760` | unchanged — already correct |
| Jetson | `2606488089 3408192` | (new) | **replaced** |

Two things fall out of that table. The lock moved `d210601 -> cccbd71`, which
is the commit that *adds* the batch NIF, so 247 and the Jetson genuinely had
to rebuild — no ambiguity about whether the copy landed. And **247 and 248
produced byte-identical artifacts**: the Rust build is reproducible across
that FreeBSD pair, which is what makes `cksum` a valid staleness detector
there rather than a coincidence. 248 was unchanged because the earlier forced
rebuild had already produced exactly these bytes.

### The mac-247 anomaly was a BEAM segfault in Vulkan instance init

On mac-247 the scripted bare `mix test` printed nothing and returned within
seconds. It was recorded as unexplained; it is not. The checkout held a
**524 MB `beam.smp.core` timestamped `Sep 3 02:50`** — the same minute the
script gave up. Preserved at
`~/cores/beam.smp.247.2026-09-03T0250Z.core` on 247, moved out of the working
tree because an untracked core in a git checkout is one `git clean -fd` from
being gone.

    thread #1, name = 'erts_dios_3', stop reason = signal SIGSEGV
      frame #0: 0x0000000000000000
      frame #1: libvulkan.so.1`___lldb_unnamed_symbol1062 + 114
      frame #5: libvulkan.so.1`vkEnumerateInstanceExtensionProperties + 477
      frame #6: libnx_vulkan_vulkano.so`vulkano::library::VulkanLibrary::get_extension_properties
      frame #7: libnx_vulkan_vulkano.so`nx_vulkan_vulkano::ctx
      frame #8: libnx_vulkan_vulkano.so`...::leapfrog_chain_synth_f64
      frame #10: beam.smp`erts_call_dirty_nif

A jump through a **null function pointer** inside the ICD loader's dispatch
chain, on first-touch Vulkan initialisation, on a dirty-IO scheduler thread.

What this is and is not:

  * **Not the batched path, and not this commit.** It is `ctx()` on the
    `leapfrog_chain_synth_f64` entry — the live single-instance path — during
    instance creation, before any shader is dispatched. A pre-existing hazard
    that this fleet run happened to catch.
  * **Intermittent.** The same binary ran the full suite to 683/1 minutes
    later, and the batched shader test had passed on it moments before.
  * **Not a stale or dirty build.** 247 had just had
    `mix deps.clean nx_vulkan --build`, its artifact checksum changed, and the
    result was byte-identical to 248's. The crash happened *on* a
    verified-clean build, which is why wiping or rebuilding 247 is neither a
    diagnosis nor a fix.
  * **Not obviously the ICD config.** 247 carries four ICD manifests (intel,
    lvp, nvidia, radeon) with all four libraries present — and 248 carries
    exactly the same four and does not crash.

**Worth reporting upstream, carefully labelled.** `ctx()` is
`CTX.get()` -> build -> `CTX.set()`, not `get_or_init`. Several dirty-IO
threads can therefore build complete `VkInstance`s concurrently, and
`let _ = CTX.set(ctx)` **drops every loser**, running vulkano's `Drop` on a
live instance while another thread may still be inside the loader. That is a
real teardown race sitting exactly where this crash is — but it is a
hypothesis, not a finding: only `erts_dios_3` was inside Vulkan at dump time,
and the dump shows the state after the fault, not microseconds before it.

Second, smaller point for upstream: a NIF that segfaults during first-touch
init takes the whole VM with it, and the only trace is a core file in the
caller's working directory.

**Method note.** The run that crashed exited 0 and printed nothing. A suite
that prints nothing and exits 0 is indistinguishable from one that passed, if
the harness only checks the exit code — this one was caught because the script
grepped for a summary line and found none. Keep that grep.

## NEXT TASK — wire the coordinator to the parallel path

The shader, the NIF, the coordinator, the partition key, the padding and the
trim are all done. Nothing dispatches through them yet.

Set `:exmc_chain_coord` in the `Task.async_stream` path (`sampler.ex:119`),
starting and stopping a coordinator per multi-chain run. `route_chain` in
`tree.ex` already reads it.

**Before claiming a win, build the instrument.** The 3.4-4.3x is the
*in-dispatch term only* and what fraction of wall time it represents is
**unknown** — both estimates were withdrawn, one of them after three
consecutive runs of the same commit produced -125.0, -130.4 and 1554.5 µs for
the same quantity. super-io cannot resolve this: it is a desktop with a ~900 µs
noise band that manufactured a 13x error once already. mac-248 is headless and
resolves the chain-dispatch path at 0.3%.

Also still open: `bulkhead_test.exs` and `server_test.exs` share the unguarded
teardown pattern `chaos_test.exs` was fixed for.

### What NOT to do

**Do not touch the vectorized path.** `sampler.ex:1256` is "Phase 3: Sample all
chains sequentially (no XLA contention)" — chains run to completion one at a
time, so nothing to batch. Making it interleave is a refactor of the default
multi-chain loop reversing a decision made for EXLA, and it is not justified by
a Vulkan number.

**Do not size the payoff from the 3.4-4.3x alone.** That is the *in-dispatch
term only*. What fraction of wall time it represents is **unknown** — both
estimates were withdrawn. Build an instrument that resolves it before claiming
an end-to-end win.

---

## Host hazard — a swapped `.so` survives an ordinary rebuild

**Confirmed on mac-248, 2026-09-02.** After `.so`-swap benchmarking left an
artifact of unknown provenance in `deps/nx_vulkan/priv/native/`, the agreed
mitigation was to force `mix deps.compile nx_vulkan` before any run there.
**That does not work.** Cargo sees unchanged Rust sources, reports the crate up
to date, and Rustler never re-copies — the copy into `priv/native` happens only
when cargo produces a new artifact. Upstream demonstrated it by replacing the
`.so` with 25 bytes of text and watching a plain `mix compile` leave it in place
for 1622 failures.

Verified here rather than accepted: a clean rebuild on 248 changed the
checksum, `f1ab9681753dde0f -> 7cf6facf98be03e3`. **The artifact was stale and
my mitigation would not have replaced it.**

    # the only thing that actually replaces it
    mix deps.clean nx_vulkan --build && mix deps.compile nx_vulkan
    # or: rm -rf deps/nx_vulkan/native/nx_vulkan_vulkano/target

**Impact on measurements taken on 248 in that window: none that changes a
conclusion**, checked rather than assumed. The stale artifact was a `d210601`
build; it differs from the lock by the buffer pool, which pools allocations and
does not alter arithmetic — so the depth histogram (48/300 chains agreeing,
`n_steps` 1..16) stands, and the kinetic-energy A/B used the *same* binary in
both arms so its comparison is internally valid. The pure-Elixir microbenchmarks
never touched the NIF.

**Generalises past this incident:** any workflow that swaps a `.so` — benchmark
arms, a cross-built deploy, a bisect — leaves an artifact ordinary rebuilds will
not clear. Checksum `priv/native` before and after, always.

Also from upstream (`c2c7a5b`): `NXV_SKIP_NIF_BUILD=1` now disables the crate
build via `skip_compilation?`, so a Jetson verification is a ~2 minute
cross-build on super-io plus a short Elixir compile for **any** commit, not just
Rust-only ones.

---

## Status — 2026-09-02 (latest), the batching contract, and one method note

`gate1/reconcile-core` @ **`abfbe8e81`**.

### Differences of noisy measurements — the day's actual lesson

Three separate things died of this today, in both repos:

| claim | error | how |
|---|---|---|
| upstream's ~0.16 ms per submit-and-fence | **3x** | benchmark median minus benchmark median, across hosts |
| my "~4500 chain dispatches" | **9x** | remembered from a different workload, then used as a divisor |
| my in/out-of-dispatch split | **impossible** | marginal between two sample counts, giving negative time |

Every one produced a number that looked *more* precise than either operand it
came from. Subtracting two noisy quantities keeps both variances and discards
the scale that would have made the noise visible.

**Rule: do not difference two measurements to get a third.** Instrument the
thing directly, or A/B the whole workload with medians. Where a difference is
unavoidable, publish the repeatability of the difference before the difference
— running the same commit three times would have caught all three of these in
minutes.

### A second pattern, three instances in one day

**A constraint that was true when written and false by the time it mattered**,
each one documented confidently enough that nobody re-derived it:

* **Our 128-byte push cap.** Real arithmetic on bytes the GPU never receives.
  Cost: models past ~6 free Normal RVs silently routed to per-op sampling.
  13.1x when removed, and five green tests were defending it.
* **`nx_vulkan`'s glslang version pin.** Documented as load-bearing. 81 of 81
  shaders are byte-identical across 15.1.0, 16.2.0 and 16.5.0 — the generator
  word encodes the generator version, not the release.
* **`BatchCoordinator` partitioning on K.** Correct about the shader — a single
  K does parameterise the whole workgroup — and wrong about the conclusion,
  because padding was available. Written when no batched f64 NIF existed, so it
  reasoned about a capability nobody could exercise.

The shape: a true observation, a conclusion that followed from it *at the
time*, and no mechanism that notices when the premise moves. None of the three
would have been caught by a test, because each was consistent with the code as
it stood. What catches them is asking "is this still true?" of the constraints
you are about to design around — which is what all three of these cost.

### The batching contract, and the key that would have broken it

`nx_vulkan` shipped the f64 batched chain path (`bcfed0a`, bounds check
`ab16b24`). Measured on mac-248 at our depth histogram:

| chains | depths | serial us/draw | batched us/draw | speedup |
|---|---|---|---|---|
| 4 | 7,7,7,7 | 478.5 | 111.3 | 4.3x |
| 4 | 7,7,3,3 | 424.2 | 123.5 | 3.4x |
| 4 | 7,3,3,1 | 448.3 | 128.4 | 3.5x |
| 2 | 7,7 | 267.0 | 119.2 | 2.2x |
| 8 | 7x8 | 1143.6 | 119.3 | **9.6x** |

**Batched cost is flat in chain count** — the GPU runs the workgroups
concurrently, and at d=4 a single-instance dispatch occupies 4 of 256 threads.
More chains is nearly free.

**Scope, stated precisely because it is easy to overclaim:** this is 3.4-4.3x
of the *in-dispatch term only*. What fraction of wall time that term is was my
68-80% figure, now withdrawn along with the 1.9 ms it replaced. **The
end-to-end benefit is currently unknown.** What is established is upstream's
own measurement at our operating point: intercept 91.3 us against slope
2.5 us/step over K<=16, so 86% of a dispatch is fixed cost.

Contract: instances share one SPV, so priors and `d` must match; **K may
differ and is padded to the deepest**; inputs and outputs are instance-major;
`n_instances` is bounded by the device workgroup limit.

**`BatchCoordinator`'s key is wrong for this**, `batch_coordinator.ex:434`:

    {:erlang.phash2(meta), k, eps}

K is in the key, so chains of differing depth are partitioned into singletons
rather than padded. Against our measured histogram — 48 of 300 draws with all
four chains at the same `n_steps` — **it would batch fully in 16% of draws and
fragment in 84%**, which is most of the benefit gone. The comment justifying
it is not wrong about the shader; it concluded "partition" where "pad" was
available, and it predates any batched f64 NIF existing.

Fix, when the coordinator is wired: **drop `k` from the key** to
`{phash2(meta), eps}` and pad at flush. `d` is implicit in `meta`.

**A hazard to write before anyone codes the flush:** a padded instance runs
MORE leapfrog steps than it asked for. If chain A wants K=3 in a group padded
to K=7, its buffers come back with seven steps, and the coordinator must hand
back only the first three. The extra steps are computed from valid state, so
they are not garbage — they are trajectory the sampler never requested.
Getting this wrong produces a plausible wrong posterior, not an error.

**The property that makes the slice sound is now checked rather than assumed**
(`nx_vulkan` cccbd71). A K=7 dispatch's first n steps are **bit-identical** to
a K=n dispatch, on all four output buffers, verified at n = 1, 3 and 5 and
again through the batched path with an instance sliced out of a padded group.
It is a property of their shader so the test lives in their suite, which means
a future change to the step loop cannot silently break our flush.

`ab16b24` also verified on **unified memory**: 833 doctests / 884 tests / 0
failures on the Jetson, where `record_upload` and `record_readback` both take
their no-op branches — a genuinely different path from the discrete box every
other batched measurement came from.

### Our chains do not advance in lockstep

The reason the batched path is not reachable from the default:
`sampler.ex:1256` is **"Phase 3: Sample all chains sequentially (no XLA
contention)"** — `Enum.map(chain_states, &run_sampling/…)`. Chain 0 finishes
all its draws before chain 1 starts, so four chains never want a leapfrog at
once.

The **non-vectorized** path does: `sampler.ex:119` runs chains through
`Task.async_stream` with `max_concurrency: num_chains`, in separate processes,
hitting `Dispatch.chain/8` independently. They do not synchronise per draw and
do not need to — coalescing requests from independent processes on a
size-or-timer flush is exactly what `BatchCoordinator` was built for and has
never done.

So the work splits:

1. **Wire `BatchCoordinator` to the parallel path.** Tractable. Machinery
   exists, D1/D2/D3 fixed, callers already concurrent, and the batched NIF is
   the half it was waiting for. Needs the key fix and the padding slice above.
2. **Restructure the vectorized path to advance chains together.** A real
   refactor of the default multi-chain loop, reversing a decision made for
   EXLA. Not on the strength of a Vulkan number.

---

## Status — 2026-09-02 (later), the tree-logic split, measured — and D4 gets a number

`gate1/reconcile-core` @ **`4ba9e8a6a`**.

### The in-dispatch / out-of-dispatch split, measured rather than subtracted

`Dispatch.dispatch_micros/0` now accumulates time inside `chain/8` alongside
the counter, so both halves come from one run. `bench/tree_logic_split.exs`
takes the marginal between 200 and 800 draws, so warmup, compilation, shader
synthesis and BEAM startup cancel exactly.

**RETRACTED — the instrument cannot resolve anything. Do not use these
numbers.** They are left visible because the way they failed is the useful
part.

The first run looked clean and tight:

| d | disp/draw | us/draw | in-dispatch | out-of-dispatch | out % |
|---|---|---|---|---|---|
| 1 | 1.37 | 350.5 | 246.1 | 109.0 | 31.7% |
| 2 | 1.47 | 415.0 | 280.9 | 134.4 | 32.4% |
| 4 | 1.64 | 629.6 | 430.0 | 178.1 | 30.9% |
| 8 | 1.68 | 920.8 | 379.9 | 184.5 | 20.0% |

Three consecutive runs of the **same commit** on the same quiet box:

| d | run 1 | run 2 | run 3 |
|---|---|---|---|
| 1 | 111.5 | 123.2 | 180.3 |
| 2 | 138.2 | **-125.0** | **-130.4** |
| 4 | 83.4 | 177.2 | 161.1 |
| 8 | 148.5 | **1554.5** | 250.3 |

Negative out-of-dispatch is physically impossible — it is wall minus
in-dispatch — and d=8 spans 10x. The table above was one draw from that
distribution that happened to look tidy.

**The design is the fault.** Taking the marginal between 200 and 800 draws
cancels warmup, which is what it was for, but it differences two noisy
quantities and amplifies both variances. A direct A/B of the same workload
with medians spreads 9-19% on the same box, which is bad but at least finite.

So the "~1.9 ms of every 4.1 ms is tree logic" figure this file has quoted
remains unverified rather than corrected. Both it and its replacement are
now known to come from instruments that cannot support them. **What is
actually established is only that `disp/draw` is 1.4-1.7** — that number is
stable across every run above.

Building an instrument that resolves this is the prerequisite for any further
host-side work, and it is unfinished.

### Our trees are shallow, and that changes what is worth optimising

| model | n_steps max | mean | histogram |
|---|---|---|---|
| d=1 | 7 | 2.83 | `1:140 3:311 7:49` |
| d=4 | 7 | 5.44 | `1:3 3:191 7:306` |
| d=8 | 15 | 6.44 | `3:74 7:424 15:2` |

**K = 2^depth is 1 to 16, effectively never large.** `tree.ex` calls
`Nx.to_flat_list(Nx.slice(all_logp, [0], [n_steps]))` per dispatch, which
measures 8.5 us at K=32, 105 us at K=256 and **419 us at K=1024**, against
4.3 / 36 / 112 for a binary comprehension over `Nx.to_binary`. A 3.7x win — on
a term that costs about **2 us** at the K we actually run.

Right scaling curve, wrong operating point. Measure the operating point before
writing the patch, not after. (`Nx.backend_copy` to a backend a tensor is
already on costs ~1.1 us regardless of size, so the four of them in `tree.ex`
are also not worth touching.)

### The kinetic-energy change: written, verified, reverted

`cached_step_fn` computes `0.5 * sum(p * inv_mass * p)` as four Nx operations
on a d-element BinaryBackend tensor, per leaf. On mac-248 at d=4 the whole
leaf body is **21.84 us and that line is 14.3 of it — 63%**, for twelve
floating-point operations. Hoisting the row extraction to one
`Nx.to_flat_list` per dispatch and summing in plain Elixir measures
**21.84 -> 9.56 us per leaf, -56.2%**, with the hoist cost accounted
separately (1.7-12.2 us per dispatch, breaking even inside one leaf against
~3.3 leaves per dispatch) and posteriors **bit-identical to four decimal
places**, mean and sd, every parameter.

It was reverted (`7ad45d90d`). End to end it is ~4350 leaves x 12.28 us =
~53 ms against a 2200 ms run — **2.4%, below the 9-19% noise floor** of the
best instrument available — and the direct A/B point estimate came out 5%
*slower*. Unmeasurable benefit against a permanent branch plus a fallback for
rank-2 mass matrices.

Same standard applied to `nx_vulkan`'s buffer pool an hour earlier, and it
would have been inconsistent to keep this one. **Re-apply it if leaf count
ever rises** — deeper trees, larger K, or a model that stops being
4-dimensional — because the 56% is real, it is just currently multiplied by
too small a number.

### D4 now has a number attached: up to 3.9x

The batched chain path — `BatchCoordinator`, `synthesise_batched/1`,
`:exmc_chain_coord` — has been inert since it was written and this file has
carried it as "decide: wire it or retire it" with no way to price the decision.
It can be priced now.

`nx_vulkan` swept K on mac-248 at d=4, 2500 dispatches/sample, median of 5,
fitted over K<=16:

    K= 1   92.9 us      intercept  91.3 us
    K= 2   95.4 us      slope       2.5 us/step
    K= 4  102.8 us
    K= 8  111.8 us
    K=16  130.4 us
    K=32  167.4 us

At our d=4 mean of 5.44 steps: **91.3 us fixed + 14.9 us of steps — 86% of the
call is intercept.**

    4 chains serial     4 x (91.3 + 2.5*6)    = 424.7 us
    4 chains batched    91.3 + 2.5*7 (padded) = 108.7 us      ~3.9x

I measured whether our chains agree on depth, since batching pads to the
deepest: **48 of 300 draws agree exactly, 31.6% padding waste.** That sounds
disqualifying and is not — the waste lands on the slope, which is 14% of the
call, so padding costs ~2.5 us while collapsing three intercepts saves ~274 us.
**31.6% of 14%.**

At d=4 the shader also dispatches ONE workgroup of 256 threads with four doing
work — 1.5% occupancy. A batched `[n_instances, 1, 1]` runs the instances
concurrently, so batched wall time should be `intercept + slope * max(K)`
rather than `sum(K)`, and the padding may cost no wall time at all.

**The exmc half is D4.** The 3.9x is the ceiling on the nx_vulkan side; we
collect it only if we can prepare ONE batched call rather than four, which is
exactly what `BatchCoordinator` exists to do and has never been wired to do.
Three of four NIF intercepts go for certain; three of four Elixir-side
marshalling costs go only if the coordinator is real. **That is the strongest
argument yet for wiring D4 rather than retiring it**, and the first one with a
measurement behind it.

nx_vulkan needs an f64 batched NIF and template first — `leapfrog_chain_synth_batch`
exists but is f32-only. They are taking that to their user as a proposal.

### Correction: super-io inflated an intercept 3x

Their earlier intercept was **296.8 us on super-io against 91.3 us on 248** —
the same 2-13x inflation as every other per-dispatch figure from that desktop.
Had we not redone it at our operating point, the batching case would have been
built on a 3x overstated fixed cost.

Also worth retracting from my side: I told them "your side of the boundary is
spent, the 1.9 ms is the whole game", generalising the Jetson subtraction. The
248 split says in-dispatch is 68-80% of a draw. The 3%-of-ceiling result argues
we cannot fix per-dispatch cost by running **more** dispatches concurrently; it
was never an argument that per-dispatch cost does not matter. Reducing the
**count** attacks the majority term.

---

## Status — 2026-09-02, fleet green, and the ceiling that decides what is next

`gate1/reconcile-core` @ **`7284a57e6`**. `nx_vulkan` `d210601 -> 6d3a651`.

### Fleet at `ae2e1927a`, all three hosts

| host | before | now | what moved |
|---|---|---|---|
| mac-247 | 652 / **4** | 658 / **3** | `ChaosTest` flake gone |
| mac-248 | 652 / **3** | 658 / **3** | identical to 247, by name |
| Jetson (MAXN) | 652 / **7** | 658 / **4** | `ChaosTest` x2 + overhead gate gone |

Both Keplers now fail on exactly the same three: Poker, LevelSet, CustomDist,
all `SynthUnsupportedError` for **non-synthesisable IR, not width** — so the
push-cap change correctly does not touch them. The Jetson's three
disappearances are precisely the three things fixed that day. **No regression
anywhere from removing the cap.** 652 -> 658 reconciles exactly: +5
`push_width_test`, +2 `spv_recovery_test`, -1 from the P0 cap block.

### The number that decides the next piece of work

`nx_vulkan` measured concurrent dispatch on mac-248 — M processes each driving
`leapfrog_chain_synth_f64`, d=13, K=32, both arms, two rounds:

    M    us/dispatch     throughput
    1    172 / 169       5800-5900 disp/s
    2    135 / 135       ~7390
    4    136 / 136       ~7350
    8    136 / 136       ~7325

**The chain path saturates at ~7350 dispatches/s from M=2 onward.** Every
dispatch does `submit_and_wait` on a single queue, so two concurrent callers
fill it and more buy nothing.

Our sustained sampling run does **18550 dispatches in 76.1 s = 244/s — about
3% of that ceiling.**

Three consequences, and they are the clearest direction this file has had:

1. **The bottleneck is confirmed to be ours, from an independent direction.**
   The GPU-side decomposition already said ~1.9 ms of each ~4.1 ms dispatch is
   NUTS tree logic. This says the same thing without reference to that
   measurement: we are running at 3% of what the queue would allow.
2. **Parallelising chains is not the lever.** Throughput is flat past M=2 on
   this path. Adding concurrent samplers to buy dispatch throughput would be
   work spent against a ceiling we are nowhere near and could not raise.
3. **Further per-dispatch optimisation upstream has little left to give us at
   our current rate.** The fence fold was worth 18.2%; at 3% queue occupancy
   the remaining in-NIF cost is not what is holding us.

### A hypothesis of mine that measurement killed

I argued the buffer pool's global mutex might serialise our concurrent
dispatch — seven `async: true` modules call `Sampler.sample`, so contention is
real for us — and that a 1.3% win at M=1 could be a net loss at M=4.

**The mutex does not hurt at any M.** No scaling penalty in either direction,
despite 8 lock acquisitions per dispatch. What the measurement did find is that
the pool's advantage *disappears entirely* at M >= 2, because per-dispatch CPU
savings stop mattering once the queue saturates. So insisting on the
measurement was right and my mechanism was wrong — the arms are
indistinguishable everywhere except M=1. The pool has been dropped upstream on
those grounds rather than on the 2.2 us.

### Two corrections to figures this file has quoted

`nx_vulkan` re-measured its own super-io numbers on mac-248 and both were
inflated: the fence fold **-36% -> -18.2%**, the buffer pool **-17% -> -1.3%**.
The cumulative chain-dispatch cost on 248, arms chained:

    ab2e779           365 us
    096d7bd fast OFF  238 us    8cce91c    -35%
    096d7bd fast ON   224 us    b59c4a7     -6%
    f4c00f4           210 us
    8cd19ee           172 us    fence fold -18%
    d210601           170 us    pool       -1.3%   (now dropped)

**365 -> 170 us, about -53%**, with both large wins on the readback side —
which is what our own `3*K*d*8` down against `2*d*8` up asymmetry predicted.

The lesson attached to it is theirs and it is sharper than "super-io is noisy":
**a ~900 us noise band does not merely fail to resolve a small effect, it
manufactures a large one.** Treat every per-dispatch figure taken on super-io
as an upper bound.

Related, and it retires a documented constraint: **81 of 81 shaders are
byte-identical at glslang 15.1.0, 16.2.0 and 16.5.0.** The generator word
encodes glslang's generator version, not its release. The version pin was
documented as load-bearing and is not.

### Housekeeping

`mac-248`'s `deps/nx_vulkan/priv/native/libnx_vulkan_vulkano.so` was left dated
01:12 by upstream's `.so`-swap benchmarking. The git tree there is clean and at
`ae2e1927a`, but **force `mix deps.compile nx_vulkan` before any run on that
box** rather than trusting the artifact matches the lock. Same trap as
deploying with `--no-compile`.

---

## Status — 2026-09-01 (later), the push cap: 13.1x from deleting a guard

`gate1/reconcile-core` @ **`ae2e1927a`**. `nx_vulkan` `6b38aee -> d210601`.
super-io Vulkan arm: **658 tests, 2 failures** — the cleanest this branch has
been. Both remaining are pre-existing and characterised: `LevelSet` (300s
timeout) and a `ValidatorTest` Cauchy KS at `d=0.0999` against `crit=0.0975`,
confirmed by reverting to HEAD and reproducing it. **Poker now passes here.**

### The push-constants cap was measuring bytes nothing read

`Push.pack/1` built a 24-byte header plus one f64 per prior parameter and
rejected the result past 128 bytes, degrading the model to per-op sampling.
That capped models at ~13 free RVs with one-parameter priors, 6 with Normal, 3
with TruncatedNormal.

The tail was read by nothing. `MultiRvCustomSpec` bakes prior parameters into
the GLSL as literals — disassembling a cached SPV for `Normal(0.0, 7.3125)`
shows `OpConstant %double 7.3125` and its precomputed normalisation term, in a
push struct of exactly `OpTypeStruct %uint %uint %uint %uint %double`. And
`leapfrog_chain_synth_f64` pushes `sizeof(PushBlockF64) = 24` and drops the
rest. **But the NIF rejects `push.len() > 128` before dispatching**, so the
unread tail was counted against a budget it never spent.

| 8-RV conjugate model | dispatches | wall |
|---|---|---|
| with the tail | **0** | 160.9 s |
| header only | **2564** | 12.3 s |

**13.1x**, posterior unchanged and correct against the closed-form conjugate on
both arms. Negative control: restoring the tail makes `detect_meta` refuse the
model again and `push_width_test.exs` go 5/5 -> 3 failures.

Since wide models are now reachable, synthesis refuses `d > 256` — the shader's
actual bound (`local_size_x = 256`, `q_shared[256]`) — so it degrades rather
than reaching `do_chain`'s single guarded clause and raising a
`FunctionClauseError` naming nothing.

### Five passing tests were defending it

This is the part worth keeping.

Four sat in a describe block named **"chain shader: the push block caps model
width at 13 prior floats"**, carrying a measured table — `d <= 13` for
one-parameter priors, 6 for Normal, 4 for StudentT, 3 for TruncatedNormal —
and a comment saying the number "changes what the synthesis path is FOR". A
fifth, `PushFallbackTest`, asserted that a ten-parameter model *correctly*
degraded to per-op. All five were green at HEAD.

They were not testing a limit. They were pinning a defect, and pinning it
precisely enough to look thoroughly established. **A passing test can be
evidence for a defect.** All five are rewritten to assert the corrected
behaviour and kept as regressions: reinstating the tail fails them.

Four doc sites asserted the wrong bound and all had to move together —
`push.ex`'s moduledoc, `tree.ex:722`, `dispatch.ex:78`, `compiler.ex`. Each
said `d <= 256` was "never the binding one". It is the only binding one.

### A vanished shader is now rebuilt, not fatal

`Dispatch` holds an `spv_path` from synthesis and hands it to the NIF every
call. If the file disappears in between, the NIF returns `{:error,
:dispatch_failed, "read spv: No such file or directory"}` and the `{:ok, _} =`
match raises inside `Tree.do_build`, in whatever test is running.

Two unrelated causes so far: this module's own shared temp paths (fixed
earlier), and **`Nx.Vulkan.Synthesis.clear_cache/0` doing `File.rm_rf` on
`~/.exmc/gpu_node/spv`** — the two projects shared that directory, from that
project's `setup` AND `on_exit`. It deleted a shader out from under a
full-suite run here. nx_vulkan has since moved its caches under
`~/.nx_vulkan/` (b024ad1, verified with sentinel files).

`Compile.compile_glsl/1` now remembers each source against its hash and
`ensure!/1` rebuilds from it; dispatch retries once. Rebuilding is always
correct — the GLSL is deterministic and the hash IS the filename. This removes
the class, not one cause: eviction, partial writes, an operator `rm` and a
restored home directory all reach the same state.

Negative control, deleting the shader **during** a run: without the retry,
`MatchError ... "read spv"` after **1 dispatch**, twice; with it, 600 draws and
804 dispatches, posterior unchanged, twice.

### Two test fixes, and what each cost to get right

**`ChaosTest`'s teardown race** — six sightings, three hosts, five test names.
ExUnit exits the test process with `:shutdown` when the body returns, reaping
everything `start_link`ed in `setup`; `on_exit` then runs from a *different*
process, so `Process.whereis/1` can return a pid dead microseconds later. The
old teardown was `for name <- names, do: GenServer.stop(...)`, and a
single-generator comprehension compiles to `Enum.map/2` — which is why the
stack frames ran through `enum.ex:1714`, and why the first dead pid also
aborted cleanup of every name after it. Deterministic control: 5 failures -> 0;
12/12 clean after.

**`FaultTolerantTest`'s overhead gate** — failed on the Jetson only after MAXN,
because overhead is a *ratio* and a faster box shrinks the denominator faster
than the numerator. The measurement found worse: seven alternating pairs of the
same runs read -5.7, -1.5, +19.3, -11.8, +21.6, +3.4, +0.3 percent. **A +-20%
instrument cannot adjudicate a 10% claim on any host.** Replaced with BEAM
reductions and a same-run control arm (`supervised: :task`, a real
implementation that spawns per subtree), normalised per leapfrog step.

That rewrite then failed in the full suite while passing 8/8 in isolation —
`:task` diverged from the unsupervised trajectory under load. A load-sensitive
assertion inside the test written to remove load-sensitive assertions, and only
a contended full-suite run could show it.

### Four controls caught four things review did not

Worth listing together, because the hit rate is the argument:

1. The `ValidatorTest` Cauchy failure looked like mine; reverting to HEAD
   reproduced it. Pre-existing.
2. My first shader-recovery test deleted the artifact *before* sampling. That
   proved nothing — `Sampler.sample` re-synthesises and rebuilds it first — and
   the negative control **passed with the fix reverted**, which is the only
   reason it was caught.
3. My closing `assert File.exists?(spv)` was itself a race: the killer's last
   `rm` can land after the last dispatch. It failed while the recovery it
   tested had worked perfectly.
4. The `:task` divergence above, caught only by the full suite.

---

## Status — 2026-09-01, the Jetson at MAXN — and the count lies a third time

**The Jetson's power mode was never recorded in any baseline.** It ran at
nvpmodel 5W: **2 of 4 cores**, CPU capped 918 MHz, GPU capped 640 MHz —
roughly a third of the board. `/etc/nvpmodel.conf` declares `DEFAULT=0`
(MAXN), so 5W came from a persisted `/var/lib/nvpmodel/status` and had been
there for as long as anyone had run tests on that host. Every Jetson number in
this file above this line was taken that way and none of them said so.

Switched to MAXN (4 cores, 1479 MHz, GPU 921.6 MHz) and re-ran the identical
arm. The suite now logs `nvpmodel`, `nproc`, `cpu_max` and `gpu_max` in its own
header, because "trust me, it was MAXN" is the class of claim that produced this
whole section.

| | 5W | MAXN |
|---|---|---|
| commit | `4d1c4800d` | `c01ee78e7` |
| nx_vulkan | `6b38aee` | `096d7bd` |
| result | 652 tests, **7 failures** | 652 tests, **7 failures** |
| wall | 6054 s | **3802 s** |

### Seven and seven, and almost nothing in common

| test | 5W | MAXN |
|---|---|---|
| `NewDistTest` Lognormal mean | FAIL | FAIL |
| `PokerTest` parameter recovery | FAIL | FAIL |
| `LevelSetIntegrationTest` 6x6 | FAIL | FAIL |
| `IntegrationTest` large model | FAIL | FAIL |
| `MCLMCTest` EEVPD target | FAIL (timeout) | **cleared** |
| `MCLMCTest` raising EEVPD | FAIL (timeout) | **cleared** |
| `NUTSTest` Leapfrog 4 momentum | FAIL (timeout) | **cleared** |
| `FaultTolerantTest` supervised overhead | pass | **NEW — assertion** |
| `ChaosTest` suicide window, age out | pass | **NEW — flake** |
| `ChaosTest` suicide window, emergency brake | pass | **NEW — flake** |

**Three timeouts cleared, three new failures appeared, and the total did not
move.** This is the third time today a failure count has been misleading and it
is the worst of the three: on mac-247 the count stayed at 4 while composition
changed; on the Jetson's EXLA arm the population itself shrank; here the count
is *identical* and seven of ten rows differ. A total is not a result.

The three that cleared were all `ExUnit.TimeoutError`. That part was predicted
in advance and held: those tests were starved, not slow.

**The three that appeared are the interesting half, and they were not
predicted.**

* `FaultTolerantTest` — "Supervised overhead 10.7% exceeds 10% threshold". A
  real assertion, not a timeout. Supervision overhead is a **ratio**, and a
  faster box shrinks the denominator faster than the numerator, so the test
  gets *harder* as the machine improves. It has presumably been passing on the
  Keplers and here for the same reason the wall-clock test used to "pass"
  somewhere: the hardware happened to sit on the friendly side of a threshold.
* `ChaosTest` x2 — the same `on_exit` `GenServer.stop` on an already-dead pid,
  `(EXIT) no process`, after the assertions passed. **Fifth and sixth
  sightings**, now on a third host and two further test names. It is not
  host-specific and never was; four cores simply surface the teardown race more
  readily. This needs fixing rather than re-observing.

### Do not read the 37% as a power-mode number

6054 s -> 3802 s crosses **two** variables: the power mode and `nx_vulkan`
`6b38aee -> 096d7bd`, which is independently worth ~39% per chain dispatch on
Kepler. A clean power-mode figure needs `4d1c4800d` rebuilt at MAXN. The
failure composition above is unaffected by the bump and is the comparable half.

### Chain-dispatch cost, third architecture

`bench/chain_dispatch_cost.exs`, N=3000, warmup 3000, 6 replicates:

| host | GPU | per-dispatch | spread |
|---|---|---|---|
| mac-248 | GT 750M, headless | **365 us** | 0.3% |
| Jetson | Tegra X1, MAXN, unified | **2225 us** | 2.5% |
| super-io | RTX 3060 Ti, live desktop | 822-1741 us | 57% — unusable |

The Jetson is 6.1x the 750M but tight, so it is a real number. `staging path:
OFF` confirms the unified branch.

### The GPU-utilisation scare: the instrument was fine, the operator was not

tegrastats showing `GR3D_FREQ 0%` during a Vulkan run looked like evidence that
the arm was secretly CPU. Under *confirmed* dispatch it reads **53-58%**, with
sysfs `gpu load: 514`/1000. `scripts/tegrastats_bars.sh` parses it correctly.

Four sampling attempts preceded that answer and every one measured something
else: a process blocked on a build lock; a host whose checkout predated the
bench file; a run still inside `mix compile`; and a run where `EXMC_COMPILER`
was inert. A gate checking for `replicate 1:` in the log existed the whole time
and the sample was printed regardless of whether it fired. **Writing the check
and not letting it gate anything is the failure this whole document is about.**

Two real defects fell out:

* `bench/chain_dispatch_cost.exs` told the reader to "re-run with
  EXMC_COMPILER=vulkan", which `config/test.exs` honours only under
  `MIX_ENV=test`. Under `mix run` it was inert. It appeared to work on super-io
  (EXLA unusable without `LD_LIBRARY_PATH`) and mac-248 (no EXLA) because
  auto-detect fell through to Vulkan anyway. On the Jetson it selected EXLA and
  the file's own compiler guard refused. Fixed in `c01ee78e7`.
* `scripts/tegrastats_bars.sh` (nx_vulkan) extracts GPU/EMC clock only from a
  `%@<freq>` suffix this board never emits, so those two bars **silently never
  render**. Reported upstream.

Also: `glslangValidator` lives in `~/.local/bin` on the Jetson. Omit it from
`PATH` and synthesis fails with a bare `:enoent` and `detect_meta/2` returns
`:unsupported` — which reads as "this model cannot be synthesised" rather than
"the compiler binary is missing".

### Where the time actually goes: the GPU is ~29% busy, and that is the story

The MAXN suite nearly halving (6054 -> 3802 s) prompted the right question — a
GPU-bound workload should not care much about doubling CPU cores. Measured, on
a sustained `Sampler.sample` loop with the gate confirmed open first:

    GR3D_FREQ:  34 34 34 33 25 30 33 27 28 0 27 41 25 27 34 27 33 34 27   mean ~29%
    18550 dispatches in 76.1 s   ->  4.1 ms of wall per dispatch
    isolated bench               ->  2.2 ms inside Dispatch.chain

Decomposing one dispatch:

| | per dispatch | share |
|---|---|---|
| GPU actually executing | ~1.2 ms | ~29% |
| CPU **inside** the NIF call — marshalling, alloc, submit, fence | ~1.0 ms | ~25% |
| CPU **outside** it — NUTS tree logic, scalar Nx, BEAM | ~1.9 ms | ~46% |

**About 70% of a sampling run is CPU**, and the suite figure is lower still
because many tests never sample. That is why more cores and more clock nearly
halved it. The GPU is not being bypassed — it is being starved.

The `nx_vulkan` session measured the same thing from the other end, on a 32-step
f64 Normal leapfrog at d=13:

    fused chain shader    0.469 ms     1 dispatch
    per-op via Nx        76.658 ms   ~224 dispatches      163x
                                      ~342 us per dispatch

At d=13 the real GPU work per op is nanoseconds inside ~342 us of host and
driver overhead, so utilisation averages toward zero while the card waits to be
fed. **The fix for the per-op path is fusion or batching, not widening a gate.**

**Consequence for optimisation effort:** the readback batching (`8cce91c`,
4 fences -> 1) attacks the ~1.0 ms CPU-inside-the-call half, not the ~1.2 ms of
GPU compute. That is the right half to attack. But neither Kepler has had those
two halves separated — the 365 us on mac-248 and the 2225 us here presumably
decompose the same way, and the split has only been measured on the Jetson.

### The per-op path does NOT host-fall-back — and how I nearly got that backwards

Confirmed from both sides. A real `Sampler.sample` on the chain path reports
**53 dispatches, 0 host fallbacks**. So "Vulkan arm" on this box really is
Vulkan; the earlier worry that Jetson Vulkan numbers contained an unknown
quantity of CPU-via-fallback is dead. They contain a lot of CPU, but it is BEAM
and marshalling, not silent backend fallback.

Getting there took a retraction. I ran the census, got 0, ran a "positive
control" that also returned 0, and concluded `Nx.Vulkan.Fallback.count/1` was
inert — one message away from telling the upstream session their instrument was
broken and their own 0-fallback result needed rechecking.

**Both zeros were mine.** I had set `Application.put_env(:exmc, :compiler,
:vulkan)` — exmc's compiler selection — and not `Nx.default_backend/1`. Every
tensor was built on `BinaryBackend`. Nothing was ever on the GPU, so nothing
could fall back, and the counter was correctly answering a question I had not
meant to ask.

With the backend actually set, on the Jetson at `096d7bd`:

| op | result backend | counter |
|---|---|---|
| `pow(t, 2.0)` | **BinaryBackend** | 1 `[pow: 3]` |
| `pow(t, t2)` same-shape | VulkanoBackend | 0 |
| `add(t, 1.0)` | VulkanoBackend | 0 |
| `sort(t)` | **BinaryBackend** | 1 `[sort: 3]` |
| `exp(t)` | VulkanoBackend | 0 |

Which independently reproduces upstream's `pow_ok_bcast?` finding on a third
architecture: broadcast `pow` leaves the GPU because the broadcast elementwise
shaders have no `pow` arm, while same-shape `pow` stays. `sort` is a second one.
Weibull is the family in the f64 set whose per-op form needs `pow`.

**The rule this broke was already written in this file:** a control that fails
to trigger is not evidence the instrument is broken. It is evidence of nothing
until the control is shown able to trigger. Mine could not, because the setup
was wrong upstream of the thing being tested. Verify the control fires before
trusting a null — the same discipline as the non-vacuity guards in
`chain_meta_routing_test.exs`, applied to ad-hoc measurement.

**Tally for the day, since the pattern is the point:** the GPU-utilisation
question alone took six invalid measurements before a valid one — four
tegrastats samples of a blocked, stale, still-compiling or wrong-arm process,
then two fallback censuses on tensors that were never on the GPU. Every one
produced a plausible number. None measured what its label said.

---

## Status — 2026-08-31 (latest), nx_vulkan ab2e779 — and a retracted measurement

`gate1/reconcile-core` @ **`f680412a3`**. `nx_vulkan` `6b38aee -> ab2e779`.

### Fleet at `f680412a3`

| host | @ `4d1c4800d` (6b38aee) | @ `f680412a3` (ab2e779) |
|---|---|---|
| mac-247 | 652 / **4**, 960s | 652 / **4**, 932s |
| mac-248 | 652 / **3**, 576s | 652 / **3**, 529s |

Composition identical by name on both: Poker, LevelSet, CustomDist
(`SynthUnsupportedError`), plus the roaming `ChaosTest` teardown flake on 247 —
**fourth sighting, fourth test name** (`cross-shader suicide window old timeouts
age out of the window`). It has now appeared on both hosts under four different
test names with one signature. Fix the `on_exit`; stop reading it as signal.

Suite time improved on both, 960 -> 932 and 576 -> 529.

### The chain-dispatch measurement is withdrawn

Do not cite any per-dispatch number this branch has produced, including the ones
already sent upstream.

Matched conditions (both arms run immediately after a full suite, same box):

| | probe, 8 tests |
|---|---|
| `6b38aee` | 4.0 4.1 4.1 |
| `ab2e779` | 4.4 4.5 4.4 4.5 4.5 |

The warm/cold confound was real but small — the post-rebuild spread (4.4-5.0,
median 4.6) collapses to 4.5 once machine state is matched — and a ~0.4s gap
survives it. That is not the null the code predicted, and there is a candidate
explanation: `ab2e779` is not a no-op relative to `6b38aee` on the **upload**
side. At `6b38aee` `upload_buffer` was still a plain host-visible write with no
submission; at `ab2e779` it is a device-local alloc + staging buffer + recorded
copy. The fences are gone, the staging allocation is not. Our uploads are
`2*d*8` bytes with d <= 13 — about 200 bytes — so there is no VRAM-residency win
to offset it.

**But the instrument cannot support that conclusion, and here is why.** The
probe drives **~520** chain dispatches, not the ~4500 previously claimed:

| workload | dispatches |
|---|---|
| vectorized, 4 chains, 50+50 | 64 |
| sequential, 4 chains, 50+50 | 64 |
| `sample_compiled`, 100 samples | 126 |
| `sample_compiled_tuned`, 100 samples | 132 |
| `sample_stream`, 50+50 | 89 |

The ~4500 figure came from the `chain_meta` fix-verification runs, which used
500-sample workloads; the probe tests use 50 and 100. It was quoted from memory
after being flagged as load-bearing, and it was passed upstream, where it was
used to size their proposed mechanism. Corrected, their ~2.9s prediction becomes
~0.33s against ~1.0s observed — the discrepancy inverts from 3x under to 3x
over.

The real problem: ~520 dispatches inside a ~4s measurement means the probe is
dominated by BEAM boot, compilation, GLSL synthesis and backend init, not by
dispatch. A sub-millisecond per-dispatch effect cannot be resolved in it. The
`amplify` advice was given, and accepted, and not actually followed.

**To measure this properly**, a standalone microbenchmark driving the f64 chain
NIF in a loop: tens of thousands of dispatches, startup excluded, warmup
discarded, n replicates with arm order rotated. Offered upstream; build it
before their four-downloads-into-one-fence change lands, so the instrument
exists before the change does.

### The download mechanism, verified in source

The `2617e5e -> 6b38aee` regression is **downloads, not uploads** — an earlier
attribution in this file was wrong and is retracted. Verified by `git show` on
both revisions:

* `2617e5e` `download_buffer` is `buf.read()` in place; `grep -c "fn
  staging_read"` returns **0** — the function does not exist at that revision.
* `6b38aee` `download_buffer` delegates to `staging_read`, which does
  `copy_buffer` + `submit_and_wait`.
* `6b38aee` `upload_buffer` still carries `PREFER_DEVICE |
  HOST_SEQUENTIAL_WRITE` and no submission — so uploads contributed **zero** to
  that interval.
* The chain NIF makes **four** `download_buffer` calls per dispatch —
  `q_chain`, `p_chain`, `grad_chain`, `logp_chain`.
* `git diff 6b38aee..ab2e779` touches neither `download_buffer` nor
  `staging_read`.

So the chain path went 0 -> 4 fences per dispatch on readback and is still at 4.
Our own design note is why it lands hardest here: `3*K*d*8` down against
`2*d*8` up. The upstream fix (four staging reads into one command buffer, 4
fences -> 1) is the one that will matter for us.

---

## Status — 2026-08-31 (later), nx_vulkan 6b38aee — the staging-buffer bump

`gate1/reconcile-core` @ **`4d1c4800d`**, pushed. `nx_vulkan` `2617e5e ->
6b38aee`.

Not a routine dep roll. `native/nx_vulkan_vulkano/src/lib.rs` gained ~299 lines
that change the buffer memory model: compute buffers were `PREFER_DEVICE |
HOST_RANDOM_ACCESS` and are now **`PREFER_DEVICE` alone**, with every host read
or write going through a new `staging_read`/`staging_write` copy via a
`PREFER_HOST` buffer. It is gated at runtime on a unified-memory probe.
`download_buffer` is on that path, which is how exmc reads every chain-dispatch
result back — so this can change results, not only speed.

The probe splits the fleet, and the NIF prints its verdict at init:

| host | probe line |
|---|---|
| super-io | `unified memory: false (staging path: ON)` |
| mac-247 | `unified memory: false (staging path: ON)` |
| mac-248 | `unified memory: false (staging path: ON)` |
| the Jetson | `device: NVIDIA Tegra X1 (nvgpu) (IntegratedGpu)`, `unified memory: true (staging path: OFF)` |

### Results at `4d1c4800d`

| run | @ `a178a0833` | @ `4d1c4800d` |
|---|---|---|
| super-io EXLA | 652 / **0**, 449s | 652 / **0**, 566s |
| super-io Vulkan | 652 / **2**, 1217s | 652 / **1**, 1223s |
| mac-247 | 652 / **4**, 883s | 652 / **4**, 960s |
| mac-248 | 652 / **3**, 521s | 652 / **3**, 576s |

**No regression from the bump.** Failure composition on the Keplers is
identical by name: Poker, CustomDist and LevelSet (all
`SynthUnsupportedError`), plus the roaming `ChaosTest` teardown flake, which
landed on 247 this time as `different metas track independently` — the exact
test that failed on **248** at `3e58fb70f`. Third sighting, third
host/test combination, same `on_exit` `GenServer.stop` on a dead pid. It is one
flake, and it moves.

super-io's Vulkan arm went 2 -> 1: **Poker passed.** Do not read that as the
bump fixing anything. Poker is recorded below as borderline on this host —
passes sometimes, times out at 300s otherwise — and one green run is exactly
what "borderline" predicts. LevelSet still times out at 300s.

### The Jetson: a bare `mix test` there tests the WRONG ARM

Worth reading before anyone runs the Jetson again, because the result looks
green and is not comparable.

`Exmc.JIT.detect_compiler/0` resolves EXLA > Vulkan > Evaluator, and **the
Jetson has EXLA**. So a bare `mix test` there selects the EXLA arm, exactly as
it does on super-io — unlike the Keplers, where bare `mix test` is the Vulkan
arm because there is nothing else. Every historical Jetson number in this file
is a `EXMC_COMPILER=vulkan` run.

It does not merely swap the backend. `test/test_helper.exs:148` flips the
exclusion set with it:

| arm | excludes |
|---|---|
| `Nx.Vulkan` | `:diag, :slow, :vulkan_known_failure` — 5 excluded, 2 skipped |
| anything else | `:diag, :slow, :requires_vulkan` — **31 excluded** |

So the unforced run drops the Vulkan-only tests entirely. On the first attempt
at `4d1c4800d` this showed as **652 tests / 6 failures** against a baseline of
640/8 — and four of the "fixed" failures (`PokerTest`,
`LevelSetIntegrationTest`, and both `IntegrationTest` cases) **were excluded,
not fixed**. Same trap as the mac-247 count, one layer deeper: there the
composition changed under a constant count; here the population itself changed.
Check the `exmc: compiler=` line the suite prints before comparing anything.

The unforced arm did establish two things worth keeping:

* The probe is right on the unified host —
  `[nx_vulkan_vulkano] device: NVIDIA Tegra X1 (nvgpu) (IntegratedGpu)` then
  `unified memory: true (staging path: OFF)`. The fleet splits 3-1 across the
  gate exactly as designed.
* Its 6 failures are **all `ExUnit.TimeoutError` at 60s**, none an assertion
  failure or a Vulkan error. Four (`NewDistTest` Lognormal mean, both
  `MCLMCTest`, `NUTSTest` Leapfrog 4) time out on the Vulkan arm too, in every
  Jetson log going back to `ad464bce5`. Two — `StepSizeBoundsTest:91` and
  `HalfNormalTransformTest:83` — are EXLA-path timeouts with **no EXLA-path
  baseline on this host**, so whether they are new is unknown, not benign.

Corrected baselines, from the logs actually on the box (neither records its own
commit, so attribution is by filename and count match, not by content):

| log | commit | result |
|---|---|---|
| `/tmp/jetson501.log` | `ad464bce5` | 632 tests, 9 failures, 6607s |
| `/tmp/jetson3e5.log` | `3e58fb70f` | **640 tests, 8 failures**, 6581s |

The 632/9 figure carried above was one commit stale. The Vulkan arm at
`4d1c4800d` is running now, forced, and 640/8 is what it should be read
against.

### The one thing worth chasing: the staging path may cost ~10% on the Keplers

Suite time rose on both Vulkan-only hosts while super-io stayed flat:

| host | @ `a178a0833` | @ `4d1c4800d` | delta |
|---|---|---|---|
| mac-247 | 883s | 960s | **+8.8%** |
| mac-248 | 521s | 576s | **+10.4%** |
| super-io Vulkan | 1217s | 1223s | +0.5% |

A narrower probe agrees in direction. `chain_meta_routing_test.exs` +
`sample_stream_test.exs` on 247 — 8 tests, pure chain-dispatch — ran **4.0,
4.1, 4.1s** at `4d1c4800d` against **3.0s** at `a178a0833`.

**Treat the magnitude as unestablished.** The `3.0s` baseline is a single
reading; only the new numbers were repeated. What is solid is the direction and
that it appears on both Keplers and not on super-io, which is what a
host-boundary copy would look like on older PCIe.

**Do not spend a run on this yet.** The `nx_vulkan` session has since read its
own code and expects its *next* commit (`d7b5f08`) to make this path worse
again, with a named mechanism: `leapfrog_chain_synth_f64` calls
`upload_buffer` three times (`q_init`, `p_init`, `extras`), and where each was
once a plain host write, each becomes a device-local alloc + staging buffer +
copy + `submit_and_wait` — three extra submit-and-fence pairs per chain
dispatch, ~0.076 ms each on Ampere and worse on Kepler submission. Our f64
chain dispatch is precisely the caller that pays it. Their intended fix is to
record those copies into the synth NIF's own command buffer ahead of the
dispatch it already submits, so they cost nothing extra.

They are running a controlled A/B on super-io first, because it is the only
box in the fleet where GPU clock can be recorded — FreeBSD reports `[N/A]` for
`clocks.sm` on the Keplers, and unrecorded clock state alone swung one of their
super-io measurements 2.6x. **The Kepler re-measure is worth doing after that
A/B lands, not before.** If a future measurement here comes back slower, the
default explanation is the input path, not anything in exmc.

---

## Status — 2026-08-31, six defects closed and fleet-verified

Written against `gate1/reconcile-core` @ `a178a0833`, **pushed to `origin`**.
Both Keplers verified at that commit after the reboot; the Jetson is not.

The session ran `nx_vulkan` `501fa08 -> 2617e5e` and then closed six defects.
Five were silent. The sixth had been on this file's own known-failures list
since 2026-08-16 as "item 4, the wall-clock one", carried for two weeks as a
scheduling artifact. It was not one.

| commit | what |
|---|---|
| `7d40ce49c` | `nx_vulkan` `501fa08 -> 2617e5e` |
| `0ff2d866c` | D3 + D1 — coordinator died on a raise; batched path packed 5 of 12 priors |
| `3e58fb70f` | Validator pinned an `:exla` reference arm it could not run, and leaked it |
| `237c0c3f8` | Vectorized (DEFAULT) multi-chain path never reached the chain shader |
| `edbeb2713` | D2 — batched push had no 128-byte cap |
| `a178a0833` | Distributed and streaming paths dropped `chain_meta` too |

### Verification as it actually stands

| run | result |
|---|---|
| `mix test` (EXLA), @ `a178a0833` | 652 tests, **0 failures**, 449s |
| `EXMC_COMPILER=vulkan mix test`, @ `a178a0833` | 652 tests, **2 failures**, 1217s |
| mac-247, @ `a178a0833` | 652 tests, **4 failures**, 883s |
| mac-248, @ `a178a0833` | 652 tests, **3 failures**, 521s |
| the Jetson, @ `ad464bce5` | 632 tests, **9 failures** — 7 of them ExUnit timeouts |

**The EXLA arm is fully green for the first time.** The Vulkan arm's two are
Poker and LevelSet, both 300s `ExUnit.TimeoutError` on this host, both
confirmed pre-existing: Poker was re-run alone against HEAD with the session's
changes reverted and timed out identically at 300.4s.

**Both Keplers verified at `a178a0833` on 2026-08-31 21:15–21:34 UTC**, after
the vulkan race was terminated. Logs: `~/fleet_a178a0833.log` on each; the
`3e58fb70f` baselines they are compared against are `/tmp/fleet3e5.log`.

### What changed on the fleet, test by test

Counts alone would mislead on mac-247 — 4 failures before, 4 after — so the
comparison is by name.

| test | 247 @ `3e58fb70f` | 247 @ `a178a0833` | 248 @ `3e58fb70f` | 248 @ `a178a0833` |
|---|---|---|---|---|
| `IntegrationTest` vectorized-chains wall-clock | FAIL | **gone** | FAIL | **gone** |
| `ChaosTest` eviction policy | pass | FAIL (flake) | FAIL (flake) | pass |
| `PokerTest` parameter recovery | FAIL | FAIL | FAIL | FAIL |
| `CustomDistTest` NUTS sampling | FAIL | FAIL | FAIL | FAIL |
| `LevelSetIntegrationTest` 6x6 grid | FAIL | FAIL | FAIL | FAIL |

**The wall-clock failure is gone on both**, so `237c0c3f8` does on the Keplers
what it did here. Its replacement was re-run on its own on mac-247 together
with the other new guards: `chain_meta_routing_test.exs` +
`sample_stream_test.exs`, **8 tests, 0 failures, 3.0s**. `:requires_vulkan` is
not in the Keplers' exclusion list, so these are the first runs of the
dispatch-count guards on a **Vulkan-only** host — super-io's Vulkan arm is a
forced override of an EXLA auto-detect, so it could not have shown this.

**`ChaosTest` swapped hosts, and that is the point.** It is one flake, not a
regression: identical signature both times — `on_exit` calls
`GenServer.stop(pid, :normal, :infinity)` on a process that has already died,
`(EXIT) no process` — but a *different test within the module* each time
(`different metas track independently` on 248 at baseline, `a successful
dispatch resets the counter` on 247 now). It failed at `3e58fb70f` too, so it
predates this session. The teardown wants to tolerate a dead pid; it is a test
defect, not a product one.

**Poker, CustomDist and LevelSet all fail identically on both Keplers** with
`Exmc.SynthUnsupportedError` — not a timeout, unlike super-io. So the "fails
FAST on the Keplers" note below now has its reason: these hosts have no EXLA to
fall back to, so a non-synthesisable IR raises at compile rather than crawling.
Three models the Vulkan-only arm cannot run. Unchanged by this session.

Suite time fell despite twelve more tests — 949s -> 883s on 247, 554s -> 521s
on 248, about 6-7% on both. Consistent with the `chain_meta` fixes, though the
suite is dominated by work those paths do not touch, so this is corroboration
and not a measurement.

`mix deps.get` returned `GET_EXIT=0` on both with no seeding fetch — second
unattended confirmation of `allowReachableSHA1InWant`.

### Do this first

1. ~~Fleet-verify `a178a0833` on mac-247/248.~~ **Done, 2026-08-31** — see the
   table above. The prediction written here beforehand said "expect 646 tests
   and 4 -> 3 on 247"; the real numbers were **652** and **4 -> 4**, and both
   misses were informative rather than cosmetic. The count was wrong because
   twelve tests were added, not six. The failure count on 247 held flat because
   the wall-clock test going green was masked by a pre-existing `ChaosTest`
   teardown flake landing on that host this time. **Compare failures by name,
   not by count.**
2. **The Jetson has not been verified since `ad464bce5`.** Three commits behind
   the Keplers. It needs `PATH=/home/io/.asdf/shims:/home/io/.cargo/bin:/home/io/.local/bin`
   and `CXX=g++-13 CC=gcc-13`, and takes ~1.8h.

### What is open

* **D4 — the batched chain path still cannot run.** Nothing sets
  `:exmc_chain_coord`; `synthesise_batched/1` has no callers; the f64 batch NIF
  does not exist in `nx_vulkan`. D1/D2/D3 are closed, so this is the last of the
  four in `docs/BATCHED_CHAIN_DISPATCH.md`, and it is a decision — wire it or
  retire it — not a fix. See that file's Option 1 vs Option 2.
* **Poker on super-io is borderline.** Passes sometimes, times out at 300s
  otherwise, and times out on the Jetson. It is the only thing between this
  branch and a clean Vulkan arm here. The Keplers' *fast*
  `SynthUnsupportedError` is no longer the mystery it was listed as: those
  hosts have no EXLA, so a non-synthesisable IR raises at compile time instead
  of falling back. Whether Poker's IR *should* be synthesisable is the open
  question, and it is the same question on all three hosts.
* **`ChaosTest`'s teardown races.** `on_exit` stops a coordinator that has
  sometimes already exited, so the test fails with `(EXIT) no process` after
  its assertions have passed. Seen on 248 @ `3e58fb70f` and on 247 @
  `a178a0833`, different test each time. Predates this session; small fix.
* **`lib/exmc/nuts/sampler.ex` is not `mix format`-clean at HEAD** (~60 lines).
  Every edit to it this session was hand-formatted to avoid churning unrelated
  lines into a behavioural diff. Wants its own commit.
* **`sample_from_compiled/3` deletes `:exmc_chain_meta` on the success path
  only** — a raise mid-sampling leaks it into the process. The three sites
  fixed this session all use `try/after`; this one was deliberately not
  changed, only not copied.
* **`Chain 0 on :worker_1_45511@... failed (:erpc, :noconnection), retrying on
  coordinator`** appears in every fleet log and predates all of this work.
  Nobody has looked at it.
* **`PlanBPrimeGuardTest` does run here** — an earlier note in this session
  claimed it never executes anywhere. It does, 6 tests. The claim was wrong.

### Two host facts that cost an hour each, so they are written down

**EXLA on super-io needs `LD_LIBRARY_PATH`.** `libexla.so` is a CUDA 12 build
needing `libnvshmem_host.so.3` and `libnvrtc-builtins.so.12.9`, and the wheels
are under **python3.12** site-packages — not the python3.10 tree that holds the
other `nvidia/*` wheels, so searching 3.10 finds nothing and it looks like the
libraries are absent:

    NV=/home/io/.local/lib/python3.12/site-packages/nvidia
    export LD_LIBRARY_PATH=$NV/nvshmem/lib:$NV/cuda_nvrtc/lib

Interactive shells inherit it; **agent, `nohup` and cron shells do not**, so a
suite launched that way silently runs a different backend. `jit.ex:84` has
documented this failure mode since 2026-08-23 without recording the path.
Without it the run is not merely EXLA-less — until `3e58fb70f` the Validator
turned it into ~100 failures across 12 unrelated modules.

**Fetch-by-sha now works.** `uploadpack.allowReachableSHA1InWant` is enabled on
`nx_vulkan.git` at 192.168.0.249, so `mix deps.get` no longer needs a per-host
seeding fetch after a bump. First unattended bump confirmed it: `GET_EXIT=0` on
both Keplers.

### The methodological point, since it is the reusable part

Three separate entry points — the default multi-chain path, the distributed
path, and the streaming path — each destructured `_chain_meta` and threw it
away, disabling the fused chain shader everywhere except single-chain
`Sampler.sample/3`. Measured: **12.9x**, **5.8x**, **2.6x**.

All three survived because the only test watching was a wall-clock inequality
against a concurrent path, which is host-dependent, so its failure was
dismissed as noise for two weeks. **A test that fails for a reason nobody
believes is worse than no test.** The replacements in
`test/exmc/nuts/vulkan/chain_meta_routing_test.exs` count dispatches: exact,
host-independent, and each was run against its own reverted fix to confirm it
fails saying `0 chain dispatches` rather than passing vacuously.

Keep doing that. Every fix this session has a negative control, and two of them
changed conclusions that code-reading alone had gotten wrong.

---

## Status — 2026-08-18, after the reboot

**All seven §2 items are closed.**
**Item 1 was confirmed on 2026-08-17** by the sweep it asked for — see below and
[`bench_results/OBSERVED_MODEL_EVIDENCE.md`](bench_results/OBSERVED_MODEL_EVIDENCE.md).
`6c1589a fix(synth): every observed node summed the WHOLE obs buffer` closed
it. **Item 7 was fixed on 2026-08-18** — three defects, not one; see
[`bench_results/CRASH_RECOVERY.md`](bench_results/CRASH_RECOVERY.md).

| run | result |
|---|---|
| `mix test` (default → EXLA), after items 7 + 2 | 476 tests, **1 failure** (the wall-clock one, item 4), 585s |
| `EXMC_COMPILER=vulkan mix test`, after items 7 + 2 | 476 tests, **3 failures**, 1301s |
| `EXMC_COMPILER=vulkan mix test`, after item 7 only | 473 tests, **2 failures**, 1047s |
| `EXMC_COMPILER=vulkan mix test`, after `6c1589a` | 472 tests, **4 failures** |
| `EXMC_COMPILER=vulkan mix test` ×2, before `6c1589a` | 472 tests, **5 failures**, identical both times |

Start `epmd -daemon` before any sweep — without it `distributed_test.exs`
contributes two failures that have nothing to do with the code.

`integration_test.exs:646` went green in the last one, and it is a real fix
rather than a tolerance passing — the scalar arm measures **mean 3.9864, sd
0.5536, 479/500 distinct** against analytic truth of mean 3.99, sd 0.577. It
had been frozen at 1 distinct draw in 500 with sd 3.29e-14 for the whole of the
preceding session.

That was one seed on one model, so it was not enough on its own. The evidence
table has now been re-run as a sweep — `bench/observed_model_evidence.exs`,
nine model shapes x four seeds x both arms, each row scored against the
closed-form conjugate posterior rather than against the other arm.
**72 of 72 rows within tolerance**; worst mean error 0.091, worst sd error
8.1%, fewest distinct draws 447/500. The frozen-chain signature appears
nowhere. `docs/OPEN_VULKAN_OBSERVED_MODEL.md` carries the table and the three
things about the sweep worth keeping; the raw output and host are in
`bench_results/OBSERVED_MODEL_EVIDENCE.md`.

Items 7 and 2 followed it and are also closed (§2), so **the whole P0 list is
now done**. Item 2 is the one worth reading the write-up for: the sweep found
that no statistical assertion in the suite could see a 20% variance error, the
0.3.1 "fix" included, and that the reason was sample size rather than tolerance
— which means a pure tolerance rewrite would have changed nothing.
[`docs/TOLERANCE_AUDIT.md`](docs/TOLERANCE_AUDIT.md).

### ✅ The crash-recovery defect, in flight when the reboot came — closed

Diagnosed before the reboot, fixed and verified 2026-08-18. It turned out to be
three defects rather than one, and two of them were vacuities that made the
first invisible. `fault_tolerant_test.exs:239` is green on both arms. See §2
item 7 and [`bench_results/CRASH_RECOVERY.md`](bench_results/CRASH_RECOVERY.md).

---

## 0. Two things to know before you touch anything

### `origin` is private. `upstream` publishes.

```
origin    git@localhost:/home/git/repos/exmc.git      # private server — working remote
upstream  git@github.com:borodark/exmc.git            # PUBLIC — pushing here is a release
```

The naming inverts the usual fork convention. `main` tracks `origin/main`.
**Never push to `upstream` as the last step of a task.** It is a separate,
outward-facing decision. See §1.

### `rm -rf _build/` — do it early, do not agonise

`_build/` regenerates from source and the lockfile. Nothing is lost. The `test`
env goes stale *independently* of `dev`, and a stale `_build/test/lib/<dep>` is
a first-class time sink: on 2026-08-16 it cost a long detour before anyone
noticed `_build/test/lib/nx_vulkan` was version **0.1.0** while `mix.lock`
pinned `7067499`, with a NIF missing `device_supports_f64/0` and
`leapfrog_chain_synth_f64/6`. Symptom was 20 integration failures that looked
like anything but a build artifact.

Suspect `_build/test/lib/` **first** on: `UndefinedFunctionError` for a NIF, a
`:bad_lib` on_load warning, a loaded version disagreeing with `mix.lock`, or
"suddenly every test in the repo fails."

```sh
rm -rf _build/                                     # the blunt instrument, and it is fine
MIX_ENV=test mix deps.compile nx_vulkan --force    # the surgical one
```

**The one exception was `exla`, and it is now resolved** — see §2 item 3. One
thing about it is still worth knowing here: neither the built `libexla.so`
(cached in `~/.cache/xla/exla/`, keyed by elixir/erts/xla/exla versions and
**not** by target) nor exla's C++ objects are reached by `rm -rf _build/`, so a
stale CUDA build survives every `_build` deletion you can think of.
[`docs/EXLA_CPU_BUILD.md`](docs/EXLA_CPU_BUILD.md) has the clearing recipe.

---

## 1. Decide: publish 0.3.1, or fix the known issue first

**This is the only item that is genuinely blocked on a human.**

`origin/main` is at 0.3.1. `upstream/main` is at `2de6c60` (0.3.0) — nine
commits behind. The whole correctness release is unpublished, and 0.3.0 is
still what a `mix deps.get` gets.

The case for publishing now: 0.3.0 ships a sampler whose posteriors are
over-dispersed — `Normal(0,1)` variance 1.378 against a true 1.0, worse under
`compiler: :vulkan`. Every day it stays up, someone can draw samples from it.
The CHANGELOG says so plainly and that is the right instinct.

The case for waiting **was item 1** — that Vulkan is the default for the users
who most need this library and was not correct for models with observations.
**That argument is gone as of 2026-08-17:** item 1 is fixed and confirmed across
nine variants and four seeds (§2). What is left against publishing is item 7 —
under `supervised: true`, a crash-recovered Vulkan run returns a destroyed
posterior and reports success — and the unswept tolerances of item 2. Item 7 is
narrower than item 1 was: it needs crashes to trigger, and it is a silent wrong
answer rather than a whole broken model class.

**Updated 2026-08-18: items 7 and 2 are closed too, so P0 is done.** Nothing on
that list blocks publishing on correctness grounds any more.

One thing the release notes should not claim, though. The tolerance sweep
(§2 item 2) makes the suite able to see a 20% variance error **in the marginals
of models with closed forms**. It says nothing about correlations, about the
joint distribution, or about any model without a closed form — which is most
real models. "No known correctness defect, and the suite can now see the defect
class that caused the last two" is the honest sentence. "Verified correct" is
not, and will not be until §3's Geweke and SBC items land.

> **Correction, 2026-08-16.** This file, `MISSION.md`, and the annotation on
> the red test all say `compiler: :vulkan` is "the default". That is not what
> the code does. `config/config.exs` sets no compiler at all, so the default is
> `Exmc.JIT.auto_detect/0`, which prefers **EXLA when it is available** and only
> falls through to Vulkan when it is not. The belief came from this host, where
> EXLA could not load and auto-detect therefore always landed on Vulkan (§2
> item 3, now fixed). Verified directly on one checkout, no config changed:
> while `libexla.so` was unloadable `Exmc.JIT.detect_compiler/0` returned
> `Nx.Vulkan`; once the CPU EXLA loaded, the same call returned `EXLA`.
>
> This narrows the §2 item 1 exposure rather than removing it: the users who get
> Vulkan by default are exactly the ones with no working EXLA — the FreeBSD and
> non-CUDA GPU hosts the backend exists for. They are still the ones who cannot
> use the alternative. But "the default compiler is silently wrong" is not an
> accurate description of what a hex user gets, and the §1 decision should not
> be argued on it. The middle path below still stands on its own terms; it is
> just a smaller change than "flip the default" makes it sound, since for most
> users the default is already EXLA.

**A middle path worth considering, and probably the right one:** publish 0.3.1
with the default compiler changed to `:none`, and `:vulkan` opt-in until §2 is
resolved. That makes the release honest end to end — correct posteriors *and* a
default that delivers them — and it costs users nothing they were actually
getting, since MODEL_SCALING says the Vulkan path is slower than
`BinaryBackend` at the widths eXMC runs anyway (MISSION §1). It is a one-line
change plus a CHANGELOG note.

Whichever way it goes, `mix hex.publish` needs an interactive password and is
the operator's to run.

---

## 2. P0 continued — what 0.3.1 did not close

Ranked. **All seven are closed.** P0 is done; §3 (verification as a
deliverable) is what follows, and `docs/TOLERANCE_AUDIT.md` names the two
things there that item 2 could not substitute for — Geweke and SBC.

| # | item | effort | why it ranks here |
|---:|---|---|---|
| 1 | ~~**The vulkan observed-model defect.**~~ **Closed 2026-08-17, confirmed by sweep.** Was: `compiler: :vulkan` returned a frozen chain (1 distinct value in 500 draws, `accept_prob` ≈ 0.002) for models with observations. Fixed in `6c1589a`; confirmed by `bench/observed_model_evidence.exs` — 9 variants x 4 seeds x both arms, **72/72 rows** within tolerance of the closed form, worst sd error 8.1%, fewest distinct draws 447/500. The distinct-sigma variants are the load-bearing ones (equal sigmas make a mis-assigned span bit-identical), and a per-row GPU dispatch count rules out rows that silently fell back to the host. [`docs/OPEN_VULKAN_OBSERVED_MODEL.md`](docs/OPEN_VULKAN_OBSERVED_MODEL.md), raw in [`bench_results/OBSERVED_MODEL_EVIDENCE.md`](bench_results/OBSERVED_MODEL_EVIDENCE.md). | — | — |
| 2 | ~~**The `assert_in_delta` sweep.**~~ **Done 2026-08-18.** The answer was that **no** statistical assertion in the suite could see a 20% variance error — including the one 0.3.1 added as the fix, which resolved 37.8% against a defect that was 37.8%. Three findings: 89 of 99 sampling tests asserted nothing about dispersion at all; where a gate existed the sample size made it decorative (ESS ~0.35/draw, so the resolution limits ran 36-86%); and two gates were wrong in the *other* direction (Student-t at df<=4 has no valid variance gate, Cauchy's IQR gate was 6x too tight). Fixed with `TestHelper.assert_posterior!/3`, which fails as INCONCLUSIVE when the chain cannot resolve the error it claims to check. [`docs/TOLERANCE_AUDIT.md`](docs/TOLERANCE_AUDIT.md). | — | — |
| 3 | ~~**The EXLA build.**~~ **Done — both halves.** The library bug is fixed (`exla` is `runtime: false`, `Exmc.JIT` starts it lazily and treats a failed start as "backend unavailable", covered by `test/optional_deps_test.exs`), *and* this host now has a working CPU EXLA. Recipe and its two traps in [`docs/EXLA_CPU_BUILD.md`](docs/EXLA_CPU_BUILD.md); the short version is `EXLA_CPU_ONLY=1 XLA_TARGET=cpu`, not `XLA_TARGET=cpu`. | — | — |
| 4 | **The wall-clock test.** `mix test` is now **0 failures** on the default (EXLA) path — 375 tests on `main`, 472 on `gate1/reconcile-core` with the MCLMC/MAMS/SBI suites. The only default-path failure left is `integration_test.exs:738` — `assert t_vec < t_par` — and it is **timing-flaky**, not consistently red: it failed at `1034ms < 659ms` on one run and passed on the next with no code change. Move it to `bench/`. The other three failures NEXT.md originally listed were artefacts of Vulkan-by-default and are green under EXLA — **not fixed, not exercised**. | 1 hour | a flaky red trains people to ignore red faster than a stable one |
| 5 | ~~**`config/test.exs` had never been loaded.**~~ **Fixed.** `config/config.exs` was one line, `import Config`, with no `import_config` — and Mix auto-loads only `config/config.exs`, so every setting in `config/test.exs` was dead: the `EXMC_COMPILER` switch, `config :exla, default_client: :host`, `allow_vulkan_perop_sampling`. **Every `EXMC_COMPILER=vulkan mix test` ever run sampled with whatever auto-detect picked and reported a pass for it.** Now imported, with `test/config_test.exs` as the tripwire. | — | — |
| 6 | ~~**Tests leak `:exmc` application env into each other.**~~ **Done, and verified by the check that matters: two consecutive Vulkan sweeps now return identical failure sets.** Three separate instances, all restoring wrongly or not at all: `p0_correctness_test.exs` leaked `compiler: :none`; `nuts_test.exs:618` "reset" `full_tree_nif` to `true` when its default is `false`; `fault_tolerant_test.exs` did the same via `get_env(..., true)` and skipped its restore entirely on a raised assertion. `native_tree_test.exs` was `async: true` while setting `use_nif` globally, so it raced concurrent tests rather than merely later ones. Fixed with `Exmc.TestHelper.put_env_scoped/3` (reads the previous value instead of assuming a default — the assumption is what went wrong three times) plus an `ExUnit.after_suite` tripwire over **all twelve** `:exmc` keys that gate behaviour, since an ordinary assertion only sees leaks from files that ran *before* it. | — | — |
| 7 | ~~**Vulkan crash recovery destroys the posterior, and reports success.**~~ **Closed 2026-08-18.** Three defects, all fixed and all measured: crash-recovered iterations were fed to dual averaging as acceptance 0.0 (a feedback loop that drove eps to 2.41e-11 and the posterior to variance 1.45e-15); `supervised: true` did **nothing at all** wherever the speculative buffer was live, i.e. the default path; and warmup recoveries were never counted, so 176 placeholders reported `recoveries: 0`. Before/after matrix in [`bench_results/CRASH_RECOVERY.md`](bench_results/CRASH_RECOVERY.md), reproducible via `bench/crash_recovery.exs`. | — | — |

### Item 7 in full — Vulkan crash recovery, and how it closed

Diagnosed 2026-08-17, fixed and verified 2026-08-18. Every link below was
measured, not inferred; the before/after matrix is in
[`bench_results/CRASH_RECOVERY.md`](bench_results/CRASH_RECOVERY.md) and
re-runnable with `bench/crash_recovery.exs`.

`fault_tolerant_test.exs:239` samples a **prior-only** `N(0,1)` — no
observations at all, so item 1's model class was never involved — with
`FaultInjector` set to crash at depth 3 and `supervised: true`.

| backend | injector consulted | placeholders | result | adapted eps |
|---|---:|---:|---|---|
| `:none` | 1463 | **0** | mean −0.0707, var 1.0212, 290/300 distinct, 6 div | 0.928 |
| `:exla` | 1463 | — | identical to `:none`, bit for bit | — |
| `:vulkan`, before | 3405 | **176** | mean −0.5225, **var 1.45e-15**, 300/300 distinct, **178 div** | **2.41e-11** |
| `:vulkan`, after | 1973 | 29 | mean 0.0373, **var 0.8807**, 298/300 distinct, 3 div, **29 recoveries** | **0.810** |

Uninjected, Vulkan samples this model at var 0.955 / eps 1.053. The collapse
needed the crashes; the fixed path degrades to 0.881 instead of collapsing.

**It was never item 1.** Item 1's frozen chain is literally 1 distinct draw in
500. This was 300 distinct draws in 300, inside a neighbourhood of ~4e-8. Both
read as "variance collapsed"; they were different failures.

#### Defect 1 — a crashed subtree was treated as evidence about the step size

ε fell from 1.053 to 2.41e-11 — eleven orders of magnitude:

1. A subtree crashes; the supervision wrapper catches it and substitutes
   `divergent_placeholder/4`.
2. The placeholder carried `accept_sum: 0.0`, so `nuts_step_warmup` computed
   `accept_stat = 0.0` and fed it to dual averaging.
3. Acceptance 0.0 reads as "step size catastrophically too large", so ε shrinks.
4. Smaller ε means longer trajectories to a U-turn, so trees get **deeper**.
5. Deeper trees hit the depth-3 injection point more often — 3405 consultations
   against 2018 uninjected.
6. Back to 1.

A crashed subtree measured nothing. The iteration is now **excluded from the
dual-averaging update** (`Sampler.maybe_adapt/3`) rather than fed a zero. The
placeholder also stopped fabricating `n_steps: 2 ** depth` for leaves that were
never integrated — those phantom leaves went into the denominator of
`accept_sum / n_steps` and into per-draw `sample_stats`.

**`n_steps` alone was not the lever**, despite how `divergent_placeholder`
reads: `sampler.ex` already guarded `n_steps > 0` and yielded `0.0` either way,
so zeroing the leaf count fixes the reported statistic but not the loop.
Checked before concluding; both changes are in.

#### Defect 2 — `supervised: true` did nothing on the default path

`safe_build_subtree` was reachable only from the *non-speculative* branch of
`do_build/11`, and `speculative_precompute` defaults to `true`. Measured: on
`:none` with `supervised: true`, 861 subtree builds, all speculative, the
supervision wrapper entered **zero** times. Injecting a crash at depth 1 there
killed the run outright — an unhandled `RuntimeError` straight through
`run_phase/10` — on a run that had explicitly asked to be supervised.

That is also the only reason the depth-3 row above produced numbers at all:
Vulkan's warmup downgrade disables speculation, so supervision happened to be
reachable during Vulkan warmup and nowhere else.

Fixed by hoisting the guard to wrap **whichever** dispatch runs
(`Tree.with_supervision/7`), speculative included, with `ensure_available`
inside the guarded region since the bulk pre-compute can fail the same ways.

#### Defect 3 — warmup recoveries were invisible

The before row built 176 placeholders and reported `recoveries: 0`.
`nuts_step_warmup` never read the `:recovered` flag; only `nuts_step_with_stats`
did, and that runs in the sampling phase, where the Vulkan hot path never
consults the injector. Every crash was a warmup crash, so the one counter an
operator could have noticed was structurally zero. `placeholders` and
`recoveries` now agree on every row of the matrix.

**The divergence count was fabricated too:** 176 placeholders, 178 reported
divergences. Nearly every "divergence" was a placeholder marked
`divergent: true` — which it must be, to stop the doubling — not an integrator
divergence. Crash-recovered iterations now count as recoveries and not as
divergences. That attribution is exact rather than approximate: `do_build/11`
breaks on `subtree.divergent or subtree.turning`, so a genuine divergence in an
earlier subtree would have stopped the loop before the crashing one was built,
and the two cannot co-occur in one iteration.

#### The test that proved nothing

`fault_tolerant_test.exs`'s end-to-end recovery test injected at **depth 3**,
and `:none` hit `divergent_placeholder` **zero** times while consulting the
injector 1463 times: a well-adapted host sampler never builds trees that deep.
It passed on `:none` and `:exla` for months without recovery ever running.

It now injects at depth 1, which fires on every backend (357 recoveries on
`:none`, 29 at depth 3 on `:vulkan`), and asserts `recoveries > 0` so the
vacuity cannot come back silently. A second test guards the mechanism directly
— `assert eps > 0.01` catches the feedback loop eleven orders of magnitude
before the variance assertion would. Both fail against the unfixed tree.

#### What recovery still costs

It is not free. At depth 1, where 357 of 500 iterations crash, ε adapts to 2.04
against 0.93 uninjected, variance lands at 0.877 against 1.0, and 245 of 300
draws are distinct. That is a degraded chain — and it is a chain, which is the
point. Both no-injection controls are **bit-identical** before and after, so
nothing on the non-crash path moved.

### Do not skip the red test — and mind which backend it is running

The annotated test is `integration_test.exs:646`, "vector obs produces same
posterior as equivalent scalar obs" (this file previously said 611, then 639 —
the annotation was rewritten on 2026-08-17 when the defect closed, which moved
it again; grep for the test name rather than trusting the number). It fails
for a real reason and is annotated to say so. Skipping a test that fails for a
real reason is precisely the habit that let both 0.3.1 defects ship.

**It is green under Vulkan as of 2026-08-17**, verified directly:
`EXMC_COMPILER=vulkan mix test test/integration_test.exs:646` passes, and the
sweep in item 1 says the same thing across nine variants and four seeds. The
annotation on the test still describes it as known-failing and should be
rewritten to describe the closed defect instead — do not simply delete it, the
test is the regression guard for `6c1589a`.

What follows is the state it was in before the fix, kept because it is what the
signature looks like. Since item 5 was fixed,
`EXMC_COMPILER=vulkan mix test test/integration_test.exs:646` used to fail on
exactly the documented assertion:

```
code: assert_in_delta scalar_summary["mu"].std, vector_summary["mu"].std, 0.3
```

Item 1 was confirmed alive at that point, and reproduced the numbers in
`docs/OPEN_VULKAN_OBSERVED_MODEL.md` to the digit — scalar arm mean **3.6503**,
sd **3.29e-14**, **1 distinct draw in 500**; vector arm mean 3.9716, sd 0.5516,
472/500. Under `EXMC_COMPILER=none` both arms are correct.
`allow_vulkan_perop_sampling` made no difference to it either way, which ruled
that out as the route around the chain shader.

The standing lesson: it was **green under a bare `mix test`** the whole time it
was broken, and that was not a fix — auto-detect picks EXLA on this host and the
Vulkan path is never entered (see the correction in §1). A bare `mix test`
passing says nothing about the Vulkan path. Use `EXMC_COMPILER=vulkan`, and
note that it only means anything now that item 5 is fixed.

### What the Vulkan sweep actually says

Latest, 2026-08-18, after items 7 and 2 — **476 tests, 3 failures** (was 4,
was 5; item 2 added three tests and made several much slower). The default
(EXLA) path is **476 tests, 1 failure** — the wall-clock one — on the same tree.

| test | failure |
|---|---|
| `level_set_integration_test.exs:11` | timed out at 300s |
| `poker_test.exs:228` | timed out at 300s — **back**, see below |
| `integration_test.exs:745` | the wall-clock assertion (item 4), fails on both paths |

**The poker timeout came back, and item 2 is the likely reason.** It dropped
off the list on the sweep immediately after item 7, which was run on an
otherwise idle box; item 2 raised several tests from a few hundred draws to
five figures and the Vulkan sweep went from 1047s to **1301s**. A test that
hangs until ExUnit kills it at 300s is exactly the kind that a busier box tips
over. It is a timeout with no containment behind it (see below), not a
correctness failure, and it should be treated as load-sensitive rather than
fixed or broken.

Dropped off since the last sweep:

* `fault_tolerant_test.exs:239` — **item 7, fixed.** Was `Variance collapsed:
  1.45e-15`.
* `poker_test.exs:228` — dropped off after item 7 and came back after item 2,
  both times a 300s timeout with no change that should affect it. Load-flaky;
  see the note above.
* `integration_test.exs:646` dropped off at `6c1589a` — see item 1.
* `new_dist_test.exs:271`'s `read spv: No such file or directory` dropped off
  earlier and is not expected back: it was a shader-cache race, fixed in
  `e167734`.

**Run `epmd -daemon` before a sweep.** Two `distributed_test.exs` failures on
2026-08-17 were nothing but a missing epmd after the reboot — `Cannot start
distribution ... econnrefused`. With epmd up the file is 5 tests, 0 failures.
It costs a minute to chase and looks like a real distributed-sampling defect.

The remaining timeout is unexplained and is **not** contained: per the findings
below there is no timeout containment on the Vulkan path at all, so it hangs
until ExUnit kills it at 300s.

The sweep is reproducible as of `1e735bc` — two consecutive runs before
`6c1589a` produced identical failure sets, differing only in ordering.

### The shader cache was racy, and it served empty shaders

`Exmc.NUTS.CustomSynth.Compile.compile_fresh/2` pointed `glslangValidator -o`
straight at the content-addressed cache path, and derived its temp GLSL path
the same way. Concurrent callers synthesising the same shader therefore shared
both paths — and ten-plus test modules are `async: true` and sample.

Measured, 24 concurrent compiles of one shader over 40 rounds:
**50 of 960 callers received `{:ok, spv_path}` for a zero-byte file.**
`glslangValidator` creates its output before writing it, so the `File.exists?/1`
fast path returned a module that had no contents yet. After compiling to
per-caller temp paths and `File.rename/2`-ing into place: **960/960 clean.**

Note what the repro corrected. The predicted mechanism was the shared `.comp`
path letting one caller delete another's source mid-compile, surfacing as
ENOENT. That is real and is also fixed, but it is *not* what dominates — the
existence check racing the validator's file creation is, and it fails in a
worse way, because an empty shader is a successful return rather than an error.
Fixing on the hypothesis alone would have left the common case in place.

The ENOENT interleaving was never directly reproduced (it needs a validator
failure, and none occurred in 960 runs); it is eliminated by construction
rather than by observation.

**Related, and still open:** `exmc` and `nx_vulkan` hardcode the *same*
`~/.exmc/gpu_node/spv` directory in two separate codebases, and `nx_vulkan`
ships `Nx.Vulkan.Synthesis.clear_cache/0` — an `File.rm_rf` of it. Nothing
calls it today. Any future caller silently deletes exmc's synthesised shaders
mid-run.

### Two documented safety features never run

Found while chasing the above, and worth knowing before item 1:

- **`Exmc.NUTS.Vulkan.SuspectTracker` is never started.** No `start_link`
  anywhere in `lib/` or `test/`, so `alive?/0` is always false and the
  per-shader eviction policy in its moduledoc — three consecutive timeouts
  evicts a shader and routes around the GPU — has never executed.
- **`:gpu_node` is read but never set** (`tree.ex:885`), so the watchdog path
  through `Nx.Vulkan.Node.with_node` is unreachable and `route_chain_direct`
  always takes the bare-dispatch branch.

Net: there is currently **no timeout containment on the Vulkan path**. The two
300s timeouts in the sweep hang until ExUnit kills them.

---

## 3. P1 — verification as a deliverable

Unchanged from `MISSION.md` §7 P1, except that its foundation now exists: 0.3.1
landed `Validator.ess/1` (Geyer), `analytic_moments/1`, `check_analytic/3`, a
variance SE that does not assume normality, and `bench/nuts_truth.exs`. Build on
those rather than starting over.

The ranked plan is in
`/home/io/projects/learn_erl/pymc/exmc/docs/VERIFICATION_METHODS.md`
(cross-repo, 1,641 lines). Its rank-1 item was §2 item 2 above, now closed —
though only partly in the form that document expected. Item 2 did its own
items 1 and 2 (the variance SE, and refusing a variance gate for
`2 < nu <= 4` Student-t), fixed a third gate it had not spotted (Cauchy's IQR
band, six times too tight), and added the thing that was missing from all of
them: a **power** check, so a gate that cannot see the defect it claims to
check fails rather than passes. See `docs/TOLERANCE_AUDIT.md`.

**Still open from that rank-1 list**, and cheap: thin to independence before
the KS gate (or drop it), make `:unknown` visible at suite level, and add
split-R-hat with an `ESS >= 100` precondition. Items 7-9 there — un-tag
`nuts_test.exs`, parametrise the reversibility test over all five leapfrog
implementations, ungate `native_tree_test.exs`'s `:vulkan_known_failure` — are
untouched and are the ones that make existing tests run against shipping code.
(`nuts_test.exs` no longer carries `@moduletag :gpu_state`, so item 7 there may
already be moot; check before doing it.)

Its next rank is **Geweke's joint distribution test**, which is the one check
that would have caught the tree defect at the point of introduction rather than
months later. Nothing in item 2 substitutes for it: the tolerance sweep can see
a wrong marginal in a model with a closed form, and says nothing about the
joint or about models without one.

**One addition to that plan, from what 0.3.1 found:** a **leaf-level
differential between the chain shader and the host**. Fix `q0`, `p0`, `eps`,
`inv_mass`, `K`; dispatch `leapfrog_chain_synth_f64`; read back all four
arrays; run the same K leapfrog steps through `Exmc.Compiler`'s `vag_fn`;
compare element-wise. No existing test does this, and it would have caught the
`logp_chain` off-by-one immediately. It is also the next experiment §2 item 1
needs.

### A methodological note worth keeping

Every statistical check in this repo before 0.3.1 was **differential** — run the
model two ways, assert agreement. That is structurally blind to any defect the
two arms share, and both arms share the NUTS tree. Two real defects lived behind
a green suite for months because of it.

When you add a check, ask: *what would this see that comparing two arms would
not?* If the answer is nothing, it is not buying much.

---

## 4. Beyond correctness

Only after §2 items 1 and 2. From `MISSION.md` §7 P2 onward, unchanged, with
the ordering intact — graph shape (16–20×) before shaders, and the README
honesty fix (§1 of the mission: reach, not speed) before any new performance
claim.

One item that moved up as a result of 0.3.1: **every claim should point at raw
output.** `bench_results/` now exists and holds four files — `MCLMC_BIAS.md` (partial,
§6), `OBSERVED_MODEL_EVIDENCE.md`, `CRASH_RECOVERY.md` and `TOLERANCE_AUDIT.md`
(complete). `bench/` has seven scripts. Every performance claim in the README should point at a file
containing raw output and the host it ran on, the way `nx_vulkan/bench_results/`
does; none of them do yet.

---

## 5. Where the numbers came from

So the next person can re-run rather than trust:

```sh
# posterior moments vs analytic truth — the check that can see a shared defect
mix run --no-deps-check bench/nuts_truth.exs
COMPILER=vulkan SEEDS=1,2,3 mix run --no-deps-check bench/nuts_truth.exs
USE_NIF=0      mix run --no-deps-check bench/nuts_truth.exs   # pure Elixir tree
FULL_TREE_NIF=1 mix run --no-deps-check bench/nuts_truth.exs  # Rust build_full_tree

# the observed-model evidence sweep — the check that closed §2 item 1
COMPILER=none   SEEDS=42,1,2,3 mix run --no-deps-check bench/observed_model_evidence.exs
COMPILER=vulkan SEEDS=42,1,2,3 mix run --no-deps-check bench/observed_model_evidence.exs

# what the suite's statistical tolerances admit — the check that closed §2
# item 2. Each test appears twice, `was` and `now`; the `4sd floor` column is
# the resolution limit no tolerance rewrite can beat.
mix run --no-deps-check bench/tolerance_audit.exs

# crash recovery under supervision — the check that closed §2 item 7.
# INJECT=0 is the control; `placeholders: 0` with injection on means the run
# never exercised recovery, however green it looks.
COMPILER=none   DEPTH=1  mix run --no-deps-check bench/crash_recovery.exs
COMPILER=vulkan DEPTH=3  mix run --no-deps-check bench/crash_recovery.exs
COMPILER=vulkan INJECT=0 mix run --no-deps-check bench/crash_recovery.exs

# the regression tests (move _build/test/lib/exla aside first — §0)
mix test --no-deps-check test/nuts/p0_correctness_test.exs
```

**There are three tree implementations**, selected by `use_nif` and
`full_tree_nif`, and each carries its own copy of the doubling logic. A guard
added to one and not the others is a defect that only appears under whichever
flag nobody set. `bench/nuts_truth.exs` sweeps all three for this reason.

To prove a code path actually ran — which matters more than it sounds, since a
vacuous check reads exactly like a passing one — use `:call_count` tracing, and
**force-load the modules first** or `trace_pattern` silently matches nothing:

```elixir
Code.ensure_loaded!(Exmc.NUTS.NativeTree)
:erlang.trace_pattern({Exmc.NUTS.NativeTree, :build_subtree_bin, 10}, true, [:call_count])
# ... run ...
:erlang.trace_info({Exmc.NUTS.NativeTree, :build_subtree_bin, 10}, :call_count)
```

---

## 6. B1 (MCLMC / MAMS) — landed, with one measurement unfinished

**Written 2026-08-16.** Roadmap item B1 from
`/home/io/projects/learn_erl/pymc/exmc/docs/PLAN_SAMPLER_ROADMAP.md` §3.
Stages B1.1–B1.3 are complete and gated. **B1.4, the bias measurement, is
partially run and needs finishing — a host reboot interrupted it.** B1.5 (the
GLSL arm) is deferred to Gate 5 and was deliberately not started.

### What landed

| file | what |
|---|---|
| `lib/exmc/mclmc/integrator.ex` | the isokinetic step — minimal-norm `V T V T V` splitting, λ = 0.1931833275037836, plus leapfrog |
| `lib/exmc/mclmc/tuning.ex` | EEVPD step-size adaptation, the two `L` estimators, the moment accumulator |
| `lib/exmc/mclmc.ex` | `sample/3`, `sample_compiled/3` — biased, one draw per integrator step |
| `lib/exmc/mams.ex` | the same dynamics with a Metropolis accept — asymptotically unbiased |
| `test/mclmc/{integrator,tuning,mclmc,mams}_test.exs` | 42 tests, all green |
| `bench/mclmc_bias.exs` | the B1.4 sweep |
| `bench_results/MCLMC_BIAS.md` | **partial** — see below |

Nothing in the model layer changed. Both samplers take
`Exmc.Compiler.compile_for_sampling/1`'s tuple and use only its `vag_fn` and
`PointMap` slots.

### To continue after the reboot: finish the bias sweep

**This is the one unfinished thing. Start here.**

`bench_results/MCLMC_BIAS.md` carries a `PARTIAL RUN` banner. What it actually
contains, and what it does not:

| block | state |
|---|---|
| `Normal(0,1)`, `HalfNormal(1)`, `Exponential(2)` at `d = 2` | **complete** — all ten rows each |
| `Normal(0,1)` at `d = 8` | **partial** — MCLMC at six step sizes, MCLMC (tuned) and MAMS are there; **the NUTS row is missing** |
| `HalfNormal`/`Exponential` at `d = 8` | missing |
| everything at `d = 32` | missing |

The missing NUTS row at `d = 8` is not a cosmetic gap: at `d = 8` MAMS reaches
**1.6964 ESS/gradient** and MCLMC (tuned) **1.5258**, against 0.4169 and 0.2283
for the same two at `d = 2`. Whether that beats NUTS is the entire B1 case and
**the number that would answer it is the one that did not finish.** `d = 32` is
what the roadmap's "abandon if" is really about.

**Do not quote this file as evidence about high dimensions until the sweep is
re-run.** The command, unchanged:

```sh
DIMS=2,8,32 SEEDS=1,2,3 WARMUP=1000 SAMPLES=3000 \
  EPS=0.1,0.25,0.5,1.0,2.0,4.0 OUT=bench_results/MCLMC_BIAS.md \
  mix run --no-deps-check bench/mclmc_bias.exs 2>&1 | tee /tmp/mclmc_bias.log
```

Budget roughly **3–4 hours** on `super-io` under load — the three `d = 2`
blocks alone took about an hour of process time with a second agent on the box.
Run it on an otherwise idle host if one is available, and `tee` it: three
things worth knowing before starting.

- **Fix the incremental write first — it is ten lines and it already cost one
  run.** The script builds the whole document in memory and writes `OUT` once,
  at the very end, so a kill loses the file. Every row *is* printed to stdout as
  it is produced, which is how the current partial file was recovered, but that
  recovery should not have been necessary. Append each block to `OUT` as it
  completes.

- **NUTS is the slow arm by a wide margin, and not because of tree depth.**
  Measured directly on this host, `compiler: :none`, two `HalfNormal(1)` RVs:
  600 NUTS iterations took 18.8 s at a mean tree depth of 2.1 and 1106 total
  gradient evaluations — about **8 ms per gradient**. MCLMC on the same model
  and backend runs ~2150 gradients/second, i.e. **~0.5 ms per gradient**. That
  is a 15× per-gradient overhead in the tree machinery, not an algorithmic
  difference, and it is not visible in any ESS-per-gradient table. It deserves
  a profile of its own.

- **`mix test` is green before you start.** The full suite was **461 tests, 1
  failure** with all of this in place, and the one failure is the known
  pre-existing wall-clock assertion at `integration_test.exs:762` (§2 item 4).
  The four new files add 42 tests and take about 5 minutes of the run. If the
  suite is not in that state after the reboot, fix that before trusting any
  benchmark number.

### What is not done, and should be

- **B1.4 at `d = 8` and `d = 32`.** Above. This is the item that decides
  whether B1 was worth doing.
- **A Geweke joint-distribution run against MAMS.** The roadmap asks for it and
  it is the right check for a novel accept step. It needs `simulate_from_prior/2`
  and a public single transition — §3 / `MISSION.md` §7 P1 item 9. MAMS's
  accept step is currently gated by an exact involution test
  (`test/mclmc/mams_test.exs`) plus the analytic-moment battery, which is
  strong but is not a joint-distribution test.
- **`init_values` under NCP.** Both samplers raise rather than guess;
  `Exmc.NUTS.Sampler.invert_ncp_init/2` is private and was not duplicated.

### One general finding, and it is not confined to B1

**A bare Elixir float in an `Nx` binary op silently computes at f32, even
against an f64 tensor.** `Nx.divide(f64_tensor, 0.9695359714832659)` returns an
f32-accurate result: the scalar becomes a default-typed `{:f, 32}` tensor and
the promotion widens *after* the arithmetic. Measured on `Nx.BinaryBackend`.

This cost real time here — the `‖u‖ = 1` invariant failed at **3e-8**, which is
f32 epsilon, and read exactly like an algebra error in a new integrator.
`Exmc.MCLMC.Integrator` now routes every scalar through a `c/2` helper that
builds it at the tensor's own type, and the note is in that module's source.

**The rest of the repository has not been audited for this.** `sampler.ex`,
`leapfrog.ex`, `tree.ex` and `mass_matrix.ex` all mix Elixir floats with
tensors. Any place that does is computing at f32 while believing it is at f64,
and `MISSION.md` §4's "the f64 default is not costing anything and it is a
correctness asset" is only true where the default actually applies. Worth a
grep before the next precision question is diagnosed as a backend problem.

---

## 7. C2 (SBI / ABC) — landed, and the thing it was built for is not

**Written 2026-08-20.** Roadmap item C2 from
`/home/io/projects/learn_erl/pymc/exmc/docs/PLAN_SAMPLER_ROADMAP.md` §5, whose
title is "simulation-based inference **on `sim_ex`**". Four of its five stages
are complete and gated. **C2.4 — the `sim_ex` bridge and the calibration
notebook — was not started**, and since that is the stage the item is named
after, read the rest of this section with that in mind: what landed is a
correct, tested ABC library with no consumer.

Committed as `82db4f8`.

### What landed

| file | what |
|---|---|
| `lib/exmc/sbi/simulator.ex` | the behaviour — `simulate(params, rng) :: {summary, rng}`, a parameter map of plain floats and a functional `:rand` state in, a summary vector and the advanced state out |
| `lib/exmc/sbi/prior.ex` | independent scalar priors, drawable and evaluable. Deliberately standalone rather than a route into `Exmc.IR`: ABC needs draw and `logpdf` and nothing else — no gradient, no transform |
| `lib/exmc/sbi/abc.ex` | rejection ABC — the reference arm |
| `lib/exmc/sbi/abc_smc.ex` | Toni et al. (2009), Del Moral's adaptive tolerance, the Beaumont et al. (2009) perturbation kernel |
| `lib/exmc/sbi/engine.ex` | option resolution, the distance, RNG splitting, batched evaluation |
| `lib/exmc/sbi.ex` | `run/3`, `run!/3`, and the posterior accessors — `posterior_mean/1`, `credible_interval/3`, `weight_ess/1`, `resample/3`, `rank/5`, `prior_predictive_scale/3` |
| `test/sbi/*` | 55 tests, all green |

Nothing outside `lib/exmc/sbi/` changed except `mix.exs`, which gained the
ex_doc groups. There is no dependency on the model layer at all: `Exmc.SBI`
never calls `Exmc.Compiler`, never builds an `Exmc.IR`, and never
differentiates anything. It is the only inference path in the repository that
does not.

### The stages, and which one is missing

| # | work | state |
|---|---|---|
| C2.1 | `Simulator` behaviour + rejection ABC | ✅ |
| C2.2 | ABC-SMC — schedule, kernel, weights, resampling | ✅ |
| C2.3 | parallel evaluation over `Task.async_stream`, **then `Mesh.Pool`** | ✅ / **✗** |
| C2.4 | the `sim_ex_exmc` bridge + the M/M/1 calibration notebook | **✗ not started** |
| C2.5 | the SBC gate | ✅ |

C2.3 is half done and the half that is missing is the one that matters for the
BEAM claim. `Task.async_stream` fans a population out across schedulers on one
node; `Mesh.Pool` is what would fan it across the cluster. **`lib/exmc/mesh/`
does not exist in this repository** — it is one of the private-only subtrees the
Gate 1 survey found. So the distributed arm of C2.3 is blocked on the
core/applications split, not on ABC.

### The gate, and the two tests the default run does not execute

`test/sbi/sbc_test.exs` is the primary gate, and it is the one place in the
whole roadmap where SBC is the right tool: Geweke is ~40× cheaper but needs an
exact-invariance argument about a Markov kernel, and a likelihood-free
posterior has none. It is affordable here because the target is a
Normal–Normal conjugate model whose simulator is ten normal draws — 800
complete ABC-SMC fits cost about ten seconds, not the hours a NUTS SBC would.

The target is conjugate for a second reason worth keeping: its summary, the
sample mean, is **sufficient**. Run SBC on the M/M/1 queue instead and a red
gate is ambiguous between "the sampler is wrong" and "mean waiting time is not
sufficient for (λ, μ)" — and an ambiguous gate teaches people to ignore it.

The module documents its own error rates rather than asserting them. Null, 300
experiments of 800 replicates: rejection at α = 0.01 measured **0.0033**
against a nominal 0.01, median p-value **0.485**. Power at α = 0.01 against a
posterior whose standard deviation is wrong by a fixed factor: **0.913** at
sd × 0.85, **0.427** at sd × 0.90, **0.307** at sd × 1.10. 800 replicates and
10 bins were both chosen off that table rather than picked. And the gate is
then measured *in situ* against the real sampler by throwing the importance
weights away — the single most likely way for an ABC-SMC implementation to be
wrong — which gives **p = 7.3e-8** against **0.299** for the unmodified
posteriors.

**But the two tests that produce those error rates are `@tag :slow` and are
excluded from the default run** — `sbc_test.exs:183` (false-positive rate under
the null) and `:213` (power against a known scale error). That is a defensible
call, since between them they are 300 complete SBC experiments. It does mean
the default `mix test` re-checks the gate but never re-checks the gate's
credibility. The numbers above are as measured on the day and nothing in CI
would notice them drifting.

```sh
mix test --include slow test/sbi/sbc_test.exs   # runtime not measured; budget generously
```

The M/M/1 arm (`test/sbi/mm1_test.exs`) validates the fixture against
`ρ = λ/μ`, `Wq = ρ/(μ−λ)` and `Lq = ρ²/(1−ρ)` **before** using it to gate the
inference, which is what makes it a gate rather than a comparison against
another piece of our own code. Every tolerance in it is a standard error
computed from the run's own replications or a binomial bound on a coverage
count.

### What is not done, and should be

- **C2.4, the whole of it.** No `sim_ex_exmc` bridge — `sim_ex` is not
  referenced anywhere under `lib/exmc/sbi/` or `test/sbi/`. No calibration
  notebook; `notebooks/` has nothing on ABC. The M/M/1 that exists is a test
  fixture in `test/sbi/support/mm1.exs`, not a document anyone would read.
  This is the deliverable the roadmap calls P-3 and describes as "the only item
  on this roadmap that is plausibly publishable on its own", and it is the item
  the "abandon if" condition is written about — whether the simulator call
  dominates so completely that useful particle counts are out of reach. **That
  question is currently unanswered**, because the only simulator ABC has been
  run against is a fixture designed to be fast.
- **`Mesh.Pool` evaluation.** Above. Blocked on the split.
- **No `bench/` script and no `bench_results/` file.** §4 of this document says
  every claim should point at raw output; C2 is the newest subsystem in the
  repo and is the one with no such file. The SBC error-rate table lives in a
  `@moduledoc` and the M/M/1 coverage numbers live in assertions. Both should
  be re-derivable by running one script.
- **Neither arm has been run on Vulkan or EXLA**, because neither touches Nx.
  That is correct — a simulator is a process, not a tensor lane — but it means
  C2 contributes nothing to the backend sweep in §2 and should not be counted
  toward it.

### One design decision worth not undoing

Proposals are generated **sequentially** from the parent `:rand` state and
evaluated in a **batch whose size is fixed before any simulator runs**, so
nothing about the result depends on the concurrency. `parallel: false` and
`parallel: true` over the same seed produce bit-identical particles, weights
and simulation counts — which turns "is the parallel path the same algorithm?"
from a hopeful assertion into an equality test, and
`test/sbi/determinism_test.exs` asserts exactly that, including across worker
counts.

The cost is real and is not hidden: a batch may simulate more proposals than
the population needs, because it cannot stop early. `n_simulations` in every
result counts what actually ran. Anyone tempted to reclaim those simulations by
generating proposals inside the workers should understand they are trading the
equality test for them.

### The caveat that belongs on the front page

`Exmc.SBI`'s `@moduledoc` carries it as a warning block and it should stay
there: ABC targets `p(θ | S(y*))`, not `p(θ | y*)`, and those are the same
distribution only when `S` is sufficient. Driving `ε → 0` does not repair it.
There are two approximations in every ABC posterior and only `ε` is under the
user's control — insufficiency does not appear in any diagnostic, does not
shrink with budget, and does not show up as a divergence or a low ESS. The
roadmap made this an explicit gate requirement ("in the docs and not only in
the tests") because the characteristic failure of ABC software is that the
sentence exists somewhere and nobody reads it. If the README ever grows an SBI
section, it goes there too.
