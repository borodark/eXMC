# Review plan — eXMC as the middle of the stack

**Written:** 2026-09-12, against `main` @ `dc671b62c` with the working tree's
pin at nx_vulkan `bae9221`, from the three-repo sweep recorded at the top of
`NEXT.md`. Sibling plans: `../../../nx_vulkan/REVIEW_PLAN.md`,
`../../../pathmc_ex/REVIEW_PLAN.md`. Shared vocabulary:

| arm | exmc setting | deploy | research |
|---|---|---|---|
| **CPU** | `compiler: :none` → `Nx.Defn.Evaluator` on BinaryBackend | everywhere; the reference | — |
| **EXLA** | `compiler: :exla` (CUDA build, or the CPU build in `docs/EXLA_CPU_BUILD.md`) | Linux | — |
| **Vulkan** | `compiler: :vulkan` → nx_vulkan | **FreeBSD, the only GPU option there** | Linux, against EXLA on super-io |

Auto-detection is EXLA > Vulkan > Evaluator (`lib/exmc/jit.ex`). That order
has a consequence the sweep kept running into: on a Linux box with a working
EXLA, the Vulkan arm is never exercised unless forced, and on the same box
with EXLA present-but-unloadable (no `LD_LIBRARY_PATH`), the run silently
becomes the Vulkan arm. `test_helper.exs` prints which arm ran; nothing else
in the repo records it.

## What the sweep established (do not re-derive)

- nx_vulkan `5f65398` (increment 2) segfaults this suite on super-io within
  10 s, twice, in Vulkan loader initialisation; single files pass. Diagnosis
  and backtrace in `NEXT.md` (2026-09-12 later). The lock stays at `bae9221`.
- At `bae9221`, `mix test` on super-io with auto-detected Vulkan: 723 tests,
  2 failures. One is the host-specific Cauchy KS check. The other,
  `CustomDistTest` "custom dist works with NUTS sampler", passes under
  `EXMC_COMPILER=vulkan` and fails without it, because
  `:allow_vulkan_perop_sampling` is set by `config/runtime.exs` **from the env
  var**, not from the detected backend — deliberately, per its comment and
  `Exmc.JIT.describe/0`'s moduledoc, which record the same 16/1 vs 16/0 on
  mac-248. The fleet script sets the variable, so the fleet never sees this.
  MEASURED.
- The EXLA arm was last recorded fully green on 2026-08 at `a178a0833`
  (652 tests). Every status section since 2026-09-02 is Vulkan-only. The EXLA
  arm at HEAD is unmeasured.
- pathmc_ex calls a small, stable surface (listed in Track 4). The one lib
  change between `147305261` and HEAD that reached it (`994305de4`,
  `stats.divergences` no longer counts warmup) broke its guide notebook while
  its gate stayed green — the commit message predicted this.
- `EXMC_COMPILER` is read only in `config/runtime.exs`; a consumer's runtime
  never loads it. The only consumer-side knob is
  `Application.put_env(:exmc, :compiler, ...)`, undocumented in README.
- `mix format --check-formatted` fails on 26 files at HEAD. There is no
  format gate.
- `@moduletag :vulkan` on four modules is excluded by nothing;
  `jit_vulkan_test.exs` claims `mix test` skips them.
- `CHANGELOG.md` Unreleased was empty across 40 commits; filled 2026-09-12.
- README mentions neither Vulkan nor how the dependency wiring works.

## Track 1 — Make the arm an explicit input, everywhere

**Question.** Can a run's arm be chosen, recorded, and reasoned about without
reading a log?

1. `config/runtime.exs` sets `:allow_vulkan_perop_sampling` only under an
   explicit `EXMC_COMPILER=vulkan`, deliberately (its comment: a global flag
   would turn a loud refusal into a silent per-op run). The one test that
   needs the fallback, `CustomDistTest` "custom dist works with NUTS sampler",
   should set it for itself with `put_env_scoped/2`, and the env-var block in
   `runtime.exs` should then go. Then `mix test` on a Vulkan host and
   `EXMC_COMPILER=vulkan mix test` are the same arm, and the auto-detect
   run's second failure disappears for the right reason.
2. Write `docs/ARMS.md`: the three arms, how each is selected (env var for
   `mix` invocations from this repo; `put_env` for consumers), what each host
   in the fleet runs, and the expected suite result per arm per host — a
   `SUITE_COUNTS.md` for this repo, since every fleet status section in
   `NEXT.md` currently re-derives it.
3. Give consumers a programmatic knob and document it:
   `Exmc.JIT.configure!(compiler: :vulkan | :exla | :none)` that validates and
   sets the env, or at minimum a README section naming the two config keys
   (`:compiler`, `:force_precision`) and their valid values. pathmc_ex's docs
   currently talk about `EXMC_COMPILER` as if it reached them.
4. `Exmc.JIT.describe/0` already exists; put it in the stats map every sampler
   returns (`arm: ...`), so a posterior carries the arm that produced it.
   pathmc_ex's "do not report a speedup from eligibility" note is the symptom
   of this being absent.

**Acceptance.** `mix test` and `EXMC_COMPILER=vulkan mix test` report the same
failures on super-io; `docs/ARMS.md` has one row per fleet host with a measured
count; a sampler result says which arm ran.

## Track 2 — Correctness across arms, not per arm

**Question.** Where the three arms all run a model, do they agree, and is
that asserted anywhere that fails?

Existing pieces: `bench/nuts_truth.exs` and `bench/observed_model_evidence.exs`
("both arms": EXLA and Vulkan), `test/nuts/conjugate_oracle_test.exs`,
`test/nuts/leapfrog_leaf_diff_test.exs` (Vulkan against a host reference at
`1e-13`), the Cauchy KS Validator check that fails on super-io only.

1. Promote the cross-arm comparison into a test the way the leaf diff was
   promoted: `test/nuts/arm_parity_test.exs`, the same seeded models sampled
   under every arm available on the host, asserting the posterior summaries
   agree to a tolerance measured first and written beside the assertion. Tag
   it by what it needs (`:requires_exla`, `:requires_vulkan`), not by what it
   is.
2. The leaf-diff test compares Vulkan to the host. Add the EXLA column on
   super-io: if EXLA's leapfrog differs from the host reference at the same
   order as Vulkan's, the `1e-13` bound is about f64 rounding; if it is
   bit-exact, the bound is about Vulkan. Either answer belongs in
   `docs/TWO_GATES_THAT_DO_NOT_GATE.md`.
3. The Cauchy KS failure is host-specific on super-io. Run the Validator under
   the EXLA arm on the same box. If the KS check passes on EXLA and fails on
   Vulkan with identical seeds, the draws differ by arm and the "bit-identical
   q/p/grad" claim needs a scope note; if both fail, it is the test.
   **Hypothesis to test first (INFERRED, 2026-09-12):** the chain shader's
   transcendentals are f32-cast by default (`:chain_shader_transcendentals`
   is `:f32_cast`; `multi_rv_custom_spec.ex:945`), Cauchy's logpdf is
   `log(1 + z²)`, and f32 `log` approximations differ by GPU architecture —
   which would make draws host-specific in exactly the observed pattern
   (fails on Ampere, passes on two Keplers and the Jetson, same seeds). The
   leaf-diff fixtures are Normal-family and never exercise `log` of a sampled
   value, so their 1e-15 agreement does not speak to this. Measure: a
   leaf-diff fixture with a Cauchy or Student-t likelihood, per host, and the
   same under `:polynomial` on asus (the only box where that lever does not
   segfault). nx_vulkan's elementwise f64 shaders make the same trade
   (`MISSION.md` §3.2 there), but eXMC's chain path never calls them; they are
   reached only by the per-op fallback, where `Nx.pow(t, 2)` falls back to
   the host (exact, 604x slower) and `t * t` stays on the GPU (exact f64).
4. Retire `:vulkan_known_failure` as a concept or generalise it:
   `:known_failure` with an arm and a host in the tag value, read by
   `test_helper.exs`.

**Acceptance.** One parity test that fails when an arm drifts; the leaf-diff
table has an EXLA column; the Cauchy question has a measured answer.

## Track 3 — The EXLA arm on Linux, measured again

**Question.** Is the arm that Linux deploys on still green, and can a fresh
shell reach it?

1. Run the full suite under `EXMC_COMPILER=exla` on super-io at HEAD with the
   CUDA build (`LD_LIBRARY_PATH` per the recipe — put the recipe in
   `docs/EXLA_CPU_BUILD.md` or a new `docs/EXLA_CUDA_SUPER_IO.md`; today it
   exists only in an agent's memory file) and once with the CPU build. Record
   both in `docs/ARMS.md`.
2. `test/run_all.sh` is a CUDA-era wrapper nothing references. Fold what it
   still does (the libdevice symlink) into the EXLA doc and delete it, or make
   it the documented EXLA-arm runner.
3. `Exmc.JIT.usable?/1` treats a failed EXLA start as "unavailable". Add a test
   that a present-but-unloadable EXLA (the `libnvshmem_host.so.3` case) yields
   the Vulkan arm *and says so* — that is the 2026-08-23 hour, as a test.

**Acceptance.** Two rows in `docs/ARMS.md` for super-io EXLA (CUDA, CPU) with
counts; no undocumented environment variable on the path to either.

## Track 4 — The contract with pathmc_ex

**Question.** What can a consumer rely on, and how does it learn of a change
before its gate goes green over a broken notebook?

The surface pathmc_ex uses, MEASURED 2026-09-12: `Exmc.Builder.{new_ir/0, rv/5,
obs/4}`, `Exmc.Dist.Custom.{new/2, rv/4}`, `Exmc.Dist.{Normal, HalfNormal,
Gamma, Beta}`, `Exmc.NUTS.Sampler.{sample/3, sample_chains/3}` and the option
keys `:init_values :seed :vectorized :num_samples :num_warmup`, the `{trace,
stats}` shape and `stats.divergences`, `Exmc.Diagnostics.rhat/1` on a list of
lists, `%Exmc.Node{op: {:rv, dist, params[, transform]}}`,
`Exmc.NUTS.ChainShaderCodegen.detect_meta/2` and its refusal atoms,
`Exmc.Stan.sample/3`, and the app env keys `:force_precision`, `:compiler`.

1. Write that list into `docs/PUBLIC_API.md` and mark each item stable or
   internal. `detect_meta/2` is a probe pathmc_ex depends on; its `@spec`
   (`chain_shader_codegen.ex:47`) does not describe what it returns
   (`{:synthesised, …}`, `{:unsupported, reason}`). Fix the spec or the
   function.
2. The refusal atoms are a contract: the seven remaining bare `:unsupported`
   sites (`chain_shader_codegen.ex:106,108,169,279,290`, `custom_synth.ex:154,231,237`)
   should each carry a reason, and the reasons should be listed in one place
   with what a consumer can do about each. `:multiple_custom_nodes` is the one
   pathmc_ex hits on every multi-equation model; the plan for lifting it (R2)
   needs a sentence, even if the sentence is "not planned".
3. Changelog discipline: a change to anything in `PUBLIC_API.md` gets an
   Unreleased entry in the same commit. `994305de4` is the case study.
4. Init values: `sample_chains/3` starts every chain from the one
   `init_values` map, varying only the seed. R-hat's over-dispersed-starts
   rationale is not delivered. Either add init jitter or document that chains
   share a start, so pathmc_ex's comment stops claiming otherwise.

**Acceptance.** `docs/PUBLIC_API.md` exists; `detect_meta/2` has a spec that
Dialyzer would enforce; no bare `:unsupported` remains.

## Track 5 — The contract with nx_vulkan

1. The pin policy in `mix.exs` is good and long; move it to `docs/ARMS.md`
   and leave a pointer. Add the bump checklist nx_vulkan's `NEXT.md` asks for
   (`bench/nuts_truth.exs` on both arms after a bump) and run it for the
   current pin, which has not been done for any pin since `9a8427c`.
2. `NXV_DEVICE` is in the pin, and since nx_vulkan `77bb61f` (increment 2) a
   tensor carries its device: every chain NIF this repo calls has an `_on`
   variant taking a device slot (`leapfrog_chain_synth_f64_on/7`,
   `..._batch_f64_on/7`), and any multi-tensor NIF refuses mixed operands with
   `{:error, :cross_device, msg}`. `Exmc.NUTS.Vulkan.Dispatch` still calls the
   default-slot forms, so on a two-GPU host every chain lands on the default
   card. Decide whether `Dispatch.chain/8` takes a device, and whether
   `Exmc.NUTS.BatchCoordinator` handles `:cross_device` as a refusal or a bug.
   `scripts/fleet_verify.sh` should pin the device per host by UUID the way
   nx_vulkan's item 4 proposes, and refuse on mismatch; on asus it currently
   tests whichever card enumerates first. **Read nx_vulkan
   `docs/MULTI_DEVICE.md` (`3c5ae2d`) first** — it is the consumer description
   of the surface: `Nx.Vulkan.Device.resolve/1` (selector or slot →
   `{:ok, slot, info}`), `ChainTrace.dispatch_f64/7`, `Node` per device,
   `Device.{class,f64?,weak?}/1` for a named device, and the startup cost
   (13 s for the first Vulkan client in a BEAM). One rough edge it records:
   `:cross_device` is a return on the NIF and chain paths this repo uses, but
   a raise (a `MatchError` carrying the tuple) on Nx-level ops — irrelevant
   while this repo stays on the chain path, and to be said so if that changes.
3. Rustler range: `~> 0.36` here admits 0.37, which nx_vulkan says is broken.
   Narrow to `~> 0.36.0` or measure 0.37.
4. Docstrings that cite nx_vulkan state were fixed 2026-09-12; add a grep to
   the bump checklist for `nx_vulkan/` paths and short shas in `lib/` and
   `test/` so the next bump catches the next one.

## Track 6 — Hygiene that blocks the rest

- A whole-tree `mix format` as its own commit, then `format --check-formatted`
  as the first step of whatever the gate is (pathmc_ex's `mix check` alias is
  the model). Done 2026-09-12: `mix check` = format check, compile, test.
  Not yet `compile --warnings-as-errors`: the tree carries 16 warnings at
  HEAD (clauses of `do_eval/3`, `do_emit/3`, `detect_meta/2`,
  `analytic_moments/1` not grouped; `@doc` on private functions; unused
  defaults on `nuts_step_with_stats/8`). Fix those, then add the flag.
- README: a dependency section (git server, `NX_VULKAN_PATH`, `NX_PATH`,
  optional EXLA and how a consumer opts in), an arms section, a testing
  section naming the gate.
- Tag cleanup: `:vulkan` (dead), `:requires_f64` (excluded, never tagged),
  `:statistical :poker_integration :glslang :benchmark :stan :integration
  :level_set_integration` (tagged, never read). Bulkhead and server tests get
  `:requires_vulkan` or a guard.

## Order

1 (arms explicit) → 6 (format, so later diffs are readable) → 3 (EXLA arm
measured) → 2 (parity) → 4 (public API) → 5 (nx_vulkan contract). Track 1 is
first because every later measurement is only as good as the arm label on it.
