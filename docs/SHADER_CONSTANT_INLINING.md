# The chain shader inlines its data — and the driver stops accepting it

Status as of 2026-09-05: **21 of 33 posteriordb models cannot run on Vulkan at
all.** The fused f64 chain shader bakes closure-captured rank-1 tensors into the
GLSL as `const double[]` literals, so SPIR-V size grows with the data. Past
roughly 1300 inlined elements the NVIDIA driver refuses to create the compute
pipeline.

This was invisible until `benchmark/posteriordb` was made to name its compiler.
Its "33/33 PASS" is an EXLA-CPU result; nothing had ever run that suite on the
GPU.

## Evidence

`--compiler vulkan` on the 33 posteriordb models: **1 PASS / 10 FAIL / 22 CRASH**.
21 of the 22 crashes are the same error, from `ComputePipeline::new`
(`nx_vulkan/native/nx_vulkan_vulkano/src/lib.rs:139-144`):

    {:error, :dispatch_failed, "ComputePipeline: a non-validation error occurred"}

It is pipeline **creation**, not dispatch, and it reproduces at `parallel=1`, so
it is not GPU contention. The separation is monotone in `n_obs x n_beta`:

| model | n_obs | n_beta | product | result |
| --- | ---: | ---: | ---: | --- |
| mesquite-* | 46 | 7 | 322 | no crash |
| sblr* | 100 | 5 | 500 | no crash |
| kidiq-kidscore_momiq | 434 | 2 | 868 | no crash |
| kidiq-kidscore_momhsiq | 434 | 3 | **1302** | CRASH |
| earnings-earn_height | 1192 | 2 | 2384 | CRASH |
| nes2000-nes | 476 | 9 | 4284 | CRASH |

A clean threshold between 868 and 1302, no exceptions across all 33.

Corroborated by the artifacts in `~/.exmc/gpu_node/spv/`: synthesised chain
shaders run **78 KB to 2.15 MB** of SPIR-V, against ~8 KB for every hand-written
shader in `nx_vulkan/priv/shaders`.

## Mechanism — not the one shader size first suggests

The likelihood is **not** unrolled. `multi_rv_custom_spec.ex:1071` emits a serial
loop, which is size-independent:

    for (uint j = <lo>; j < <hi>; j++) { double obs_j = obs_inv_mass[...]; ... }

What scales is the **captured data**. A linear regression closes over its design
matrix as `n_beta` rank-1 column tensors of length `n_obs`. Only `y` reaches the
GPU as a buffer; every column of `X` is a closure capture, and
`Glsl.register_capture/1` (`glsl.ex:605-625`) lowers each one to

    const double __captured_t<hash>[N] = double[](v0, v1, ... vN-1);

emitted at file scope by `build_captured_decls/1`
(`multi_rv_custom_spec.ex:757`). So the shader carries `n_obs * n_beta` float
literals.

**The code already knows.** `glsl.ex:52-56`:

> Inline `const float[]` works for the synth-coverage probe and for **small** obs
> tensors. Per-instance batched dispatch requires the captures to move to SSBO
> bindings, with the obs registered via `Builder.data/2` at the model-builder
> layer — that is follow-up work, not this clause.

and `multi_rv_custom_spec.ex:751`:

> Inline `const float[]` is correct for the synth-coverage probe and for closures
> over fixed reference data. Per-instance batched dispatch requires the obs
> tensors to move to SSBO bindings [...] closures over rank-1 obs bake
> instance-0's data into the shader.

Both name the same fix and both scope it to *batching*. The size ceiling is the
part nobody costed: it breaks single-instance dispatch too, and it is the reason
the largest two thirds of posteriordb has never run on a GPU.

`Builder.data/2` exists (`builder.ex:19`) but only forces a tensor onto
BinaryBackend and stores it at `ir.data`. It does not create an SSBO binding.
The mechanism the comments point at **has not been built.**

---

## The fix: captures become a slice of the extras buffer

The shader already has an SSBO carrying heterogeneous host data:

    layout (std430, binding = 2) readonly buffer In_extras { double obs_inv_mass[]; };

with layout `obs[0..n_obs)` then `inv_mass[0..d)`, packed in `dispatch.ex:153` as
`extras_bin = obs_bin <> inv_mass_bin`. Captures become a third region.

**Layout: `obs | inv_mass | captures`** — deliberately appended *after* inv_mass
rather than inserted between. Every existing index expression
(`obs_inv_mass[j]`, `obs_inv_mass[pc.n_obs + tid]`) then stays byte-identical,
and the change cannot perturb a path it was not meant to touch. Captures sit at
`pc.n_obs + pc.d + <offset>`.

A storage buffer has none of the size problem: `maxStorageBufferRange` is
gigabytes, and the largest case here is 1192 x 9 x 8 B = 86 KB.

### Step 1 — assign offsets at registration (`glsl.ex`)

`register_capture/1` must return an accessor that already knows its offset, so
maintain a running total in the process dictionary beside the capture map. First
registration of a hash gets `offset = running_total`, and the total advances by
`n`. Registration is already idempotent by `phash2`, so a repeated capture
reuses its offset.

    accessor = "obs_inv_mass[pc.n_obs + pc.d + #{offset} + j]"

`collect_captures/0` then returns entries **in offset order**, each carrying its
`:offset`.

### Step 2 — one encoder, not two

`build_captured_decls/1` stops emitting `const double[]` and returns `""`; the
`{{captured_decls}}` placeholder collapses to the f64 helpers alone.

The packed binary **must be built from the same entry list that assigned the
offsets** — never re-derived from the tensors, never re-sorted. Two independent
encoders that agree by construction today and drift later is exactly defect D1 in
`BATCHED_CHAIN_DISPATCH.md`, where `dispatch.ex` grew a 5-clause copy of a
12-clause `Push.prior_param_floats/1`. Add `CustomSynth.captures_bin/1` next to
the collection point and let dispatch call it.

### Step 3 — carry it in the meta

    {:synthesised, sha, layout, push_spec, spv_path, obs_bin}

becomes a 7-tuple with `captures_bin` appended. `dispatch.ex:153` then packs

    extras_bin = obs_bin <> inv_mass_bin <> captures_bin

Fourteen destructuring sites across `lib/` and `test/` (`tree.ex:828`,
`dispatch.ex:119,136,272`, `sampler.ex:1650`, `batch_coordinator.ex:94`, plus
tests). All mechanical — only `dispatch.ex:136` actually *reads* element 6.

There is a cheaper alternative: keep arity 6 and redefine element 6 as the whole
static prefix `obs_bin <> captures_bin`, with layout `obs | captures | inv_mass`
and inv_mass moved to `pc.n_obs + <CAPT_TOTAL>`. One shader edit instead of
fourteen call sites. **Not recommended**: five sites bind that element as
`_obs_bin`/`_empty_obs`, and silently changing what it holds while the name says
otherwise is precisely the class of trap this file exists to close.

### Step 4 — the batched path

`@template` at `multi_rv_custom_spec.ex:1426` has its own `{{captured_decls}}`.
The batched renderer needs the same treatment, and its per-instance offsets must
account for the captures region. If batching is not being revived now, make
`synthesise_batched/1` **raise** on a non-empty capture set rather than silently
emit instance-0's data into a shader shared by every instance — which is the bug
its own comment predicts.

---

## The correctness question the implementation must answer first

Capture accessors are indexed by `@loop_index_var` = `"j"`, the reduce loop
variable. With `spans: :full` there is one loop over `[0, pc.n_obs)` and `j` is
the global observation index, so a capture aligned to the full obs buffer is
correct.

With **per-node spans** (`multi_rv_custom_spec.ex:940-946`) marker *i* ranges over
`[off_i, off_i + cnt_i)`. `j` is still a global index, so a capture that is
node-local rather than buffer-global would be read at the wrong offset. Establish
which it is **before** writing the packer, and put a fixture with two observed
nodes and a captured vector in the suite either way. Getting this wrong produces
a finite, plausible, wrong log-density — not a crash.

---

## Verification

The failure mode is a wrong posterior, so a green suite proves nothing.

1. **Equivalence gate.** Host `Compiler.compile/1` vs the composed synth
   log-density, 200 random q, on a model with captures. Expect **<= 1e-15**, not
   "close": moving data from a literal to a buffer does not change the
   arithmetic or its order, so byte equality is the right standard.
2. **Mutation check.** Perturb one capture's offset by 1 and assert the gate
   fails. A gate that has not been shown to fail is not evidence.
3. **Size regression — this is the actual bug, so assert it directly.**
   Synthesise `earnings-earn_height` (n_obs=1192) and `mesquite-logmesquite`
   (n_obs=46) and assert their SPIR-V sizes are within a small constant factor.
   Today they differ by orders of magnitude. Without this the fix can silently
   regress.
4. **The 21 crashing models synthesise and dispatch.** They will not all pass —
   see below — but none may fail with `dispatch_failed`.
5. **Two observed nodes plus a captured vector**, per the spans question above.
6. **Fleet.** Both Keplers and the Jetson. The Keplers are Vulkan-only and are
   the hosts that exercise this for real.

## What this fix does NOT do

Three distinct failures were found under `--compiler vulkan`. This plan addresses
**one**.

- **Pipeline size** — 21 models. This document.
- **Inf/NaN out of the chain shader** — at 300/300, `sblrc-blr`, `sblri-blr`,
  `mesquite-mesquite` and `mesquite-logmesquite_logvash` die in
  `Exmc.NUTS.NativeTree.build_subtree_bin/9` with `badarg`. The dumped binaries
  decode to `0x7FF0000000000000` (+Inf) and `0x7FFFFFFFE0000000` (NaN) in the
  gradient and momentum. These models are *small* — `n_obs x n_beta` of 322-500,
  far under the pipeline threshold — so this is a separate numerical bug and
  will survive this fix untouched.
- **Step-size collapse** (`eps -> 0.0`, ESS ~3). Partly a short-warmup artifact:
  `earnings-earn_height` shows the same collapse under **EXLA** at 50/50. Needs a
  full-protocol run before any of it is attributed to Vulkan.

Fixing the pipeline ceiling is what makes the other two *measurable* — today they
are hidden behind 21 models that never reach the sampler.

## Order

1. The spans question. It decides the packer, and it is free to answer now.
2. Steps 1-3, with verification 1-3. This is the fix.
3. Step 4, or the guard raise if batching stays parked.
4. Verification 4-6, then re-run posteriordb under Vulkan and re-triage what is
   left with the other two classes now visible.
