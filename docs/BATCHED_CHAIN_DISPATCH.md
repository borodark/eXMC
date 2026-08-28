# Batched chain dispatch — where it belongs, and what it would take

Status as of 2026-08-28: **inert and structurally incomplete.** Nothing in
production reaches it, and it could not work if it did.

## What exists

| piece | where | state |
| --- | --- | --- |
| batched GLSL renderer | `MultiRvCustomSpec.render_batched/1` | **done**, f64, per-instance `obs_j` slices |
| batched synth meta | `CustomSynth.synthesise_batched/1` | **done**, compiles SPV, enforces the 128 B push cap |
| request coalescer | `NUTS.Vulkan.BatchCoordinator` | **done**, timer + size flush, per-caller reply |
| dispatch | `NUTS.Vulkan.Dispatch.chain_batch/5` | **packs f64, calls a NIF that does not exist** |
| the NIF | `nx_vulkan` | **never written** |
| activation | `tree.ex` `:exmc_chain_coord` | **nothing ever sets it** |

`synthesise_batched/1` has no callers anywhere in `lib/` or `test/`. Nothing
sets `:exmc_chain_coord`, so `tree.ex:865` never takes the batching branch.
The feature is unreachable from a real sample.

## Does BatchCoordinator belong in nx_vulkan?

Mostly no. Decompose it:

1. **Coalescing mechanism** — accumulate requests, group by a compatibility
   key, flush on size-or-timer, scatter results back to callers. Domain-free.
   Any `nx_vulkan` consumer would want this.
2. **Compatibility key** — same SPV, same `K`, same `|ε|`, same `dir_sign`.
   Half shader identity, half NUTS.
3. **Payload** — `{q, p, inv_mass, obs}` in, `{q_chain, p_chain, logp_chain,
   grad_chain}` out. Pure HMC.
4. **Fallback policy** — `{:fallback, reason}` routes back to
   `route_chain_direct`. An `exmc` sampling decision, and a correctness-critical
   one: a dropped batch must become an unbatched draw, never a missing draw.
5. **The f64 batch NIF** — `nx_vulkan`'s, unambiguously.

Only (5), and optionally a generic form of (1), belong in the backend. Items
2–4 are sampler policy. Moving the coordinator wholesale would push `dir_sign`,
`ε` and leapfrog trajectories into a general-purpose Nx backend.

**Worth naming, since it is the reason the question feels ambiguous:** that
layering violation already happened. `nx_vulkan` hosts
`leapfrog_chain_synth{,_f64,_batch}` — a NUTS integrator living inside a
general-purpose compute backend. The batched work now inherits that boundary.
Two ways forward.

---

## Option 1 — follow the existing boundary (recommended)

Keep `BatchCoordinator` in `exmc`. Add the missing NIF to `nx_vulkan`.

### nx_vulkan

Add `leapfrog_chain_synth_batch_f64/6`. It is a merge of two functions that
already sit 150 lines apart in `native/nx_vulkan_vulkano/src/lib.rs`:

- `leapfrog_chain_synth_batch/6` (~143 lines) — the batching, `n_instances`
  push header, per-instance buffer offsets, one `vkQueueSubmit`
- `leapfrog_chain_synth_f64/6` (~136 lines) — the f64 strides and
  `parse_push_block_f64`

The delta is arithmetic: 4-byte strides become 8, `parse_push_block_batch`
gains an f64 variant (its `eps` field is already f64), and `chain_bytes`
becomes `n_instances * k * d * 8`. Add the Elixir stub in `native_v.ex`.

The shader is **not** needed on this side — `exmc` renders and compiles the
batched f64 GLSL itself and passes an SPV path.

Estimate: one file, ~150 lines, no new concepts.

### exmc

Nothing to change for the NIF itself: `Dispatch.chain_batch/5` already calls it
via `apply/3` and starts working the day it lands. The work is the four defects
underneath it.

---

## Option 2 — fix the layering

Give `nx_vulkan` a generic `dispatch_spv(buffers, push, workgroups, spv_path)`
and move the whole leapfrog family — f32, f64, batched — into `exmc`, which
already owns the GLSL that those NIFs execute.

Better boundary, and it removes the reason this question keeps coming up. Much
larger: touches both repos, every chain dispatch path, and the whole fleet
revalidates. Not the right change to make while the f64 batch path has never
run once.

Recommendation: Option 1 now, Option 2 recorded as the real fix.

---

## The four defects underneath, independent of which option

These are `exmc`'s in both options, and three of them are silent.

### D1 — a second push encoder, with fewer distributions

`Push.prior_param_floats/1` has **12** clauses. `dispatch.ex` hand-rolls a
private copy with **5**, and raises for anything else:

    defp prior_param_floats({id, mod, _}),
      do: raise("chain_batch: no prior_param_floats clause for #{id}")

So the batched path silently supports fewer models than the single-instance
path. A model that samples fine unbatched raises the moment batching turns on.

Fix: make `Push.prior_param_floats/1` public and delete the copy. One source of
truth for the push layout.

### D2 — the batched push has no 128-byte cap

`chain_batch` builds `push = header <> prior_bin` with no length check.
`Push.pack/1` returns `{:error, :push_too_large}` above 128 bytes; this path
just hands an oversized block to the NIF, which rejects it with
`{:error, :bad_input}`, which fails the `{:ok, {...}} =` match as a `MatchError`
that names nothing useful.

Note `synthesise_batched/1` *does* check the cap at synth time — but with
`K: 32, eps: 0.05` placeholders, not the values dispatch actually packs. The
check and the pack are different code paths.

Fix: route the batched push through `Push.pack/1` too, or give it the same
guard and error.

### D3 — one dispatch site rescues, the other does not

`batch_coordinator.ex:428` (`request_synth_chain`, the path `tree.ex` uses)
wraps `Dispatch.chain_batch` in `try/rescue` and replies `{:fallback, reason}`.
`batch_coordinator.ex:313` (`do_flush_group`, the older `request_chain` path)
does not. A raise there kills the coordinator GenServer, and every caller
waiting on it, instead of falling back.

Fix: same `try/rescue` at both sites.

### D4 — activation was never wired

`route_chain` batches only when `Process.get(:exmc_chain_coord)` returns
`{coord_pid, obs}`. Nothing sets it. The comment points at "BatchedSampler
before sample_compiled in Step 3" — Step 3 was never done.

Decide explicitly: wire it, or retire the path. What must not continue is a
feature that looks finished, has tests, and cannot run.

---

## Verification, when the NIF lands

The failure mode here is a **plausible but wrong** posterior, so a green suite
proves nothing on its own.

1. **Equivalence gate.** Same model, same seed, same `K`, same `ε`: N
   single-instance dispatches vs one batched dispatch of the same N. Compare
   `q_chain`, `p_chain`, `grad_chain`, `logp_chain` elementwise. Expect ≤ 1e-15,
   not "close" — the arithmetic is identical, only the buffer offsets differ.
2. **Mutation check.** Permute one instance's slice offset and confirm the gate
   fails. A gate that has not been shown to fail is not evidence.
3. **Heterogeneous batch.** N instances with genuinely different `q`, `p`,
   `inv_mass` and `obs`. A batch of identical instances passes even when
   per-instance offsets are wrong — that is the bug this shape exists to catch.
4. **Partial flush.** Fewer instances than `batch_size`, flushed by timer. The
   `n_instances` push field must match what was actually packed.
5. **Fallback still works.** With the NIF present, force a failure and confirm
   callers still get `{:fallback, _}` and the coordinator survives — at **both**
   dispatch sites (see D3).
6. **Fleet.** Both Keplers and the Jetson. The Keplers are Vulkan-only, so they
   are the ones that exercise this path for real.

## Order

1. D3 and D1 — small, independent of the NIF, and D1 removes a silent
   capability gap. Do these first regardless of what happens with batching.
2. D2 — same, slightly more thought.
3. nx_vulkan NIF (Option 1), pushed to its origin, then `mix deps.update
   nx_vulkan` here.
4. Verification 1–4 locally, then D4 (the activation decision), then 5–6.
