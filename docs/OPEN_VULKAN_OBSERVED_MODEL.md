# ~~Open defect~~ FIXED: the vulkan chain path was wrong for observed models

**Found:** 2026-08-16, while verifying the 0.3.1 P0 backport.
**Status:** **fixed** 2026-08-16 in `MultiRvCustomSpec`. NOT introduced by the
0.3.1 fixes — but they changed its symptom, which is what made it visible
instead of merely wrong.
**Was blocking:** claiming `compiler: :vulkan` is correct for anything with
observations.

## The answer, up front

The "next experiment" below was run, and it found the fault on **this** side of
the NIF. `compose_logp_defn/1` gave **every** observed node the **whole**
observation vector:

```elixir
lp = mod.logpdf(obs, resolved) |> Nx.sum()   # obs = the ENTIRE buffer
```

With three separate `Builder.obs` nodes, the likelihood was therefore counted
three times over — 3× the log-density *and* 3× the gradient. The emitter then
faithfully turned each node's `Nx.sum` into its own `for (j < pc.n_obs)` loop.
`observed_obs_bin/1` had always concatenated the nodes' values in iteration
order and its comment claimed the read side matched ("matches the order
compose_logp_defn reads them"); nothing enforced it, and it did not.

That also explains the freeze without needing the step-size lead below: a
likelihood counted 3× is a posterior ~sqrt(3) too narrow with 3× steeper
gradients, so eps = 1.139 was far past stable and acceptance collapsed to
~0.002. **The "adapted eps identical to sixteen digits" observation was a
saturated adaptation reporting a real problem, not a second bug.** It is also
why the *vector* arm was correct all along: one obs node, one loop, no
double-count.

### The fix

`transform_reduce_sum/3` now takes per-marker `{offset, count}` spans, so each
observed node's loop ranges over its **own** slice of the buffer. `:full` (the
old whole-buffer bound) is kept for the two cases where it is the correct
reading: a single observed node, which owns the buffer by definition, and a
Custom likelihood, whose markers cannot be attributed positionally.

**One trap, recorded because it nearly shipped.** Attribution is positional,
and the gradient's markers arrive **mirrored** relative to the forward
log-density's — reverse-mode AD walks `compose_logp_defn/1`'s left fold
backwards, so `_gacc*_0` is the LAST observed node while `_lpacc0` is the
first. With all three observations `Normal(mu, 1)` a mirrored assignment gives
a **bit-identical** answer, so the first version of this fix looked correct and
was not. `bench/leapfrog_leaf_diff.exs` therefore uses **distinct per-node
sigmas** (1.0 / 2.0 / 3.0), where any permutation changes the numbers. Keep it
that way. A marker-count guard raises (degrading to `:unsupported`, i.e. the
slower host path) if the correspondence ever breaks.

### Verification

`bench/leapfrog_leaf_diff.exs` — all four arrays agree with the host leapfrog
to ~1e-15 across three (eps, q0, p0) settings, and the logp offset is constant
along the trajectory, so the Metropolis ratio is equivalent.

The posterior, 300 warmup + 500 samples, seed 42, `compiler: :vulkan`:

| | mean | sd | distinct draws |
|---|---|---|---|
| analytic | 3.99 | 0.577 | — |
| **after the fix** | **3.966** | **0.539** | **469 / 500** |
| before | 3.650 | 3.3e-14 | 1 / 500 |

`test/integration_test.exs:639` ("vector obs produces same posterior as
equivalent scalar obs"), left deliberately red, now passes.

### Still open, found alongside

* The observation buffer and prior params are **f32-rounded** on a nominally
  f64 path — `3.8` arrives as `3.799999952316284`. Shared by host and GPU (both
  read the same f32 IR tensors), so it does not show up as a divergence, but it
  is not f64.
* `transform_reduce_sum_batched/2` (the Task #154 batched path) carries the
  **same** whole-buffer defect and was NOT fixed here. It is currently
  unreachable — `Nx.Vulkan.NativeV.leapfrog_chain_synth_batch_f64/6` is
  undefined, which the compiler warns about on every build.

---

## Original write-up

**Status at the time:** open.

## What happens

Model — `integration_test.exs:611`, the scalar arm:

```elixir
Builder.new_ir()
|> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(10.0)})
|> Builder.rv("x1", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
|> Builder.obs("x1_obs", "x1", Nx.tensor(4.0))
|> Builder.rv("x2", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
|> Builder.obs("x2_obs", "x2", Nx.tensor(3.8))
|> Builder.rv("x3", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
|> Builder.obs("x3_obs", "x3", Nx.tensor(4.2))
```

Analytic posterior for `mu`: mean **3.99**, sd **0.577**.
300 warmup + 500 samples, seed 42:

| arm | mean | sd | distinct draws | ε |
|---|---|---|---|---|
| `compiler: :none` | 3.970 | 0.552 | — | 1.0838 |
| `compiler: :vulkan`, 0.3.1 | 3.650 | **3.3e-14** | **1 / 500** | 1.1391216000810296 |
| `compiler: :vulkan`, 0.3.0 + tree fix only | 5.194 | 0.696 | — | 1.1391216000810296 |
| `compiler: :vulkan`, 0.3.0 | — | 1.779 | — | — |

The host path is correct. **Every** vulkan variant is wrong; they differ only
in how. At 0.3.1 the chain does not move at all: 1 distinct value in 500 draws,
with `accept_prob` around **0.0023** per iteration.

## What has been ruled out

- **The template reorder is not the cause.** The generated GLSL was read
  directly. `lp_i` is assigned from `q_shared[0]` *after* the post-update store
  (`if (in_bounds) q_shared[tid] = qi; barrier();`), so it describes exactly the
  state written to `q_chain[k]` on the same iteration. The invariant the fix
  was meant to establish does hold. Reverting the fix does not make the arm
  correct — it makes it wrong in the other direction (mean 5.19 against 3.99).
- **`n_obs` packing.** `detect_meta/1` reports `d=1, n_obs=3` and a 24-byte
  observation buffer (3 × f64) for this model. Correct.
- **The tree fix.** `compiler: :none` runs the same fixed tree and lands on the
  analytic answer.
- **Step-size adaptation being merely unlucky.** See below — it is not adapting.

## The strongest lead

ε is **1.1391216000810296 in both vulkan rows, identical to sixteen digits**,
across a change that alters the log-density of every leaf in every trajectory.

Dual averaging is driven by accept probabilities, and accept probabilities come
from those densities. If the densities all change and the adapted step size does
not move even in the last bit, warmup on this path is not consuming the shader's
`logp_chain` output. That would also explain the freeze: ε = 1.139 against a
posterior sd of 0.577 is roughly 2σ per step, which a working adaptation would
have shrunk, and which produces exactly the ~0.002 acceptance seen.

Note the direction of causation. Before the logp fix, `logp_chain[k]` described
the state *before* step k — so leaf 0 reported the density at `q0`, the same
position `joint_logp_0` is measured at. The Metropolis ratio for the first leaf
therefore looked healthy for the wrong reason, and masked a step size that was
already far too large. Fixing the lag removed the mask. **A low acceptance rate
here is the correct report of a real problem, not a new one.**

## Next experiment

Differential the shader against the host at the leaf level, which no existing
test does:

1. Fix `q0`, `p0`, `eps`, `inv_mass`, `K = 32`.
2. Dispatch `leapfrog_chain_synth_f64` and read back `q_chain`, `p_chain`,
   `grad_chain`, `logp_chain`.
3. Run the same K leapfrog steps on the host via `Exmc.Compiler`'s `vag_fn`.
4. Compare all four arrays element-wise.

If they agree, the defect is in how the host consumes the arrays (step-size
adaptation being the first place to look, per above). If they diverge, the
step index at which they first diverge names the bug.

This is the check that would have caught both 0.3.1 defects at the point of
introduction, and it is worth building regardless of what it finds here.

## Scope note

`compiler: :vulkan` is the **default**. Until this is resolved, any observed
model sampled on the default compiler can silently return a degenerate
posterior. `Exmc.NUTS.Vulkan.Validator.check_analytic/3` will catch it —
`bench/nuts_truth.exs` is the entry point — but nothing runs that automatically
yet. That is P1's job (see `MISSION.md` §7).
