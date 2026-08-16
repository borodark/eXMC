# Open defect: the vulkan chain path is wrong for observed models

**Found:** 2026-08-16, while verifying the 0.3.1 P0 backport.
**Status:** open. NOT introduced by the 0.3.1 fixes — but they changed its
symptom, so it is now visible instead of merely wrong.
**Blocks:** claiming `compiler: :vulkan` is correct for anything with
observations. It is not.

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
