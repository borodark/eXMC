# macOS / EMLX Port: Benchmark Results & Patches

## Overview

Exmc was ported from Linux/EXLA (CPU f64 + XLA JIT + Rust NIF) to
macOS Apple Silicon / EMLX (Metal f32 + Evaluator fallback). This
document records the benchmark comparison and all patches required.

## Benchmark: EMLX vs EXLA on Apple Silicon M4

**Config:** 500 warmup + 500 draws, seed 42, single chain

| Model          | EXLA (ms) | EMLX (ms) | Slowdown | EXLA ESS/s | EMLX ESS/s | Div (E/M) |
|----------------|-----------|-----------|----------|------------|------------|-----------|
| simple (2d)    | 629       | 9,867     | 15.7x    | 414.5      | 25.6       | 11/9      |
| medium (5d)    | 1,432     | 65,667    | 45.9x    | 85.3       | 0.2        | 25/44     |
| stress (8d)    | 5,174     | 139,718   | 27.0x    | 26.9       | 1.1        | 14/25     |
| eight_schools  | 568       | 40,358    | 71.1x    | 592.3      | 9.1        | 7/9       |
| funnel (10d)   | 1,024     | 37,202    | 36.3x    | 3.0        | 0.2        | 24/103    |

### Why EMLX is slower

EMLX does not support `value_and_grad` on closures containing captured
tensors. When Exmc calls `Nx.Defn.value_and_grad(q, logp_fn)`, EMLX
falls back to `Nx.Defn.Evaluator` which:

1. **No JIT compilation** — every Nx op executes as an individual
   BinaryBackend call instead of a fused XLA kernel
2. **No batched leapfrog** — the `while` loop in `BatchedLeapfrog`
   compiles through Evaluator, losing the O(1)-dispatch advantage
3. **No Rust NIF tree** — disabled because the NIF expects f64 binary
   layout; the f32↔f64 conversion overhead would dominate anyway
4. **f32 precision** — more divergences due to reduced numerical range,
   especially on funnel-geometry models

### What EMLX still provides

- **Correct results** — posterior means match EXLA within f32 tolerance
- **Full API compatibility** — same `Sampler.sample` call, zero code changes
- **Apple Silicon native** — no Rosetta, no Docker, instant `mix compile`

## Patches Required for macOS/EMLX Support

### 1. Precision Portability (8 files)

All hardcoded `:f64` type annotations replaced with `Exmc.JIT.precision()`,
which returns `:f32` for EMLX, `:f64` for EXLA.

**Files:** `tree.ex`, `sampler.ex`, `mass_matrix.ex`, `leapfrog.ex`,
`batched_leapfrog.ex`, `distributed.ex`, `point_map.ex`

### 2. NIF Binary Boundary (`tree.ex`)

The Rust NIF reads f64 binaries (8 bytes/value). Two new helpers:
- `to_nif_binary/1` — casts f32 tensors to f64 before `Nx.to_binary()`
- `from_nif_binary/2` — decodes f64 from Rust, reshapes, casts back to working precision

### 3. Numerical Safety — Transform Clamping (`transform.ex`)

`apply(:log, z)` now clamps z before `Nx.exp()`:
- f32: `[-20, 20]` → sigma ∈ [2e-9, 5e8]
- f64: `[-200, 200]` → sigma ∈ [1e-87, 7e86]

Uses `Nx.max`/`Nx.min` instead of `Nx.clip` because **`Nx.clip` has broken
gradient in Evaluator autodiff** (returns wrong gradient when composed with
other ops inside `value_and_grad` closures).

### 4. Numerical Safety — Distribution Scale Guards (10 files)

All distributions that divide by a scale parameter now floor it at `1e-30`:

```elixir
safe_sigma = Nx.max(sigma, Nx.tensor(1.0e-30))
```

**Protected:** Normal, HalfNormal, Laplace, Cauchy, HalfCauchy, Lognormal,
StudentT, TruncatedNormal, GaussianRandomWalk, Censored

**Why:** On BinaryBackend, Erlang arithmetic throws `ArithmeticError` on
divide-by-zero (unlike GPU which silently returns NaN/Inf per IEEE 754).

### 5. Tree Builder Crash Resilience (`tree.ex`)

**A. try/rescue in build_subtree base case:**
The `step_fn` call is wrapped in try/rescue. On `ArithmeticError`, returns
a divergent leaf at the starting position instead of crashing the tree.

**B. NaN-safe divergent leaf:**
When `joint_logp` is NaN (detected via `is_number/1` guard), the leaf
falls back to original `q`/`p` for all flat lists. This prevents `:nan`
atoms (from `Nx.to_flat_list`) from reaching Erlang `+` in `zip_add`.

### 6. Batched Leapfrog Type Cast (`batched_leapfrog.ex`)

After `value_and_grad` inside the `while` loop, results are cast back to
target precision:

```elixir
{logp_new, grad_new} = value_and_grad(q_new, logp_fn)
logp_new = Nx.as_type(logp_new, fp)
grad_new = Nx.as_type(grad_new, fp)
```

The Evaluator returns f64 (BinaryBackend default) even when the loop
accumulator tensors are f32, causing a while-loop type mismatch.

### 7. Notebooks (4 files)

- Conditional backend: `{:emlx, "~> 0.2"}` on macOS, `{:exla, "~> 0.10"}` elsewhere
- Replaced custom `Chart` module with `kino_vega_lite` for native Livebook rendering
- Added persisted outputs for self-documenting notebooks

### 8. Livebook Launch (`start_livebook.sh`)

Replaced Docker/nerdctl launch with native `livebook` escript.

## Nx/EXLA/EMLX Gotchas Discovered

| Issue | Workaround |
|-------|------------|
| `Nx.clip` broken gradient in Evaluator | Use `Nx.max`/`Nx.min` |
| `Nx.to_number` returns `:nan`/`:infinity` atoms | Guard with `is_number/1` |
| `Nx.to_flat_list` returns `:nan` atoms | Erlang `+` on `:nan` crashes |
| EMLX `value_and_grad` on closures | Falls back to Evaluator (slow) |
| BinaryBackend ArithmeticError on Inf/NaN | Clamp inputs, try/rescue |
| Evaluator returns f64 in f32 while loops | Cast outputs with `Nx.as_type` |

## Test Results

- **262 tests + 11 doctests: 0 failures** (EXLA backend)
- **4 Livebook notebooks: all pass end-to-end** (EMLX backend)
- **5 benchmark models: all complete** (EMLX backend, with numerical guards)
