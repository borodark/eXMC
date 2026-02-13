# eXMC — macOS Build & Test Guide

Tested on macOS (Darwin 25.2.0, Apple Silicon). This guide covers everything needed to compile eXMC (including the Rust NIF) and run the full test suite and Livebook notebooks.

## Prerequisites

### 1. Homebrew

If not already installed:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Erlang/OTP 27+

Option A — Homebrew (simplest):

```bash
brew install erlang
```

Option B — kerl (version manager, recommended for development):

```bash
brew install kerl
kerl build 27.3.4.6
kerl install 27.3.4.6 ~/.kerl/27.3.4.6
source ~/.kerl/27.3.4.6/activate
```

Verify: `erl -noshell -eval 'io:format("~s~n", [erlang:system_info(otp_release)]), halt().'` should print `27`.

### 3. Elixir 1.18+

Option A — Homebrew:

```bash
brew install elixir
```

Option B — kiex (version manager):

```bash
curl -sSL https://raw.githubusercontent.com/taylor/kiex/master/install | bash -s
kiex install 1.18.4
kiex use 1.18.4
```

Verify: `elixir --version` should show Elixir 1.18+ compiled with Erlang/OTP 27.

### 4. Rust (required for the NIF tree builder)

```bash
brew install rust
```

This puts `rustc` and `cargo` on `/opt/homebrew/bin/` which is already on PATH for Homebrew users. The Rustler build system invokes `cargo` during `mix compile`.

Verify: `rustc --version` should show 1.70+.

### 5. cmake (required for EXLA/XLA compilation)

```bash
brew install cmake
```

## Summary of brew installs

If starting from scratch, the minimal set:

```bash
brew install erlang elixir rust cmake
```

## Build

```bash
cd exmc
mix deps.get
mix compile
```

The first `mix compile` will:
- Download and compile all Elixir dependencies (Nx, EXLA, Rustler, etc.)
- Build the Rust NIF (`native/exmc_tree/`) via Rustler — produces `priv/native/libexmc_tree.so`
- EXLA will download and cache the XLA binary (~200MB, first time only)

## Test

```bash
mix test
```

Expected: **262 tests, 0 failures** in ~14 seconds. Without the Rust NIF compiled, 9 NativeTree tests will fail and test time increases to ~96 seconds (falls back to pure Elixir tree builder).

To run without GPU (recommended for testing):

```bash
CUDA_VISIBLE_DEVICES="" mix test
```

## Livebook Notebooks

The `notebooks/` directory contains 4 interactive notebooks demonstrating eXMC features.

### Install Livebook

```bash
mix escript.install hex livebook
```

Or run directly without installing:

```bash
mix run --no-halt -e 'Application.ensure_all_started(:livebook)'
```

The simplest approach — run Livebook as a standalone:

```bash
livebook server --open
```

Then open any notebook from `exmc/notebooks/`:

| Notebook | Topics |
|----------|--------|
| `01_getting_started.livemd` | Model building, NUTS sampling, trace/histogram plots |
| `02_hierarchical_model.livemd` | Hierarchical models, NCP, multi-chain, R-hat, forest plots |
| `03_model_comparison.livemd` | WAIC, LOO-CV, model comparison tables |
| `04_variational_inference.livemd` | NCP vs centered, multi-chain diagnostics, Lognormal |

Notebooks use `kino_vega_lite` for chart rendering — VegaLite specs auto-render natively in Livebook cells.

## Troubleshooting

**`cargo` not found during compile:**
Ensure Rust is installed via `brew install rust` and `/opt/homebrew/bin` is on your PATH.

**EXLA compilation fails:**
Make sure `cmake` is installed: `brew install cmake`. EXLA downloads a prebuilt XLA binary for most platforms, but cmake is needed for the Elixir NIF wrapper.

**NIF tests fail (9 failures):**
The Rust NIF wasn't compiled. Run `mix clean && mix compile` and verify `priv/native/libexmc_tree.so` exists.

**Livebook can't find exmc:**
Notebooks use `{:exmc, path: Path.expand("../", __DIR__)}` — they must be opened from within the `exmc/notebooks/` directory structure. Open them via Livebook's file browser, not by URL.
