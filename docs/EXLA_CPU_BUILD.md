# Building a CPU EXLA

For hosts where the CUDA build of `exla` is installed but cannot load — the
common symptom being

```
Failed to load NIF library .../priv/libexla:
  'libnvshmem_host.so.3: cannot open shared object file: No such file or directory'
```

eXMC no longer *needs* you to fix this. `exla` is declared `runtime: false`
(see the `exla_dep` comment in `mix.exs`), so a broken EXLA makes
`Exmc.JIT.detect_compiler/0` fall through to the next backend instead of
aborting the VM at boot. Everything below is for when you want EXLA working,
not for when you want the test suite to run.

## The recipe

```sh
# 1. Clear the stale C++ objects and every cached libexla.so — see trap 2.
rm -rf deps/exla/cache/0.13.1/objs deps/exla/cache/libexla.so ~/.cache/xla/exla

# 2. Build, once per MIX_ENV.
EXLA_CPU_ONLY=1 XLA_TARGET=cpu mix deps.compile exla --force
EXLA_CPU_ONLY=1 XLA_TARGET=cpu MIX_ENV=test mix deps.compile exla --force

# 3. Verify. Both should print nothing.
ldd _build/test/lib/exla/priv/libexla.so | grep 'not found'
nm -D --undefined-only _build/test/lib/exla/priv/libexla.so | grep -i cuda
```

Neither variable is remembered — both are read at dep-compile time, so they
must be set on every forced rebuild, in each environment separately. A later
bare `mix deps.compile exla --force` silently gives you the CUDA build back.

## Trap 1 — `XLA_TARGET=cpu` alone is not enough

It selects the CPU `xla_extension` archive, but exla's Makefile gates the CUDA
custom calls on `EXLA_CPU_ONLY` and probes for `nvcc` independently:

```
EXLA_CPU_ONLY is not set, checking for nvcc availability
CUDA is available.
```

So on any host with the CUDA toolkit installed it still compiles
`c_src/exla/custom_calls/runtime_callback_cuda.cc` with `-DCUDA_ENABLED`, which
fails against g++ 13 with

```
error: no matching function for call to 'exla::callback_bridge::OutputBuffer::OutputBuffer()'
```

— an error about a C++ default constructor, in a CUDA file, from what you asked
to be a CPU build. `EXLA_CPU_ONLY=1` is what produces
`EXLA_CPU_ONLY is set, skipping nvcc step` and drops the `-DCUDA_ENABLED`.

## Trap 2 — `--force` does not force the C++ objects

`--force` re-runs `make`, and make keeps any `.o` newer than its source.
Objects left in `deps/exla/cache/<version>/objs/` from an earlier
`-DCUDA_ENABLED` compile get linked into the CPU `libexla.so`, which then
builds, installs, and reports success — and fails at load with

```
undefined symbol: cudaGetErrorString
```

`~/.cache/xla/exla/<elixir-erts-xla-exla>/libexla.so` compounds it. That key
does **not** include the target, so a stale entry is copied straight back into
a fresh `_build` and survives `rm -rf _build/` entirely. Step 1 above clears
both.
