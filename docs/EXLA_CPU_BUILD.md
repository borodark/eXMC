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

## Before building a CPU EXLA: the CUDA build may only need its libraries found

On super-io the CUDA `libexla.so` is fine; four of the libraries it links are
simply not on the loader's path. See which:

```sh
ldd _build/test/lib/exla/priv/libexla.so | grep 'not found'
```

There (measured 2026-09-12) that prints `libnvshmem_host.so.3`,
`nvshmem_bootstrap_uid.so.3`, `nvshmem_transport_ibrc.so.3` and
`libnvrtc-builtins.so.12.9`. All four ship in pip wheels under the
**python3.12** site-packages — not the python3.10 tree holding the other
`nvidia/*` wheels, where searching finds nvrtc 12.1 and no nvshmem, which reads
as "not installed":

```sh
find / -name 'libnvshmem_host.so.3' 2>/dev/null
```

Make them visible to the system loader, once per host. It takes two steps,
because `ldconfig` only indexes files whose names start with `lib`, and two of
the four do not:

```sh
NV=/home/io/.local/lib/python3.12/site-packages/nvidia
# 1. the lib-prefixed ones (libnvshmem_host.so.3, libnvrtc-builtins.so.12.9)
printf '%s\n' "$NV/nvshmem/lib" "$NV/cuda_nvrtc/lib" |
  sudo tee /etc/ld.so.conf.d/zz-nvidia-pip-wheels.conf
sudo ldconfig
# 2. the two ldconfig will never index; the loader still searches its
#    built-in directories by exact name (`ld.so --help` lists them)
sudo ln -s $NV/nvshmem/lib/nvshmem_bootstrap_uid.so.3 \
           $NV/nvshmem/lib/nvshmem_transport_ibrc.so.3 /usr/lib/x86_64-linux-gnu/
# from the exmc checkout root; prints nothing
ldd _build/test/lib/exla/priv/libexla.so | grep 'not found'
```

Step 1 alone leaves `nvshmem_bootstrap_uid.so.3` and
`nvshmem_transport_ibrc.so.3` unresolved; that was measured on 2026-09-12,
after an earlier version of this page gave step 1 as the whole fix. Both are
`NEEDED` by `libxla_extension.so`, not by `libexla.so`. The distro package
`libnvshmem3-cuda-12` (3.7.2), if installed, does not help: it keeps its files
under `nvshmem/12/`, where nothing searches, and ships
`nvshmem_transport_ibrc.so.6`, while XLA was linked against `.so.3`.

No new shell is needed after either step. Neither touches the environment; the
loader reads the cache and its directories when each process starts.

A per-checkout alternative with no sudo: `libxla_extension.so`'s RUNPATH
already looks for a pip layout at `$ORIGIN/../../nvidia`, so `ln -s $NV
deps/exla/cache/nvidia` also works. But every checkout needs its own link, and
`mix deps.clean exla` removes it.

**Why the loader and not `LD_LIBRARY_PATH`.** This host ran on an exported
`LD_LIBRARY_PATH` for a month, and it failed the same way every time: a shell
that had not exported it — an agent, `nohup`, cron, a fresh terminal — got a
different backend, silently. The loader reads that variable once at process
start, so nothing inside the BEAM can set it, and `test/distributed_test.exs`
starts `:exla` on `:peer` nodes that inherit whatever the launching shell had.
The loader's own configuration has no such gap. Verified 2026-09-12 from a
non-interactive shell with no `LD_LIBRARY_PATH`: EXLA starts, `client: :cuda`
computes on the RTX 3060 Ti, and `Exmc.JIT.describe/0` reports `compiler=EXLA`.

**Why `zz-`.** The `cuda_nvrtc` wheel also carries `libnvrtc.so.12` (12.9),
and the system has its own (12.6, `/usr/local/cuda`). Files are read in sort
order and the first directory providing a soname wins, so sorting after
`000_cuda.conf` and `988_cuda-12.conf` leaves every other CUDA program on the
box on 12.6 and only adds the sonames nothing else provides. Measured with
both orders: `Nx.Defn.jit(..., compiler: EXLA, client: :cuda)` returns the
same f64 result either way (`908.1872256586632` for
`sum(exp(x) * sin(x))`, `x = iota(1000) / 1000`) on the RTX 3060 Ti.

Once EXLA loads, `Exmc.JIT` auto-detection picks it over Vulkan, so a bare
`mix test` on this host is the EXLA arm; name `EXMC_COMPILER=vulkan` for the
other one. Tests still run EXLA on the host client (`config/test.exs`);
`client: :cuda` is opt-in. The CUDA client preallocates 90% of the card and
logs `CUDA_ERROR_OUT_OF_MEMORY` when the desktop already holds some of it,
then continues with less — noise, not a failure.

If the libraries are genuinely absent, build for the CPU instead:

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
