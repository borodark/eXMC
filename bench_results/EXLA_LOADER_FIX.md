# EXLA loader fix on super-io — record

What was changed on the host so the CUDA `exla` NIF loads with no environment
variables. Facts only; the reasoning is in `docs/EXLA_CPU_BUILD.md`.

## Host

| | |
|---|---|
| host | super-io, Linux Mint 22.2, Linux 6.8.0-139-generic x86_64 |
| glibc | 2.39 (Ubuntu GLIBC 2.39-0ubuntu8.9) |
| GPU / driver | NVIDIA GeForce RTX 3060 Ti / 580.178.04 |
| CUDA toolkit | 12.6 at `/usr/local/cuda` |
| Erlang / Elixir | OTP 27, erts 15.2.7.2 / Elixir 1.18.4 |
| exla / xla | exla 0.13.1 (hex), `xla_extension-0.10.0-x86_64-linux-gnu-cuda12` |
| pip wheels | `nvidia-nvshmem-cu12` 3.5.19, `nvidia-cuda-nvrtc-cu12` 12.9.86, under `/home/io/.local/lib/python3.12/site-packages/nvidia` |
| distro nvshmem | `libnvshmem3-cuda-12` 3.7.2-1, files in `/usr/lib/x86_64-linux-gnu/nvshmem/12/`, not used |
| exmc | `gate1/reconcile-core` @ `d5f6abb4a` |

## Symptom before

Any process without `LD_LIBRARY_PATH` (nothing on the host set it):

    $ ldd _build/test/lib/exla/priv/libexla.so | grep 'not found'
    	libnvshmem_host.so.3 => not found
    	nvshmem_bootstrap_uid.so.3 => not found
    	nvshmem_transport_ibrc.so.3 => not found
    	libnvrtc-builtins.so.12.9 => not found

    Failed to load NIF library .../_build/test/lib/exla/priv/libexla:
      'libnvshmem_host.so.3: cannot open shared object file: No such file or directory'

`Exmc.JIT` auto-detect fell through to `Nx.Vulkan`; `test/distributed_test.exs`
peer nodes logged the failure above on every run.

## Where the names come from

    $ readelf -d deps/exla/cache/xla_extension/lib/libxla_extension.so | grep 'NEEDED.*nvshmem'
     (NEEDED) Shared library: [libnvshmem_host.so.3]
     (NEEDED) Shared library: [nvshmem_bootstrap_uid.so.3]
     (NEEDED) Shared library: [nvshmem_transport_ibrc.so.3]

`libnvrtc-builtins.so.12.9` comes in through the same closure. The system
CUDA has only `libnvrtc-builtins.so.12.6` and `.12.0`. The distro nvshmem
ships `nvshmem_transport_ibrc.so.6`, not `.so.3`.

## Changes made, 2026-09-12, by the operator with sudo

**1. 19:57:44 -0400**: loader config entry, then `ldconfig`.

    $ cat /etc/ld.so.conf.d/zz-nvidia-pip-wheels.conf
    /home/io/.local/lib/python3.12/site-packages/nvidia/nvshmem/lib
    /home/io/.local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib

Result: `libnvshmem_host.so.3` and `libnvrtc-builtins.so.12.9` resolved, and
`libnvrtc.so.12` still resolved to `/usr/local/cuda` (12.6). Still missing:

    	nvshmem_bootstrap_uid.so.3 => not found
    	nvshmem_transport_ibrc.so.3 => not found

`ldconfig` does not index file names without a `lib` prefix.

**2. 20:00:52 -0400**: two symlinks in a built-in loader directory.

    NV=/home/io/.local/lib/python3.12/site-packages/nvidia
    sudo ln -s $NV/nvshmem/lib/nvshmem_bootstrap_uid.so.3 \
               $NV/nvshmem/lib/nvshmem_transport_ibrc.so.3 /usr/lib/x86_64-linux-gnu/

    lrwxrwxrwx root root /usr/lib/x86_64-linux-gnu/nvshmem_bootstrap_uid.so.3 -> $NV/nvshmem/lib/nvshmem_bootstrap_uid.so.3
    lrwxrwxrwx root root /usr/lib/x86_64-linux-gnu/nvshmem_transport_ibrc.so.3 -> $NV/nvshmem/lib/nvshmem_transport_ibrc.so.3

## Verification after, 20:01 -0400

From a non-interactive shell, `LD_LIBRARY_PATH` empty:

    $ ldd _build/test/lib/exla/priv/libexla.so | grep 'not found'
    (no output)
    $ ldd _build/test/lib/exla/priv/libexla.so | grep -E 'nvshmem|nvrtc'
    	libnvshmem_host.so.3 => $NV/nvshmem/lib/libnvshmem_host.so.3
    	nvshmem_bootstrap_uid.so.3 => /lib/x86_64-linux-gnu/nvshmem_bootstrap_uid.so.3
    	nvshmem_transport_ibrc.so.3 => /lib/x86_64-linux-gnu/nvshmem_transport_ibrc.so.3
    	libnvrtc.so.12 => /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so.12
    	libnvrtc-builtins.so.12.9 => $NV/cuda_nvrtc/lib/libnvrtc-builtins.so.12.9

    $ MIX_ENV=test mix run --no-deps-check --no-start probe.exs
    exla start: {:ok, [:complex, :telemetry, :nx, :nimble_pool, :exla]}
    StreamExecutor [0]: NVIDIA GeForce RTX 3060 Ti, Compute Capability 8.6 (Driver: 13.0.0; Runtime: 12.6.0; Toolkit: 12.9.0; DNN: 9.26.0)
    cuda result: {908.1872256586632, EXLA.Backend}
    platforms: %{host: 88, cuda: 1}

    $ MIX_ENV=test mix run --no-deps-check -e 'IO.puts(Exmc.JIT.describe())'
    compiler=EXLA (configured: nil) backend=EXLA.Backend precision=:f64 ...

`probe.exs` is `Nx.Defn.jit(&Nx.sum(Nx.multiply(Nx.exp(&1), Nx.sin(&1))),
compiler: EXLA, client: :cuda)` on `Nx.iota({1000}, type: :f64) / 1000`. Before
the fix, the same probe with
`LD_LIBRARY_PATH=$NV/nvshmem/lib:$NV/cuda_nvrtc/lib` gave the same
`908.1872256586632`, with `libnvrtc.so.12` from either the wheel or
`/usr/local/cuda`.

No new shell was needed; no environment variable is involved.

## Consequence

`Exmc.JIT` auto-detect now selects EXLA on super-io. A bare `mix check` is
the EXLA arm; the Vulkan arm must be named `EXMC_COMPILER=vulkan`. Vulkan-arm
counts recorded before 20:00 on 2026-09-12 (723/0 at nx_vulkan `16d13f3` and
`8116a19`) ran with EXLA unloadable.

## Undo

    sudo rm /usr/lib/x86_64-linux-gnu/nvshmem_bootstrap_uid.so.3 \
            /usr/lib/x86_64-linux-gnu/nvshmem_transport_ibrc.so.3 \
            /etc/ld.so.conf.d/zz-nvidia-pip-wheels.conf
    sudo ldconfig

## Breaks if

- the pip wheels move or are uninstalled (the symlinks dangle, and the conf
  entry points at nothing);
- `nvidia-nvshmem-cu12` changes the sonames XLA needs;
- exla or xla is upgraded to a build with a different `NEEDED` list: re-run
  the `ldd` check.
