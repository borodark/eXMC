import Config

# Force EXLA to use CPU (host) client for tests.
# Without this, EXLA tries to init a CUDA client which may:
# 1. Fail with CUDA_ERROR_OUT_OF_MEMORY on machines with small/busy GPUs
# 2. Crash the EXLA.Client GenServer
# 3. Cascade to every subsequent test that touches JIT
#
# Run EXLA on the GPU instead: CUDA_VISIBLE_DEVICES=0 mix test
config :exla, default_client: :host

# Backend selection for the test run:
#
#   mix test                       → auto-detect (EXLA on this host)
#   EXMC_COMPILER=exla   mix test  → force EXLA
#   EXMC_COMPILER=vulkan mix test  → force nx_vulkan (GPU compute via VulkanoBackend)
#   EXMC_COMPILER=none   mix test  → pure Evaluator / BinaryBackend
#
# (EMLX / Apple Metal is postponed until real hardware is available —
#  see the `Exmc.JIT` moduledoc.)
case System.get_env("EXMC_COMPILER") do
  backend when backend in [nil, ""] ->
    :ok

  "vulkan" ->
    config :exmc, compiler: :vulkan

    # Non-synthesisable models (observed-data likelihoods, deep hierarchies)
    # normally trip the Plan B' guard under Vulkan. For a full-suite backend
    # sweep we let them fall through to per-op GPU dispatch (VulkanoBackend)
    # instead of raising, so the whole suite exercises the Vulkan path.
    # Chain-shader-eligible models still take the fused f64 shader.
    config :exmc, allow_vulkan_perop_sampling: true

  name ->
    config :exmc, compiler: String.to_atom(name)
end
