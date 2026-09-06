import Config

# The tripwire `test/config_test.exs` asserts on. It has to be something this
# file sets unconditionally on every platform, which is why it is a marker and
# not a real setting: the original tripwire was `:exla, :default_client`, and
# that key is deliberately absent on FreeBSD (below), so a proxy would have
# reported this file as dead on the one platform whose test run nobody watches.
config :exmc, test_config_loaded: true

# Force EXLA to use CPU (host) client for tests.
# Without this, EXLA tries to init a CUDA client which may:
# 1. Fail with CUDA_ERROR_OUT_OF_MEMORY on machines with small/busy GPUs
# 2. Crash the EXLA.Client GenServer
# 3. Cascade to every subsequent test that touches JIT
#
# Run EXLA on the GPU instead: CUDA_VISIBLE_DEVICES=0 mix test
#
# Guarded on the platform because exla is not a dependency on FreeBSD — the
# `xla` archive ships darwin and linux-gnu targets only, see the `@freebsd?`
# conditional in mix.exs and b536a40. Configuring an application that is not
# available is not an error, but Mix prints a nine-line "you have configured
# application :exla ... but the application is not available" block on every
# single test run, and a warning nobody can act on is one people learn to read
# past. This became visible only when b2f462c made this file load at all.
unless match?({:unix, :freebsd}, :os.type()) do
  config :exla, default_client: :host
end

# Backend selection for the test run:
#
#   mix test                       → auto-detect (EXLA on this host)
#   EXMC_COMPILER=exla   mix test  → force EXLA
#   EXMC_COMPILER=vulkan mix test  → force nx_vulkan (GPU compute via VulkanoBackend)
#   EXMC_COMPILER=none   mix test  → pure Evaluator / BinaryBackend
#
# (EMLX / Apple Metal is postponed until real hardware is available —
#  see the `Exmc.JIT` moduledoc.)
# The EXMC_COMPILER switch MOVED to config/runtime.exs (2026-09-06).
#
# It lived here, and `config/config.exs` imports this file only when
# `config_env() == :test` -- so under `mix run` (which is :dev) it was inert and
# the benchmark suite silently ran EXLA while reporting a Vulkan arm.
# runtime.exs is loaded in every environment, which is the property this needed
# and this file cannot provide.
#
# `config :exla, default_client: :host` above STAYS here: it must keep its
# @freebsd? guard and its test scoping, or every FreeBSD run reintroduces the
# "you have configured application :exla but it is not available" block.
