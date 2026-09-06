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
# Guarded on whether exla IS HERE, not on which OS this is.
#
# Configuring an absent application is not an error, but Mix prints a nine-line
# "you have configured application :exla ... but the application is not
# available" block on every single test run, and a warning nobody can act on is
# one people learn to read past. That is what needs suppressing.
#
# This used to read `unless match?({:unix, :freebsd}, :os.type())`, which is a
# PROXY for the real condition. The proxy came from mix.exs, where the OS check
# is correct and load-bearing: the `xla` archive ships darwin and linux-gnu
# targets only, so on FreeBSD declaring exla at all breaks `mix deps.get`
# before a single module compiles (see the @freebsd? conditional and b536a40).
# But mix.exs has ALREADY made that decision by the time this file is read, and
# re-deriving it here means the same fact is expressed twice, in two different
# ways, free to drift.
#
# It is also wrong in cases nobody had hit yet. The proxy says "not FreeBSD,
# therefore exla" — which is false on any Linux host that resolved without it,
# on macOS, and on any future target the archive does not cover. The honest
# predicate has no OS in it: is the module on the code path?
#
# The predicate has to be one that is TRUE AT CONFIG-LOAD TIME, and the two
# obvious candidates are not. Measured on this host, by instrumenting this file:
#
#   Code.ensure_loaded?(EXLA)       = false   <- module not on the path yet
#   Application.spec(:exla, :vsn)   = nil     <- applications not loaded yet
#   Mix.Project.deps_paths()[:exla] = true    <- the dependency list, which IS
#                                                resolved by now
#
# The first two are false here even on a host where exla is present and working
# — config runs before modules are loadable and before any application is
# loaded. Guarding on either silently drops the setting on EVERY platform, which
# is a quieter failure than the warning it was meant to suppress: I tried
# `Application.spec/2` first and `default_client` came back nil.
#
# `Mix.Project.deps_paths/0` reads the resolved dependency list, which is what
# mix.exs's @freebsd? conditional actually decides. Safe here because
# config/test.exs is build-time only and never evaluated in a release, where
# Mix is absent.
if Map.has_key?(Mix.Project.deps_paths(), :exla) do
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
