import Config

# Mix loads THIS file at runtime, in every environment, for `mix run`,
# `mix test` and releases alike. `config/config.exs` and `config/<env>.exs` are
# loaded at BUILD time and, worse here, `config/config.exs` imports `test.exs`
# only when `config_env() == :test`.
#
# That asymmetry is why this file exists. `EXMC_COMPILER` used to be read only
# from `config/test.exs`, so under `mix run` -- which is `:dev` -- it was
# INERT. The posteriordb benchmark suite ran under `mix run`. For months it
# reported "33/33 PASS" for an arm it had never selected: with the variable
# ignored, `Exmc.JIT.auto_detect/0` picked EXLA, and nothing in the output said
# so. The number was real and the attribution was not.
#
# Two harnesses had already hand-rolled a workaround --
# `bench/chain_dispatch_cost.exs` and `benchmark/posteriordb/harness.exs` both
# call `Application.put_env(:exmc, :compiler, ...)` themselves -- which is the
# usual sign that the mechanism belongs one level down.
#
# See docs/NX_BACKEND_HANDLING.md P1/P2.
case System.get_env("EXMC_COMPILER") do
  v when v in [nil, ""] ->
    # Unset is the normal state, including on the FreeBSD fleet. Configure
    # nothing and let `Exmc.JIT.auto_detect/0` behave exactly as before.
    :ok

  "vulkan" ->
    config :exmc, compiler: :vulkan

  "exla" ->
    config :exmc, compiler: :exla

  "none" ->
    config :exmc, compiler: :none

  "auto" ->
    config :exmc, compiler: :auto

  other ->
    # Previously `String.to_atom(name)`, which turned a typo into
    # `compiler: :vulcan` and then into a bare CaseClauseError from
    # `detect_compiler/0`, naming nothing. Fail here, where the typo is.
    raise """
    EXMC_COMPILER=#{other} is not a compiler this project knows.

    Expected one of: vulkan | exla | none | auto (or unset for auto-detect).
    """
end

# On `:allow_vulkan_perop_sampling`, which this file no longer sets.
#
# This flag lets a model that CustomSynth refuses fall through to per-op
# dispatch instead of raising `SynthUnsupportedError`, so a full-suite backend
# sweep exercises the Vulkan path end to end. Outside a sweep the raise is the
# CORRECT outcome -- it is the Plan B' guard doing its job -- and enabling this
# globally would quietly convert a loud refusal into a slow, silent, per-op run
# on the CPU.
#
# It is scoped to an EXPLICIT `EXMC_COMPILER=vulkan`, and that asymmetry has
# teeth: on a Vulkan-only host that AUTO-DETECTS, the flag is false and a
# refused model raises. Measured on mac-248 (FreeBSD, GT 750M), same tree, same
# file: `mix test test/custom_dist_test.exs` gives 16 tests / 1 failure under
# auto-detect and 16 / 0 under `EXMC_COMPILER=vulkan`. That failure reads like a
# code regression and is not one -- the model has been `:unsupported` since
# 5b99e02af. The fleet convention for Vulkan-only hosts is therefore the
# explicit form. `Exmc.JIT.describe/0` now prints `perop_fallback=` so the two
# paths are distinguishable from the banner.
# REMOVED 2026-09-12: the block that set `allow_vulkan_perop_sampling: true`
# under `config_env() == :test and EXMC_COMPILER == "vulkan"`. It was
# deliberate (see above) and it made the two Vulkan invocations different
# arms: plain `mix test` on a Vulkan host reported one more failure than
# `EXMC_COMPILER=vulkan mix test` for the same tree, and `test_helper.exs`,
# which keys its excludes off the DETECTED backend so the arm cannot be
# misread, could not see the difference. Exactly one test needed the
# allowance — `CustomDistTest` "custom dist works with NUTS sampler", a
# Custom-only model with no free RVs — and it now sets the flag for itself
# with `put_env_scoped/2`. `PlanBPrimeGuardTest` sets and restores it
# explicitly as it always did. Nothing else in the suite reads it.
