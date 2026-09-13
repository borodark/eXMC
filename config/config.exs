import Config

# Mix auto-loads *only* this file. A `config/<env>.exs` sitting next to it is
# inert unless imported from here — no warning, no error, it simply never runs.
#
# `config/test.exs` was inert for exactly that reason: the `EXMC_COMPILER`
# switch, `config :exla, default_client: :host`, and `allow_vulkan_perop_sampling`
# all did nothing. That made the backend sweep vacuous — every
# `EXMC_COMPILER=vulkan mix test` ran against whatever `Exmc.JIT.auto_detect/0`
# happened to pick, and reported a pass for it. A vacuous check reads exactly
# like a passing one.
#
# `test/config_test.exs` asserts this import is live, so the same silence
# cannot come back.
#
# Imported conditionally rather than as `import_config "#{config_env()}.exs"`
# because :test is the only environment with a file; that form would break
# `mix compile` in :dev and :prod.
if config_env() == :test do
  import_config "test.exs"
end

# NXV_SKIP_NIF_BUILD=1: link the nx_vulkan `.so` already in
# deps/nx_vulkan/priv/native instead of letting Rustler build the crate.
# scripts/fleet_verify.sh sets it only when that `.so` is a cross-built artifact
# whose provenance names this checkout's lock sha and whose hash still matches.
#
# nx_vulkan's own config/config.exs reads the same variable, but a dependency's
# config files are never loaded by the project that depends on it. MEASURED
# 2026-09-13: with the variable set, `Application.get_env(:nx_vulkan,
# Nx.Vulkan.NativeV)` in this project is `nil`. Without this block the variable
# does nothing here: Rustler rebuilds natively over the shipped artifact, and a
# run reports "prebuilt" about a NIF it compiled itself.
#
# Rustler reads this with compile_env, so the value must be the same for
# `mix compile` and for every `mix test` after it, or the VM refuses to boot.
if System.get_env("NXV_SKIP_NIF_BUILD") == "1" do
  config :nx_vulkan, Nx.Vulkan.NativeV, skip_compilation?: true
end
