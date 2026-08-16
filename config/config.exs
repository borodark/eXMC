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
