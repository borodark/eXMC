defmodule Exmc.ConfigTest do
  use ExUnit.Case, async: true

  @moduledoc """
  `config/test.exs` must actually be loaded.

  Mix auto-loads only `config/config.exs`; a per-environment file is inert
  unless imported from there. `config/test.exs` was inert for the whole life of
  this repo, which is why `EXMC_COMPILER=vulkan mix test` silently ran against
  whatever `Exmc.JIT.auto_detect/0` picked and reported a pass for it.

  A dead config file fails silently in both directions — nothing warns you it
  was skipped, and every test still passes. This test is the tripwire.
  """

  test "config/test.exs is loaded" do
    assert Application.get_env(:exmc, :test_config_loaded) == true, """
    config/test.exs is not being loaded. It sets `config :exmc, test_config_loaded: true`
    as its first line and that has not arrived, so every other setting in that
    file — the EXLA host client — is dead too. (The EXMC_COMPILER switch lives
    in config/runtime.exs, which Mix loads on its own.)

    Restore the `import_config "test.exs"` in config/config.exs.
    """
  end

  test "the EXLA host client is configured wherever exla is a dependency" do
    # This was the tripwire above until it turned out to be platform-specific:
    # exla is not a dependency on FreeBSD, so the key is legitimately absent
    # there and a proxy assertion would have called a live config file dead.
    # It is still worth asserting where it applies — it is what keeps EXLA off
    # a busy CUDA device during the suite.
    if match?({:unix, :freebsd}, :os.type()) do
      refute Code.ensure_loaded?(EXLA),
             "exla is loadable on FreeBSD, so config/test.exs should be configuring it"

      assert Application.get_env(:exla, :default_client) == nil,
             "config/test.exs is configuring :exla on FreeBSD, which makes Mix warn " <>
               "on every run about an application that is not a dependency there"
    else
      assert Application.get_env(:exla, :default_client) == :host
    end
  end

  describe "EXMC_COMPILER" do
    test "reaches application config when set" do
      # The switch is read at config time, so this asserts the wiring for the
      # value this run was actually started with rather than setting it here.
      case System.get_env("EXMC_COMPILER") do
        env when env in [nil, ""] ->
          assert Application.get_env(:exmc, :compiler) == nil,
                 "no EXMC_COMPILER was set, so :exmc :compiler should be unset (auto-detect)"

        name ->
          assert Application.get_env(:exmc, :compiler) == String.to_existing_atom(name),
                 "EXMC_COMPILER=#{name} did not reach `config :exmc, :compiler`"
      end
    end

    test "the compiler in config is the compiler actually selected" do
      # Guards the gap that made the sweep vacuous: config saying :vulkan while
      # detect_compiler/0 quietly returns something else.
      case Application.get_env(:exmc, :compiler) do
        nil ->
          :ok

        :none ->
          assert Exmc.JIT.detect_compiler() == nil

        :vulkan ->
          assert Exmc.JIT.detect_compiler() == Nx.Vulkan,
                 "configured :vulkan but detect_compiler/0 returned " <>
                   inspect(Exmc.JIT.detect_compiler())

        :exla ->
          assert Exmc.JIT.detect_compiler() == EXLA,
                 "configured :exla but detect_compiler/0 returned " <>
                   inspect(Exmc.JIT.detect_compiler())
      end
    end
  end
end
