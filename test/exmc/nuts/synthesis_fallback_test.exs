defmodule Exmc.NUTS.SynthesisFallbackTest do
  @moduledoc """
  Under Vulkan, a single-RV model is synthesised or refused -- never handed
  a family meta, which no chain dispatch accepts.

  The defect this pins: `ChainShaderCodegen.detect_meta/2` fell back to
  `{:normal, mu, sigma}` and friends whenever synthesis did not succeed, and
  `try_synthesise/2` turned any raise into a bare `:unsupported`. On
  2026-09-13 CustomSynth raised on every FreeBSD host (`:crypto` was not on
  the code path; fixed in mix.exs), the fallback produced `{:normal, 0.0,
  1.0}`, and `bench/nuts_truth.exs` died after a second with a
  FunctionClauseError in `Exmc.NUTS.Vulkan.Dispatch.do_chain/8` that named
  neither crypto nor synthesis.

  No ordinary model makes synthesis raise, so these tests pass a raising
  synthesiser through the `:synthesiser` option.
  """

  use ExUnit.Case, async: false

  import ExUnit.CaptureLog
  import Exmc.TestHelper

  alias Exmc.{Builder, Compiler}
  alias Exmc.NUTS.ChainShaderCodegen

  defmodule RaisingSynth do
    # apply/3, so the undefined module is a runtime UndefinedFunctionError --
    # the shape of the real :crypto failure -- and not a compile warning.
    def synthesise(_ir, _opts), do: apply(:crypto_stand_in, :hash, [:sha256, "glsl"])
  end

  defmodule RecordingSynth do
    def synthesise(_ir, opts) do
      send(self(), {:synthesise_called, opts})
      :unsupported
    end
  end

  defp normal_ir do
    Builder.new_ir()
    |> Builder.rv("x", Exmc.Dist.Normal, %{
      mu: Nx.tensor(0.0, type: :f64),
      sigma: Nx.tensor(1.0, type: :f64)
    })
  end

  describe "under Vulkan" do
    @describetag :requires_vulkan

    setup do
      put_env_scoped(:compiler, :vulkan)
      put_env_scoped(:allow_vulkan_perop_sampling, false)
      :ok
    end

    test "a raising synthesis is a tagged refusal, not a family meta, and it is logged" do
      log =
        capture_log(fn ->
          assert {:unsupported, :synthesis_raised} =
                   ChainShaderCodegen.detect_meta(normal_ir(), synthesiser: RaisingSynth)
        end)

      assert log =~ "chain-shader synthesis raised"
      # The exception itself, so the cause is readable from the log.
      assert log =~ ":crypto_stand_in"
    end

    test "a clean refusal from synthesis is returned as is, never replaced by a family meta" do
      assert :unsupported =
               ChainShaderCodegen.detect_meta(normal_ir(), synthesiser: RecordingSynth)

      # The option is a seam, not something synthesise/2 should ever see.
      assert_received {:synthesise_called, opts}
      refute Keyword.has_key?(opts, :synthesiser)
    end

    test "the compile-time guard refuses loudly and says synthesis raised" do
      error =
        assert_raise Exmc.SynthUnsupportedError, fn ->
          capture_log(fn ->
            Compiler.compile_for_sampling(normal_ir(), synthesiser: RaisingSynth)
          end)
        end

      assert Exception.message(error) =~ "CHAIN-SHADER SYNTHESIS RAISED"
      assert Exception.message(error) =~ "{:unsupported, :synthesis_raised}"
    end

    test "an ordinary Normal still synthesises" do
      assert {:ok, {:synthesised, _, _, _, _, _, _}} =
               ChainShaderCodegen.detect_meta(normal_ir(), [])
    end
  end

  describe "off Vulkan" do
    setup do
      put_env_scoped(:compiler, :none)
      :ok
    end

    test "the family meta is unchanged and synthesis is not consulted" do
      # Sampler seeds the initial mass matrix from it (prior_inv_mass_per_rv/2),
      # and nothing dispatches a chain shader off Vulkan.
      assert {:ok, {:normal, +0.0, 1.0}} =
               ChainShaderCodegen.detect_meta(normal_ir(), synthesiser: RaisingSynth)
    end
  end
end
