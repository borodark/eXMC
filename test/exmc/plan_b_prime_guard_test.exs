defmodule Exmc.PlanBPrimeGuardTest do
  @moduledoc """
  Tests for the Plan B' guard in `Exmc.Compiler.compile_for_sampling/2`.

  Pre-Plan-B': vulkan compiler + non-synth IR silently fell through to
  per-op CPU via Evaluator + VulkanoBackend host fallbacks. DTrace
  confirmed zero vkQueueSubmit during a 600-iteration NUTS run on
  RegimeModel; the production trial ran 68h in that state producing
  zero posterior updates.

  Post-Plan-B': raises `Exmc.SynthUnsupportedError` at compile time.
  Escape hatch via `config :exmc, :allow_vulkan_perop_sampling, true`
  logs a warning instead.
  """

  use ExUnit.Case, async: false

  alias Exmc.{Builder, Compiler}
  alias Exmc.Dist.{Normal, HalfNormal}

  setup do
    # Save and restore application env around each test so leaked state
    # doesn't bleed across test runs.
    prior_compiler = Application.get_env(:exmc, :compiler)
    prior_allow = Application.get_env(:exmc, :allow_vulkan_perop_sampling)

    on_exit(fn ->
      if prior_compiler do
        Application.put_env(:exmc, :compiler, prior_compiler)
      else
        Application.delete_env(:exmc, :compiler)
      end

      if prior_allow do
        Application.put_env(:exmc, :allow_vulkan_perop_sampling, prior_allow)
      else
        Application.delete_env(:exmc, :allow_vulkan_perop_sampling)
      end
    end)

    :ok
  end

  defp t(v), do: Nx.tensor(v, type: :f64, backend: Nx.BinaryBackend)

  defp single_normal_ir do
    # Single Normal RV — should synth (Phase A).
    Builder.new_ir()
    |> Builder.rv("theta", Normal, %{mu: t(0.0), sigma: t(1.0)})
  end

  defp non_synth_ir do
    # MvNormal, because its parameters need dense linear algebra.
    #
    # This used to be a prior-only Normal + HalfNormal pair, described here as
    # "hierarchical-ish, no synthesised path". That was never true of the
    # model — it was true of a `cond` in detect_meta/1, which only attempted
    # synthesis for IRs carrying a Custom or observed likelihood and dropped
    # everything else to :unsupported. d6f128dee removed that gate (the models
    # synthesise, and match the host log-density to 1.7e-9), and this fixture
    # promptly became synthesisable, so three tests below stopped seeing the
    # guard they exist to check.
    #
    # A fixture for "unsynthesisable" has to be something the GLSL emitter
    # cannot reach on its merits rather than by an accident of routing.
    # MvNormal qualifies structurally: Exmc.Dist.MvNormal.prepare_params/1
    # needs a Cholesky factorisation and a matrix inverse, and the emitter
    # covers elementwise ops, slices, reshapes and reductions — not dense
    # LinAlg. Measured: detect_meta/1 returns :unsupported.
    #
    # Deliberately NOT used here: a hierarchical model with string parameter
    # refs, which is also :unsupported today but is an open work item and
    # would silently turn this test into a no-op the day it lands.
    Builder.new_ir()
    |> Builder.rv("z", Exmc.Dist.MvNormal, %{
      mu: Nx.tensor([0.0, 0.0], type: :f64, backend: Nx.BinaryBackend),
      cov: Nx.tensor([[1.0, 0.2], [0.2, 1.0]], type: :f64, backend: Nx.BinaryBackend)
    })
  end

  describe "guard fires when (vulkan compiler, non-synth IR)" do
    @tag :requires_vulkan
    test "raises SynthUnsupportedError with helpful message" do
      # Only meaningful when Nx.Vulkan is loadable; the guard checks
      # `Exmc.JIT.detect_compiler() == Nx.Vulkan`.
      if Code.ensure_loaded?(Nx.Vulkan) do
        Application.put_env(:exmc, :compiler, :vulkan)
        Application.delete_env(:exmc, :allow_vulkan_perop_sampling)

        ir = non_synth_ir()

        assert_raise Exmc.SynthUnsupportedError, ~r/Vulkan compiler requires/, fn ->
          Compiler.compile_for_sampling(ir)
        end
      end
    end

    @tag :requires_vulkan
    test "exception carries the offending IR" do
      if Code.ensure_loaded?(Nx.Vulkan) do
        Application.put_env(:exmc, :compiler, :vulkan)
        Application.delete_env(:exmc, :allow_vulkan_perop_sampling)

        ir = non_synth_ir()

        try do
          Compiler.compile_for_sampling(ir)
          flunk("guard did not raise")
        rescue
          e in Exmc.SynthUnsupportedError ->
            assert e.ir == ir
            assert e.message =~ "compiler: :exla"
            assert e.message =~ "compiler: :none"
        end
      end
    end
  end

  describe "guard bypassed when escape hatch is enabled" do
    @tag :requires_vulkan
    test "allow_vulkan_perop_sampling: true logs warning instead of raising" do
      if Code.ensure_loaded?(Nx.Vulkan) do
        Application.put_env(:exmc, :compiler, :vulkan)
        Application.put_env(:exmc, :allow_vulkan_perop_sampling, true)

        ir = non_synth_ir()

        log =
          ExUnit.CaptureLog.capture_log(fn ->
            assert {_vag_fn, _step_fn, _pm, _ncp_info, _multi_step_fn, nil} =
                     Compiler.compile_for_sampling(ir)
          end)

        assert log =~ "Plan B' guard bypassed"
      end
    end
  end

  describe "guard does NOT fire when chain_meta is non-nil" do
    @tag :requires_vulkan
    test "single Normal IR passes through cleanly" do
      if Code.ensure_loaded?(Nx.Vulkan) do
        Application.put_env(:exmc, :compiler, :vulkan)

        ir = single_normal_ir()

        # Should NOT raise, and chain_meta should be non-nil.
        #
        # This asserted {:normal, _, _} — the f32-era family fast path. Under
        # D88's f64 Vulkano default, detect_meta/1 deliberately routes
        # single-family models to the synth path instead: the family SPVs are
        # f32-only and trigger the D87 silent-collapse pathology at f64. So
        # {:synthesised, ...} is the correct answer here, and the old
        # expectation was stale rather than the code being wrong.
        #
        # It went unnoticed because this file lived only in the applications
        # tree until d4146a6af, and the assertion needs a Vulkan host to reach.
        # The point of the test is that the guard does not fire, so it asserts
        # that: a meta was produced, of either shape.
        result = Compiler.compile_for_sampling(ir)
        chain_meta = elem(result, 5)

        assert match?({:normal, _, _}, chain_meta) or
                 match?({:synthesised, _, _, _, _, _}, chain_meta),
               "expected a chain meta, got: #{inspect(chain_meta)}"
      end
    end
  end

  describe "guard does NOT fire when compiler is not vulkan" do
    test "exla compiler skips guard regardless of synth status" do
      Application.put_env(:exmc, :compiler, :exla)
      ir = non_synth_ir()

      # Should not raise even though IR is unsupported by synth.
      # (May raise for other reasons — EXLA not loaded, etc — but not
      # SynthUnsupportedError.)
      try do
        Compiler.compile_for_sampling(ir)
      rescue
        e ->
          refute e.__struct__ == Exmc.SynthUnsupportedError
      end
    end

    test "none compiler skips guard" do
      Application.put_env(:exmc, :compiler, :none)
      ir = non_synth_ir()

      # Pure CPU path. Should complete without raising.
      result = Compiler.compile_for_sampling(ir)
      assert is_tuple(result)
      assert tuple_size(result) == 6
    end
  end
end
