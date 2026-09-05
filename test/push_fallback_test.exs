defmodule Exmc.PushFallbackTest do
  @moduledoc """
  A ten-parameter Stan model reaches the fused chain shader.

  ## What this file used to assert, and why it changed

  It was called "push_too_large model falls back to per-op instead of
  crashing", and it asserted `detect_meta/1 == {:unsupported,
  :push_too_large}` for this model — ten free RVs with Normal priors, twenty
  f64 prior floats, comfortably past the old 128-byte push-constants budget.

  That budget was measuring bytes nothing consumed. `Push.pack/1` appended
  one f64 per prior parameter; `leapfrog_chain_synth_f64` forwards only the
  24-byte header; and `MultiRvCustomSpec` bakes prior parameters into the
  generated GLSL as literals, so the shader read them from the SPIR-V, not
  from push constants. The tail existed only to be counted against the NIF's
  `push.len() > 128` check.

  So this model was never too wide. It was being routed to per-op sampling by
  a check on a quantity that did not matter, and this test asserted that the
  routing was correct.

  The end-to-end value of the file survives: a ten-parameter observed model
  built through the Stan front end must still sample correctly. That part is
  unchanged. Only the claim about where it runs has flipped.

  See `test/exmc/nuts/custom_synth/push_width_test.exs` for the posterior
  check against a closed-form conjugate, and `Exmc.NUTS.CustomSynth.Push`
  for the mechanism.
  """
  use ExUnit.Case, async: false

  # Vulkan-specific: the routing this pins only exists under the Vulkan
  # chain-shader path. Under EXLA it would add no coverage, and its NUTS
  # sampling perturbs the EXLA-host BLAS stream enough to break a later
  # LinAlg.solve test in the same run.
  @moduletag :requires_vulkan

  alias Exmc.NUTS.Vulkan.Dispatch

  defp ten_param_ir do
    code = """
    data { real y; }
    parameters {
      real a1; real a2; real a3; real a4; real a5;
      real a6; real a7; real a8; real a9; real a10;
    }
    model {
      a1 ~ normal(0, 1); a2 ~ normal(0, 1); a3 ~ normal(0, 1);
      a4 ~ normal(0, 1); a5 ~ normal(0, 1); a6 ~ normal(0, 1);
      a7 ~ normal(0, 1); a8 ~ normal(0, 1); a9 ~ normal(0, 1);
      a10 ~ normal(0, 1);
      y ~ normal(a1, 1);
    }
    """

    Exmc.Stan.compile!(code, %{"y" => Nx.tensor(3.0)})
  end

  test "a ten-parameter model synthesises instead of degrading to per-op" do
    assert {:ok, {:synthesised, _sha, layout, _spec, _spv, _obs, _capt}} =
             Exmc.NUTS.ChainShaderCodegen.detect_meta(ten_param_ir()),
           "twenty prior floats used to return {:unsupported, :push_too_large} " <>
             "and sample per-op; the push block no longer carries them"

    assert length(layout) == 10
  end

  test "it dispatches the chain shader, and still samples correctly" do
    Dispatch.reset_dispatch_count()

    {trace, _stats} =
      Exmc.Sampler.sample(ten_param_ir(), %{}, num_warmup: 40, num_samples: 40, seed: 42)

    dispatches = Dispatch.dispatch_count()

    assert dispatches > 0,
           "the ten-parameter model issued #{dispatches} chain dispatches — it is " <>
             "still falling back to per-op sampling"

    # The original end-to-end assertion, unchanged: whatever path it takes,
    # the model must produce a usable trace.
    vals = trace["a1"] |> Nx.to_flat_list()
    assert length(vals) == 40
    assert Enum.all?(vals, &is_number/1)

    # a1 is the only parameter the likelihood touches (y ~ normal(a1, 1)),
    # with a standard Normal prior and one observation at 3.0. Conjugate
    # posterior mean is 1.5. A loose band: this is a routing test, and 40
    # draws cannot resolve much, but a wholesale wrong answer would show.
    mean = Enum.sum(vals) / length(vals)
    assert abs(mean - 1.5) < 1.5, "a1 posterior mean #{Float.round(mean, 3)} vs 1.5 expected"
  end
end
