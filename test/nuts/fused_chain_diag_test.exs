defmodule Exmc.NUTS.FusedChainDiagTest do
  @moduledoc """
  Diagnostic comparison: EXLA (f64 reference) vs unfused Vulkan
  (f32, no chain) vs fused Vulkan chain (f32, K-step).

  All three sample the same trivial Normal-Normal model
  (`x ~ N(0, 1)`) with the same seed. Posterior variance must
  recover to ~1.0 within MCMC noise. Divergence between the three
  isolates *where* a discrepancy lives:

  * EXLA ≈ 1.0, unfused Vulkan ≈ 0.5, fused chain ≈ 0.5
    → f32 precision is the issue (chain is innocent)

  * EXLA ≈ 1.0, unfused Vulkan ≈ 1.0, fused chain ≈ 0.5
    → chain integration has a real bug (off-by-one, sign,
      contract mismatch, etc.)

  * EXLA ≈ 1.0, unfused Vulkan ≈ 0.5, fused chain ≈ 1.0
    → fused chain is *correcting* something the unfused path
      gets wrong; investigate before declaring a win

  ## Tags

  All three tests are tagged `:diag` (excluded from default runs;
  enable with `mix test --include diag`). The unfused-Vulkan test
  is also `:slow` because it takes several minutes — the per-step
  dispatch overhead this whole project is designed to eliminate.

  ## Running

      # All three diagnostics under EXLA + Vulkan:
      EXMC_COMPILER=vulkan mix test --include diag \\
        test/nuts/fused_chain_diag_test.exs

      # Just the fast ones (skip unfused):
      EXMC_COMPILER=vulkan mix test --include diag --exclude slow \\
        test/nuts/fused_chain_diag_test.exs

  ## Acceptance criterion

  Once Stage 1.5.4 of `nx_vulkan/PLAN_FUSED_LEAPFROG.md` is
  resolved, the fused-chain assertion should pass under the
  bound `var in [0.7, 1.3]`. Until then it's expected to fail
  (var ≈ 0.5 per the May 2026 measurement) and the test is
  excluded from normal CI via `:diag`.
  """

  use ExUnit.Case, async: false

  import Exmc.TestHelper, only: [put_env_scoped: 2, delete_env_scoped: 1]

  alias Exmc.{Builder, Dist.Normal, NUTS.Sampler}

  # Generous bounds: MCMC noise + small sample (1000 draws after
  # 200 warmup) easily moves variance ±0.3.
  @var_low 0.7
  @var_high 1.3
  @num_warmup 200
  @num_samples 1000
  @seed 42

  defp run_one do
    ir =
      Builder.new_ir()
      |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

    {trace, _stats} =
      Sampler.sample(ir, %{},
        num_warmup: @num_warmup,
        num_samples: @num_samples,
        seed: @seed
      )

    xs = trace["x"] |> Nx.to_flat_list()
    mean = Enum.sum(xs) / length(xs)
    var = Enum.sum(Enum.map(xs, fn x -> (x - mean) * (x - mean) end)) / length(xs)
    {mean, var}
  end

  defp report(label, mean, var) do
    IO.puts(
      "  #{String.pad_trailing(label, 24)}: mean=#{Float.round(mean, 4)}, var=#{Float.round(var, 4)}"
    )
  end

  @tag :diag
  test "reference: EXLA produces var ≈ 1.0 on Normal-Normal" do
    {mean, var} = run_one()
    report("EXLA reference", mean, var)

    assert var >= @var_low and var <= @var_high,
           "EXLA var=#{var} outside [#{@var_low}, #{@var_high}] — sampler smoke is broken"
  end

  @tag :diag
  @tag :slow
  @tag :requires_vulkan
  test "unfused Vulkan: per-step dispatch path produces var ≈ 1.0" do
    delete_env_scoped(:fused_leapfrog_normal_meta)
    {mean, var} = run_one()
    report("Unfused Vulkan (f32)", mean, var)

    assert var >= @var_low and var <= @var_high,
           "Unfused Vulkan var=#{var} outside [#{@var_low}, #{@var_high}] — " <>
             "if this fails, f32 precision is the cause and the chain is innocent"
  end

  @tag :diag
  @tag :requires_vulkan
  test "fused chain: leapfrog_chain_normal produces var ≈ 1.0" do
    put_env_scoped(:fused_leapfrog_normal_meta, {0.0, 1.0})

    {mean, var} = run_one()
    report("Fused Vulkan chain", mean, var)

    assert var >= @var_low and var <= @var_high,
           "Fused chain var=#{var} outside [#{@var_low}, #{@var_high}] — " <>
             "the Stage 1.5.4 variance-bias issue. Compare against the unfused-Vulkan " <>
             "test to isolate (chain bug vs f32 precision)."
  end
end
