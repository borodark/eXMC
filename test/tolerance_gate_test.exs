defmodule Exmc.ToleranceGateTest do
  @moduledoc """
  Tests for the test harness itself — specifically that `assert_posterior!`
  can fail, and fails for the right reason.

  This exists because the failure mode it guards against is invisible by
  construction. A distributional assertion whose tolerance is wider than the
  defect it is written for passes, reports green, and is indistinguishable from
  a check that ran and found nothing wrong. That is how a 37.8% variance
  inflation shipped in 0.3.0 behind a passing suite, and how the assertion
  0.3.1 added as the fix was itself, at its original sample size, a coin flip
  on the very defect it was written for (see NEXT.md item 2).

  So the gate is exercised directly: once against an under-powered chain, once
  against a chain carrying the exact defect that shipped, once against an
  honest chain.
  """
  use ExUnit.Case

  import Exmc.TestHelper, only: [assert_posterior!: 3]

  alias Exmc.Builder
  alias Exmc.Dist.Normal
  alias Exmc.NUTS.Sampler

  @moduletag timeout: 300_000

  # mu ~ N(0, 10), x | mu ~ N(mu, 1), observe x = 5.
  # Conjugate posterior: precision 1.01, mean 5/1.01, var 1/1.01.
  @meta {:normal, 5.0 / 1.01, :math.sqrt(1.0 / 1.01)}

  defp draws(n) do
    ir =
      Builder.new_ir()
      |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(10.0)})
      |> Builder.rv("x", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
      |> Builder.obs("x_obs", "x", Nx.tensor(5.0))

    {trace, _} = Sampler.sample(ir, %{}, num_warmup: 300, num_samples: n, seed: 42)
    trace |> Map.fetch!("mu") |> Nx.to_flat_list()
  end

  test "an under-powered chain is reported INCONCLUSIVE, not passed" do
    # 500 draws is what integration_test.exs used until 2026-08-18. It reaches
    # ESS ~180, where a 4-sigma analytic gate resolves a 38% variance error —
    # the same size as the defect that shipped.
    err =
      assert_raise ExUnit.AssertionError, fn ->
        assert_posterior!(draws(500), @meta, resolution: 0.20)
      end

    assert err.message =~ "INCONCLUSIVE"
    assert err.message =~ "resolves a variance error of 3"
  end

  test "the 0.3.0 defect — variance inflated 37.8% — is caught as a variance failure" do
    d = draws(4000)
    m = Enum.sum(d) / length(d)

    # Widen the spread about the mean by exactly the factor CHANGELOG 0.3.1
    # records: Normal(0,1) variance 1.378 against a true 1.0.
    inflated = Enum.map(d, fn x -> m + (x - m) * :math.sqrt(1.378) end)

    err =
      assert_raise ExUnit.AssertionError, fn ->
        assert_posterior!(inflated, @meta, resolution: 0.20)
      end

    # It must be reported as a WRONG ANSWER, not as an inconclusive one. An
    # earlier version of the helper checked power before moments, and the
    # inflated chain's own fourth moment pushed it over the resolution
    # threshold — so a real defect was reported as "not enough draws".
    assert err.message =~ "analytic_variance failed"
    refute err.message =~ "INCONCLUSIVE"
  end

  test "an honest chain with enough draws passes" do
    assert :ok == assert_posterior!(draws(4000), @meta, resolution: 0.20)
  end
end
