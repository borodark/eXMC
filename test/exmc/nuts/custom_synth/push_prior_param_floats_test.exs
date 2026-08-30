defmodule Exmc.NUTS.CustomSynth.PushPriorParamFloatsTest do
  use ExUnit.Case, async: true

  alias Exmc.NUTS.CustomSynth.Push

  # `Push.prior_param_floats/1` became public on 2026-08-29 so the batched
  # dispatch path could stop carrying its own copy (D1 in
  # docs/BATCHED_CHAIN_DISPATCH.md).
  #
  # These tests pin the behaviour the copy did NOT have. They are the only
  # verification D1 can currently get: `Dispatch.chain_batch/5` calls
  # `ensure_batch_nif!/0` before it packs anything, so with no f64 batch NIF
  # in nx_vulkan the encoder is unreachable through that function. Testing it
  # here rather than pretending the batched path covers it.

  describe "clauses the deleted dispatch.ex copy also had" do
    test "Normal, HalfCauchy, HalfNormal, Exponential encode unchanged" do
      assert Push.prior_param_floats({"x", Exmc.Dist.Normal, %{mu: 1.5, sigma: 2.0}}) ==
               [1.5, 2.0]

      assert Push.prior_param_floats({"x", Exmc.Dist.HalfCauchy, %{scale: 3.0}}) == [3.0]
      assert Push.prior_param_floats({"x", Exmc.Dist.HalfNormal, %{sigma: 0.5}}) == [0.5]
      assert Push.prior_param_floats({"x", Exmc.Dist.Exponential, %{lambda: 4.0}}) == [4.0]
    end
  end

  describe "clauses the copy lacked — these raised before D1" do
    test "the seven distributions the batched path could not encode" do
      assert Push.prior_param_floats({"x", Exmc.Dist.StudentT, %{df: 3.0, loc: 0.0, scale: 1.0}}) ==
               [3.0, 0.0, 1.0]

      assert Push.prior_param_floats({"x", Exmc.Dist.Cauchy, %{loc: 0.0, scale: 1.0}}) == [
               0.0,
               1.0
             ]

      assert Push.prior_param_floats({"x", Exmc.Dist.Weibull, %{k: 2.0, lambda: 1.0}}) == [
               2.0,
               1.0
             ]

      assert Push.prior_param_floats({"x", Exmc.Dist.Lognormal, %{mu: 0.0, sigma: 1.0}}) ==
               [0.0, 1.0]

      assert Push.prior_param_floats(
               {"x", Exmc.Dist.TruncatedNormal, %{mu: 0.0, sigma: 1.0, lower: -1.0, upper: 1.0}}
             ) == [0.0, 1.0, -1.0, 1.0]

      assert Push.prior_param_floats({"x", Exmc.Dist.Gamma, %{alpha: 2.0, beta: 3.0}}) == [
               2.0,
               3.0
             ]

      assert Push.prior_param_floats({"x", Exmc.Dist.Beta, %{alpha: 2.0, beta: 3.0}}) == [
               2.0,
               3.0
             ]
    end

    test "an unknown distribution raises rather than packing a short block" do
      assert_raise RuntimeError, ~r/no encoder for prior/, fn ->
        Push.prior_param_floats({"x", Exmc.Dist.NotAThing, %{}})
      end
    end
  end

  describe "scalar/2 shapes the copy mishandled" do
    # The copy's scalar/2 was `case ... do v when is_number(v) -> ...;
    # %Nx.Tensor{} = t -> Nx.to_number(t) end`. Both cases below fell through
    # it: a hierarchical param hit no clause (CaseClauseError), and a {d}
    # tensor reached Nx.to_number/1, which rejects a non-scalar.

    test "a hierarchical param naming another RV takes a slot as 0.0" do
      # The value is the NAME of another RV, resolved in the shader from q.
      assert Push.prior_param_floats({"x", Exmc.Dist.Normal, %{mu: :mu_pop, sigma: 1.0}}) ==
               [0.0, 1.0]

      assert Push.prior_param_floats({"x", Exmc.Dist.Normal, %{mu: "mu_pop", sigma: 1.0}}) ==
               [0.0, 1.0]
    end

    test "a scalar tensor param is unwrapped" do
      t = Nx.tensor(2.5, backend: Nx.BinaryBackend)
      assert Push.prior_param_floats({"x", Exmc.Dist.HalfNormal, %{sigma: t}}) == [2.5]
    end

    test "a vectorized {d} param takes element 0" do
      t = Nx.tensor([1.5, 1.5, 1.5], backend: Nx.BinaryBackend)
      assert Push.prior_param_floats({"x", Exmc.Dist.HalfNormal, %{sigma: t}}) == [1.5]
    end
  end
end
