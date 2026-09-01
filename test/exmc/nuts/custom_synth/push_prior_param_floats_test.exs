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

  describe "ensure_fits!/2 — the batched path's 128-byte cap (D2)" do
    # chain_batch/5 builds its own header so it cannot go through pack/1, and
    # had no size check at all: an oversized block reached the NIF, came back
    # {:error, :bad_input}, and failed the {:ok, {...}} = match as a MatchError
    # naming nothing.
    #
    # Tested here rather than through chain_batch/5 because that function calls
    # ensure_batch_nif!/0 first, and the f64 batch NIF does not exist -- the
    # pack is unreachable through it. Same reason as prior_param_floats above.

    test "a block at the budget passes through unchanged" do
      bin = :binary.copy(<<0>>, Push.max_bytes())
      assert Push.ensure_fits!(bin, "test") == bin
    end

    test "one byte over the budget raises, naming the overflow" do
      bin = :binary.copy(<<0>>, Push.max_bytes() + 1)

      assert_raise RuntimeError, ~r/129 B, budget is 128 B \(over by 1 B\)/, fn ->
        Push.ensure_fits!(bin, "test")
      end
    end

    test "the message names the caller it was given" do
      bin = :binary.copy(<<0>>, Push.max_bytes() + 8)

      assert_raise RuntimeError, ~r/chain_batch\/5/, fn ->
        Push.ensure_fits!(bin, "chain_batch/5")
      end
    end

    test "the budget binds the batched path only — pack/1 no longer has one" do
      # This test used to assert "two encoders, one number": that a 16-prior
      # model overflowed BOTH `ensure_fits!/2` and `pack/1`. That is no longer
      # true and the change is deliberate.
      #
      # `pack/1` emits the 24-byte header alone. The prior floats it used to
      # append were baked into the synthesised GLSL as literals and read by
      # nothing, while still counting against the NIF's `push.len() > 128`
      # check — so the cap rejected models the shader could have run. Removing
      # the tail took an 8-RV model from 0 chain dispatches to 2564. See
      # `push_width_test.exs`.
      #
      # `ensure_fits!/2` still guards `chain_batch/5`, which builds its own
      # header-plus-tail for the batched f32 path and does not go through
      # `pack/1`. That path is unreachable today (D4) but its encoder is real,
      # so the budget it checks is real.
      priors = for i <- 1..16, do: {"p#{i}", Exmc.Dist.HalfNormal, %{sigma: 1.0}}

      # pack/1: no budget, no width at which it fails.
      assert {:ok, _bin, 24} = Push.pack(%{K: 8, n_obs: 4, d: 16, eps: 0.1, priors: priors})

      # ensure_fits!/2: still enforces 128 B, on a block the batched path builds.
      floats = Enum.flat_map(priors, &Push.prior_param_floats/1)
      oversized = <<0::128>> <> for f <- floats, into: <<>>, do: <<f::little-float-64>>

      assert byte_size(oversized) > Push.max_bytes()

      assert_raise RuntimeError, ~r/budget is #{Push.max_bytes()} B/, fn ->
        Push.ensure_fits!(oversized, "chain_batch/5")
      end
    end
  end
end
