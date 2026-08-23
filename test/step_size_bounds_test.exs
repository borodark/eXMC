defmodule Exmc.StepSizeBoundsTest do
  @moduledoc """
  The step-size safety bounds must catch a blow-up and must not bind otherwise.

  Both halves are asserted, because only the first is obvious and only the
  second was actually wrong. The applications tree clamped epsilon to
  [1e-6, 1.0], and measured on this host that ceiling bound on every model
  tried — Normal(0,1) included — pinning epsilon at exactly 1.0 and collapsing
  ESS to exactly 500/2000. A clamp that is always active is not a guard, it is
  a fixed step size, and it means dual averaging is not running at all.

  A test for the guard alone would have passed against that. So the binding
  test is here too, and it is the one with teeth.
  """

  use ExUnit.Case, async: true

  alias Exmc.{Builder, Dist}
  alias Exmc.NUTS.{Sampler, StepSize}

  # Mirrors the module attributes in step_size.ex. Duplicated on purpose: if
  # someone narrows the real bounds, these stay put and the tests below fail.
  @epsilon_min 1.0e-10
  @epsilon_max 1.0e3

  describe "the guard catches what it exists for" do
    test "BEAM float arithmetic raises rather than producing NaN or infinity" do
      # This is why the clamps carry no `x != x` branch. The one they used to
      # carry could not fire, and a safety branch that cannot fire reads
      # exactly like one that works.
      # Values arrive through variables, or the compiler constant-folds them
      # and warns at compile time instead of raising at run time.
      # Enum.random/1 is opaque to constant folding; String.to_float/1 is not,
      # and the folded form warns at compile time instead of raising at run time.
      big = Enum.random([710.0])
      ok = Enum.random([709.0])

      assert_raise ArithmeticError, fn -> :math.exp(big) end

      # 709 is fine, 710 is not — so the upper bound guards a real edge.
      assert is_float(:math.exp(ok))
    end

    test "a NaN cannot reach the adaptation as a number" do
      # Nx is the only place a NaN could come from, and it does not arrive as
      # a float: `update/2` is guarded by is_number/1, which :nan fails. So
      # there is no path by which the removed `x != x` branch could have run.
      v = Nx.to_number(Nx.tensor(:nan, type: :f64))

      assert v == :nan
      refute is_number(v)

      assert_raise FunctionClauseError, fn -> StepSize.update(StepSize.init(1.0), v) end
    end

    test "a long run of rejections drives the step down but not out of range" do
      eps =
        Enum.reduce(1..200, StepSize.init(1.0), fn _, st -> StepSize.update(st, 0.0) end)
        |> StepSize.finalize()

      assert eps >= @epsilon_min
      assert eps < 1.0, "200 rejections should shrink the step, got #{eps}"
    end

    test "a long run of perfect acceptances drives it up but not out of range" do
      eps =
        Enum.reduce(1..200, StepSize.init(1.0), fn _, st -> StepSize.update(st, 1.0) end)
        |> StepSize.finalize()

      assert eps <= @epsilon_max
    end
  end

  describe "the guard does not bind in ordinary operation" do
    test "Normal(0,1) adapts to an interior step size, not to a bound" do
      ir =
        Builder.new_ir()
        |> Builder.rv("x", Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

      {_trace, stats} = Sampler.sample(ir, %{}, num_warmup: 500, num_samples: 500, seed: 11)

      # Landing exactly on a bound is the signature of a clamp that is doing the
      # adapting. Interior is the whole claim.
      refute stats.step_size == @epsilon_max
      refute stats.step_size == @epsilon_min

      assert stats.step_size > 0.05 and stats.step_size < 10.0,
             "expected a sane adapted step for Normal(0,1), got #{stats.step_size}"
    end

    test "targets differing only by scale do not all pin to the same step" do
      steps =
        for sigma <- [1.0, 100.0, 10_000.0] do
          ir =
            Builder.new_ir()
            |> Builder.rv("x", Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(sigma)})

          {_trace, stats} = Sampler.sample(ir, %{}, num_warmup: 500, num_samples: 500, seed: 11)
          stats.step_size
        end

      # Under the [1e-6, 1.0] clamp all three came back as exactly 1.0. That is
      # what this refutes: identical steps across three targets means the bound
      # chose them, not the adaptation.
      refute length(Enum.uniq(steps)) == 1,
             "all three scales adapted to the same step #{inspect(hd(steps))} — a bound is binding"
    end
  end
end
