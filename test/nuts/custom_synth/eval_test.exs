defmodule Exmc.NUTS.CustomSynth.EvalTest do
  use ExUnit.Case, async: true

  alias Exmc.NUTS.CustomSynth.Eval

  @moduledoc """
  The check `Exmc.NUTS.CustomSynth.Eval` exists to perform, performed on it.

  That module's `@moduledoc` describes R1.5: trace a function to an
  `Nx.Defn.Expr`, walk it with `Eval`, and compare against
  `Nx.Defn.Evaluator` on the same inputs. Agreement certifies that the GLSL
  emitter beside it — same dispatch structure, string leaves instead of
  numeric ones — is structurally faithful to Defn's graph.

  Until now nothing ran that check. `Eval` had no test and no caller in
  `lib/`, and two of its op handlers were captures of functions that do not
  exist: `&:math.log1p/1` and `&:math.expm1/1`. Erlang's `:math` has neither.
  A capture of a missing remote function is legal, so it compiled with a
  warning and waited to raise `UndefinedFunctionError` until something
  evaluated one — which is any model with a positive-constrained or bounded
  RV, because `softplus` and `logit` inverse transforms are built out of
  `Nx.log1p/1` and `Nx.expm1/1`.

  A dead validator reads exactly like a passing one.
  """

  # Every op is exercised on a value where it is defined and where the naive
  # and careful implementations of it would visibly differ if one were used.
  @unary [
    {:exp, &Nx.exp/1, 0.7},
    {:log, &Nx.log/1, 2.5},
    {:log1p, &Nx.log1p/1, 0.5},
    {:expm1, &Nx.expm1/1, 0.5},
    {:sqrt, &Nx.sqrt/1, 2.0},
    {:rsqrt, &Nx.rsqrt/1, 2.0},
    {:sin, &Nx.sin/1, 0.7},
    {:cos, &Nx.cos/1, 0.7},
    {:tan, &Nx.tan/1, 0.7},
    {:asin, &Nx.asin/1, 0.5},
    {:acos, &Nx.acos/1, 0.5},
    {:atan, &Nx.atan/1, 0.7},
    {:sinh, &Nx.sinh/1, 0.7},
    {:cosh, &Nx.cosh/1, 0.7},
    {:tanh, &Nx.tanh/1, 0.7},
    {:floor, &Nx.floor/1, 2.7},
    {:ceil, &Nx.ceil/1, 2.3},
    {:abs, &Nx.abs/1, -1.5},
    {:negate, &Nx.negate/1, 1.5},
    {:sigmoid, &Nx.sigmoid/1, 0.7}
  ]

  @binary [
    {:add, &Nx.add/2, 1.5, 2.25},
    {:subtract, &Nx.subtract/2, 1.5, 2.25},
    {:multiply, &Nx.multiply/2, 1.5, 2.25},
    {:divide, &Nx.divide/2, 1.5, 2.25},
    {:pow, &Nx.pow/2, 2.0, 3.0},
    {:min, &Nx.min/2, 1.5, 2.25},
    {:max, &Nx.max/2, 1.5, 2.25},
    {:remainder, &Nx.remainder/2, 7.5, 2.0},
    {:atan2, &Nx.atan2/2, 1.5, 2.25}
  ]

  defp trace1(fun), do: Nx.Defn.debug_expr(fun).(Nx.template({}, :f64))
  defp trace2(fun), do: Nx.Defn.debug_expr(fun).(Nx.template({}, :f64), Nx.template({}, :f64))

  defp reference1(fun, x), do: Nx.to_number(fun.(Nx.tensor(x, type: :f64)))

  defp reference2(fun, x, y),
    do: Nx.to_number(fun.(Nx.tensor(x, type: :f64), Nx.tensor(y, type: :f64)))

  describe "every unary op agrees with the Nx reference" do
    for {name, fun, x} <- @unary do
      test "#{name}" do
        {fun, x} = {unquote(fun), unquote(x)}

        assert {:ok, ours} = Eval.evaluate(trace1(fun), [x]),
               "#{unquote(name)} did not evaluate"

        assert_in_delta ours, reference1(fun, x), 1.0e-12
      end
    end
  end

  describe "every binary op agrees with the Nx reference" do
    for {name, fun, x, y} <- @binary do
      test "#{name}" do
        {fun, x, y} = {unquote(fun), unquote(x), unquote(y)}

        assert {:ok, ours} = Eval.evaluate(trace2(fun), [x, y]),
               "#{unquote(name)} did not evaluate"

        assert_in_delta ours, reference2(fun, x, y), 1.0e-12
      end
    end
  end

  describe "log1p and expm1 — the regression, and the shape of the fix" do
    test "they evaluate at all" do
      # The literal defect. Before the fix both of these raised
      # UndefinedFunctionError, because &:math.log1p/1 and &:math.expm1/1 name
      # functions Erlang does not have.
      refute function_exported?(:math, :log1p, 1),
             "Erlang's :math grew log1p — revisit the fix in eval.ex, the naive " <>
               "form may no longer be the one that matches Nx"

      refute function_exported?(:math, :expm1, 1),
             "Erlang's :math grew expm1 — same"

      assert {:ok, _} = Eval.evaluate(trace1(&Nx.log1p/1), [0.5])
      assert {:ok, _} = Eval.evaluate(trace1(&Nx.expm1/1), [0.5])
    end

    test "they match Nx across four decades, including where Nx loses the value" do
      # This is the test that pins the *choice*, not just the behaviour.
      #
      # Nx computes both naively — Nx.log1p(1.0e-16) is 0.0, not 1.0e-16 — and
      # so do the GLSL helpers this walker's sibling emits
      # (multi_rv_custom_spec.ex:680,684, `_safe` in the name
      # notwithstanding). Eval's job is to agree with the reference, so it
      # computes naively too.
      #
      # If someone "improves" eval.ex to the numerically stable form, this
      # test goes red at 1.0e-16 — which is the correct outcome, because the
      # accuracy of these ops belongs to Nx and to the shaders, and moving it
      # means moving all three together.
      for x <- [1.0e-16, 1.0e-8, 1.0e-4, 0.5, 5.0] do
        {:ok, ours} = Eval.evaluate(trace1(&Nx.log1p/1), [x])
        assert ours === reference1(&Nx.log1p/1, x), "log1p disagreed at x = #{x}"

        {:ok, ours} = Eval.evaluate(trace1(&Nx.expm1/1), [x])
        assert ours === reference1(&Nx.expm1/1, x), "expm1 disagreed at x = #{x}"
      end
    end

    test "the transforms that reach these ops evaluate end to end" do
      # Why the defect mattered: softplus and logit inverse transforms are
      # where log1p and expm1 enter a real model. `Exmc.Transform`'s softplus
      # is log1p(exp(-|x|)) + max(x, 0).
      softplus = fn x ->
        Nx.add(Nx.max(x, Nx.tensor(0.0, type: :f64)), Nx.log1p(Nx.exp(Nx.negate(Nx.abs(x)))))
      end

      inv_softplus = fn x -> Nx.log(Nx.expm1(x)) end

      for x <- [-3.0, -0.25, 0.25, 3.0] do
        assert {:ok, ours} = Eval.evaluate(trace1(softplus), [x])
        assert_in_delta ours, reference1(softplus, x), 1.0e-12
      end

      for x <- [0.25, 1.0, 3.0] do
        assert {:ok, ours} = Eval.evaluate(trace1(inv_softplus), [x])
        assert_in_delta ours, reference1(inv_softplus, x), 1.0e-12
      end
    end
  end

  describe "the walker on a composite expression" do
    test "a log-density-shaped expression agrees with the reference" do
      # Nothing here is new machinery; the point is that the ops compose the
      # way a traced logpdf composes rather than only standing alone.
      logpdf = fn x, sigma ->
        z = Nx.divide(x, sigma)

        Nx.subtract(
          Nx.multiply(Nx.tensor(-0.5, type: :f64), Nx.multiply(z, z)),
          Nx.log(Nx.multiply(sigma, Nx.tensor(2.5066282746310002, type: :f64)))
        )
      end

      for {x, sigma} <- [{0.0, 1.0}, {1.5, 2.0}, {-2.25, 0.5}] do
        assert {:ok, ours} = Eval.evaluate(trace2(logpdf), [x, sigma])
        assert_in_delta ours, reference2(logpdf, x, sigma), 1.0e-12
      end
    end
  end
end
