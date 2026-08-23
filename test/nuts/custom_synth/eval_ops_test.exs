defmodule Exmc.NUTS.CustomSynth.EvalOpsTest do
  @moduledoc """
  Op-level behaviour of the Eval walker: arithmetic, comparison/select, and
  reverse-mode gradients through the ops the emitter actually produces.

  Sibling to `eval_test.exs`, which covers a different thing — the fidelity of
  log1p/expm1 against Nx across four decades. The two were separate files in
  separate trees under the SAME module name, so porting this one collided; the
  collision is the only reason anyone noticed they were complementary rather
  than duplicates.
  """

  use ExUnit.Case, async: true

  # R1.5 — correctness gate via the parallel Elixir walker.
  # The hypothesis under test: the GLSL emitter visits ops in the
  # right order and applies the right transformations.  We verify
  # this by writing a parallel walker (Eval) with the same op
  # dispatch structure but Elixir-float leaves, then asserting
  # Eval agrees with Nx.Defn.Evaluator on random inputs.
  #
  # If Eval matches Defn on ~all inputs, the GLSL walker is
  # structurally correct.  The remaining unknown — GLSL on Vulkan
  # vs reference — gets validated at R2 when we wire the leapfrog
  # template integration.

  alias Exmc.NUTS.CustomSynth.Eval

  defp trace(fun, templates), do: Nx.Defn.debug_expr_apply(fun, templates)

  defp reference(fun, args) do
    prev = Nx.default_backend()
    Nx.default_backend(Nx.BinaryBackend)

    try do
      Nx.Defn.jit_apply(fun, args, compiler: Nx.Defn.Evaluator)
      |> Nx.to_number()
    after
      Nx.default_backend(prev)
    end
  end

  describe "Eval matches Defn — simple ops" do
    test "add / multiply" do
      fun = fn x, y ->
        Nx.add(Nx.multiply(x, Nx.tensor(2.0, type: :f64, backend: Nx.BinaryBackend)), y)
      end

      for _ <- 1..50 do
        x = :rand.uniform() * 10.0 - 5.0
        y = :rand.uniform() * 10.0 - 5.0

        ref =
          reference(fun, [
            Nx.tensor(x, type: :f64, backend: Nx.BinaryBackend),
            Nx.tensor(y, type: :f64, backend: Nx.BinaryBackend)
          ])

        expr = trace(fun, [Nx.template({}, :f64), Nx.template({}, :f64)])
        {:ok, ours} = Eval.evaluate(expr, [x, y])

        assert_in_delta(ours, ref, 1.0e-6)
      end
    end

    test "log / exp / sigmoid" do
      fun = fn x ->
        Nx.sigmoid(
          Nx.log(Nx.add(Nx.exp(x), Nx.tensor(1.0, type: :f64, backend: Nx.BinaryBackend)))
        )
      end

      for _ <- 1..50 do
        x = :rand.uniform() * 4.0 - 2.0
        ref = reference(fun, [Nx.tensor(x, type: :f64, backend: Nx.BinaryBackend)])
        expr = trace(fun, [Nx.template({}, :f64)])
        {:ok, ours} = Eval.evaluate(expr, [x])
        assert_in_delta(ours, ref, 1.0e-6)
      end
    end

    test "min / max + comparison + select (via grad of min)" do
      # grad of min(x, c) → indicator → exercises :less + :select
      fun = fn x ->
        Nx.Defn.grad(x, fn x ->
          Nx.min(x, Nx.tensor(0.5, type: :f64, backend: Nx.BinaryBackend))
        end)
      end

      for _ <- 1..50 do
        x = :rand.uniform() * 2.0 - 1.0
        ref = reference(fun, [Nx.tensor(x, type: :f64, backend: Nx.BinaryBackend)])
        expr = trace(fun, [Nx.template({}, :f64)])
        {:ok, ours} = Eval.evaluate(expr, [x])
        assert_in_delta(ours, ref, 1.0e-6)
      end
    end
  end

  describe "Eval matches Defn — Normal log_pdf and its gradient" do
    test "Normal logpdf value" do
      # logpdf(x | μ=0, σ=1) = -0.5 * x² - 0.5 * log(2π)
      fun = fn x ->
        Nx.subtract(
          Nx.multiply(
            Nx.tensor(-0.5, type: :f64, backend: Nx.BinaryBackend),
            Nx.pow(x, Nx.tensor(2.0, type: :f64, backend: Nx.BinaryBackend))
          ),
          Nx.multiply(
            Nx.tensor(0.5, type: :f64, backend: Nx.BinaryBackend),
            Nx.log(Nx.tensor(2.0 * :math.pi(), type: :f64, backend: Nx.BinaryBackend))
          )
        )
      end

      for _ <- 1..100 do
        x = :rand.uniform() * 6.0 - 3.0
        ref = reference(fun, [Nx.tensor(x, type: :f64, backend: Nx.BinaryBackend)])
        expr = trace(fun, [Nx.template({}, :f64)])
        {:ok, ours} = Eval.evaluate(expr, [x])
        assert_in_delta(ours, ref, 1.0e-6)
      end
    end

    test "grad of Normal logpdf — should be -x" do
      fun = fn x ->
        Nx.Defn.grad(x, fn x ->
          Nx.multiply(
            Nx.tensor(-0.5, type: :f64, backend: Nx.BinaryBackend),
            Nx.pow(x, Nx.tensor(2.0, type: :f64, backend: Nx.BinaryBackend))
          )
        end)
      end

      for _ <- 1..100 do
        x = :rand.uniform() * 4.0 - 2.0
        ref = reference(fun, [Nx.tensor(x, type: :f64, backend: Nx.BinaryBackend)])
        expr = trace(fun, [Nx.template({}, :f64)])
        {:ok, ours} = Eval.evaluate(expr, [x])
        # ref should ≈ -x; eval should match ref
        assert_in_delta(ours, ref, 1.0e-6)
        assert_in_delta(ours, -x, 1.0e-6)
      end
    end
  end

  describe "Eval matches Defn — softmax-mixture (regime model's signature shape)" do
    test "log_w0 = -log(1 + exp(w1) + exp(w2)) — value" do
      fun = fn w1, w2 ->
        ew1 = Nx.exp(Nx.min(w1, Nx.tensor(10.0, type: :f64, backend: Nx.BinaryBackend)))
        ew2 = Nx.exp(Nx.min(w2, Nx.tensor(10.0, type: :f64, backend: Nx.BinaryBackend)))
        z = Nx.add(Nx.add(Nx.tensor(1.0, type: :f64, backend: Nx.BinaryBackend), ew1), ew2)
        Nx.subtract(Nx.tensor(0.0, type: :f64, backend: Nx.BinaryBackend), Nx.log(z))
      end

      for _ <- 1..100 do
        w1 = :rand.uniform() * 6.0 - 3.0
        w2 = :rand.uniform() * 6.0 - 3.0

        ref =
          reference(fun, [
            Nx.tensor(w1, type: :f64, backend: Nx.BinaryBackend),
            Nx.tensor(w2, type: :f64, backend: Nx.BinaryBackend)
          ])

        expr = trace(fun, [Nx.template({}, :f64), Nx.template({}, :f64)])
        {:ok, ours} = Eval.evaluate(expr, [w1, w2])

        assert_in_delta(ours, ref, 1.0e-6)
      end
    end

    test "grad(log_w0) w.r.t. w1 — exercises chain rule on exp/log/min/div" do
      fun = fn w1, w2 ->
        Nx.Defn.grad(w1, fn w1 ->
          ew1 = Nx.exp(Nx.min(w1, Nx.tensor(10.0, type: :f64, backend: Nx.BinaryBackend)))
          ew2 = Nx.exp(Nx.min(w2, Nx.tensor(10.0, type: :f64, backend: Nx.BinaryBackend)))
          z = Nx.add(Nx.add(Nx.tensor(1.0, type: :f64, backend: Nx.BinaryBackend), ew1), ew2)
          Nx.subtract(Nx.tensor(0.0, type: :f64, backend: Nx.BinaryBackend), Nx.log(z))
        end)
      end

      for _ <- 1..100 do
        w1 = :rand.uniform() * 6.0 - 3.0
        w2 = :rand.uniform() * 6.0 - 3.0

        ref =
          reference(fun, [
            Nx.tensor(w1, type: :f64, backend: Nx.BinaryBackend),
            Nx.tensor(w2, type: :f64, backend: Nx.BinaryBackend)
          ])

        expr = trace(fun, [Nx.template({}, :f64), Nx.template({}, :f64)])
        {:ok, ours} = Eval.evaluate(expr, [w1, w2])

        assert_in_delta(ours, ref, 1.0e-6)
      end
    end
  end
end
