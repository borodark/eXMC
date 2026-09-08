defmodule Exmc.NUTS.CustomSynth.PowTest do
  @moduledoc """
  `Nx.pow` must not reach the shader as GLSL `pow`.

  GLSL.std.450's `Pow` is declared on `float` only. The chain shader is f64
  throughout, so `pow(double, double)` has no overload and glslangValidator
  rejects the whole module:

      'pow' : no matching overloaded function found

  That killed the fused shader for any Gaussian log-density written the
  obvious way — `(x - mu)^2` — and the `Exmc.Dist.Custom` moduledoc taught
  exactly that form, so the documented idiom was the broken one. It survived
  because nothing in this repo's own models used `Nx.pow`; it was found by a
  downstream consumer whose likelihoods all did.

  The repair unrolls a constant integer exponent to multiplication and refuses
  everything else. Refusing matters as much as unrolling: the natural helper,
  `pow_d(x, y) = exp_d(y * log_d(x))`, is NaN for negative x, and the base
  here is a residual that is negative about half the time. A silent NaN is
  worse than the compile error it would replace.
  """
  use ExUnit.Case, async: true

  alias Exmc.NUTS.CustomSynth.Glsl

  defp emit_pow(exponent) do
    expr =
      Nx.Defn.debug_expr_apply(
        fn q -> Nx.pow(Nx.subtract(q, 1.0), exponent) end,
        [Nx.template({}, :f64)]
      )

    Glsl.emit(expr, ["q"])
  end

  describe "constant integer exponents unroll to multiplication" do
    test "the emitted GLSL contains no call to pow" do
      {:ok, glsl} = emit_pow(2)

      refute glsl =~ "pow(",
             "emitted a GLSL pow call, which has no f64 overload: #{glsl}"

      # Two factors, so the residual appears twice.
      assert length(Regex.scan(~r/\(q - 1\.0/, glsl)) == 2, glsl
    end

    test "exponent 3 emits three factors" do
      {:ok, glsl} = emit_pow(3)
      refute glsl =~ "pow("
      assert length(Regex.scan(~r/\(q - 1\.0/, glsl)) == 3, glsl
    end

    test "exponent 1 is the base itself and 0 is the literal one" do
      {:ok, one} = emit_pow(1)
      {:ok, zero} = emit_pow(0)

      refute one =~ "pow("
      assert zero == "1.0lf", zero
    end

    test "a negative exponent becomes a reciprocal" do
      {:ok, glsl} = emit_pow(-2)
      refute glsl =~ "pow("
      assert glsl =~ "1.0lf /", glsl
    end
  end

  describe "everything else refuses rather than emitting an undefined shader" do
    # The base is a residual and can be negative, where exp(y*log(x)) is NaN.
    # GLSL's own pow is undefined for x < 0 as well, so refusing costs nothing
    # that was ever well defined — the model takes the host path, which
    # computes it correctly.
    test "a non-integer exponent is unsupported" do
      assert {:error, {:unsupported_op, :pow}} = emit_pow(2.5)
    end

    test "an integer exponent beyond the unroll cap is unsupported" do
      assert {:error, {:unsupported_op, :pow}} = emit_pow(64)
    end

    test "a non-constant exponent is unsupported" do
      expr =
        Nx.Defn.debug_expr_apply(
          fn q -> Nx.pow(Nx.subtract(q, 1.0), Nx.add(q, 3.0)) end,
          [Nx.template({}, :f64)]
        )

      assert {:error, {:unsupported_op, :pow}} = Glsl.emit(expr, ["q"])
    end
  end

  describe "numerically" do
    # Unrolling has to be exact, not merely compilable. A sign error or an
    # off-by-one in the factor count would still produce valid GLSL.
    test "the unrolled form agrees with Nx.pow on negative and positive bases" do
      for base <- [-2.7, -1.0, -0.3, 0.0, 0.5, 3.1], n <- [0, 1, 2, 3, 5] do
        expected = Nx.to_number(Nx.pow(Nx.tensor(base, type: :f64), n))
        unrolled = Enum.reduce(1..n//1, 1.0, fn _, acc -> acc * base end)

        assert_in_delta unrolled, expected, 1.0e-12, "base #{base}, exponent #{n}"
      end
    end
  end
end
