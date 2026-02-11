defmodule Exmc.CompilerTest do
  use ExUnit.Case

  @moduledoc false

  alias Exmc.{Builder, Compiler, PointMap, Rewrite, LogProb}
  alias Exmc.Dist.{Normal, Exponential}
  import Exmc.TestHelper

  # --- Finite difference gradient helper ---

  defp finite_diff_grad(f, x, eps \\ 1.0e-5) do
    x_flat = Nx.to_flat_list(x)

    grads =
      x_flat
      |> Enum.with_index()
      |> Enum.map(fn {_val, i} ->
        x_plus =
          Nx.indexed_put(
            x,
            Nx.tensor([[i]]),
            Nx.tensor([Nx.to_flat_list(x) |> Enum.at(i) |> Kernel.+(eps)])
          )

        x_minus =
          Nx.indexed_put(
            x,
            Nx.tensor([[i]]),
            Nx.tensor([Nx.to_flat_list(x) |> Enum.at(i) |> Kernel.-(eps)])
          )

        fp = f.(x_plus) |> Nx.to_number()
        fm = f.(x_minus) |> Nx.to_number()
        (fp - fm) / (2.0 * eps)
      end)

    Nx.tensor(grads, type: :f64)
  end

  # =============================================
  # PointMap tests
  # =============================================

  describe "PointMap.build" do
    test "identifies free vs observed RVs" do
      ir =
        Builder.new_ir()
        |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
        |> Builder.rv("y", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
        |> Builder.obs("y_obs", "y", Nx.tensor(0.5))
        |> Rewrite.apply()

      pm = PointMap.build(ir)

      assert PointMap.has_entry?(pm, "x")
      refute PointMap.has_entry?(pm, "y")
      assert length(pm.entries) == 1
      assert pm.size == 1
    end

    test "all free RVs" do
      ir =
        Builder.new_ir()
        |> Builder.rv("a", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
        |> Builder.rv("b", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
        |> Rewrite.apply()

      pm = PointMap.build(ir)

      assert length(pm.entries) == 2
      assert pm.size == 2
      assert Enum.map(pm.entries, & &1.id) == ["a", "b"]
    end

    test "no free RVs (all observed)" do
      ir =
        Builder.new_ir()
        |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
        |> Builder.obs("x_obs", "x", Nx.tensor(0.5))
        |> Rewrite.apply()

      pm = PointMap.build(ir)

      assert length(pm.entries) == 0
      assert pm.size == 0
    end

    test "mixed free and observed" do
      ir =
        Builder.new_ir()
        |> Builder.rv("alpha", Exponential, %{lambda: Nx.tensor(1.0)})
        |> Builder.rv("beta", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
        |> Builder.rv("gamma", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
        |> Builder.obs("gamma_obs", "gamma", Nx.tensor(0.3))
        |> Rewrite.apply()

      pm = PointMap.build(ir)

      assert length(pm.entries) == 2
      ids = Enum.map(pm.entries, & &1.id)
      # Alphabetically sorted
      assert ids == ["alpha", "beta"]
    end

    test "transform metadata recorded correctly" do
      ir =
        Builder.new_ir()
        |> Builder.rv("x", Exponential, %{lambda: Nx.tensor(1.0)})
        |> Builder.rv("y", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
        |> Rewrite.apply()

      pm = PointMap.build(ir)

      x_entry = Enum.find(pm.entries, &(&1.id == "x"))
      y_entry = Enum.find(pm.entries, &(&1.id == "y"))

      # Exponential gets :log transform from rewrite
      assert x_entry.transform == :log
      assert y_entry.transform == nil
    end
  end

  describe "PointMap pack/unpack" do
    test "roundtrip preserves values" do
      ir =
        Builder.new_ir()
        |> Builder.rv("a", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
        |> Builder.rv("b", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
        |> Rewrite.apply()

      pm = PointMap.build(ir)

      value_map = %{"a" => Nx.tensor(1.5), "b" => Nx.tensor(-0.3)}
      flat = PointMap.pack(value_map, pm)
      recovered = PointMap.unpack(flat, pm)

      assert_close(recovered["a"], Nx.tensor(1.5))
      assert_close(recovered["b"], Nx.tensor(-0.3))
    end
  end

  describe "PointMap transforms" do
    test "to_constrained / to_unconstrained roundtrip" do
      ir =
        Builder.new_ir()
        |> Builder.rv("x", Exponential, %{lambda: Nx.tensor(1.0)})
        |> Builder.rv("y", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
        |> Rewrite.apply()

      pm = PointMap.build(ir)

      # Start with unconstrained values
      unconstrained = %{"x" => Nx.tensor(0.5), "y" => Nx.tensor(-0.2)}
      constrained = PointMap.to_constrained(unconstrained, pm)

      # x should be exp(0.5), y should stay the same (nil transform)
      assert_close(constrained["x"], Nx.exp(Nx.tensor(0.5)))
      assert_close(constrained["y"], Nx.tensor(-0.2))

      # Roundtrip back
      recovered = PointMap.to_unconstrained(constrained, pm)
      assert_close(recovered["x"], Nx.tensor(0.5))
      assert_close(recovered["y"], Nx.tensor(-0.2))
    end
  end

  # =============================================
  # Compiler tests
  # =============================================

  describe "Compiler.compile" do
    test "single Normal matches LogProb.eval" do
      ir =
        Builder.new_ir()
        |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

      {logp_fn, pm} = Compiler.compile(ir)

      x_val = Nx.tensor(0.3)
      flat = PointMap.pack(%{"x" => x_val}, pm)
      compiled_logp = logp_fn.(flat)
      expected = LogProb.eval(ir, %{"x" => x_val})

      assert_close(compiled_logp, expected)
    end

    test "Exponential with transform matches LogProb.eval" do
      ir =
        Builder.new_ir()
        |> Builder.rv("z", Exponential, %{lambda: Nx.tensor(1.5)})

      {logp_fn, pm} = Compiler.compile(ir)

      z_val = Nx.tensor(0.1)
      flat = PointMap.pack(%{"z" => z_val}, pm)
      compiled_logp = logp_fn.(flat)
      expected = LogProb.eval(ir, %{"z" => z_val})

      assert_close(compiled_logp, expected)
    end

    test "model with observations matches LogProb.eval" do
      ir =
        Builder.new_ir()
        |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
        |> Builder.obs("x_obs", "x", Nx.tensor(0.2))

      {logp_fn, pm} = Compiler.compile(ir)

      assert pm.size == 0
      compiled_logp = logp_fn.(Nx.tensor(0.0))
      expected = LogProb.eval(ir, %{})

      assert_close(compiled_logp, expected)
    end

    test "mixed model (free + observed) matches LogProb.eval" do
      ir =
        Builder.new_ir()
        |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
        |> Builder.rv("y", Normal, %{mu: Nx.tensor(1.0), sigma: Nx.tensor(2.0)})
        |> Builder.obs("y_obs", "y", Nx.tensor(0.5))

      {logp_fn, pm} = Compiler.compile(ir)

      assert pm.size == 1
      x_val = Nx.tensor(0.3)
      flat = PointMap.pack(%{"x" => x_val}, pm)
      compiled_logp = logp_fn.(flat)
      expected = LogProb.eval(ir, %{"x" => x_val})

      assert_close(compiled_logp, expected)
    end

    test "measurable obs in compiled model" do
      a = Nx.tensor([[2.0]])

      ir =
        Builder.new_ir()
        |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
        |> Builder.det("y", :matmul, [a, "x"])
        |> Builder.obs("y_obs", "y", Nx.tensor([[0.4]]))

      {logp_fn, pm} = Compiler.compile(ir)

      # x is observed through meas_obs, so no free RVs
      assert pm.size == 0
      compiled_logp = logp_fn.(Nx.tensor(0.0))
      expected = LogProb.eval(ir, %{})

      assert_close(compiled_logp, expected)
    end
  end

  describe "Compiler.value_and_grad" do
    test "Normal gradient: grad = -(x - mu) / sigma^2" do
      ir =
        Builder.new_ir()
        |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

      {vag_fn, pm} = Compiler.value_and_grad(ir)

      x_val = Nx.tensor(0.3)
      flat = PointMap.pack(%{"x" => x_val}, pm)
      {logp, grad} = vag_fn.(flat)

      expected_logp = LogProb.eval(ir, %{"x" => x_val})
      assert_close(logp, expected_logp)

      # For N(0,1): d/dx logp = -x
      expected_grad = Nx.negate(x_val)
      assert_close(Nx.reshape(grad, {}), expected_grad)
    end

    test "gradient with transform verified via finite differences" do
      ir =
        Builder.new_ir()
        |> Builder.rv("z", Exponential, %{lambda: Nx.tensor(1.5)})

      {vag_fn, pm} = Compiler.value_and_grad(ir)
      {logp_fn, _} = Compiler.compile(ir)

      z_val = Nx.tensor(0.1)
      flat = PointMap.pack(%{"z" => z_val}, pm)
      {logp, grad} = vag_fn.(flat)

      expected_logp = LogProb.eval(ir, %{"z" => z_val})
      assert_close(logp, expected_logp)

      fd_grad = finite_diff_grad(logp_fn, flat)
      assert_close(grad, fd_grad, 1.0e-4)
    end

    test "gradient with multiple free RVs" do
      ir =
        Builder.new_ir()
        |> Builder.rv("a", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
        |> Builder.rv("b", Normal, %{mu: Nx.tensor(1.0), sigma: Nx.tensor(2.0)})

      {vag_fn, pm} = Compiler.value_and_grad(ir)
      {logp_fn, _} = Compiler.compile(ir)

      value_map = %{"a" => Nx.tensor(0.5), "b" => Nx.tensor(-0.3)}
      flat = PointMap.pack(value_map, pm)
      {_logp, grad} = vag_fn.(flat)

      fd_grad = finite_diff_grad(logp_fn, flat)
      assert_close(grad, fd_grad, 1.0e-3)
    end
  end
end
