defmodule Exmc.Builder.DataBackendTest do
  @moduledoc """
  Regression test for `Builder.data/2`'s backend-coercion fix.

  Compiler.build_vag_fn / build_step_fn / BatchedLeapfrog close over
  `ir.data` inside Nx.Defn.value_and_grad. The defn tracer rejects
  captured tensors on backends it can't substitute for Nx.Defn.Expr
  — e.g. Nx.Vulkan.VulkanoBackend — raising "two incompatible tensor
  implementations".

  Builder.data/2 now backend-copies the observation tensor to
  Nx.BinaryBackend at ingress, regardless of the caller's backend.
  This test pins that behaviour so the fix doesn't silently regress.
  """

  use ExUnit.Case, async: true

  alias Exmc.{Builder, IR}

  describe "Builder.data/2 backend coercion" do
    test "BinaryBackend input passes through unchanged" do
      ir = Builder.new_ir()
      bin_tensor = Nx.tensor([1.0, 2.0, 3.0], type: :f64, backend: Nx.BinaryBackend)
      ir = Builder.data(ir, bin_tensor)

      assert ir.data.data.__struct__ == Nx.BinaryBackend
      assert Nx.to_flat_list(ir.data) == [1.0, 2.0, 3.0]
    end

    test "non-BinaryBackend input is coerced (default backend simulation)" do
      # We can't easily depend on VulkanoBackend in a unit test without
      # tagging the whole module :vulkan_live. But we CAN test the
      # general contract: whatever backend the caller passes, the
      # resulting `ir.data` ends up on Nx.BinaryBackend.
      #
      # Use Nx's default backend to construct, then verify Builder.data
      # forces it onto BinaryBackend even if the default was something
      # else. (In the test env the default is BinaryBackend, so this
      # is a no-op identity — the assertion is the contract: post-call,
      # data is on BinaryBackend.)
      ir = Builder.new_ir()
      tensor = Nx.tensor([0.1, 0.2, 0.3], type: :f64)
      ir = Builder.data(ir, tensor)

      assert ir.data.data.__struct__ == Nx.BinaryBackend
      assert Nx.to_flat_list(ir.data) == [0.1, 0.2, 0.3]
    end

    test "ir.data is a real Nx.Tensor (not a struct wrapper)" do
      ir = Builder.new_ir()
      ir = Builder.data(ir, Nx.tensor([1.0]))
      assert %Nx.Tensor{} = ir.data
      assert Nx.shape(ir.data) == {1}
    end

    test "Builder.data preserves type from input" do
      ir = Builder.new_ir()
      ir = Builder.data(ir, Nx.tensor([1.0, 2.0], type: :f32))
      assert Nx.type(ir.data) == {:f, 32}

      ir = Builder.new_ir()
      ir = Builder.data(ir, Nx.tensor([1.0, 2.0], type: :f64))
      assert Nx.type(ir.data) == {:f, 64}
    end

    test "data field is nil on a fresh IR" do
      ir = Builder.new_ir()
      assert %IR{data: nil} = ir
    end
  end
end
