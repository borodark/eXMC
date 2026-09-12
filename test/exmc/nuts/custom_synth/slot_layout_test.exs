defmodule Exmc.NUTS.CustomSynth.SlotLayoutTest do
  @moduledoc """
  The synth path and the host sampler must agree on where each RV lives in q.

  They are two implementations reading one vector. When they disagree the
  symptom is not a crash — it is a finite, plausible, wrong posterior, which
  this project has shipped twice from exactly this class (the NCP/centred
  mismatch in `compiler.ex`, and a layout that relied on `Map` iteration order
  and permuted past 32 RVs).

  So `extract_components/1` does not compute offsets; it reads them from
  `Exmc.PointMap.build/1`, the same function `Compiler` uses. These tests pin
  that the derivation is actually shared rather than merely equivalent today.
  """
  use ExUnit.Case, async: true

  alias Exmc.{Builder, Dist, IR, PointMap}
  alias Exmc.NUTS.CustomSynth

  @f64 [type: :f64]

  defp components(ir) do
    {:ok, c} = CustomSynth.extract_components(Exmc.Rewrite.apply(ir, []))
    c
  end

  defp point_map(ir), do: PointMap.build(Exmc.Rewrite.apply(ir, []))

  describe "slots are derived from PointMap, not recomputed" do
    test "scalar RVs: one slot each, offsets and order match PointMap" do
      ir =
        IR.new()
        |> Builder.rv("b", Dist.Normal, %{mu: Nx.tensor(0.0, @f64), sigma: Nx.tensor(1.0, @f64)})
        |> Builder.rv("a", Dist.Normal, %{mu: Nx.tensor(0.0, @f64), sigma: Nx.tensor(1.0, @f64)})

      c = components(ir)
      pm = point_map(ir)

      assert Enum.map(c.slots, & &1.id) == Enum.map(pm.entries, & &1.id)
      assert Enum.map(c.slots, & &1.offset) == Enum.map(pm.entries, & &1.offset)
      assert length(c.layout) == pm.size

      # PointMap sorts free RVs by id, so "a" precedes "b" despite the build
      # order. If the synth layout ever stops agreeing, q is read in one order
      # and written in another.
      assert c.layout == ["a", "b"]
    end

    test "a vector RV occupies as many layout entries as it has coordinates" do
      ir =
        IR.new()
        |> Builder.rv(
          "beta",
          Dist.Normal,
          %{mu: Nx.tensor(0.0, @f64), sigma: Nx.tensor(5.0, @f64)},
          shape: {3}
        )
        |> Builder.rv("s", Dist.HalfNormal, %{sigma: Nx.tensor(1.0, @f64)})

      c = components(ir)
      pm = point_map(ir)

      # This is the whole point of the change: `layout` is the q order and
      # `d = length(layout)` is the shader's thread count, so a shape: {3} RV
      # must contribute three entries. It used to contribute one, and the trace
      # template built from it handed the closure a SCALAR beta.
      assert length(c.layout) == pm.size
      assert length(c.layout) == 4

      beta_slot = Enum.find(c.slots, &(&1.id == "beta"))
      assert beta_slot.length == 3
      assert beta_slot.shape == {3}

      # Names are labels; `slots` is the authority.
      assert "beta[0]" in c.layout
      assert "beta[2]" in c.layout
    end

    test "offsets are contiguous and cover exactly PointMap's size" do
      ir =
        IR.new()
        |> Builder.rv("v", Dist.Normal, %{mu: Nx.tensor(0.0, @f64), sigma: Nx.tensor(1.0, @f64)},
          shape: {4}
        )
        |> Builder.rv("w", Dist.Normal, %{mu: Nx.tensor(0.0, @f64), sigma: Nx.tensor(1.0, @f64)},
          shape: {2}
        )

      c = components(ir)

      covered =
        c.slots
        |> Enum.flat_map(fn s -> Enum.to_list(s.offset..(s.offset + s.length - 1)) end)
        |> Enum.sort()

      assert covered == Enum.to_list(0..(point_map(ir).size - 1))
    end
  end

  describe "a transformed RV uses its UNCONSTRAINED width" do
    # PointMap resolves the unconstrained length, which for some transforms
    # differs from the constrained shape. Recomputing the width locally would
    # have had to mirror that rule; deriving it cannot get it wrong.
    test "a :log-transformed scalar still occupies one slot" do
      ir =
        IR.new()
        |> Builder.rv("sigma", Dist.HalfNormal, %{sigma: Nx.tensor(1.0, @f64)})

      c = components(ir)
      pm = point_map(ir)

      assert Enum.map(c.slots, &{&1.id, &1.offset, &1.length}) ==
               Enum.map(pm.entries, &{&1.id, &1.offset, &1.length})
    end
  end
end
