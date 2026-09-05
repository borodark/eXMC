defmodule Exmc.NUTS.Vulkan.BatchPaddingTest do
  @moduledoc """
  Batched chain requests of differing depth group together and are trimmed
  back per caller.

  ## Why the partition key changed

  `BatchCoordinator` used to partition on `{meta_hash, K, |eps|}`. The comment
  justifying it was right about the shader — a single K parameterises the whole
  workgroup — and drew the wrong conclusion, because padding to the deepest
  satisfies that constraint too. It was written when no batched f64 NIF existed,
  so it reasoned about a capability nobody could exercise.

  Measured: of 300 draws with 4 chains, **48 had all four at the same
  `n_steps`**. Keying on K batches fully in 16% of draws and falls back to
  singletons in the other 84%.

  ## Why trimming is correctness-critical

  A padded instance runs more leapfrog steps than its caller asked for. Those
  steps are computed from valid state, so returning them is not a crash — it is
  trajectory the sampler never requested, which reaches the posterior. The
  prefix property that makes trimming sound (a K=k0 dispatch's first k steps are
  bit-identical to a K=k dispatch) is pinned upstream in `nx_vulkan` cccbd71.
  """
  use ExUnit.Case, async: true

  alias Exmc.NUTS.Vulkan.BatchCoordinator, as: C

  defp req(meta, eps, k), do: {:from, meta, :q, :p, :im, :obs, eps, k}

  describe "partition key" do
    test "requests differing ONLY in K group together" do
      meta = {:synthesised, "sha", ["mu"], %{}, "/tmp/x.spv", <<>>, <<>>}

      keys = for k <- [1, 3, 7, 16], do: C.partition_key(req(meta, 0.05, k))

      assert length(Enum.uniq(keys)) == 1,
             "K is back in the partition key — chains of differing depth will " <>
               "be dispatched as singletons, which measured 84% of draws"
    end

    test "different shaders and different |eps| still separate" do
      m1 = {:synthesised, "sha1", ["mu"], %{}, "/tmp/a.spv", <<>>, <<>>}
      m2 = {:synthesised, "sha2", ["mu"], %{}, "/tmp/b.spv", <<>>, <<>>}

      refute C.partition_key(req(m1, 0.05, 4)) == C.partition_key(req(m2, 0.05, 4)),
             "two different synthesised shaders cannot share one batched dispatch"

      refute C.partition_key(req(m1, 0.05, 4)) == C.partition_key(req(m1, 0.09, 4)),
             "|eps| is a single push field for the whole workgroup"
    end
  end

  describe "trim_to_requested/3" do
    defp bufs(k, d) do
      {Nx.iota({k, d}, type: :f64), Nx.iota({k, d}, type: :f64), Nx.iota({k}, type: :f64),
       Nx.iota({k, d}, type: :f64)}
    end

    test "a caller that asked for fewer steps gets exactly those steps" do
      {q, p, logp, grad} = C.trim_to_requested(bufs(8, 3), 3, 8)

      assert Nx.shape(q) == {3, 3}
      assert Nx.shape(p) == {3, 3}
      assert Nx.shape(grad) == {3, 3}
      assert Nx.shape(logp) == {3}, "logp is one scalar per step, not per dimension"
    end

    test "the trimmed prefix is the untrimmed prefix, value for value" do
      full = bufs(8, 3)
      {q, _, logp, _} = C.trim_to_requested(full, 3, 8)
      {fq, _, flogp, _} = full

      assert Nx.to_flat_list(q) == Nx.to_flat_list(Nx.slice(fq, [0, 0], [3, 3]))
      assert Nx.to_flat_list(logp) == Nx.to_flat_list(Nx.slice(flogp, [0], [3]))
    end

    test "a caller that asked for the padded depth is untouched" do
      full = bufs(8, 3)
      assert C.trim_to_requested(full, 8, 8) == full
    end

    test "a fallback tuple passes through unreshaped" do
      # Trimming must not try to slice an error reply.
      assert C.trim_to_requested({:fallback, :whatever}, 2, 8) == {:fallback, :whatever}
      assert C.trim_to_requested(:crashed, 2, 8) == :crashed
    end
  end
end
