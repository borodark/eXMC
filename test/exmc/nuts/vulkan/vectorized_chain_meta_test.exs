defmodule Exmc.NUTS.Vulkan.VectorizedChainMetaTest do
  @moduledoc """
  Regression guard: the vectorized multi-chain path must reach the fused
  chain shader.

  `Sampler.sample_chains_vectorized_compiled/3` destructured the compiled
  tuple and threw `chain_meta` away. `Tree` reads it from
  `Process.get(:exmc_chain_meta)` (tree.ex:699) and silently falls back to
  per-op dispatch when it is missing, so every vectorized run bypassed the
  fused f64 chain shader this project exists to use. `vectorized` defaults to
  true for `num_chains > 1`, so that was the default multi-chain path.

  It went unnoticed because the only test watching this compared wall clocks
  against the concurrent path, which is host-dependent and had been on the
  known-failures list for two rounds. This test counts dispatches instead:
  the count is exact, it does not care how fast the machine is, and it fails
  for the actual reason rather than a proxy for it.

  Vulkan-only: `Dispatch.dispatch_count/0` counts chain-shader dispatches,
  and there are none to count under EXLA or BinaryBackend.
  """

  use ExUnit.Case, async: false

  @moduletag :requires_vulkan
  @moduletag timeout: 120_000

  alias Exmc.Builder
  alias Exmc.Dist.Normal
  alias Exmc.NUTS.Sampler
  alias Exmc.NUTS.Vulkan.Dispatch

  defp ir do
    Builder.new_ir()
    |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(5.0)})
    |> Builder.rv("x", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
    |> Builder.obs("x_obs", "x", Nx.tensor(3.0))
  end

  @opts [num_warmup: 50, num_samples: 50, seed: 42]

  test "the vectorized path dispatches the fused chain shader" do
    # Both arms run in THIS process, so the per-process counter is comparable
    # and concurrency is not a confound. Pre-fix this was 0 against ~1099.
    Dispatch.reset_dispatch_count()
    {traces_vec, _} = Sampler.sample_chains(ir(), 4, [vectorized: true] ++ @opts)
    vec_dispatches = Dispatch.dispatch_count()

    Dispatch.reset_dispatch_count()

    {traces_seq, _} =
      Sampler.sample_chains(ir(), 4, [vectorized: false, parallel: false] ++ @opts)

    seq_dispatches = Dispatch.dispatch_count()

    assert length(traces_vec) == 4
    assert length(traces_seq) == 4

    # The sequential arm establishes that this model routes to the chain
    # shader at all on this host. Without it, a zero on both sides would look
    # like a pass on a box where nothing dispatches.
    assert seq_dispatches > 0,
           "the non-vectorized path issued no chain dispatches either -- " <>
             "this model is not routing to the chain shader on this host, " <>
             "so the vectorized assertion below would be vacuous"

    assert vec_dispatches > 0,
           "vectorized path issued #{vec_dispatches} chain dispatches " <>
             "against #{seq_dispatches} for the sequential path -- chain_meta " <>
             "is not reaching Tree.fused_leapfrog_meta/0"
  end

  test "chain_meta does not leak into the calling process afterwards" do
    # The put is process-global. sample_from_compiled/3 deletes it on the
    # success path only; the vectorized path uses try/after. Either way a
    # later sample without a chain meta must not inherit this one.
    refute Process.get(:exmc_chain_meta)

    {_traces, _} = Sampler.sample_chains(ir(), 2, [vectorized: true] ++ @opts)

    refute Process.get(:exmc_chain_meta),
           "vectorized sampling left :exmc_chain_meta set in the caller"
  end
end
