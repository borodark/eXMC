defmodule Exmc.NUTS.Vulkan.ChainMetaRoutingTest do
  @moduledoc """
  Regression guard: every entry point that accepts a compiled tuple must
  route its `chain_meta` to `Tree`.

  `Tree` reads the meta from `Process.get(:exmc_chain_meta)` (tree.ex:699)
  and silently falls back to per-op dispatch when it is missing. Three
  entry points destructured the compiled tuple and threw the meta away, so
  each bypassed the fused f64 chain shader this project exists to use:

    * `sample_chains_vectorized_compiled/3` — the DEFAULT multi-chain path
      (`vectorized` defaults to true for `num_chains > 1`). 7791 ms and 0
      dispatches against 2405 ms and 1099 for the sequential path.
    * `sample_from_compiled_tuned/4` — the DISTRIBUTED path
      (`distributed.ex` `run_chain_remote`/`run_chain_local`), so the shader
      was off precisely on the gpu_node workers. 3209 ms and 0 dispatches
      against 249 ms and 508 for `sample_compiled`, with no difference in
      work to account for any of it.
    * `stream_from_compiled/4` — `sample_stream/4`, which dropped
      `multi_step_fn` as well.

  All three went unnoticed for the same reason: nothing counted dispatches.
  The one test in the area compared wall clocks against a concurrent path,
  which is host-dependent, so its failure had been carried as a known
  flake for two rounds. These tests count instead — the count is exact, it
  does not care how fast the machine is, and it fails naming the cause
  rather than a proxy for it.

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

  test "the distributed tuned path dispatches the fused chain shader" do
    # sample_from_compiled_tuned/4, reached through sample_compiled_tuned/4 —
    # what distributed.ex calls on every gpu_node worker and on the
    # coordinator's :erpc fallback.
    {_t, stats} = Sampler.sample(ir(), %{}, num_warmup: 50, num_samples: 20, seed: 42)
    tuning = %{epsilon: stats.step_size, inv_mass: stats.inv_mass_diag, chol_cov: nil}
    compiled = Exmc.Compiler.compile_for_sampling(ir(), [])

    # Non-vacuity: this model must route to the chain shader at all here,
    # otherwise zero on both sides would read as a pass.
    Dispatch.reset_dispatch_count()
    Sampler.sample_compiled(compiled, %{}, num_warmup: 0, num_samples: 100, seed: 99)
    baseline = Dispatch.dispatch_count()

    assert baseline > 0,
           "sample_compiled issued no chain dispatches on this host, so the " <>
             "assertion below would be vacuous"

    Dispatch.reset_dispatch_count()
    Sampler.sample_compiled_tuned(compiled, tuning, %{}, num_samples: 100, seed: 99)
    tuned = Dispatch.dispatch_count()

    assert tuned > 0,
           "sample_compiled_tuned issued #{tuned} chain dispatches against " <>
             "#{baseline} for sample_compiled — chain_meta is not reaching " <>
             "Tree.fused_leapfrog_meta/0 on the distributed path"
  end

  test "the streaming path dispatches the fused chain shader" do
    # sample_stream/4. Runs in THIS process, so the counter applies directly.
    # This path dropped multi_step_fn as well as chain_meta, so it was the
    # furthest from the fused path of the three.
    Dispatch.reset_dispatch_count()

    assert :ok =
             Sampler.sample_stream(ir(), self(), %{}, num_warmup: 50, num_samples: 50, seed: 42)

    # Drain so the mailbox does not leak into a later test in this module.
    receive_all = fn f ->
      receive do
        {:exmc_sample, _, _, _} -> f.(f)
        {:exmc_done, n} -> n
      after
        60_000 -> flunk("sample_stream did not finish within 60s")
      end
    end

    assert receive_all.(receive_all) == 50

    streamed = Dispatch.dispatch_count()

    assert streamed > 0,
           "sample_stream issued #{streamed} chain dispatches — chain_meta is " <>
             "not reaching Tree.fused_leapfrog_meta/0 on the streaming path"
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
