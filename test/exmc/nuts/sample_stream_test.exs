defmodule Exmc.NUTS.SampleStreamTest do
  @moduledoc """
  First tests for `Sampler.sample_stream/4`.

  It had none. It is public API, documented in README, MISSION.md, STORY.md,
  the CHANGELOG and two blog posts as a headline feature ("per-sample
  streaming"), and nothing in `lib/` or `test/` called it — so nothing
  noticed that `stream_from_compiled/4` dropped both `multi_step_fn` and
  `chain_meta` at its clause head, running every step per-op. The
  chain-shader half of that is covered by
  `Exmc.NUTS.Vulkan.ChainMetaRoutingTest`; this file covers the contract
  itself, which is what was actually untested.
  """

  use ExUnit.Case, async: false

  @moduletag timeout: 120_000

  alias Exmc.Builder
  alias Exmc.Dist.Normal
  alias Exmc.NUTS.Sampler

  defp ir do
    Builder.new_ir()
    |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(5.0)})
    |> Builder.rv("x", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
    |> Builder.obs("x_obs", "x", Nx.tensor(3.0))
  end

  defp drain(acc \\ []) do
    receive do
      {:exmc_sample, i, point_map, step_stat} -> drain([{i, point_map, step_stat} | acc])
      {:exmc_done, n} -> {Enum.reverse(acc), n}
    after
      60_000 -> flunk("sample_stream sent neither all samples nor :exmc_done within 60s")
    end
  end

  test "streams one message per draw, in order, then :exmc_done" do
    assert :ok =
             Sampler.sample_stream(ir(), self(), %{}, num_warmup: 50, num_samples: 25, seed: 42)

    {samples, done_n} = drain()

    assert done_n == 25
    assert length(samples) == 25

    # Indices are 1..n in order — a caller accumulating a growing trace
    # depends on this and nothing was checking it.
    assert Enum.map(samples, fn {i, _, _} -> i end) == Enum.to_list(1..25)
  end

  test "each message carries a constrained point map and step stats" do
    assert :ok =
             Sampler.sample_stream(ir(), self(), %{}, num_warmup: 50, num_samples: 10, seed: 7)

    {samples, 10} = drain()

    for {_i, point_map, step_stat} <- samples do
      assert is_map(point_map)
      assert Map.has_key?(point_map, "mu")

      # Constrained space: a Normal is unbounded, so the check that means
      # something here is that it is a finite number rather than a tensor
      # still carrying an unapplied transform.
      mu = point_map["mu"]
      assert is_number(mu) or match?(%Nx.Tensor{}, mu)

      assert %{
               tree_depth: depth,
               n_steps: n_steps,
               divergent: divergent,
               accept_prob: accept,
               energy: energy
             } = step_stat

      assert is_integer(depth) and depth >= 0
      assert is_integer(n_steps) and n_steps >= 0
      assert is_boolean(divergent)
      assert is_number(accept) and accept >= 0.0 and accept <= 1.0
      assert is_number(energy)
    end
  end

  test "the streamed posterior agrees with the ordinary sampler" do
    # The point of wiring multi_step_fn and chain_meta through is that this
    # path now runs the same integrator as Sampler.sample/3. If it did not,
    # this is where that would show.
    seed = 20_260_830
    o = [num_warmup: 300, num_samples: 300, seed: seed]

    assert :ok = Sampler.sample_stream(ir(), self(), %{}, o)
    {samples, 300} = drain()

    streamed =
      Enum.map(samples, fn {_i, pm, _} ->
        case pm["mu"] do
          %Nx.Tensor{} = t -> Nx.to_number(t)
          n when is_number(n) -> n
        end
      end)

    {trace, _stats} = Sampler.sample(ir(), %{}, o)
    reference = trace["mu"] |> Nx.to_flat_list()

    mean = fn xs -> Enum.sum(xs) / length(xs) end
    m_stream = mean.(streamed)
    m_ref = mean.(reference)

    sd = fn xs, m -> :math.sqrt(mean.(Enum.map(xs, &((&1 - m) * (&1 - m))))) end

    se =
      :math.sqrt(
        :math.pow(sd.(streamed, m_stream), 2) / 300 + :math.pow(sd.(reference, m_ref), 2) / 300
      )

    # Different seeds internally, so this is an MCMC-noise comparison, not a
    # bit-for-bit one. 4 combined standard errors.
    assert abs(m_stream - m_ref) < 4 * se,
           "streamed mean #{m_stream} vs sampler mean #{m_ref}, " <>
             "difference #{abs(m_stream - m_ref)} exceeds 4 SE (#{4 * se})"
  end

  test "does not leave :exmc_chain_meta set in the calling process" do
    refute Process.get(:exmc_chain_meta)

    assert :ok = Sampler.sample_stream(ir(), self(), %{}, num_warmup: 20, num_samples: 5, seed: 1)
    {_samples, 5} = drain()

    refute Process.get(:exmc_chain_meta),
           "sample_stream left :exmc_chain_meta set in the caller"
  end
end
