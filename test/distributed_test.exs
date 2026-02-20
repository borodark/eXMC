defmodule Exmc.DistributedTest do
  use ExUnit.Case, async: false

  @moduletag timeout: 120_000

  alias Exmc.Builder
  alias Exmc.NUTS.{Distributed, Sampler}
  alias Exmc.Dist.{Normal, Exponential}

  # --- Helper: build simple Normal-Normal model ---

  defp build_simple_model do
    y = Nx.tensor([2.1, 1.8, 2.3, 1.9, 2.0], type: :f64)

    Builder.new_ir()
    |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(10.0)})
    |> Builder.rv("sigma", Exponential, %{lambda: Nx.tensor(1.0)})
    |> Builder.rv("y", Normal, %{mu: "mu", sigma: "sigma"})
    |> Builder.obs("y_obs", "y", y)
  end

  # --- Tests that run without peer nodes (coordinator-only) ---

  describe "coordinator-only (single node)" do
    test "distributed sample with nodes: [node()] matches local sampling" do
      ir = build_simple_model()
      init = %{"mu" => Nx.tensor(2.0), "sigma" => Nx.tensor(1.0)}

      # Run distributed (coordinator only, 1 chain)
      {[trace_d], [stats_d]} =
        Distributed.sample_chains(ir,
          nodes: [node()],
          chains_per_node: 1,
          init_values: init,
          num_warmup: 200,
          num_samples: 300,
          seed: 42
        )

      assert Map.has_key?(trace_d, "mu")
      assert Map.has_key?(trace_d, "sigma")
      assert {300} == Nx.shape(trace_d["mu"])
      assert stats_d.num_warmup == 0
      assert stats_d.num_samples == 300
      assert stats_d.step_size > 0.0

      # Verify posterior recovery
      mu_mean = Nx.to_number(Nx.mean(trace_d["mu"]))
      assert_in_delta mu_mean, 2.0, 0.5
    end

    test "distributed sample with multiple chains on coordinator" do
      ir = build_simple_model()
      init = %{"mu" => Nx.tensor(2.0), "sigma" => Nx.tensor(1.0)}

      {traces, stats_list} =
        Distributed.sample_chains(ir,
          nodes: [node()],
          chains_per_node: 3,
          init_values: init,
          num_warmup: 200,
          num_samples: 200,
          seed: 42
        )

      assert length(traces) == 3
      assert length(stats_list) == 3

      # Each chain should produce valid samples
      for {trace, stats} <- Enum.zip(traces, stats_list) do
        assert {200} == Nx.shape(trace["mu"])
        assert stats.divergences >= 0
      end

      # Chains should have different samples (different seeds)
      [t1, t2, _t3] = traces
      mu1 = Nx.to_number(Nx.mean(t1["mu"]))
      mu2 = Nx.to_number(Nx.mean(t2["mu"]))
      # Means should be similar but samples should differ
      assert_in_delta mu1, mu2, 1.0
    end

    test "sample_compiled_tuned produces valid samples with pre-computed tuning" do
      ir = build_simple_model()
      init = %{"mu" => Nx.tensor(2.0), "sigma" => Nx.tensor(1.0)}

      # First, run a normal sampling to get tuning params
      {_trace, stats} = Sampler.sample(ir, init, num_warmup: 200, num_samples: 50, seed: 42)

      tuning = %{
        epsilon: stats.step_size,
        inv_mass: stats.inv_mass_diag,
        chol_cov: nil
      }

      # Now run with pre-computed tuning (no warmup)
      compiled = Sampler.compile(ir)

      {trace, stats2} =
        Sampler.sample_compiled_tuned(compiled, tuning, init,
          num_samples: 300,
          seed: 99
        )

      assert {300} == Nx.shape(trace["mu"])
      assert stats2.num_warmup == 0
      assert stats2.step_size == stats.step_size

      mu_mean = Nx.to_number(Nx.mean(trace["mu"]))
      assert_in_delta mu_mean, 2.0, 0.5
    end
  end

  # --- Tests that require peer nodes ---

  describe "multi-node" do
    @describetag :distributed

    setup do
      # Skip if distribution is not available or :peer module not present
      unless function_exported?(:peer, :start_link, 1) do
        flunk("OTP :peer module not available (run with --exclude distributed)")
      end

      case ensure_distribution() do
        :ok ->
          start_peer_nodes(2)

        {:skip, reason} ->
          flunk("Cannot start distribution: #{reason} (run with --exclude distributed)")
      end
    end

    test "distributed chains across peer nodes recover posterior", %{nodes: nodes} do
      ir = build_simple_model()
      init = %{"mu" => Nx.tensor(2.0), "sigma" => Nx.tensor(1.0)}

      all_nodes = [node() | nodes]

      {traces, stats_list} =
        Distributed.sample_chains(ir,
          nodes: all_nodes,
          chains_per_node: 1,
          init_values: init,
          num_warmup: 200,
          num_samples: 300,
          seed: 42
        )

      assert length(traces) == length(all_nodes)

      # All chains should produce valid samples
      for {trace, stats} <- Enum.zip(traces, stats_list) do
        assert {300} == Nx.shape(trace["mu"])
        assert {300} == Nx.shape(trace["sigma"])
        assert stats.num_samples == 300

        mu_mean = Nx.to_number(Nx.mean(trace["mu"]))
        assert_in_delta mu_mean, 2.0, 1.0
      end
    end

    test "fault recovery: chain retries on coordinator when peer dies", %{peers: peers} do
      ir = build_simple_model()
      init = %{"mu" => Nx.tensor(2.0), "sigma" => Nx.tensor(1.0)}

      # Kill the first peer before sampling
      [{first_peer_pid, first_node} | _] = peers
      :peer.stop(first_peer_pid)

      # Wait for node to actually disconnect
      Process.sleep(100)

      # Should still succeed — the dead node's chain falls back to coordinator
      {traces, _stats_list} =
        Distributed.sample_chains(ir,
          nodes: [first_node, node()],
          chains_per_node: 1,
          init_values: init,
          num_warmup: 200,
          num_samples: 200,
          seed: 42,
          timeout: 30_000
        )

      assert length(traces) == 2

      for trace <- traces do
        assert {200} == Nx.shape(trace["mu"])
      end
    end
  end

  # --- Peer node helpers ---

  defp ensure_distribution do
    if Node.alive?() do
      :ok
    else
      name = :"coordinator_#{System.unique_integer([:positive])}"

      case Node.start(name, :shortnames) do
        {:ok, _} -> :ok
        {:error, {:already_started, _}} -> :ok
        {:error, reason} -> {:skip, "Cannot start distribution: #{inspect(reason)}"}
      end
    end
  end

  defp start_peer_nodes(n) do
    peers =
      for i <- 1..n do
        name = :"worker_#{i}_#{:rand.uniform(100_000)}"

        case :peer.start_link(%{name: name, connection: :standard_io}) do
          {:ok, pid, node_name} ->
            # Load code paths from coordinator using MFA form (anonymous fns are :undef on peers)
            paths = :code.get_path()

            for path <- paths do
              :erpc.call(node_name, :code, :add_path, [path])
            end

            # Start required applications on peer
            :erpc.call(node_name, Application, :ensure_all_started, [:logger])
            :erpc.call(node_name, Application, :ensure_all_started, [:nx])
            :erpc.call(node_name, Application, :ensure_all_started, [:exla])

            {pid, node_name}

          {:error, reason} ->
            raise "Failed to start peer #{name}: #{inspect(reason)}"
        end
      end

    on_exit(fn ->
      for {pid, _node} <- peers do
        try do
          :peer.stop(pid)
        catch
          _, _ -> :ok
        end
      end
    end)

    %{peers: peers, nodes: Enum.map(peers, fn {_, n} -> n end)}
  end
end
