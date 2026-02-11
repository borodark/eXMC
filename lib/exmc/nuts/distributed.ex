defmodule Exmc.NUTS.Distributed do
  @moduledoc """
  Distributed NUTS sampling across Erlang nodes.

  Sends model IR to each node for independent compilation. Runs warmup
  on the coordinator node, then broadcasts adapted tuning parameters
  (step size, mass matrix) to all nodes. Each node runs its chain(s)
  independently with these shared parameters.

  Results are collected back as `{[traces], [stats]}`, same format as
  `Sampler.sample_chains/3`.

  ## Example

      nodes = [:"worker1@host", :"worker2@host"]
      {traces, stats} = Distributed.sample_chains(ir, nodes: nodes, seed: 42)

  ## Requirements

  - All nodes must have the `Exmc` application code loaded.
  - EXLA is compiled independently on each node (supports heterogeneous hardware).
  - Nx tensors cross node boundaries as BinaryBackend binaries.
  """

  alias Exmc.Compiler
  alias Exmc.NUTS.Sampler

  require Logger

  @default_opts [
    chains_per_node: 1,
    num_warmup: 1000,
    num_samples: 1000,
    max_tree_depth: 10,
    target_accept: 0.8,
    seed: 0,
    timeout: :infinity
  ]

  @doc """
  Sample chains distributed across connected Erlang nodes.

  Each node runs `chains_per_node` chains. Model is compiled
  independently on each node. Warmup runs on coordinator only.

  Returns `{[traces], [stats]}`.

  ## Options

  - `:nodes` — list of node names to use (default: `[node()]`, coordinator only)
  - `:chains_per_node` — chains per node (default: 1)
  - `:init_values` — initial parameter values (default: `%{}`)
  - `:timeout` — per-chain timeout in ms (default: `:infinity`)
  - Plus all options from `Sampler.sample/3` (`:num_warmup`, `:num_samples`, `:seed`, etc.)
  """
  def sample_chains(ir, opts \\ []) do
    opts = Keyword.merge(@default_opts, opts)
    nodes = Keyword.get(opts, :nodes, [node()])
    chains_per_node = opts[:chains_per_node]
    init_values = Keyword.get(opts, :init_values, %{})
    timeout = opts[:timeout]
    compile_opts = Keyword.take(opts, [:ncp, :device])

    # Validate nodes are reachable and have Exmc loaded
    validate_nodes!(nodes)

    # Phase 1: Warmup on coordinator
    {tuning, _pm, _ncp_info} = run_coordinator_warmup(ir, init_values, opts, compile_opts)

    # Phase 2: Dispatch chains to nodes
    sampler_opts =
      opts
      |> Keyword.drop([:nodes, :chains_per_node, :init_values, :timeout])
      |> Keyword.put(:num_warmup, 0)

    base_seed = opts[:seed]

    chain_assignments =
      for {node_name, node_idx} <- Enum.with_index(nodes),
          chain_idx <- 0..(chains_per_node - 1) do
        global_id = node_idx * chains_per_node + chain_idx
        seed = base_seed + global_id * 7919
        {node_name, global_id, seed}
      end

    # Launch all chains in parallel
    tasks =
      Enum.map(chain_assignments, fn {target_node, global_id, seed} ->
        chain_opts = Keyword.put(sampler_opts, :seed, seed)

        Task.async(fn ->
          run_chain_on_node(ir, target_node, tuning, init_values, chain_opts, compile_opts, timeout, global_id)
        end)
      end)

    results = Task.await_many(tasks, :infinity)

    traces = Enum.map(results, fn {trace, _} -> trace end)
    stats = Enum.map(results, fn {_, stats} -> stats end)
    {traces, stats}
  end

  # Validate that remote nodes are reachable and have Exmc code loaded.
  # Logs warnings for unreachable nodes (fault recovery will handle them at dispatch time).
  defp validate_nodes!(nodes) do
    for n <- nodes, n != node() do
      try do
        case :erpc.call(n, Code, :ensure_loaded, [Exmc.Compiler], 10_000) do
          {:module, _} ->
            :ok

          {:error, reason} ->
            Logger.warning("Node #{inspect(n)} does not have Exmc loaded: #{inspect(reason)}")
        end
      catch
        _, reason ->
          Logger.warning("Node #{inspect(n)} is unreachable: #{inspect(reason)}")
      end
    end

    :ok
  end

  # Run full warmup on the coordinator node.
  # Returns {tuning, pm, ncp_info} where tuning has epsilon, inv_mass, chol_cov.
  defp run_coordinator_warmup(ir, init_values, opts, compile_opts) do
    compiled = Compiler.compile_for_sampling(ir, compile_opts)

    {_vag_fn, _step_fn, pm, ncp_info, _multi_step_fn} = compiled

    if pm.size == 0 do
      tuning = %{
        epsilon: 0.0,
        inv_mass: Nx.broadcast(Nx.tensor(0.0, type: :f64), {0}),
        chol_cov: nil
      }

      {tuning, pm, ncp_info}
    else
      # Run a full warmup sampling run (1 sample is enough — we just want the tuning)
      warmup_opts =
        opts
        |> Keyword.drop([:nodes, :chains_per_node, :init_values, :timeout])
        |> Keyword.put(:num_samples, 1)

      {_trace, stats} = Sampler.sample_compiled(compiled, init_values, warmup_opts)

      tuning = %{
        epsilon: stats.step_size,
        inv_mass: stats.inv_mass_diag,
        chol_cov: nil
      }

      {tuning, pm, ncp_info}
    end
  end

  # Run a single chain on a target node.
  # Falls back to coordinator on failure.
  defp run_chain_on_node(ir, target_node, tuning, init_values, opts, compile_opts, timeout, global_id) do
    if target_node == node() do
      run_chain_local(ir, tuning, init_values, opts, compile_opts)
    else
      try do
        :erpc.call(
          target_node,
          __MODULE__,
          :run_chain_remote,
          [ir, tuning, init_values, opts, compile_opts],
          timeout
        )
      catch
        kind, reason ->
          Logger.warning(
            "Chain #{global_id} on #{inspect(target_node)} failed " <>
              "(#{inspect(kind)}: #{inspect(reason)}), retrying on coordinator"
          )

          run_chain_local(ir, tuning, init_values, opts, compile_opts)
      end
    end
  end

  # Run a chain locally (coordinator or fallback).
  defp run_chain_local(ir, tuning, init_values, opts, compile_opts) do
    compiled = Compiler.compile_for_sampling(ir, compile_opts)
    Sampler.sample_compiled_tuned(compiled, tuning, init_values, opts)
  end

  @doc false
  # Called via :erpc on remote nodes. Must be public for remote calls.
  def run_chain_remote(ir, tuning, init_values, opts, compile_opts \\ []) do
    # Ensure inv_mass is on BinaryBackend (may have been serialized)
    tuning = ensure_binary_backend(tuning)
    compiled = Compiler.compile_for_sampling(ir, compile_opts)
    Sampler.sample_compiled_tuned(compiled, tuning, init_values, opts)
  end

  # Ensure tuning tensors are on BinaryBackend after deserialization.
  defp ensure_binary_backend(tuning) do
    inv_mass = Nx.backend_copy(tuning.inv_mass, Nx.BinaryBackend)

    chol_cov =
      if tuning.chol_cov do
        Nx.backend_copy(tuning.chol_cov, Nx.BinaryBackend)
      end

    %{tuning | inv_mass: inv_mass, chol_cov: chol_cov}
  end
end
