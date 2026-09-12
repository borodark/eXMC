defmodule Exmc.NUTS.Vulkan.BulkheadTest do
  @moduledoc """
  W6 — verify the GPU node timeout + EXLA fallback path.
  """

  use ExUnit.Case, async: false

  import Exmc.TestHelper,
    only: [put_env_scoped: 2, put_env_scoped: 3, delete_env_scoped: 1]

  alias Exmc.{Builder, Dist, NUTS.Sampler}

  # Needs a Vulkan device: starts `Nx.Vulkan.Node` and names the compiler.
  # `test_helper.exs` excludes `:requires_vulkan` on hosts without one.
  @moduletag :requires_vulkan

  setup do
    put_env_scoped(:compiler, :vulkan)

    case Process.whereis(Nx.Vulkan.Node) do
      nil -> :ok
      pid -> GenServer.stop(pid, :normal)
    end

    delete_env_scoped(:gpu_node)
    delete_env_scoped(:gpu_node_timeout_ms)
    :ok
  end

  test "timeout returns {:error, :node_timeout}" do
    case Process.whereis(Nx.Vulkan.Node) do
      nil -> {:ok, _pid} = Nx.Vulkan.Node.start_link()
      _ -> :ok
    end

    put_env_scoped(:nx_vulkan, :node_timeout_ms, 1)

    result =
      Nx.Vulkan.Node.with_node(fn ->
        Process.sleep(100)
        :should_not_reach
      end)

    assert result == {:error, :node_timeout}
  end

  test "node dead returns {:error, :node_dead}" do
    case Process.whereis(Nx.Vulkan.Node) do
      nil -> :ok
      pid -> GenServer.stop(pid, :normal)
    end

    result = Nx.Vulkan.Node.with_node(fn -> :unreached end)
    assert result == {:error, :node_dead}
  end

  test "tree.ex falls back to EXLA path when GPU node times out" do
    case Process.whereis(Nx.Vulkan.Node) do
      nil -> {:ok, _pid} = Nx.Vulkan.Node.start_link()
      _ -> :ok
    end

    put_env_scoped(:gpu_node, true)
    put_env_scoped(:gpu_node_timeout_ms, 1)

    ir =
      Builder.new_ir()
      |> Builder.rv("x", Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

    Process.put(:fused_leapfrog_meta, {:normal, 0.0, 1.0})

    {trace, _stats} =
      Sampler.sample(ir, %{}, num_warmup: 100, num_samples: 100, seed: 42)

    xs = trace["x"] |> Nx.to_flat_list()
    mean = Enum.sum(xs) / length(xs)
    assert is_number(mean)

    assert abs(mean) < 0.5,
           "Posterior mean #{mean} drifted too far — fallback may not be working"
  end
end
