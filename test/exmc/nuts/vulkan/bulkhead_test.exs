defmodule Exmc.NUTS.Vulkan.BulkheadTest do
  @moduledoc """
  W6 — verify the GPU node timeout + EXLA fallback path.
  """

  use ExUnit.Case, async: false

  alias Exmc.{Builder, Dist, NUTS.Sampler}

  @moduletag :vulkan

  setup do
    Application.put_env(:exmc, :compiler, :vulkan)
    on_exit(fn -> Application.delete_env(:exmc, :compiler) end)

    case Process.whereis(Nx.Vulkan.Node) do
      nil -> :ok
      pid -> GenServer.stop(pid, :normal)
    end

    Application.delete_env(:exmc, :gpu_node)
    Application.delete_env(:exmc, :gpu_node_timeout_ms)
    :ok
  end

  test "timeout returns {:error, :node_timeout}" do
    case Process.whereis(Nx.Vulkan.Node) do
      nil -> {:ok, _pid} = Nx.Vulkan.Node.start_link()
      _ -> :ok
    end

    Application.put_env(:nx_vulkan, :node_timeout_ms, 1)

    result =
      Nx.Vulkan.Node.with_node(fn ->
        Process.sleep(100)
        :should_not_reach
      end)

    assert result == {:error, :node_timeout}
  after
    Application.delete_env(:nx_vulkan, :node_timeout_ms)
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

    Application.put_env(:exmc, :gpu_node, true)
    Application.put_env(:exmc, :gpu_node_timeout_ms, 1)

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
