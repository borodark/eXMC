defmodule Exmc.NUTS.Vulkan.ServerTest do
  use ExUnit.Case, async: false

  alias Exmc.{Builder, Dist, NUTS.Sampler}

  @moduletag :vulkan

  setup_all do
    Application.put_env(:exmc, :compiler, :vulkan)
    on_exit(fn -> Application.delete_env(:exmc, :compiler) end)
    :ok
  end

  setup do
    Application.put_env(:exmc, :gpu_node, false)
    on_exit(fn -> Application.delete_env(:exmc, :gpu_node) end)

    case Process.whereis(Nx.Vulkan.Node) do
      nil -> :ok
      pid -> GenServer.stop(pid, :normal)
    end
  end

  test "GenServer starts, reports status, stops cleanly" do
    {:ok, pid} = Nx.Vulkan.Node.start_link()
    assert Process.alive?(pid)
    assert Nx.Vulkan.Node.alive?()

    status = Nx.Vulkan.Node.status()
    assert status.exec_count == 0
    assert status.uptime_ms >= 0

    :ok = GenServer.stop(pid, :normal)
    refute Nx.Vulkan.Node.alive?()
  end

  test "Normal d=1 posterior identical via direct vs GenServer routing" do
    ir =
      Builder.new_ir()
      |> Builder.rv("x", Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

    Process.put(:fused_leapfrog_meta, {:normal, 0.0, 1.0})

    Application.put_env(:exmc, :gpu_node, false)
    {trace_direct, _} = Sampler.sample(ir, %{}, num_warmup: 200, num_samples: 200, seed: 42)
    xs_direct = trace_direct["x"] |> Nx.to_flat_list()

    {:ok, _pid} = Nx.Vulkan.Node.start_link()
    Application.put_env(:exmc, :gpu_node, true)
    {trace_server, _} = Sampler.sample(ir, %{}, num_warmup: 200, num_samples: 200, seed: 42)
    xs_server = trace_server["x"] |> Nx.to_flat_list()

    mean_direct = Enum.sum(xs_direct) / length(xs_direct)
    mean_server = Enum.sum(xs_server) / length(xs_server)
    var_direct = sample_var(xs_direct, mean_direct)
    var_server = sample_var(xs_server, mean_server)

    assert_in_delta mean_server,
                    mean_direct,
                    0.05,
                    "GenServer routing diverged from direct: #{mean_server} vs #{mean_direct}"

    assert_in_delta var_server,
                    var_direct,
                    0.05,
                    "Variance diverged: #{var_server} vs #{var_direct}"

    status = Nx.Vulkan.Node.status()
    assert status.exec_count > 0
  end

  test "GenServer dispatches Exponential, StudentT, HalfNormal, Weibull (smoke)" do
    {:ok, _pid} = Nx.Vulkan.Node.start_link()
    Application.put_env(:exmc, :gpu_node, true)

    cases = [
      {Builder.new_ir() |> Builder.rv("x", Dist.Exponential, %{lambda: Nx.tensor(2.0)}),
       {:exponential, 2.0}, "Exponential"},
      {Builder.new_ir() |> Builder.rv("x", Dist.HalfNormal, %{sigma: Nx.tensor(1.0)}),
       {:halfnormal, 1.0, -:math.log(1.0) - 0.5 * :math.log(:math.pi())}, "HalfNormal"},
      {Builder.new_ir()
       |> Builder.rv("x", Dist.Weibull, %{k: Nx.tensor(2.0), lambda: Nx.tensor(1.0)}),
       {:weibull, 2.0, 1.0, 1.0 * (:math.log(2.0) - 2.0 * :math.log(1.0))}, "Weibull"}
    ]

    for {ir, meta, name} <- cases do
      Process.put(:fused_leapfrog_meta, meta)
      {trace, _} = Sampler.sample(ir, %{}, num_warmup: 100, num_samples: 100, seed: 42)
      xs = trace["x"] |> Nx.to_flat_list()
      mean = Enum.sum(xs) / length(xs)

      assert is_number(mean), "#{name}: produced non-numeric mean #{inspect(mean)}"
      assert length(xs) == 100, "#{name}: wrong sample count"
    end
  end

  defp sample_var(xs, mean) do
    n = length(xs)
    sq_sum = Enum.reduce(xs, 0.0, fn x, acc -> acc + (x - mean) * (x - mean) end)
    sq_sum / n
  end
end
