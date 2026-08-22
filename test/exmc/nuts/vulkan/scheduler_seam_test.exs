defmodule Exmc.NUTS.Vulkan.SchedulerSeamTest do
  @moduledoc """
  The GPU compute scheduler is configurable, and this pins that it is.

  `BatchCoordinator` used to name its scheduler with an `alias`, and that one
  line was the entire difference between this file and the applications tree's
  copy of it — core code naming an application module is what makes a file
  forkable. The seam replaces it. A seam nobody tests is a seam that quietly
  stops working, so both directions are asserted here: the default when no
  application configures one, and the override when one does.
  """

  use ExUnit.Case, async: false

  alias Exmc.NUTS.Vulkan.BatchCoordinator

  defmodule FakeScheduler do
    @moduledoc false
    def run(fun) when is_function(fun, 1), do: {:fake, fun.(:no_device)}
    def run(fun) when is_function(fun, 0), do: {:fake, fun.()}
  end

  setup do
    prev = Application.get_env(:exmc, :gpu_scheduler)

    on_exit(fn ->
      case prev do
        nil -> Application.delete_env(:exmc, :gpu_scheduler)
        mod -> Application.put_env(:exmc, :gpu_scheduler, mod)
      end
    end)

    :ok
  end

  test "defaults to the core scheduler when the application configures none" do
    Application.delete_env(:exmc, :gpu_scheduler)
    assert BatchCoordinator.scheduler() == Exmc.NUTS.Vulkan.Scheduler
  end

  test "an application can supply its own without editing the core" do
    Application.put_env(:exmc, :gpu_scheduler, FakeScheduler)
    assert BatchCoordinator.scheduler() == FakeScheduler

    # And the contract the coordinator actually uses is `run/1` over a
    # one-arity function — the shape both implementations must honour.
    assert {:fake, :dispatched} =
             BatchCoordinator.scheduler().run(fn _device -> :dispatched end)
  end
end
