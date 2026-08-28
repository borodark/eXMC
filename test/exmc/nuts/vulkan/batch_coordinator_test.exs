defmodule Exmc.NUTS.Vulkan.BatchCoordinatorTest do
  use ExUnit.Case, async: true

  alias Exmc.NUTS.Vulkan.BatchCoordinator

  @synth_meta {:synthesised, "abc", [], %{eps: 0.1, n_obs: 4}, "/tmp/fake.spv", <<>>}

  # The early-guard tests above never reach Dispatch, so a push_spec stub is
  # enough for them. The live-coord test below does reach it, and needs
  # `:priors` — without that key chain_batch raises KeyError while walking the
  # prior floats, long before any dispatch. That is what this test used to
  # exercise while claiming to exercise a dead SPV path.
  @dispatchable_meta {:synthesised, "abc", [],
                      %{
                        eps: 0.1,
                        n_obs: 4,
                        priors: [{"x", Exmc.Dist.Normal, %{mu: 0.0, sigma: 1.0}}]
                      }, "/tmp/fake.spv", <<>>}

  describe "request_synth_chain/9 — early guards (no coord round-trip)" do
    test "non-synthesised meta returns {:fallback, :unsupported_meta_type}" do
      # Per-family chain shaders (:normal, :exponential, ...) don't
      # have a batched variant. Route_chain falls through to the
      # single-instance direct path.
      assert {:fallback, :unsupported_meta_type} =
               BatchCoordinator.request_synth_chain(
                 self(),
                 {:normal, 0.0, 1.0},
                 nil,
                 nil,
                 nil,
                 nil,
                 0.05,
                 16,
                 1
               )
    end

    test "non-pid coord arg returns {:fallback, :not_a_pid}" do
      assert {:fallback, :not_a_pid} =
               BatchCoordinator.request_synth_chain(
                 :not_a_pid,
                 @synth_meta,
                 nil,
                 nil,
                 nil,
                 nil,
                 0.05,
                 16,
                 1
               )
    end

    test "dead coord pid returns {:fallback, {:coord_exit, _}}" do
      # The Task #171 hook in Tree.route_chain must NOT crash the
      # caller when the coordinator is gone — it must surface
      # :fallback so the caller routes through the single-instance
      # direct path. Spawn a process that exits immediately, then
      # call request_synth_chain against its pid.
      dead = spawn(fn -> :ok end)
      :timer.sleep(10)

      assert {:fallback, {:coord_exit, _reason}} =
               BatchCoordinator.request_synth_chain(
                 dead,
                 @synth_meta,
                 nil,
                 nil,
                 nil,
                 nil,
                 0.05,
                 16,
                 1
               )
    end
  end

  describe "live coord — partition + flush dispatching" do
    setup do
      # batched_meta passed to init isn't used by the chain dispatch
      # path (per-request meta wins) but the existing multi_step
      # init contract requires something here.
      {:ok, coord} = BatchCoordinator.start_link(@synth_meta, 4, flush_ms: 50)
      on_exit(fn -> if Process.alive?(coord), do: GenServer.stop(coord) end)
      %{coord: coord}
    end

    test "single chain request to a dead-Dispatch path surfaces fallback", %{coord: coord} do
      # Dispatch.chain_batch raises because nx_vulkan exports no f64 batched
      # chain NIF (see Dispatch.ensure_batch_nif!/0). The coordinator's
      # try/rescue must surface :fallback rather than crash the coord.
      #
      # The reason is asserted, not just its shape. This test previously
      # attributed the raise to the bogus "/tmp/fake.spv" path, and would
      # have passed just as green with a perfectly valid SPV — it never got
      # near the file. If the f64 batch NIF ever lands, this assertion is
      # what fails and says to revisit the test rather than the test quietly
      # continuing to pass for a fourth different reason.
      q = Nx.tensor([0.0, 0.0], type: :f32, backend: Nx.BinaryBackend)
      p = Nx.tensor([0.1, -0.1], type: :f32, backend: Nx.BinaryBackend)
      im = Nx.tensor([1.0, 1.0], type: :f32, backend: Nx.BinaryBackend)
      obs = Nx.tensor([0.0, 0.0, 0.0, 0.0], type: :f32, backend: Nx.BinaryBackend)

      result =
        BatchCoordinator.request_synth_chain(
          coord,
          @dispatchable_meta,
          q,
          p,
          im,
          obs,
          0.05,
          4,
          1
        )

      assert {:fallback, {:dispatch_raise, msg}} = result
      assert msg =~ "leapfrog_chain_synth_batch_f64/6"
      # Coord must still be alive.
      assert Process.alive?(coord)
    end
  end
end
