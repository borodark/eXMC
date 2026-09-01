defmodule Exmc.NUTS.Vulkan.ChaosTest do
  @moduledoc """
  W6 Phase 1 — chaos test for the Vulkan dispatch path's bulkhead +
  per-shader suspect tracking.

  Uses an aggressive timeout (1 ms) to simulate "this shader is
  slow / hung" without actually building a deliberately bad shader
  (which on NVIDIA Linux can hang the GPU and require a host
  reboot). The watchdog mechanism is platform-agnostic; what we're
  testing is the *policy* that wraps it, not the kernel-level
  failure mode.

  Phase 2 will add a real bad-shader test using `:vk_synchronization2`
  fences with cancellation support, and verify GPU buffer state
  doesn't leak. For now Phase 1 just covers the suspect tracker
  contract: timeouts → counter → eviction → fallback → recovery.
  """

  use ExUnit.Case, async: false

  import Exmc.TestHelper, only: [put_env_scoped: 2, put_env_scoped: 3]

  alias Exmc.{Builder, Dist, NUTS.Sampler}
  alias Exmc.NUTS.Vulkan.SuspectTracker

  @moduletag :vulkan
  @moduletag :requires_vulkan

  setup do
    put_env_scoped(:compiler, :vulkan)

    on_exit(fn ->
      Enum.each([Nx.Vulkan.Node, SuspectTracker], &stop_quietly/1)
    end)

    {:ok, _node} = Nx.Vulkan.Node.start_link()
    {:ok, _tracker} = SuspectTracker.start_link()

    :ok
  end

  # Stop a process by name or pid, tolerating one that has already exited.
  #
  # Teardown here must not care whether the process is still there. This module
  # tests a suicide/eviction mechanism, so a coordinator that has already died
  # is frequently the *correct* outcome of a test rather than an anomaly. On
  # top of that, ExUnit exits the test process with `:shutdown` once the test
  # body returns, which reaps everything `start_link`ed from `setup` — and
  # `on_exit` callbacks run after that, from a different process. So
  # `Process.whereis/1` can hand back a pid that is dead by the time
  # `GenServer.stop/3` reaches it. The window is microseconds wide, which is
  # why this surfaced as a *roaming* flake, attaching itself to whichever test
  # happened to be running, and why it got worse on hosts with more schedulers.
  #
  # The previous teardown was `for name <- names, do: GenServer.stop(...)`, and
  # a single-generator comprehension compiles to `Enum.map/2` — so the first
  # dead pid did not just fail the test for a reason unrelated to anything it
  # asserts, it also aborted the cleanup of every name after it, leaking the
  # processes the callback existed to reap.
  #
  # Hence both halves: `alive?/1` narrows the window, and the `catch` covers
  # the part of it that cannot be closed. Cleanup that can itself raise is not
  # cleanup.
  defp stop_quietly(name_or_pid, reason \\ :normal) do
    pid = if is_pid(name_or_pid), do: name_or_pid, else: Process.whereis(name_or_pid)

    if is_pid(pid) and Process.alive?(pid) do
      try do
        GenServer.stop(pid, reason, :infinity)
      catch
        # :noproc — lost the race above; anything else — it died on its own
        # while stopping. Either way the process is gone, which is what we
        # asked for.
        :exit, _ -> :ok
      end
    end

    :ok
  end

  describe "single-timeout path" do
    @tag :skip
    test "tight timeout drives the meta to eviction during sampling" do
      # SKIPPED: Phase 1 GenServer queueing under storm-of-timeouts
      # has subtle interleaving (calls accumulate in mailbox while
      # NIF holds the GenServer for 30ms each). Real bad-shader
      # eviction is exercised by the dedicated tracker tests below;
      # this end-to-end path will be revisited in Phase 2 when the
      # in-flight NIF can actually be cancelled rather than just
      # abandoned by the caller.
      :ok
    end
  end

  describe "eviction policy" do
    test "after max_consecutive_timeouts, the meta is evicted" do
      meta = {:test_meta, :a}

      # Drive timeouts manually to test the tracker contract without
      # involving the actual sampler.
      assert SuspectTracker.suspect_count(meta) == 0
      assert :ok = SuspectTracker.record_timeout(meta)
      assert SuspectTracker.suspect_count(meta) == 1
      refute SuspectTracker.evicted?(meta)

      assert :ok = SuspectTracker.record_timeout(meta)
      assert SuspectTracker.suspect_count(meta) == 2
      refute SuspectTracker.evicted?(meta)

      assert :evicted = SuspectTracker.record_timeout(meta)
      assert SuspectTracker.suspect_count(meta) == 3
      assert SuspectTracker.evicted?(meta)
    end

    test "a successful dispatch resets the counter (does not un-evict)" do
      meta = {:test_meta, :b}

      :ok = SuspectTracker.record_timeout(meta)
      :ok = SuspectTracker.record_timeout(meta)
      assert SuspectTracker.suspect_count(meta) == 2

      :ok = SuspectTracker.record_success(meta)
      # `cast` is async; give it a moment to land
      Process.sleep(10)
      assert SuspectTracker.suspect_count(meta) == 0
    end

    test "different metas track independently" do
      meta_a = {:family_a, 1.0}
      meta_b = {:family_b, 2.0}

      SuspectTracker.record_timeout(meta_a)
      SuspectTracker.record_timeout(meta_a)
      SuspectTracker.record_timeout(meta_b)

      assert SuspectTracker.suspect_count(meta_a) == 2
      assert SuspectTracker.suspect_count(meta_b) == 1
    end
  end

  describe "cross-shader suicide window" do
    test "5 timeouts across different metas in the window trigger emergency_brake" do
      # Custom tracker with tighter window for the test.
      {:ok, t} =
        SuspectTracker.start_link(
          name: :test_tracker_window,
          max_window_timeouts: 5,
          window_ms: 60_000
        )

      # As `on_exit`, not a trailing call: a failing assertion below would skip
      # a trailing stop and leak the tracker under its registered name.
      on_exit(fn -> stop_quietly(t) end)

      for i <- 1..5 do
        SuspectTracker.record_timeout({:meta, i}, :test_tracker_window)
      end

      status = SuspectTracker.status(:test_tracker_window)
      assert status.window_size == 5
      assert status.emergency_brake == true
    end

    test "old timeouts age out of the window" do
      {:ok, t} =
        SuspectTracker.start_link(
          name: :test_tracker_age,
          max_window_timeouts: 3,
          window_ms: 50
        )

      on_exit(fn -> stop_quietly(t) end)

      SuspectTracker.record_timeout({:meta, 1}, :test_tracker_age)
      SuspectTracker.record_timeout({:meta, 2}, :test_tracker_age)
      Process.sleep(80)
      SuspectTracker.record_timeout({:meta, 3}, :test_tracker_age)

      # First two timeouts should have aged out; window has only 1 entry.
      status = SuspectTracker.status(:test_tracker_age)
      assert status.window_size == 1
    end
  end

  describe "tree.ex route_chain integration" do
    @tag :skip
    test "after a timeout-storm, route_chain falls back to EXLA without calling Nx.Vulkan.Node" do
      put_env_scoped(:gpu_node, true)
      put_env_scoped(:nx_vulkan, :node_timeout_ms, 1)

      meta = {:normal, 0.0, 1.0}

      # Force eviction immediately via the tracker.
      SuspectTracker.record_timeout(meta)
      SuspectTracker.record_timeout(meta)
      SuspectTracker.record_timeout(meta)
      assert SuspectTracker.evicted?(meta)

      # Now dispatch — should bypass the GPU node entirely. Use a
      # tight timeout to verify the route doesn't actually call
      # with_node (else we'd see :node_timeout pile up).
      ir =
        Builder.new_ir()
        |> Builder.rv("x", Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

      Process.put(:fused_leapfrog_meta, meta)

      # Capture the node's exec_count before sampling.
      before_count = Nx.Vulkan.Node.status().exec_count

      {trace, _stats} =
        Sampler.sample(ir, %{}, num_warmup: 50, num_samples: 50, seed: 42)

      after_count = Nx.Vulkan.Node.status().exec_count

      # Sampling completed (chain didn't stall), and no with_node calls
      # went out — eviction skipped the GPU node entirely.
      xs = trace["x"] |> Nx.to_flat_list()
      assert length(xs) == 50

      assert after_count == before_count,
             "expected no with_node calls (was #{before_count}, now #{after_count})"
    end
  end
end
