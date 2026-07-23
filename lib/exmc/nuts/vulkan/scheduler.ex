defmodule Exmc.NUTS.Vulkan.Scheduler do
  @moduledoc """
  GPU concurrency limiter for parallel NUTS chains.

  Prevents GPU OOM by limiting how many chains can run NUTS sampling on the
  GPU simultaneously. Uses a permit-based semaphore with automatic crash
  recovery.

  This is optional infrastructure. When the scheduler process is **not**
  running, `run/2` executes the work function directly (no concurrency
  limit) — so single-chain and Livebook usage needs no supervision tree.
  To bound concurrency, start it under your own supervisor:

      children = [Exmc.NUTS.Vulkan.Scheduler]
      Supervisor.start_link(children, strategy: :one_for_one)

  ## Multi-device support

  When multiple GPUs are detected, maintains per-device permit pools.
  `acquire/1` accepts a `device` option (`:any` picks least-loaded), and a
  CPU overflow pool absorbs work when every GPU is saturated.

  ## Usage

      Scheduler.run(fn device ->
        sample_device = if is_integer(device), do: :cuda, else: :host
        Exmc.NUTS.Sampler.sample(ir, init, device: sample_device)
      end)

  Or manual acquire/release:

      {:ok, device} = Scheduler.acquire()
      try do
        Exmc.NUTS.Sampler.sample(ir, init, opts)
      after
        Scheduler.release()
      end

  ## Tuning

  Per-GPU and CPU-pool sizes come from `MAX_GPU_CONCURRENT` /
  `MAX_CPU_CONCURRENT` env vars, or the `:cpu_slots` / `:gpu_only` start
  options. Free VRAM is probed via `nvidia-smi` when present.
  """

  use GenServer

  require Logger

  defmodule DevicePool do
    @moduledoc false
    defstruct [
      :max,
      active: 0,
      queue: :queue.new()
    ]
  end

  defstruct devices: %{},
            monitors: %{},
            total_acquired: 0,
            total_queued: 0

  @max_per_gpu 3
  @max_cpu_slots 2

  # --- Client API ---

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Acquire a compute permit. Blocks until one is available. Returns `{:ok, device_id}`."
  def acquire(opts \\ []) do
    device = Keyword.get(opts, :device, :any)
    GenServer.call(__MODULE__, {:acquire, device}, :infinity)
  end

  @doc "Release a GPU permit."
  def release do
    GenServer.cast(__MODULE__, {:release, self()})
  end

  @doc """
  Run a function with a compute permit. Handles acquire/release + crash safety.

  Accepts arity-0 or arity-1 functions. Arity-1 receives the device_id
  (integer for GPU, `:cpu` for the CPU pool) so the caller can route work.

  When the scheduler process is not running, the function is executed
  directly (with device `:none` for arity-1 callers) — no concurrency
  limiting, but no supervision tree required.

      Scheduler.run(fn device ->
        sample_device = if is_integer(device), do: :cuda, else: :host
        Exmc.NUTS.Sampler.sample(ir, init, device: sample_device)
      end)
  """
  def run(fun, opts \\ []) when is_function(fun, 0) or is_function(fun, 1) do
    if Process.whereis(__MODULE__) do
      {:ok, device} = acquire(opts)

      try do
        apply_fun(fun, device)
      after
        release()
      end
    else
      # No scheduler running — execute directly with no permit.
      apply_fun(fun, :none)
    end
  end

  defp apply_fun(fun, device) do
    case Function.info(fun, :arity) do
      {:arity, 1} -> fun.(device)
      {:arity, 0} -> fun.()
    end
  end

  @doc "Get scheduler status."
  def status do
    GenServer.call(__MODULE__, :status)
  end

  # --- Server ---

  @impl true
  def init(opts) do
    device_configs = Keyword.get_lazy(opts, :devices, &detect_devices/0)

    # Apply option overrides
    device_configs =
      device_configs
      |> maybe_override_cpu_slots(Keyword.get(opts, :cpu_slots))
      |> maybe_strip_cpu(Keyword.get(opts, :gpu_only, false))

    devices =
      Map.new(device_configs, fn {dev_id, max} ->
        {dev_id, %DevicePool{max: max}}
      end)

    total_max = devices |> Map.values() |> Enum.map(& &1.max) |> Enum.sum()
    n_devices = map_size(devices)
    Logger.info("[Vulkan.Scheduler] #{n_devices} device(s), total_max_concurrent=#{total_max}")

    for {dev_id, pool} <- devices do
      Logger.info("[Vulkan.Scheduler]   device #{dev_id}: max=#{pool.max}")
    end

    {:ok, %__MODULE__{devices: devices}}
  end

  @impl true
  def handle_call({:acquire, device_pref}, {pid, _} = from, state) do
    dev_id = pick_device(state.devices, device_pref)

    case dev_id do
      nil ->
        # All devices full — queue on least-loaded device
        target = least_loaded_device(state.devices)
        pool = state.devices[target]
        queue = :queue.in({from, pid}, pool.queue)
        devices = Map.put(state.devices, target, %{pool | queue: queue})
        {:noreply, %{state | devices: devices, total_queued: state.total_queued + 1}}

      dev ->
        ref = Process.monitor(pid)
        monitors = Map.put(state.monitors, pid, {ref, dev})
        pool = state.devices[dev]
        devices = Map.put(state.devices, dev, %{pool | active: pool.active + 1})

        {:reply, {:ok, dev},
         %{state | devices: devices, monitors: monitors, total_acquired: state.total_acquired + 1}}
    end
  end

  def handle_call(:status, _from, state) do
    per_device =
      Map.new(state.devices, fn {dev_id, pool} ->
        {dev_id, %{max: pool.max, active: pool.active, queued: :queue.len(pool.queue)}}
      end)

    total_max = state.devices |> Map.values() |> Enum.map(& &1.max) |> Enum.sum()
    total_active = state.devices |> Map.values() |> Enum.map(& &1.active) |> Enum.sum()
    total_queued = state.devices |> Map.values() |> Enum.map(&:queue.len(&1.queue)) |> Enum.sum()

    info = %{
      # Backward-compatible fields
      max_concurrent: total_max,
      active: total_active,
      queued: total_queued,
      total_acquired: state.total_acquired,
      total_queued: state.total_queued,
      # Multi-device detail
      devices: per_device
    }

    {:reply, info, state}
  end

  @impl true
  def handle_cast({:release, pid}, state) do
    {:noreply, do_release(state, pid)}
  end

  @impl true
  def handle_info({:DOWN, _ref, :process, pid, _reason}, state) do
    {:noreply, do_release(state, pid)}
  end

  # --- Internal ---

  defp pick_device(devices, :any) do
    # Pick device with most available capacity
    devices
    |> Enum.filter(fn {_dev, pool} -> pool.active < pool.max end)
    |> Enum.min_by(fn {_dev, pool} -> pool.active end, fn -> nil end)
    |> case do
      nil -> nil
      {dev, _pool} -> dev
    end
  end

  defp pick_device(devices, device_id) do
    case Map.get(devices, device_id) do
      %{active: active, max: max} when active < max -> device_id
      _ -> nil
    end
  end

  defp least_loaded_device(devices) do
    {dev, _} = Enum.min_by(devices, fn {_dev, pool} -> pool.active / max(pool.max, 1) end)
    dev
  end

  defp do_release(state, pid) do
    case Map.pop(state.monitors, pid) do
      {nil, _monitors} ->
        state

      {{ref, dev_id}, monitors} ->
        Process.demonitor(ref, [:flush])
        pool = state.devices[dev_id]
        pool = %{pool | active: pool.active - 1}
        devices = Map.put(state.devices, dev_id, pool)
        state = %{state | devices: devices, monitors: monitors}
        grant_next(state, dev_id)
    end
  end

  defp grant_next(state, dev_id) do
    pool = state.devices[dev_id]

    case :queue.out(pool.queue) do
      {:empty, _} ->
        # Work-stealing: check other devices' queues
        steal_from_other(state, dev_id)

      {{:value, {from, pid}}, queue} ->
        pool = %{pool | queue: queue}

        if Process.alive?(pid) do
          ref = Process.monitor(pid)
          monitors = Map.put(state.monitors, pid, {ref, dev_id})
          GenServer.reply(from, {:ok, dev_id})

          devices = Map.put(state.devices, dev_id, %{pool | active: pool.active + 1})

          %{state | devices: devices, monitors: monitors, total_acquired: state.total_acquired + 1}
        else
          state = %{state | devices: Map.put(state.devices, dev_id, pool)}
          grant_next(state, dev_id)
        end
    end
  end

  defp detect_devices do
    gpu_memories = detect_gpu_memories_mb()

    env_gpu = parse_env_int("MAX_GPU_CONCURRENT")
    env_cpu = parse_env_int("MAX_CPU_CONCURRENT")

    gpu_devices =
      if gpu_memories == [] do
        []
      else
        Enum.with_index(gpu_memories, fn mb, idx ->
          max_jobs = env_gpu || @max_per_gpu
          Logger.info("[Vulkan.Scheduler] GPU #{idx}: #{mb}MB free, #{max_jobs} concurrent")
          {idx, max_jobs}
        end)
      end

    # CPU overflow pool — the JIT backend serializes compilation, so high
    # concurrency wastes permits.
    cpu_slots = env_cpu || @max_cpu_slots

    if gpu_devices == [] do
      # No GPU — CPU-only mode
      cpus = env_cpu || max(div(System.schedulers_online(), 8), 2)
      Logger.info("[Vulkan.Scheduler] No GPU detected, CPU-only mode: #{cpus} slots")
      [{:cpu, cpus}]
    else
      Logger.info("[Vulkan.Scheduler] CPU overflow pool: #{cpu_slots} slots")
      gpu_devices ++ [{:cpu, cpu_slots}]
    end
  end

  defp steal_from_other(state, dev_id) do
    # Find the device with the longest queue (excluding ourselves)
    victim =
      state.devices
      |> Enum.reject(fn {id, _} -> id == dev_id end)
      |> Enum.filter(fn {_, pool} -> :queue.len(pool.queue) > 0 end)
      |> Enum.max_by(fn {_, pool} -> :queue.len(pool.queue) end, fn -> nil end)

    case victim do
      nil ->
        state

      {victim_id, victim_pool} ->
        {{:value, {from, pid}}, queue} = :queue.out(victim_pool.queue)
        victim_pool = %{victim_pool | queue: queue}
        devices = Map.put(state.devices, victim_id, victim_pool)
        state = %{state | devices: devices}

        if Process.alive?(pid) do
          ref = Process.monitor(pid)
          monitors = Map.put(state.monitors, pid, {ref, dev_id})
          GenServer.reply(from, {:ok, dev_id})
          pool = state.devices[dev_id]
          devices = Map.put(state.devices, dev_id, %{pool | active: pool.active + 1})

          %{state | devices: devices, monitors: monitors, total_acquired: state.total_acquired + 1}
        else
          # Dead process — try stealing another
          steal_from_other(state, dev_id)
        end
    end
  end

  defp maybe_override_cpu_slots(configs, nil), do: configs

  defp maybe_override_cpu_slots(configs, slots) when is_integer(slots) and slots > 0 do
    Enum.map(configs, fn
      {:cpu, _} -> {:cpu, slots}
      other -> other
    end)
  end

  defp maybe_strip_cpu(configs, false), do: configs

  defp maybe_strip_cpu(configs, true) do
    stripped = Enum.reject(configs, fn {dev, _} -> dev == :cpu end)
    # Don't strip if CPU is the only pool
    if stripped == [], do: configs, else: stripped
  end

  defp detect_gpu_memories_mb do
    case System.cmd("nvidia-smi", ["--query-gpu=memory.free", "--format=csv,noheader,nounits"]) do
      {output, 0} ->
        output
        |> String.trim()
        |> String.split("\n")
        |> Enum.map(&(&1 |> String.trim() |> String.to_integer()))

      _ ->
        []
    end
  rescue
    _ -> []
  end

  defp parse_env_int(name) do
    case System.get_env(name) do
      nil -> nil
      val -> String.to_integer(val)
    end
  rescue
    _ -> nil
  end
end
