defmodule Exmc.NUTS.Vulkan.SuspectTracker do
  @moduledoc """
  W6 Phase 1 — per-shader suspect tracking for the Vulkan dispatch
  path. Records consecutive timeouts per meta tag, and a sliding
  cross-shader window for "the driver is probably hosed" detection.

  ## Policy

  - **Per-shader eviction**: after `:max_consecutive_timeouts` (default 3)
    consecutive timeouts on the same meta tag, the shader is marked
    `:evicted`. `evicted?/1` returns true; `tree.ex:route_chain`
    skips the GPU node and falls back to EXLA without even calling
    `with_node`.

  - **Cross-shader suicide window**: if `:max_window_timeouts` (default 5)
    timeouts occur across any shaders within `:window_ms` (default 60000),
    the driver is probably hung; the tracker logs an `:emergency_brake`
    event and Future Work would suicide the `Nx.Vulkan.Node` so the
    supervisor restarts it. (Phase 1 stops at logging — actual
    Process.exit on the node is Phase 2 work, gated on knowing the
    node has a supervisor.)

  ## State

      %{
        # meta → consecutive timeout count
        suspects: %{},
        # meta → :evicted | :ok
        eviction: %{},
        # MapSet of meta tags currently in :evicted state
        evicted: MapSet.new(),
        # [{ts_ms, meta}, ...] timestamps for the cross-shader window
        window: [],
        max_consecutive_timeouts: 3,
        max_window_timeouts: 5,
        window_ms: 60_000
      }
  """

  use GenServer

  @name __MODULE__

  ## API

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: opts[:name] || @name)
  end

  @doc "Whether the named tracker is alive."
  def alive?(name \\ @name) do
    case Process.whereis(name) do
      nil -> false
      pid -> Process.alive?(pid)
    end
  end

  @doc """
  Record a timeout for the given meta tag. Returns `:ok | :evicted`
  where `:evicted` means this call crossed the eviction threshold
  (the meta is now blacklisted for future dispatches until reset).
  """
  def record_timeout(meta, name \\ @name),
    do: GenServer.call(name, {:record_timeout, meta})

  @doc """
  Record a successful dispatch. Resets the consecutive-timeout
  counter for this meta but doesn't un-evict (eviction is sticky
  until explicit `reset/0` or restart).
  """
  def record_success(meta, name \\ @name),
    do: GenServer.cast(name, {:record_success, meta})

  @doc """
  True when this meta has crossed the consecutive-timeout
  threshold. Callers should skip the GPU dispatch and go straight
  to the EXLA fallback when this returns true.
  """
  def evicted?(meta, name \\ @name) do
    case Process.whereis(name) do
      nil -> false
      _pid -> GenServer.call(name, {:evicted?, meta})
    end
  end

  @doc "Per-shader consecutive-timeout counter."
  def suspect_count(meta, name \\ @name),
    do: GenServer.call(name, {:suspect_count, meta})

  @doc "Read-only snapshot of all tracker state."
  def status(name \\ @name), do: GenServer.call(name, :status)

  @doc "Clear all suspect counters + eviction state. For tests."
  def reset(name \\ @name), do: GenServer.call(name, :reset)

  ## GenServer callbacks

  @impl true
  def init(opts) do
    {:ok,
     %{
       suspects: %{},
       evicted: MapSet.new(),
       window: [],
       max_consecutive_timeouts: Keyword.get(opts, :max_consecutive_timeouts, 3),
       max_window_timeouts: Keyword.get(opts, :max_window_timeouts, 5),
       window_ms: Keyword.get(opts, :window_ms, 60_000),
       emergency_brake: false
     }}
  end

  @impl true
  def handle_call({:record_timeout, meta}, _from, state) do
    n = Map.get(state.suspects, meta, 0) + 1
    suspects = Map.put(state.suspects, meta, n)

    evicted? = n >= state.max_consecutive_timeouts

    evicted =
      if evicted?,
        do: MapSet.put(state.evicted, meta),
        else: state.evicted

    now = System.monotonic_time(:millisecond)
    window = prune_window([{now, meta} | state.window], now, state.window_ms)
    emergency_brake = length(window) >= state.max_window_timeouts

    if emergency_brake and not state.emergency_brake do
      require Logger

      Logger.warning(
        "Exmc.NUTS.Vulkan.SuspectTracker: emergency brake — #{length(window)} timeouts in #{state.window_ms} ms across #{window |> Enum.map(&elem(&1, 1)) |> Enum.uniq() |> length()} shader(s)"
      )
    end

    new_state = %{
      state
      | suspects: suspects,
        evicted: evicted,
        window: window,
        emergency_brake: emergency_brake
    }

    reply = if evicted?, do: :evicted, else: :ok
    {:reply, reply, new_state}
  end

  @impl true
  def handle_call({:evicted?, meta}, _from, state) do
    {:reply, MapSet.member?(state.evicted, meta), state}
  end

  @impl true
  def handle_call({:suspect_count, meta}, _from, state) do
    {:reply, Map.get(state.suspects, meta, 0), state}
  end

  @impl true
  def handle_call(:status, _from, state) do
    {:reply,
     %{
       suspect_counts: state.suspects,
       evicted: MapSet.to_list(state.evicted),
       window_size: length(state.window),
       emergency_brake: state.emergency_brake
     }, state}
  end

  @impl true
  def handle_call(:reset, _from, state) do
    {:reply, :ok,
     %{state | suspects: %{}, evicted: MapSet.new(), window: [], emergency_brake: false}}
  end

  @impl true
  def handle_cast({:record_success, meta}, state) do
    {:noreply, %{state | suspects: Map.delete(state.suspects, meta)}}
  end

  defp prune_window(window, now, window_ms) do
    cutoff = now - window_ms
    Enum.filter(window, fn {ts, _meta} -> ts >= cutoff end)
  end
end
