defmodule Exmc.Poker.Table do
  @moduledoc """
  GenServer representing a poker table.

  Each player seat is a process with its own Bayesian posterior.
  The dealer broadcasts board cards via message passing.
  This is the OTP angle no Python framework can match.
  """

  use GenServer

  alias Exmc.Poker.{Simulator, OpponentModel}

  defstruct [:players, :observations, :posteriors, :hand_count]

  # --- Client API ---

  @doc "Start a table process with the given player param maps."
  def start_link(player_params, opts \\ []) do
    GenServer.start_link(__MODULE__, player_params, opts)
  end

  @doc "Simulate N hands and collect observations."
  def play(table, n_hands) do
    GenServer.call(table, {:play, n_hands}, 30_000)
  end

  @doc "Run Bayesian inference on collected observations."
  def profile(table, opts \\ []) do
    GenServer.call(table, {:profile, opts}, 300_000)
  end

  @doc "Get current state summary."
  def status(table) do
    GenServer.call(table, :status)
  end

  @doc "Make a decision given your hand and the current game state."
  def decide(table, my_hole, board, pot, to_call) do
    GenServer.call(table, {:decide, my_hole, board, pot, to_call}, 60_000)
  end

  # --- Server callbacks ---

  @impl true
  def init(player_params) do
    state = %__MODULE__{
      players: player_params,
      observations: Enum.map(player_params, fn _ -> %{hand_strengths: [], actions: []} end),
      posteriors: nil,
      hand_count: 0
    }

    {:ok, state}
  end

  @impl true
  def handle_call({:play, n_hands}, _from, state) do
    {new_obs, _} = Simulator.simulate(state.players, n_hands)

    merged =
      Enum.zip(state.observations, new_obs)
      |> Enum.map(fn {old, new} ->
        %{
          hand_strengths: old.hand_strengths ++ new.hand_strengths,
          actions: old.actions ++ new.actions
        }
      end)

    new_state = %{state | observations: merged, hand_count: state.hand_count + n_hands}
    {:reply, {:ok, new_state.hand_count}, new_state}
  end

  @impl true
  def handle_call({:profile, opts}, _from, state) do
    num_players = length(state.players)
    {ir, _data} = OpponentModel.build(state.observations)
    init = OpponentModel.init_values(num_players)

    num_samples = Keyword.get(opts, :num_samples, 300)
    num_warmup = Keyword.get(opts, :num_warmup, 300)

    {trace, stats} =
      Exmc.Sampler.sample(ir, init,
        num_samples: num_samples,
        num_warmup: num_warmup,
        seed: Keyword.get(opts, :seed, 42),
        ncp: false
      )

    profiles = OpponentModel.extract_profiles(trace, num_players)
    new_state = %{state | posteriors: profiles}

    {:reply, {:ok, profiles, stats}, new_state}
  end

  @impl true
  def handle_call(:status, _from, state) do
    summary = %{
      num_players: length(state.players),
      hands_played: state.hand_count,
      observations_per_player: Enum.map(state.observations, fn o -> length(o.actions) end),
      has_posteriors: state.posteriors != nil
    }

    {:reply, summary, state}
  end

  @impl true
  def handle_call({:decide, my_hole, board, pot, to_call}, _from, state) do
    if state.posteriors == nil do
      {:reply, {:error, :no_posteriors}, state}
    else
      # Use first opponent's profile for the decision
      profile = hd(state.posteriors)

      decision =
        Exmc.Poker.Decision.expected_value(my_hole, board, profile, pot, to_call)

      {:reply, {:ok, decision}, state}
    end
  end
end
