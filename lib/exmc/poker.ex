defmodule Exmc.Poker do
  @moduledoc """
  Bayesian Poker Strategy Simulator.

  Hierarchical opponent modeling via NUTS — the first application of
  Hamiltonian Monte Carlo to real-time poker opponent profiling.

  ## Quick start

      # Simulate a table with 3 archetypal players
      {obs, true_params} = Exmc.Poker.simulate()

      # Run Bayesian inference to profile opponents
      {profiles, stats} = Exmc.Poker.profile(obs)

      # Make a decision with your hand
      decision = Exmc.Poker.decide(
        my_hole: ~w(Ah Kh),
        board: ~w(Td Jd 2c),
        opponent: Enum.at(profiles, 0),
        pot: 100,
        to_call: 30
      )

      IO.puts(Exmc.Poker.Decision.format_decision(decision))

  ## OTP mode (each player is a process)

      {:ok, table} = Exmc.Poker.Table.start_link(Exmc.Poker.Simulator.archetypes())
      Exmc.Poker.Table.play(table, 50)
      {:ok, profiles, _stats} = Exmc.Poker.Table.profile(table)

  *Probabiliers de tous les a priori, unissez-vous !*
  """

  alias Exmc.Poker.{Cards, Simulator, OpponentModel, Decision}

  @doc """
  Simulate a poker table. Returns observations and true params.

  ## Options
    - :players - list of param maps, or :archetypes (default: first 3 archetypes)
    - :hands - number of hands per player (default: 50)
  """
  def simulate(opts \\ []) do
    players =
      case Keyword.get(opts, :players, :archetypes) do
        :archetypes -> Enum.take(Simulator.archetypes(), 3)
        custom -> custom
      end

    hands = Keyword.get(opts, :hands, 50)
    Simulator.simulate(players, hands)
  end

  @doc """
  Run Bayesian inference on observations to profile opponents.

  ## Options
    - :num_samples - posterior samples (default: 300)
    - :num_warmup - warmup iterations (default: 300)
    - :seed - PRNG seed (default: 42)
  """
  def profile(observations, opts \\ []) do
    num_players = length(observations)
    {ir, _data} = OpponentModel.build(observations)
    init = OpponentModel.init_values(num_players)

    {trace, stats} =
      Exmc.Sampler.sample(ir, init,
        num_samples: Keyword.get(opts, :num_samples, 300),
        num_warmup: Keyword.get(opts, :num_warmup, 300),
        seed: Keyword.get(opts, :seed, 42),
        ncp: false
      )

    profiles = OpponentModel.extract_profiles(trace, num_players)
    {profiles, stats}
  end

  @doc """
  Make a decision given your hand, the board, an opponent profile, and pot/bet info.

  ## Options (keyword list)
    - :my_hole - list of card strings, e.g. ["Ah", "Kh"]
    - :board - list of card strings, e.g. ["Td", "Jd", "2c"]
    - :opponent - profile map from `profile/2`
    - :pot - current pot size
    - :to_call - amount to call
  """
  def decide(opts) do
    hole = opts |> Keyword.fetch!(:my_hole) |> Enum.map(&Cards.parse/1)
    board = opts |> Keyword.fetch!(:board) |> Enum.map(&Cards.parse/1)
    opponent = Keyword.fetch!(opts, :opponent)
    pot = Keyword.fetch!(opts, :pot)
    to_call = Keyword.fetch!(opts, :to_call)

    Decision.expected_value(hole, board, opponent, pot, to_call)
  end

  @doc """
  Pretty-print a comparison of true vs inferred player types.
  """
  def compare(true_params, profiles) do
    Enum.zip(true_params, profiles)
    |> Enum.with_index()
    |> Enum.map(fn {{true_p, profile}, i} ->
      label = Map.get(true_p, :label, "Player #{i}")

      vpip_post = profile.vpip |> Nx.mean() |> Nx.to_number() |> Float.round(3)
      pfr_post = profile.pfr |> Nx.mean() |> Nx.to_number() |> Float.round(3)
      agg_post = profile.agg |> Nx.mean() |> Nx.to_number() |> Float.round(3)
      bluff_post = profile.bluff |> Nx.mean() |> Nx.to_number() |> Float.round(3)

      IO.puts("""
      #{label}:
        VPIP:  true=#{true_p.vpip}  post=#{vpip_post}
        PFR:   true=#{true_p.pfr}   post=#{pfr_post}
        AGG:   true=#{true_p.agg}   post=#{agg_post}
        BLUFF: true=#{true_p.bluff} post=#{bluff_post}
      """)

      %{player: label, true: true_p, posterior_mean: %{vpip: vpip_post, pfr: pfr_post, agg: agg_post, bluff: bluff_post}}
    end)
  end
end
