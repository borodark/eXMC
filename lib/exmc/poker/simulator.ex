defmodule Exmc.Poker.Simulator do
  @moduledoc """
  Generate synthetic poker observations from known player types.
  Feed to OpponentModel for Bayesian parameter recovery.
  """

  alias Exmc.Poker.{Cards, ActionModel}

  @doc """
  Generate synthetic observations for a table of players.

  ## Args
    - player_params: list of %{vpip: float, pfr: float, agg: float, bluff: float}
    - hands: number of hands to simulate per player

  ## Returns
    {observations, true_params} where observations is the format OpponentModel.build expects
  """
  def simulate(player_params, hands \\ 50) do
    observations =
      Enum.map(player_params, fn params ->
        {strengths, actions} =
          1..hands
          |> Enum.map(fn _ ->
            # Random hand strength (simplified: uniform [0,1])
            hs = :rand.uniform()
            action = ActionModel.sample_action(params.vpip, params.pfr, params.agg, params.bluff, hs)
            {hs, action}
          end)
          |> Enum.unzip()

        %{hand_strengths: strengths, actions: actions}
      end)

    {observations, player_params}
  end

  @doc """
  Generate a set of archetypal player types for testing.
  """
  def archetypes do
    [
      # TAG: tight-aggressive (the winning style)
      %{vpip: 0.22, pfr: 0.18, agg: 1.8, bluff: 0.25, label: :tag},
      # LAG: loose-aggressive (tricky, exploitative)
      %{vpip: 0.35, pfr: 0.28, agg: 2.2, bluff: 0.40, label: :lag},
      # Nit: ultra-tight (only plays premiums)
      %{vpip: 0.12, pfr: 0.10, agg: 1.2, bluff: 0.10, label: :nit},
      # Calling station: loose-passive (calls everything)
      %{vpip: 0.45, pfr: 0.08, agg: 0.5, bluff: 0.05, label: :station},
      # Maniac: hyper-aggressive (raises everything)
      %{vpip: 0.50, pfr: 0.40, agg: 3.0, bluff: 0.55, label: :maniac}
    ]
  end

  @doc """
  Simulate a full poker hand with card dealing and showdown.
  Returns detailed hand history.
  """
  def simulate_hand(player_holes, board_cards \\ nil) do
    deck = Cards.deck() |> Enum.shuffle()

    {holes, deck} =
      if player_holes do
        dead = List.flatten(player_holes)
        {player_holes, Enum.reject(deck, &(&1 in dead))}
      else
        n = 2
        # Deal 2 cards to each of N imaginary players
        {dealt, rest} =
          Enum.reduce(1..2, {[], deck}, fn _, {acc, d} ->
            hand = Enum.take(d, n)
            {acc ++ [hand], Enum.drop(d, n)}
          end)
        {dealt, rest}
      end

    board =
      if board_cards do
        board_cards
      else
        Enum.take(deck, 5)
      end

    evaluations =
      Enum.map(holes, fn hole ->
        hand = Cards.evaluate_7(hole ++ board)
        strength = Cards.hand_strength(hole, Enum.take(board, 3), 200)
        %{hole: Enum.map(hole, &Cards.card_name/1), hand_rank: hand, flop_strength: strength}
      end)

    %{board: Enum.map(board, &Cards.card_name/1), players: evaluations}
  end
end
