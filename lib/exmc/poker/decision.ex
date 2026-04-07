defmodule Exmc.Poker.Decision do
  @moduledoc """
  Expected utility calculation using posterior opponent profiles.

  Given posterior samples of opponent parameters and the current game state,
  compute EV of each action (fold, call, raise) with full uncertainty.
  """

  alias Exmc.Poker.{Cards, ActionModel}

  @doc """
  Compute expected value of each action given posterior uncertainty.

  ## Args
    - my_hole: list of 2 card integers (your hole cards)
    - board: list of 0-5 card integers (community cards so far)
    - profile: %{vpip: {N}, pfr: {N}, agg: {N}, bluff: {N}} posterior samples
    - pot: current pot size
    - to_call: amount to call

  ## Returns
    %{fold: float, call: float, raise: float,
      p_call_positive: float, p_raise_positive: float}
  """
  def expected_value(my_hole, board, profile, pot, to_call, opts \\ []) do
    raise_to = Keyword.get(opts, :raise_to, pot)
    equity_samples = Keyword.get(opts, :equity_samples, 200)

    # My hand equity on current board
    equity = Cards.hand_strength(my_hole, board, equity_samples)

    # EV(fold) = 0 always
    ev_fold = 0.0

    # Get posterior sample count
    n_samples = Nx.axis_size(profile.vpip, 0)

    # For each posterior sample, compute opponent's likely response to our raise
    # and the resulting EV
    vpip_list = Nx.to_flat_list(profile.vpip)
    pfr_list = Nx.to_flat_list(profile.pfr)
    agg_list = Nx.to_flat_list(profile.agg)
    bluff_list = Nx.to_flat_list(profile.bluff)

    call_evs =
      for _ <- 1..n_samples do
        # EV(call) = equity * (pot + to_call) - to_call
        equity * (pot + to_call) - to_call
      end

    raise_evs =
      Enum.zip([vpip_list, pfr_list, agg_list, bluff_list])
      |> Enum.map(fn {vpip, pfr, agg, bluff} ->
        # Opponent's response to our raise depends on their params
        # We model: facing a raise, opponent sees it as a strong action
        # Their fold probability increases for tighter players
        {p_fold, _p_call, _p_raise} =
          ActionModel.action_probs(vpip, pfr, agg, bluff, 0.5)

        # If they fold, we win the pot
        # If they call, we play for pot + raise_to + their call
        # If they re-raise, simplify: we're committed
        total_pot = pot + raise_to + raise_to
        ev_they_fold = pot
        ev_they_call = equity * total_pot - raise_to

        p_fold * ev_they_fold + (1 - p_fold) * ev_they_call
      end)

    ev_call = Enum.sum(call_evs) / n_samples
    ev_raise = Enum.sum(raise_evs) / n_samples

    p_call_pos = Enum.count(call_evs, &(&1 > 0)) / n_samples
    p_raise_pos = Enum.count(raise_evs, &(&1 > 0)) / n_samples

    %{
      fold: ev_fold,
      call: ev_call,
      raise: ev_raise,
      equity: equity,
      p_call_positive: p_call_pos,
      p_raise_positive: p_raise_pos,
      recommended: recommend(ev_fold, ev_call, ev_raise)
    }
  end

  defp recommend(ev_fold, ev_call, ev_raise) do
    cond do
      ev_raise >= ev_call and ev_raise >= ev_fold -> :raise
      ev_call >= ev_fold -> :call
      true -> :fold
    end
  end

  @doc """
  Quick summary of a decision for display.
  """
  def format_decision(decision) do
    """
    Equity: #{Float.round(decision.equity * 100, 1)}%
    EV(fold):  #{Float.round(decision.fold, 1)}
    EV(call):  #{Float.round(decision.call, 1)} (#{Float.round(decision.p_call_positive * 100, 1)}% positive)
    EV(raise): #{Float.round(decision.raise, 1)} (#{Float.round(decision.p_raise_positive * 100, 1)}% positive)
    >>> #{String.upcase(to_string(decision.recommended))}
    """
  end
end
