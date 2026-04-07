defmodule Exmc.Poker.Cards do
  @moduledoc """
  Card representation and hand evaluation for Texas Hold'em.

  Cards are integers 0-51:
    rank = div(card, 4)   # 0=2, 1=3, ..., 8=T, 9=J, 10=Q, 11=K, 12=A
    suit = rem(card, 4)   # 0=c, 1=d, 2=h, 3=s
  """

  @ranks ~w(2 3 4 5 6 7 8 9 T J Q K A)
  @suits ~w(c d h s)

  @doc "Extract rank index (0=2 .. 12=A) from a card integer."
  def rank(card), do: div(card, 4)

  @doc "Extract suit index (0=c, 1=d, 2=h, 3=s) from a card integer."
  def suit(card), do: rem(card, 4)

  @doc "Convert a card integer to its two-character name, e.g. `\"Ah\"`, `\"Td\"`."
  def card_name(card), do: Enum.at(@ranks, rank(card)) <> Enum.at(@suits, suit(card))

  @doc "Parse a two-character card name (e.g. `\"Ah\"`) into its integer representation."
  def parse(<<r::binary-size(1), s::binary-size(1)>>) do
    ri = Enum.find_index(@ranks, &(&1 == r))
    si = Enum.find_index(@suits, &(&1 == s))
    ri * 4 + si
  end

  @doc "Return all 52 cards as integers 0..51."
  def deck, do: Enum.to_list(0..51)

  @doc """
  Evaluate a 5-card poker hand.

  Returns `{category, kickers}` where category is:
  0=high, 1=pair, 2=two_pair, 3=trips, 4=straight,
  5=flush, 6=full_house, 7=quads, 8=straight_flush.
  Tuples compare naturally for hand ranking.
  """
  def evaluate_5(five) do
    ranks = five |> Enum.map(&rank/1) |> Enum.sort(:desc)
    suits = five |> Enum.map(&suit/1)

    flush? = length(Enum.uniq(suits)) == 1
    straight? = straight?(ranks)
    groups = ranks |> Enum.frequencies() |> Map.values() |> Enum.sort(:desc)

    cond do
      flush? and straight? -> {8, straight_high(ranks)}
      groups == [4, 1] -> {7, grouped_kickers(ranks, 4)}
      groups == [3, 2] -> {6, grouped_kickers(ranks, 3)}
      flush? -> {5, ranks}
      straight? -> {4, straight_high(ranks)}
      groups == [3, 1, 1] -> {3, grouped_kickers(ranks, 3)}
      groups == [2, 2, 1] -> {2, two_pair_kickers(ranks)}
      groups == [2, 1, 1, 1] -> {1, grouped_kickers(ranks, 2)}
      true -> {0, ranks}
    end
  end

  defp straight?([12, 3, 2, 1, 0]), do: true
  defp straight?([a, b, c, d, e]), do: a - e == 4 and length(Enum.uniq([a, b, c, d, e])) == 5

  defp straight_high([12, 3, 2, 1, 0]), do: [3]
  defp straight_high([high | _]), do: [high]

  defp grouped_kickers(ranks, group_size) do
    freq = Enum.frequencies(ranks)
    {main, _} = Enum.find(freq, fn {_, v} -> v == group_size end)
    others = freq |> Enum.reject(fn {_, v} -> v == group_size end) |> Enum.map(&elem(&1, 0)) |> Enum.sort(:desc)
    [main | others]
  end

  defp two_pair_kickers(ranks) do
    freq = Enum.frequencies(ranks)
    pairs = freq |> Enum.filter(fn {_, v} -> v == 2 end) |> Enum.map(&elem(&1, 0)) |> Enum.sort(:desc)
    {kicker, _} = Enum.find(freq, fn {_, v} -> v == 1 end)
    pairs ++ [kicker]
  end

  @doc "Find the best 5-card hand from 7 cards (flop/turn/river). Evaluates all 21 combinations."
  def evaluate_7(seven) do
    combinations(seven, 5)
    |> Enum.map(&evaluate_5/1)
    |> Enum.max()
  end

  @doc "Generate all k-element combinations from a list."
  def combinations(_list, 0), do: [[]]
  def combinations([], _k), do: []
  def combinations([h | t], k) do
    (combinations(t, k - 1) |> Enum.map(&[h | &1])) ++ combinations(t, k)
  end

  @doc """
  Estimate hand strength as win probability vs a random opponent hand.
  Uses Monte Carlo sampling over remaining cards.
  """
  def hand_strength(hole, board, samples \\ 500) do
    dead = MapSet.new(hole ++ board)
    remaining = Enum.reject(deck(), &MapSet.member?(dead, &1))
    board_needed = 5 - length(board)

    {wins, ties, _total} =
      1..samples
      |> Enum.reduce({0, 0, 0}, fn _, {w, t, n} ->
        shuffled = Enum.shuffle(remaining)
        {opp, rest} = Enum.split(shuffled, 2)
        {fill, _} = Enum.split(rest, board_needed)
        full_board = board ++ fill

        my = evaluate_7(hole ++ full_board)
        theirs = evaluate_7(opp ++ full_board)

        cond do
          my > theirs -> {w + 1, t, n + 1}
          my == theirs -> {w, t + 1, n + 1}
          true -> {w, t, n + 1}
        end
      end)

    (wins + ties * 0.5) / samples
  end
end
