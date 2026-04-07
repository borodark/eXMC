defmodule Exmc.PokerTest do
  use ExUnit.Case, async: true

  alias Exmc.Poker.{Cards, ActionModel, Simulator, OpponentModel}

  describe "Cards" do
    test "parse and name roundtrip" do
      for card <- Cards.deck() do
        name = Cards.card_name(card)
        assert Cards.parse(name) == card
      end
    end

    test "deck has 52 unique cards" do
      deck = Cards.deck()
      assert length(deck) == 52
      assert length(Enum.uniq(deck)) == 52
    end

    test "evaluate_5 recognizes hand categories" do
      # Royal flush (straight flush, A-high)
      royal = [Cards.parse("Ah"), Cards.parse("Kh"), Cards.parse("Qh"), Cards.parse("Jh"), Cards.parse("Th")]
      {cat, _} = Cards.evaluate_5(royal)
      assert cat == 8

      # Four of a kind
      quads = [Cards.parse("Ac"), Cards.parse("Ad"), Cards.parse("Ah"), Cards.parse("As"), Cards.parse("2c")]
      {cat, _} = Cards.evaluate_5(quads)
      assert cat == 7

      # Full house
      boat = [Cards.parse("Kc"), Cards.parse("Kd"), Cards.parse("Kh"), Cards.parse("Qs"), Cards.parse("Qc")]
      {cat, _} = Cards.evaluate_5(boat)
      assert cat == 6

      # Flush
      flush = [Cards.parse("2h"), Cards.parse("5h"), Cards.parse("7h"), Cards.parse("9h"), Cards.parse("Jh")]
      {cat, _} = Cards.evaluate_5(flush)
      assert cat == 5

      # Straight
      straight = [Cards.parse("5c"), Cards.parse("6d"), Cards.parse("7h"), Cards.parse("8s"), Cards.parse("9c")]
      {cat, _} = Cards.evaluate_5(straight)
      assert cat == 4

      # Wheel (A-2-3-4-5)
      wheel = [Cards.parse("Ac"), Cards.parse("2d"), Cards.parse("3h"), Cards.parse("4s"), Cards.parse("5c")]
      {cat, _} = Cards.evaluate_5(wheel)
      assert cat == 4

      # Three of a kind
      trips = [Cards.parse("7c"), Cards.parse("7d"), Cards.parse("7h"), Cards.parse("Ks"), Cards.parse("2c")]
      {cat, _} = Cards.evaluate_5(trips)
      assert cat == 3

      # Two pair
      two_pair = [Cards.parse("Jc"), Cards.parse("Jd"), Cards.parse("5h"), Cards.parse("5s"), Cards.parse("Ac")]
      {cat, _} = Cards.evaluate_5(two_pair)
      assert cat == 2

      # One pair
      pair = [Cards.parse("Tc"), Cards.parse("Td"), Cards.parse("8h"), Cards.parse("5s"), Cards.parse("2c")]
      {cat, _} = Cards.evaluate_5(pair)
      assert cat == 1

      # High card
      high = [Cards.parse("Ac"), Cards.parse("Kd"), Cards.parse("9h"), Cards.parse("5s"), Cards.parse("2c")]
      {cat, _} = Cards.evaluate_5(high)
      assert cat == 0
    end

    test "hand ranking comparison" do
      flush = [Cards.parse("2h"), Cards.parse("5h"), Cards.parse("7h"), Cards.parse("9h"), Cards.parse("Jh")]
      straight = [Cards.parse("5c"), Cards.parse("6d"), Cards.parse("7h"), Cards.parse("8s"), Cards.parse("9c")]
      assert Cards.evaluate_5(flush) > Cards.evaluate_5(straight)
    end

    test "evaluate_7 picks best 5 from 7" do
      # 7 cards containing a flush
      cards = [
        Cards.parse("2h"), Cards.parse("5h"), Cards.parse("7h"),
        Cards.parse("9h"), Cards.parse("Jh"), Cards.parse("3c"), Cards.parse("Kd")
      ]
      {cat, _} = Cards.evaluate_7(cards)
      assert cat == 5
    end

    test "hand_strength returns value in [0, 1]" do
      hole = [Cards.parse("Ah"), Cards.parse("Kh")]
      board = [Cards.parse("Td"), Cards.parse("Jd"), Cards.parse("2c")]
      strength = Cards.hand_strength(hole, board, 100)
      assert strength >= 0.0 and strength <= 1.0
      # AK with broadway board should be decent
      assert strength > 0.3
    end
  end

  describe "ActionModel" do
    test "action_probs sum to 1" do
      for _ <- 1..20 do
        vpip = :rand.uniform()
        pfr = :rand.uniform() * vpip
        agg = :rand.uniform() * 3
        bluff = :rand.uniform() * 0.5
        hs = :rand.uniform()

        {pf, pc, pr} = ActionModel.action_probs(vpip, pfr, agg, bluff, hs)
        assert_in_delta(pf + pc + pr, 1.0, 1.0e-10)
        assert pf >= 0.0
        assert pc >= 0.0
        assert pr >= 0.0
      end
    end

    test "tighter player folds more" do
      hs = 0.3
      {pf_tight, _, _} = ActionModel.action_probs(0.15, 0.10, 1.0, 0.1, hs)
      {pf_loose, _, _} = ActionModel.action_probs(0.50, 0.10, 1.0, 0.1, hs)
      assert pf_tight > pf_loose
    end

    test "more aggressive player raises more with strong hands" do
      hs = 0.8
      {_, _, pr_passive} = ActionModel.action_probs(0.3, 0.15, 0.5, 0.1, hs)
      {_, _, pr_aggro} = ActionModel.action_probs(0.3, 0.15, 2.5, 0.1, hs)
      assert pr_aggro > pr_passive
    end

    test "bluffier player raises more with weak hands" do
      hs = 0.15
      {_, _, pr_honest} = ActionModel.action_probs(0.3, 0.20, 1.0, 0.05, hs)
      {_, _, pr_bluff} = ActionModel.action_probs(0.3, 0.20, 1.0, 0.50, hs)
      assert pr_bluff > pr_honest
    end

    test "Nx version matches pure Elixir version" do
      vpip = 0.3
      pfr = 0.2
      agg = 1.5
      bluff = 0.25
      hs_list = [0.1, 0.3, 0.5, 0.7, 0.9]

      hs_tensor = Nx.tensor(hs_list)
      log_probs = ActionModel.log_action_probs_nx(
        Nx.tensor(vpip), Nx.tensor(pfr), Nx.tensor(agg), Nx.tensor(bluff), hs_tensor
      )

      for {hs, i} <- Enum.with_index(hs_list) do
        {pf, pc, pr} = ActionModel.action_probs(vpip, pfr, agg, bluff, hs)
        nx_pf = log_probs[i][0] |> Nx.to_number() |> :math.exp()
        nx_pc = log_probs[i][1] |> Nx.to_number() |> :math.exp()
        nx_pr = log_probs[i][2] |> Nx.to_number() |> :math.exp()

        assert_in_delta(pf, nx_pf, 1.0e-5)
        assert_in_delta(pc, nx_pc, 1.0e-5)
        assert_in_delta(pr, nx_pr, 1.0e-5)
      end
    end
  end

  describe "Simulator" do
    test "generates correct observation structure" do
      players = [
        %{vpip: 0.3, pfr: 0.2, agg: 1.5, bluff: 0.2},
        %{vpip: 0.5, pfr: 0.1, agg: 0.5, bluff: 0.05}
      ]

      {obs, _params} = Simulator.simulate(players, 30)
      assert length(obs) == 2

      for o <- obs do
        assert length(o.hand_strengths) == 30
        assert length(o.actions) == 30
        assert Enum.all?(o.actions, &(&1 in [0, 1, 2]))
        assert Enum.all?(o.hand_strengths, &(&1 >= 0.0 and &1 <= 1.0))
      end
    end

    test "archetypes returns 5 player types" do
      types = Simulator.archetypes()
      assert length(types) == 5
      assert Enum.all?(types, &Map.has_key?(&1, :label))
    end
  end

  describe "OpponentModel" do
    test "builds valid IR" do
      obs = [
        %{hand_strengths: [0.3, 0.7, 0.5], actions: [0, 2, 1]},
        %{hand_strengths: [0.1, 0.9], actions: [0, 2]}
      ]

      {ir, data} = OpponentModel.build(obs)
      assert data.num_players == 2
      # 8 hyperparams + 4*2 player raws + 1 likelihood = 17 nodes + 1 obs
      assert map_size(ir.nodes) > 0
    end

    test "init_values covers all params" do
      init = OpponentModel.init_values(3)
      assert Map.has_key?(init, "mu_vpip")
      assert Map.has_key?(init, "sigma_vpip")
      assert Map.has_key?(init, "vpip_raw_0")
      assert Map.has_key?(init, "bluff_raw_2")
    end

    @tag timeout: 120_000
    @tag :poker_integration
    test "compiles and produces finite logp" do
      obs = [
        %{hand_strengths: [0.3, 0.7, 0.5, 0.2, 0.8], actions: [0, 2, 1, 0, 2]}
      ]

      {ir, _} = OpponentModel.build(obs)
      {logp_fn, pm} = Exmc.Compiler.compile(ir, ncp: false)

      # Create a flat vector of the right size
      flat = Nx.broadcast(Nx.tensor(0.0), {pm.size})
      logp = logp_fn.(flat) |> Nx.to_number()
      assert is_number(logp)
      refute logp == :nan
    end
  end

  describe "integration" do
    @tag timeout: 300_000
    @tag :poker_integration
    test "parameter recovery on synthetic data" do
      :rand.seed(:exsss, 42)
      # One player: TAG
      true_params = [%{vpip: 0.25, pfr: 0.20, agg: 1.5, bluff: 0.20}]
      {obs, _} = Simulator.simulate(true_params, 100)

      {ir, _} = OpponentModel.build(obs)
      init = OpponentModel.init_values(1)

      {trace, _stats} =
        Exmc.Sampler.sample(ir, init,
          num_samples: 200,
          num_warmup: 300,
          seed: 42,
          ncp: false
        )

      [profile] = OpponentModel.extract_profiles(trace, 1)

      vpip_mean = profile.vpip |> Nx.mean() |> Nx.to_number()
      pfr_mean = profile.pfr |> Nx.mean() |> Nx.to_number()
      agg_mean = profile.agg |> Nx.mean() |> Nx.to_number()
      bluff_mean = profile.bluff |> Nx.mean() |> Nx.to_number()

      # Generous tolerances for MCMC with 100 observations
      assert_in_delta(vpip_mean, 0.25, 0.15)
      assert_in_delta(pfr_mean, 0.20, 0.15)
      assert_in_delta(agg_mean, 1.5, 1.0)
      assert_in_delta(bluff_mean, 0.20, 0.15)
    end
  end
end
