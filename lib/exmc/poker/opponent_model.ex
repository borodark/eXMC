defmodule Exmc.Poker.OpponentModel do
  @moduledoc """
  Hierarchical Bayesian opponent model using Exmc.

  Population-level hyperparameters capture the "meta" at a stake level.
  Per-player parameters are partially pooled via NCP.
  Actions are modeled as softmax over (fold, call, raise) conditioned on
  hand strength and player params.

  This is where NUTS earns its keep: the posterior over 8 + 4*num_players
  continuous correlated parameters is exactly the geometry HMC navigates well.
  """

  alias Exmc.Builder
  alias Exmc.Dist.{Normal, HalfCauchy, Custom}

  @doc """
  Build Exmc IR for the hierarchical opponent model.

  ## Args
    - observations: list of per-player observation maps
      [%{hand_strengths: [float], actions: [0|1|2]}, ...]

  ## Returns
    {ir, data} where data contains pre-built tensors for the likelihood
  """
  def build(observations) do
    num_players = length(observations)

    # Pre-stack observation data as Nx tensors (captured in closure)
    player_data =
      Enum.map(observations, fn obs ->
        %{
          hs: Nx.tensor(obs.hand_strengths, type: :f64),
          acts: Nx.tensor(obs.actions, type: :s64),
          n: length(obs.hand_strengths)
        }
      end)

    ir = Builder.new_ir()

    # --- Population hyperparams ---
    # Logit-scale for [0,1] params (vpip, pfr, bluff)
    # Log-scale for positive params (agg)
    ir =
      ir
      |> Builder.rv("mu_vpip", Normal, %{mu: Nx.tensor(-0.85), sigma: Nx.tensor(0.5)})
      |> Builder.rv("sigma_vpip", HalfCauchy, %{scale: Nx.tensor(0.5)}, transform: :log)
      |> Builder.rv("mu_pfr", Normal, %{mu: Nx.tensor(-1.4), sigma: Nx.tensor(0.5)})
      |> Builder.rv("sigma_pfr", HalfCauchy, %{scale: Nx.tensor(0.5)}, transform: :log)
      |> Builder.rv("mu_agg", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(0.5)})
      |> Builder.rv("sigma_agg", HalfCauchy, %{scale: Nx.tensor(0.5)}, transform: :log)
      |> Builder.rv("mu_bluff", Normal, %{mu: Nx.tensor(-1.4), sigma: Nx.tensor(0.5)})
      |> Builder.rv("sigma_bluff", HalfCauchy, %{scale: Nx.tensor(0.5)}, transform: :log)

    # --- Per-player NCP raw params ---
    ir =
      Enum.reduce(0..(num_players - 1), ir, fn i, acc ->
        acc
        |> Builder.rv("vpip_raw_#{i}", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
        |> Builder.rv("pfr_raw_#{i}", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
        |> Builder.rv("agg_raw_#{i}", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
        |> Builder.rv("bluff_raw_#{i}", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
      end)

    # --- Grouped likelihood via Custom dist ---
    likelihood_fn = fn _x, params ->
      mu_vpip = params["mu_vpip"]
      sigma_vpip = params["sigma_vpip"]
      mu_pfr = params["mu_pfr"]
      sigma_pfr = params["sigma_pfr"]
      mu_agg = params["mu_agg"]
      sigma_agg = params["sigma_agg"]
      mu_bluff = params["mu_bluff"]
      sigma_bluff = params["sigma_bluff"]

      Enum.reduce(0..(num_players - 1), Nx.tensor(0.0), fn i, ll ->
        pd = Enum.at(player_data, i)

        if pd.n == 0 do
          ll
        else
          # NCP reconstruction
          logit_vpip = Nx.add(mu_vpip, Nx.multiply(sigma_vpip, params["vpip_raw_#{i}"]))
          logit_pfr = Nx.add(mu_pfr, Nx.multiply(sigma_pfr, params["pfr_raw_#{i}"]))
          log_agg = Nx.add(mu_agg, Nx.multiply(sigma_agg, params["agg_raw_#{i}"]))
          logit_bluff = Nx.add(mu_bluff, Nx.multiply(sigma_bluff, params["bluff_raw_#{i}"]))

          # Constrained params (clamp to prevent overflow)
          vpip = Nx.sigmoid(Nx.max(Nx.min(logit_vpip, Nx.tensor(10.0)), Nx.tensor(-10.0)))
          pfr = Nx.sigmoid(Nx.max(Nx.min(logit_pfr, Nx.tensor(10.0)), Nx.tensor(-10.0)))
          agg = Nx.exp(Nx.max(Nx.min(log_agg, Nx.tensor(3.0)), Nx.tensor(-3.0)))
          bluff = Nx.sigmoid(Nx.max(Nx.min(logit_bluff, Nx.tensor(10.0)), Nx.tensor(-10.0)))

          # Action log-probs for all hands of this player
          log_probs =
            Exmc.Poker.ActionModel.log_action_probs_nx(vpip, pfr, agg, bluff, pd.hs)

          selected =
            Exmc.Poker.ActionModel.gather_log_probs(log_probs, pd.acts)

          Nx.add(ll, Nx.sum(selected))
        end
      end)
    end

    dist = Custom.new(likelihood_fn)

    # All free RVs as string refs so the compiler defers this obs
    # (routes to deferred_obs_term which resolves refs at sample time)
    param_refs =
      %{
        "mu_vpip" => "mu_vpip",
        "sigma_vpip" => "sigma_vpip",
        "mu_pfr" => "mu_pfr",
        "sigma_pfr" => "sigma_pfr",
        "mu_agg" => "mu_agg",
        "sigma_agg" => "sigma_agg",
        "mu_bluff" => "mu_bluff",
        "sigma_bluff" => "sigma_bluff"
      }
      |> Map.merge(
        for i <- 0..(num_players - 1),
            p <- ~w(vpip pfr agg bluff),
            into: %{} do
          {"#{p}_raw_#{i}", "#{p}_raw_#{i}"}
        end
      )

    ir = Custom.rv(ir, "likelihood", dist, param_refs)
    ir = Builder.obs(ir, "likelihood_obs", "likelihood", Nx.tensor(0.0))

    {ir, %{num_players: num_players, player_data: player_data}}
  end

  @doc """
  Generate init values (all NCP raws at 0 = prior mode).
  """
  def init_values(num_players) do
    base = %{
      "mu_vpip" => -0.85,
      "sigma_vpip" => 0.5,
      "mu_pfr" => -1.4,
      "sigma_pfr" => 0.5,
      "mu_agg" => 0.0,
      "sigma_agg" => 0.5,
      "mu_bluff" => -1.4,
      "sigma_bluff" => 0.5
    }

    per_player =
      for i <- 0..(num_players - 1), param <- ~w(vpip pfr agg bluff), into: %{} do
        {"#{param}_raw_#{i}", 0.0}
      end

    Map.merge(base, per_player)
  end

  @doc """
  Extract constrained player profiles from MCMC trace.

  Returns a list of maps, one per player:
    %{vpip: {N}, pfr: {N}, agg: {N}, bluff: {N}}
  where each value is a tensor of posterior samples.
  """
  def extract_profiles(trace, num_players) do
    mu_vpip = trace["mu_vpip"]
    sigma_vpip = trace["sigma_vpip"]
    mu_pfr = trace["mu_pfr"]
    sigma_pfr = trace["sigma_pfr"]
    mu_agg = trace["mu_agg"]
    sigma_agg = trace["sigma_agg"]
    mu_bluff = trace["mu_bluff"]
    sigma_bluff = trace["sigma_bluff"]

    for i <- 0..(num_players - 1) do
      vpip_raw = trace["vpip_raw_#{i}"]
      pfr_raw = trace["pfr_raw_#{i}"]
      agg_raw = trace["agg_raw_#{i}"]
      bluff_raw = trace["bluff_raw_#{i}"]

      clamp = fn x, lo, hi -> Nx.max(Nx.min(x, Nx.tensor(hi)), Nx.tensor(lo)) end

      logit_vpip = clamp.(Nx.add(mu_vpip, Nx.multiply(sigma_vpip, vpip_raw)), -10.0, 10.0)
      logit_pfr = clamp.(Nx.add(mu_pfr, Nx.multiply(sigma_pfr, pfr_raw)), -10.0, 10.0)
      log_agg = clamp.(Nx.add(mu_agg, Nx.multiply(sigma_agg, agg_raw)), -3.0, 3.0)
      logit_bluff = clamp.(Nx.add(mu_bluff, Nx.multiply(sigma_bluff, bluff_raw)), -10.0, 10.0)

      %{
        vpip: Nx.sigmoid(logit_vpip),
        pfr: Nx.sigmoid(logit_pfr),
        agg: Nx.exp(log_agg),
        bluff: Nx.sigmoid(logit_bluff)
      }
    end
  end
end
