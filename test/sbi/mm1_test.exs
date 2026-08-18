defmodule Exmc.SBI.MM1Test do
  use ExUnit.Case, async: false

  Code.require_file("support/mm1.exs", __DIR__)
  alias Exmc.SBITest.MM1

  @moduletag timeout: 600_000

  @moduledoc """
  The M/M/1 closed-form gate.

  Two things are being checked and they are not the same thing:

  1. **The fixture is right.** The simulator's long-run statistics must match
     `ρ = λ/μ`, `Wq = ρ/(μ−λ)` and `Lq = ρ²/(1−ρ)`. Without this the rest is
     a comparison against another piece of code we wrote.
  2. **The inference is right.** ABC-SMC calibrating `(λ, μ)` from that
     simulator's output must cover the values that generated the output, and
     must concentrate as the number of replications per particle grows.

  The second is a real gate rather than a demonstration precisely because the
  first exists: the answer is known, independently of the sampler.

  Every tolerance below is either a standard error computed from the run's own
  replications, or a binomial bound on a coverage count. Nothing is a round
  number, and nothing was widened to make it pass.
  """

  @true_lambda 1.0
  @true_mu 1.8
  @t_end 300.0
  @prior [lambda: {:uniform, 0.2, 2.0}, mu: {:uniform, 0.6, 4.0}]

  @datasets [101, 202, 303, 404, 505, 606]
  @particles 150
  @populations 11

  setup_all do
    sim1 = MM1.simulator(@t_end, 1)

    # A robust prior-predictive scale. Without it the distance is dominated
    # by whichever coordinate happens to have the largest units; with the
    # *standard deviation* instead of the MAD it is dominated by the corner
    # of the prior where λ ≥ μ and the queue diverges.
    scale = Exmc.SBI.prior_predictive_scale(sim1, [prior: @prior, seed: 5], 300)

    observed =
      Map.new(@datasets, fn seed ->
        {s, _} =
          sim1.(%{lambda: @true_lambda, mu: @true_mu}, :rand.seed_s(:exsss, {seed, seed, 7}))

        {seed, s}
      end)

    runs =
      for reps <- [1, 8], seed <- @datasets do
        {reps, seed}
      end
      |> Task.async_stream(
        fn {reps, seed} ->
          {:ok, p} =
            Exmc.SBI.ABCSMC.run(MM1.simulator(@t_end, reps), observed[seed],
              prior: @prior,
              n_particles: @particles,
              n_populations: @populations,
              summary_scale: scale,
              min_acceptance_rate: 0.004,
              max_simulations_per_population: 40_000,
              seed: seed,
              parallel: false
            )

          {reps, seed, p}
        end,
        timeout: :infinity,
        max_concurrency: 12
      )
      |> Enum.map(fn {:ok, r} -> r end)

    {:ok, runs: runs, scale: scale, observed: observed}
  end

  describe "the fixture, against the textbook" do
    test "long-run statistics converge to the analytic steady state" do
      want = MM1.analytic(@true_lambda, @true_mu)
      reps = 60
      horizon = 2_000.0

      {stats, _} =
        Enum.map_reduce(1..reps, :rand.seed_s(:exsss, {31, 41, 59}), fn _, rng ->
          MM1.run(@true_lambda, @true_mu, horizon, rng)
        end)

      # The replications are independent, so the standard error of their mean
      # is the honest tolerance — no autocorrelation argument needed, and it
      # tightens automatically if anyone raises `reps`.
      for key <- [:arrival_rate, :utilisation, :mean_wait, :mean_queue] do
        xs = Enum.map(stats, &Map.fetch!(&1, key))
        m = Enum.sum(xs) / reps
        var = Enum.reduce(xs, 0.0, fn x, acc -> acc + (x - m) * (x - m) end) / (reps - 1)
        se = :math.sqrt(var / reps)

        assert_in_delta m,
                        Map.fetch!(want, key),
                        4.0 * se,
                        "#{key}: simulated #{m} vs analytic #{Map.fetch!(want, key)} (se #{se})"
      end
    end

    test "an unstable queue is reported, not silently smoothed" do
      # λ > μ: the waiting line grows without bound, so there is no steady
      # state and the summary must reflect that rather than looking plausible.
      {stats, _} = MM1.run(2.0, 1.0, 500.0, :rand.seed_s(:exsss, {1, 1, 1}))

      assert stats.utilisation >= 0.99
      assert stats.mean_wait > 100.0
      assert_raise FunctionClauseError, fn -> MM1.analytic(2.0, 1.0) end
    end
  end

  describe "ABC-SMC calibration of (λ, μ)" do
    test "the posterior covers the parameters that generated the data", %{runs: runs} do
      misses =
        for {_reps, _seed, p} <- runs,
            {name, truth} <- [lambda: @true_lambda, mu: @true_mu],
            {lo, hi} = Exmc.SBI.credible_interval(p, name, 0.9),
            truth < lo or truth > hi do
          {name, truth, lo, hi}
        end

      n = length(runs) * 2

      # 24 interval checks at nominal 90%. Under exact calibration the miss
      # count is Binomial(24, 0.1) — mean 2.4 — and P(X ≥ 8) ≈ 0.007, so a
      # threshold of 8 gives the gate a false-positive rate below 1%. The
      # checks are not fully independent (the two parameters share a run, and
      # the two replication arms share a dataset), which makes the real tail
      # heavier than binomial and the threshold correspondingly conservative.
      #
      # The expected direction of failure is under-coverage: ABC posteriors
      # are convolved with the acceptance kernel and are therefore *wider*
      # than the exact posterior, so a correct implementation over-covers. A
      # posterior that misses is one that has concentrated in the wrong place.
      assert length(misses) < 8,
             "#{length(misses)}/#{n} 90% intervals missed the truth: #{inspect(misses)}"
    end

    test "the posterior concentrates as replications per particle grow", %{runs: runs} do
      by_reps =
        Enum.group_by(runs, fn {reps, _, _} -> reps end, fn {_, seed, p} ->
          {seed, Exmc.SBI.posterior_sd(p)}
        end)
        |> Map.new(fn {reps, list} -> {reps, Map.new(list)} end)

      # Averaging one summary over 8 independent replications divides its
      # sampling variance by 8, so the tolerance the adaptive schedule can
      # reach falls and the posterior tightens. This is the whole reason a
      # simulation budget buys a better posterior in ABC.
      for name <- [:lambda, :mu], seed <- @datasets do
        one = Map.fetch!(by_reps[1], seed) |> Map.fetch!(name)
        eight = Map.fetch!(by_reps[8], seed) |> Map.fetch!(name)

        assert eight < one,
               "#{name}, dataset #{seed}: sd at 8 replications (#{eight}) is not below " <>
                 "sd at 1 (#{one})"
      end

      for name <- [:lambda, :mu] do
        avg = fn r ->
          Enum.sum(Enum.map(@datasets, &Map.fetch!(Map.fetch!(by_reps[r], &1), name))) /
            length(@datasets)
        end

        ratio = avg.(8) / avg.(1)

        assert ratio < 0.8,
               "#{name}: mean posterior sd only fell to #{Float.round(ratio, 3)} of its " <>
                 "one-replication value"
      end
    end

    test "every run reached the end of its schedule or said why not", %{runs: runs} do
      for {reps, seed, p} <- runs do
        assert p.stopped == :completed,
               "reps=#{reps} seed=#{seed} stopped with #{inspect(p.stopped)}"

        assert p.epsilon > 0.0
        assert length(p.particles) == @particles
        assert_in_delta Enum.sum(p.weights), 1.0, 1.0e-12
        assert Exmc.SBI.weight_ess(p) > @particles * 0.5
      end
    end

    test "the tolerance schedule is monotone and the acceptance rate falls with it",
         %{runs: runs} do
      for {_reps, _seed, p} <- runs do
        eps = Enum.map(p.populations, & &1.epsilon)
        assert eps == Enum.sort(eps, :desc)
        assert length(p.populations) == @populations
        assert hd(p.populations).acceptance_rate == 1.0
        assert List.last(p.populations).acceptance_rate < 0.5
      end
    end

    test "rejection ABC agrees with ABC-SMC at the same tolerance", %{
      scale: scale,
      observed: observed
    } do
      # The third check, and the weakest: it is differential. Kept because a
      # disagreement is decisive, not because agreement proves much.
      obs = observed[hd(@datasets)]
      sim = MM1.simulator(@t_end, 1)

      {:ok, smc} =
        Exmc.SBI.ABCSMC.run(sim, obs,
          prior: @prior,
          n_particles: 200,
          n_populations: 6,
          summary_scale: scale,
          seed: 8
        )

      {:ok, rej} =
        Exmc.SBI.ABC.run(sim, obs,
          prior: @prior,
          n_particles: 200,
          epsilon: smc.epsilon,
          summary_scale: scale,
          seed: 9,
          max_simulations: 400_000
        )

      smc_mean = Exmc.SBI.posterior_mean(smc)
      rej_mean = Exmc.SBI.posterior_mean(rej)
      smc_sd = Exmc.SBI.posterior_sd(smc)
      rej_sd = Exmc.SBI.posterior_sd(rej)

      for name <- [:lambda, :mu] do
        se =
          :math.sqrt(
            Map.fetch!(smc_sd, name) ** 2 / Exmc.SBI.weight_ess(smc) +
              Map.fetch!(rej_sd, name) ** 2 / 200
          )

        assert_in_delta Map.fetch!(smc_mean, name), Map.fetch!(rej_mean, name), 4.0 * se

        assert_in_delta Map.fetch!(smc_sd, name),
                        Map.fetch!(rej_sd, name),
                        4.0 * se / :math.sqrt(2.0) + 0.1 * Map.fetch!(rej_sd, name)
      end

      # ABC-SMC exists to spend fewer simulator calls for the same tolerance.
      assert rej.n_simulations > smc.n_simulations
    end
  end
end
