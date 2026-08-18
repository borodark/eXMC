defmodule Exmc.SBI.ConjugateTest do
  use ExUnit.Case, async: false

  Code.require_file("support/conjugate.exs", __DIR__)
  alias Exmc.SBITest.Conjugate

  @moduletag timeout: 300_000

  @moduledoc """
  Both arms against a **closed-form posterior**, not against each other.

  `NEXT.md` §3 records that every statistical check in this repository before
  0.3.1 was differential — run it two ways, assert agreement — and that two real
  defects lived behind a green suite because the two ways shared the code that
  was wrong. So the primary assertions here compare each arm separately to
  `Normal(n·ȳ/(n+1), 1/√(n+1))`, which is the answer whatever the code does.
  The SMC-versus-rejection comparison is kept, but as a third check.

  Tolerances are `4 × SE` of the estimator being asserted on, where the SE is
  computed from the sampler's own effective sample size and the number of
  independent runs averaged. Nothing here is a round number.
  """

  @n 10
  @seeds [1, 2, 3, 4, 5, 6]
  @particles 300
  @pops 6
  # ȳ is fixed rather than simulated: the point is to compare against the
  # posterior for a *known* summary, so the summary must not move.
  @ybar 0.7

  defp exact, do: Conjugate.exact_posterior(@ybar, @n)

  # The SMC arm is used by three of the four tests and is the expensive half;
  # run it once.
  setup_all do
    {:ok, smc: smc_runs()}
  end

  defp smc_runs do
    Enum.map(@seeds, fn seed ->
      {:ok, p} =
        Exmc.SBI.ABCSMC.run(Conjugate.simulator(@n), [@ybar],
          prior: Conjugate.prior(),
          n_particles: @particles,
          n_populations: @pops,
          seed: seed
        )

      p
    end)
  end

  defp mean_of(runs, f), do: Enum.sum(Enum.map(runs, f)) / length(runs)

  test "ABC-SMC recovers the closed-form Normal-Normal posterior", %{smc: runs} do
    {want_mean, want_sd} = exact()

    assert Enum.all?(runs, &(&1.stopped == :completed))

    got_mean = mean_of(runs, &Exmc.SBI.posterior_mean(&1).theta)
    got_sd = mean_of(runs, &Exmc.SBI.posterior_sd(&1).theta)
    eps = mean_of(runs, & &1.epsilon)

    # SE of the weighted posterior mean is sd/√ESS per run, and we average
    # `length(@seeds)` independent runs.
    ess = mean_of(runs, &Exmc.SBI.weight_ess/1)
    se_mean = want_sd / :math.sqrt(ess * length(@seeds))
    # SE of a standard deviation estimated from `m` effective draws is
    # sd/√(2m) for a Gaussian.
    se_sd = want_sd / :math.sqrt(2.0 * ess * length(@seeds))

    # The ABC target is the posterior given |ȳ_sim − ȳ| ≤ ε, which is the exact
    # posterior convolved with a Uniform(−ε, ε): the likelihood variance goes
    # from 1/n to 1/n + ε²/3. That inflation is a property of the algorithm,
    # not an error, so it is added to the tolerance explicitly rather than
    # being absorbed into a rounder number.
    inflated_sd = :math.sqrt(1.0 / (1.0 + 1.0 / (1.0 / @n + eps * eps / 3.0)))
    abc_bias = abs(inflated_sd - want_sd)

    assert_in_delta got_mean, want_mean, 4.0 * se_mean
    assert_in_delta got_sd, want_sd, 4.0 * se_sd + abc_bias
  end

  test "rejection ABC recovers the same closed form at the tolerance SMC reached", %{smc: smc} do
    {want_mean, want_sd} = exact()
    eps = mean_of(smc, & &1.epsilon)

    runs =
      Enum.map(@seeds, fn seed ->
        {:ok, p} =
          Exmc.SBI.ABC.run(Conjugate.simulator(@n), [@ybar],
            prior: Conjugate.prior(),
            n_particles: @particles,
            epsilon: eps,
            seed: 1000 + seed,
            max_simulations: 5_000_000
          )

        p
      end)

    got_mean = mean_of(runs, &Exmc.SBI.posterior_mean(&1).theta)
    got_sd = mean_of(runs, &Exmc.SBI.posterior_sd(&1).theta)

    m = @particles * length(@seeds)
    se_mean = want_sd / :math.sqrt(m)
    se_sd = want_sd / :math.sqrt(2.0 * m)
    inflated_sd = :math.sqrt(1.0 / (1.0 + 1.0 / (1.0 / @n + eps * eps / 3.0)))

    assert_in_delta got_mean, want_mean, 4.0 * se_mean
    assert_in_delta got_sd, want_sd, 4.0 * se_sd + abs(inflated_sd - want_sd)

    # And the efficiency claim that is the entire reason ABC-SMC exists.
    smc_sims = mean_of(smc, & &1.n_simulations)
    rej_sims = mean_of(runs, & &1.n_simulations)
    assert rej_sims > smc_sims
  end

  test "SMC and rejection agree with each other at the same tolerance", %{smc: smc} do
    # Third check, and deliberately last. It is differential, and a differential
    # check cannot see a defect the two arms share.
    eps = mean_of(smc, & &1.epsilon)

    rej =
      Enum.map(@seeds, fn seed ->
        {:ok, p} =
          Exmc.SBI.ABC.run(Conjugate.simulator(@n), [@ybar],
            prior: Conjugate.prior(),
            n_particles: @particles,
            epsilon: eps,
            seed: 2000 + seed,
            max_simulations: 5_000_000
          )

        p
      end)

    {_, want_sd} = exact()
    ess = mean_of(smc, &Exmc.SBI.weight_ess/1)
    m = @particles * length(@seeds)

    se_mean = want_sd * :math.sqrt(1.0 / (ess * length(@seeds)) + 1.0 / m)
    se_sd = se_mean / :math.sqrt(2.0)

    assert_in_delta mean_of(smc, &Exmc.SBI.posterior_mean(&1).theta),
                    mean_of(rej, &Exmc.SBI.posterior_mean(&1).theta),
                    4.0 * se_mean

    assert_in_delta mean_of(smc, &Exmc.SBI.posterior_sd(&1).theta),
                    mean_of(rej, &Exmc.SBI.posterior_sd(&1).theta),
                    4.0 * se_sd
  end

  test "the tolerance is a real approximation, and it widens the posterior" do
    # The honesty requirement in `Exmc.SBI`'s moduledoc, made executable. ABC's
    # target at tolerance ε is *not* the posterior; it is the posterior
    # convolved with the acceptance kernel. At a large ε that is visible, and a
    # user who reads only the mean would never notice.
    {_, want_sd} = exact()

    wide =
      Enum.map(@seeds, fn seed ->
        {:ok, p} =
          Exmc.SBI.ABCSMC.run(Conjugate.simulator(@n), [@ybar],
            prior: Conjugate.prior(),
            n_particles: @particles,
            n_populations: 2,
            seed: seed
          )

        p
      end)

    eps = mean_of(wide, & &1.epsilon)
    got_sd = mean_of(wide, &Exmc.SBI.posterior_sd(&1).theta)
    predicted = :math.sqrt(1.0 / (1.0 + 1.0 / (1.0 / @n + eps * eps / 3.0)))

    assert eps > 0.3, "expected a deliberately loose tolerance, got #{eps}"
    assert got_sd > want_sd * 1.05

    # And the inflation is the size the convolution argument predicts, which is
    # what makes it an understood approximation rather than an unexplained one.
    ess = mean_of(wide, &Exmc.SBI.weight_ess/1)
    se = want_sd / :math.sqrt(2.0 * ess * length(@seeds))
    assert_in_delta got_sd, predicted, 6.0 * se
  end
end
