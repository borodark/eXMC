defmodule Exmc.SBI.SBCTest do
  use ExUnit.Case, async: false

  Code.require_file("support/conjugate.exs", __DIR__)
  Code.require_file("support/uniformity.exs", __DIR__)

  alias Exmc.SBITest.{Conjugate, Uniformity}

  @moduletag timeout: 1_800_000

  @moduledoc """
  Simulation-based calibration — the primary gate for `Exmc.SBI`.

  `MISSION.md` §8 says not to implement SBC, and `PLAN_SAMPLER_ROADMAP.md` §5
  says this is the one place in the whole roadmap where it is the right tool.
  Both are correct, and the reason they are is worth stating: Geweke's joint
  distribution test is ~40× cheaper than SBC for the same job, but it needs an
  exact-invariance argument about a Markov kernel. ABC has no Markov kernel and
  no likelihood. SBC is what is left, and here it is affordable — the
  simulator is ten normal draws, so 800 complete ABC-SMC fits cost about ten
  seconds rather than the hours a NUTS SBC would.

  ## The construction

  Draw `θ* ~ π`, simulate `y*` from it, run ABC-SMC on `y*`, resample the
  weighted posterior down to 99 unweighted draws, and count how many fall below
  `θ*`. Under a calibrated posterior that rank is uniform on `0 … 99`. Repeat
  800 times; bin the ranks into 10 bins of 10; Pearson chi-squared with 9
  degrees of freedom.

  The target is the Normal–Normal conjugate model, whose summary — the sample
  mean — is **sufficient**. That is not incidental. Run SBC on the M/M/1 queue
  and a red gate would be ambiguous between "the sampler is wrong" and "mean
  waiting time is not sufficient for (λ, μ)", and an ambiguous gate teaches
  people to ignore it.

  ## The gate's own error rates, measured rather than assumed

  A calibration check whose false-positive rate has never been measured is a
  ritual. Both arms below were run before this test was written, using the same
  rank and chi-squared code, against an *exact* conjugate posterior sampler
  standing in for ABC-SMC (`Conjugate.exact_posterior_sample/5`).

  **Null (`sd_scale = 1.0`), 300 independent SBC experiments of 800 replicates
  each, 10 bins:** (`p` is one draw from Uniform(0,1) per experiment, so these
  are the rates the threshold below is set against)

  | | measured | nominal |
  |---|---|---|
  | reject at α = 0.01 | **0.0033** | 0.01 |
  | reject at α = 0.05 | **0.0367** | 0.05 |
  | median p-value | **0.485** | 0.5 |

  **Power at α = 0.01, same design, against a posterior whose standard
  deviation is wrong by a fixed factor:**

  | error | 100 reps | 200 | 400 | 800 | 1500 |
  |---|---|---|---|---|---|
  | sd × 0.95 | 0.003 | 0.008 | 0.033 | 0.047 | 0.133 |
  | sd × 0.90 | 0.008 | 0.053 | 0.203 | **0.427** | 0.830 |
  | sd × 0.85 | 0.053 | 0.193 | 0.583 | **0.913** | 1.000 |
  | sd × 1.10 | 0.018 | 0.038 | 0.115 | **0.307** | 0.713 |

  The 100/200/400 columns are over 400 experiments each; the 800 and 1500
  columns over 300.

  800 replicates was chosen from that table: it is the smallest design in it
  that exceeds 90% power against a 15% posterior-scale error while costing
  about ten seconds. Bin count was chosen the same way — 4, 5, 10, 20 and 25
  bins were all measured and 10 was the most powerful at every replicate count.

  ## Power in situ, and why that test is here rather than in a comment

  The table above measures the *statistic*. The second test in this module
  measures the **gate**, against the real sampler: it takes the same 800
  ABC-SMC posteriors and throws the importance weights away, which is the
  single most likely way for an ABC-SMC implementation to be wrong, and asserts
  that the gate catches it. Measured p-value for that arm: **7.3e-8**, against
  **0.299** for the unmodified posteriors.

  It caught something real, too, though not this way. The `L·Lᵀ = Σ` unit test
  in `Exmc.SBI.KernelTest` found a Cholesky factor whose diagonal skipped its
  subtraction — the kernel was wider than `2 × Σ̂`. SBC was **green** across
  that defect and correctly so: `mvn_sample/3` and `mvn_logpdf/4` used the same
  wrong factor, so the sampler still targeted the right posterior with the
  wrong proposal. That is exactly the division of labour to expect. SBC checks
  the answer; the exact unit tests check the algorithm.
  """

  @n 10
  @l 99
  @bins 10
  @replicates 800
  @particles 200
  @populations 8

  setup_all do
    sim = Conjugate.simulator(@n)

    posteriors =
      1..@replicates
      |> Task.async_stream(
        fn i ->
          rng = :rand.seed_s(:exsss, {i, i * 977 + 13, 20_260_816})
          {theta, rng} = :rand.normal_s(rng)
          {[ybar], rng} = sim.(%{theta: theta}, rng)

          {:ok, post} =
            Exmc.SBI.ABCSMC.run(sim, [ybar],
              prior: Conjugate.prior(),
              n_particles: @particles,
              n_populations: @populations,
              rng: rng,
              parallel: false
            )

          {theta, post, rng}
        end,
        timeout: :infinity,
        max_concurrency: max(System.schedulers_online(), 8)
      )
      |> Enum.map(fn {:ok, r} -> r end)

    {:ok, posteriors: posteriors}
  end

  defp ranks(posteriors, transform) do
    Enum.map(posteriors, fn {theta, post, rng} ->
      {r, _} = Exmc.SBI.rank(transform.(post), :theta, theta, @l, rng)
      r
    end)
  end

  test "ABC-SMC posterior ranks are uniform", %{posteriors: posteriors} do
    assert Enum.all?(posteriors, fn {_, p, _} -> p.stopped == :completed end)

    eps = Enum.map(posteriors, fn {_, p, _} -> p.epsilon end)
    max_eps = Enum.max(eps)

    # ABC's ε inflates the posterior variance from 1/(1+n) to
    # 1/(1 + 1/(1/n + ε²/3)). The gate detects a 10% scale error 43% of the
    # time, so the schedule has to leave ε small enough that its own bias is
    # far below that — otherwise a red gate would be reporting the tolerance
    # rather than the sampler. At the worst replicate this bounds the
    # inflation, and it is asserted rather than assumed.
    worst_inflation =
      :math.sqrt(1.0 / (1.0 + 1.0 / (1.0 / @n + max_eps * max_eps / 3.0))) *
        :math.sqrt(1.0 + @n) - 1.0

    assert worst_inflation < 0.05,
           "the loosest tolerance reached (#{max_eps}) inflates the posterior sd by " <>
             "#{Float.round(worst_inflation * 100, 2)}%, which is inside the range this " <>
             "gate is meant to detect. Raise :n_populations."

    {stat, df, p} = Uniformity.chisq_uniform(ranks(posteriors, & &1), @l, @bins)

    assert p > 0.01,
           "rank histogram is not uniform: chi2 = #{Float.round(stat, 3)} on #{df} df, " <>
             "p = #{p}, counts = #{inspect(Uniformity.bin_ranks(ranks(posteriors, & &1), @l, @bins))}"
  end

  test "the gate has power against the defect it is most likely to face",
       %{posteriors: posteriors} do
    # Throw away the importance weights. This is not a synthetic perturbation:
    # it is what ABC-SMC degenerates into if `importance_weights/6` is wrong,
    # forgotten, or normalised in the wrong place, and it leaves a posterior
    # that is tilted toward the previous population's kernel smear rather than
    # the prior. Every other diagnostic in the result map — acceptance rate,
    # tolerance schedule, weight ESS, agreement with rejection ABC at a loose
    # tolerance — looks entirely healthy in that state.
    flatten = fn post ->
      %{post | weights: List.duplicate(1.0, length(post.particles))}
    end

    {stat, _df, p} = Uniformity.chisq_uniform(ranks(posteriors, flatten), @l, @bins)

    assert p < 1.0e-3,
           "SBC failed to detect a posterior with its importance weights discarded " <>
             "(chi2 = #{Float.round(stat, 3)}, p = #{p}). The gate is not measuring anything."
  end

  @tag :slow
  test "the gate's false-positive rate under the null" do
    # Reproduces the null row of the table in this module's docs. Excluded from
    # the default run because it is 200 complete SBC experiments, but it is the
    # thing that makes the p-value threshold above mean something, so it is a
    # test and not a paragraph.
    m = 200
    sim = Conjugate.simulator(@n)

    ps =
      1..m
      |> Task.async_stream(fn s -> null_experiment(sim, @replicates, 1.0, s) end,
        timeout: :infinity,
        max_concurrency: max(System.schedulers_online(), 8)
      )
      |> Enum.map(fn {:ok, p} -> p end)

    rejected = Enum.count(ps, &(&1 < 0.01))
    median = Enum.at(Enum.sort(ps), div(m, 2))

    # Under the null the count is Binomial(200, 0.01): mean 2, sd 1.41.
    # P(X ≥ 8) ≈ 0.001, so 8 is the threshold with a ~0.1% false alarm rate.
    assert rejected < 8,
           "#{rejected}/#{m} null experiments rejected at α = 0.01; the chi-squared " <>
             "approximation or the rank statistic is miscalibrated"

    # And the p-values should be uniform, not merely non-small.
    assert_in_delta median, 0.5, 4.0 * 0.5 / :math.sqrt(m)
  end

  @tag :slow
  test "the gate's power against a known posterior-scale error" do
    # Reproduces the sd × 0.85 row at 800 replicates.
    m = 100
    sim = Conjugate.simulator(@n)

    ps =
      1..m
      |> Task.async_stream(fn s -> null_experiment(sim, @replicates, 0.85, s + 500_000) end,
        timeout: :infinity,
        max_concurrency: max(System.schedulers_online(), 8)
      )
      |> Enum.map(fn {:ok, p} -> p end)

    power = Enum.count(ps, &(&1 < 0.01)) / m

    # Measured 0.913 over 300 experiments. Asserted at 0.80 so the test is not
    # a restatement of a single measurement, and the threshold is a floor on
    # the gate's usefulness rather than a fitted value.
    assert power > 0.80,
           "power against a 15% posterior-scale error is only #{power} at α = 0.01"
  end

  defp null_experiment(sim, replicates, sd_scale, seed) do
    rng0 = :rand.seed_s(:exsss, {seed, seed * 31 + 7, 1234})

    {ranks, _} =
      Enum.map_reduce(1..replicates, rng0, fn _, rng ->
        {theta, rng} = :rand.normal_s(rng)
        {[ybar], rng} = sim.(%{theta: theta}, rng)
        {post, rng} = Conjugate.exact_posterior_sample(ybar, @n, @l + 1, rng, sd_scale)
        {r, rng} = Exmc.SBI.rank(post, :theta, theta, @l, rng)
        {r, rng}
      end)

    {_stat, _df, p} = Uniformity.chisq_uniform(ranks, @l, @bins)
    p
  end
end
