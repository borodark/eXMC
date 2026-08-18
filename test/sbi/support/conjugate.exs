defmodule Exmc.SBITest.Conjugate do
  @moduledoc """
  The Normal–Normal conjugate model, wearing a simulator's clothes.

      θ  ~ Normal(0, 1)
      yᵢ ~ Normal(θ, 1)      i = 1 … n
      S(y) = ȳ                (sufficient for θ)

  so the posterior is available in closed form,

      θ | ȳ  ~  Normal( n·ȳ / (n + 1),  1 / √(n + 1) )

  and ABC's tolerance is the *only* approximation in the chain. That is the
  whole reason this model is the SBC target rather than the M/M/1 queue: the
  sample mean is sufficient, so `ε → 0` recovers the exact posterior, uniform
  ranks are genuinely the correct expectation, and a failure of the uniformity
  test is a defect in the sampler rather than a restatement of the fact that
  summaries are lossy.

  Run SBC on M/M/1 instead and a red gate would be ambiguous between "the
  sampler is wrong" and "mean waiting time is not sufficient for (λ, μ)", which
  is not a gate at all.

  It also gives a **null arm**: `exact_posterior_sample/5` draws from the
  closed-form posterior directly. Running the whole SBC pipeline against it
  measures the gate's false-positive rate under a sampler that is correct by
  construction, and running it against a deliberately mis-scaled version
  measures the gate's power. A calibration check whose own error rates have
  never been measured is a ritual.
  """

  @doc """
  A `Exmc.SBI.Simulator`-shaped closure: `n` draws from `Normal(θ, 1)`,
  summarised by their mean.
  """
  @spec simulator(pos_integer()) :: (map(), :rand.state() -> {[float()], :rand.state()})
  def simulator(n) do
    fn %{theta: theta}, rng ->
      {sum, rng} =
        Enum.reduce(1..n, {0.0, rng}, fn _, {acc, rng} ->
          {z, rng} = :rand.normal_s(rng)
          {acc + theta + z, rng}
        end)

      {[sum / n], rng}
    end
  end

  @doc "The prior, in `Exmc.SBI.Prior` form."
  @spec prior() :: keyword()
  def prior, do: [theta: {:normal, 0.0, 1.0}]

  @doc "Closed-form posterior `{mean, sd}` given the summary `ȳ` and `n`."
  @spec exact_posterior(float(), pos_integer()) :: {float(), float()}
  def exact_posterior(ybar, n), do: {n * ybar / (n + 1), :math.sqrt(1.0 / (n + 1))}

  @doc """
  A posterior map of `l` exact draws — the shape `Exmc.SBI` returns, so it
  drops straight into `Exmc.SBI.rank/5`.

  `sd_scale` multiplies the posterior standard deviation. `1.0` is the null
  arm: a sampler that is right. Anything else is a sampler that is wrong in a
  specific, known way, which is what a power measurement needs.
  """
  @spec exact_posterior_sample(float(), pos_integer(), pos_integer(), :rand.state(), float()) ::
          {map(), :rand.state()}
  def exact_posterior_sample(ybar, n, l, rng, sd_scale \\ 1.0) do
    {m, sd} = exact_posterior(ybar, n)

    {particles, rng} =
      Enum.map_reduce(1..l, rng, fn _, rng ->
        {z, rng} = :rand.normal_s(rng)
        {%{theta: m + sd * sd_scale * z}, rng}
      end)

    {%{
       method: :exact,
       names: [:theta],
       particles: particles,
       weights: List.duplicate(1.0 / l, l),
       distances: List.duplicate(0.0, l),
       epsilon: 0.0
     }, rng}
  end
end
