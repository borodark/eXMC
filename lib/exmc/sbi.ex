defmodule Exmc.SBI do
  @moduledoc """
  Simulation-based (likelihood-free) inference: approximate Bayesian
  computation for models you can run forward but cannot write a density for.

  A discrete-event simulation, an agent model, a queueing network — you can
  execute it, you cannot differentiate it, and there is no `logpdf` to hand
  `Exmc.NUTS`. ABC needs neither. It needs a prior, a simulator, a summary
  statistic, and a notion of "close enough".

      summary = Exmc.SBI.Simulator.to_vector(observed_stats)

      {:ok, posterior} =
        Exmc.SBI.run(MyQueue, summary,
          prior: [lambda: {:lognormal, 0.0, 0.5}, mu: {:lognormal, 0.7, 0.5}],
          n_particles: 300,
          n_populations: 5
        )

      Exmc.SBI.posterior_mean(posterior)
      #=> %{lambda: 1.02, mu: 1.98}

      Exmc.SBI.credible_interval(posterior, :mu, 0.9)
      #=> {1.71, 2.28}

  ## The caveat that belongs on the front page, not in a paper

  > #### ABC targets the posterior given the *summary statistics* {: .warning}
  >
  > Every method in this module approximates
  >
  >     p(θ | S(y*))     — the posterior given the summaries
  >
  > and **not**
  >
  >     p(θ | y*)        — the posterior given the data.
  >
  > These are the same distribution only when `S` is **sufficient** for `θ`. For
  > a queueing model, an agent-based model, or essentially any simulation worth
  > calibrating, it is not, and no choice of tolerance repairs that: driving
  > `ε → 0` converges to `p(θ | S(y*) = s*)`, which is a different — generally
  > wider, sometimes differently located — distribution than the one you would
  > get from the full data.
  >
  > There are therefore *two* approximations in an ABC posterior, and only one
  > of them is under your control:
  >
  > 1. **`ε > 0`.** Tunable. Shrinks with simulator budget, and the tolerance
  >    the run actually reached is reported as `posterior.epsilon`.
  > 2. **`S` is not sufficient.** **Not** tunable. It does not appear in any
  >    diagnostic, it does not shrink with budget, it does not show up as a
  >    divergence or a low effective sample size, and a beautifully converged
  >    ABC run tells you nothing about its size.
  >
  > Choose summaries that you can argue capture what you are estimating, report
  > which ones you used, and treat the result as inference about those
  > summaries. This is stated here — rather than in a docstring three modules
  > down — because the characteristic failure of ABC software is that the
  > sentence exists somewhere and nobody reads it.

  ## What is here

  | module | |
  |---|---|
  | `Exmc.SBI.Simulator` | the behaviour: `simulate(params, rng) :: {summary, rng}` |
  | `Exmc.SBI.Prior` | independent scalar priors, drawable and evaluable |
  | `Exmc.SBI.ABC` | rejection ABC — the slow, obviously correct reference arm |
  | `Exmc.SBI.ABCSMC` | populations, adaptive tolerance, perturbation kernel, importance weights |

  ## Why this is a BEAM library and not a wrapper

  ABC's cost is `N` independent simulator runs per population, and a simulator
  run is a process, not a tensor lane — the one workload in this repository
  where the BEAM's unit of concurrency is exactly the right size. `Exmc.SBI`
  fans a population out over `Task.async_stream` by default.

  It does so without giving up reproducibility. Proposals are generated
  sequentially from the parent `:rand` state and each carries its own derived
  child state, so a run's particles depend on its seed and on nothing else —
  not on the number of schedulers, not on which core got which particle, not on
  whether `parallel: false` was set. That property is asserted in the test
  suite as an equality, not hoped for.

  ## Posterior shape

  Both arms return a map:

      %{
        method: :abc_smc | :rejection,
        names: [:lambda, :mu],          # coordinate order
        particles: [%{lambda: .., mu: ..}, ...],
        weights: [float],               # normalised; uniform for rejection ABC
        distances: [float],
        summaries: [[float]],           # the simulated summary of each particle
        epsilon: float,                 # the tolerance actually reached
        n_simulations: pos_integer,
        acceptance_rate: float,
        populations: [%{...}],          # per-population diagnostics (ABC-SMC)
        stopped: :completed | {reason, detail}   # ABC-SMC only
      }

  **Check `:stopped`.** ABC-SMC returns `{:ok, posterior}` even when it ran out
  of budget or the acceptance rate collapsed, because a truncated population is
  more useful than an error — but only if you can see that it was truncated.
  """

  alias Exmc.SBI.{ABC, ABCSMC, Engine, Prior, Simulator}

  @doc """
  Run simulation-based inference.

  Dispatches on `:method` — `:abc_smc` (default) or `:rejection`. See
  `Exmc.SBI.ABCSMC` and `Exmc.SBI.ABC` for the options each accepts.
  """
  @spec run(Simulator.t(), term(), keyword()) :: {:ok, map()} | {:error, term()}
  def run(simulator, observed, opts) do
    {method, opts} = Keyword.pop(opts, :method, :abc_smc)

    case method do
      :abc_smc ->
        ABCSMC.run(simulator, observed, opts)

      :rejection ->
        ABC.run(simulator, observed, opts)

      other ->
        {:error, {:bad_options, ":method must be :abc_smc or :rejection, got #{inspect(other)}"}}
    end
  end

  @doc "As `run/3`, raising on failure."
  @spec run!(Simulator.t(), term(), keyword()) :: map()
  def run!(simulator, observed, opts) do
    case run(simulator, observed, opts) do
      {:ok, posterior} -> posterior
      {:error, reason} -> raise "Exmc.SBI: #{inspect(reason)}"
    end
  end

  @doc """
  Weighted posterior mean of every parameter.
  """
  @spec posterior_mean(map()) :: %{atom() => float()}
  def posterior_mean(%{names: names} = posterior) do
    Map.new(names, fn name ->
      {name, weighted_mean(values(posterior, name), posterior.weights)}
    end)
  end

  @doc """
  Weighted posterior standard deviation of every parameter.
  """
  @spec posterior_sd(map()) :: %{atom() => float()}
  def posterior_sd(%{names: names} = posterior) do
    Map.new(names, fn name ->
      xs = values(posterior, name)
      m = weighted_mean(xs, posterior.weights)

      var =
        [xs, posterior.weights]
        |> Enum.zip_reduce(0.0, fn [x, w], acc -> acc + w * (x - m) * (x - m) end)

      {name, :math.sqrt(max(var / Enum.sum(posterior.weights), 0.0))}
    end)
  end

  @doc """
  Weighted quantile of one parameter, by the inverse of the weighted ECDF.
  """
  @spec posterior_quantile(map(), atom(), float()) :: float()
  def posterior_quantile(posterior, name, q) when q >= 0.0 and q <= 1.0 do
    total = Enum.sum(posterior.weights)

    pairs =
      [values(posterior, name), posterior.weights]
      |> Enum.zip()
      |> Enum.sort_by(&elem(&1, 0))

    target = q * total

    {_, x} =
      Enum.reduce_while(pairs, {0.0, nil}, fn {x, w}, {acc, _} ->
        acc = acc + w
        if acc >= target, do: {:halt, {acc, x}}, else: {:cont, {acc, x}}
      end)

    x
  end

  @doc """
  Equal-tailed weighted credible interval for one parameter.
  """
  @spec credible_interval(map(), atom(), float()) :: {float(), float()}
  def credible_interval(posterior, name, mass \\ 0.9) do
    tail = (1.0 - mass) / 2.0
    {posterior_quantile(posterior, name, tail), posterior_quantile(posterior, name, 1.0 - tail)}
  end

  @doc """
  Effective sample size of the importance weights, `1 / Σ wᵢ²`.

  For rejection ABC this is exactly `n_particles`. For ABC-SMC it is the number
  that says whether the population is still a population or has collapsed onto
  a handful of particles — a weight ESS far below `n_particles` means the
  perturbation kernel is fighting the prior and the posterior summaries above
  are being computed from far fewer distinct draws than the particle count
  suggests.
  """
  @spec weight_ess(map()) :: float()
  def weight_ess(%{weights: weights}) do
    total = Enum.sum(weights)
    1.0 / Enum.reduce(weights, 0.0, fn w, acc -> acc + w / total * (w / total) end)
  end

  @doc """
  Draw `n` unweighted particles from a weighted posterior, by systematic
  resampling.

  Systematic rather than multinomial: it has strictly lower variance for the
  same cost, and for the rank statistic in a simulation-based-calibration
  check that variance is noise added on top of the thing being measured.
  """
  @spec resample(map(), pos_integer(), :rand.state() | integer() | nil) ::
          {[%{atom() => float()}], :rand.state()}
  def resample(posterior, n, rng \\ nil) do
    rng = Engine.rng_from(rng)
    {u, rng} = :rand.uniform_s(rng)
    total = Enum.sum(posterior.weights)
    cum = Enum.scan(posterior.weights, 0.0, fn w, acc -> acc + w / total end)
    particles = List.to_tuple(posterior.particles)

    positions = Enum.map(0..(n - 1), fn i -> (i + u) / n end)

    {drawn, _} =
      Enum.map_reduce(positions, {cum, 0}, fn p, {remaining, idx} ->
        {remaining, idx} = advance(remaining, idx, p)
        {elem(particles, min(idx, tuple_size(particles) - 1)), {remaining, idx}}
      end)

    {drawn, rng}
  end

  defp advance([c | rest], idx, p) when c < p, do: advance(rest, idx + 1, p)
  defp advance(remaining, idx, _p), do: {remaining, idx}

  @doc """
  The simulation-based-calibration rank of `truth` for one parameter.

  Resamples the weighted posterior down to `n_draws` unweighted particles and
  counts how many fall strictly below `truth`. Under a correctly calibrated
  posterior, and with `θ*` drawn from the prior and the data simulated from
  `θ*`, this rank is uniform on `0 … n_draws`.

  Ties are broken at random, which matters for a discrete simulator: without it
  a duplicated particle value biases the rank downward and the uniformity test
  fires on an artefact of the tie-breaking rather than on the sampler.
  """
  @spec rank(map(), atom(), number(), pos_integer(), :rand.state() | integer() | nil) ::
          {non_neg_integer(), :rand.state()}
  def rank(posterior, name, truth, n_draws, rng \\ nil) do
    rng = Engine.rng_from(rng)
    {draws, rng} = resample(posterior, n_draws, rng)

    Enum.reduce(draws, {0, rng}, fn p, {count, rng} ->
      x = Map.fetch!(p, name)

      cond do
        x < truth ->
          {count + 1, rng}

        x > truth ->
          {count, rng}

        true ->
          {u, rng} = :rand.uniform_s(rng)
          {count + if(u < 0.5, do: 1, else: 0), rng}
      end
    end)
  end

  @doc """
  Per-coordinate scale for the distance, estimated from the prior predictive.

  Draws `n` parameter sets from the prior, simulates each, and returns a scale
  for every summary coordinate. Passed as `:summary_scale`, this makes the
  Euclidean distance invariant to the units the summaries happen to be measured
  in — without it, a summary that happens to be a count dominates one that
  happens to be a probability, and the tolerance is really a tolerance on the
  count alone.

  It is a *prior* predictive scale on purpose: computing it from the observed
  data would make the distance depend on the data twice.

  ## Why the default is the MAD and not the standard deviation

  `statistic: :mad` (the default) returns `1.4826 × median|x − median(x)|`,
  which equals the standard deviation for Gaussian data and ignores the tail
  otherwise. `statistic: :sd` returns the standard deviation.

  The difference is not cosmetic. A prior wide enough to be honest will contain
  corners where the simulator's output diverges — an M/M/1 queue with `λ ≥ μ`
  has no steady state and its mean wait grows without bound. A handful of such
  draws inflates the *standard deviation* of that coordinate by two orders of
  magnitude, the coordinate is then divided by a number far larger than its
  real spread, and the summary silently stops contributing to the distance at
  all. That is not a crash and not a warning; it is a summary statistic that
  has been switched off, and the run still reports a posterior. Measured on
  the M/M/1 fixture in this repository's test suite (300 prior draws, horizon
  300): the SD scale for mean waiting time is **51.2**, the MAD scale
  **0.43** — a factor of 119 between them, on the coordinate that carries most
  of the information about the service rate.
  """
  @spec prior_predictive_scale(Simulator.t(), keyword(), pos_integer()) :: [float()]
  def prior_predictive_scale(simulator, opts, n \\ 200) do
    simulator = Simulator.validate!(simulator)
    prior = opts |> Keyword.fetch!(:prior) |> Prior.new()
    rng = Engine.rng_from(Keyword.get(opts, :rng) || Keyword.get(opts, :seed))

    {children, rng} = Engine.split_rng(rng, n)

    {proposals, _rng} =
      Enum.map_reduce(children, rng, fn child, rng ->
        {params, rng} = Prior.sample(prior, rng)
        {{nil, params, child}, rng}
      end)

    summaries =
      simulator
      |> Engine.evaluate(proposals, fn _ -> 0.0 end, opts)
      |> Enum.map(fn {_, _, _, s} -> s end)

    statistic = Keyword.get(opts, :statistic, :mad)

    summaries
    |> Enum.zip_with(& &1)
    |> Enum.map(fn column ->
      s = column_scale(statistic, column)
      if s > 0.0, do: s, else: 1.0
    end)
  end

  defp column_scale(:sd, column) do
    m = Enum.sum(column) / length(column)

    :math.sqrt(
      Enum.reduce(column, 0.0, fn x, acc -> acc + (x - m) * (x - m) end) / (length(column) - 1)
    )
  end

  defp column_scale(:mad, column) do
    med = median(column)
    1.4826 * median(Enum.map(column, &abs(&1 - med)))
  end

  defp column_scale(other, _column) do
    raise ArgumentError, "Exmc.SBI: :statistic must be :mad or :sd, got #{inspect(other)}"
  end

  defp median(values) do
    sorted = Enum.sort(values)
    n = length(sorted)

    if rem(n, 2) == 1 do
      Enum.at(sorted, div(n, 2))
    else
      (Enum.at(sorted, div(n, 2) - 1) + Enum.at(sorted, div(n, 2))) / 2.0
    end
  end

  defp values(posterior, name), do: Enum.map(posterior.particles, &Map.fetch!(&1, name))

  defp weighted_mean(xs, weights) do
    total = Enum.sum(weights)
    [xs, weights] |> Enum.zip_reduce(0.0, fn [x, w], acc -> acc + w * x end) |> Kernel./(total)
  end
end
