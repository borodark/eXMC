defmodule Exmc.SBI.ABCSMC do
  @moduledoc """
  Sequential Monte Carlo ABC — Toni et al. (2009), with the adaptive tolerance
  of Del Moral et al. and the perturbation kernel of Beaumont et al. (2009).

  Rejection ABC (`Exmc.SBI.ABC`) draws every proposal from the prior, so at a
  tight tolerance almost every simulator call is wasted on a region the data
  ruled out long ago. ABC-SMC spends the budget where the posterior is: a
  sequence of populations `t = 1 … T` with tolerances `ε₁ > ε₂ > … > ε_T`, each
  proposing from the last rather than from the prior.

  ## The algorithm, exactly as implemented

  **Population 1.** Draw `θ ~ π(θ)`, simulate, accept on
  `d(S(y), s*) ≤ ε₁`. All weights `1/N`. With the default `epsilon_1: nil` the
  first population accepts everything it draws, and `ε₁` is reported as the
  largest distance it saw — the prior-predictive envelope, which is the only
  honest starting tolerance when you do not already know the scale of `d`.

  **Population `t > 1`.** Repeat until `N` particles are accepted:

  1. resample `θ*` from population `t−1` with probability `w_{t−1}`;
  2. perturb: `θ** ~ K_t(· | θ*)`;
  3. **if `π(θ**) = 0`, reject without simulating** — this is not an
     optimisation, it is the step that keeps the target the posterior and not
     the posterior smeared outside its own support;
  4. simulate `y ~ p(· | θ**)`, accept on `d(S(y), s*) ≤ ε_t`.

  **Weights.** For each accepted particle,

  ```
                       π(θ_t^(i))
      w_t^(i)  =  ────────────────────────────────
                  Σ_j  w_{t-1}^(j) · K_t(θ_t^(i) | θ_{t-1}^(j))
  ```

  normalised to sum to one. Computed in log space via log-sum-exp, because the
  denominator is a sum of `N` Gaussian densities and in more than two
  dimensions the individual terms underflow long before the sum does.

  **Tolerance schedule.** `ε_t` is the `:epsilon_quantile` quantile (default
  `0.5`) of the distances accepted in population `t−1`, floored at
  `:epsilon_final`. Pass `:epsilon_schedule` to fix it explicitly instead.

  **Perturbation kernel.** Multivariate normal, mean `θ*`, covariance
  `kernel_scale × Σ̂_{t−1}` where `Σ̂_{t−1}` is the *weighted* empirical
  covariance of population `t−1` and `kernel_scale` defaults to `2.0`. The
  factor of two is Beaumont et al.'s result, not a fudge: for a Gaussian target
  it minimises the Kullback–Leibler divergence between the proposal and the
  target of the next population.

  ## What it is not

  ABC-SMC is an **efficiency device**. At the same final tolerance it targets
  the same distribution as rejection ABC, and if the two disagree, one of them
  is wrong — which is why `Exmc.SBI.ABC` is kept in the library and run in the
  test suite. That comparison is differential, though, and this project has
  been burned by differential-only testing before; the primary gates are
  simulation-based calibration and the closed-form M/M/1 check, not the
  agreement of the two arms.

  > #### ABC targets the summaries, not the data {: .warning}
  >
  > See `Exmc.SBI`. It applies here in full, and it is not weakened by any
  > amount of sequential machinery — `ε → 0` recovers
  > `p(θ | S(y) = S(y*))`, never `p(θ | y*)`, unless `S` is sufficient.

  ## Options

  | option | default | meaning |
  |---|---|---|
  | `:prior` | — | required; see `Exmc.SBI.Prior` |
  | `:n_particles` | `200` | population size `N` |
  | `:n_populations` | `5` | `T` |
  | `:epsilon_1` | `nil` | tolerance for population 1; `nil` accepts everything |
  | `:epsilon_quantile` | `0.5` | `ε_t` from this quantile of population `t−1` |
  | `:epsilon_schedule` | `nil` | explicit `[ε₁, …, ε_T]`, overrides the above |
  | `:epsilon_final` | `0.0` | floor; the run stops once `ε_t` reaches it |
  | `:kernel_scale` | `2.0` | multiplier on the weighted empirical covariance |
  | `:min_acceptance_rate` | `1.0e-4` | stop rather than grind |
  | `:max_simulations_per_population` | `500 × n_particles` | |
  | `:batch_size` | `2 × n_particles` | proposals per parallel batch |
  | `:distance` | `:euclidean` | or `fn simulated, observed -> float end` |
  | `:summary_scale` | `nil` | per-coordinate divisor before the distance |
  | `:parallel` | `true` | `Task.async_stream` over the batch |
  | `:max_concurrency` | schedulers | |
  | `:seed` / `:rng` | `{1,2,3}` | |

  The result carries `:stopped`, which is `:completed` only when the full
  schedule ran. Assert on it; a truncated run that silently reports a posterior
  is exactly the failure mode this repository keeps rediscovering.
  """

  alias Exmc.SBI.{Engine, Prior}

  @log_2pi :math.log(2.0 * :math.pi())

  @doc """
  Run ABC-SMC.
  """
  @spec run(Exmc.SBI.Simulator.t(), term(), keyword()) :: {:ok, map()} | {:error, term()}
  def run(simulator, observed, opts) do
    simulator = Exmc.SBI.Simulator.validate!(simulator)
    %{prior: prior, n_particles: n, rng: rng} = Engine.common!(opts)
    observed = Engine.observed!(observed)
    distance = Engine.build_distance(opts, observed)

    cfg = %{
      simulator: simulator,
      prior: prior,
      n: n,
      distance: distance,
      opts: opts,
      d: Prior.dimension(prior),
      n_populations: Keyword.get(opts, :n_populations, 5),
      schedule: Keyword.get(opts, :epsilon_schedule),
      epsilon_1: Keyword.get(opts, :epsilon_1),
      epsilon_quantile: Keyword.get(opts, :epsilon_quantile, 0.5),
      epsilon_final: Keyword.get(opts, :epsilon_final, 0.0) * 1.0,
      kernel_scale: Keyword.get(opts, :kernel_scale, 2.0) * 1.0,
      min_rate: Keyword.get(opts, :min_acceptance_rate, 1.0e-4),
      budget: Keyword.get(opts, :max_simulations_per_population, 500 * n),
      batch: Keyword.get(opts, :batch_size, 2 * n)
    }

    cfg =
      case cfg.schedule do
        nil -> cfg
        list when is_list(list) and length(list) > 0 -> %{cfg | n_populations: length(list)}
        other -> throw({:bad_schedule, other})
      end

    with {:ok, pop1} <- population_one(cfg, rng) do
      loop(cfg, pop1, [summarise(pop1)], 2)
    end
  catch
    {:bad_schedule, other} ->
      {:error,
       {:bad_options, ":epsilon_schedule must be a non-empty list, got #{inspect(other)}"}}
  end

  @doc """
  Run ABC-SMC, raising on failure.
  """
  @spec run!(Exmc.SBI.Simulator.t(), term(), keyword()) :: map()
  def run!(simulator, observed, opts) do
    case run(simulator, observed, opts) do
      {:ok, posterior} -> posterior
      {:error, reason} -> raise "Exmc.SBI.ABCSMC: #{inspect(reason)}"
    end
  end

  # --- population 1 ------------------------------------------------------

  defp population_one(cfg, rng) do
    eps =
      cond do
        is_list(cfg.schedule) -> hd(cfg.schedule) * 1.0
        is_number(cfg.epsilon_1) -> cfg.epsilon_1 * 1.0
        true -> :infinity
      end

    fill_from_prior(cfg, eps, rng, [], 0, 0)
  end

  defp fill_from_prior(cfg, eps, rng, acc, used, batches) do
    need = cfg.n - length(acc)

    cond do
      need <= 0 ->
        accepted = acc |> Enum.reverse() |> Enum.take(cfg.n)
        distances = Enum.map(accepted, fn {_, d, _} -> d end)
        w = 1.0 / cfg.n

        eps = if eps == :infinity, do: Enum.max(distances), else: eps

        {:ok,
         %{
           index: 1,
           epsilon: eps,
           particles: Enum.map(accepted, fn {p, _, _} -> p end),
           weights: List.duplicate(w, cfg.n),
           distances: distances,
           summaries: Enum.map(accepted, fn {_, _, s} -> s end),
           n_simulations: used,
           n_prior_rejections: 0,
           acceptance_rate: cfg.n / used,
           rng: rng
         }}

      used >= cfg.budget ->
        {:error,
         {:budget_exhausted,
          %{population: 1, epsilon: eps, accepted: length(acc), n_simulations: used}}}

      true ->
        # With no tolerance on population 1 every draw is accepted, so a batch
        # wider than `need` is pure waste and would report a fictitious 50%
        # acceptance rate for a step that accepts everything.
        wanted = if eps == :infinity, do: need, else: max(cfg.batch, need)
        size = min(wanted, cfg.budget - used)
        {children, rng} = Engine.split_rng(rng, size)

        {proposals, rng} =
          Enum.map_reduce(children, rng, fn child, rng ->
            {params, rng} = Prior.sample(cfg.prior, rng)
            {{nil, params, child}, rng}
          end)

        results = Engine.evaluate(cfg.simulator, proposals, cfg.distance, cfg.opts)

        accepted =
          results
          |> Enum.filter(fn {_, _, dist, _} -> eps == :infinity or dist <= eps end)
          |> Enum.take(need)
          |> Enum.map(fn {_, p, dist, s} -> {p, dist, s} end)

        fill_from_prior(cfg, eps, rng, Enum.reverse(accepted) ++ acc, used + size, batches + 1)
    end
  end

  # --- populations 2..T --------------------------------------------------

  defp loop(cfg, prev, history, t) when t > cfg.n_populations do
    {:ok, assemble(cfg, prev, history, :completed)}
  end

  defp loop(cfg, prev, history, t) do
    if prev.epsilon <= cfg.epsilon_final do
      {:ok, assemble(cfg, prev, history, {:epsilon_final, prev.epsilon})}
    else
      case population(cfg, prev, next_epsilon(cfg, prev, t), t) do
        {:ok, pop} ->
          history = history ++ [summarise(pop)]

          if pop.acceptance_rate < cfg.min_rate do
            {:ok, assemble(cfg, pop, history, {:min_acceptance_rate, pop.acceptance_rate})}
          else
            loop(cfg, pop, history, t + 1)
          end

        {:error, {:budget_exhausted, info}} ->
          {:ok, assemble(cfg, prev, history, {:budget_exhausted, info})}

        {:error, _} = err ->
          err
      end
    end
  end

  defp next_epsilon(cfg, prev, t) do
    raw =
      case cfg.schedule do
        list when is_list(list) -> Enum.at(list, t - 1) * 1.0
        _ -> Engine.quantile(prev.distances, cfg.epsilon_quantile)
      end

    # Never let the schedule go backwards: an ε that grew would make the
    # population a strictly worse approximation than the one it came from.
    raw
    |> min(prev.epsilon)
    |> max(cfg.epsilon_final)
  end

  defp population(cfg, prev, eps, t) do
    kernel = build_kernel(cfg, prev)
    cum = cumulative(prev.weights)
    prev_vectors = prev.particles |> Enum.map(&vec(cfg.prior, &1)) |> List.to_tuple()

    fill(cfg, prev, eps, t, kernel, cum, prev_vectors, [], 0, 0)
  end

  defp fill(cfg, prev, eps, t, kernel, cum, prev_vectors, acc, used, prior_rejections) do
    need = cfg.n - length(acc)

    cond do
      need <= 0 ->
        accepted = acc |> Enum.reverse() |> Enum.take(cfg.n)
        vectors = Enum.map(accepted, fn {v, _, _, _} -> v end)
        log_priors = Enum.map(accepted, fn {_, lp, _, _} -> lp end)

        {_, chol, log_det} = kernel

        weights =
          importance_weights(
            log_priors,
            vectors,
            prev.weights,
            Tuple.to_list(prev_vectors),
            chol,
            log_det
          )

        {:ok,
         %{
           index: t,
           epsilon: eps,
           particles: Enum.map(vectors, &Prior.from_vector(cfg.prior, Tuple.to_list(&1))),
           weights: weights,
           distances: Enum.map(accepted, fn {_, _, d, _} -> d end),
           summaries: Enum.map(accepted, fn {_, _, _, s} -> s end),
           n_simulations: used,
           n_prior_rejections: prior_rejections,
           acceptance_rate: cfg.n / used,
           rng: elem(kernel, 0)
         }}

      used >= cfg.budget ->
        {:error,
         {:budget_exhausted,
          %{population: t, epsilon: eps, accepted: length(acc), n_simulations: used}}}

      true ->
        size = min(max(cfg.batch, need), cfg.budget - used)

        {proposals, rejected, kernel} =
          propose_batch(cfg, kernel, cum, prev_vectors, size)

        results = Engine.evaluate(cfg.simulator, proposals, cfg.distance, cfg.opts)

        accepted =
          results
          |> Enum.filter(fn {_, _, dist, _} -> dist <= eps end)
          |> Enum.take(need)
          |> Enum.map(fn {{v, lp}, _params, dist, s} -> {v, lp, dist, s} end)

        fill(
          cfg,
          prev,
          eps,
          t,
          kernel,
          cum,
          prev_vectors,
          Enum.reverse(accepted) ++ acc,
          used + size,
          prior_rejections + rejected
        )
    end
  end

  # Generate `size` proposals with strictly positive prior density. Proposals
  # that land outside the prior's support are discarded here, before any
  # simulator call — cheap, and required for correctness (step 3 above).
  defp propose_batch(cfg, kernel, cum, prev_vectors, size) do
    {rng, chol, _log_det} = kernel

    {proposals, rejected, rng} =
      Enum.reduce(1..size//1, {[], 0, rng}, fn _, {acc, rej, rng} ->
        {v, lp, rej, rng} = one_proposal(cfg, chol, cum, prev_vectors, rej, rng, 0)
        {[{v, lp} | acc], rej, rng}
      end)

    proposals = Enum.reverse(proposals)
    {children, rng} = Engine.split_rng(rng, size)

    tagged =
      proposals
      |> Enum.zip(children)
      |> Enum.map(fn {{v, lp}, child} ->
        {{v, lp}, Prior.from_vector(cfg.prior, Tuple.to_list(v)), child}
      end)

    {tagged, rejected, {rng, chol, elem(kernel, 2)}}
  end

  @max_support_retries 10_000

  defp one_proposal(_cfg, _chol, _cum, _prev, _rej, _rng, tries)
       when tries > @max_support_retries do
    raise "Exmc.SBI.ABCSMC: #{@max_support_retries} consecutive perturbations landed " <>
            "outside the prior's support. The kernel is far too wide for the prior, or " <>
            "the prior's support is a vanishing fraction of the previous population's " <>
            "spread. Lower :kernel_scale, or widen the prior."
  end

  defp one_proposal(cfg, chol, cum, prev_vectors, rej, rng, tries) do
    {j, rng} = draw_index(cum, rng)
    {v, rng} = mvn_sample(elem(prev_vectors, j), chol, rng)

    case Prior.logpdf(cfg.prior, Prior.from_vector(cfg.prior, Tuple.to_list(v))) do
      :neg_infinity -> one_proposal(cfg, chol, cum, prev_vectors, rej + 1, rng, tries + 1)
      lp -> {v, lp, rej, rng}
    end
  end

  # --- importance weights ------------------------------------------------

  @doc """
  The ABC-SMC importance weight, `π(θᵢ) / Σⱼ wⱼ K(θᵢ | θⱼ)`, normalised.

  Public so it can be tested against a hand-computed case; not part of the
  supported API.
  """
  @spec importance_weights([float()], [tuple()], [float()], [tuple()], tuple(), float()) ::
          [float()]
  def importance_weights(log_priors, vectors, prev_weights, prev_vectors, chol, log_det) do
    log_prev = Enum.map(prev_weights, &safe_log/1)

    log_w =
      Enum.zip_with(log_priors, vectors, fn lp, v ->
        denom =
          [log_prev, prev_vectors]
          |> Enum.zip_with(fn [lw, mu] -> lw + mvn_logpdf(v, mu, chol, log_det) end)
          |> logsumexp()

        lp - denom
      end)

    total = logsumexp(log_w)
    Enum.map(log_w, &:math.exp(&1 - total))
  end

  defp safe_log(w) when w > 0.0, do: :math.log(w)
  defp safe_log(_), do: -1.0e308

  defp logsumexp(values) do
    m = Enum.max(values)
    m + :math.log(Enum.reduce(values, 0.0, fn v, acc -> acc + :math.exp(v - m) end))
  end

  # --- the perturbation kernel ------------------------------------------

  defp build_kernel(cfg, prev) do
    vectors = Enum.map(prev.particles, &vec(cfg.prior, &1))
    cov = weighted_cov(vectors, prev.weights, cfg.d)
    scaled = scale_matrix(cov, cfg.kernel_scale)
    chol = cholesky!(scaled, cfg.d)

    log_det =
      2.0 * Enum.reduce(0..(cfg.d - 1), 0.0, fn i, acc -> acc + :math.log(at(chol, i, i)) end)

    {prev.rng, chol, log_det}
  end

  @doc false
  @spec weighted_cov([tuple()], [float()], pos_integer()) :: tuple()
  def weighted_cov(vectors, weights, d) do
    total = Enum.sum(weights)
    w = Enum.map(weights, &(&1 / total))

    means =
      Enum.map(0..(d - 1), fn k ->
        [w, vectors] |> Enum.zip_reduce(0.0, fn [wi, v], acc -> acc + wi * elem(v, k) end)
      end)
      |> List.to_tuple()

    for i <- 0..(d - 1) do
      for j <- 0..(d - 1) do
        [w, vectors]
        |> Enum.zip_reduce(0.0, fn [wi, v], acc ->
          acc + wi * (elem(v, i) - elem(means, i)) * (elem(v, j) - elem(means, j))
        end)
      end
      |> List.to_tuple()
    end
    |> List.to_tuple()
  end

  defp scale_matrix(m, s) do
    d = tuple_size(m)

    for i <- 0..(d - 1) do
      for j <- 0..(d - 1), do: s * at(m, i, j)
    end
    |> Enum.map(&List.to_tuple/1)
    |> List.to_tuple()
  end

  @doc false
  @spec cholesky!(tuple(), pos_integer()) :: tuple()
  def cholesky!(m, d) do
    trace = Enum.reduce(0..(d - 1), 0.0, fn i, acc -> acc + at(m, i, i) end)

    if trace <= 0.0 do
      raise "Exmc.SBI.ABCSMC: the population's weighted covariance has zero trace — every " <>
              "particle is identical. The population has collapsed; reduce " <>
              ":epsilon_quantile so the tolerance falls more slowly, or raise :n_particles."
    end

    base = trace / d

    # A degenerate population — every particle identical, or a parameter that
    # the data has pinned to machine precision — gives a singular covariance
    # and no Cholesky factor. Jitter until it factorises rather than crashing,
    # but bound the jitter and raise if it does not help: a kernel that needs
    # 1e-4 of the trace added to its diagonal is no longer the kernel the
    # algorithm is specified with, and pretending otherwise would be the same
    # move as widening a tolerance.
    Enum.reduce_while([0.0, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4], nil, fn jitter, _ ->
      case try_cholesky(jitter_matrix(m, jitter * base, d), d) do
        {:ok, l} -> {:halt, l}
        :error -> {:cont, nil}
      end
    end)
    |> case do
      nil ->
        raise "Exmc.SBI.ABCSMC: the weighted covariance of the population is not positive " <>
                "definite even with 1e-4 of its trace on the diagonal. The population has " <>
                "collapsed; reduce :epsilon_quantile so the tolerance falls more slowly, " <>
                "or raise :n_particles."

      l ->
        l
    end
  end

  defp jitter_matrix(m, eps, d) do
    for i <- 0..(d - 1) do
      for j <- 0..(d - 1), do: at(m, i, j) + if(i == j, do: eps, else: 0.0)
    end
    |> Enum.map(&List.to_tuple/1)
    |> List.to_tuple()
  end

  defp try_cholesky(m, d) do
    Enum.reduce_while(0..(d - 1), {:ok, empty_rows(d)}, fn i, {:ok, l} ->
      case cholesky_row(m, l, i, d) do
        {:ok, row} -> {:cont, {:ok, put_elem(l, i, row)}}
        :error -> {:halt, :error}
      end
    end)
  end

  defp empty_rows(d), do: List.to_tuple(List.duplicate(List.to_tuple(List.duplicate(0.0, d)), d))

  defp cholesky_row(m, l, i, d) do
    Enum.reduce_while(0..i, {:ok, List.to_tuple(List.duplicate(0.0, d))}, fn j, {:ok, row} ->
      # Row `i` is still under construction and has not been written into `l`,
      # so on the diagonal (j == i) the factor's own row is `row`, not
      # `elem(l, i)` — which is still all zeros. Reading it from `l` silently
      # skips the subtraction and leaves L_ii = √Σ_ii, giving a factor whose
      # L·Lᵀ is not Σ at all. It is self-consistent between `mvn_sample/3` and
      # `mvn_logpdf/4`, so the sampler still targets the right posterior with
      # the wrong kernel, and no statistical gate can see it.
      l_j = if j == i, do: row, else: elem(l, j)

      s =
        Enum.reduce(0..(j - 1)//1, at(m, i, j), fn k, acc ->
          acc - elem(row, k) * elem(l_j, k)
        end)

      cond do
        i == j and s <= 0.0 -> {:halt, :error}
        i == j -> {:cont, {:ok, put_elem(row, j, :math.sqrt(s))}}
        true -> {:cont, {:ok, put_elem(row, j, s / at(l, j, j))}}
      end
    end)
  end

  defp at(m, i, j), do: m |> elem(i) |> elem(j)

  @doc false
  @spec mvn_sample(tuple(), tuple(), :rand.state()) :: {tuple(), :rand.state()}
  def mvn_sample(mean, chol, rng) do
    d = tuple_size(mean)

    {zs, rng} =
      Enum.map_reduce(0..(d - 1), rng, fn _, rng ->
        :rand.normal_s(rng)
      end)

    z = List.to_tuple(zs)

    v =
      Enum.map(0..(d - 1), fn i ->
        Enum.reduce(0..i, elem(mean, i), fn k, acc -> acc + at(chol, i, k) * elem(z, k) end)
      end)

    {List.to_tuple(v), rng}
  end

  @doc false
  @spec mvn_logpdf(tuple(), tuple(), tuple(), float()) :: float()
  def mvn_logpdf(x, mean, chol, log_det) do
    d = tuple_size(x)

    # Solve L y = (x - mean) by forward substitution; ‖y‖² is the Mahalanobis
    # distance. No explicit inverse is ever formed.
    {quad, _} =
      Enum.reduce(0..(d - 1), {0.0, List.to_tuple(List.duplicate(0.0, d))}, fn i, {acc, y} ->
        s =
          Enum.reduce(0..(i - 1)//1, elem(x, i) - elem(mean, i), fn k, a ->
            a - at(chol, i, k) * elem(y, k)
          end)

        yi = s / at(chol, i, i)
        {acc + yi * yi, put_elem(y, i, yi)}
      end)

    -0.5 * (d * @log_2pi + log_det + quad)
  end

  # --- resampling --------------------------------------------------------

  defp cumulative(weights) do
    total = Enum.sum(weights)

    weights
    |> Enum.scan(0.0, fn w, acc -> acc + w / total end)
    |> List.to_tuple()
  end

  defp draw_index(cum, rng) do
    {u, rng} = :rand.uniform_s(rng)
    n = tuple_size(cum)
    {search(cum, u, 0, n - 1), rng}
  end

  defp search(_cum, _u, lo, hi) when lo >= hi, do: lo

  defp search(cum, u, lo, hi) do
    mid = div(lo + hi, 2)
    if u <= elem(cum, mid), do: search(cum, u, lo, mid), else: search(cum, u, mid + 1, hi)
  end

  # --- plumbing ----------------------------------------------------------

  defp vec(prior, params), do: prior |> Prior.to_vector(params) |> List.to_tuple()

  defp summarise(pop) do
    %{
      population: pop.index,
      epsilon: pop.epsilon,
      n_simulations: pop.n_simulations,
      n_prior_rejections: pop.n_prior_rejections,
      acceptance_rate: pop.acceptance_rate,
      weight_ess: weight_ess(pop.weights),
      mean_distance: Enum.sum(pop.distances) / length(pop.distances)
    }
  end

  defp weight_ess(weights) do
    total = Enum.sum(weights)
    sq = Enum.reduce(weights, 0.0, fn w, acc -> acc + w / total * (w / total) end)
    1.0 / sq
  end

  defp assemble(cfg, pop, history, stopped) do
    %{
      method: :abc_smc,
      prior: cfg.prior,
      names: Prior.names(cfg.prior),
      particles: pop.particles,
      weights: pop.weights,
      distances: pop.distances,
      summaries: pop.summaries,
      epsilon: pop.epsilon,
      n_simulations: Enum.reduce(history, 0, &(&1.n_simulations + &2)),
      acceptance_rate: pop.acceptance_rate,
      populations: history,
      stopped: stopped
    }
  end
end
