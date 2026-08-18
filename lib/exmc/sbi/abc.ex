defmodule Exmc.SBI.ABC do
  @moduledoc """
  Rejection ABC — the reference arm.

  Draw `θ` from the prior, run the simulator, keep `θ` if
  `d(S(y*), S(y)) ≤ ε`. That is the whole algorithm. The accepted particles are
  exact draws from

      π_ε(θ | s*)  ∝  π(θ) · Pr[ d(S(y), s*) ≤ ε | θ ]

  with no approximation beyond `ε` itself and no tuning parameters to get
  wrong. It is also, at any interesting tolerance, extremely slow: the
  acceptance rate falls like the volume of the `ε`-ball, so halving `ε` in two
  dimensions quarters it.

  **This module exists to be slow.** `Exmc.SBI.ABCSMC` reaches the same target
  for a fraction of the simulator budget, and the way you find out whether it
  actually does is by running this one at the same final `ε` and comparing.
  Do not delete it because it is boring; it is the only arm in `Exmc.SBI` with
  nothing in it to be subtly wrong.

  > #### ABC targets the summaries, not the data {: .warning}
  >
  > See `Exmc.SBI` — the caveat applies to this module identically and it is
  > the one users skip.

  ## Two ways to say how close is close enough

  Fixed tolerance, when you know it:

      Exmc.SBI.ABC.run(MySim, observed,
        prior: [lambda: {:lognormal, 0.0, 0.5}],
        epsilon: 0.2,
        n_particles: 500
      )

  Or a quantile of the prior-predictive distance, when you do not — draw
  `n_draws` proposals, keep the closest `n_particles`, and report the `ε` that
  implies. This is the honest way to pick a first tolerance, and the resulting
  `ε` is a sensible `epsilon_1` for `Exmc.SBI.ABCSMC`:

      Exmc.SBI.ABC.run(MySim, observed,
        prior: [lambda: {:lognormal, 0.0, 0.5}],
        n_particles: 200,
        n_draws: 20_000
      )

  ## Options

  | option | default | meaning |
  |---|---|---|
  | `:prior` | — | required; see `Exmc.SBI.Prior` |
  | `:n_particles` | `200` | accepted particles wanted |
  | `:epsilon` | — | fixed tolerance; mutually exclusive with `:n_draws` |
  | `:n_draws` | — | quantile mode: total prior draws to take |
  | `:max_simulations` | `2_000_000` | budget in `:epsilon` mode |
  | `:batch_size` | `4 × n_particles` | proposals evaluated per batch |
  | `:distance` | `:euclidean` | or a `fn simulated, observed -> float end` |
  | `:summary_scale` | `nil` | per-coordinate divisor before the distance |
  | `:parallel` | `true` | `Task.async_stream` over the batch |
  | `:max_concurrency` | schedulers | |
  | `:seed` / `:rng` | `{1,2,3}` | integer, 3-tuple, or a `:rand` state |

  Returns `{:ok, posterior}` or `{:error, reason}`; see `Exmc.SBI` for the
  posterior shape and the helpers that read it.
  """

  alias Exmc.SBI.{Engine, Prior}

  @doc """
  Run rejection ABC.
  """
  @spec run(Exmc.SBI.Simulator.t(), term(), keyword()) ::
          {:ok, map()} | {:error, term()}
  def run(simulator, observed, opts) do
    simulator = Exmc.SBI.Simulator.validate!(simulator)
    %{prior: prior, n_particles: n, rng: rng} = Engine.common!(opts)
    observed = Engine.observed!(observed)
    distance = Engine.build_distance(opts, observed)

    case {Keyword.get(opts, :epsilon), Keyword.get(opts, :n_draws)} do
      {nil, nil} ->
        {:error,
         {:missing_tolerance,
          "Exmc.SBI.ABC.run/3 needs either :epsilon (fixed tolerance) or :n_draws " <>
            "(keep the closest :n_particles of :n_draws prior draws)"}}

      {eps, nil} when is_number(eps) and eps >= 0 ->
        fixed(simulator, prior, distance, n, eps * 1.0, rng, opts)

      {nil, draws} when is_integer(draws) and draws > 0 ->
        if draws < n do
          {:error, {:bad_options, ":n_draws (#{draws}) must be at least :n_particles (#{n})"}}
        else
          quantile(simulator, prior, distance, n, draws, rng, opts)
        end

      {_, _} ->
        {:error, {:bad_options, ":epsilon and :n_draws are mutually exclusive"}}
    end
  end

  @doc """
  Run rejection ABC, raising on failure.
  """
  @spec run!(Exmc.SBI.Simulator.t(), term(), keyword()) :: map()
  def run!(simulator, observed, opts) do
    case run(simulator, observed, opts) do
      {:ok, posterior} -> posterior
      {:error, reason} -> raise "Exmc.SBI.ABC: #{inspect(reason)}"
    end
  end

  # --- fixed tolerance ---------------------------------------------------

  defp fixed(simulator, prior, distance, n, eps, rng, opts) do
    batch = Keyword.get(opts, :batch_size, 4 * n)
    budget = Keyword.get(opts, :max_simulations, 2_000_000)

    fixed_loop(simulator, prior, distance, n, eps, rng, opts, batch, budget, [], 0)
  end

  defp fixed_loop(_sim, _prior, _distance, n, eps, _rng, _opts, _batch, budget, acc, used)
       when used >= budget do
    {:error,
     {:budget_exhausted,
      %{
        epsilon: eps,
        wanted: n,
        accepted: length(acc),
        n_simulations: used,
        hint:
          "no particle reached epsilon within :max_simulations. Either the tolerance is " <>
            "below what the summaries can resolve, or the prior does not cover the " <>
            "observed summary. Run in :n_draws mode to see the prior-predictive " <>
            "distance distribution before choosing an epsilon."
      }}}
  end

  defp fixed_loop(simulator, prior, distance, n, eps, rng, opts, batch, budget, acc, used) do
    need = n - length(acc)

    if need <= 0 do
      {:ok, finish(prior, Enum.take(Enum.reverse(acc), n), eps, used)}
    else
      size = min(batch, budget - used)
      {proposals, rng} = propose(prior, size, rng)
      results = Engine.evaluate(simulator, proposals, distance, opts)

      accepted =
        results
        |> Enum.filter(fn {_, _, d, _} -> d <= eps end)
        |> Enum.take(need)

      acc = Enum.reverse(accepted) ++ acc
      used = used + size

      # Widen the batch when acceptance is rare, so the budget is spent in
      # simulator calls rather than in round trips. Never narrow it: a batch
      # that shrinks toward `need` spends its whole life one particle short.
      rate = (length(acc) + 1) / (used + 1)
      batch = max(batch, min(round((n - length(acc) + 1) / max(rate, 1.0e-6)), 20 * n))

      fixed_loop(simulator, prior, distance, n, eps, rng, opts, batch, budget, acc, used)
    end
  end

  # --- quantile mode -----------------------------------------------------

  defp quantile(simulator, prior, distance, n, draws, rng, opts) do
    batch = Keyword.get(opts, :batch_size, max(4 * n, 1000))

    results =
      Stream.unfold({rng, draws}, fn
        {_rng, 0} ->
          nil

        {rng, left} ->
          size = min(batch, left)
          {proposals, rng} = propose(prior, size, rng)
          {Engine.evaluate(simulator, proposals, distance, opts), {rng, left - size}}
      end)
      |> Enum.flat_map(& &1)

    kept =
      results
      |> Enum.sort_by(fn {_, _, d, _} -> d end)
      |> Enum.take(n)

    {_, _, eps, _} = List.last(kept)
    {:ok, finish(prior, kept, eps, draws)}
  end

  # --- shared ------------------------------------------------------------

  defp propose(prior, size, rng) do
    {children, rng} = Engine.split_rng(rng, size)

    {proposals, rng} =
      Enum.map_reduce(children, rng, fn child, rng ->
        {params, rng} = Prior.sample(prior, rng)
        {{nil, params, child}, rng}
      end)

    {proposals, rng}
  end

  defp finish(prior, accepted, eps, used) do
    n = length(accepted)
    w = 1.0 / n

    %{
      method: :rejection,
      prior: prior,
      names: Prior.names(prior),
      particles: Enum.map(accepted, fn {_, p, _, _} -> p end),
      weights: List.duplicate(w, n),
      distances: Enum.map(accepted, fn {_, _, d, _} -> d end),
      summaries: Enum.map(accepted, fn {_, _, _, s} -> s end),
      epsilon: eps,
      n_simulations: used,
      acceptance_rate: n / used,
      populations: []
    }
  end
end
