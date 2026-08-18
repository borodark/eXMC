defmodule Exmc.SBI.Engine do
  @moduledoc false

  # Machinery shared by `Exmc.SBI.ABC` and `Exmc.SBI.ABCSMC`: option
  # resolution, the distance function, deterministic RNG splitting, and the
  # batched particle evaluation that C2.3 is about.
  #
  # The one design decision worth stating here, because it is what makes the
  # parallel arm testable: proposals are generated **sequentially** from the
  # parent RNG and evaluated in a **batch** whose size is fixed before any
  # simulator runs. Nothing about the result depends on the concurrency. A
  # `parallel: false` run and a `parallel: true` run over the same seed produce
  # bit-identical particles, weights and simulation counts, which turns "is the
  # parallel path the same algorithm?" from a hopeful assertion into an
  # equality test.
  #
  # The cost of that choice is real and should not be hidden: a batch may
  # simulate more proposals than the population needs, because it cannot stop
  # early. `n_simulations` in every result counts the simulations actually run,
  # not the ones a sequential implementation would have needed, so the reported
  # acceptance rate stays honest.

  import Bitwise

  alias Exmc.SBI.{Prior, Simulator}

  @seed_bits 58

  @doc """
  Split `n` independent child RNG states off `rng`, deterministically.
  """
  @spec split_rng(:rand.state(), non_neg_integer()) :: {[:rand.state()], :rand.state()}
  def split_rng(rng, n) when n >= 0 do
    Enum.map_reduce(1..n//1, rng, fn _, rng ->
      {a, rng} = :rand.uniform_s(1 <<< @seed_bits, rng)
      {b, rng} = :rand.uniform_s(1 <<< @seed_bits, rng)
      {c, rng} = :rand.uniform_s(1 <<< @seed_bits, rng)
      {:rand.seed_s(:exsss, {a, b, c}), rng}
    end)
  end

  @doc """
  Coerce a seed option into a `:rand` state.
  """
  @spec rng_from(term()) :: :rand.state()
  def rng_from(nil), do: :rand.seed_s(:exsss, {1, 2, 3})
  def rng_from(seed) when is_integer(seed), do: :rand.seed_s(:exsss, {seed, seed * 7 + 1, 17})

  def rng_from({a, b, c} = seed) when is_integer(a) and is_integer(b) and is_integer(c),
    do: :rand.seed_s(:exsss, seed)

  # A live `:rand` state is `{alg_handler, internal_state}`. On OTP 27 the
  # handler is a *map*, not the atom it was in older releases; matching only
  # atoms here silently rejected every already-seeded state that was passed in.
  def rng_from({alg, _} = state) when is_map(alg) or is_atom(alg), do: state

  @doc """
  Build the distance function `fn simulated_vector -> float`.
  """
  @spec build_distance(keyword(), [float()]) :: ([float()] -> float())
  def build_distance(opts, observed) do
    len = length(observed)
    scale = normalise_scale(Keyword.get(opts, :summary_scale), len)

    check = fn sim ->
      if length(sim) != len do
        raise ArgumentError,
              "Exmc.SBI: the simulator returned a summary of length #{length(sim)} but the " <>
                "observed summary has length #{len}. A summary whose dimension depends on " <>
                "the parameters has no meaningful distance."
      end

      sim
    end

    case Keyword.get(opts, :distance, :euclidean) do
      fun when is_function(fun, 2) ->
        fn sim -> fun.(check.(sim), observed) * 1.0 end

      :euclidean ->
        fn sim -> euclidean(check.(sim), observed, scale) end

      :weighted_euclidean ->
        fn sim -> euclidean(check.(sim), observed, scale) end

      other ->
        raise ArgumentError,
              "Exmc.SBI: :distance must be :euclidean, :weighted_euclidean, or a " <>
                "2-arity function, got #{inspect(other)}"
    end
  end

  defp normalise_scale(nil, _len), do: nil

  defp normalise_scale(scale, len) when is_list(scale) do
    if length(scale) != len do
      raise ArgumentError,
            "Exmc.SBI: :summary_scale has #{length(scale)} entries but the observed " <>
              "summary has #{len}"
    end

    Enum.map(scale, fn s ->
      s = s * 1.0
      if s <= 0.0, do: raise(ArgumentError, "Exmc.SBI: :summary_scale entries must be > 0")
      s
    end)
  end

  defp euclidean(sim, obs, nil) do
    [sim, obs]
    |> Enum.zip_reduce(0.0, fn [a, b], acc -> acc + (a - b) * (a - b) end)
    |> :math.sqrt()
  end

  defp euclidean(sim, obs, scale) do
    [sim, obs, scale]
    |> Enum.zip_reduce(0.0, fn [a, b, s], acc ->
      d = (a - b) / s
      acc + d * d
    end)
    |> :math.sqrt()
  end

  @doc """
  Evaluate a batch of proposals: run the simulator, take the distance.

  `proposals` is a list of `{tag, params, child_rng}`. Returns
  `[{tag, params, distance, summary}]` **in input order**, whether or not the
  evaluation was parallel.
  """
  @spec evaluate(
          Simulator.t(),
          [{term(), map(), :rand.state()}],
          ([float()] -> float()),
          keyword()
        ) ::
          [{term(), map(), float(), [float()]}]
  def evaluate(simulator, proposals, distance, opts) do
    work = fn {tag, params, child_rng} ->
      {summary, _rng} = Simulator.invoke(simulator, params, child_rng)
      {tag, params, distance.(summary), summary}
    end

    if Keyword.get(opts, :parallel, true) and length(proposals) > 1 do
      proposals
      |> Task.async_stream(work,
        ordered: true,
        max_concurrency: Keyword.get(opts, :max_concurrency, System.schedulers_online()),
        timeout: Keyword.get(opts, :timeout, :infinity)
      )
      |> Enum.map(fn {:ok, r} -> r end)
    else
      Enum.map(proposals, work)
    end
  end

  @doc """
  Resolve and validate the options every entry point shares.
  """
  @spec common!(keyword()) :: %{prior: Prior.t(), n_particles: pos_integer(), rng: :rand.state()}
  def common!(opts) do
    prior =
      opts
      |> Keyword.fetch!(:prior)
      |> Prior.new()

    n = Keyword.get(opts, :n_particles, 200)

    unless is_integer(n) and n > 1 do
      raise ArgumentError, "Exmc.SBI: :n_particles must be an integer > 1, got #{inspect(n)}"
    end

    %{
      prior: prior,
      n_particles: n,
      rng: rng_from(Keyword.get(opts, :rng) || Keyword.get(opts, :seed))
    }
  end

  @doc """
  Normalise the observed summary the same way a simulated one is normalised.
  """
  @spec observed!(term()) :: [float()]
  def observed!(observed), do: Simulator.to_vector(observed)

  @doc """
  Quantile of a list of distances, by linear interpolation on sorted order.
  """
  @spec quantile([float()], float()) :: float()
  def quantile([], _q), do: raise(ArgumentError, "Exmc.SBI: quantile of an empty list")

  def quantile(values, q) when q >= 0.0 and q <= 1.0 do
    sorted = Enum.sort(values)
    n = length(sorted)
    pos = q * (n - 1)
    lo = trunc(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    Enum.at(sorted, lo) * (1.0 - frac) + Enum.at(sorted, hi) * frac
  end
end
