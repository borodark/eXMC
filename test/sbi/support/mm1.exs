defmodule Exmc.SBITest.MM1 do
  @moduledoc """
  A self-contained M/M/1 queue, for testing `Exmc.SBI` against a known answer.

  Single FIFO server, Poisson(λ) arrivals, Exponential(μ) service. Chosen
  because its steady state is closed form, so a calibration run has a right
  answer that does not come from another arm of the same code:

      ρ  = λ / μ                     utilisation
      Wq = ρ / (μ − λ) = λ/(μ(μ−λ))  mean wait in queue
      Lq = ρ² / (1 − ρ)              mean number waiting
      W  = 1 / (μ − λ)               mean time in system

  Implemented as Lindley's recursion rather than an event calendar. For a
  single-server FIFO queue the two are *exactly* equivalent — the k-th
  customer's departure is `D_k = max(A_k, D_{k−1}) + S_k` — and the recursion
  has nothing in it to get wrong, which is the property a test fixture wants.
  The point here is to exercise `Exmc.SBI`; a bug in the fixture would be
  indistinguishable from a bug in the sampler.

  ## Identifiability

  Utilisation alone cannot separate λ from μ — `(2, 4)` and `(1, 2)` share
  `ρ = 0.5`. The summary therefore carries the observed arrival rate as well,
  which pins λ directly; ρ then pins μ. Mean wait and mean queue length are
  the third and fourth coordinates and are largely redundant with the first
  two by Little's law (`Lq = λ·Wq`), which is deliberate: they are the
  coordinates a practitioner would actually record, and including a
  near-collinear summary is the normal condition of applied ABC rather than a
  pathology.
  """

  @warmup_fraction 0.1

  @doc """
  Run one replication and return `%{arrival_rate:, utilisation:, mean_wait:,
  mean_queue:}` plus the advanced RNG state.

  `t_end` is the horizon in time units. The first `#{trunc(@warmup_fraction * 100)}%`
  of customers are discarded as the initial transient — the queue starts empty,
  which is not its steady state.
  """
  @spec run(float(), float(), float(), :rand.state()) :: {map(), :rand.state()}
  def run(lambda, mu, t_end, rng) when lambda > 0 and mu > 0 and t_end > 0 do
    {arrivals, services, rng} = generate(lambda, mu, t_end, rng)

    case arrivals do
      [] ->
        {%{arrival_rate: 0.0, utilisation: 0.0, mean_wait: 0.0, mean_queue: 0.0}, rng}

      _ ->
        {%{
           arrival_rate: length(arrivals) / t_end,
           utilisation: utilisation(services, t_end),
           mean_wait: mean_wait(arrivals, services),
           mean_queue: mean_queue(arrivals, services, t_end)
         }, rng}
    end
  end

  @doc """
  Summary vector in the order `Exmc.SBI` sees it.
  """
  @spec summary(map()) :: [float()]
  def summary(stats) do
    [stats.arrival_rate, stats.utilisation, stats.mean_wait, stats.mean_queue]
  end

  @doc """
  A `Exmc.SBI.Simulator`-shaped closure.

  `replications` averages that many independent runs per particle. The summary
  noise falls like `1/√R`, so raising it is the knob the "the posterior must
  concentrate as replications grow" gate turns.
  """
  @spec simulator(float(), pos_integer()) :: (map(), :rand.state() -> {[float()], :rand.state()})
  def simulator(t_end, replications \\ 1) do
    fn %{lambda: lambda, mu: mu}, rng ->
      # A parameter draw with λ ≥ μ is an unstable queue: the waiting line
      # grows without bound and no steady-state summary exists. Return the
      # observed transient rather than raising — the distance will be enormous
      # and ABC will reject it, which is the correct handling of a parameter
      # the model cannot support.
      {vectors, rng} =
        Enum.map_reduce(1..replications, rng, fn _, rng ->
          {stats, rng} = run(lambda, mu, t_end, rng)
          {summary(stats), rng}
        end)

      {average(vectors), rng}
    end
  end

  @doc "Closed-form steady state. Raises for an unstable queue."
  @spec analytic(float(), float()) :: map()
  def analytic(lambda, mu) when lambda < mu do
    rho = lambda / mu

    %{
      arrival_rate: lambda,
      utilisation: rho,
      mean_wait: rho / (mu - lambda),
      mean_queue: rho * rho / (1.0 - rho),
      mean_time_in_system: 1.0 / (mu - lambda)
    }
  end

  # --- internals ---------------------------------------------------------

  defp generate(lambda, mu, t_end, rng) do
    gen(lambda, mu, t_end, 0.0, [], [], rng)
  end

  defp gen(lambda, mu, t_end, clock, arrivals, services, rng) do
    {u, rng} = :rand.uniform_s(rng)
    clock = clock - :math.log(u) / lambda

    if clock > t_end do
      {Enum.reverse(arrivals), Enum.reverse(services), rng}
    else
      {v, rng} = :rand.uniform_s(rng)
      s = -:math.log(v) / mu
      gen(lambda, mu, t_end, clock, [clock | arrivals], [s | services], rng)
    end
  end

  # Lindley: W_k = max(0, W_{k-1} + S_{k-1} - (A_k - A_{k-1}))
  defp waits(arrivals, services) do
    [arrivals, services]
    |> Enum.zip()
    |> Enum.reduce({[], 0.0, 0.0, nil}, fn {a, s}, {acc, w_prev, s_prev, a_prev} ->
      w =
        case a_prev do
          nil -> 0.0
          prev -> max(0.0, w_prev + s_prev - (a - prev))
        end

      {[w | acc], w, s, a}
    end)
    |> elem(0)
    |> Enum.reverse()
  end

  defp drop_transient(list) do
    n = length(list)
    k = trunc(n * @warmup_fraction)
    if n - k < 2, do: list, else: Enum.drop(list, k)
  end

  defp mean_wait(arrivals, services) do
    kept = arrivals |> waits(services) |> drop_transient()
    Enum.sum(kept) / length(kept)
  end

  defp utilisation(services, t_end), do: min(Enum.sum(services) / t_end, 1.0)

  # Little's law applied to the queue: Lq = λ · Wq. Estimated from the same
  # run rather than from the analytic λ, so it carries its own sampling noise.
  defp mean_queue(arrivals, services, t_end) do
    length(arrivals) / t_end * mean_wait(arrivals, services)
  end

  defp average([single]), do: single

  defp average(vectors) do
    n = length(vectors)
    vectors |> Enum.zip_with(& &1) |> Enum.map(fn col -> Enum.sum(col) / n end)
  end
end
