defmodule Exmc.Diagnostics do
  @moduledoc """
  MCMC diagnostics: summary statistics, ESS, R-hat, and autocorrelation.

  Performance: autocorrelation uses direct summation in Erlang floats
  (not Nx ops) for speed on BinaryBackend.
  """

  @doc """
  Compute summary statistics for a trace.

  Returns `%{var_name => %{mean:, std:, q5:, q25:, q50:, q75:, q95:}}`.
  """
  def summary(trace) when is_map(trace) do
    Map.new(trace, fn {name, samples} ->
      flat = Nx.to_flat_list(samples)
      n = length(flat)
      mean = Enum.sum(flat) / n
      variance = Enum.sum(Enum.map(flat, fn x -> (x - mean) * (x - mean) end)) / n
      std = :math.sqrt(variance)
      sorted = Enum.sort(flat)

      {name,
       %{
         mean: mean,
         std: std,
         q5: quantile(sorted, n, 0.05),
         q25: quantile(sorted, n, 0.25),
         q50: quantile(sorted, n, 0.50),
         q75: quantile(sorted, n, 0.75),
         q95: quantile(sorted, n, 0.95)
       }}
    end)
  end

  @doc """
  Effective sample size via initial positive sequence estimator (Geyer 1992).

  Takes a 1D tensor or list of samples from a single chain.
  Uses raw (non-rank-normalized) autocorrelation.
  """
  def ess(samples) do
    values = to_float_list(samples)
    n = length(values)

    if n < 4 do
      n * 1.0
    else
      acf = autocorrelation(values, min(n - 1, n))
      ess_from_acf(acf, n)
    end
  end

  @doc """
  Bulk effective sample size (Vehtari et al. 2021).

  Rank-normalizes the chain, then computes ESS on the normalized values.
  This matches ArviZ's `az.ess(data, method="bulk")` for single-chain input.
  """
  def ess_bulk(samples) do
    values = to_float_list(samples)
    n = length(values)

    if n < 4 do
      n * 1.0
    else
      # Rank-normalize: rank → normal quantile
      z = rank_normalize(values)
      acf = autocorrelation(z, min(n - 1, n))
      ess_from_acf(acf, n)
    end
  end

  @doc """
  Split R-hat (Vehtari et al. 2021).

  Takes a list of 1D tensors/lists, one per chain.
  Returns a float. Values near 1.0 indicate convergence.
  """
  def rhat(chains) when is_list(chains) and length(chains) >= 2 do
    # Split each chain in half
    split_chains =
      Enum.flat_map(chains, fn chain ->
        values = to_float_list(chain)
        mid = div(length(values), 2)
        {first, second} = Enum.split(values, mid)
        [first, second]
      end)

    m = length(split_chains)
    chain_lengths = Enum.map(split_chains, &length/1)
    n = Enum.min(chain_lengths)

    # Trim all to same length
    trimmed = Enum.map(split_chains, &Enum.take(&1, n))

    chain_means = Enum.map(trimmed, fn c -> Enum.sum(c) / n end)
    grand_mean = Enum.sum(chain_means) / m

    # Between-chain variance B
    b = n / (m - 1) * Enum.sum(Enum.map(chain_means, fn cm -> (cm - grand_mean) ** 2 end))

    # Within-chain variance W
    chain_vars =
      Enum.zip(trimmed, chain_means)
      |> Enum.map(fn {c, cm} ->
        Enum.sum(Enum.map(c, fn x -> (x - cm) ** 2 end)) / (n - 1)
      end)

    w = Enum.sum(chain_vars) / m

    # R-hat
    var_hat = (n - 1) / n * w + b / n
    :math.sqrt(var_hat / w)
  end

  @doc """
  Raw autocorrelation function via direct computation.

  Takes a 1D tensor or list and max lag. Returns list of ACF values [lag 0 .. max_lag].
  Uses Erlang floats for performance on BinaryBackend.
  """
  def autocorrelation(samples, max_lag) do
    values = to_float_list(samples)
    n = length(values)
    mean = Enum.sum(values) / n
    centered = Enum.map(values, fn x -> x - mean end)
    centered_arr = :array.from_list(centered)
    var = Enum.sum(Enum.map(centered, fn x -> x * x end))

    if var == 0.0 do
      List.duplicate(0.0, max_lag + 1)
    else
      Enum.map(0..max_lag, fn lag ->
        sum =
          Enum.reduce(0..(n - lag - 1), 0.0, fn i, acc ->
            acc + :array.get(i, centered_arr) * :array.get(i + lag, centered_arr)
          end)

        sum / var
      end)
    end
  end

  # --- Private helpers ---

  defp ess_from_acf(acf, n) do
    # Initial positive sequence estimator (Geyer 1992)
    # Gamma_m = rho_{2m} + rho_{2m+1}, tau = -1 + 2 * sum Gamma_m while Gamma_m > 0
    # Note: rho_0 = 1 (included in Gamma_0 = rho_0 + rho_1 = 1 + rho_1)
    max_k = div(length(acf) - 1, 2)

    tau =
      Enum.reduce_while(0..max_k, -1.0, fn k, tau_acc ->
        rho_2k = Enum.at(acf, 2 * k, 0.0)
        rho_2k1 = Enum.at(acf, 2 * k + 1, 0.0)
        pair_sum = rho_2k + rho_2k1

        if pair_sum > 0 do
          {:cont, tau_acc + 2 * pair_sum}
        else
          {:halt, tau_acc}
        end
      end)

    n / max(tau, 1.0)
  end

  defp quantile(sorted, n, p) do
    # Linear interpolation quantile
    h = (n - 1) * p
    lo = floor(h)
    hi = ceil(h)
    frac = h - lo

    lo_val = Enum.at(sorted, lo)
    hi_val = Enum.at(sorted, hi)
    lo_val + frac * (hi_val - lo_val)
  end

  # Rank-normalize: map values to normal quantiles via their ranks.
  # Uses average rank for ties, then Blom's approximation: z = Φ^{-1}((r - 3/8) / (n + 1/4))
  defp rank_normalize(values) do
    n = length(values)
    indexed = Enum.with_index(values)
    sorted = Enum.sort_by(indexed, fn {v, _} -> v end)

    # Compute average ranks (1-based) for tied groups
    ranks = :array.new(n)

    ranks =
      sorted
      |> Enum.chunk_by(fn {v, _} -> v end)
      |> Enum.reduce({ranks, 1}, fn group, {r_arr, pos} ->
        avg_rank = pos + (length(group) - 1) / 2.0

        r_arr =
          Enum.reduce(group, r_arr, fn {_, orig_idx}, acc ->
            :array.set(orig_idx, avg_rank, acc)
          end)

        {r_arr, pos + length(group)}
      end)
      |> elem(0)

    # Transform to normal quantiles
    Enum.map(0..(n - 1), fn i ->
      r = :array.get(i, ranks)
      p = (r - 0.375) / (n + 0.25)
      # Approximate Φ^{-1}(p) using rational approximation (Abramowitz & Stegun)
      probit(p)
    end)
  end

  # Probit function (inverse normal CDF) via rational approximation.
  # Accurate to ~4.5e-4 for 0 < p < 1.
  defp probit(p) when p > 0.0 and p < 1.0 do
    if p < 0.5 do
      -probit_inner(p)
    else
      probit_inner(1.0 - p)
    end
  end

  defp probit_inner(p) do
    # Rational approximation for Φ^{-1}(1-p) where p < 0.5
    t = :math.sqrt(-2.0 * :math.log(p))
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308
    t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
  end

  defp to_float_list(%Nx.Tensor{} = t), do: Nx.to_flat_list(t)
  defp to_float_list(list) when is_list(list), do: list
end
