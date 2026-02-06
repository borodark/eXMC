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
  """
  def ess(samples) do
    values = to_float_list(samples)
    n = length(values)

    if n < 4 do
      n * 1.0
    else
      acf = autocorrelation(values, min(n - 1, n))
      # Initial positive sequence: sum pairs of consecutive ACF values
      # while the sum remains positive
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
    # Initial positive sequence estimator
    # Sum autocorrelation pairs (rho_{2k}, rho_{2k+1}) while their sum > 0
    max_k = div(length(acf) - 2, 2)

    tau =
      Enum.reduce_while(0..max_k, -1.0, fn k, tau_acc ->
        rho_2k = Enum.at(acf, 2 * k + 1, 0.0)
        rho_2k1 = Enum.at(acf, 2 * k + 2, 0.0)
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

  defp to_float_list(%Nx.Tensor{} = t), do: Nx.to_flat_list(t)
  defp to_float_list(list) when is_list(list), do: list
end
