defmodule Exmc.ModelComparison do
  @moduledoc """
  Model comparison via WAIC and LOO-CV.

  Uses pointwise log-likelihood evaluated at posterior samples to compute
  information criteria for model selection.
  """

  alias Exmc.{IR, Rewrite, Transform}

  @doc """
  Compute pointwise log-likelihood for each observation at each posterior sample.

  Returns `%{obs_node_id => [log_lik_1, log_lik_2, ...]}` where each list
  has length equal to the number of posterior samples.

  The trace should contain constrained-space values (as returned by the sampler).
  """
  def pointwise_log_likelihood(%IR{} = ir, trace) do
    ir = Rewrite.apply(ir)
    obs_evaluators = build_obs_evaluators(ir)

    n_samples = trace |> Map.values() |> hd() |> Nx.shape() |> elem(0)

    Enum.reduce(0..(n_samples - 1), %{}, fn i, acc ->
      sample_map = extract_sample(trace, i)

      Enum.reduce(obs_evaluators, acc, fn {obs_id, eval_fn}, acc ->
        logp_result = eval_fn.(sample_map)

        case Nx.shape(logp_result) do
          {} ->
            # Scalar: single data point
            logp = Nx.to_number(logp_result)
            Map.update(acc, obs_id, [logp], &[logp | &1])

          {_n} ->
            # Vector: n data points, keyed as {obs_id, index}
            values = Nx.to_flat_list(logp_result)

            Enum.with_index(values)
            |> Enum.reduce(acc, fn {v, idx}, acc ->
              key = {obs_id, idx}
              Map.update(acc, key, [v], &[v | &1])
            end)
        end
      end)
    end)
    |> Map.new(fn {key, values} -> {key, Enum.reverse(values)} end)
  end

  @doc """
  Compute WAIC (Widely Applicable Information Criterion).

  Takes pointwise log-likelihood from `pointwise_log_likelihood/2`.

  Returns `%{waic:, elpd_waic:, p_waic:, se:, n_obs:}` where:
  - `waic`: -2 * elpd_waic (lower is better)
  - `elpd_waic`: expected log pointwise predictive density
  - `p_waic`: effective number of parameters
  - `se`: standard error of elpd_waic
  """
  def waic(pointwise_ll) when is_map(pointwise_ll) do
    obs_ids = Map.keys(pointwise_ll)
    n = length(obs_ids)

    if n == 0, do: raise(ArgumentError, "No observations for WAIC computation")

    results =
      Enum.map(obs_ids, fn obs_id ->
        ll = pointwise_ll[obs_id]
        lppd_i = log_mean_exp(ll)
        p_waic_i = variance(ll)
        {lppd_i, p_waic_i}
      end)

    lppd = Enum.sum(Enum.map(results, &elem(&1, 0)))
    p_waic = Enum.sum(Enum.map(results, &elem(&1, 1)))
    elpd_waic = lppd - p_waic
    waic_val = -2 * elpd_waic

    elpd_pointwise = Enum.map(results, fn {l, p} -> l - p end)
    se = if n > 1, do: :math.sqrt(n * variance(elpd_pointwise)), else: 0.0

    %{waic: waic_val, elpd_waic: elpd_waic, p_waic: p_waic, se: se, n_obs: n}
  end

  @doc """
  Compute LOO-CV via importance sampling.

  Takes pointwise log-likelihood from `pointwise_log_likelihood/2`.

  Returns `%{loo:, elpd_loo:, p_loo:, se:, n_obs:}` where:
  - `loo`: -2 * elpd_loo (lower is better)
  - `elpd_loo`: expected log pointwise predictive density (LOO)
  - `p_loo`: effective number of parameters
  - `se`: standard error of elpd_loo
  """
  def loo(pointwise_ll) when is_map(pointwise_ll) do
    obs_ids = Map.keys(pointwise_ll)
    n = length(obs_ids)

    if n == 0, do: raise(ArgumentError, "No observations for LOO computation")

    results =
      Enum.map(obs_ids, fn obs_id ->
        ll = pointwise_ll[obs_id]
        loo_i_basic(ll)
      end)

    elpd_loo = Enum.sum(Enum.map(results, & &1.elpd))
    p_loo_val = Enum.sum(Enum.map(results, & &1.p_loo))
    loo_val = -2 * elpd_loo

    elpd_pointwise = Enum.map(results, & &1.elpd)
    se = if n > 1, do: :math.sqrt(n * variance(elpd_pointwise)), else: 0.0

    %{loo: loo_val, elpd_loo: elpd_loo, p_loo: p_loo_val, se: se, n_obs: n}
  end

  @doc """
  Compare multiple models by their information criterion results.

  Takes a list of `{label, result}` tuples where result is from `waic/1` or `loo/1`.
  Returns a list sorted by IC (best first) with relative differences.
  """
  def compare(results) when is_list(results) do
    sorted =
      Enum.sort_by(results, fn {_label, r} ->
        Map.get(r, :waic, Map.get(r, :loo, 0))
      end)

    best_elpd =
      case hd(sorted) do
        {_, %{elpd_waic: e}} -> e
        {_, %{elpd_loo: e}} -> e
      end

    Enum.map(sorted, fn {label, r} ->
      ic = Map.get(r, :waic, Map.get(r, :loo))
      elpd = Map.get(r, :elpd_waic, Map.get(r, :elpd_loo))
      se = r.se
      d_elpd = elpd - best_elpd

      %{label: label, ic: ic, elpd: elpd, se: se, d_elpd: d_elpd}
    end)
  end

  # --- Private: build obs evaluators from IR ---

  defp extract_sample(trace, i) do
    Map.new(trace, fn {rv_id, samples} ->
      val = Nx.slice_along_axis(samples, i, 1, axis: 0) |> Nx.squeeze()
      {rv_id, val}
    end)
  end

  defp build_obs_evaluators(%IR{} = ir) do
    ir.nodes
    |> Map.values()
    |> Enum.filter(&obs_node?/1)
    |> Enum.map(fn node -> {node.id, build_obs_eval(node, ir)} end)
    |> Enum.reject(fn {_id, f} -> is_nil(f) end)
  end

  defp build_obs_eval(%{op: {:obs, target_id, value, meta}}, ir) do
    target = IR.get_node!(ir, target_id)
    build_obs_eval_for_target(target, value, meta)
  end

  defp build_obs_eval(%{op: {:obs, target_id, value}}, ir) do
    target = IR.get_node!(ir, target_id)
    build_obs_eval_for_target(target, value, %{})
  end

  defp build_obs_eval(_node, _ir), do: nil

  defp build_obs_eval_for_target(%{op: {:rv, dist, params}}, value, meta) do
    # Strip :reduce for pointwise evaluation (we want per-element logp)
    meta_pw = Map.delete(meta, :reduce)

    if has_string_refs?(params) do
      fn sample_map ->
        resolved = resolve_from_trace(params, sample_map)
        logp = dist.logpdf(value, resolved)
        apply_obs_meta(logp, meta_pw)
      end
    else
      logp = dist.logpdf(value, params)
      logp = apply_obs_meta(logp, meta_pw)
      fn _sample_map -> logp end
    end
  end

  defp build_obs_eval_for_target(%{op: {:rv, dist, params, transform}}, value, meta) do
    # Strip :reduce for pointwise evaluation (we want per-element logp)
    meta_pw = Map.delete(meta, :reduce)
    z = inverse_transform(transform, value)
    x = Transform.apply(transform, z)

    if has_string_refs?(params) do
      fn sample_map ->
        resolved = resolve_from_trace(params, sample_map)
        logp = dist.logpdf(x, resolved)
        jac = Transform.log_abs_det_jacobian(transform, z)
        apply_obs_meta(Nx.add(logp, jac), meta_pw)
      end
    else
      logp = dist.logpdf(x, params)
      jac = Transform.log_abs_det_jacobian(transform, z)
      combined = apply_obs_meta(Nx.add(logp, jac), meta_pw)
      fn _sample_map -> combined end
    end
  end

  defp build_obs_eval_for_target(_target, _value, _meta), do: nil

  defp resolve_from_trace(params, sample_map) do
    Map.new(params, fn
      {k, v} when is_binary(v) -> {k, Map.fetch!(sample_map, v)}
      {k, v} -> {k, v}
    end)
  end

  defp obs_node?(%{op: {:obs, _, _}}), do: true
  defp obs_node?(%{op: {:obs, _, _, _}}), do: true
  defp obs_node?(_), do: false

  defp has_string_refs?(params) do
    Enum.any?(Map.values(params), &is_binary/1)
  end

  defp apply_obs_meta(logp, meta) when map_size(meta) == 0, do: logp

  defp apply_obs_meta(logp, meta) do
    weight = Map.get(meta, :weight, 1.0)

    if weight != 1.0 do
      Nx.multiply(logp, Nx.tensor(weight))
    else
      logp
    end
  end

  defp inverse_transform(nil, x), do: x
  defp inverse_transform(:log, x), do: Nx.log(x)
  defp inverse_transform(:softplus, x), do: Nx.log(Nx.expm1(x))
  defp inverse_transform(:logit, x), do: Nx.subtract(Nx.log(x), Nx.log1p(Nx.negate(x)))

  # --- Math helpers (Erlang floats) ---

  defp log_mean_exp(values) do
    n = length(values)
    max_val = Enum.max(values)
    sum_exp = Enum.sum(Enum.map(values, fn v -> :math.exp(v - max_val) end))
    max_val + :math.log(sum_exp) - :math.log(n)
  end

  defp variance(values) do
    n = length(values)

    if n < 2 do
      0.0
    else
      mean = Enum.sum(values) / n
      Enum.sum(Enum.map(values, fn x -> (x - mean) * (x - mean) end)) / (n - 1)
    end
  end

  defp loo_i_basic(log_liks) do
    neg_ll = Enum.map(log_liks, &(-&1))
    raw_elpd = -log_mean_exp(neg_ll)
    lppd_i = log_mean_exp(log_liks)
    p_loo_i = lppd_i - raw_elpd
    %{elpd: raw_elpd, p_loo: p_loo_i}
  end
end
