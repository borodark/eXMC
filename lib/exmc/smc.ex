defmodule Exmc.SMC do
  @moduledoc """
  Sequential Monte Carlo sampler with likelihood tempering.

  Uses a sequence of tempered distributions:
    p_t(z) ~ prior(z) * likelihood(z)^beta_t
  where beta_t goes from 0 to 1.
  """

  alias Exmc.{Compiler, Transform}

  @default_opts [
    num_particles: 500,
    threshold_ratio: 0.5,
    num_mh_steps: 5,
    seed: 0
  ]

  def sample(ir, opts \\ []) do
    opts = Keyword.merge(@default_opts, opts)
    {logp_fn, pm} = Compiler.compile(ir)

    if pm.size == 0 do
      {%{}, %{num_stages: 0, betas: [], ess_history: [], acceptance_rates: []}}
    else
      d = pm.size
      n = opts[:num_particles]
      rng = :rand.seed_s(:exsss, opts[:seed])

      {particles, rng} = init_particles(d, n, rng)

      logps =
        Enum.map(particles, fn q ->
          val = Nx.to_number(logp_fn.(q))
          if is_number(val), do: val, else: -1.0e10
        end)

      {particles, _logps, info, _rng} =
        smc_loop(logp_fn, particles, logps, d, n, rng, opts)

      trace = build_trace(particles, pm)
      {trace, info}
    end
  end

  defp init_particles(d, n, rng) do
    Enum.map_reduce(1..n, rng, fn _, rng ->
      {vals, rng} =
        Enum.map_reduce(1..d, rng, fn _, rng ->
          {v, rng} = :rand.normal_s(rng)
          {v * 0.5, rng}
        end)

      {Nx.tensor(vals, type: :f64), rng}
    end)
  end

  defp smc_loop(logp_fn, particles, logps, d, n, rng, opts) do
    threshold = opts[:threshold_ratio] * n
    num_mh = opts[:num_mh_steps]
    do_smc(logp_fn, particles, logps, d, n, rng, 0.0, threshold, num_mh, [], [], [])
  end

  defp do_smc(
         _logp_fn,
         particles,
         logps,
         _d,
         _n,
         rng,
         beta,
         _threshold,
         _num_mh,
         betas,
         ess_hist,
         acc_hist
       )
       when beta >= 1.0 do
    info = %{
      num_stages: length(betas),
      betas: Enum.reverse(betas),
      ess_history: Enum.reverse(ess_hist),
      acceptance_rates: Enum.reverse(acc_hist)
    }

    {particles, logps, info, rng}
  end

  defp do_smc(
         logp_fn,
         particles,
         logps,
         d,
         n,
         rng,
         beta,
         threshold,
         num_mh,
         betas,
         ess_hist,
         acc_hist
       ) do
    new_beta = find_next_beta(logps, beta, n, threshold)
    new_beta = min(new_beta, 1.0)
    delta_beta = new_beta - beta

    log_weights =
      Enum.map(logps, fn lp ->
        if is_number(lp), do: delta_beta * lp, else: -1.0e10
      end)

    max_lw = Enum.max(log_weights)
    weights = Enum.map(log_weights, fn lw -> :math.exp(lw - max_lw) end)
    sum_w = Enum.sum(weights)
    norm_weights = Enum.map(weights, fn w -> w / sum_w end)

    ess = 1.0 / Enum.sum(Enum.map(norm_weights, fn w -> w * w end))

    {particles, logps, rng} =
      if ess < threshold do
        resample(particles, logps, norm_weights, n, rng)
      else
        {particles, logps, rng}
      end

    scale = estimate_scale(particles, d)

    {particles, logps, acc_rate, rng} =
      mh_mutate(logp_fn, particles, logps, d, n, scale, new_beta, num_mh, rng)

    do_smc(
      logp_fn,
      particles,
      logps,
      d,
      n,
      rng,
      new_beta,
      threshold,
      num_mh,
      [new_beta | betas],
      [ess | ess_hist],
      [acc_rate | acc_hist]
    )
  end

  defp find_next_beta(logps, current_beta, n, threshold) do
    finite_logps = Enum.filter(logps, &is_number/1)

    if finite_logps == [] do
      1.0
    else
      do_bisect(finite_logps, current_beta, current_beta, 1.0, n, threshold, 0)
    end
  end

  defp do_bisect(_logps, _current, lo, hi, _n, _threshold, iter) when iter > 50, do: (lo + hi) / 2

  defp do_bisect(logps, current, lo, hi, n, threshold, iter) do
    mid = (lo + hi) / 2
    delta = mid - current
    log_w = Enum.map(logps, fn lp -> delta * lp end)
    max_lw = Enum.max(log_w)
    w = Enum.map(log_w, fn lw -> :math.exp(lw - max_lw) end)
    sum_w = Enum.sum(w)
    nw = Enum.map(w, fn wi -> wi / sum_w end)
    ess = 1.0 / Enum.sum(Enum.map(nw, fn wi -> wi * wi end))

    if abs(ess - threshold) < 1.0 do
      mid
    else
      if ess < threshold do
        do_bisect(logps, current, lo, mid, n, threshold, iter + 1)
      else
        do_bisect(logps, current, mid, hi, n, threshold, iter + 1)
      end
    end
  end

  defp resample(particles, logps, weights, n, rng) do
    {u, rng} = :rand.uniform_s(rng)
    u_start = u / n
    positions = Enum.map(0..(n - 1), fn i -> u_start + i / n end)

    cumulative = Enum.scan(weights, &(&1 + &2))

    indices =
      Enum.map(positions, fn pos ->
        Enum.find_index(cumulative, fn c -> c >= pos end) || n - 1
      end)

    new_particles = Enum.map(indices, fn i -> Enum.at(particles, i) end)
    new_logps = Enum.map(indices, fn i -> Enum.at(logps, i) end)
    {new_particles, new_logps, rng}
  end

  defp estimate_scale(particles, d) do
    particle_lists = Enum.map(particles, &Nx.to_flat_list/1)
    n = length(particle_lists)

    Enum.map(0..(d - 1), fn dim ->
      vals = Enum.map(particle_lists, fn p -> Enum.at(p, dim) end)
      mean = Enum.sum(vals) / n
      var = Enum.sum(Enum.map(vals, fn v -> (v - mean) * (v - mean) end)) / max(n - 1, 1)
      :math.sqrt(max(var, 1.0e-10)) * 2.38 / :math.sqrt(d)
    end)
  end

  defp mh_mutate(logp_fn, particles, logps, _d, n, scale, beta, num_steps, rng) do
    {particles, logps, total_accepted, rng} =
      Enum.reduce(1..num_steps, {particles, logps, 0, rng}, fn _, {parts, lps, acc_count, rng} ->
        {new_pairs, {step_acc, rng}} =
          Enum.zip(parts, lps)
          |> Enum.map_reduce({0, rng}, fn {q, lp}, {accepted, rng} ->
            {perturbation, rng} =
              Enum.map_reduce(scale, rng, fn s, rng ->
                {z, rng} = :rand.normal_s(rng)
                {z * s, rng}
              end)

            q_prop = Nx.add(q, Nx.tensor(perturbation, type: :f64))
            lp_prop_raw = Nx.to_number(logp_fn.(q_prop))
            lp_prop = if is_number(lp_prop_raw), do: lp_prop_raw, else: -1.0e10

            log_alpha =
              if is_number(lp) do
                beta * (lp_prop - lp)
              else
                -1.0e10
              end

            {u, rng} = :rand.uniform_s(rng)

            if :math.log(u) < log_alpha do
              {{q_prop, lp_prop}, {accepted + 1, rng}}
            else
              {{q, lp}, {accepted, rng}}
            end
          end)

        {new_q, new_lp} = Enum.unzip(new_pairs)
        {new_q, new_lp, acc_count + step_acc, rng}
      end)

    acc_rate = total_accepted / max(n * num_steps, 1)
    {particles, logps, acc_rate, rng}
  end

  defp build_trace(particles, pm) do
    stacked = Nx.stack(particles)

    Map.new(pm.entries, fn entry ->
      sliced = Nx.slice_along_axis(stacked, entry.offset, entry.length, axis: 1)
      num = elem(Nx.shape(stacked), 0)
      target_shape = Tuple.insert_at(entry.shape, 0, num)
      reshaped = Nx.reshape(sliced, target_shape)
      transformed = Transform.apply(entry.transform, reshaped)
      {entry.id, transformed}
    end)
  end
end
