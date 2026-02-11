defmodule Exmc.ADVI do
  @moduledoc """
  Automatic Differentiation Variational Inference (Kucukelbir et al. 2017).

  Fits a mean-field (diagonal) normal approximation q(z) = N(mu, diag(sigma^2))
  in unconstrained space by maximizing the ELBO via stochastic gradient ascent.
  """

  alias Exmc.{Compiler, Transform}

  @default_opts [
    num_draws: 1000,
    max_iters: 10_000,
    learning_rate: 0.01,
    num_mc_samples: 1,
    seed: 0,
    convergence_tol: 1.0e-4,
    window_size: 100
  ]

  def fit(ir, opts \\ []) do
    opts = Keyword.merge(@default_opts, opts)
    {vag_fn, pm} = Compiler.value_and_grad(ir)

    if pm.size == 0 do
      {%{}, %{elbo_history: [], mu: nil, log_sigma: nil, converged: true, num_iters: 0}}
    else
      d = pm.size
      rng = :rand.seed_s(:exsss, opts[:seed])

      mu = Nx.broadcast(Nx.tensor(0.0, type: :f64), {d})
      log_sigma = Nx.broadcast(Nx.tensor(-1.0, type: :f64), {d})

      {mu, log_sigma, elbo_history, rng, num_iters, converged} =
        advi_loop(vag_fn, mu, log_sigma, d, rng, opts)

      {draws_flat, _rng} = draw_samples(mu, log_sigma, opts[:num_draws], rng)
      trace = build_trace(draws_flat, pm)

      info = %{
        elbo_history: elbo_history,
        mu: mu,
        log_sigma: log_sigma,
        converged: converged,
        num_iters: num_iters
      }

      {trace, info}
    end
  end

  defp advi_loop(vag_fn, mu, log_sigma, d, rng, opts) do
    lr = opts[:learning_rate]
    max_iters = opts[:max_iters]
    n_mc = opts[:num_mc_samples]
    tol = opts[:convergence_tol]
    window = opts[:window_size]

    state = %{mu: mu, log_sigma: log_sigma, rng: rng, elbos: [], iter: 0, converged: false}

    result =
      Enum.reduce_while(1..max_iters, state, fn i, state ->
        sigma = Nx.exp(state.log_sigma)

        {eps_list, rng} = sample_eps(d, n_mc, state.rng)

        {elbo, grad_mu, grad_log_sigma} =
          compute_elbo_grads(vag_fn, state.mu, sigma, state.log_sigma, eps_list, d)

        new_mu = Nx.add(state.mu, Nx.multiply(Nx.tensor(lr, type: :f64), grad_mu))

        new_log_sigma =
          Nx.add(state.log_sigma, Nx.multiply(Nx.tensor(lr, type: :f64), grad_log_sigma))

        elbos = [elbo | state.elbos]

        converged =
          if length(elbos) >= window do
            recent = Enum.take(elbos, div(window, 2))
            old = elbos |> Enum.drop(div(window, 2)) |> Enum.take(div(window, 2))
            mean_recent = Enum.sum(recent) / length(recent)
            mean_old = Enum.sum(old) / length(old)
            abs(mean_recent - mean_old) / (abs(mean_old) + 1.0e-8) < tol
          else
            false
          end

        new_state = %{
          state
          | mu: new_mu,
            log_sigma: new_log_sigma,
            rng: rng,
            elbos: elbos,
            iter: i,
            converged: converged
        }

        if converged, do: {:halt, new_state}, else: {:cont, new_state}
      end)

    {result.mu, result.log_sigma, Enum.reverse(result.elbos), result.rng, result.iter,
     result.converged}
  end

  defp sample_eps(d, n_mc, rng) do
    Enum.map_reduce(1..n_mc, rng, fn _, rng ->
      {vals, rng} =
        Enum.map_reduce(1..d, rng, fn _, rng ->
          {v, rng} = :rand.normal_s(rng)
          {v, rng}
        end)

      {Nx.tensor(vals, type: :f64), rng}
    end)
  end

  defp compute_elbo_grads(vag_fn, mu, sigma, log_sigma, eps_list, d) do
    n = length(eps_list)

    results =
      Enum.map(eps_list, fn eps ->
        z = Nx.add(mu, Nx.multiply(sigma, eps))
        {logp_t, grad_t} = vag_fn.(z)
        logp = Nx.to_number(logp_t)

        entropy = Nx.to_number(Nx.sum(log_sigma)) + 0.5 * d * (1.0 + :math.log(2.0 * :math.pi()))

        elbo = if is_number(logp), do: logp + entropy, else: -1.0e10

        grad_mu = grad_t

        grad_log_sigma =
          Nx.add(
            Nx.multiply(Nx.multiply(grad_t, sigma), eps),
            Nx.tensor(1.0, type: :f64)
          )

        {elbo, grad_mu, grad_log_sigma}
      end)

    avg_elbo = Enum.sum(Enum.map(results, &elem(&1, 0))) / n

    avg_grad_mu =
      results
      |> Enum.map(&elem(&1, 1))
      |> Enum.reduce(&Nx.add/2)
      |> Nx.divide(Nx.tensor(n * 1.0, type: :f64))

    avg_grad_ls =
      results
      |> Enum.map(&elem(&1, 2))
      |> Enum.reduce(&Nx.add/2)
      |> Nx.divide(Nx.tensor(n * 1.0, type: :f64))

    {avg_elbo, avg_grad_mu, avg_grad_ls}
  end

  defp draw_samples(mu, log_sigma, n, rng) do
    sigma = Nx.exp(log_sigma)
    d = elem(Nx.shape(mu), 0)

    Enum.map_reduce(1..n, rng, fn _, rng ->
      {vals, rng} =
        Enum.map_reduce(1..d, rng, fn _, rng ->
          {v, rng} = :rand.normal_s(rng)
          {v, rng}
        end)

      eps = Nx.tensor(vals, type: :f64)
      z = Nx.add(mu, Nx.multiply(sigma, eps))
      {z, rng}
    end)
  end

  defp build_trace(draws_flat, pm) do
    stacked = Nx.stack(draws_flat)

    Map.new(pm.entries, fn entry ->
      sliced = Nx.slice_along_axis(stacked, entry.offset, entry.length, axis: 1)
      num_draws = elem(Nx.shape(stacked), 0)
      target_shape = Tuple.insert_at(entry.shape, 0, num_draws)
      reshaped = Nx.reshape(sliced, target_shape)
      transformed = Transform.apply(entry.transform, reshaped)
      {entry.id, transformed}
    end)
  end
end
