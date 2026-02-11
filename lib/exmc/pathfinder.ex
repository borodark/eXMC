defmodule Exmc.Pathfinder do
  @moduledoc """
  Pathfinder variational inference (Zhang et al. 2022).

  Traces an L-BFGS path from an initial point toward the mode,
  fitting a diagonal normal approximation at each step. Selects
  the approximation with highest ELBO.

  Returns draws from the best approximation - useful as:
  - Fast approximate posterior
  - Initialization for NUTS
  """

  alias Exmc.{Compiler, Transform}

  @default_opts [
    num_draws: 1000,
    max_iters: 100,
    history_size: 6,
    seed: 0
  ]

  @doc """
  Run Pathfinder on a model IR.

  Returns `{draws, info}` where:
  - `draws`: `%{var_name => Nx.t({num_draws, ...shape})}` in constrained space
  - `info`: `%{elbo:, mu:, sigma:, num_iters:}`
  """
  def fit(ir, opts \\ []) do
    opts = Keyword.merge(@default_opts, opts)
    {vag_fn, pm} = Compiler.value_and_grad(ir)

    if pm.size == 0 do
      {%{}, %{elbo: 0.0, mu: nil, sigma: nil, num_iters: 0}}
    else
      d = pm.size
      seed = opts[:seed]
      rng = :rand.seed_s(:exsss, seed)

      {path, grads} = lbfgs_path(vag_fn, d, opts[:max_iters], opts[:history_size], rng)
      approximations = fit_approximations(path, grads, vag_fn, d)
      best = Enum.max_by(approximations, & &1.elbo)
      {draws_flat, _rng} = draw_from_normal(best.mu, best.sigma, opts[:num_draws], rng)
      trace = build_trace(draws_flat, pm)

      info = %{
        elbo: best.elbo,
        mu: best.mu,
        sigma: best.sigma,
        num_iters: length(path)
      }

      {trace, info}
    end
  end

  defp lbfgs_path(vag_fn, d, max_iters, history_size, rng) do
    {init_vals, _rng} =
      Enum.map_reduce(1..d, rng, fn _, rng ->
        {v, rng} = :rand.normal_s(rng)
        {v * 0.1, rng}
      end)

    q0 = Nx.tensor(init_vals, type: :f64)
    {_logp0, grad0} = vag_fn.(q0)

    state = %{
      q: q0,
      grad: grad0,
      s_list: [],
      y_list: [],
      path: [q0],
      grads: [grad0]
    }

    result =
      Enum.reduce_while(1..max_iters, state, fn _i, state ->
        direction = lbfgs_direction(state.grad, state.s_list, state.y_list)
        alpha = 0.01
        q_new = Nx.add(state.q, Nx.multiply(Nx.tensor(alpha, type: :f64), direction))
        {logp_new, grad_new} = vag_fn.(q_new)
        logp_val = Nx.to_number(logp_new)

        if not is_number(logp_val) do
          {:halt, state}
        else
          s = Nx.subtract(q_new, state.q)
          y = Nx.subtract(grad_new, state.grad)
          ys = Nx.to_number(Nx.dot(y, s))

          {s_list, y_list} =
            if ys > 1.0e-10 do
              s_new = [s | Enum.take(state.s_list, history_size - 1)]
              y_new = [y | Enum.take(state.y_list, history_size - 1)]
              {s_new, y_new}
            else
              {state.s_list, state.y_list}
            end

          new_state = %{
            q: q_new,
            grad: grad_new,
            s_list: s_list,
            y_list: y_list,
            path: [q_new | state.path],
            grads: [grad_new | state.grads]
          }

          {:cont, new_state}
        end
      end)

    {Enum.reverse(result.path), Enum.reverse(result.grads)}
  end

  defp lbfgs_direction(grad, [], []), do: grad

  defp lbfgs_direction(grad, s_list, y_list) do
    pairs = Enum.zip(s_list, y_list)

    rhos =
      Enum.map(pairs, fn {s, y} ->
        ys = Nx.to_number(Nx.dot(y, s))
        if ys > 0, do: 1.0 / ys, else: 0.0
      end)

    {q, alphas} =
      Enum.zip(pairs, rhos)
      |> Enum.reduce({grad, []}, fn {{s, _y}, rho}, {q, alphas} ->
        alpha_val = rho * Nx.to_number(Nx.dot(s, q))
        y_curr = Enum.at(y_list, length(alphas))
        q = Nx.subtract(q, Nx.multiply(Nx.tensor(alpha_val, type: :f64), y_curr))
        {q, [alpha_val | alphas]}
      end)

    alphas = Enum.reverse(alphas)

    {s0, y0} = {hd(s_list), hd(y_list)}
    gamma = Nx.to_number(Nx.dot(s0, y0)) / max(Nx.to_number(Nx.dot(y0, y0)), 1.0e-10)
    r = Nx.multiply(Nx.tensor(gamma, type: :f64), q)

    {r, _} =
      Enum.zip(pairs, rhos)
      |> Enum.zip(alphas)
      |> Enum.reverse()
      |> Enum.reduce({r, 0}, fn {{{s, y}, rho}, alpha_val}, {r, _} ->
        beta = rho * Nx.to_number(Nx.dot(y, r))
        r = Nx.add(r, Nx.multiply(Nx.tensor(alpha_val - beta, type: :f64), s))
        {r, 0}
      end)

    r
  end

  defp fit_approximations(path, grads, vag_fn, d) do
    Enum.zip(path, grads)
    |> Enum.map(fn {q, _grad} ->
      logp = Nx.to_number(elem(vag_fn.(q), 0))
      grad_at_q = elem(vag_fn.(q), 1)
      grad_abs = Nx.abs(grad_at_q)
      sigma = Nx.rsqrt(Nx.add(grad_abs, Nx.tensor(1.0e-6, type: :f64)))

      entropy =
        0.5 * d * (1.0 + :math.log(2.0 * :math.pi())) +
          Nx.to_number(Nx.sum(Nx.log(sigma)))

      elbo = logp + entropy
      %{mu: q, sigma: sigma, elbo: elbo}
    end)
  end

  defp draw_from_normal(mu, sigma, n, rng) do
    d = elem(Nx.shape(mu), 0)

    {draws, rng} =
      Enum.map_reduce(1..n, rng, fn _, rng ->
        {z_vals, rng} =
          Enum.map_reduce(1..d, rng, fn _, rng ->
            {z, rng} = :rand.normal_s(rng)
            {z, rng}
          end)

        z = Nx.tensor(z_vals, type: :f64)
        draw = Nx.add(mu, Nx.multiply(sigma, z))
        {draw, rng}
      end)

    {draws, rng}
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
