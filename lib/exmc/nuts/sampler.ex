defmodule Exmc.NUTS.Sampler do
  @moduledoc """
  Top-level NUTS sampler orchestrator.

  Operates in unconstrained space, returns constrained-space trace.
  Uses Stan-style three-phase warmup with doubling windows.

  PRNG strategy: uses `:rand` (Erlang) for all scalar random decisions and
  `Nx.Random` only for momentum sampling (normal draws). This avoids the
  expensive Nx.Random.split/uniform overhead with BinaryBackend.
  """

  alias Exmc.{Compiler, PointMap, Transform}
  alias Exmc.NUTS.{Leapfrog, MassMatrix, StepSize, Tree}

  @default_opts [
    num_warmup: 1000,
    num_samples: 1000,
    max_tree_depth: 10,
    target_accept: 0.8,
    seed: 0
  ]

  @doc """
  Sample from a probabilistic model.

  Returns `{trace, stats}` where:
  - `trace`: `%{var_name => Nx.t() of shape {num_samples, ...var_shape}}` in constrained space
  - `stats`: `%{step_size:, inv_mass_diag:, divergences:, num_warmup:, num_samples:, sample_stats:}`
    where `sample_stats` is a list of per-step maps with keys `:tree_depth`, `:n_steps`, `:divergent`, `:accept_prob`
  """
  def sample(ir, init_values \\ %{}, opts \\ []) do
    compiled = Compiler.compile_for_sampling(ir)
    sample_from_compiled(compiled, init_values, opts)
  end

  # Run sampling from pre-compiled artifacts. Used by sample_chains to compile once.
  defp sample_from_compiled({vag_fn, step_fn, pm, ncp_info}, init_values, opts) do
    opts = Keyword.merge(@default_opts, opts)
    num_warmup = opts[:num_warmup]
    num_samples = opts[:num_samples]
    max_tree_depth = opts[:max_tree_depth]
    target_accept = opts[:target_accept]
    seed = opts[:seed]

    if pm.size == 0 do
      empty_trace = %{}

      stats = %{
        step_size: 0.0,
        inv_mass_diag: Nx.broadcast(Nx.tensor(0.0, type: :f64), {0}),
        divergences: 0,
        num_warmup: num_warmup,
        num_samples: num_samples
      }

      {empty_trace, stats}
    else
      rng = :rand.seed_s(:exsss, seed)
      d = pm.size

      # Initialize position
      {q, rng} = init_position(pm, init_values, d, rng)

      # Evaluate initial logp and gradient
      {logp, grad} = vag_fn.(q)

      # Initialize mass matrix (identity)
      inv_mass_diag = Nx.broadcast(Nx.tensor(1.0, type: :f64), {d})

      # Find reasonable initial step size
      {epsilon, rng} = find_reasonable_epsilon_with_rng(step_fn, q, logp, grad, inv_mass_diag, rng)

      # Run warmup
      state = %{
        q: q,
        logp: logp,
        grad: grad,
        rng: rng,
        divergences: 0
      }

      {state, epsilon, inv_mass_diag} =
        run_warmup(step_fn, state, epsilon, inv_mass_diag, d, num_warmup, max_tree_depth, target_accept)

      # Freeze step size
      epsilon_final = epsilon

      # Run sampling
      {draws, sample_stats, state} =
        run_sampling(step_fn, state, epsilon_final, inv_mass_diag, num_samples, max_tree_depth)

      # Build trace (with NCP reconstruction if applicable)
      trace = build_trace(draws, pm, ncp_info)

      stats = %{
        step_size: epsilon_final,
        inv_mass_diag: inv_mass_diag,
        divergences: state.divergences,
        num_warmup: num_warmup,
        num_samples: num_samples,
        sample_stats: sample_stats
      }

      {trace, stats}
    end
  end

  # --- Position initialization ---

  defp init_position(_pm, init_values, d, rng) when map_size(init_values) == 0 do
    # Random initialization near zero using :rand for normal draws
    {values, rng} =
      Enum.map_reduce(1..d, rng, fn _i, rng ->
        {val, rng} = :rand.normal_s(rng)
        {val * 0.1, rng}
      end)

    q = Nx.tensor(values, type: :f64)
    {q, rng}
  end

  defp init_position(pm, init_values, _d, rng) do
    unconstrained = PointMap.to_unconstrained(init_values, pm)
    q = PointMap.pack(unconstrained, pm)
    {q, rng}
  end

  # --- Momentum sampling using :rand (fast, no Nx.Random overhead) ---

  defp sample_momentum_fast(rng, inv_mass_diag) do
    inv_mass_list = Nx.to_flat_list(inv_mass_diag)

    {p_values, rng} =
      Enum.map_reduce(inv_mass_list, rng, fn inv_m, rng ->
        {z, rng} = :rand.normal_s(rng)
        p = z / :math.sqrt(inv_m)
        {p, rng}
      end)

    p = Nx.tensor(p_values, type: :f64)
    {p, rng}
  end

  # --- Step size finding using :rand ---

  defp find_reasonable_epsilon_with_rng(step_fn, q, logp, grad, inv_mass_diag, rng) do
    {p, rng} = sample_momentum_fast(rng, inv_mass_diag)
    joint_logp_0 = Leapfrog.joint_logp(logp, p, inv_mass_diag) |> Nx.to_number()

    epsilon = 1.0
    {_q_new, p_new, logp_new, _grad_new} = step_fn.(q, p, grad, epsilon, inv_mass_diag)

    joint_logp_new = Leapfrog.joint_logp(logp_new, p_new, inv_mass_diag) |> Nx.to_number()

    log_accept =
      if is_number(joint_logp_0) and is_number(joint_logp_new) do
        joint_logp_new - joint_logp_0
      else
        -1000.0
      end

    direction = if log_accept > :math.log(0.5), do: 1.0, else: -1.0
    epsilon = search_epsilon(step_fn, q, p, grad, inv_mass_diag, epsilon, direction, joint_logp_0, 0)
    {epsilon, rng}
  end

  defp search_epsilon(_step_fn, _q, _p, _grad, _inv_mass_diag, epsilon, _direction, _joint_logp_0, count)
       when count >= 100 do
    max(epsilon, 1.0e-10)
  end

  defp search_epsilon(step_fn, q, p, grad, inv_mass_diag, epsilon, direction, joint_logp_0, count) do
    factor = :math.pow(2.0, direction)
    new_epsilon = epsilon * factor

    {_q_new, p_new, logp_new, _grad_new} = step_fn.(q, p, grad, new_epsilon, inv_mass_diag)

    joint_logp_new = Leapfrog.joint_logp(logp_new, p_new, inv_mass_diag) |> Nx.to_number()

    log_accept =
      if is_number(joint_logp_0) and is_number(joint_logp_new) do
        joint_logp_new - joint_logp_0
      else
        -1000.0
      end

    crossed =
      if direction > 0 do
        log_accept < :math.log(0.5)
      else
        log_accept > :math.log(0.5)
      end

    if crossed or not is_finite(log_accept) do
      max(new_epsilon, 1.0e-10)
    else
      search_epsilon(step_fn, q, p, grad, inv_mass_diag, new_epsilon, direction, joint_logp_0, count + 1)
    end
  end

  defp is_finite(x) when is_float(x), do: x != :infinity and x != :neg_infinity and x == x
  defp is_finite(_), do: false

  # --- Warmup ---

  defp run_warmup(step_fn, state, epsilon, inv_mass_diag, d, num_warmup, max_tree_depth, target_accept) do
    if num_warmup == 0 do
      {state, epsilon, inv_mass_diag}
    else
      # Three-phase schedule
      init_buffer = min(75, div(num_warmup, 3))
      term_buffer = min(50, div(num_warmup, 5))
      adapt_end = num_warmup - term_buffer

      # Phase I: step size only (0..init_buffer-1)
      da_state = StepSize.init(epsilon, target_accept)

      {state, da_state} =
        run_phase(step_fn, state, epsilon, inv_mass_diag, max_tree_depth, da_state, 0, init_buffer)

      epsilon = current_epsilon(da_state)

      if adapt_end <= init_buffer do
        epsilon_final = StepSize.finalize(da_state)
        {state, epsilon_final, inv_mass_diag}
      else
        # Phase II: step size + mass matrix with doubling windows
        {state, epsilon, inv_mass_diag, _} =
          run_phase_ii(
            step_fn, state, epsilon, inv_mass_diag, d,
            max_tree_depth, target_accept, init_buffer, adapt_end
          )

        # Phase III: step size only (adapt_end..num_warmup-1)
        da_state = StepSize.init(epsilon, target_accept)

        {state, da_state} =
          run_phase(step_fn, state, epsilon, inv_mass_diag, max_tree_depth, da_state, adapt_end, num_warmup)

        epsilon_final = StepSize.finalize(da_state)
        {state, epsilon_final, inv_mass_diag}
      end
    end
  end

  defp run_phase(step_fn, state, _epsilon, inv_mass_diag, max_tree_depth, da_state, from, to) do
    if from >= to do
      {state, da_state}
    else
      Enum.reduce(from..(to - 1)//1, {state, da_state}, fn _i, {state, da_state} ->
        # Use the DA's current epsilon for each step
        eps = current_epsilon(da_state)
        {state, accept_stat} = nuts_step(step_fn, state, eps, inv_mass_diag, max_tree_depth)
        da_state = StepSize.update(da_state, accept_stat)
        {state, da_state}
      end)
    end
  end

  defp run_phase_ii(step_fn, state, epsilon, inv_mass_diag, d, max_tree_depth, target_accept, from, to) do
    windows = build_windows(from, to)

    Enum.reduce(windows, {state, epsilon, inv_mass_diag, nil}, fn {win_start, win_end}, {state, epsilon, inv_mass_diag, _} ->
      welford = MassMatrix.init(d)
      da_state = StepSize.init(epsilon, target_accept)

      {state, da_state, welford} =
        Enum.reduce(win_start..(win_end - 1)//1, {state, da_state, welford}, fn _i, {state, da_state, welford} ->
          eps = current_epsilon(da_state)
          {state, accept_stat} = nuts_step(step_fn, state, eps, inv_mass_diag, max_tree_depth)
          da_state = StepSize.update(da_state, accept_stat)
          welford = MassMatrix.update(welford, state.q)
          {state, da_state, welford}
        end)

      # Finalize window
      inv_mass_diag = MassMatrix.finalize(welford)

      # Use current DA smoothed epsilon as starting point for next window
      epsilon = StepSize.finalize(da_state)

      {state, epsilon, inv_mass_diag, nil}
    end)
    |> then(fn {state, epsilon, inv_mass_diag, _} ->
      {state, epsilon, inv_mass_diag, nil}
    end)
  end

  defp build_windows(from, to) do
    total = to - from
    if total <= 0 do
      []
    else
      do_build_windows(from, to, 25, [])
      |> Enum.reverse()
    end
  end

  defp do_build_windows(current, to, _window_size, acc) when current >= to do
    acc
  end

  defp do_build_windows(current, to, window_size, acc) do
    remaining = to - current
    actual_size = if remaining <= window_size * 1.5, do: remaining, else: window_size
    win_end = current + actual_size
    acc = [{current, win_end} | acc]
    do_build_windows(win_end, to, window_size * 2, acc)
  end

  defp current_epsilon(da_state) do
    :math.exp(da_state.log_epsilon)
  end

  # --- NUTS step ---

  defp nuts_step(step_fn, state, epsilon, inv_mass_diag, max_tree_depth) do
    {p, rng} = sample_momentum_fast(state.rng, inv_mass_diag)
    joint_logp_0 = Leapfrog.joint_logp(state.logp, p, inv_mass_diag)

    result =
      Tree.build(
        step_fn, state.q, p, state.logp, state.grad,
        epsilon, inv_mass_diag, max_tree_depth, rng, joint_logp_0
      )

    accept_stat =
      if result.n_steps > 0 do
        result.accept_sum / result.n_steps
      else
        0.0
      end

    # Advance rng to avoid correlation between steps
    {_, rng} = :rand.uniform_s(rng)

    divergences = if result.divergent, do: state.divergences + 1, else: state.divergences

    new_state = %{
      q: result.q,
      logp: result.logp,
      grad: result.grad,
      rng: rng,
      divergences: divergences
    }

    {new_state, accept_stat}
  end

  # NUTS step returning additional stats for diagnostics
  defp nuts_step_with_stats(step_fn, state, epsilon, inv_mass_diag, max_tree_depth) do
    {p, rng} = sample_momentum_fast(state.rng, inv_mass_diag)
    joint_logp_0 = Leapfrog.joint_logp(state.logp, p, inv_mass_diag)

    result =
      Tree.build(
        step_fn, state.q, p, state.logp, state.grad,
        epsilon, inv_mass_diag, max_tree_depth, rng, joint_logp_0
      )

    accept_stat =
      if result.n_steps > 0 do
        result.accept_sum / result.n_steps
      else
        0.0
      end

    {_, rng} = :rand.uniform_s(rng)
    divergences = if result.divergent, do: state.divergences + 1, else: state.divergences

    new_state = %{
      q: result.q,
      logp: result.logp,
      grad: result.grad,
      rng: rng,
      divergences: divergences
    }

    step_info = %{
      depth: result.depth,
      n_steps: result.n_steps,
      divergent: result.divergent
    }

    {new_state, accept_stat, step_info}
  end

  # --- Sampling ---

  defp run_sampling(step_fn, state, epsilon, inv_mass_diag, num_samples, max_tree_depth) do
    {draws_reversed, sample_stats_reversed, state} =
      Enum.reduce(1..num_samples, {[], [], state}, fn _i, {draws, stats_acc, state} ->
        {state, accept_stat, step_info} =
          nuts_step_with_stats(step_fn, state, epsilon, inv_mass_diag, max_tree_depth)

        step_stat = %{
          tree_depth: step_info.depth,
          n_steps: step_info.n_steps,
          divergent: step_info.divergent,
          accept_prob: accept_stat
        }

        {[state.q | draws], [step_stat | stats_acc], state}
      end)

    {Enum.reverse(draws_reversed), Enum.reverse(sample_stats_reversed), state}
  end

  @doc """
  Run multiple chains with different seeds.

  Compiles the model once and runs chains in parallel by default.

  Returns `{[traces], [stats]}` where each element corresponds to one chain.

  ## Options

  - `:parallel` — run chains concurrently (default: `true`)
  - `:max_concurrency` — max parallel chains (default: `num_chains`)
  - `:init_values` — initial values for all chains (default: `%{}`)

  Plus all options from `sample/3` (`:num_warmup`, `:num_samples`, `:seed`, etc.).
  """
  def sample_chains(ir, num_chains, opts \\ []) when num_chains >= 1 do
    base_seed = Keyword.get(opts, :seed, 0)
    init_values = Keyword.get(opts, :init_values, %{})
    parallel = Keyword.get(opts, :parallel, true)
    max_concurrency = Keyword.get(opts, :max_concurrency, num_chains)

    # Strip chain-specific opts before passing to sample
    sample_opts = Keyword.drop(opts, [:init_values, :parallel, :max_concurrency])

    # Compile once, share across all chains
    compiled = Compiler.compile_for_sampling(ir)

    chain_opts_list =
      Enum.map(0..(num_chains - 1), fn i ->
        Keyword.put(sample_opts, :seed, base_seed + i * 7919)
      end)

    results =
      if parallel and num_chains > 1 do
        chain_opts_list
        |> Task.async_stream(
          fn chain_opts -> sample_from_compiled(compiled, init_values, chain_opts) end,
          max_concurrency: max_concurrency,
          timeout: :infinity,
          ordered: true
        )
        |> Enum.map(fn {:ok, result} -> result end)
      else
        Enum.map(chain_opts_list, fn chain_opts ->
          sample_from_compiled(compiled, init_values, chain_opts)
        end)
      end

    traces = Enum.map(results, fn {trace, _stats} -> trace end)
    stats = Enum.map(results, fn {_trace, stats} -> stats end)
    {traces, stats}
  end

  # --- Trace building ---

  defp build_trace(draws, pm, ncp_info) do
    stacked = Nx.stack(draws)

    base_trace =
      Map.new(pm.entries, fn entry ->
        sliced = Nx.slice_along_axis(stacked, entry.offset, entry.length, axis: 1)

        num_samples = elem(Nx.shape(stacked), 0)
        target_shape = Tuple.insert_at(entry.shape, 0, num_samples)
        reshaped = Nx.reshape(sliced, target_shape)

        transformed = Transform.apply(entry.transform, reshaped)

        {entry.id, transformed}
      end)

    reconstruct_ncp(base_trace, ncp_info)
  end

  # Reconstruct NCP'd variables: x = mu + sigma * z (in topological order)
  defp reconstruct_ncp(trace, ncp_info) when map_size(ncp_info) == 0, do: trace

  defp reconstruct_ncp(trace, ncp_info) do
    order = ncp_topo_order(ncp_info)

    Enum.reduce(order, trace, fn rv_id, trace ->
      %{mu: mu_src, sigma: sigma_src} = ncp_info[rv_id]
      z = Map.fetch!(trace, rv_id)
      mu = resolve_trace_value(mu_src, trace)
      sigma = resolve_trace_value(sigma_src, trace)
      Map.put(trace, rv_id, Nx.add(mu, Nx.multiply(sigma, z)))
    end)
  end

  defp resolve_trace_value(v, trace) when is_binary(v), do: Map.fetch!(trace, v)
  defp resolve_trace_value(%Nx.Tensor{} = v, _trace), do: v
  defp resolve_trace_value(v, _trace) when is_number(v), do: Nx.tensor(v, type: :f64)

  # Topological sort for NCP entries: process entries whose NCP dependencies are resolved first
  defp ncp_topo_order(ncp_info) do
    ncp_ids = MapSet.new(Map.keys(ncp_info))
    remaining = Map.keys(ncp_info)
    do_ncp_topo(remaining, ncp_info, ncp_ids, MapSet.new(), [])
  end

  defp do_ncp_topo([], _ncp_info, _ncp_ids, _done, acc), do: Enum.reverse(acc)

  defp do_ncp_topo(remaining, ncp_info, ncp_ids, done, acc) do
    ready =
      Enum.filter(remaining, fn id ->
        %{mu: mu, sigma: sigma} = ncp_info[id]
        ncp_dep_resolved?(mu, ncp_ids, done) and ncp_dep_resolved?(sigma, ncp_ids, done)
      end)

    if ready == [] do
      Enum.reverse(acc) ++ remaining
    else
      new_done = Enum.reduce(ready, done, &MapSet.put(&2, &1))
      do_ncp_topo(remaining -- ready, ncp_info, ncp_ids, new_done, Enum.reverse(Enum.sort(ready)) ++ acc)
    end
  end

  defp ncp_dep_resolved?(src, ncp_ids, done) when is_binary(src) do
    not MapSet.member?(ncp_ids, src) or MapSet.member?(done, src)
  end

  defp ncp_dep_resolved?(_src, _ncp_ids, _done), do: true
end
