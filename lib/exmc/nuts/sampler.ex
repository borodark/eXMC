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
    seed: 0,
    supervised: false
  ]

  @doc """
  Sample from a probabilistic model.

  Returns `{trace, stats}` where:
  - `trace`: `%{var_name => Nx.t() of shape {num_samples, ...var_shape}}` in constrained space
  - `stats`: `%{step_size:, inv_mass_diag:, divergences:, num_warmup:, num_samples:, sample_stats:}`
    where `sample_stats` is a list of per-step maps with keys `:tree_depth`, `:n_steps`, `:divergent`, `:accept_prob`
  """
  def sample(ir, init_values \\ %{}, opts \\ []) do
    compile_opts = Keyword.take(opts, [:ncp, :device])
    compiled = Compiler.compile_for_sampling(ir, compile_opts)
    sample_from_compiled(compiled, init_values, opts)
  end

  @doc """
  Pre-compile a model for sampling.

  Returns compiled artifacts that can be passed to `sample_compiled/3`
  or `sample_chains_compiled/4` to avoid repeated JIT compilation.
  """
  def compile(ir, opts \\ []) do
    compile_opts = if is_list(opts), do: Keyword.take(opts, [:ncp, :device]), else: []
    Compiler.compile_for_sampling(ir, compile_opts)
  end

  @doc """
  Sample using pre-compiled artifacts from `compile/1`.
  Same interface as `sample/3` but skips compilation.
  """
  def sample_compiled(compiled, init_values \\ %{}, opts \\ []) do
    sample_from_compiled(compiled, init_values, opts)
  end

  @doc """
  Sample using pre-compiled artifacts and pre-computed tuning parameters.
  Skips warmup entirely — uses the provided step size and mass matrix directly.

  `tuning` is a map with:
  - `:epsilon` — adapted step size (float)
  - `:inv_mass` — adapted inverse mass matrix (Nx tensor, rank 1 or 2)
  - `:chol_cov` — Cholesky factor for dense mass (Nx tensor or nil)

  Used by distributed sampling to run chains with shared warmup parameters.
  """
  def sample_compiled_tuned(compiled, tuning, init_values \\ %{}, opts \\ []) do
    sample_from_compiled_tuned(compiled, tuning, init_values, opts)
  end

  @doc """
  Run multiple chains using pre-compiled artifacts from `compile/1`.
  Same interface as `sample_chains/3` but skips compilation.
  """
  def sample_chains_compiled(compiled, num_chains, opts \\ []) when num_chains >= 1 do
    vectorized = Keyword.get(opts, :vectorized, num_chains > 1)

    if vectorized and num_chains > 1 do
      sample_chains_vectorized_compiled(compiled, num_chains, opts)
    else
      sample_chains_compiled_parallel(compiled, num_chains, opts)
    end
  end

  defp sample_chains_compiled_parallel(compiled, num_chains, opts) do
    base_seed = Keyword.get(opts, :seed, 0)
    init_values = Keyword.get(opts, :init_values, %{})
    parallel = Keyword.get(opts, :parallel, true)
    max_concurrency = Keyword.get(opts, :max_concurrency, num_chains)
    sample_opts = Keyword.drop(opts, [:init_values, :parallel, :max_concurrency, :vectorized])

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

  # Run sampling from pre-compiled artifacts.
  # Accepts both 4-tuple (legacy) and 5-tuple (with multi_step_fn) for backwards compat.
  defp sample_from_compiled({vag_fn, step_fn, pm, ncp_info}, init_values, opts) do
    sample_from_compiled({vag_fn, step_fn, pm, ncp_info, nil}, init_values, opts)
  end

  defp sample_from_compiled({vag_fn, step_fn, pm, ncp_info, multi_step_fn}, init_values, opts) do
    opts = Keyword.merge(@default_opts, opts)
    num_warmup = opts[:num_warmup]
    num_samples = opts[:num_samples]
    max_tree_depth = opts[:max_tree_depth]
    target_accept = opts[:target_accept]
    seed = opts[:seed]
    supervised = opts[:supervised]

    # Set supervised mode for Tree.build (process dictionary, zero-overhead when false)
    if supervised, do: Process.put(:exmc_supervised, supervised)

    # Reset depth tracker for hybrid dispatch (full-tree NIF vs speculative)
    Process.put(:exmc_max_tree_depth_seen, 0)

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
      use_dense = Keyword.get(opts, :dense_mass, false)

      # Initialize position (invert NCP so user's constrained values map correctly)
      {q, rng} = init_position(pm, init_values, d, rng, ncp_info)

      # Evaluate initial logp and gradient (copy from EXLA.Backend to BinaryBackend
      # so all downstream sampler/tree arithmetic stays on fast BinaryBackend)
      {logp, grad} = vag_fn.(q)
      logp = Nx.backend_copy(logp, Nx.BinaryBackend)
      grad = Nx.backend_copy(grad, Nx.BinaryBackend)

      # Initialize mass matrix (identity, diagonal)
      inv_mass_diag = Nx.broadcast(Nx.tensor(1.0, type: :f64, backend: Nx.BinaryBackend), {d})

      # Use generic step_fn for dense mode (dispatches on mass shape)
      active_step_fn = if use_dense, do: build_generic_step_fn(vag_fn), else: step_fn

      # Find reasonable initial step size (always with diagonal identity)
      {epsilon, rng} =
        find_reasonable_epsilon_with_rng(active_step_fn, q, logp, grad, inv_mass_diag, rng)

      # Run warmup
      state = %{
        q: q,
        logp: logp,
        grad: grad,
        rng: rng,
        divergences: 0,
        recoveries: 0
      }

      # Use batched leapfrog for diagonal mass (both warmup and sampling)
      active_multi = if use_dense, do: nil, else: multi_step_fn

      {state, epsilon, inv_mass, chol_cov} =
        run_warmup(
          active_step_fn,
          state,
          epsilon,
          inv_mass_diag,
          d,
          num_warmup,
          max_tree_depth,
          target_accept,
          use_dense,
          active_multi
        )

      # Freeze step size
      epsilon_final = epsilon

      # Run sampling
      {draws, sample_stats, state} =
        run_sampling(
          active_step_fn,
          state,
          epsilon_final,
          inv_mass,
          num_samples,
          max_tree_depth,
          chol_cov,
          active_multi
        )

      # Build trace (with NCP reconstruction if applicable)
      trace = build_trace(draws, pm, ncp_info)

      # Extract diagonal for stats (consistent API)
      inv_mass_diag_out =
        case Nx.rank(inv_mass) do
          1 -> inv_mass
          2 -> Nx.take_diagonal(inv_mass)
        end

      stats = %{
        step_size: epsilon_final,
        inv_mass_diag: inv_mass_diag_out,
        divergences: state.divergences,
        recoveries: Map.get(state, :recoveries, 0),
        num_warmup: num_warmup,
        num_samples: num_samples,
        sample_stats: sample_stats
      }

      # Clean up supervised mode
      Process.delete(:exmc_supervised)

      {trace, stats}
    end
  end

  # Run sampling with pre-computed tuning (no warmup).
  defp sample_from_compiled_tuned(compiled, tuning, init_values, opts) do
    {vag_fn, step_fn, pm, ncp_info, multi_step_fn} =
      case compiled do
        {v, s, p, n, m} -> {v, s, p, n, m}
        {v, s, p, n} -> {v, s, p, n, nil}
      end

    opts = Keyword.merge(@default_opts, opts)
    num_samples = opts[:num_samples]
    max_tree_depth = opts[:max_tree_depth]
    seed = opts[:seed]

    epsilon = tuning.epsilon
    inv_mass = tuning.inv_mass
    chol_cov = Map.get(tuning, :chol_cov)

    if pm.size == 0 do
      empty_trace = %{}

      stats = %{
        step_size: 0.0,
        inv_mass_diag: Nx.broadcast(Nx.tensor(0.0, type: :f64), {0}),
        divergences: 0,
        num_warmup: 0,
        num_samples: num_samples,
        sample_stats: []
      }

      {empty_trace, stats}
    else
      rng = :rand.seed_s(:exsss, seed)
      d = pm.size
      use_dense = chol_cov != nil

      {q, rng} = init_position(pm, init_values, d, rng, ncp_info)
      {logp, grad} = vag_fn.(q)
      logp = Nx.backend_copy(logp, Nx.BinaryBackend)
      grad = Nx.backend_copy(grad, Nx.BinaryBackend)

      active_step_fn = if use_dense, do: build_generic_step_fn(vag_fn), else: step_fn
      active_multi = if use_dense, do: nil, else: multi_step_fn

      state = %{q: q, logp: logp, grad: grad, rng: rng, divergences: 0}

      {draws, sample_stats, state} =
        run_sampling(
          active_step_fn,
          state,
          epsilon,
          inv_mass,
          num_samples,
          max_tree_depth,
          chol_cov,
          active_multi
        )

      trace = build_trace(draws, pm, ncp_info)

      inv_mass_diag_out =
        case Nx.rank(inv_mass) do
          1 -> inv_mass
          2 -> Nx.take_diagonal(inv_mass)
        end

      stats = %{
        step_size: epsilon,
        inv_mass_diag: inv_mass_diag_out,
        divergences: state.divergences,
        num_warmup: 0,
        num_samples: num_samples,
        sample_stats: sample_stats
      }

      {trace, stats}
    end
  end

  # --- Position initialization ---

  defp init_position(_pm, init_values, d, rng, _ncp_info) when map_size(init_values) == 0 do
    # Random initialization near zero using :rand for normal draws
    {values, rng} =
      Enum.map_reduce(1..d, rng, fn _i, rng ->
        {val, rng} = :rand.normal_s(rng)
        {val * 0.1, rng}
      end)

    q = Nx.tensor(values, type: :f64, backend: Nx.BinaryBackend)
    {q, rng}
  end

  defp init_position(pm, init_values, _d, rng, ncp_info) do
    init_values = invert_ncp_init(init_values, ncp_info)
    unconstrained = PointMap.to_unconstrained(init_values, pm)
    q = PointMap.pack(unconstrained, pm)
    {q, rng}
  end

  # Invert NCP reparameterization on init values.
  # User provides constrained values (e.g. alpha=4.0 the group mean),
  # but after NCP rewrite the free variable is alpha_raw ~ N(0,1).
  # Inverse: z = (x - mu) / sigma
  defp invert_ncp_init(init_values, ncp_info) when map_size(ncp_info) == 0, do: init_values

  defp invert_ncp_init(init_values, ncp_info) do
    # Compute all raw values using original constrained init_values
    raw_updates =
      Enum.reduce(ncp_info, %{}, fn {id, %{mu: mu_src, sigma: sigma_src}}, acc ->
        case Map.get(init_values, id) do
          nil ->
            acc

          value ->
            mu = resolve_init_source(mu_src, init_values)
            sigma = resolve_init_source(sigma_src, init_values)
            z = Nx.divide(Nx.subtract(value, mu), sigma)
            Map.put(acc, id, z)
        end
      end)

    Map.merge(init_values, raw_updates)
  end

  defp resolve_init_source(src, init_values) when is_binary(src) do
    Map.fetch!(init_values, src)
  end

  defp resolve_init_source(%Nx.Tensor{} = src, _init_values), do: src
  defp resolve_init_source(src, _init_values) when is_number(src), do: Nx.tensor(src)

  # --- Momentum sampling using :rand (fast, no Nx.Random overhead) ---

  # Fast momentum sampling with pre-cached inv_mass list
  defp sample_momentum_fast(rng, inv_mass_list) when is_list(inv_mass_list) do
    {p_values, rng} =
      Enum.map_reduce(inv_mass_list, rng, fn inv_m, rng ->
        {z, rng} = :rand.normal_s(rng)
        p = z / :math.sqrt(inv_m)
        {p, rng}
      end)

    p = Nx.tensor(p_values, type: :f64, backend: Nx.BinaryBackend)
    {p, rng}
  end

  # Fallback: extract list from tensor (used during init/warmup)
  defp sample_momentum_fast(rng, inv_mass_diag) do
    sample_momentum_fast(rng, Nx.to_flat_list(inv_mass_diag))
  end

  # Dense momentum sampling: p ~ N(0, M) where M = Cov^{-1}.
  # With Cov = L @ L^T (chol_cov = L), sample p = L^{-T} @ z.
  defp sample_momentum_dense(rng, chol_cov) do
    {d, _} = Nx.shape(chol_cov)

    {z_values, rng} =
      Enum.map_reduce(1..d, rng, fn _i, rng ->
        {z, rng} = :rand.normal_s(rng)
        {z, rng}
      end)

    z = Nx.tensor(z_values, type: :f64, backend: Nx.BinaryBackend)
    # Solve L^T @ p = z (L^T is upper triangular)
    z_col = Nx.reshape(z, {d, 1})
    p_col = Nx.LinAlg.triangular_solve(Nx.transpose(chol_cov), z_col, lower: false)
    p = Nx.reshape(p_col, {d})
    {p, rng}
  end

  # Dispatch momentum sampling based on mass type
  defp sample_momentum_for(rng, inv_mass, nil), do: sample_momentum_fast(rng, inv_mass)
  defp sample_momentum_for(rng, _inv_mass, chol_cov), do: sample_momentum_dense(rng, chol_cov)

  # Generic step function that works with both diagonal and dense mass matrices.
  # Not JIT'd, but vag_fn inside IS JIT'd. For d=5-8, tensor overhead is negligible.
  # Returns 5-tuple: {q, p, logp, grad, joint_logp} to match compiled step_fn.
  defp build_generic_step_fn(vag_fn) do
    fn q, p, grad, epsilon, inv_mass ->
      eps = Nx.tensor(epsilon, type: :f64, backend: Nx.BinaryBackend)
      half_eps = Nx.divide(eps, Nx.tensor(2.0, type: :f64, backend: Nx.BinaryBackend))
      p_half = Nx.add(p, Nx.multiply(half_eps, grad))
      q_new = Nx.add(q, Nx.multiply(eps, Leapfrog.mass_times_p(inv_mass, p_half)))
      {logp_new, grad_new} = vag_fn.(q_new)
      p_new = Nx.add(p_half, Nx.multiply(half_eps, grad_new))
      joint_logp = Leapfrog.joint_logp(logp_new, p_new, inv_mass)
      {q_new, p_new, logp_new, grad_new, joint_logp}
    end
  end

  # --- Step size finding using :rand ---

  defp find_reasonable_epsilon_with_rng(step_fn, q, logp, grad, inv_mass_diag, rng) do
    {p, rng} = sample_momentum_fast(rng, inv_mass_diag)
    joint_logp_0 = Leapfrog.joint_logp(logp, p, inv_mass_diag) |> Nx.to_number()

    epsilon = 1.0

    {_q_new, _p_new, _logp_new, _grad_new, joint_logp_t} =
      step_fn.(q, p, grad, epsilon, inv_mass_diag)

    joint_logp_new = Nx.to_number(joint_logp_t)

    log_accept =
      if is_number(joint_logp_0) and is_number(joint_logp_new) do
        joint_logp_new - joint_logp_0
      else
        -1000.0
      end

    direction = if log_accept > :math.log(0.5), do: 1.0, else: -1.0

    epsilon =
      search_epsilon(step_fn, q, p, grad, inv_mass_diag, epsilon, direction, joint_logp_0, 0)

    {epsilon, rng}
  end

  defp search_epsilon(
         _step_fn,
         _q,
         _p,
         _grad,
         _inv_mass_diag,
         epsilon,
         _direction,
         _joint_logp_0,
         count
       )
       when count >= 100 do
    max(epsilon, 1.0e-10)
  end

  defp search_epsilon(step_fn, q, p, grad, inv_mass_diag, epsilon, direction, joint_logp_0, count) do
    factor = :math.pow(2.0, direction)
    new_epsilon = epsilon * factor

    {_q_new, _p_new, _logp_new, _grad_new, joint_logp_t} =
      step_fn.(q, p, grad, new_epsilon, inv_mass_diag)

    joint_logp_new = Nx.to_number(joint_logp_t)

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
      search_epsilon(
        step_fn,
        q,
        p,
        grad,
        inv_mass_diag,
        new_epsilon,
        direction,
        joint_logp_0,
        count + 1
      )
    end
  end

  defp is_finite(x) when is_float(x), do: x != :infinity and x != :neg_infinity and x == x
  defp is_finite(_), do: false

  # --- Warmup ---

  defp run_warmup(
         step_fn,
         state,
         epsilon,
         inv_mass,
         d,
         num_warmup,
         max_tree_depth,
         target_accept,
         use_dense,
         multi_step_fn
       ) do
    if num_warmup == 0 do
      {state, epsilon, inv_mass, nil}
    else
      # Three-phase schedule
      # Stan uses term_buffer=50. After the log_epsilon_bar initialization fix
      # (lesson #26), DA converges in 50 iterations since it starts from the
      # correct step size. The shorter term_buffer gives Phase II 150 more
      # iterations for mass matrix adaptation.
      init_buffer = min(75, div(num_warmup, 3))
      term_buffer = 50
      adapt_end = num_warmup - term_buffer

      # Phase I: step size only (0..init_buffer-1), always diagonal
      da_state = StepSize.init(epsilon, target_accept)

      {state, da_state} =
        run_phase(
          step_fn,
          state,
          epsilon,
          inv_mass,
          max_tree_depth,
          da_state,
          0,
          init_buffer,
          nil,
          multi_step_fn
        )

      epsilon = current_epsilon(da_state)

      if adapt_end <= init_buffer do
        epsilon_final = StepSize.finalize(da_state)
        {state, epsilon_final, inv_mass, nil}
      else
        # Phase II: step size + mass matrix with doubling windows
        {state, epsilon, inv_mass, chol_cov} =
          run_phase_ii(
            step_fn,
            state,
            epsilon,
            inv_mass,
            d,
            max_tree_depth,
            target_accept,
            init_buffer,
            adapt_end,
            use_dense,
            multi_step_fn
          )

        # Phase III: step size only (adapt_end..num_warmup-1)
        da_state = StepSize.init(epsilon, target_accept)

        {state, da_state} =
          run_phase(
            step_fn,
            state,
            epsilon,
            inv_mass,
            max_tree_depth,
            da_state,
            adapt_end,
            num_warmup,
            chol_cov,
            multi_step_fn
          )

        epsilon_final = StepSize.finalize(da_state)
        {state, epsilon_final, inv_mass, chol_cov}
      end
    end
  end

  defp run_phase(
         step_fn,
         state,
         _epsilon,
         inv_mass,
         max_tree_depth,
         da_state,
         from,
         to,
         chol_cov,
         multi_step_fn
       ) do
    if from >= to do
      {state, da_state}
    else
      # Cache inv_mass as flat list (mass matrix doesn't change within a phase)
      inv_mass_list =
        case Nx.rank(inv_mass) do
          1 -> Nx.to_flat_list(inv_mass)
          2 -> nil
        end

      Enum.reduce(from..(to - 1)//1, {state, da_state}, fn _i, {state, da_state} ->
        # Use the DA's current epsilon for each step
        eps = current_epsilon(da_state)

        {state, accept_stat} =
          nuts_step_warmup(
            step_fn,
            state,
            eps,
            inv_mass,
            chol_cov,
            max_tree_depth,
            inv_mass_list,
            multi_step_fn
          )

        da_state = StepSize.update(da_state, accept_stat)

        {state, da_state}
      end)
    end
  end

  defp run_phase_ii(
         step_fn,
         state,
         epsilon,
         inv_mass,
         d,
         max_tree_depth,
         target_accept,
         from,
         to,
         use_dense,
         multi_step_fn
       ) do
    # Dense mode needs larger windows for stable covariance estimation
    base_window = if use_dense, do: max(25, 10 * d), else: 25
    windows = build_windows(from, to, base_window)

    # Per-window Welford reset (Stan-style): each window starts fresh so early
    # samples drawn under wrong geometry don't contaminate later mass matrix estimates.
    chol_cov = nil

    Enum.reduce(windows, {state, epsilon, inv_mass, chol_cov}, fn {win_start, win_end},
                                                                  {state, epsilon, inv_mass,
                                                                   chol_cov} ->
      # Fresh Welford for this window
      welford = if use_dense, do: MassMatrix.init_dense(d), else: MassMatrix.init(d)
      da_state = StepSize.init(epsilon, target_accept)

      # Cache inv_mass as flat list for this window (mass matrix doesn't change within a window)
      inv_mass_list =
        case Nx.rank(inv_mass) do
          1 -> Nx.to_flat_list(inv_mass)
          2 -> nil
        end

      {state, _da_state, welford} =
        Enum.reduce(win_start..(win_end - 1)//1, {state, da_state, welford}, fn i,
                                                                                {state, da_state,
                                                                                 welford} ->
          eps = current_epsilon(da_state)
          # Cap tree depth to 8 during first 200 warmup iterations (PyMC-style)
          effective_depth = if i < 200, do: min(max_tree_depth, 8), else: max_tree_depth

          div_before = state.divergences

          {state, accept_stat} =
            nuts_step_warmup(
              step_fn,
              state,
              eps,
              inv_mass,
              chol_cov,
              effective_depth,
              inv_mass_list,
              multi_step_fn
            )

          da_state = StepSize.update(da_state, accept_stat)

          # Stan excludes divergent transitions from mass matrix estimation
          # (divergent trees are truncated, biasing the selected position)
          was_divergent = state.divergences > div_before

          welford =
            if was_divergent,
              do: welford,
              else: MassMatrix.update(welford, state.q)

          {state, da_state, welford}
        end)

      {inv_mass, chol_cov} =
        if use_dense do
          %{cov: cov, chol_cov: chol} = MassMatrix.finalize_dense(welford)
          {cov, chol}
        else
          {MassMatrix.finalize(welford), nil}
        end

      # Re-search step size with new mass matrix (Stan-style)
      {epsilon, rng} =
        find_reasonable_epsilon_with_rng(
          step_fn,
          state.q,
          state.logp,
          state.grad,
          inv_mass,
          state.rng
        )

      state = %{state | rng: rng}

      {state, epsilon, inv_mass, chol_cov}
    end)
  end

  defp build_windows(from, to, base_window) do
    total = to - from

    if total <= 0 do
      []
    else
      do_build_windows(from, to, base_window, [])
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

  # NUTS step with pre-cached inv_mass_list and optional batched leapfrog
  defp nuts_step_warmup(
         step_fn,
         state,
         epsilon,
         inv_mass,
         chol_cov,
         max_tree_depth,
         inv_mass_list,
         multi_step_fn
       ) do
    {p, rng} =
      if inv_mass_list do
        sample_momentum_fast(state.rng, inv_mass_list)
      else
        sample_momentum_for(state.rng, inv_mass, chol_cov)
      end

    joint_logp_0 = Leapfrog.joint_logp(state.logp, p, inv_mass)

    result =
      Tree.build(
        step_fn,
        state.q,
        p,
        state.logp,
        state.grad,
        epsilon,
        inv_mass,
        max_tree_depth,
        rng,
        joint_logp_0,
        multi_step_fn,
        inv_mass_list
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
      divergences: divergences,
      recoveries: Map.get(state, :recoveries, 0)
    }

    {new_state, accept_stat}
  end

  # NUTS step returning additional stats for diagnostics
  # inv_mass_list: pre-cached flat list for diagonal mass (nil for dense)
  # multi_step_fn: batched leapfrog function (nil to use individual step_fn)
  defp nuts_step_with_stats(
         step_fn,
         state,
         epsilon,
         inv_mass,
         chol_cov,
         max_tree_depth,
         inv_mass_list \\ nil,
         multi_step_fn \\ nil
       ) do
    # Use cached list for momentum if available
    {p, rng} =
      if inv_mass_list do
        sample_momentum_fast(state.rng, inv_mass_list)
      else
        sample_momentum_for(state.rng, inv_mass, chol_cov)
      end

    joint_logp_0 = Leapfrog.joint_logp(state.logp, p, inv_mass)

    result =
      Tree.build(
        step_fn,
        state.q,
        p,
        state.logp,
        state.grad,
        epsilon,
        inv_mass,
        max_tree_depth,
        rng,
        joint_logp_0,
        multi_step_fn,
        inv_mass_list
      )

    accept_stat =
      if result.n_steps > 0 do
        result.accept_sum / result.n_steps
      else
        0.0
      end

    {_, rng} = :rand.uniform_s(rng)
    divergences = if result.divergent, do: state.divergences + 1, else: state.divergences
    recovered = Map.get(result, :recovered, false)

    recoveries =
      if recovered, do: Map.get(state, :recoveries, 0) + 1, else: Map.get(state, :recoveries, 0)

    # Energy = -joint_logp (Hamiltonian)
    energy = -Nx.to_number(joint_logp_0)

    new_state = %{
      q: result.q,
      logp: result.logp,
      grad: result.grad,
      rng: rng,
      divergences: divergences,
      recoveries: recoveries
    }

    step_info = %{
      depth: result.depth,
      n_steps: result.n_steps,
      divergent: result.divergent,
      energy: energy,
      recovered: recovered
    }

    {new_state, accept_stat, step_info}
  end

  # --- Sampling ---

  defp run_sampling(
         step_fn,
         state,
         epsilon,
         inv_mass,
         num_samples,
         max_tree_depth,
         chol_cov,
         multi_step_fn
       ) do
    # Cache inv_mass as flat list for fast momentum sampling and U-turn checks
    inv_mass_list =
      case Nx.rank(inv_mass) do
        1 -> Nx.to_flat_list(inv_mass)
        2 -> nil
      end

    {draws_reversed, sample_stats_reversed, state} =
      Enum.reduce(1..num_samples, {[], [], state}, fn _i, {draws, stats_acc, state} ->
        {state, accept_stat, step_info} =
          nuts_step_with_stats(
            step_fn,
            state,
            epsilon,
            inv_mass,
            chol_cov,
            max_tree_depth,
            inv_mass_list,
            multi_step_fn
          )

        step_stat = %{
          tree_depth: step_info.depth,
          n_steps: step_info.n_steps,
          divergent: step_info.divergent,
          accept_prob: accept_stat,
          energy: step_info.energy,
          recovered: Map.get(step_info, :recovered, false)
        }

        {[state.q | draws], [step_stat | stats_acc], state}
      end)

    {Enum.reverse(draws_reversed), Enum.reverse(sample_stats_reversed), state}
  end

  @doc """
  Run multiple chains with different seeds.

  Compiles the model once. Uses vectorized (sequential, shared-warmup) sampling
  by default for multi-chain runs, which eliminates XLA thread pool contention.

  Returns `{[traces], [stats]}` where each element corresponds to one chain.

  ## Options

  - `:vectorized` — use shared-warmup vectorized path (default: `true` for multi-chain)
  - `:parallel` — run chains concurrently when not vectorized (default: `true`)
  - `:max_concurrency` — max parallel chains (default: `num_chains`)
  - `:init_values` — initial values for all chains (default: `%{}`)

  Plus all options from `sample/3` (`:num_warmup`, `:num_samples`, `:seed`, etc.).
  """
  def sample_chains(ir, num_chains, opts \\ []) when num_chains >= 1 do
    vectorized = Keyword.get(opts, :vectorized, num_chains > 1)

    if vectorized and num_chains > 1 do
      sample_chains_vectorized(ir, num_chains, opts)
    else
      sample_chains_parallel(ir, num_chains, opts)
    end
  end

  @doc """
  Vectorized multi-chain sampling: warmup once, run N chains sequentially.

  Eliminates XLA thread pool contention by running all chains in a single
  BEAM process with shared JIT-compiled functions. Warmup runs on chain 0
  only; the resulting step size and mass matrix are shared across all chains.

  Returns `{[traces], [stats]}`.
  """
  def sample_chains_vectorized(ir, num_chains, opts \\ []) when num_chains >= 1 do
    compile_opts = Keyword.take(opts, [:ncp, :device])
    compiled = Compiler.compile_for_sampling(ir, compile_opts)
    sample_chains_vectorized_compiled(compiled, num_chains, opts)
  end

  @doc """
  Vectorized multi-chain sampling using pre-compiled artifacts.
  """
  def sample_chains_vectorized_compiled(compiled, num_chains, opts \\ []) when num_chains >= 1 do
    # Accept both 4-tuple (legacy) and 5-tuple (with multi_step_fn)
    {vag_fn, step_fn, pm, ncp_info, multi_step_fn} =
      case compiled do
        {v, s, p, n, m} -> {v, s, p, n, m}
        {v, s, p, n} -> {v, s, p, n, nil}
      end

    opts = Keyword.merge(@default_opts, opts)
    num_warmup = opts[:num_warmup]
    num_samples = opts[:num_samples]
    max_tree_depth = opts[:max_tree_depth]
    target_accept = opts[:target_accept]
    base_seed = opts[:seed]
    init_values = Keyword.get(opts, :init_values, %{})

    if pm.size == 0 do
      empty_trace = %{}

      empty_stats = %{
        step_size: 0.0,
        inv_mass_diag: Nx.broadcast(Nx.tensor(0.0, type: :f64), {0}),
        divergences: 0,
        num_warmup: num_warmup,
        num_samples: num_samples,
        sample_stats: []
      }

      {List.duplicate(empty_trace, num_chains), List.duplicate(empty_stats, num_chains)}
    else
      d = pm.size
      use_dense = Keyword.get(opts, :dense_mass, false)

      # --- Phase 1: Warmup on chain 0 only ---
      warmup_seed = base_seed
      rng0 = :rand.seed_s(:exsss, warmup_seed)
      {q0, rng0} = init_position(pm, init_values, d, rng0, ncp_info)
      {logp0, grad0} = vag_fn.(q0)
      logp0 = Nx.backend_copy(logp0, Nx.BinaryBackend)
      grad0 = Nx.backend_copy(grad0, Nx.BinaryBackend)
      inv_mass_diag = Nx.broadcast(Nx.tensor(1.0, type: :f64, backend: Nx.BinaryBackend), {d})
      active_step_fn = if use_dense, do: build_generic_step_fn(vag_fn), else: step_fn

      {epsilon, rng0} =
        find_reasonable_epsilon_with_rng(active_step_fn, q0, logp0, grad0, inv_mass_diag, rng0)

      warmup_state = %{q: q0, logp: logp0, grad: grad0, rng: rng0, divergences: 0}

      {_warmup_state, epsilon_final, inv_mass, chol_cov} =
        run_warmup(
          active_step_fn,
          warmup_state,
          epsilon,
          inv_mass_diag,
          d,
          num_warmup,
          max_tree_depth,
          target_accept,
          use_dense,
          nil
        )

      # --- Phase 2: Initialize N chains with different seeds ---
      chain_seeds = Enum.map(0..(num_chains - 1), fn i -> base_seed + i * 7919 end)

      chain_states =
        Enum.map(chain_seeds, fn seed ->
          rng = :rand.seed_s(:exsss, seed)
          {q, rng} = init_position(pm, init_values, d, rng, ncp_info)
          {logp, grad} = vag_fn.(q)
          logp = Nx.backend_copy(logp, Nx.BinaryBackend)
          grad = Nx.backend_copy(grad, Nx.BinaryBackend)
          %{q: q, logp: logp, grad: grad, rng: rng, divergences: 0}
        end)

      # --- Phase 3: Sample all chains sequentially (no XLA contention) ---
      inv_mass_diag_out =
        case Nx.rank(inv_mass) do
          1 -> inv_mass
          2 -> Nx.take_diagonal(inv_mass)
        end

      active_multi = if use_dense, do: nil, else: multi_step_fn

      results =
        Enum.map(chain_states, fn state ->
          {draws, sample_stats, final_state} =
            run_sampling(
              active_step_fn,
              state,
              epsilon_final,
              inv_mass,
              num_samples,
              max_tree_depth,
              chol_cov,
              active_multi
            )

          trace = build_trace(draws, pm, ncp_info)

          stats = %{
            step_size: epsilon_final,
            inv_mass_diag: inv_mass_diag_out,
            divergences: final_state.divergences,
            num_warmup: num_warmup,
            num_samples: num_samples,
            sample_stats: sample_stats
          }

          {trace, stats}
        end)

      traces = Enum.map(results, fn {trace, _} -> trace end)
      stats = Enum.map(results, fn {_, stats} -> stats end)
      {traces, stats}
    end
  end

  # Original parallel implementation (used when vectorized: false)
  defp sample_chains_parallel(ir, num_chains, opts) do
    base_seed = Keyword.get(opts, :seed, 0)
    init_values = Keyword.get(opts, :init_values, %{})
    parallel = Keyword.get(opts, :parallel, true)
    max_concurrency = Keyword.get(opts, :max_concurrency, num_chains)

    # Strip chain-specific opts before passing to sample
    sample_opts = Keyword.drop(opts, [:init_values, :parallel, :max_concurrency, :vectorized])

    # Compile once, share across all chains
    compile_opts = Keyword.take(opts, [:ncp, :device])
    compiled = Compiler.compile_for_sampling(ir, compile_opts)

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

  @doc """
  Stream samples to a receiver process.

  Runs warmup as normal, then sends `{:exmc_sample, i, point_map, step_stat}` to
  `receiver_pid` after each sampling step. The point map is in constrained space.

  Returns `:ok` when done.
  """
  def sample_stream(ir, receiver_pid, init_values \\ %{}, opts \\ []) do
    compile_opts = Keyword.take(opts, [:ncp, :device])
    compiled = Compiler.compile_for_sampling(ir, compile_opts)
    stream_from_compiled(compiled, receiver_pid, init_values, opts)
  end

  defp stream_from_compiled(
         {vag_fn, step_fn, pm, ncp_info, _multi_step_fn},
         receiver_pid,
         init_values,
         opts
       ) do
    stream_from_compiled_impl(vag_fn, step_fn, pm, ncp_info, receiver_pid, init_values, opts)
  end

  defp stream_from_compiled({vag_fn, step_fn, pm, ncp_info}, receiver_pid, init_values, opts) do
    stream_from_compiled_impl(vag_fn, step_fn, pm, ncp_info, receiver_pid, init_values, opts)
  end

  defp stream_from_compiled_impl(vag_fn, step_fn, pm, ncp_info, receiver_pid, init_values, opts) do
    opts = Keyword.merge(@default_opts, opts)
    num_warmup = opts[:num_warmup]
    num_samples = opts[:num_samples]
    max_tree_depth = opts[:max_tree_depth]
    target_accept = opts[:target_accept]
    seed = opts[:seed]

    if pm.size == 0 do
      send(receiver_pid, {:exmc_done, 0})
      :ok
    else
      rng = :rand.seed_s(:exsss, seed)
      d = pm.size

      use_dense = Keyword.get(opts, :dense_mass, false)
      {q, rng} = init_position(pm, init_values, d, rng, ncp_info)
      {logp, grad} = vag_fn.(q)
      logp = Nx.backend_copy(logp, Nx.BinaryBackend)
      grad = Nx.backend_copy(grad, Nx.BinaryBackend)
      inv_mass_diag = Nx.broadcast(Nx.tensor(1.0, type: :f64, backend: Nx.BinaryBackend), {d})
      active_step_fn = if use_dense, do: build_generic_step_fn(vag_fn), else: step_fn

      {epsilon, rng} =
        find_reasonable_epsilon_with_rng(active_step_fn, q, logp, grad, inv_mass_diag, rng)

      state = %{
        q: q,
        logp: logp,
        grad: grad,
        rng: rng,
        divergences: 0,
        recoveries: 0
      }

      {state, epsilon, inv_mass, chol_cov} =
        run_warmup(
          active_step_fn,
          state,
          epsilon,
          inv_mass_diag,
          d,
          num_warmup,
          max_tree_depth,
          target_accept,
          use_dense,
          nil
        )

      # Sampling phase: send each sample to receiver
      Enum.reduce(1..num_samples, state, fn i, state ->
        {state, accept_stat, step_info} =
          nuts_step_with_stats(active_step_fn, state, epsilon, inv_mass, chol_cov, max_tree_depth)

        # Build constrained point map for this step
        unconstrained = PointMap.unpack(state.q, pm)
        constrained = PointMap.to_constrained(unconstrained, pm)
        point_map = reconstruct_ncp(constrained, ncp_info)

        step_stat = %{
          tree_depth: step_info.depth,
          n_steps: step_info.n_steps,
          divergent: step_info.divergent,
          accept_prob: accept_stat,
          energy: step_info.energy,
          recovered: Map.get(step_info, :recovered, false)
        }

        send(receiver_pid, {:exmc_sample, i, point_map, step_stat})
        state
      end)

      send(receiver_pid, {:exmc_done, num_samples})
      :ok
    end
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

      do_ncp_topo(
        remaining -- ready,
        ncp_info,
        ncp_ids,
        new_done,
        Enum.reverse(Enum.sort(ready)) ++ acc
      )
    end
  end

  defp ncp_dep_resolved?(src, ncp_ids, done) when is_binary(src) do
    not MapSet.member?(ncp_ids, src) or MapSet.member?(done, src)
  end

  defp ncp_dep_resolved?(_src, _ncp_ids, _done), do: true
end
