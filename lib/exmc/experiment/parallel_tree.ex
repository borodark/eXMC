defmodule Exmc.Experiment.ParallelTree do
  @moduledoc """
  **EXPERIMENT (NEGATIVE RESULT)** — Speculative parallel NUTS tree builder.

  Branch: `experiment/dist-default-numerics` (merged 2026-02-12)
  Status: Architecturally sound, situationally unprofitable. Not used in production.

  ## Results (E25)

  Tested with identity mass matrix (no warmup). Average tree depth 1.5-2.0.
  Task.async spawn overhead (~50μs) exceeds savings for shallow trees.
  Stress model: **0.85x regression**. Simple/medium: neutral.
  Profitability threshold: depth >= 3 (~1.6ms subtree, 25% hit rate → 400μs saving).

  The test used identity mass matrix, giving artificially shallow trees. With adapted
  mass matrix (depth 3-5), speculation should become profitable. See E31 (planned).

  See `EXPERIMENT_RESULTS.md` for full data, `benchmark/gradient_service_bench.exs`
  phase_b to reproduce.

  ## Design

  At each doubling level, spawn the next subtree as a Task *before* merging
  the current one. If the merge triggers a U-turn or divergence, cancel the
  speculative Task. If it continues, await it.

  This exploits the fact that at depth `d`, the next subtree (depth `d`)
  starts from the same endpoint and uses the same step_fn — its computation
  is independent of whether the current merge accepts or rejects.

  The tricky part is RNG: we pre-split RNG states for each depth level
  so speculative subtrees get deterministic-but-independent seeds.

  ## Limitations

  - Only works with the "plain" path (no speculative buffer / NIF).
    The existing speculative buffer already amortizes JIT dispatch;
    layering process parallelism on top would be redundant.
  - Parallel benefit only materializes when subtree build time > merge time,
    i.e., for deep trees (depth >= 3) where leapfrog dominates.
  """

  @doc """
  Build a NUTS tree with speculative parallelism.

  Same signature as `Tree.build/12`, but uses Tasks for subtree speculation.
  Falls through to `Tree.build` for the actual subtree construction (reuses
  all existing logic: U-turn checks, merge, etc.).

  Returns `%{q:, logp:, grad:, n_steps:, divergent:, accept_sum:, depth:}`.
  """
  def build(
        step_fn,
        q,
        p,
        logp,
        grad,
        epsilon,
        inv_mass_diag,
        max_depth,
        rng,
        joint_logp_0,
        inv_mass_list \\ nil
      ) do
    joint_logp_0_scalar = Nx.to_number(joint_logp_0)

    q_list = Nx.to_flat_list(q)
    p_list = Nx.to_flat_list(p)

    initial = %{
      q_left: q,
      p_left: p,
      grad_left: grad,
      q_left_list: q_list,
      p_left_list: p_list,
      q_right: q,
      p_right: p,
      grad_right: grad,
      q_right_list: q_list,
      p_right_list: p_list,
      q_prop: q,
      logp_prop: logp,
      grad_prop: grad,
      rho_list: p_list,
      depth: 0,
      log_sum_weight: 0.0,
      n_steps: 0,
      divergent: false,
      accept_sum: 0.0,
      turning: false
    }

    # Pre-split RNG for up to max_depth levels (each depth gets independent RNG)
    {rngs, _} =
      Enum.map_reduce(1..max(max_depth, 1), rng, fn _, r ->
        {_, r1} = :rand.uniform_s(r)
        {_, r2} = :rand.uniform_s(r1)
        {r2, r2}
      end)

    do_build_parallel(
      step_fn,
      initial,
      epsilon,
      inv_mass_diag,
      max_depth,
      rng,
      rngs,
      joint_logp_0_scalar,
      0,
      inv_mass_list,
      %{speculated: 0, used: 0, cancelled: 0}
    )
  end

  # --- Termination clauses ---

  defp do_build_parallel(
         _step_fn,
         traj,
         _eps,
         _inv_mass,
         max_depth,
         _rng,
         _rngs,
         _joint_logp_0,
         depth,
         _inv_mass_list,
         speculation_stats
       )
       when depth >= max_depth do
    {result_map(traj, depth), speculation_stats}
  end

  defp do_build_parallel(
         _step_fn,
         %{divergent: true} = traj,
         _eps,
         _inv_mass,
         _max_depth,
         _rng,
         _rngs,
         _joint_logp_0,
         depth,
         _inv_mass_list,
         speculation_stats
       ) do
    {result_map(traj, depth), speculation_stats}
  end

  defp do_build_parallel(
         _step_fn,
         %{turning: true} = traj,
         _eps,
         _inv_mass,
         _max_depth,
         _rng,
         _rngs,
         _joint_logp_0,
         depth,
         _inv_mass_list,
         speculation_stats
       ) do
    {result_map(traj, depth), speculation_stats}
  end

  # --- Main recursive case with speculation ---

  defp do_build_parallel(
         step_fn,
         traj,
         epsilon,
         inv_mass_diag,
         max_depth,
         rng,
         rngs,
         joint_logp_0,
         depth,
         inv_mass_list,
         speculation_stats
       ) do
    # Random direction for THIS depth
    {rand_val, rng} = :rand.uniform_s(rng)
    go_right = rand_val > 0.5
    dir_epsilon = if go_right, do: epsilon, else: -epsilon

    {start_q, start_p, start_grad} =
      if go_right do
        {traj.q_right, traj.p_right, traj.grad_right}
      else
        {traj.q_left, traj.p_left, traj.grad_left}
      end

    # Determine if we should speculatively start the NEXT subtree.
    # Only speculate if:
    # 1. We're not at the second-to-last depth (next depth would exceed max)
    # 2. Current tree isn't already divergent/turning
    # 3. Subtree depth >= 2 (enough work to justify Task overhead)
    can_speculate = depth + 1 < max_depth and depth >= 2

    # Pre-pick the NEXT direction and endpoint (for speculation)
    # We use the pre-split RNG for the next depth level
    {next_rand_val, next_rng_rest} =
      if can_speculate do
        :rand.uniform_s(Enum.at(rngs, depth))
      else
        {0.5, rng}
      end

    next_go_right = next_rand_val > 0.5

    # Speculatively start next subtree as a Task
    # Key: next subtree endpoint depends on THIS subtree's result (for the
    # direction that extends the trajectory). But we can predict:
    # - If go_right NOW, the right endpoint changes. Next subtree will extend
    #   from either the NEW right (if next_go_right) or existing left (if not).
    # - The "other direction" endpoint is unchanged, so we can pre-start that.
    #
    # CONSERVATIVE approach: only speculate when next direction is OPPOSITE
    # to current. In that case, the start point is from the existing trajectory
    # (unchanged by current subtree), so we can start it immediately.
    {speculative_task, speculation_stats} =
      if can_speculate and next_go_right != go_right do
        # Next subtree extends from the side that current subtree doesn't touch
        {spec_q, spec_p, spec_grad} =
          if next_go_right do
            # Next goes right, but current went left → right endpoint unchanged
            {traj.q_right, traj.p_right, traj.grad_right}
          else
            # Next goes left, but current went right → left endpoint unchanged
            {traj.q_left, traj.p_left, traj.grad_left}
          end

        spec_dir_epsilon = if next_go_right, do: epsilon, else: -epsilon
        spec_rng = next_rng_rest

        task =
          Task.async(fn ->
            build_subtree_plain(
              step_fn,
              spec_q,
              spec_p,
              spec_grad,
              spec_dir_epsilon,
              inv_mass_diag,
              depth,
              spec_rng,
              joint_logp_0,
              inv_mass_list
            )
          end)

        {task, Map.update!(speculation_stats, :speculated, &(&1 + 1))}
      else
        {nil, speculation_stats}
      end

    # Build current subtree (blocking)
    {subtree, rng} =
      build_subtree_plain(
        step_fn,
        start_q,
        start_p,
        start_grad,
        dir_epsilon,
        inv_mass_diag,
        depth,
        rng,
        joint_logp_0,
        inv_mass_list
      )

    # Merge current subtree into trajectory
    {new_traj, rng} =
      merge_trajectories(traj, subtree, go_right, inv_mass_diag, rng, inv_mass_list)

    # Check if tree should continue
    if new_traj.divergent or new_traj.turning do
      # Cancel speculative task if running
      speculation_stats =
        if speculative_task do
          Task.shutdown(speculative_task, :brutal_kill)
          Map.update!(speculation_stats, :cancelled, &(&1 + 1))
        else
          speculation_stats
        end

      {result_map(new_traj, depth + 1), speculation_stats}
    else
      # Tree continues — can we use the speculative result?
      if speculative_task do
        # Await the speculative subtree
        {next_subtree, _spec_rng} = Task.await(speculative_task, :infinity)
        speculation_stats = Map.update!(speculation_stats, :used, &(&1 + 1))

        # Merge speculative subtree into trajectory
        {new_traj2, rng} =
          merge_trajectories(
            new_traj,
            next_subtree,
            next_go_right,
            inv_mass_diag,
            rng,
            inv_mass_list
          )

        # Skip ahead by 2 depths (current + speculative both done)
        do_build_parallel(
          step_fn,
          new_traj2,
          epsilon,
          inv_mass_diag,
          max_depth,
          rng,
          rngs,
          joint_logp_0,
          depth + 2,
          inv_mass_list,
          speculation_stats
        )
      else
        # No speculative result — recurse normally for remaining depths
        do_build_parallel(
          step_fn,
          new_traj,
          epsilon,
          inv_mass_diag,
          max_depth,
          rng,
          rngs,
          joint_logp_0,
          depth + 1,
          inv_mass_list,
          speculation_stats
        )
      end
    end
  end

  # --- Subtree building (plain path, reuses Tree's recursive logic) ---

  # Build a subtree using plain step_fn calls (no speculative buffer).
  # This reimplements the core recursive subtree build to avoid depending
  # on Tree's private functions.
  defp build_subtree_plain(
         step_fn,
         q,
         p,
         grad,
         epsilon,
         inv_mass_diag,
         0,
         rng,
         joint_logp_0,
         _inv_mass_list
       ) do
    {q_new, p_new, logp_new, grad_new, joint_logp_t} =
      step_fn.(q, p, grad, epsilon, inv_mass_diag)

    q_new = Nx.backend_copy(q_new, Nx.BinaryBackend)
    p_new = Nx.backend_copy(p_new, Nx.BinaryBackend)
    logp_new = Nx.backend_copy(logp_new, Nx.BinaryBackend)
    grad_new = Nx.backend_copy(grad_new, Nx.BinaryBackend)

    joint_logp_new = Nx.to_number(joint_logp_t)

    {divergent, log_weight, accept_prob} =
      if is_number(joint_logp_new) do
        d = joint_logp_new - joint_logp_0
        {d < -1000.0, d, min(1.0, :math.exp(min(d, 0.0)))}
      else
        {true, -1001.0, 0.0}
      end

    q_list = Nx.to_flat_list(q_new)
    p_list = Nx.to_flat_list(p_new)

    subtree = %{
      q_left: q_new,
      p_left: p_new,
      grad_left: grad_new,
      q_left_list: q_list,
      p_left_list: p_list,
      q_right: q_new,
      p_right: p_new,
      grad_right: grad_new,
      q_right_list: q_list,
      p_right_list: p_list,
      q_prop: q_new,
      logp_prop: logp_new,
      grad_prop: grad_new,
      rho_list: p_list,
      depth: 0,
      log_sum_weight: log_weight,
      n_steps: 1,
      divergent: divergent,
      accept_sum: accept_prob,
      turning: false
    }

    {subtree, rng}
  end

  defp build_subtree_plain(
         step_fn,
         q,
         p,
         grad,
         epsilon,
         inv_mass_diag,
         depth,
         rng,
         joint_logp_0,
         inv_mass_list
       )
       when depth > 0 do
    # Build first half
    {first, rng} =
      build_subtree_plain(
        step_fn,
        q,
        p,
        grad,
        epsilon,
        inv_mass_diag,
        depth - 1,
        rng,
        joint_logp_0,
        inv_mass_list
      )

    if first.divergent or first.turning do
      {first, rng}
    else
      # Build second half from first's endpoint
      {next_q, next_p, next_grad} =
        if epsilon > 0 do
          {first.q_right, first.p_right, first.grad_right}
        else
          {first.q_left, first.p_left, first.grad_left}
        end

      {second, rng} =
        build_subtree_plain(
          step_fn,
          next_q,
          next_p,
          next_grad,
          epsilon,
          inv_mass_diag,
          depth - 1,
          rng,
          joint_logp_0,
          inv_mass_list
        )

      {merged, rng} = merge_subtrees(first, second, epsilon, inv_mass_diag, rng, inv_mass_list)
      {merged, rng}
    end
  end

  # --- Merge operations (duplicated from Tree for self-containment) ---

  defp merge_subtrees(first, second, epsilon, inv_mass_diag, rng, inv_mass_list) do
    combined_log_weight = log_sum_exp(first.log_sum_weight, second.log_sum_weight)
    combined_n_steps = first.n_steps + second.n_steps
    combined_accept_sum = first.accept_sum + second.accept_sum
    combined_divergent = first.divergent or second.divergent

    {rand_val, rng} = :rand.uniform_s(rng)
    accept_prob = :math.exp(second.log_sum_weight - combined_log_weight)
    use_second = rand_val < accept_prob

    {q_prop, logp_prop, grad_prop} =
      if use_second do
        {second.q_prop, second.logp_prop, second.grad_prop}
      else
        {first.q_prop, first.logp_prop, first.grad_prop}
      end

    rho_list = zip_add(first.rho_list, second.rho_list)

    {q_left, p_left, grad_left, q_left_list, p_left_list, q_right, p_right, grad_right,
     q_right_list,
     p_right_list} =
      if epsilon > 0 do
        {first.q_left, first.p_left, first.grad_left, first.q_left_list, first.p_left_list,
         second.q_right, second.p_right, second.grad_right, second.q_right_list,
         second.p_right_list}
      else
        {second.q_left, second.p_left, second.grad_left, second.q_left_list, second.p_left_list,
         first.q_right, first.p_right, first.grad_right, first.q_right_list, first.p_right_list}
      end

    iml = inv_mass_list || Nx.to_flat_list(inv_mass_diag)

    turning =
      combined_divergent or second.turning or
        check_uturn_rho(rho_list, p_left_list, p_right_list, iml)

    turning =
      if not turning and first.depth > 0 do
        {left_sub, right_sub} = if epsilon > 0, do: {first, second}, else: {second, first}
        partial_rho_2 = zip_add(left_sub.rho_list, right_sub.p_left_list)

        if check_uturn_rho(partial_rho_2, left_sub.p_left_list, right_sub.p_left_list, iml) do
          true
        else
          partial_rho_3 = zip_add(left_sub.p_right_list, right_sub.rho_list)
          check_uturn_rho(partial_rho_3, left_sub.p_right_list, right_sub.p_right_list, iml)
        end
      else
        turning
      end

    merged = %{
      q_left: q_left,
      p_left: p_left,
      grad_left: grad_left,
      q_left_list: q_left_list,
      p_left_list: p_left_list,
      q_right: q_right,
      p_right: p_right,
      grad_right: grad_right,
      q_right_list: q_right_list,
      p_right_list: p_right_list,
      q_prop: q_prop,
      logp_prop: logp_prop,
      grad_prop: grad_prop,
      rho_list: rho_list,
      depth: max(first.depth, second.depth) + 1,
      log_sum_weight: combined_log_weight,
      n_steps: combined_n_steps,
      divergent: combined_divergent,
      accept_sum: combined_accept_sum,
      turning: turning
    }

    {merged, rng}
  end

  defp merge_trajectories(traj, subtree, go_right, inv_mass_diag, rng, inv_mass_list) do
    combined_log_weight = log_sum_exp(traj.log_sum_weight, subtree.log_sum_weight)
    combined_n_steps = traj.n_steps + subtree.n_steps
    combined_accept_sum = traj.accept_sum + subtree.accept_sum
    combined_divergent = traj.divergent or subtree.divergent

    {rand_val, rng} = :rand.uniform_s(rng)
    use_subtree = :math.log(rand_val) < subtree.log_sum_weight - traj.log_sum_weight

    {q_prop, logp_prop, grad_prop} =
      if use_subtree do
        {subtree.q_prop, subtree.logp_prop, subtree.grad_prop}
      else
        {traj.q_prop, traj.logp_prop, traj.grad_prop}
      end

    rho_list = zip_add(traj.rho_list, subtree.rho_list)

    {q_left, p_left, grad_left, q_left_list, p_left_list, q_right, p_right, grad_right,
     q_right_list,
     p_right_list} =
      if go_right do
        {traj.q_left, traj.p_left, traj.grad_left, traj.q_left_list, traj.p_left_list,
         subtree.q_right, subtree.p_right, subtree.grad_right, subtree.q_right_list,
         subtree.p_right_list}
      else
        {subtree.q_left, subtree.p_left, subtree.grad_left, subtree.q_left_list,
         subtree.p_left_list, traj.q_right, traj.p_right, traj.grad_right, traj.q_right_list,
         traj.p_right_list}
      end

    iml = inv_mass_list || Nx.to_flat_list(inv_mass_diag)

    turning =
      combined_divergent or subtree.turning or
        check_uturn_rho(rho_list, p_left_list, p_right_list, iml)

    turning =
      if not turning do
        {left_sub, right_sub} = if go_right, do: {traj, subtree}, else: {subtree, traj}
        partial_rho_2 = zip_add(left_sub.rho_list, right_sub.p_left_list)

        if check_uturn_rho(partial_rho_2, left_sub.p_left_list, right_sub.p_left_list, iml) do
          true
        else
          partial_rho_3 = zip_add(left_sub.p_right_list, right_sub.rho_list)
          check_uturn_rho(partial_rho_3, left_sub.p_right_list, right_sub.p_right_list, iml)
        end
      else
        turning
      end

    merged = %{
      q_left: q_left,
      p_left: p_left,
      grad_left: grad_left,
      q_left_list: q_left_list,
      p_left_list: p_left_list,
      q_right: q_right,
      p_right: p_right,
      grad_right: grad_right,
      q_right_list: q_right_list,
      p_right_list: p_right_list,
      q_prop: q_prop,
      logp_prop: logp_prop,
      grad_prop: grad_prop,
      rho_list: rho_list,
      depth: traj.depth + 1,
      log_sum_weight: combined_log_weight,
      n_steps: combined_n_steps,
      divergent: combined_divergent,
      accept_sum: combined_accept_sum,
      turning: turning
    }

    {merged, rng}
  end

  # --- Helpers ---

  defp result_map(traj, depth) do
    %{
      q: traj.q_prop,
      logp: traj.logp_prop,
      grad: traj.grad_prop,
      n_steps: traj.n_steps,
      divergent: traj.divergent,
      accept_sum: traj.accept_sum,
      depth: depth,
      recovered: false
    }
  end

  defp check_uturn_rho(rho, pl, pr, inv_mass_list) do
    {dot_right, dot_left} = zip_reduce_rho(rho, pl, pr, inv_mass_list, 0.0, 0.0)
    dot_right < 0.0 or dot_left < 0.0
  end

  defp zip_reduce_rho([], [], [], [], dr, dl), do: {dr, dl}

  defp zip_reduce_rho([r | rs], [pl | pls], [pr | prs], [im | ims], dr, dl) do
    v = r * im
    zip_reduce_rho(rs, pls, prs, ims, dr + v * pr, dl + v * pl)
  end

  defp zip_add([], []), do: []
  defp zip_add([a | as_], [b | bs]), do: [a + b | zip_add(as_, bs)]

  defp log_sum_exp(a, b) do
    m = max(a, b)

    if m == :neg_infinity or m == -1.0e300 do
      -1.0e300
    else
      m + :math.log(:math.exp(a - m) + :math.exp(b - m))
    end
  end
end
