defmodule Exmc.NUTS.Tree do
  @moduledoc """
  NUTS binary tree building with multinomial sampling and U-turn detection.

  Implements the No-U-Turn Sampler with iterative doubling, per Hoffman & Gelman (2014)
  and Betancourt (2017) multinomial variant.

  Uses `:rand` for scalar random decisions (direction, proposal selection) for performance
  with BinaryBackend. The PRNG state is seeded deterministically from the caller's key.
  """

  alias Exmc.NUTS.Leapfrog

  @doc """
  Build a NUTS tree via iterative doubling.

  `rng` is an `:rand` state (seeded by caller for reproducibility).

  Returns `%{q:, logp:, grad:, n_steps:, divergent:, accept_sum:, depth:}`.
  """
  def build(step_fn, q, p, logp, grad, epsilon, inv_mass_diag, max_depth, rng, joint_logp_0) do
    joint_logp_0_scalar = Nx.to_number(joint_logp_0)

    initial = %{
      q_left: q,
      p_left: p,
      grad_left: grad,
      q_right: q,
      p_right: p,
      grad_right: grad,
      q_prop: q,
      logp_prop: logp,
      grad_prop: grad,
      depth: 0,
      log_sum_weight: 0.0,
      n_steps: 0,
      divergent: false,
      accept_sum: 0.0,
      turning: false
    }

    do_build(step_fn, initial, epsilon, inv_mass_diag, max_depth, rng, joint_logp_0_scalar, 0)
  end

  defp do_build(_step_fn, traj, _epsilon, _inv_mass_diag, max_depth, _rng, _joint_logp_0, depth)
       when depth >= max_depth do
    result(traj, depth)
  end

  defp do_build(_step_fn, %{divergent: true} = traj, _epsilon, _inv_mass_diag, _max_depth, _rng, _joint_logp_0, depth) do
    result(traj, depth)
  end

  defp do_build(_step_fn, %{turning: true} = traj, _epsilon, _inv_mass_diag, _max_depth, _rng, _joint_logp_0, depth) do
    result(traj, depth)
  end

  defp do_build(step_fn, traj, epsilon, inv_mass_diag, max_depth, rng, joint_logp_0, depth) do
    # Random direction
    {rand_val, rng} = :rand.uniform_s(rng)
    go_right = rand_val > 0.5
    dir_epsilon = if go_right, do: epsilon, else: -epsilon

    # Build subtree from appropriate end (use endpoint gradient, not proposal gradient)
    {start_q, start_p, start_grad} =
      if go_right do
        {traj.q_right, traj.p_right, traj.grad_right}
      else
        {traj.q_left, traj.p_left, traj.grad_left}
      end

    {subtree, rng} =
      build_subtree(
        step_fn, start_q, start_p, start_grad,
        dir_epsilon, inv_mass_diag, depth, rng, joint_logp_0
      )

    # Merge subtree into trajectory
    {new_traj, rng} = merge_trajectories(traj, subtree, go_right, inv_mass_diag, rng)

    do_build(step_fn, new_traj, epsilon, inv_mass_diag, max_depth, rng, joint_logp_0, depth + 1)
  end

  # Base case: single leapfrog step
  defp build_subtree(step_fn, q, p, grad, epsilon, inv_mass_diag, 0, rng, joint_logp_0) do
    {q_new, p_new, logp_new, grad_new} = step_fn.(q, p, grad, epsilon, inv_mass_diag)

    joint_logp_new = Leapfrog.joint_logp(logp_new, p_new, inv_mass_diag) |> Nx.to_number()

    # Guard against NaN/Inf from numerical issues (e.g., log(0) gradient)
    {divergent, log_weight, accept_prob} =
      if is_number(joint_logp_new) do
        d = joint_logp_new - joint_logp_0
        {d < -1000.0, min(0.0, d), min(1.0, :math.exp(min(d, 0.0)))}
      else
        {true, -1001.0, 0.0}
      end

    subtree = %{
      q_left: q_new,
      p_left: p_new,
      grad_left: grad_new,
      q_right: q_new,
      p_right: p_new,
      grad_right: grad_new,
      q_prop: q_new,
      logp_prop: logp_new,
      grad_prop: grad_new,
      depth: 0,
      log_sum_weight: log_weight,
      n_steps: 1,
      divergent: divergent,
      accept_sum: accept_prob,
      turning: false
    }

    {subtree, rng}
  end

  # Recursive case: build two half-subtrees and merge
  defp build_subtree(step_fn, q, p, grad, epsilon, inv_mass_diag, depth, rng, joint_logp_0)
       when depth > 0 do
    half_depth = depth - 1

    # Build first half
    {first, rng} =
      build_subtree(step_fn, q, p, grad, epsilon, inv_mass_diag, half_depth, rng, joint_logp_0)

    if first.divergent or first.turning do
      {first, rng}
    else
      # Build second half from appropriate end (use endpoint gradient)
      {next_q, next_p, next_grad} =
        if epsilon > 0 do
          {first.q_right, first.p_right, first.grad_right}
        else
          {first.q_left, first.p_left, first.grad_left}
        end

      {second, rng} =
        build_subtree(
          step_fn, next_q, next_p, next_grad,
          epsilon, inv_mass_diag, half_depth, rng, joint_logp_0
        )

      {merged, rng} = merge_subtrees(first, second, epsilon, inv_mass_diag, rng)
      {merged, rng}
    end
  end

  defp merge_subtrees(first, second, epsilon, inv_mass_diag, rng) do
    combined_log_weight = log_sum_exp(first.log_sum_weight, second.log_sum_weight)
    combined_n_steps = first.n_steps + second.n_steps
    combined_accept_sum = first.accept_sum + second.accept_sum
    combined_divergent = first.divergent or second.divergent

    # Multinomial: accept second's proposal with prob exp(second.log_weight - combined_log_weight)
    {rand_val, rng} = :rand.uniform_s(rng)
    accept_prob = :math.exp(second.log_sum_weight - combined_log_weight)
    use_second = rand_val < accept_prob

    {q_prop, logp_prop, grad_prop} =
      if use_second do
        {second.q_prop, second.logp_prop, second.grad_prop}
      else
        {first.q_prop, first.logp_prop, first.grad_prop}
      end

    # Endpoints depend on direction
    {q_left, p_left, grad_left, q_right, p_right, grad_right} =
      if epsilon > 0 do
        {first.q_left, first.p_left, first.grad_left,
         second.q_right, second.p_right, second.grad_right}
      else
        {second.q_left, second.p_left, second.grad_left,
         first.q_right, first.p_right, first.grad_right}
      end

    # U-turn check
    turning = combined_divergent or check_uturn(q_left, p_left, q_right, p_right, inv_mass_diag)

    merged = %{
      q_left: q_left,
      p_left: p_left,
      grad_left: grad_left,
      q_right: q_right,
      p_right: p_right,
      grad_right: grad_right,
      q_prop: q_prop,
      logp_prop: logp_prop,
      grad_prop: grad_prop,
      depth: max(first.depth, second.depth) + 1,
      log_sum_weight: combined_log_weight,
      n_steps: combined_n_steps,
      divergent: combined_divergent,
      accept_sum: combined_accept_sum,
      turning: turning
    }

    {merged, rng}
  end

  defp merge_trajectories(traj, subtree, go_right, inv_mass_diag, rng) do
    combined_log_weight = log_sum_exp(traj.log_sum_weight, subtree.log_sum_weight)
    combined_n_steps = traj.n_steps + subtree.n_steps
    combined_accept_sum = traj.accept_sum + subtree.accept_sum
    combined_divergent = traj.divergent or subtree.divergent

    # Multinomial: accept subtree's proposal with prob exp(subtree.log_weight - combined_log_weight)
    {rand_val, rng} = :rand.uniform_s(rng)
    accept_prob = :math.exp(subtree.log_sum_weight - combined_log_weight)
    use_subtree = rand_val < accept_prob

    {q_prop, logp_prop, grad_prop} =
      if use_subtree do
        {subtree.q_prop, subtree.logp_prop, subtree.grad_prop}
      else
        {traj.q_prop, traj.logp_prop, traj.grad_prop}
      end

    # Update endpoints
    {q_left, p_left, grad_left, q_right, p_right, grad_right} =
      if go_right do
        {traj.q_left, traj.p_left, traj.grad_left,
         subtree.q_right, subtree.p_right, subtree.grad_right}
      else
        {subtree.q_left, subtree.p_left, subtree.grad_left,
         traj.q_right, traj.p_right, traj.grad_right}
      end

    # U-turn check
    turning = combined_divergent or subtree.turning or
              check_uturn(q_left, p_left, q_right, p_right, inv_mass_diag)

    merged = %{
      q_left: q_left,
      p_left: p_left,
      grad_left: grad_left,
      q_right: q_right,
      p_right: p_right,
      grad_right: grad_right,
      q_prop: q_prop,
      logp_prop: logp_prop,
      grad_prop: grad_prop,
      depth: traj.depth + 1,
      log_sum_weight: combined_log_weight,
      n_steps: combined_n_steps,
      divergent: combined_divergent,
      accept_sum: combined_accept_sum,
      turning: turning
    }

    {merged, rng}
  end

  defp check_uturn(q_left, p_left, q_right, p_right, inv_mass_diag) do
    dq = Nx.subtract(q_right, q_left)
    dot_right = Nx.sum(Nx.multiply(dq, Nx.multiply(inv_mass_diag, p_right))) |> Nx.to_number()
    dot_left = Nx.sum(Nx.multiply(dq, Nx.multiply(inv_mass_diag, p_left))) |> Nx.to_number()
    dot_right < 0.0 or dot_left < 0.0
  end

  defp log_sum_exp(a, b) do
    m = max(a, b)

    if m == :neg_infinity or m == -1.0e300 do
      -1.0e300
    else
      m + :math.log(:math.exp(a - m) + :math.exp(b - m))
    end
  end

  defp result(traj, depth) do
    %{
      q: traj.q_prop,
      logp: traj.logp_prop,
      grad: traj.grad_prop,
      n_steps: traj.n_steps,
      divergent: traj.divergent,
      accept_sum: traj.accept_sum,
      depth: depth
    }
  end
end
