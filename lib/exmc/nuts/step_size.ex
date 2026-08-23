defmodule Exmc.NUTS.StepSize do
  @moduledoc """
  Dual averaging step size adaptation (Nesterov/Hoffman-Gelman).

  Scalar math (Erlang float) for all adaptation logic.
  """

  alias Exmc.NUTS.Leapfrog

  # Step-size adaptation safety bounds.
  #
  # The bound exists because dual averaging can drive `log_epsilon` arbitrarily
  # far when `h_bar` diverges — a wide prior where the leapfrog blows past the
  # typical set in the first few steps will do it — and `:math.exp/1` raises
  # `badarith` above about 709.78. That kills the chain from inside the
  # adaptation, with a stack trace pointing at the step-size update rather than
  # at the model that provoked it.
  #
  # Note what the failure is NOT. The applications tree's version of this
  # comment said `exp(log_epsilon)` "overflows f64 and the integrator takes
  # infinite steps". The BEAM does not work that way — checked directly on this
  # OTP: `math:exp(710.0)`, `0.0/0.0` and `math:log(0.0)` all raise `badarith`.
  # No float arithmetic here can produce an infinity or a NaN, so there is no
  # silent infinite step to guard against, and there was no NaN to test for.
  # The `x != x` branch that used to sit in the clamps below was unreachable —
  # the same kind of thing as the supervision wrapper in D90: a safety feature
  # that reads as protection and has never once run.
  #
  # The bound must also NOT bind in ordinary operation, and this is the half
  # that was wrong. The applications tree clamped to [1e-6, 1.0], and measured
  # on this host that ceiling binds on **every** model tried, Normal(0,1)
  # included: epsilon pinned at exactly 1.0 for Normal(0,1), Normal(0,100) and
  # Normal(0,10000) alike, with ESS collapsing to exactly 500/2000 in all three.
  # A clamp that is always active is not a guard, it is a fixed step size, and
  # it means dual averaging is not running. Unclamped, the same three adapt to
  # 0.946 / 0.911 / 1.043 and reach ESS 694 / 660 / 526.
  #
  # So the range sits where the failure actually is: far enough out that
  # adaptation never reaches it, close enough in to stop the `badarith`.
  #
  # The lower bound is deliberately far below 1e-6. When crash recovery was
  # destroying the posterior (D90) the symptom was epsilon at 2.41e-11; a 1e-6
  # floor would have lifted that five orders of magnitude and made the defect
  # harder to see. A safety bound that tidies away the evidence of a real bug
  # is worse than no bound.
  @log_epsilon_min :math.log(1.0e-10)
  @log_epsilon_max :math.log(1.0e3)
  @epsilon_min 1.0e-10
  @epsilon_max 1.0e3

  @doc """
  Initialize dual averaging state.
  """
  def init(epsilon, target_accept \\ 0.8) when is_number(epsilon) do
    %{
      log_epsilon: :math.log(epsilon),
      # Initialize smoothed value from the input epsilon (not 0.0).
      # With short adaptation windows (50-200 iterations), starting at 0.0
      # (step size 1.0) causes log_epsilon_bar to lag behind the working value,
      # producing an overly conservative final step size.
      log_epsilon_bar: :math.log(epsilon),
      h_bar: 0.0,
      mu: :math.log(10.0 * epsilon),
      m: 0,
      gamma: 0.05,
      t0: 10.0,
      kappa: 0.75,
      target_accept: target_accept
    }
  end

  @doc """
  One dual averaging update. `accept_stat` is the mean accept probability from the tree.
  """
  def update(state, accept_stat) when is_number(accept_stat) do
    m = state.m + 1
    eta = 1.0 / (m + state.t0)
    h_bar = (1.0 - eta) * state.h_bar + eta * (state.target_accept - accept_stat)

    log_epsilon =
      (state.mu - :math.sqrt(m) / state.gamma * h_bar)
      |> clamp_log_epsilon()

    m_kappa = :math.pow(m, -state.kappa)

    log_epsilon_bar =
      (m_kappa * log_epsilon + (1.0 - m_kappa) * state.log_epsilon_bar)
      |> clamp_log_epsilon()

    %{state | m: m, h_bar: h_bar, log_epsilon: log_epsilon, log_epsilon_bar: log_epsilon_bar}
  end

  defp clamp_log_epsilon(x) when is_float(x) do
    x |> max(@log_epsilon_min) |> min(@log_epsilon_max)
  end

  @doc """
  Finalize: return smoothed step size `exp(log_epsilon_bar)`.
  """
  def finalize(state) do
    :math.exp(state.log_epsilon_bar)
  end

  @doc """
  Find a reasonable initial step size by doubling/halving until accept prob crosses 0.5.

  Returns `{epsilon, new_key}`.
  """
  def find_reasonable_epsilon(vag_fn, q, logp, grad, inv_mass_diag, key) do
    epsilon = 1.0
    {p, key} = Leapfrog.sample_momentum(key, inv_mass_diag)
    joint_logp_0 = Leapfrog.joint_logp(logp, p, inv_mass_diag) |> Nx.to_number()

    {_q_new, p_new, logp_new, _grad_new} =
      Leapfrog.step(vag_fn, q, p, grad, epsilon, inv_mass_diag)

    joint_logp_new = Leapfrog.joint_logp(logp_new, p_new, inv_mass_diag) |> Nx.to_number()

    log_accept = joint_logp_new - joint_logp_0

    # Determine direction: if accept prob > 0.5, double; else halve
    direction = if log_accept > :math.log(0.5), do: 1.0, else: -1.0

    epsilon =
      search_epsilon(vag_fn, q, p, grad, inv_mass_diag, epsilon, direction, joint_logp_0, 0)
      |> clamp_epsilon()

    {epsilon, key}
  end

  defp clamp_epsilon(x) when is_float(x) do
    x |> max(@epsilon_min) |> min(@epsilon_max)
  end

  defp search_epsilon(
         _vag_fn,
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
    # Safety: cap iterations and clamp to working range.
    clamp_epsilon(max(epsilon, @epsilon_min))
  end

  defp search_epsilon(vag_fn, q, p, grad, inv_mass_diag, epsilon, direction, joint_logp_0, count) do
    factor = :math.pow(2.0, direction)
    new_epsilon = epsilon * factor

    {_q_new, p_new, logp_new, _grad_new} =
      Leapfrog.step(vag_fn, q, p, grad, new_epsilon, inv_mass_diag)

    joint_logp_new = Leapfrog.joint_logp(logp_new, p_new, inv_mass_diag) |> Nx.to_number()

    log_accept = joint_logp_new - joint_logp_0

    # Check if we crossed the 0.5 threshold (log(0.5) ~ -0.693)
    crossed =
      if direction > 0 do
        log_accept < :math.log(0.5)
      else
        log_accept > :math.log(0.5)
      end

    if crossed or not is_finite(log_accept) do
      clamp_epsilon(max(new_epsilon, @epsilon_min))
    else
      search_epsilon(
        vag_fn,
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
end
