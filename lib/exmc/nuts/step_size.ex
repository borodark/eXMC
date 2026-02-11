defmodule Exmc.NUTS.StepSize do
  @moduledoc """
  Dual averaging step size adaptation (Nesterov/Hoffman-Gelman).

  Scalar math (Erlang float) for all adaptation logic.
  """

  alias Exmc.NUTS.Leapfrog

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
    log_epsilon = state.mu - :math.sqrt(m) / state.gamma * h_bar
    m_kappa = :math.pow(m, -state.kappa)
    log_epsilon_bar = m_kappa * log_epsilon + (1.0 - m_kappa) * state.log_epsilon_bar

    %{state | m: m, h_bar: h_bar, log_epsilon: log_epsilon, log_epsilon_bar: log_epsilon_bar}
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

    {epsilon, key}
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
    # Safety: cap iterations
    max(epsilon, 1.0e-10)
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
      max(new_epsilon, 1.0e-10)
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
