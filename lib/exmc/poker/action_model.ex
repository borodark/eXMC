defmodule Exmc.Poker.ActionModel do
  @moduledoc """
  Softmax action probability model for poker decisions.

  Maps continuous player parameters + hand strength to action probabilities.
  Differentiable — works inside Nx/EXLA for gradient computation.

  Player params (constrained):
    vpip  ∈ [0,1] — voluntarily put $ in pot (higher = looser)
    pfr   ∈ [0,1] — preflop raise rate (higher = more aggressive preflop)
    agg   ∈ (0,∞) — postflop aggression factor
    bluff ∈ [0,1] — bluff frequency (raise with weak hands)

  Actions: 0=fold, 1=call, 2=raise
  """

  @doc """
  Compute action log-probabilities for a batch of hands (Nx tensors).

  ## Args
    - vpip, pfr, agg, bluff: scalar Nx tensors (player params)
    - hand_strengths: {N} tensor of hand strengths in [0,1]

  ## Returns
    {N, 3} tensor of log P(action | params, hand_strength)
  """
  def log_action_probs_nx(vpip, pfr, agg, bluff, hand_strengths) do
    n = Nx.axis_size(hand_strengths, 0)

    # Clamp params to prevent overflow in logits
    vpip = Nx.max(Nx.min(vpip, Nx.tensor(5.0)), Nx.tensor(-5.0))
    pfr = Nx.max(Nx.min(pfr, Nx.tensor(5.0)), Nx.tensor(-5.0))
    agg = Nx.max(Nx.min(agg, Nx.tensor(10.0)), Nx.tensor(0.01))
    bluff = Nx.max(Nx.min(bluff, Nx.tensor(5.0)), Nx.tensor(-5.0))

    fold_logits = Nx.broadcast(Nx.tensor(0.0), {n})

    call_logits =
      hand_strengths
      |> Nx.multiply(2.0)
      |> Nx.add(Nx.multiply(vpip, 3.0))
      |> Nx.subtract(1.0)

    raise_logits =
      hand_strengths
      |> Nx.multiply(2.0)
      |> Nx.multiply(agg)
      |> Nx.add(Nx.multiply(pfr, 3.0))
      |> Nx.add(Nx.multiply(bluff, Nx.multiply(Nx.subtract(1.0, hand_strengths), 3.0)))
      |> Nx.subtract(1.5)

    logits = Nx.stack([fold_logits, call_logits, raise_logits], axis: 1)
    log_softmax(logits)
  end

  @doc """
  Select log-probs for observed actions from the {N, 3} log-prob matrix.

  ## Args
    - log_probs: {N, 3} tensor from log_action_probs_nx
    - actions: {N} integer tensor (0=fold, 1=call, 2=raise)

  ## Returns
    {N} tensor of log P(observed_action)
  """
  def gather_log_probs(log_probs, actions) do
    one_hot = Nx.equal(Nx.reshape(actions, {:auto, 1}), Nx.tensor([[0, 1, 2]]))
    Nx.sum(Nx.multiply(log_probs, one_hot), axes: [1])
  end

  @doc """
  Pure Elixir version for simulation — returns {p_fold, p_call, p_raise}.
  """
  def action_probs(vpip, pfr, agg, bluff, hand_strength) do
    fold_logit = 0.0
    call_logit = vpip * 3.0 + hand_strength * 2.0 - 1.0
    raise_logit = pfr * 3.0 + agg * hand_strength * 2.0 + bluff * (1.0 - hand_strength) * 3.0 - 1.5

    max_l = Enum.max([fold_logit, call_logit, raise_logit])
    exp_f = :math.exp(fold_logit - max_l)
    exp_c = :math.exp(call_logit - max_l)
    exp_r = :math.exp(raise_logit - max_l)
    total = exp_f + exp_c + exp_r

    {exp_f / total, exp_c / total, exp_r / total}
  end

  @doc """
  Sample an action given player params and hand strength.
  Returns 0 (fold), 1 (call), or 2 (raise).
  """
  def sample_action(vpip, pfr, agg, bluff, hand_strength) do
    {p_fold, p_call, _p_raise} = action_probs(vpip, pfr, agg, bluff, hand_strength)
    u = :rand.uniform()

    cond do
      u < p_fold -> 0
      u < p_fold + p_call -> 1
      true -> 2
    end
  end

  defp log_softmax(logits) do
    max = Nx.reduce_max(logits, axes: [1], keep_axes: true)
    shifted = Nx.subtract(logits, max)
    lse = Nx.add(max, Nx.log(Nx.sum(Nx.exp(shifted), axes: [1], keep_axes: true)))
    Nx.subtract(logits, lse)
  end
end
