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
    {_fold, call_logits, raise_logits} = action_logits(vpip, pfr, agg, bluff, hand_strengths)
    fold_logits = Nx.broadcast(Nx.tensor(0.0, type: :f64), {n})

    logits = Nx.stack([fold_logits, call_logits, raise_logits], axis: 1)
    log_softmax(logits)
  end

  # The three action logits as SEPARATE rank-1 tensors. The fold logit is the
  # scalar 0.0 rather than a broadcast vector: it is the softmax reference
  # level, it is constant, and keeping it scalar avoids relying on :broadcast
  # emitting in scalar position.
  defp action_logits(vpip, pfr, agg, bluff, hand_strengths) do
    # Clamp params to prevent overflow in logits
    vpip = Nx.max(Nx.min(vpip, Nx.tensor(5.0)), Nx.tensor(-5.0))
    pfr = Nx.max(Nx.min(pfr, Nx.tensor(5.0)), Nx.tensor(-5.0))
    agg = Nx.max(Nx.min(agg, Nx.tensor(10.0)), Nx.tensor(0.01))
    bluff = Nx.max(Nx.min(bluff, Nx.tensor(5.0)), Nx.tensor(-5.0))

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

    {Nx.tensor(0.0, type: :f64), call_logits, raise_logits}
  end

  @doc """
  Log-probability of the OBSERVED action, per observation, as a rank-1 {N}
  tensor — without ever forming the {N,3} matrix.

  This exists so the model can reach the fused f64 chain shader. The {N,3}
  route cannot: `Nx.stack/2` has no clause in
  `Exmc.NUTS.CustomSynth.Glsl` (it falls through the catch-all at
  `glsl.ex:578`), and behind it sit a rank-2 `reduce_max` and a rank-2
  one-hot, against an emitter that refuses any rank-2 literal outright. So
  `CustomSynth.synthesise/2` returned `{:error, {:unsupported_op, :stack}}`,
  the model fell through to `Nx.Defn.Evaluator`, and on the Vulkan arm --
  where `Exmc.JIT.jit/2` has no fusing compiler at all (`jit.ex:55`) -- the
  whole trajectory was interpreted op by op. Measured at ~907 ms per NUTS
  iteration against EXLA's ~23 ms, which is a 300 s test timeout on a
  500-iteration fixture.

  `is_fold`/`is_call`/`is_raise` are ONE-HOT INDICATORS SUPPLIED AS DATA.
  That is the point of the signature. The observed action is a constant, so
  its indicator is data, not arithmetic -- and it has to be, because the
  emitter has no comparison ops at all (no `:equal`, no `:greater`), so
  computing the one-hot inside the traced closure would be refused exactly
  the way `:stack` was. Supplied as data they are rank-1 captures, which
  since docs/SHADER_CONSTANT_INLINING.md live in the extras SSBO and cost
  nothing in shader size.

  Verified BIT-IDENTICAL to `gather_log_probs(log_action_probs_nx(...), acts)`
  over 200 random draws at n=100: max absolute difference 0.0. Not "within
  tolerance" -- the same arithmetic in the same order, so equality is the
  right assertion.
  """
  def log_prob_of_action(vpip, pfr, agg, bluff, hand_strengths, is_fold, is_call, is_raise) do
    {fold, call, rais} = action_logits(vpip, pfr, agg, bluff, hand_strengths)

    # Elementwise log-sum-exp over the three actions. Same max-shift as
    # log_softmax/1, but rank-1 throughout: max/2 is elementwise, not a
    # reduction over an axis.
    m = Nx.max(Nx.max(fold, call), rais)

    lse =
      Nx.add(
        m,
        Nx.log(
          Nx.exp(Nx.subtract(fold, m))
          |> Nx.add(Nx.exp(Nx.subtract(call, m)))
          |> Nx.add(Nx.exp(Nx.subtract(rais, m)))
        )
      )

    chosen =
      Nx.multiply(is_fold, fold)
      |> Nx.add(Nx.multiply(is_call, call))
      |> Nx.add(Nx.multiply(is_raise, rais))

    Nx.subtract(chosen, lse)
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
