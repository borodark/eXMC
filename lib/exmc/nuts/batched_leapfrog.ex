defmodule Exmc.NUTS.BatchedLeapfrog do
  @moduledoc """
  Batched leapfrog integrator using XLA while-loop.

  Runs N leapfrog steps inside a single JIT call, avoiding per-step
  dispatch overhead (~250us/step). For depth-8 trees this provides
  25-67x speedup over individual step_fn calls.

  Returns pre-allocated tensors of all intermediate states, which
  the tree builder can slice into to construct the merge tree.
  """

  import Nx.Defn

  @max_steps 1024

  @doc """
  Build a batched multi-step function for the given logp_fn and dimension.

  Returns a function `(q, p, grad, eps, inv_mass, n_steps) -> {all_q, all_p, all_logp, all_grad}`
  where each output has shape `{max_steps, d}` or `{max_steps}` (for logp).
  Only the first `n_steps` rows contain valid data.

  `all_logp` contains the raw model logp (NOT joint_logp). The tree builder
  computes joint_logp = logp - KE at leaf construction time, matching the
  individual step_fn path exactly and avoiding float precision issues from
  reconstructing logp as joint_logp + KE.
  """
  def build(logp_fn, d, jit_opts \\ []) do
    EXLA.jit(
      &multi_step(&1, &2, &3, &4, &5, &6, logp_fn: logp_fn, d: d, max_steps: @max_steps),
      Keyword.merge([on_conflict: :reuse], jit_opts)
    )
  end

  defnp multi_step(q, p, grad, eps, inv_mass, n_steps, opts \\ []) do
    logp_fn = opts[:logp_fn]
    d = opts[:d]
    max_steps = opts[:max_steps]

    two = Nx.tensor(2.0, type: :f64)
    half_eps = eps / two
    half = Nx.tensor(0.5, type: :f64)

    all_q = Nx.broadcast(Nx.tensor(0.0, type: :f64), {max_steps, d})
    all_p = Nx.broadcast(Nx.tensor(0.0, type: :f64), {max_steps, d})
    all_logp = Nx.broadcast(Nx.tensor(0.0, type: :f64), {max_steps})
    all_grad = Nx.broadcast(Nx.tensor(0.0, type: :f64), {max_steps, d})
    i = Nx.tensor(0, type: :s64)

    {{_q, _p, _grad, all_q, all_p, all_logp, all_grad, _i, _half_eps, _eps, _inv_mass, _half,
      _n_steps}} =
      while {{q, p, grad, all_q, all_p, all_logp, all_grad, i, half_eps, eps, inv_mass, half,
              n_steps}},
            i < n_steps do
        # Leapfrog step
        p_half = p + half_eps * grad
        q_new = q + eps * (inv_mass * p_half)
        {logp_new, grad_new} = value_and_grad(q_new, logp_fn)
        p_new = p_half + half_eps * grad_new

        # Store raw logp (not joint_logp) — tree builder computes joint_logp at leaf time
        logp_scalar = Nx.reshape(logp_new, {})

        # Store in pre-allocated arrays
        all_q = Nx.put_slice(all_q, [i, 0], Nx.reshape(q_new, {1, d}))
        all_p = Nx.put_slice(all_p, [i, 0], Nx.reshape(p_new, {1, d}))
        all_logp = Nx.indexed_put(all_logp, Nx.reshape(i, {1, 1}), Nx.reshape(logp_scalar, {1}))
        all_grad = Nx.put_slice(all_grad, [i, 0], Nx.reshape(grad_new, {1, d}))

        {{q_new, p_new, grad_new, all_q, all_p, all_logp, all_grad, i + 1, half_eps, eps,
          inv_mass, half, n_steps}}
      end

    {all_q, all_p, all_logp, all_grad}
  end
end
