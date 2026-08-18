defmodule Exmc.MAMS do
  @moduledoc """
  Metropolis-Adjusted Microcanonical Sampler — MCLMC's dynamics with an accept
  step, and therefore **asymptotically unbiased**.

  ## The difference from `Exmc.MCLMC`, in three lines

  MCLMC takes one integrator step per draw, refreshes the velocity partially,
  and keeps the result unconditionally. MAMS refreshes the velocity
  *completely*, takes `N ≈ L/ε` integrator steps, and accepts the endpoint with

      P(accept) = min(1, exp(−ΔE)),   ΔE = ΔK − Δlog p

  where `ΔK` is the isokinetic kinetic-energy change accumulated over the
  trajectory (`Exmc.MCLMC.Integrator`). That is the whole algorithm. The
  correctness argument is the ordinary Metropolis one: the isokinetic map is
  reversible under `u → −u` and the palindromic splitting makes the composed
  proposal an involution up to that flip, so `min(1, e^{−ΔE})` gives detailed
  balance with respect to `p(x) × Uniform(S^{d−1})`. Because the velocity is
  fully resampled every iteration, no explicit momentum flip is needed on
  rejection — the same reason HMC with full momentum refresh does not need one.

  The trajectory length is **jittered**. A fixed `N` makes the proposal
  periodic on near-Gaussian targets, which is the same failure mode fixed-path
  HMC has. `N = round(U(0,1)·s + 0.5)` with `s` chosen so `E[N] = L/ε`, taken
  from blackjax's `rescale/1`.

  ## Cost, stated so the comparison is honest

  One MAMS draw costs `N` integrator steps and each minimal-norm step costs
  **two** gradients, so a draw is `2N` gradient evaluations against MCLMC's 2.
  `stats.grad_evals` records the total and `bench/mclmc_bias.exs` divides by it.
  This is the axis on which MAMS is compared to NUTS: effective samples per
  gradient, not per draw and not per second.

  ## Tuning

  MAMS has an acceptance rate, so ε is tuned the ordinary way — dual averaging
  via the existing `Exmc.NUTS.StepSize`, against `Exmc.MCLMC.Tuning`'s
  `mams_target_accept/0`. The MAMS paper gives 65% as asymptotically optimal
  and ~90% as the practical recommendation; the default here is **0.9**.

  `L` comes from the same two estimators `Exmc.MCLMC` uses: the posterior
  scale in phase 2, then the trajectory autocorrelation in phase 3, the latter
  clipped to at most twice the incoming value (blackjax's
  `Lratio_upperbound = 2.0`, which exists because one unlucky ESS estimate can
  otherwise send `L` to infinity).

  ## Options

  As `Exmc.MCLMC`, plus `:target_accept` (default 0.9) and
  `:max_integration_steps` (default 1000 — a trajectory longer than that is
  almost always a symptom, not a setting).

  ## References

  - Robnik et al., *Metropolis Adjusted Microcanonical Hamiltonian Monte
    Carlo*, <https://arxiv.org/html/2503.01707>
  - `blackjax.mcmc.adjusted_mclmc_dynamic`, `blackjax.adaptation.adjusted_mclmc_adaptation`
  """

  alias Exmc.{Compiler, PointMap}
  alias Exmc.MCLMC
  alias Exmc.MCLMC.{Integrator, Tuning}
  alias Exmc.NUTS.StepSize

  @default_opts [
    num_warmup: 1000,
    num_samples: 1000,
    seed: 0,
    integrator: :minimal_norm,
    diagonal_preconditioning: true,
    frac_tune1: 0.2,
    frac_tune2: 0.2,
    frac_tune3: 0.1,
    max_integration_steps: 1000
  ]

  @divergence_threshold 1000.0
  @l_ratio_upper 2.0
  @l_factor_adjusted 0.5

  @doc "Compile `ir` and sample. See the moduledoc for options."
  @spec sample(term(), map(), keyword()) :: {map(), map()}
  def sample(ir, init_values \\ %{}, opts \\ []) do
    compiled = Compiler.compile_for_sampling(ir, Keyword.take(opts, [:ncp, :device]))
    sample_compiled(compiled, init_values, opts)
  end

  @doc "Sample from a pre-compiled model tuple."
  @spec sample_compiled(tuple(), map(), keyword()) :: {map(), map()}
  def sample_compiled(compiled, init_values \\ %{}, opts \\ []) do
    {vag_fn, pm, ncp_info} = MCLMC.unpack(compiled)
    opts = Keyword.merge(@default_opts, opts)
    d = pm.size

    cond do
      d == 0 ->
        {%{}, %{step_size: 0.0, l: 0.0, divergences: 0, grad_evals: 0, accept_rate: 0.0}}

      d < 2 ->
        raise ArgumentError, """
        MAMS's isokinetic dynamics divide by (d-1) and are undefined for d < 2.
        This model has d = #{d}. Use Exmc.NUTS.Sampler for one-dimensional targets.
        """

      true ->
        run(vag_fn, pm, ncp_info, d, init_values, opts)
    end
  end

  # ------------------------------------------------------------------ driver

  defp run(vag_fn, pm, ncp_info, d, init_values, opts) do
    rng = :rand.seed_s(:exsss, opts[:seed])
    type = Exmc.JIT.precision()

    {x, rng} = init_position(pm, ncp_info, init_values, d, rng)
    {logp, grad} = vag_fn.(x)

    state = %{
      x: x,
      u: nil,
      logp: Nx.backend_copy(logp, Nx.BinaryBackend),
      grad: Nx.backend_copy(grad, Nx.BinaryBackend)
    }

    defaults = Tuning.initial_params(d)
    eps0 = Keyword.get(opts, :step_size) || defaults.step_size
    l0 = Keyword.get(opts, :l) || defaults.l

    ctx = %{
      vag_fn: vag_fn,
      d: d,
      type: type,
      integrator: opts[:integrator],
      grads_per_step: Integrator.grads_per_step(opts[:integrator]),
      max_n: opts[:max_integration_steps]
    }

    {state, eps, l, sigma, rng, warm} =
      warmup(ctx, state, rng, eps0, l0, Tuning.sigma_from_variances([], d, type), opts)

    {draws, acc, rng} =
      sample_phase(ctx, state, rng, eps, l, sigma, opts[:num_samples])

    trace = MCLMC.build_trace(draws, pm, ncp_info)
    _ = rng

    stats = %{
      step_size: eps,
      l: l,
      sigma: sigma,
      inv_mass_diag: Nx.multiply(sigma, sigma),
      divergences: acc.divergences,
      accept_rate: safe_div(acc.accept_sum, acc.n),
      mean_integration_steps: safe_div(acc.steps, acc.n),
      integrator_steps: acc.steps + warm.integrator_steps,
      grad_evals: (acc.steps + warm.integrator_steps) * ctx.grads_per_step,
      grads_per_step: ctx.grads_per_step,
      eevpd: variance_of(acc.energies) / d,
      num_warmup: opts[:num_warmup],
      num_samples: opts[:num_samples],
      warmup: warm
    }

    {trace, stats}
  end

  # ------------------------------------------------------------------ warmup

  defp warmup(ctx, state, rng, eps0, l0, sigma0, opts) do
    nw = opts[:num_warmup]

    if nw <= 0 do
      {state, eps0, l0, sigma0, rng, %{integrator_steps: 0, phases: %{}}}
    else
      do_warmup(ctx, state, rng, eps0, l0, sigma0, opts)
    end
  end

  defp do_warmup(ctx, state, rng, eps0, l0, sigma0, opts) do
    nw = opts[:num_warmup]
    pin_eps? = Keyword.has_key?(opts, :step_size)
    pin_l? = Keyword.has_key?(opts, :l)
    precond? = opts[:diagonal_preconditioning]
    target = Keyword.get(opts, :target_accept, Tuning.mams_target_accept())

    n1 = round(nw * opts[:frac_tune1])
    n2 = round(nw * opts[:frac_tune2])
    n3 = round(nw * opts[:frac_tune3])

    da = StepSize.init(eps0, target)

    # phase 1 — epsilon only
    {state, da, rng, _mom, _dr, s1} =
      block(ctx, state, rng, eps0, l0, sigma0, n1, da: da, pin_eps?: pin_eps?)

    eps1 = if pin_eps?, do: eps0, else: StepSize.finalize(da)

    # phase 2 — epsilon plus the variance accumulator
    {state, da, rng, mom, _dr, s2} =
      block(ctx, state, rng, eps1, l0, sigma0, n2,
        da: da,
        pin_eps?: pin_eps?,
        moments: Tuning.moments_init(ctx.d)
      )

    eps2 = if pin_eps?, do: eps0, else: StepSize.finalize(da)
    vars = Tuning.variances(mom)

    {sigma, l2} =
      cond do
        vars == [] ->
          {sigma0, l0}

        precond? ->
          {Tuning.sigma_from_variances(vars, ctx.d, ctx.type),
           Tuning.l_from_variances(vars, preconditioned?: true)}

        true ->
          {sigma0, Tuning.l_from_variances(vars)}
      end

    l2 = if pin_l?, do: l0, else: l2

    # phase 2b — whitening moved the geometry; re-settle epsilon from scratch
    # (a fresh dual-averaging window, not a continuation, because the old
    # window's mu is anchored to the un-whitened step size)
    n2b = if precond? and not pin_eps? and vars != [], do: div(n2, 2), else: 0
    da2 = StepSize.init(eps2, target)

    {state, da2, rng, _mom, _dr, s2b} =
      block(ctx, state, rng, eps2, l2, sigma, n2b, da: da2, pin_eps?: false)

    eps3 =
      cond do
        pin_eps? -> eps0
        n2b > 0 -> StepSize.finalize(da2)
        true -> eps2
      end

    # phase 3 — L from the trajectory autocorrelation
    {state, _da, rng, _mom, draws3, s3} =
      block(ctx, state, rng, eps3, l2, sigma, n3, da: da2, pin_eps?: true, record: true)

    # blackjax's adjusted path scales the *incoming* L by mean(n/ESS) rather
    # than rebuilding it from eps the way unadjusted MCLMC does, and caps the
    # growth at 2x because one unlucky ESS estimate otherwise sends L to
    # infinity. `l_factor = 0.5` here, against 0.4 for unadjusted MCLMC.
    l3 =
      if pin_l? do
        l0
      else
        case Tuning.l_from_ess(draws3, 1.0, l_factor: 1.0) do
          {:ok, ratio} when ratio > 0.0 ->
            min(@l_factor_adjusted * l2 * ratio, l2 * @l_ratio_upper)

          _ ->
            l2
        end
      end

    warm = %{
      phases: %{p1: n1, p2: n2, p2b: n2b, p3: n3},
      integrator_steps: s1 + s2 + s2b + s3,
      l_from_variances: l2,
      l_final: l3
    }

    {state, eps3, l3, sigma, rng, warm}
  end

  # A block of `n` MAMS iterations with optional dual averaging, moment
  # accumulation and draw recording. Returns
  # {state, da, rng, moments, draws, integrator_steps}.
  defp block(_ctx, state, rng, _eps, _l, _sigma, n, opts) when n <= 0 do
    {state, Keyword.fetch!(opts, :da), rng, Keyword.get(opts, :moments), [], 0}
  end

  defp block(ctx, state, rng, eps, l, sigma, n, opts) do
    da0 = Keyword.fetch!(opts, :da)
    pin_eps? = Keyword.get(opts, :pin_eps?, false)
    record? = Keyword.get(opts, :record, false)

    init = {state, da0, rng, Keyword.get(opts, :moments), [], 0, eps}

    {state, da, rng, mom, rev, steps, _eps} =
      Enum.reduce(1..n, init, fn _i, {st, da, rng, mom, draws, steps, eps} ->
        {st2, info, rng} = transition(ctx, st, rng, eps, l, sigma)

        da = if pin_eps?, do: da, else: StepSize.update(da, info.accept_prob)
        eps = if pin_eps?, do: eps, else: working_eps(da)

        mom = if mom, do: Tuning.moments_update(mom, st2.x, 1.0), else: nil
        draws = if record?, do: [st2.x | draws], else: draws

        {st2, da, rng, mom, draws, steps + info.n, eps}
      end)

    {state, da, rng, mom, Enum.reverse(rev), steps}
  end

  # `StepSize` exposes only the *smoothed* epsilon via `finalize/1`; the
  # working value the next iteration should actually use is
  # `exp(log_epsilon)`. Read it from the state rather than keeping a shadow
  # copy that can drift out of sync with the dual averaging.
  defp working_eps(%{log_epsilon: le}), do: :math.exp(le)

  # ---------------------------------------------------------------- sampling

  defp sample_phase(_ctx, _state, rng, _eps, _l, _sigma, n) when n <= 0 do
    {[], %{accept_sum: 0.0, n: 0, steps: 0, divergences: 0, energies: []}, rng}
  end

  defp sample_phase(ctx, state, rng, eps, l, sigma, n) do
    init =
      {state, rng, [], %{accept_sum: 0.0, n: 0, steps: 0, divergences: 0, energies: []}}

    {_state, rng, rev, acc} =
      Enum.reduce(1..n, init, fn _i, {st, rng, draws, acc} ->
        {st2, info, rng} = transition(ctx, st, rng, eps, l, sigma)

        acc = %{
          accept_sum: acc.accept_sum + info.accept_prob,
          n: acc.n + 1,
          steps: acc.steps + info.n,
          divergences: acc.divergences + if(info.divergent?, do: 1, else: 0),
          energies: [info.delta_energy | acc.energies]
        }

        {st2, rng, [st2.x | draws], acc}
      end)

    {Enum.reverse(rev), acc, rng}
  end

  # One MAMS transition.
  defp transition(ctx, state, rng, eps, l, sigma) do
    {n, rng} = draw_num_steps(l / eps, ctx.max_n, rng)
    {u, rng} = Integrator.random_unit(ctx.d, rng)

    proposal = %{state | u: u}

    {proposed, delta_energy} =
      Integrator.run(proposal, n, eps, sigma, ctx.d, ctx.vag_fn, ctx.integrator)

    ok? = finite?(delta_energy) and finite?(Nx.to_number(proposed.logp))
    divergent? = not ok? or delta_energy > @divergence_threshold

    accept_prob =
      cond do
        not ok? -> 0.0
        delta_energy <= 0.0 -> 1.0
        true -> :math.exp(-delta_energy)
      end

    {uni, rng} = :rand.uniform_s(rng)
    accepted? = uni < accept_prob

    next = if accepted?, do: proposed, else: state

    {next,
     %{
       accept_prob: accept_prob,
       accepted?: accepted?,
       n: n,
       delta_energy: if(ok?, do: delta_energy, else: 0.0),
       divergent?: divergent?
     }, rng}
  end

  @doc """
  A jittered integration-step count with mean `mu`.

  `N = round(U(0,1)·s + 0.5)` where `s = rescale(mu)` solves `E[N] = mu`;
  blackjax's `rescale/1` verbatim. Falls back to `ceil(mu)` when `mu < 1`,
  where the rescaling has no solution.
  """
  @spec draw_num_steps(float(), pos_integer(), :rand.state()) :: {pos_integer(), :rand.state()}
  def draw_num_steps(mu, max_n, rng) do
    {uni, rng} = :rand.uniform_s(rng)

    n =
      if mu > 1.0 do
        k = Float.floor(2.0 * mu - 1.0)
        x = k * (mu - 0.5 * (k + 1.0)) / (k + 1.0 - mu)
        round(uni * (k + x) + 0.5)
      else
        ceil(mu)
      end

    {n |> max(1) |> min(max_n), rng}
  end

  # ------------------------------------------------------------------ shared

  defp init_position(pm, ncp, init_values, d, rng) when map_size(init_values) == 0 do
    _ = {pm, ncp}

    {vals, rng} =
      Enum.map_reduce(1..d, rng, fn _i, r ->
        {z, r} = :rand.normal_s(r)
        {z * 0.1, r}
      end)

    {Nx.tensor(vals, type: Exmc.JIT.precision(), backend: Nx.BinaryBackend), rng}
  end

  defp init_position(pm, ncp, init_values, _d, rng) do
    if map_size(ncp) > 0 do
      raise ArgumentError, """
      Exmc.MAMS does not yet invert the non-centred reparameterisation for
      user-supplied init_values. Either omit init_values, or compile with
      `ncp: false`.
      """
    end

    x =
      init_values
      |> PointMap.to_unconstrained(pm)
      |> PointMap.pack(pm)
      |> Nx.backend_copy(Nx.BinaryBackend)

    {x, rng}
  end

  defp variance_of(es) do
    finite = Enum.filter(es, &finite?/1)
    n = length(finite)

    if n < 2 do
      0.0
    else
      m = Enum.sum(finite) / n
      Enum.reduce(finite, 0.0, fn e, a -> a + (e - m) * (e - m) end) / (n - 1)
    end
  end

  defp safe_div(_a, 0), do: 0.0
  defp safe_div(a, b), do: a / b

  defp finite?(x) when is_float(x), do: x == x and x != :infinity and x != :neg_infinity
  defp finite?(x) when is_integer(x), do: true
  defp finite?(_), do: false
end
