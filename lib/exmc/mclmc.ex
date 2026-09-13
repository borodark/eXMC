defmodule Exmc.MCLMC do
  @moduledoc """
  Microcanonical Langevin Monte Carlo — a sibling of `Exmc.NUTS.Sampler`.

  ## What it is, and the one thing to know before using it

  MCLMC replaces Hamiltonian dynamics with *isokinetic* Langevin dynamics: the
  state is a position `x` and a **unit** velocity `u` (`Exmc.MCLMC.Integrator`),
  and one draw costs one integrator step — two gradients — rather than a whole
  NUTS trajectory. There is no tree, no U-turn test, and no dynamic control
  flow: the loop is `for k in 0..K`.

  **It is biased.** Not "biased in principle, negligible in practice" — biased
  by construction, with the bias controlled by the step size. Larger ε is
  faster *and* more wrong. That is the trade it makes, and this repository
  publishes the number rather than hiding it: see `bench/mclmc_bias.exs` and
  `bench_results/MCLMC_BIAS.md`.

  If you want an asymptotically exact answer from the same dynamics, use
  `Exmc.MAMS`, which adds a Metropolis accept on the energy error.

  ## Interface

  Mirrors `Exmc.NUTS.Sampler`:

      {trace, stats} = Exmc.MCLMC.sample(ir, %{}, num_warmup: 500, num_samples: 2000)
      {trace, stats} = Exmc.MCLMC.sample_compiled(compiled, %{}, opts)

  `compiled` is whatever `Exmc.NUTS.Sampler.compile/1` returns — MCLMC uses
  only the `vag_fn` and `PointMap` slots, so nothing in the model layer, the
  IR or the compiler changes to support it.

  `trace` is `%{name => Nx.t() of shape {num_samples, ...}}` in constrained
  space. `stats` carries `:step_size`, `:l`, `:sigma`, `:grad_evals`,
  `:divergences` and `:eevpd` (the *measured* energy error variance per
  dimension, so a run can be checked against the target it was tuned to).

  ## Options

  | option | default | meaning |
  |---|---|---|
  | `:num_warmup` | 1000 | tuning steps |
  | `:num_samples` | 1000 | draws; one integrator step each |
  | `:seed` | 0 | `:rand` seed |
  | `:integrator` | `:minimal_norm` | or `:leapfrog` (1 gradient/step) |
  | `:diagonal_preconditioning` | `true` | whiten with the warmup variance |
  | `:step_size` | tuned | pin ε and skip its adaptation |
  | `:l` | tuned | pin the decoherence length |
  | `:desired_energy_var` | 5e-4 | the EEVPD target — **the bias knob** |
  | `:frac_tune1/2/3` | 0.1 each | warmup split; phase 3 tunes `L` from ESS |

  Pinning `:step_size` is how the bias sweep works: hold everything else and
  move ε.
  """

  alias Exmc.{Compiler, PointMap, Transform}
  alias Exmc.MCLMC.{Integrator, Tuning}

  @default_opts [
    num_warmup: 1000,
    num_samples: 1000,
    seed: 0,
    integrator: :minimal_norm,
    diagonal_preconditioning: true,
    frac_tune1: 0.1,
    frac_tune2: 0.1,
    frac_tune3: 0.1
  ]

  @doc "Compile `ir` and sample. See the moduledoc for options."
  @spec sample(term(), map(), keyword()) :: {map(), map()}
  def sample(ir, init_values \\ %{}, opts \\ []) do
    compiled = Compiler.compile_for_sampling(ir, Keyword.take(opts, [:ncp, :device]))
    sample_compiled(compiled, init_values, opts)
  end

  @doc "Sample from a pre-compiled model tuple."
  @spec sample_compiled(tuple(), map(), keyword()) :: {map(), map()}
  def sample_compiled(compiled, init_values \\ %{}, opts \\ []) do
    {vag_fn, pm, ncp_info} = unpack(compiled)
    opts = Keyword.merge(@default_opts, opts)

    d = pm.size

    cond do
      d == 0 ->
        {%{}, empty_stats(opts)}

      d < 2 ->
        raise ArgumentError, """
        MCLMC's isokinetic dynamics divide by (d-1) and are undefined for d < 2.
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

    {u, rng} = Integrator.random_unit(d, rng)
    state = %{state | u: u}

    defaults = Tuning.initial_params(d)
    eps0 = Keyword.get(opts, :step_size) || defaults.step_size
    l0 = Keyword.get(opts, :l) || defaults.l
    sigma0 = Tuning.sigma_from_variances([], d, type)

    ctx = %{
      vag_fn: vag_fn,
      d: d,
      type: type,
      integrator: opts[:integrator],
      grads_per_step: Integrator.grads_per_step(opts[:integrator])
    }

    {state, eps, l, sigma, rng, warm} =
      warmup(ctx, state, rng, eps0, l0, sigma0, opts)

    {draws, energies, divergences, state, _rng} =
      sample_phase(ctx, state, rng, eps, l, sigma, opts[:num_samples])

    trace = build_trace(draws, pm, ncp_info)

    n_steps = opts[:num_warmup] + opts[:num_samples]

    stats = %{
      step_size: eps,
      l: l,
      sigma: sigma,
      inv_mass_diag: Nx.multiply(sigma, sigma),
      divergences: divergences,
      eevpd: eevpd_of(energies, d),
      grad_evals: n_steps * ctx.grads_per_step,
      grads_per_draw: ctx.grads_per_step,
      num_warmup: opts[:num_warmup],
      num_samples: opts[:num_samples],
      warmup: warm,
      final_logp: Nx.to_number(state.logp)
    }

    {trace, stats}
  end

  # ------------------------------------------------------------------ warmup

  # Three phases, following blackjax's mclmc_find_L_and_step_size:
  #
  #   1  ε only, moments discarded (the chain is still finding the typical set)
  #   2  ε plus the variance accumulator, which yields both the preconditioner
  #      and the first L; then a short ε re-adjustment because whitening moved
  #      the geometry under it
  #   3  L from the trajectory ESS
  #
  # Whatever warmup budget is left over runs at the settled parameters. That
  # tail is not in blackjax (which sizes warmup to the phases); it is here
  # because `num_warmup` is a user-facing number in this repository and
  # silently ignoring most of it would be a surprise.
  defp warmup(ctx, state, rng, eps0, l0, sigma0, opts) do
    if opts[:num_warmup] <= 0 do
      {state, eps0, l0, sigma0, rng, %{phases: %{}, divergences: 0, nan_steps: 0}}
    else
      do_warmup(ctx, state, rng, eps0, l0, sigma0, opts)
    end
  end

  defp do_warmup(ctx, state, rng, eps0, l0, sigma0, opts) do
    nw = opts[:num_warmup]
    pin_eps? = Keyword.has_key?(opts, :step_size)
    pin_l? = Keyword.has_key?(opts, :l)
    precond? = opts[:diagonal_preconditioning]

    n1 = round(nw * opts[:frac_tune1])
    n2 = round(nw * opts[:frac_tune2])
    n3 = round(nw * opts[:frac_tune3])

    tune = Tuning.eevpd_init(eps0, Keyword.take(opts, [:desired_energy_var]))

    # --- phase 1: epsilon only
    {state, tune, rng, _acc, _dr, div1} =
      steps(ctx, state, rng, eps_or(pin_eps?, tune, eps0), l0, sigma0, n1,
        tune: tune,
        pin_eps?: pin_eps?
      )

    eps1 = if pin_eps?, do: eps0, else: tune.step_size

    # --- phase 2: epsilon + moments
    acc0 = Tuning.moments_init(ctx.d)

    {state, tune, rng, acc, _dr, div2} =
      steps(ctx, state, rng, eps1, l0, sigma0, n2, tune: tune, pin_eps?: pin_eps?, moments: acc0)

    eps2 = if pin_eps?, do: eps0, else: tune.step_size
    vars = Tuning.variances(acc)

    {sigma, l2} =
      cond do
        pin_l? and not precond? ->
          {sigma0, l0}

        vars == [] ->
          {sigma0, l0}

        precond? ->
          {Tuning.sigma_from_variances(vars, ctx.d, ctx.type),
           Tuning.l_from_variances(vars, preconditioned?: true)}

        true ->
          {sigma0, Tuning.l_from_variances(vars)}
      end

    l2 = if pin_l?, do: l0, else: l2

    # --- phase 2b: whitening moved the geometry; re-settle epsilon
    n2b = if precond? and not pin_eps?, do: div(n2, 3), else: 0

    {state, tune, rng, _acc, _dr, div2b} =
      steps(ctx, state, rng, eps2, l2, sigma, n2b, tune: tune, pin_eps?: false)

    eps3 = if pin_eps?, do: eps0, else: tune.step_size

    # --- phase 3: L from the trajectory ESS
    {state, tune, rng, _acc, draws3, div3} =
      steps(ctx, state, rng, eps3, l2, sigma, n3, tune: tune, pin_eps?: pin_eps?, record: true)

    eps4 = if pin_eps?, do: eps0, else: tune.step_size

    l3 =
      if pin_l? do
        l0
      else
        case Tuning.l_from_ess(draws3, eps4, Keyword.take(opts, [:l_factor])) do
          {:ok, l} when l > 0.0 -> l
          _ -> l2
        end
      end

    # --- tail: whatever budget is left, at the settled parameters
    tail = max(nw - (n1 + n2 + n2b + n3), 0)

    {state, _tune, rng, _acc, _dr, div4} =
      steps(ctx, state, rng, eps4, l3, sigma, tail, tune: tune, pin_eps?: true)

    warm = %{
      phases: %{p1: n1, p2: n2, p2b: n2b, p3: n3, tail: tail},
      l_from_variances: l2,
      l_from_ess: l3,
      divergences: div1 + div2 + div2b + div3 + div4,
      nan_steps: tune.nan_steps
    }

    {state, eps4, l3, sigma, rng, warm}
  end

  defp eps_or(true, _tune, eps0), do: eps0
  defp eps_or(false, tune, _eps0), do: tune.step_size

  # One block of `n` MCLMC steps. Returns
  # {state, tune, rng, moments, recorded_draws, divergences}.
  defp steps(_ctx, state, rng, _eps, _l, _sigma, n, opts) when n <= 0 do
    {state, Keyword.fetch!(opts, :tune), rng, Keyword.get(opts, :moments), [], 0}
  end

  defp steps(ctx, state, rng, eps, l, sigma, n, opts) do
    tune0 = Keyword.fetch!(opts, :tune)
    pin_eps? = Keyword.get(opts, :pin_eps?, false)
    record? = Keyword.get(opts, :record, false)
    moments0 = Keyword.get(opts, :moments)

    init = {state, tune0, rng, moments0, [], 0, eps}

    {state, tune, rng, moments, rev_draws, div, _eps} =
      Enum.reduce(1..n, init, fn _i, {st, tune, rng, mom, draws, div, eps} ->
        {st2, de, rng} = one_step(ctx, st, rng, eps, l, sigma)

        {st2, div} =
          if usable?(st2, de) do
            {st2, div}
          else
            # Reject the excursion rather than carry NaN forward. The step
            # size ceiling drops inside eevpd_update.
            {st, div + 1}
          end

        {eps, tune} =
          if pin_eps? do
            {eps, tune}
          else
            Tuning.eevpd_update(tune, de, ctx.d)
          end

        mom = if mom, do: Tuning.moments_update(mom, st2.x, eps), else: nil
        draws = if record?, do: [st2.x | draws], else: draws

        {st2, tune, rng, mom, draws, div, eps}
      end)

    {state, tune, rng, moments, Enum.reverse(rev_draws), div}
  end

  # ---------------------------------------------------------------- sampling

  defp sample_phase(_ctx, state, rng, _eps, _l, _sigma, n) when n <= 0 do
    {[], [], 0, state, rng}
  end

  defp sample_phase(ctx, state, rng, eps, l, sigma, n) do
    init = {state, rng, [], [], 0}

    {state, rng, rev_draws, rev_energies, div} =
      Enum.reduce(1..n, init, fn _i, {st, rng, draws, energies, div} ->
        {st2, de, rng} = one_step(ctx, st, rng, eps, l, sigma)

        {st2, div} = if usable?(st2, de), do: {st2, div}, else: {st, div + 1}

        {st2, rng, [st2.x | draws], [de | energies], div}
      end)

    {Enum.reverse(rev_draws), Enum.reverse(rev_energies), div, state, rng}
  end

  # One MCLMC transition: one integrator step, then partial velocity
  # refreshment at the noise scale L implies. Returns {state, ΔE, rng}.
  defp one_step(ctx, state, rng, eps, l, sigma) do
    logp0 = Nx.to_number(state.logp)

    {state, dk} =
      Integrator.step(state, eps, sigma, ctx.d, ctx.vag_fn, ctx.integrator)

    de = dk - (Nx.to_number(state.logp) - logp0)

    nu = Integrator.nu_for(l, eps, ctx.d)
    {u, rng} = Integrator.partial_refresh(state.u, nu, ctx.d, rng)

    {%{state | u: u}, de, rng}
  end

  defp usable?(state, de) do
    finite?(de) and finite?(Nx.to_number(state.logp))
  end

  defp eevpd_of([], _d), do: 0.0

  defp eevpd_of(energies, d) do
    finite = Enum.filter(energies, &finite?/1)
    n = length(finite)

    if n < 2 do
      0.0
    else
      mean = Enum.sum(finite) / n
      var = Enum.reduce(finite, 0.0, fn e, a -> a + (e - mean) * (e - mean) end) / (n - 1)
      var / d
    end
  end

  # ------------------------------------------------------- shared plumbing

  @doc """
  Normalise the compiled tuple to `{vag_fn, point_map, ncp_info}`.

  `Exmc.Compiler` emits 4-, 5- and 6-tuples depending on which optional slots
  are populated. MCLMC needs the first, third and fourth of them and nothing
  else — which is the whole reason this sampler required no model-layer change.
  """
  @spec unpack(tuple()) :: {fun(), map(), map()}
  def unpack({vag_fn, _step_fn, pm, ncp_info}), do: {vag_fn, pm, ncp_info}
  def unpack({vag_fn, _step_fn, pm, ncp_info, _multi}), do: {vag_fn, pm, ncp_info}
  def unpack({vag_fn, _step_fn, pm, ncp_info, _multi, _meta}), do: {vag_fn, pm, ncp_info}

  @doc """
  Stack unconstrained draws into a constrained-space trace.

  Shared with `Exmc.MAMS`. Mirrors `Exmc.NUTS.Sampler`'s private
  `build_trace/3`, including the non-centred reconstruction, so a trace from
  either sampler is interchangeable with a NUTS one.
  """
  @spec build_trace([Nx.t()], map(), map()) :: map()
  def build_trace([], _pm, _ncp), do: %{}

  def build_trace(draws, pm, ncp_info) do
    stacked = Nx.stack(draws)
    n = elem(Nx.shape(stacked), 0)

    base =
      Map.new(pm.entries, fn entry ->
        value =
          stacked
          |> Nx.slice_along_axis(entry.offset, entry.length, axis: 1)
          |> Nx.reshape(Tuple.insert_at(entry.shape, 0, n))
          |> then(&Transform.apply(entry.transform, &1))

        {entry.id, value}
      end)

    reconstruct_ncp(base, ncp_info)
  end

  defp reconstruct_ncp(trace, ncp),
    do: Exmc.Rewrite.NonCenteredParameterization.reconstruct(trace, ncp)

  defp init_position(_pm, _ncp, init_values, d, rng) when map_size(init_values) == 0 do
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
      Exmc.MCLMC does not yet invert the non-centred reparameterisation for
      user-supplied init_values (Exmc.NUTS.Sampler does, in invert_ncp_init/2).
      Either omit init_values, or compile with `ncp: false`.
      """
    end

    x =
      init_values
      |> PointMap.to_unconstrained(pm)
      |> PointMap.pack(pm)
      |> Nx.backend_copy(Nx.BinaryBackend)

    {x, rng}
  end

  defp empty_stats(opts) do
    %{
      step_size: 0.0,
      l: 0.0,
      divergences: 0,
      grad_evals: 0,
      eevpd: 0.0,
      num_warmup: opts[:num_warmup],
      num_samples: opts[:num_samples]
    }
  end

  defp finite?(x) when is_float(x), do: x == x and x != :infinity and x != :neg_infinity
  defp finite?(x) when is_integer(x), do: true
  defp finite?(_), do: false
end
