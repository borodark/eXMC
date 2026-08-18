defmodule Exmc.MCLMC.Tuning do
  @moduledoc """
  Hyperparameter adaptation for `Exmc.MCLMC` and `Exmc.MAMS`.

  Two knobs, tuned by two different mechanisms.

  ## ε, from the energy error variance per dimension (EEVPD)

  This is **not** dual averaging on an acceptance rate — unadjusted MCLMC has
  no acceptance rate to average. It targets a fixed *energy error variance per
  dimension*:

      Var[ΔE] / d  =  desired_energy_var        (default 5e-4)

  Note what the roadmap brief asked and what the reference implementations
  actually do: the target is a **constant**, not a function of `d`. Dimension
  enters only as the normaliser — `ξ = ΔE²/(d · target)` — so that the same
  constant means the same thing at `d = 2` and `d = 200`. `blackjax`'s
  `mclmc_find_L_and_step_size` defaults to `5e-4`;
  `make_L_step_size_adaptation` defaults to `1e-3` internally. There is no
  `d`-dependent formula in either the code or the papers.

  The update exploits `Var[ΔE] = O(ε⁶)` for a second-order integrator
  (Bou-Rabee & Sanz-Serna 2018), so the optimum is recovered by inverting a
  sixth power:

      ξ       = ΔE²/(d·target) + 1e-8
      weight  = exp(−½·(log ξ / (6·trust))²)          trust = 1.5
      x̄      ← ρ·x̄ + weight·(ξ/ε⁶)                  ρ = (n−1)/(n+1), n = 150
      t       ← ρ·t + weight
      ε       ← (x̄/t)^(−1/6),  capped at ε_max

  The Gaussian `weight` down-weights observations from step sizes far from the
  optimum, which is what stops a single wild `ΔE` from collapsing ε.

  ## ε, for MAMS

  MAMS has an accept step, so it can be tuned the ordinary way. `Exmc.MAMS`
  uses `Exmc.NUTS.StepSize` (the existing dual-averaging module) against an
  acceptance target. The MAMS paper reports 65% as asymptotically optimal and
  ~90% as the practical recommendation; this module's `mams_target_accept/0`
  returns **0.9** for that reason.

  ## L, the decoherence length — two estimators, used in sequence

  1. **From the posterior scale.** `L = ‖sd(x)‖₂ = sqrt(Σᵢ Var(xᵢ))`. Under
     diagonal preconditioning the coordinates are whitened, so this collapses
     to `L = sqrt(d)`. Cheap, and it is what the second warmup phase produces
     as a by-product of estimating the preconditioner.
  2. **From the trajectory autocorrelation.** `L = l_factor · ε · (n / ESS)`
     with `l_factor = 0.4`. This needs ~10 effective samples to be meaningful,
     which is why it is a separate third phase.

  The MAMS paper's ALBA rule is the same shape with a different constant —
  `0.3` without Langevin noise, `0.23` with it — quoted there as empirically
  determined on Gaussian targets.

  ## References

  - `blackjax.adaptation.mclmc_adaptation` — the implementation every constant
    here is taken from.
  - Robnik & Seljak, *Microcanonical Langevin Ensembles*, ICLR 2025.
  - Robnik et al., *Metropolis Adjusted Microcanonical Hamiltonian Monte
    Carlo*, <https://arxiv.org/html/2503.01707> (§ ALBA).
  """

  alias Exmc.NUTS.Vulkan.Validator

  @desired_energy_var 5.0e-4
  @trust_in_estimate 1.5
  @num_effective_samples 150
  @l_factor 0.4
  @nan_shrink 0.8
  @mams_target_accept 0.9

  @doc "The default EEVPD target. A constant — see the moduledoc."
  @spec desired_energy_var() :: float()
  def desired_energy_var, do: @desired_energy_var

  @doc "MAMS's dual-averaging acceptance target (the paper's practical 90%)."
  @spec mams_target_accept() :: float()
  def mams_target_accept, do: @mams_target_accept

  @doc "`l_factor` in `L = l_factor · ε · n/ESS`."
  @spec l_factor() :: float()
  def l_factor, do: @l_factor

  @doc """
  The reference implementations' cold-start parameters:
  `L = sqrt(d)`, `ε = sqrt(d)/4`, identity preconditioner.
  """
  @spec initial_params(pos_integer()) :: %{l: float(), step_size: float()}
  def initial_params(d) do
    %{l: :math.sqrt(d), step_size: :math.sqrt(d) * 0.25}
  end

  @doc """
  Start the EEVPD adaptor at step size `eps`.

  `opts` accepts `:desired_energy_var`, `:trust_in_estimate` and
  `:num_effective_samples` so the bias sweep can move the target and watch the
  bias move with it — which is the whole point of B1.4.
  """
  @spec eevpd_init(float(), keyword()) :: map()
  def eevpd_init(eps, opts \\ []) do
    n_eff = Keyword.get(opts, :num_effective_samples, @num_effective_samples)

    %{
      step_size: eps,
      step_size_max: :infinity,
      time: 0.0,
      x_average: 0.0,
      decay: (n_eff - 1.0) / (n_eff + 1.0),
      target: Keyword.get(opts, :desired_energy_var, @desired_energy_var),
      trust: Keyword.get(opts, :trust_in_estimate, @trust_in_estimate),
      nan_steps: 0
    }
  end

  @doc """
  Fold one observed energy error into the step-size estimate.

  Returns `{step_size, state'}`. A non-finite `delta_energy` is treated as a
  divergence: the step size ceiling drops to `0.8·ε` and the observation is
  discarded rather than folded in as a huge `ξ`.
  """
  @spec eevpd_update(map(), float(), pos_integer()) :: {float(), map()}
  def eevpd_update(state, delta_energy, d) do
    if finite?(delta_energy) do
      eps = state.step_size
      xi = delta_energy * delta_energy / (d * state.target) + 1.0e-8
      weight = :math.exp(-0.5 * :math.pow(:math.log(xi) / (6.0 * state.trust), 2))

      x_average = state.decay * state.x_average + weight * (xi / :math.pow(eps, 6.0))
      time = state.decay * state.time + weight

      new_eps =
        if time > 0.0 and x_average > 0.0 do
          :math.pow(x_average / time, -1.0 / 6.0)
        else
          eps
        end

      new_eps = cap(new_eps, state.step_size_max)

      {new_eps, %{state | step_size: new_eps, x_average: x_average, time: time}}
    else
      shrunk = state.step_size * @nan_shrink

      {shrunk,
       %{state | step_size: shrunk, step_size_max: shrunk, nan_steps: state.nan_steps + 1}}
    end
  end

  defp cap(eps, :infinity), do: eps
  defp cap(eps, max) when eps > max, do: max
  defp cap(eps, _max), do: eps

  @doc """
  Welford-style accumulator for the per-coordinate posterior variance that
  feeds both the preconditioner and the first `L` estimate.

  Weighted by step size, matching blackjax's `incremental_value_update`: the
  unadjusted chain spends unequal *time* per step when ε is still moving, and
  weighting by ε is what makes the accumulated moments an estimate of the
  continuous-time average rather than of the discrete one.
  """
  @spec moments_init(pos_integer()) :: map()
  def moments_init(d) do
    %{
      w: 0.0,
      mean: List.duplicate(0.0, d),
      mean_sq: List.duplicate(0.0, d)
    }
  end

  @spec moments_update(map(), Nx.t(), float()) :: map()
  def moments_update(acc, x, weight) when weight > 0.0 do
    xs = Nx.to_flat_list(x)
    w = acc.w + weight
    r = weight / w

    mean = Enum.zip_with(acc.mean, xs, fn m, v -> m + r * (v - m) end)
    mean_sq = Enum.zip_with(acc.mean_sq, xs, fn m, v -> m + r * (v * v - m) end)

    %{acc | w: w, mean: mean, mean_sq: mean_sq}
  end

  def moments_update(acc, _x, _weight), do: acc

  @doc """
  Per-coordinate variances from the accumulator, floored at a tiny positive
  value so a frozen coordinate cannot produce a zero or negative scale.
  """
  @spec variances(map() | nil) :: [float()]
  # `nil` when the phase that would have accumulated moments was zero-length —
  # a legitimate configuration (frac_tune2: 0), not an error.
  def variances(nil), do: []
  def variances(%{w: w}) when w <= 0.0, do: []

  def variances(acc) do
    Enum.zip_with(acc.mean_sq, acc.mean, fn m2, m -> max(m2 - m * m, 1.0e-12) end)
  end

  @doc """
  `L = sqrt(Σᵢ Var(xᵢ))` — the posterior's total scale.

  Under diagonal preconditioning the coordinates are whitened first, so pass
  `preconditioned?: true` and get `sqrt(d)` instead.
  """
  @spec l_from_variances([float()], keyword()) :: float()
  def l_from_variances(vars, opts \\ []) do
    if Keyword.get(opts, :preconditioned?, false) do
      :math.sqrt(length(vars))
    else
      :math.sqrt(Enum.sum(vars))
    end
  end

  @doc """
  `L = l_factor · ε · mean_i(n / ESS_i)` from a warmup trajectory.

  `draws` is a list of position tensors. ESS is `Exmc.NUTS.Vulkan.Validator`'s
  Geyer estimator, applied per coordinate — the same estimator every other
  gate in this repository is written against, so a change in it moves the
  gates and this together rather than letting them drift apart.

  Returns `{:ok, l}` or `:insufficient` when there are too few draws for the
  autocorrelation to mean anything.
  """
  @spec l_from_ess([Nx.t()], float(), keyword()) :: {:ok, float()} | :insufficient
  def l_from_ess(draws, eps, opts \\ [])

  def l_from_ess(draws, _eps, _opts) when length(draws) < 16, do: :insufficient

  def l_from_ess(draws, eps, opts) do
    factor = Keyword.get(opts, :l_factor, @l_factor)
    n = length(draws)
    columns = draws |> Enum.map(&Nx.to_flat_list/1) |> Enum.zip_with(& &1)

    ratios =
      Enum.map(columns, fn col ->
        ess = Validator.ess(col)
        if ess > 0.0, do: n / ess, else: 1.0
      end)

    case ratios do
      [] -> :insufficient
      _ -> {:ok, factor * eps * (Enum.sum(ratios) / length(ratios))}
    end
  end

  @doc """
  `σ = sqrt(inv_mass_diag)` as a rank-1 `BinaryBackend` tensor.

  This is the `sigma` the integrator expects. Identity when `vars` is empty.
  """
  @spec sigma_from_variances([float()], pos_integer(), Nx.Type.t()) :: Nx.t()
  def sigma_from_variances([], d, type) do
    Nx.broadcast(Nx.tensor(1.0, type: type, backend: Nx.BinaryBackend), {d})
  end

  def sigma_from_variances(vars, _d, type) do
    Nx.tensor(Enum.map(vars, &:math.sqrt/1), type: type, backend: Nx.BinaryBackend)
  end

  defp finite?(x) when is_float(x), do: x == x and x != :infinity and x != :neg_infinity
  defp finite?(x) when is_integer(x), do: true
  defp finite?(_), do: false
end
