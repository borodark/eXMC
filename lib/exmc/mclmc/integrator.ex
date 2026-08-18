defmodule Exmc.MCLMC.Integrator do
  @moduledoc """
  The isokinetic (ESH-dynamics) integrator that MCLMC and MAMS are both built on.

  ## What is different from leapfrog

  The state is a position `x` and a **unit** velocity `u`. Where a Hamiltonian
  half-step does `p ← p + (ε/2)·∇log p`, the isokinetic velocity update applies
  the gradient as a *rotation on the unit sphere*, so `‖u‖ = 1` is preserved by
  construction rather than approximately. That is the one exact invariant this
  module has and `test/mclmc/integrator_test.exs` asserts it to 1e-13 after
  10⁴ steps.

  ## The velocity update

  For `g = ∇log p(x)`, `e = g/‖g‖`, `w = u·e` and `δ = ε‖g‖/(d−1)`:

      u' = [ u + e·(sinh δ + w·(cosh δ − 1)) ] / [ cosh δ + w·sinh δ ]

  This module does **not** evaluate that form. It evaluates the algebraically
  identical `ζ = e^{−δ}` form used by the reference implementation
  (`JakobRobnik/MicroCanonicalHMC`, `mclmc/dynamics.py:update_momentum`, and
  `blackjax.mcmc.integrators.esh_dynamics_momentum_update_one_step`):

      ζ  = exp(−δ)
      ũ  = e·(1−ζ)·(1 + ζ + w·(1−ζ)) + 2ζ·u
      u' = ũ / ‖ũ‖

  because `sinh`/`cosh` overflow for large `‖g‖` and the `ζ` form does not.
  The two agree exactly: dividing `ũ` by `2ζ` gives
  `u + e·(sinh δ + w·(cosh δ − 1))`, since `(1−ζ)(1+ζ)/2ζ = sinh δ` and
  `(1−ζ)²/2ζ = cosh δ − 1`; and `‖u + e·A‖ = cosh δ + w·sinh δ` follows from
  `‖u‖ = 1` by expanding the square. `kinetic_change_closed_form/3` evaluates
  the `sinh`/`cosh` form so a test can pin the identity.

  Note the sign convention. The reference implementation carries `g = ∇(−log p)`
  and writes `e = −g/‖g‖`; blackjax carries `g = ∇log p` and writes
  `e = +g/‖g‖`. This module takes the **log-density** gradient — the same thing
  `Exmc.Compiler`'s `vag_fn` returns — so `e = +g/‖g‖`.

  ## The kinetic energy change

  The isokinetic map is not volume-preserving on `(x, u)`; the compensating
  Jacobian factor is what MAMS's Metropolis ratio needs. It is exactly

      ΔK = (d−1)·log(cosh δ + w·sinh δ)

  which is `(d−1)·log` of the normalising denominator above. Evaluated in the
  stable form as `(d−1)·(δ − log 2 + log(1 + w + (1−w)ζ²))`.

  The total energy is `E = −log p(x) + K`, so the per-step energy error is
  `ΔE = ΔK − (log p(x') − log p(x))`.

  ## The splitting

  The papers do not use plain Verlet. They use the **minimal-norm** second-order
  integrator of Takaishi & de Forcrand (hep-lat/0505020, Eq. 20), a palindromic
  `V T V T V` splitting with

      λ_c = 0.1931833275037836

  applied as `V(λ_c ε) T(ε/2) V((1−2λ_c) ε) T(ε/2) V(λ_c ε)`. This is the same
  coefficient blackjax calls `mclachlan` (`b1`), and it costs **two** gradient
  evaluations per step against leapfrog's one. `leapfrog_step/5` (`V T V`) is
  provided so the bias sweep can price the difference.

  ## Preconditioning

  `sigma` is the per-coordinate scale — `sqrt(inv_mass_diag)`. The velocity
  update sees `sigma * grad` and the position update moves by
  `ε * sigma * u`, matching `forward_L`/`adjoint_L` in blackjax's diagonal case.

  ## References

  - Robnik, De Luca, Silverstein, Seljak, *Microcanonical Hamiltonian Monte
    Carlo*, JMLR 24 (2023). <https://arxiv.org/abs/2212.08549>
  - Robnik & Seljak, *Microcanonical Langevin Ensembles*, ICLR 2025.
    <https://arxiv.org/abs/2502.06335>
  - Robnik et al., *Metropolis Adjusted Microcanonical Hamiltonian Monte Carlo*.
    <https://arxiv.org/html/2503.01707>
  - Reference implementation: <https://github.com/JakobRobnik/MicroCanonicalHMC>
  """

  # Critical value of the lambda parameter for the minimal-norm integrator.
  # Takaishi & de Forcrand (2006); blackjax's `mclachlan` b1.
  @lambda_c 0.1931833275037836

  @doc "The minimal-norm integrator's λ coefficient."
  @spec lambda_c() :: float()
  def lambda_c, do: @lambda_c

  @doc """
  Gradient evaluations consumed by one step of the named integrator.

  Every ESS/gradient number in `bench_results/` is divided by this, because
  the minimal-norm integrator buys its accuracy with a second gradient and a
  comparison that ignores that is not a comparison.
  """
  @spec grads_per_step(:minimal_norm | :leapfrog) :: pos_integer()
  def grads_per_step(:minimal_norm), do: 2
  def grads_per_step(:leapfrog), do: 1

  @doc """
  One isokinetic velocity update.

  Takes the **log-density** gradient `g` (already multiplied by `sigma` if
  preconditioning), the current unit velocity `u`, the effective step `eps`
  (already multiplied by the splitting coefficient) and the dimension `d`.

  Returns `{u', kinetic_change}` with `u'` renormalised so `‖u'‖ = 1` to
  machine precision. The renormalisation is not cosmetic: dividing by the
  closed-form denominator alone lets `‖u‖` drift, and the drift is exactly the
  class of error the 1e-13 gate exists to catch.
  """
  @spec velocity_step(Nx.t(), Nx.t(), float(), pos_integer()) :: {Nx.t(), float()}
  def velocity_step(u, g, eps, d) when is_integer(d) and d > 1 do
    g_norm = norm(g)

    # At a stationary point the direction e is undefined. The dynamics'
    # limit there is "do nothing", which is also what δ → 0 gives.
    if g_norm == 0.0 or not finite?(g_norm) do
      {u, 0.0}
    else
      e = Nx.divide(g, c(g_norm, g))
      w = Nx.to_number(Nx.dot(u, e))
      delta = eps * g_norm / (d - 1)
      zeta = :math.exp(-delta)

      u_raw =
        Nx.add(
          Nx.multiply(e, c((1.0 - zeta) * (1.0 + zeta + w * (1.0 - zeta)), e)),
          Nx.multiply(u, c(2.0 * zeta, u))
        )

      u_new = normalize(u_raw)

      # (d-1)·log(cosh δ + w·sinh δ), in the form that does not overflow.
      inner = max(1.0 + w + (1.0 - w) * zeta * zeta, 1.0e-300)
      dk = (delta - :math.log(2.0) + :math.log(inner)) * (d - 1)

      {u_new, dk}
    end
  end

  def velocity_step(_u, _g, _eps, d) do
    raise ArgumentError,
          "MCLMC's isokinetic dynamics divide by (d-1) and are undefined for d < 2; got d=#{d}"
  end

  @doc """
  `ΔK = (d−1)·log(cosh δ + w·sinh δ)`, evaluated directly.

  Present so a test can pin the identity between this and the `ζ` form
  `velocity_step/4` actually uses. Overflows for large `δ`; that is the point.
  """
  @spec kinetic_change_closed_form(float(), float(), pos_integer()) :: float()
  def kinetic_change_closed_form(delta, w, d) do
    (d - 1) * :math.log(:math.cosh(delta) + w * :math.sinh(delta))
  end

  @doc """
  `u' = [u + e·(sinh δ + w·(cosh δ − 1))] / [cosh δ + w·sinh δ]`, evaluated
  directly. Present for the same reason as `kinetic_change_closed_form/3`.
  """
  @spec velocity_step_closed_form(Nx.t(), Nx.t(), float(), pos_integer()) :: {Nx.t(), float()}
  def velocity_step_closed_form(u, g, eps, d) do
    g_norm = norm(g)
    e = Nx.divide(g, c(g_norm, g))
    w = Nx.to_number(Nx.dot(u, e))
    delta = eps * g_norm / (d - 1)

    a = :math.sinh(delta) + w * (:math.cosh(delta) - 1.0)
    denom = :math.cosh(delta) + w * :math.sinh(delta)

    u_new = Nx.divide(Nx.add(u, Nx.multiply(e, c(a, e))), c(denom, u))
    {u_new, (d - 1) * :math.log(denom)}
  end

  @doc """
  One drift (position) update: `x ← x + ε·σ·u`, then re-evaluate the model.

  Returns `{x', logp', grad'}` with both pulled onto `Nx.BinaryBackend`, the
  same convention `Exmc.NUTS.Sampler` uses so the scalar arithmetic downstream
  never round-trips through a device.
  """
  @spec position_step(Nx.t(), Nx.t(), float(), Nx.t(), (Nx.t() -> {Nx.t(), Nx.t()})) ::
          {Nx.t(), Nx.t(), Nx.t()}
  def position_step(x, u, eps, sigma, vag_fn) do
    x_new = Nx.add(x, Nx.multiply(Nx.multiply(u, sigma), c(eps, x)))
    {logp, grad} = vag_fn.(x_new)

    {x_new, Nx.backend_copy(logp, Nx.BinaryBackend), Nx.backend_copy(grad, Nx.BinaryBackend)}
  end

  @doc """
  One minimal-norm (`V T V T V`) step. **Two** gradient evaluations.

  `state` is `%{x:, u:, logp:, grad:}`. Returns `{state', kinetic_change}`.
  """
  @spec minimal_norm_step(map(), float(), Nx.t(), pos_integer(), (Nx.t() -> {Nx.t(), Nx.t()})) ::
          {map(), float()}
  def minimal_norm_step(%{x: x, u: u, grad: grad}, eps, sigma, d, vag_fn) do
    lam = @lambda_c

    {u, r1} = velocity_step(u, Nx.multiply(grad, sigma), eps * lam, d)
    {x, _l, grad} = position_step(x, u, 0.5 * eps, sigma, vag_fn)
    {u, r2} = velocity_step(u, Nx.multiply(grad, sigma), eps * (1.0 - 2.0 * lam), d)
    {x, logp, grad} = position_step(x, u, 0.5 * eps, sigma, vag_fn)
    {u, r3} = velocity_step(u, Nx.multiply(grad, sigma), eps * lam, d)

    {%{x: x, u: u, logp: logp, grad: grad}, r1 + r2 + r3}
  end

  @doc """
  One leapfrog (`V T V`) step. **One** gradient evaluation.

  Second-order like the minimal-norm splitting but with a larger error
  constant. Kept so the bias sweep can price the extra gradient rather than
  assume it.
  """
  @spec leapfrog_step(map(), float(), Nx.t(), pos_integer(), (Nx.t() -> {Nx.t(), Nx.t()})) ::
          {map(), float()}
  def leapfrog_step(%{x: x, u: u, grad: grad}, eps, sigma, d, vag_fn) do
    {u, r1} = velocity_step(u, Nx.multiply(grad, sigma), 0.5 * eps, d)
    {x, logp, grad} = position_step(x, u, eps, sigma, vag_fn)
    {u, r2} = velocity_step(u, Nx.multiply(grad, sigma), 0.5 * eps, d)

    {%{x: x, u: u, logp: logp, grad: grad}, r1 + r2}
  end

  @doc "Dispatch by integrator name. `:minimal_norm` is the papers' default."
  @spec step(map(), float(), Nx.t(), pos_integer(), (Nx.t() -> {Nx.t(), Nx.t()}), atom()) ::
          {map(), float()}
  def step(state, eps, sigma, d, vag_fn, :minimal_norm),
    do: minimal_norm_step(state, eps, sigma, d, vag_fn)

  def step(state, eps, sigma, d, vag_fn, :leapfrog),
    do: leapfrog_step(state, eps, sigma, d, vag_fn)

  @doc """
  Run `k` steps, accumulating the energy error.

  `ΔE = ΔK − Δlog p`. Returns `{state', delta_energy}`.
  """
  @spec run(
          map(),
          pos_integer(),
          float(),
          Nx.t(),
          pos_integer(),
          (Nx.t() -> {Nx.t(), Nx.t()}),
          atom()
        ) ::
          {map(), float()}
  def run(state, k, eps, sigma, d, vag_fn, integrator \\ :minimal_norm) when k >= 0 do
    logp0 = Nx.to_number(state.logp)

    {final, kinetic} =
      Enum.reduce(1..k//1, {state, 0.0}, fn _i, {st, acc} ->
        {st, dk} = step(st, eps, sigma, d, vag_fn, integrator)
        {st, acc + dk}
      end)

    {final, kinetic - (Nx.to_number(final.logp) - logp0)}
  end

  @doc """
  A uniformly random unit vector in `d` dimensions, using Erlang's `:rand`.

  `:rand` rather than `Nx.Random` for the same reason
  `Exmc.NUTS.Sampler.sample_momentum_fast/2` uses it: `Nx.Random.split/normal`
  on `BinaryBackend` costs more than the dynamics do.
  """
  @spec random_unit(pos_integer(), :rand.state()) :: {Nx.t(), :rand.state()}
  def random_unit(d, rng) do
    {vals, rng} =
      Enum.map_reduce(1..d, rng, fn _i, r ->
        {z, r} = :rand.normal_s(r)
        {z, r}
      end)

    n = :math.sqrt(Enum.reduce(vals, 0.0, fn v, a -> a + v * v end))

    u =
      Nx.tensor(Enum.map(vals, &(&1 / n)), type: Exmc.JIT.precision(), backend: Nx.BinaryBackend)

    {u, rng}
  end

  @doc """
  Partial velocity refreshment: `u ← (u + ν·z)/‖u + ν·z‖`, `z ~ N(0, I_d)`.

  This is the "Langevin" in Microcanonical Langevin Monte Carlo — the
  decoherence that stops the trajectory from being periodic. `nu_for/3`
  gives the ν that decorrelates over a trajectory of length `L`.
  """
  @spec partial_refresh(Nx.t(), float(), pos_integer(), :rand.state()) :: {Nx.t(), :rand.state()}
  def partial_refresh(u, nu, d, rng) when nu > 0.0 do
    {vals, rng} =
      Enum.map_reduce(1..d, rng, fn _i, r ->
        {z, r} = :rand.normal_s(r)
        {z * nu, r}
      end)

    z = Nx.tensor(vals, type: Nx.type(u), backend: Nx.BinaryBackend)
    uu = Nx.add(u, z)
    {normalize(uu), rng}
  end

  def partial_refresh(u, _nu, _d, rng), do: {u, rng}

  @doc """
  `ν = sqrt((exp(2ε/L) − 1)/d)`, the MCHMC paper's noise scale for a
  decoherence length `L` at step size `ε`. `L = :inf` means no refreshment.
  """
  @spec nu_for(float() | :infinity, float(), pos_integer()) :: float()
  def nu_for(:infinity, _eps, _d), do: 0.0

  def nu_for(l, eps, d) when l > 0.0 do
    :math.sqrt((:math.exp(2.0 * eps / l) - 1.0) / d)
  end

  def nu_for(_l, _eps, _d), do: 0.0

  @doc "Euclidean norm of a rank-1 tensor, as a float."
  @spec norm(Nx.t()) :: float()
  def norm(t), do: Nx.to_number(Nx.sqrt(Nx.sum(Nx.multiply(t, t))))

  @doc "`t / ‖t‖`, computed without ever putting a bare float into an Nx op."
  @spec normalize(Nx.t()) :: Nx.t()
  def normalize(t), do: Nx.divide(t, c(norm(t), t))

  # Nx silently computes at f32 when a bare Elixir float meets an f64 tensor:
  #
  #     Nx.divide(f64_tensor, 0.9695359714832659)   # -> f32-accurate result
  #
  # The scalar becomes a default-typed ({:f, 32}) tensor and the promotion
  # widens *after* the arithmetic, not before. Measured here on
  # Nx.BinaryBackend: the 1e-13 invariant gate in
  # `test/mclmc/integrator_test.exs` failed at 3e-8, which is exactly f32
  # epsilon and looked for all the world like an algebra error.
  #
  # So every scalar that meets a tensor in this module goes through `c/2`,
  # which builds it at the tensor's own type. `Nx.BinaryBackend` is
  # hard-coded because that is where the whole MCLMC loop lives by
  # construction (`position_step/5` copies `vag_fn`'s outputs onto it, the
  # same convention `Exmc.NUTS.Sampler` uses), and mixing backends in one
  # binary op raises.
  @compile {:inline, c: 2}
  defp c(v, like), do: Nx.tensor(v, type: Nx.type(like), backend: Nx.BinaryBackend)

  defp finite?(x) when is_float(x), do: x == x and x != :infinity and x != :neg_infinity
  defp finite?(_), do: false
end
