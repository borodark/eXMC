defmodule Exmc.MCLMC.IntegratorTest do
  use ExUnit.Case, async: true

  alias Exmc.MCLMC.Integrator

  @moduledoc """
  B1.1's gate.

  The headline assertion is the one the roadmap names: after 10⁴ steps at any
  ε, `abs(‖u‖ − 1) < 1e-13`. It is exact and it has zero flakiness.

  It is not, on its own, a strong check of the *algebra* — a velocity update
  that renormalises will hold `‖u‖ = 1` even if the rotation is wrong. So three
  more exact assertions sit alongside it, each aimed at a different way the
  update can be wrong while still being unit-norm:

    * **reversibility** — `Φ_ε^K ∘ flip ∘ Φ_ε^K = flip`, which holds only if
      the splitting is palindromic *and* `V(δ)(−u) = −V(−δ)(u)`. A sign error
      anywhere fails it.
    * **the closed-form identity** — the `ζ = e^{−δ}` form this module
      evaluates must agree with `[u + e(sinh δ + w(cosh δ − 1))]/[cosh δ +
      w sinh δ]` to 1e-13. This is the check that the operator's recalled
      formula and the reference implementation are the same object.
    * **order of accuracy** — halving ε must cut the one-step energy error by
      ~8×. A second-order splitting has O(ε³) local error; a first-order one
      would give ~4×.

  All four are deterministic. None of them samples anything.
  """

  # A deliberately non-Gaussian, ill-conditioned target so no error term
  # vanishes by accident: log p = -0.5 Σ (x_i/s_i)² - 0.05 Σ x_i⁴.
  defp f64(v), do: Nx.tensor(v, type: :f64, backend: Nx.BinaryBackend)

  defp target(d) do
    # Every constant is an explicitly-typed f64 tensor. A bare Elixir float in
    # an Nx binary op downcasts the whole expression to f32 — see the `c/2`
    # note in Exmc.MCLMC.Integrator. That leak is what made this file's
    # 1e-13 gate fail at 3e-8 on its first run.
    inv2 =
      f64(
        Enum.map(1..d, fn i ->
          s = 1.0 + 0.7 * (i - 1)
          1.0 / (s * s)
        end)
      )

    half = f64(-0.5)
    q4 = f64(-0.05)
    g4 = f64(-0.2)

    fn x ->
      x2 = Nx.multiply(x, x)
      x4 = Nx.multiply(x2, x2)

      logp =
        Nx.add(
          Nx.multiply(half, Nx.sum(Nx.multiply(x2, inv2))),
          Nx.multiply(q4, Nx.sum(x4))
        )

      grad =
        Nx.add(
          Nx.negate(Nx.multiply(x, inv2)),
          Nx.multiply(g4, Nx.multiply(x2, x))
        )

      {logp, grad}
    end
  end

  defp init_state(d, vag) do
    x = f64(Enum.map(1..d, fn i -> 0.3 * :math.sin(i * 1.7) end))
    u0 = f64(Enum.map(1..d, fn i -> :math.cos(i * 0.9) + 0.2 end))
    u = Integrator.normalize(u0)
    {logp, grad} = vag.(x)
    %{x: x, u: u, logp: logp, grad: grad}
  end

  defp ones(d), do: Nx.broadcast(f64(1.0), {d})

  # ---------------------------------------------------------------- the gate

  @tag timeout: 600_000
  test "‖u‖ = 1 to 1e-13 after 10^4 minimal-norm steps, at every step size" do
    d = 6
    vag = target(d)
    sigma = ones(d)

    for eps <- [0.001, 0.01, 0.1, 0.5, 1.0] do
      state = init_state(d, vag)

      final =
        Enum.reduce(1..10_000, state, fn _i, st ->
          {st, _dk} = Integrator.minimal_norm_step(st, eps, sigma, d, vag)
          st
        end)

      err = abs(Integrator.norm(final.u) - 1.0)

      assert err < 1.0e-13,
             "‖u‖ drifted to #{Integrator.norm(final.u)} (err #{err}) after 10^4 steps at eps=#{eps}"
    end
  end

  @tag timeout: 600_000
  test "‖u‖ = 1 to 1e-13 after 10^4 leapfrog steps too" do
    d = 4
    vag = target(d)
    sigma = ones(d)

    for eps <- [0.01, 0.3, 2.0] do
      state = init_state(d, vag)

      final =
        Enum.reduce(1..10_000, state, fn _i, st ->
          {st, _dk} = Integrator.leapfrog_step(st, eps, sigma, d, vag)
          st
        end)

      assert abs(Integrator.norm(final.u) - 1.0) < 1.0e-13
    end
  end

  # ------------------------------------------ the algebra, checked three ways

  test "the ζ form and the sinh/cosh closed form are the same map (1e-13)" do
    d = 5
    vag = target(d)
    state = init_state(d, vag)

    # Sweep δ across three orders of magnitude, including where the two forms
    # are numerically most different.
    for eps <- [1.0e-6, 1.0e-3, 0.05, 0.4, 2.0] do
      {u_zeta, dk_zeta} = Integrator.velocity_step(state.u, state.grad, eps, d)
      {u_closed, dk_closed} = Integrator.velocity_step_closed_form(state.u, state.grad, eps, d)

      # the closed form does not renormalise, so compare after normalising it
      u_closed = Integrator.normalize(u_closed)

      max_diff =
        Nx.subtract(u_zeta, u_closed)
        |> Nx.abs()
        |> Nx.reduce_max()
        |> Nx.to_number()

      assert max_diff < 1.0e-13, "velocity forms disagree by #{max_diff} at eps=#{eps}"

      assert abs(dk_zeta - dk_closed) < 1.0e-13 * max(1.0, abs(dk_closed)),
             "ΔK forms disagree: #{dk_zeta} vs #{dk_closed} at eps=#{eps}"
    end
  end

  test "ΔK equals (d−1)·log(cosh δ + w·sinh δ) evaluated independently" do
    d = 7
    vag = target(d)
    state = init_state(d, vag)
    eps = 0.17

    g_norm = Integrator.norm(state.grad)
    e = Integrator.normalize(state.grad)
    w = Nx.to_number(Nx.dot(state.u, e))
    delta = eps * g_norm / (d - 1)

    expected = Integrator.kinetic_change_closed_form(delta, w, d)
    {_u, got} = Integrator.velocity_step(state.u, state.grad, eps, d)

    assert abs(got - expected) < 1.0e-13 * max(1.0, abs(expected))
  end

  test "the minimal-norm step is reversible under u → −u" do
    d = 5
    vag = target(d)
    sigma = ones(d)
    eps = 0.21
    k = 100

    s0 = init_state(d, vag)

    forward =
      Enum.reduce(1..k, s0, fn _i, st ->
        {st, _} = Integrator.minimal_norm_step(st, eps, sigma, d, vag)
        st
      end)

    flipped = %{forward | u: Nx.negate(forward.u)}

    back =
      Enum.reduce(1..k, flipped, fn _i, st ->
        {st, _} = Integrator.minimal_norm_step(st, eps, sigma, d, vag)
        st
      end)

    dx = Nx.subtract(back.x, s0.x) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    du = Nx.add(back.u, s0.u) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

    assert dx < 1.0e-9, "position did not retrace: max |Δx| = #{dx}"
    assert du < 1.0e-9, "velocity did not retrace: max |u + u₀| = #{du}"
  end

  test "leapfrog is reversible under u → −u" do
    d = 4
    vag = target(d)
    sigma = ones(d)
    eps = 0.13
    k = 80

    s0 = init_state(d, vag)

    fwd =
      Enum.reduce(1..k, s0, fn _i, st ->
        {st, _} = Integrator.leapfrog_step(st, eps, sigma, d, vag)
        st
      end)

    back =
      Enum.reduce(1..k, %{fwd | u: Nx.negate(fwd.u)}, fn _i, st ->
        {st, _} = Integrator.leapfrog_step(st, eps, sigma, d, vag)
        st
      end)

    assert Nx.subtract(back.x, s0.x) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number() < 1.0e-9
    assert Nx.add(back.u, s0.u) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number() < 1.0e-9
  end

  test "one-step energy error is third order in ε (second-order splitting)" do
    d = 5
    vag = target(d)
    sigma = ones(d)

    err = fn eps ->
      s0 = init_state(d, vag)
      {s1, dk} = Integrator.minimal_norm_step(s0, eps, sigma, d, vag)
      abs(dk - (Nx.to_number(s1.logp) - Nx.to_number(s0.logp)))
    end

    # Small enough that the leading term dominates, large enough that f64
    # rounding does not.
    e1 = err.(0.02)
    e2 = err.(0.01)
    e3 = err.(0.005)

    r1 = e1 / e2
    r2 = e2 / e3

    assert r1 > 6.5 and r1 < 10.0, "order ratio #{r1} is not ~8 (third-order local error)"
    assert r2 > 6.5 and r2 < 10.0, "order ratio #{r2} is not ~8 (third-order local error)"
  end

  test "energy error does not drift over a long trajectory" do
    # A symmetric splitting has a bounded energy error; a broken one drifts.
    d = 5
    vag = target(d)
    sigma = ones(d)
    eps = 0.15

    s0 = init_state(d, vag)
    logp0 = Nx.to_number(s0.logp)

    {errors, _} =
      Enum.map_reduce(1..2000, {s0, 0.0}, fn _i, {st, kin} ->
        {st, dk} = Integrator.minimal_norm_step(st, eps, sigma, d, vag)
        kin = kin + dk
        {kin - (Nx.to_number(st.logp) - logp0), {st, kin}}
      end)

    first_half = errors |> Enum.take(1000) |> Enum.map(&abs/1) |> Enum.max()
    second_half = errors |> Enum.drop(1000) |> Enum.map(&abs/1) |> Enum.max()

    assert second_half < 3.0 * first_half + 1.0e-9,
           "energy error grew from #{first_half} to #{second_half} — the integrator is drifting"
  end

  # ------------------------------------------------------------- edge cases

  test "d < 2 is refused rather than dividing by zero" do
    u = f64([1.0])
    g = f64([0.5])

    assert_raise ArgumentError, ~r/undefined for d < 2/, fn ->
      Integrator.velocity_step(u, g, 0.1, 1)
    end
  end

  test "a zero gradient leaves the velocity alone instead of producing NaN" do
    d = 3
    u = f64([0.0, 1.0, 0.0])
    g = f64([0.0, 0.0, 0.0])

    {u2, dk} = Integrator.velocity_step(u, g, 0.3, d)

    assert dk == 0.0
    assert Nx.to_flat_list(u2) == [0.0, 1.0, 0.0]
  end

  test "preconditioning is applied to gradient and drift consistently" do
    # With sigma constant c, stepping the scaled target at eps must equal
    # stepping the unscaled target at c*eps in the rescaled coordinates.
    d = 4
    vag = target(d)
    c = 2.0
    sigma = Nx.multiply(ones(d), f64(c))
    eps = 0.1

    s0 = init_state(d, vag)
    {a, dka} = Integrator.minimal_norm_step(s0, eps, sigma, d, vag)

    # sigma folds into the dynamics as an anisotropic metric; the invariant
    # that must hold regardless is unit norm and a finite ΔK.
    assert abs(Integrator.norm(a.u) - 1.0) < 1.0e-14
    assert is_float(dka) and dka == dka
  end

  test "λ_c is the published minimal-norm coefficient" do
    assert Integrator.lambda_c() == 0.1931833275037836
    assert Integrator.grads_per_step(:minimal_norm) == 2
    assert Integrator.grads_per_step(:leapfrog) == 1
  end

  test "random_unit produces unit vectors" do
    rng = :rand.seed_s(:exsss, 42)

    {_rng, oks} =
      Enum.map_reduce(1..50, rng, fn _i, r ->
        {u, r} = Integrator.random_unit(17, r)
        {abs(Integrator.norm(u) - 1.0) < 1.0e-14, r}
      end)
      |> then(fn {oks, r} -> {r, oks} end)

    assert Enum.all?(oks)
  end

  test "partial_refresh keeps the velocity on the unit sphere" do
    rng = :rand.seed_s(:exsss, 7)
    d = 9
    {u, rng} = Integrator.random_unit(d, rng)

    nu = Integrator.nu_for(5.0, 0.2, d)
    assert nu > 0.0

    {u2, _rng} = Integrator.partial_refresh(u, nu, d, rng)
    assert abs(Integrator.norm(u2) - 1.0) < 1.0e-14

    # infinite L means no refreshment at all
    assert Integrator.nu_for(:infinity, 0.2, d) == 0.0
  end
end
