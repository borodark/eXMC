defmodule Exmc.MCLMC.TuningTest do
  use ExUnit.Case, async: true

  @moduledoc """
  Unit tests for the adaptation arithmetic, with no sampling involved.

  These are the cheap exact checks that catch an algebra error before it turns
  into a subtly-wrong posterior. Every constant asserted here is pinned against
  `blackjax.adaptation.mclmc_adaptation`, so if one of them changes the change
  is deliberate.
  """

  alias Exmc.MCLMC.Tuning

  test "the published constants are what the reference implementations use" do
    assert Tuning.desired_energy_var() == 5.0e-4
    assert Tuning.l_factor() == 0.4
    assert Tuning.mams_target_accept() == 0.9
  end

  test "cold-start parameters are L = sqrt(d), eps = sqrt(d)/4" do
    for d <- [2, 8, 100] do
      %{l: l, step_size: eps} = Tuning.initial_params(d)
      assert_in_delta l, :math.sqrt(d), 1.0e-12
      assert_in_delta eps, :math.sqrt(d) * 0.25, 1.0e-12
    end
  end

  test "a too-large energy error drives the step size down, a too-small one up" do
    d = 10
    eps0 = 0.5
    target = Tuning.desired_energy_var()

    # An error far above target: xi >> 1.
    big = :math.sqrt(100.0 * d * target)
    {eps_down, _} = Tuning.eevpd_update(Tuning.eevpd_init(eps0), big, d)

    # An error far below target: xi << 1.
    small = :math.sqrt(0.01 * d * target)
    {eps_up, _} = Tuning.eevpd_update(Tuning.eevpd_init(eps0), small, d)

    assert eps_down < eps0, "an energy error 100x the target should shrink eps"
    assert eps_up > eps0, "an energy error 1/100 of the target should grow eps"
  end

  test "an on-target energy error is a fixed point of the step size update" do
    d = 16
    eps0 = 0.37
    on_target = :math.sqrt(d * Tuning.desired_energy_var())

    # One observation exactly on target reproduces eps up to the `+1e-8`
    # log(0) guard inside xi, which is a documented constant and not slack:
    # xi = 1 + 1e-8, so eps' = eps·(1+1e-8)^(-1/6). Asserting the exact value
    # rather than a loosened delta keeps this a real check on the algebra.
    expected = eps0 * :math.pow(1.0 + 1.0e-8, -1.0 / 6.0)

    {eps1, _} = Tuning.eevpd_update(Tuning.eevpd_init(eps0), on_target, d)
    assert abs(eps1 - expected) < 1.0e-15 * eps0
  end

  test "the step size update inverts a sixth power, not some other power" do
    # If the true error variance scales as eps^6, then feeding an error
    # measured at 2*eps_opt (so xi = 2^6 = 64) must return eps_opt exactly.
    d = 8
    eps = 0.4
    xi = 64.0
    de = :math.sqrt(xi * d * Tuning.desired_energy_var())

    {eps1, _} = Tuning.eevpd_update(Tuning.eevpd_init(eps), de, d)

    # xi/eps^6 with weight w gives x_average/time = xi/eps^6, so
    # eps' = eps/xi^(1/6) = eps/2.
    assert_in_delta eps1, eps / 2.0, 1.0e-9
  end

  test "a non-finite energy error shrinks the ceiling instead of poisoning the average" do
    st0 = Tuning.eevpd_init(1.0)
    {eps, st} = Tuning.eevpd_update(st0, :nan, 5)

    assert eps == 0.8
    assert st.step_size_max == 0.8
    assert st.nan_steps == 1
    # the running averages must be untouched
    assert st.x_average == st0.x_average
    assert st.time == st0.time

    # and the ceiling must bind on a subsequent healthy observation
    on_target = :math.sqrt(5 * Tuning.desired_energy_var())
    {eps2, _} = Tuning.eevpd_update(st, on_target * 1.0e-6, 5)
    assert eps2 <= 0.8
  end

  test "the moment accumulator recovers mean and variance" do
    xs = for i <- 1..500, do: [:math.sin(i * 0.31) * 3.0, 10.0 + :math.cos(i * 0.17)]

    acc =
      Enum.reduce(xs, Tuning.moments_init(2), fn v, a ->
        Tuning.moments_update(a, Nx.tensor(v, type: :f64, backend: Nx.BinaryBackend), 1.0)
      end)

    [m0, m1] = acc.mean
    [v0, v1] = Tuning.variances(acc)

    ref_mean = fn col -> Enum.sum(col) / length(col) end

    ref_var = fn col ->
      m = ref_mean.(col)
      Enum.reduce(col, 0.0, fn x, a -> a + (x - m) * (x - m) end) / length(col)
    end

    c0 = Enum.map(xs, &hd/1)
    c1 = Enum.map(xs, &List.last/1)

    assert_in_delta m0, ref_mean.(c0), 1.0e-10
    assert_in_delta m1, ref_mean.(c1), 1.0e-10
    assert_in_delta v0, ref_var.(c0), 1.0e-9
    assert_in_delta v1, ref_var.(c1), 1.0e-9
  end

  test "L from variances is the total posterior scale, or sqrt(d) once whitened" do
    vars = [4.0, 9.0, 1.0]
    assert_in_delta Tuning.l_from_variances(vars), :math.sqrt(14.0), 1.0e-12
    assert_in_delta Tuning.l_from_variances(vars, preconditioned?: true), :math.sqrt(3), 1.0e-12
  end

  test "sigma is the square root of the variance, and identity when there is none" do
    sigma = Tuning.sigma_from_variances([4.0, 0.25], 2, :f64)
    assert Nx.to_flat_list(sigma) == [2.0, 0.5]

    assert Nx.to_flat_list(Tuning.sigma_from_variances([], 3, :f64)) == [1.0, 1.0, 1.0]
  end

  test "L from ESS refuses to guess from too few draws" do
    few = for i <- 1..8, do: Nx.tensor([i * 1.0], type: :f64, backend: Nx.BinaryBackend)
    assert Tuning.l_from_ess(few, 0.1) == :insufficient
  end

  test "L from ESS scales with the correlation time" do
    # An i.i.d. sequence has n/ESS ~= 1; a heavily autocorrelated one has more.
    rng = :rand.seed_s(:exsss, 5)

    {iid, rng} =
      Enum.map_reduce(1..2000, rng, fn _i, r ->
        {z, r} = :rand.normal_s(r)
        {Nx.tensor([z], type: :f64, backend: Nx.BinaryBackend), r}
      end)

    {corr, _rng} =
      Enum.map_reduce(1..2000, {rng, 0.0}, fn _i, {r, prev} ->
        {z, r} = :rand.normal_s(r)
        x = 0.95 * prev + 0.31 * z
        {Nx.tensor([x], type: :f64, backend: Nx.BinaryBackend), {r, x}}
      end)
      |> then(fn {xs, {r, _}} -> {xs, r} end)

    {:ok, l_iid} = Tuning.l_from_ess(iid, 1.0, l_factor: 1.0)
    {:ok, l_corr} = Tuning.l_from_ess(corr, 1.0, l_factor: 1.0)

    assert l_iid < 3.0, "an i.i.d. sequence should give n/ESS near 1, got #{l_iid}"
    assert l_corr > 3.0 * l_iid, "AR(0.95) should give a much longer L: #{l_corr} vs #{l_iid}"
  end
end
