defmodule Exmc.DirichletTest do
  use ExUnit.Case, async: true

  alias Exmc.{Builder, Compiler, PointMap, Transform, Rewrite}
  alias Exmc.Dist.Dirichlet
  import Exmc.TestHelper

  # --- logpdf tests ---

  test "Dirichlet(1,1,1) is uniform on 2-simplex, logpdf = log(2!)" do
    alpha = Nx.tensor([1.0, 1.0, 1.0])
    x = Nx.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    result = Dirichlet.logpdf(x, %{alpha: alpha}) |> Nx.to_number()
    # Dirichlet(1,1,1) normalizing constant: lgamma(3) - 3*lgamma(1) = log(2!) - 0 = log(2)
    expected = :math.log(2.0)
    assert_in_delta result, expected, 1.0e-5
  end

  test "Dirichlet(2,2,2) at center" do
    alpha = Nx.tensor([2.0, 2.0, 2.0])
    x = Nx.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    result = Dirichlet.logpdf(x, %{alpha: alpha}) |> Nx.to_number()
    # kernel: sum((2-1)*log(1/3)) = 3 * (-log(3)) = -3*log(3)
    # norm: lgamma(6) - 3*lgamma(2) = log(120) - 0 = log(120)
    expected = -3.0 * :math.log(3.0) + :math.log(120.0)
    assert_in_delta result, expected, 1.0e-4
  end

  test "Dirichlet(1,1) is uniform on 1-simplex (Beta(1,1))" do
    alpha = Nx.tensor([1.0, 1.0])
    x = Nx.tensor([0.5, 0.5])

    result = Dirichlet.logpdf(x, %{alpha: alpha}) |> Nx.to_number()
    # lgamma(2) - 2*lgamma(1) = 0 - 0 = 0
    # kernel = 0
    expected = 0.0
    assert_in_delta result, expected, 1.0e-6
  end

  test "Dirichlet support and transform" do
    params = %{alpha: Nx.tensor([1.0, 1.0, 1.0])}
    assert Dirichlet.support(params) == :simplex
    assert Dirichlet.transform(params) == :stick_breaking
  end

  # --- Stick-breaking transform tests ---

  test "stick-breaking forward: z=0 gives uniform" do
    z = Nx.tensor([0.0, 0.0])
    x = Transform.apply(:stick_breaking, z)

    # sigmoid(0) = 0.5
    # x[0] = 0.5 * 1.0 = 0.5
    # x[1] = 0.5 * 0.5 = 0.25
    # x[2] = 0.25
    x_list = Nx.to_flat_list(x)
    assert_in_delta Enum.at(x_list, 0), 0.5, 1.0e-6
    assert_in_delta Enum.at(x_list, 1), 0.25, 1.0e-6
    assert_in_delta Enum.at(x_list, 2), 0.25, 1.0e-6
    # Sum = 1
    assert_in_delta Enum.sum(x_list), 1.0, 1.0e-6
  end

  test "stick-breaking forward produces valid simplex" do
    z = Nx.tensor([1.0, -0.5, 0.3])
    x = Transform.apply(:stick_breaking, z)

    x_list = Nx.to_flat_list(x)
    # All positive
    Enum.each(x_list, fn xi -> assert xi > 0.0 end)
    # Sum to 1
    assert_in_delta Enum.sum(x_list), 1.0, 1.0e-6
  end

  test "stick-breaking round-trip: z -> x -> z" do
    z = Nx.tensor([0.5, -1.0])
    x = Transform.apply(:stick_breaking, z)
    z_recovered = Transform.inverse_stick_breaking(x)

    assert_close(z_recovered, z, 1.0e-5)
  end

  test "stick-breaking round-trip K=5" do
    z = Nx.tensor([0.2, -0.3, 1.0, -0.5])
    x = Transform.apply(:stick_breaking, z)
    z_recovered = Transform.inverse_stick_breaking(x)

    assert_close(z_recovered, z, 1.0e-5)
  end

  test "stick-breaking Jacobian via finite differences" do
    z = Nx.tensor([0.5, -0.3])

    ladj = Transform.log_abs_det_jacobian(:stick_breaking, z) |> Nx.to_number()

    # Finite-difference Jacobian: compute det(J) numerically
    eps = 1.0e-5
    k_minus_1 = 2

    # Build Jacobian matrix (K x K-1)
    jac_cols =
      for j <- 0..(k_minus_1 - 1) do
        z_plus =
          Nx.indexed_put(
            z,
            Nx.tensor([[j]]),
            Nx.tensor([Nx.to_flat_list(z) |> Enum.at(j) |> Kernel.+(eps)])
          )

        z_minus =
          Nx.indexed_put(
            z,
            Nx.tensor([[j]]),
            Nx.tensor([Nx.to_flat_list(z) |> Enum.at(j) |> Kernel.-(eps)])
          )

        x_plus = Transform.apply(:stick_breaking, z_plus) |> Nx.to_flat_list()
        x_minus = Transform.apply(:stick_breaking, z_minus) |> Nx.to_flat_list()
        Enum.zip(x_plus, x_minus) |> Enum.map(fn {p, m} -> (p - m) / (2.0 * eps) end)
      end

    # Jacobian is K x (K-1). For the log|det J| of a non-square matrix,
    # we need log(sqrt(det(J^T J))), but for stick-breaking the standard
    # convention is to drop the last row (since x_K is determined).
    # So we use the (K-1) x (K-1) sub-Jacobian.
    sub_jac =
      for j <- 0..(k_minus_1 - 1) do
        col = Enum.at(jac_cols, j)
        Enum.take(col, k_minus_1)
      end

    # Transpose: sub_jac is indexed as [col][row], need [row][col]
    sub_jac_t =
      for i <- 0..(k_minus_1 - 1) do
        for j <- 0..(k_minus_1 - 1) do
          Enum.at(Enum.at(sub_jac, j), i)
        end
      end

    # Compute det of sub_jac_t (2x2)
    [[a, b], [c, d]] = sub_jac_t
    det = a * d - b * c
    fd_ladj = :math.log(abs(det))

    assert_in_delta ladj, fd_ladj, 0.01
  end

  test "unconstrained_length for stick_breaking" do
    assert Transform.unconstrained_length(:stick_breaking, {3}) == 2
    assert Transform.unconstrained_length(:stick_breaking, {5}) == 4
  end

  test "unconstrained_shape for stick_breaking" do
    assert Transform.unconstrained_shape(:stick_breaking, {3}) == {2}
    assert Transform.unconstrained_shape(:stick_breaking, {5}) == {4}
  end

  # --- Builder + PointMap integration ---

  test "Dirichlet in Builder with shape" do
    alpha = Nx.tensor([1.0, 1.0, 1.0])

    ir =
      Builder.new_ir()
      |> Builder.rv("w", Dirichlet, %{alpha: alpha}, shape: {3})
      |> Rewrite.apply()

    pm = PointMap.build(ir)

    # K=3 on simplex -> K-1=2 unconstrained positions
    assert pm.size == 2
    entry = hd(pm.entries)
    assert entry.id == "w"
    assert entry.length == 2
    assert entry.shape == {2}
    assert entry.transform == :stick_breaking
  end

  test "Dirichlet pack/unpack via unconstrained" do
    alpha = Nx.tensor([1.0, 1.0, 1.0])

    ir =
      Builder.new_ir()
      |> Builder.rv("w", Dirichlet, %{alpha: alpha}, shape: {3})
      |> Rewrite.apply()

    pm = PointMap.build(ir)

    # Start with constrained simplex value
    x_constrained = Nx.tensor([0.5, 0.3, 0.2])
    unconstrained = PointMap.to_unconstrained(%{"w" => x_constrained}, pm)

    # Pack to flat
    flat = PointMap.pack(unconstrained, pm)
    assert Nx.shape(flat) == {2}

    # Unpack back
    recovered_unc = PointMap.unpack(flat, pm)

    # Transform to constrained
    recovered_con = PointMap.to_constrained(recovered_unc, pm)

    assert_close(recovered_con["w"], x_constrained, 1.0e-4)
  end

  test "Dirichlet compile + logp" do
    alpha = Nx.tensor([1.0, 1.0, 1.0])

    ir =
      Builder.new_ir()
      |> Builder.rv("w", Dirichlet, %{alpha: alpha}, shape: {3})

    {logp_fn, pm} = Compiler.compile(ir)

    # Evaluate at z = [0, 0] -> x = [0.5, 0.25, 0.25]
    z_val = Nx.tensor([0.0, 0.0])
    flat = PointMap.pack(%{"w" => z_val}, pm)
    compiled_logp = logp_fn.(flat) |> Nx.to_number()

    # logp = Dirichlet.logpdf([0.5, 0.25, 0.25], alpha) + log|J| of stick-breaking
    x = Transform.apply(:stick_breaking, z_val)
    logp_dist = Dirichlet.logpdf(x, %{alpha: alpha}) |> Nx.to_number()
    log_jac = Transform.log_abs_det_jacobian(:stick_breaking, z_val) |> Nx.to_number()
    expected = logp_dist + log_jac

    assert_in_delta compiled_logp, expected, 1.0e-5
  end

  test "Dirichlet gradient via value_and_grad" do
    alpha = Nx.tensor([2.0, 3.0, 1.0])

    ir =
      Builder.new_ir()
      |> Builder.rv("w", Dirichlet, %{alpha: alpha}, shape: {3})

    {logp_fn, pm} = Compiler.compile(ir)
    {vag_fn, _pm} = Compiler.value_and_grad(ir)

    z_val = Nx.tensor([0.5, -0.3])
    flat = PointMap.pack(%{"w" => z_val}, pm)
    {_logp, grad} = vag_fn.(flat)

    # Finite difference check
    eps = 1.0e-5

    fd_grad =
      Enum.map(0..1, fn i ->
        flat_plus =
          Nx.indexed_put(
            flat,
            Nx.tensor([[i]]),
            Nx.tensor([Nx.to_flat_list(flat) |> Enum.at(i) |> Kernel.+(eps)])
          )

        flat_minus =
          Nx.indexed_put(
            flat,
            Nx.tensor([[i]]),
            Nx.tensor([Nx.to_flat_list(flat) |> Enum.at(i) |> Kernel.-(eps)])
          )

        fp = logp_fn.(flat_plus) |> Nx.to_number()
        fm = logp_fn.(flat_minus) |> Nx.to_number()
        (fp - fm) / (2.0 * eps)
      end)

    assert_close(grad, Nx.tensor(fd_grad), 0.01)
  end

  test "Dirichlet sample moments" do
    alpha = Nx.tensor([2.0, 3.0, 5.0])
    rng = :rand.seed_s(:exsss, 42)
    n = 5000

    {samples, _rng} =
      Enum.map_reduce(1..n, rng, fn _i, r ->
        Dirichlet.sample(%{alpha: alpha}, r)
      end)

    stacked = Nx.stack(samples)
    means = Nx.mean(stacked, axes: [0]) |> Nx.to_flat_list()

    # E[x_i] = alpha_i / sum(alpha)
    alpha_sum = 10.0
    expected = [2.0 / alpha_sum, 3.0 / alpha_sum, 5.0 / alpha_sum]

    Enum.zip(means, expected)
    |> Enum.each(fn {m, e} -> assert_in_delta m, e, 0.03 end)
  end

  # --- lgamma vectorized test ---

  test "lgamma works on vectors" do
    x = Nx.tensor([1.0, 2.0, 5.0])
    result = Exmc.Math.lgamma(x) |> Nx.to_flat_list()

    expected = [0.0, 0.0, :math.log(24.0)]

    Enum.zip(result, expected)
    |> Enum.each(fn {r, e} -> assert_in_delta r, e, 1.0e-5 end)
  end
end
