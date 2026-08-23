defmodule Exmc.GaussianRandomWalkTest do
  use ExUnit.Case, async: true

  alias Exmc.{Builder, Compiler, PointMap, Rewrite}
  alias Exmc.Dist.{GaussianRandomWalk, Normal}
  import Exmc.TestHelper

  # --- logpdf tests ---

  test "GRW logpdf matches manual Normal sum for T=3" do
    sigma = Nx.tensor(1.0)
    x = Nx.tensor([0.1, 0.3, 0.2])

    result = GaussianRandomWalk.logpdf(x, %{sigma: sigma}) |> Nx.to_number()

    # Manual: x[0] ~ N(0,1), x[1]-x[0] ~ N(0,1), x[2]-x[1] ~ N(0,1)
    logp_0 = Normal.logpdf(Nx.tensor(0.1), %{mu: Nx.tensor(0.0), sigma: sigma}) |> Nx.to_number()
    logp_1 = Normal.logpdf(Nx.tensor(0.2), %{mu: Nx.tensor(0.0), sigma: sigma}) |> Nx.to_number()
    logp_2 = Normal.logpdf(Nx.tensor(-0.1), %{mu: Nx.tensor(0.0), sigma: sigma}) |> Nx.to_number()
    expected = logp_0 + logp_1 + logp_2

    assert_in_delta result, expected, 1.0e-5
  end

  test "GRW logpdf with sigma=0.5" do
    sigma = Nx.tensor(0.5)
    x = Nx.tensor([0.1, 0.2])

    result = GaussianRandomWalk.logpdf(x, %{sigma: sigma}) |> Nx.to_number()

    # Manual: x[0]=0.1 ~ N(0,0.5), diff=0.1 ~ N(0,0.5)
    logp_0 = Normal.logpdf(Nx.tensor(0.1), %{mu: Nx.tensor(0.0), sigma: sigma}) |> Nx.to_number()
    logp_1 = Normal.logpdf(Nx.tensor(0.1), %{mu: Nx.tensor(0.0), sigma: sigma}) |> Nx.to_number()
    expected = logp_0 + logp_1

    assert_in_delta result, expected, 1.0e-5
  end

  test "GRW logpdf T=1 is just Normal logpdf" do
    sigma = Nx.tensor(2.0)
    x = Nx.tensor([0.5])

    result = GaussianRandomWalk.logpdf(x, %{sigma: sigma}) |> Nx.to_number()

    expected =
      Normal.logpdf(Nx.tensor(0.5), %{mu: Nx.tensor(0.0), sigma: sigma}) |> Nx.to_number()

    assert_in_delta result, expected, 1.0e-6
  end

  test "GRW support and transform" do
    params = %{sigma: Nx.tensor(1.0)}
    assert GaussianRandomWalk.support(params) == :real
    assert GaussianRandomWalk.transform(params) == nil
  end

  # --- Builder + PointMap integration ---

  test "GRW in Builder with shape" do
    ir =
      Builder.new_ir()
      |> Builder.rv("s", GaussianRandomWalk, %{sigma: Nx.tensor(1.0)}, shape: {5})
      |> Rewrite.apply()

    pm = PointMap.build(ir)

    assert pm.size == 5
    entry = hd(pm.entries)
    assert entry.id == "s"
    assert entry.length == 5
    assert entry.shape == {5}
    assert entry.transform == nil
  end

  test "GRW compile + logp" do
    ir =
      Builder.new_ir()
      |> Builder.rv("s", GaussianRandomWalk, %{sigma: Nx.tensor(1.0)}, shape: {3})

    {logp_fn, pm} = Compiler.compile(ir)

    s_val = Nx.tensor([0.1, 0.3, 0.2])
    flat = PointMap.pack(%{"s" => s_val}, pm)
    compiled_logp = logp_fn.(flat) |> Nx.to_number()

    direct_logp = GaussianRandomWalk.logpdf(s_val, %{sigma: Nx.tensor(1.0)}) |> Nx.to_number()
    assert_in_delta compiled_logp, direct_logp, 1.0e-5
  end

  test "GRW gradient via value_and_grad" do
    ir =
      Builder.new_ir()
      |> Builder.rv("s", GaussianRandomWalk, %{sigma: Nx.tensor(1.0)}, shape: {3})

    {logp_fn, pm} = Compiler.compile(ir)
    {vag_fn, _pm} = Compiler.value_and_grad(ir)

    s_val = Nx.tensor([0.1, 0.3, 0.2])
    flat = PointMap.pack(%{"s" => s_val}, pm)
    {_logp, grad} = vag_fn.(flat)

    # Finite difference check
    eps = 1.0e-5

    fd_grad =
      Enum.map(0..2, fn i ->
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
            Nx.tensor([Nx.to_flat_list(flat) |> Enum.at(i) |> Kernel.+(eps * -1.0)])
          )

        fp = logp_fn.(flat_plus) |> Nx.to_number()
        fm = logp_fn.(flat_minus) |> Nx.to_number()
        (fp - fm) / (2.0 * eps)
      end)

    assert_close(grad, Nx.tensor(fd_grad), 1.0e-3)
  end

  test "GRW with string ref to sigma" do
    alias Exmc.Dist.Exponential

    ir =
      Builder.new_ir()
      |> Builder.rv("sigma", Exponential, %{lambda: Nx.tensor(1.0)})
      |> Builder.rv("s", GaussianRandomWalk, %{sigma: "sigma"}, shape: {3})

    {logp_fn, pm} = Compiler.compile(ir)

    # sigma is Exponential -> gets :log transform, entry is unconstrained log(sigma)
    # s is GRW -> unconstrained, 3 positions
    assert pm.size == 4

    # Pack: log(sigma) = 0.0 (sigma=1.0), s = [0.1, 0.3, 0.2]
    flat = PointMap.pack(%{"sigma" => Nx.tensor(0.0), "s" => Nx.tensor([0.1, 0.3, 0.2])}, pm)
    logp = logp_fn.(flat) |> Nx.to_number()
    assert is_float(logp) and logp < 0.0
  end

  test "GRW pack/unpack roundtrip" do
    ir =
      Builder.new_ir()
      |> Builder.rv("s", GaussianRandomWalk, %{sigma: Nx.tensor(1.0)}, shape: {4})
      |> Rewrite.apply()

    pm = PointMap.build(ir)

    s_val = Nx.tensor([1.0, 2.0, 3.0, 4.0])
    flat = PointMap.pack(%{"s" => s_val}, pm)
    recovered = PointMap.unpack(flat, pm)

    assert_close(recovered["s"], s_val)
  end

  test "GRW sample produces walk" do
    params = %{sigma: Nx.tensor(0.5), steps: 10}
    rng = :rand.seed_s(:exsss, 42)

    {sample, _rng} = GaussianRandomWalk.sample(params, rng)
    assert Nx.shape(sample) == {10}
  end
end
