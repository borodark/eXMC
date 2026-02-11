defmodule Exmc.LogProbTest do
  use ExUnit.Case

  @moduledoc false

  # Tests here verify numeric correctness of logprob terms, transforms,
  # observation metadata, and rewrite-driven measurable ops.

  alias Exmc.{Builder, LogProb}
  alias Exmc.Dist.{Normal, Exponential, HalfNormal, Uniform01}
  alias Exmc.DSL
  require Exmc.DSL
  use Exmc.DSL
  import Exmc.TestHelper

  doctest Exmc.Builder
  doctest Exmc.LogProb
  doctest Exmc.Rewrite
  doctest Exmc.DSL
  doctest Exmc.Dist.Gamma
  doctest Exmc.Dist.Beta
  doctest Exmc.Dist.StudentT
  doctest Exmc.Dist.Laplace
  doctest Exmc.Dist.Cauchy

  test "normal logp" do
    ir =
      Builder.new_ir()
      |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

    x = Nx.tensor(0.3)

    x2 = Nx.multiply(x, x)
    expected = Nx.multiply(Nx.tensor(-0.5), Nx.add(x2, Nx.log(Nx.tensor(2.0 * :math.pi()))))
    got = LogProb.eval(ir, %{"x" => x})

    assert_close(got, expected)
  end

  test "log transform jacobian (explicit)" do
    ir =
      Builder.new_ir()
      |> Builder.rv("z", Exponential, %{lambda: Nx.tensor(1.5)}, transform: :log)

    z = Nx.tensor(0.1)
    x = Nx.exp(z)

    expected =
      Nx.add(
        Nx.subtract(Nx.log(Nx.tensor(1.5)), Nx.multiply(Nx.tensor(1.5), x)),
        z
      )

    got = LogProb.eval(ir, %{"z" => z})

    assert_close(got, expected)
  end

  test "log transform jacobian (default rewrite)" do
    ir =
      Builder.new_ir()
      |> Builder.rv("z", Exponential, %{lambda: Nx.tensor(1.5)})

    z = Nx.tensor(0.1)
    x = Nx.exp(z)

    expected =
      Nx.add(
        Nx.subtract(Nx.log(Nx.tensor(1.5)), Nx.multiply(Nx.tensor(1.5), x)),
        z
      )

    got = LogProb.eval(ir, %{"z" => z})

    assert_close(got, expected)
  end

  test "observed value uses rv logpdf" do
    ir =
      Builder.new_ir()
      |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
      |> Builder.obs("x_obs", "x", Nx.tensor(0.2))

    x = Nx.tensor(0.2)
    x2 = Nx.multiply(x, x)
    expected = Nx.multiply(Nx.tensor(-0.5), Nx.add(x2, Nx.log(Nx.tensor(2.0 * :math.pi()))))
    got = LogProb.eval(ir, %{})

    assert_close(got, expected)
  end

  test "sum of logps from independent rvs" do
    ir =
      Builder.new_ir()
      |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
      |> Builder.rv("y", Normal, %{mu: Nx.tensor(1.0), sigma: Nx.tensor(2.0)})

    x = Nx.tensor(0.1)
    y = Nx.tensor(-0.4)

    x2 = Nx.multiply(x, x)
    logp_x = Nx.multiply(Nx.tensor(-0.5), Nx.add(x2, Nx.log(Nx.tensor(2.0 * :math.pi()))))
    z = Nx.divide(Nx.subtract(y, Nx.tensor(1.0)), Nx.tensor(2.0))
    z2 = Nx.multiply(z, z)

    logp_y =
      Nx.multiply(
        Nx.tensor(-0.5),
        Nx.add(
          z2,
          Nx.add(
            Nx.log(Nx.tensor(2.0 * :math.pi())),
            Nx.multiply(Nx.tensor(2.0), Nx.log(Nx.tensor(2.0)))
          )
        )
      )

    got = LogProb.eval(ir, %{"x" => x, "y" => y})
    expected = Nx.add(logp_x, logp_y)

    assert_close(got, expected)
  end

  test "deterministic nodes do not contribute to logp" do
    ir =
      Builder.new_ir()
      |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
      |> Builder.det("d", :add, ["x", Nx.tensor(1.0)])

    x = Nx.tensor(0.7)
    x2 = Nx.multiply(x, x)
    expected = Nx.multiply(Nx.tensor(-0.5), Nx.add(x2, Nx.log(Nx.tensor(2.0 * :math.pi()))))
    got = LogProb.eval(ir, %{"x" => x})

    assert_close(got, expected)
  end

  test "measurable matmul observation" do
    a = Nx.tensor([[2.0]])

    ir =
      Builder.new_ir()
      |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
      |> Builder.det("y", :matmul, [a, "x"])
      |> Builder.obs("y_obs", "y", Nx.tensor([[0.4]]))

    x = Nx.tensor(0.2)
    x2 = Nx.multiply(x, x)
    logp_x = Nx.multiply(Nx.tensor(-0.5), Nx.add(x2, Nx.log(Nx.tensor(2.0 * :math.pi()))))
    jac = Nx.negate(Nx.log(Nx.tensor(2.0)))

    got = LogProb.eval(ir, %{})
    expected = Nx.add(logp_x, jac)

    assert_close(got, expected)
  end

  test "dsl model macro" do
    ir =
      DSL.model do
        DSL.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
        DSL.obs("x_obs", "x", Nx.tensor(0.2))
      end

    got = LogProb.eval(ir, %{})
    x = Nx.tensor(0.2)
    x2 = Nx.multiply(x, x)
    expected = Nx.multiply(Nx.tensor(-0.5), Nx.add(x2, Nx.log(Nx.tensor(2.0 * :math.pi()))))

    assert_close(got, expected)
  end

  test "dsl unqualified model macro" do
    ir =
      DSL.model do
        rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
        obs("x_obs", "x", Nx.tensor(0.2))
      end

    got = LogProb.eval(ir, %{})
    x = Nx.tensor(0.2)
    x2 = Nx.multiply(x, x)
    expected = Nx.multiply(Nx.tensor(-0.5), Nx.add(x2, Nx.log(Nx.tensor(2.0 * :math.pi()))))

    assert_close(got, expected)
  end

  test "obs metadata weight and mask" do
    ir =
      Builder.new_ir()
      |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
      |> Builder.obs("x_obs", "x", Nx.tensor(0.3))

    # Manually inject metadata to simulate user-provided obs meta.
    ir =
      put_in(
        ir.nodes["x_obs"].op,
        {:obs, "x", Nx.tensor(0.3), %{weight: 2.0, mask: Nx.tensor(true)}}
      )

    x = Nx.tensor(0.3)
    x2 = Nx.multiply(x, x)
    base = Nx.multiply(Nx.tensor(-0.5), Nx.add(x2, Nx.log(Nx.tensor(2.0 * :math.pi()))))
    expected = Nx.multiply(base, Nx.tensor(2.0))

    got = LogProb.eval(ir, %{})

    assert_close(got, expected)
  end

  test "obs metadata vector weight and mask" do
    values = Nx.tensor([0.0, 1.0])
    weights = Nx.tensor([1.0, 0.5])
    mask = Nx.tensor([true, false])

    ir =
      Builder.new_ir()
      |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
      |> Builder.obs("x_obs", "x", values, weight: weights, mask: mask)

    base =
      Nx.multiply(
        Nx.tensor(-0.5),
        Nx.add(Nx.multiply(values, values), Nx.log(Nx.tensor(2.0 * :math.pi())))
      )

    # Vector obs auto-adds reduce: :sum, so result is scalar sum of masked weighted logp
    expected = Nx.sum(Nx.select(mask, Nx.multiply(base, weights), Nx.tensor(0.0)))

    got = LogProb.eval(ir, %{})

    assert_close(got, expected)
  end

  test "dsl obs options" do
    ir =
      DSL.model do
        DSL.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
        DSL.obs("x_obs", "x", Nx.tensor(0.1), weight: 3.0)
      end

    x = Nx.tensor(0.1)
    x2 = Nx.multiply(x, x)
    base = Nx.multiply(Nx.tensor(-0.5), Nx.add(x2, Nx.log(Nx.tensor(2.0 * :math.pi()))))
    expected = Nx.multiply(base, Nx.tensor(3.0))

    got = LogProb.eval(ir, %{})

    assert_close(got, expected)
  end

  test "obs metadata reduce sum and mean" do
    values = Nx.tensor([0.0, 1.0])

    ir_sum =
      Builder.new_ir()
      |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
      |> Builder.obs("x_obs", "x", values, reduce: :sum)

    ir_mean =
      Builder.new_ir()
      |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
      |> Builder.obs("x_obs", "x", values, reduce: :mean)

    base =
      Nx.multiply(
        Nx.tensor(-0.5),
        Nx.add(Nx.multiply(values, values), Nx.log(Nx.tensor(2.0 * :math.pi())))
      )

    expected_sum = Nx.sum(base)
    expected_mean = Nx.mean(base)

    assert_close(LogProb.eval(ir_sum, %{}), expected_sum)
    assert_close(LogProb.eval(ir_mean, %{}), expected_mean)
  end

  test "obs metadata reduce logsumexp" do
    values = Nx.tensor([0.0, 1.0])

    ir =
      Builder.new_ir()
      |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
      |> Builder.obs("x_obs", "x", values, reduce: :logsumexp)

    base =
      Nx.multiply(
        Nx.tensor(-0.5),
        Nx.add(Nx.multiply(values, values), Nx.log(Nx.tensor(2.0 * :math.pi())))
      )

    expected = Nx.logsumexp(base)

    assert_close(LogProb.eval(ir, %{}), expected)
  end

  test "measurable affine observation" do
    a = Nx.tensor(2.0)
    b = Nx.tensor(1.0)

    ir =
      Builder.new_ir()
      |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
      |> Builder.det("y", :affine, [a, b, "x"])
      |> Builder.obs("y_obs", "y", Nx.tensor(1.4))

    x = Nx.divide(Nx.subtract(Nx.tensor(1.4), b), a)
    x2 = Nx.multiply(x, x)
    base = Nx.multiply(Nx.tensor(-0.5), Nx.add(x2, Nx.log(Nx.tensor(2.0 * :math.pi()))))
    jac = Nx.negate(Nx.log(Nx.abs(a)))
    expected = Nx.add(base, jac)

    assert_close(LogProb.eval(ir, %{}), expected)
  end

  test "softplus transform jacobian (default)" do
    ir =
      Builder.new_ir()
      |> Builder.rv("z", HalfNormal, %{sigma: Nx.tensor(1.0)})

    z = Nx.tensor(0.2)
    x = Nx.log1p(Nx.exp(z))

    base =
      Nx.multiply(
        Nx.tensor(-0.5),
        Nx.add(Nx.multiply(x, x), Nx.log(Nx.tensor(2.0 * :math.pi())))
      )
      |> Nx.add(Nx.log(Nx.tensor(2.0)))

    expected = Nx.add(base, Nx.log(Nx.sigmoid(z)))

    assert_close(LogProb.eval(ir, %{"z" => z}), expected)
  end

  test "logit transform jacobian (default)" do
    ir =
      Builder.new_ir()
      |> Builder.rv("z", Uniform01, %{})

    z = Nx.tensor(0.3)
    s = Nx.sigmoid(z)
    expected = Nx.add(Nx.log(s), Nx.log1p(Nx.negate(s)))

    assert_close(LogProb.eval(ir, %{"z" => z}), expected)
  end

  test "affine broadcast with vector a and b" do
    a = Nx.tensor([2.0, 3.0])
    b = Nx.tensor([1.0, 1.0])
    y = Nx.tensor([1.4, 2.5])

    ir =
      Builder.new_ir()
      |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
      |> Builder.det("y", :affine, [a, b, "x"])
      |> Builder.obs("y_obs", "y", y)

    x = Nx.divide(Nx.subtract(y, b), a)

    base =
      Nx.multiply(
        Nx.tensor(-0.5),
        Nx.add(Nx.multiply(x, x), Nx.log(Nx.tensor(2.0 * :math.pi())))
      )

    jac = Nx.negate(Nx.log(Nx.abs(a)))
    # Vector obs auto-adds reduce: :sum, so result is scalar sum
    expected = Nx.sum(Nx.add(base, jac))

    assert_close(LogProb.eval(ir, %{}), expected)
  end
end
