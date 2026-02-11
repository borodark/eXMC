defmodule Exmc.CensoredTest do
  use ExUnit.Case, async: true

  alias Exmc.{Builder, Compiler}
  alias Exmc.Dist.Censored

  describe "erfc approximation" do
    test "matches :math.erfc at several points" do
      for x <- [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] do
        approx = Censored.erfc(Nx.tensor(x)) |> Nx.to_number()
        exact = :math.erfc(x)
        assert_in_delta approx, exact, 3.0e-6, "erfc(#{x}): got #{approx}, expected #{exact}"
      end
    end

    test "erfc of negative values" do
      for x <- [-0.5, -1.0, -2.0] do
        approx = Censored.erfc(Nx.tensor(x)) |> Nx.to_number()
        exact = :math.erfc(x)
        assert_in_delta approx, exact, 3.0e-6
      end
    end
  end

  describe "normal_cdf" do
    test "Phi(0) = 0.5" do
      result = Censored.normal_cdf(Nx.tensor(0.0)) |> Nx.to_number()
      assert_in_delta result, 0.5, 1.0e-6
    end

    test "Phi(-inf) -> 0, Phi(+inf) -> 1" do
      low = Censored.normal_cdf(Nx.tensor(-5.0)) |> Nx.to_number()
      high = Censored.normal_cdf(Nx.tensor(5.0)) |> Nx.to_number()
      assert low < 1.0e-5
      assert high > 1.0 - 1.0e-5
    end

    test "Phi(1.96) ~ 0.975" do
      result = Censored.normal_cdf(Nx.tensor(1.96)) |> Nx.to_number()
      assert_in_delta result, 0.975, 0.001
    end
  end

  describe "right-censored log_likelihood" do
    test "logp matches manual log(1 - Phi(z))" do
      mu = Nx.tensor(0.0)
      sigma = Nx.tensor(1.0)
      x = Nx.tensor(1.0)

      logp = Censored.log_likelihood(:right, x, Exmc.Dist.Normal, %{mu: mu, sigma: sigma})
      logp_val = Nx.to_number(logp)

      # Manual: log(1 - Phi(1.0)) = log(Phi(-1.0))
      phi_neg1 = :math.erfc(1.0 / :math.sqrt(2.0)) / 2.0
      expected = :math.log(phi_neg1)

      assert_in_delta logp_val, expected, 1.0e-5
    end
  end

  describe "left-censored log_likelihood" do
    test "logp matches manual log(Phi(z))" do
      mu = Nx.tensor(0.0)
      sigma = Nx.tensor(1.0)
      x = Nx.tensor(-1.0)

      logp = Censored.log_likelihood(:left, x, Exmc.Dist.Normal, %{mu: mu, sigma: sigma})
      logp_val = Nx.to_number(logp)

      # Manual: log(Phi(-1.0))
      phi_neg1 = :math.erfc(1.0 / :math.sqrt(2.0)) / 2.0
      expected = :math.log(phi_neg1)

      assert_in_delta logp_val, expected, 1.0e-5
    end
  end

  describe "interval-censored log_likelihood" do
    test "logp matches log(Phi(z_hi) - Phi(z_lo))" do
      mu = Nx.tensor(0.0)
      sigma = Nx.tensor(1.0)
      lower = Nx.tensor(-1.0)
      upper = Nx.tensor(1.0)

      logp =
        Censored.log_likelihood(:interval, %{lower: lower, upper: upper}, Exmc.Dist.Normal, %{
          mu: mu,
          sigma: sigma
        })

      logp_val = Nx.to_number(logp)

      # Manual: log(Phi(1) - Phi(-1))
      phi_1 = 0.5 * :math.erfc(-1.0 / :math.sqrt(2.0))
      phi_neg1 = 0.5 * :math.erfc(1.0 / :math.sqrt(2.0))
      expected = :math.log(phi_1 - phi_neg1)

      assert_in_delta logp_val, expected, 1.0e-5
    end
  end

  describe "censored obs in full model (eager)" do
    test "right-censored obs compiles and evaluates" do
      ir = Builder.new_ir()
      ir = Builder.rv(ir, "x", Exmc.Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
      ir = Builder.obs(ir, "x_obs", "x", Nx.tensor(2.0), censored: :right)

      {logp_fn, pm} = Compiler.compile(ir)
      # x is observed (censored), so pm.size == 0
      assert pm.size == 0
      logp = logp_fn.(Nx.tensor(0.0)) |> Nx.to_number()
      assert is_number(logp)
      # log(1 - Phi(2)) should be negative
      assert logp < 0
    end

    test "left-censored obs compiles and evaluates" do
      ir = Builder.new_ir()
      ir = Builder.rv(ir, "x", Exmc.Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
      ir = Builder.obs(ir, "x_obs", "x", Nx.tensor(-2.0), censored: :left)

      {logp_fn, pm} = Compiler.compile(ir)
      assert pm.size == 0
      logp = logp_fn.(Nx.tensor(0.0)) |> Nx.to_number()
      assert is_number(logp)
      assert logp < 0
    end

    test "interval-censored obs compiles and evaluates" do
      ir = Builder.new_ir()
      ir = Builder.rv(ir, "x", Exmc.Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

      ir =
        Builder.obs(ir, "x_obs", "x", %{lower: Nx.tensor(-1.0), upper: Nx.tensor(1.0)},
          censored: :interval
        )

      {logp_fn, pm} = Compiler.compile(ir)
      assert pm.size == 0
      logp = logp_fn.(Nx.tensor(0.0)) |> Nx.to_number()
      assert is_number(logp)
      # log(Phi(1) - Phi(-1)) ~ log(0.6827) ~ -0.38
      assert_in_delta logp, :math.log(0.6827), 0.01
    end
  end

  describe "censored with deferred params (string refs)" do
    test "right-censored with hierarchical mu" do
      ir = Builder.new_ir()
      ir = Builder.rv(ir, "mu", Exmc.Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(10.0)})

      ir =
        Builder.rv(ir, "x", Exmc.Dist.Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})

      ir = Builder.obs(ir, "x_obs", "x", Nx.tensor(2.0), censored: :right)

      {logp_fn, pm} = Compiler.compile(ir)
      # mu is free
      assert pm.size == 1

      # Evaluate at mu=0 (unconstrained)
      logp = logp_fn.(Nx.tensor([0.0])) |> Nx.to_number()
      assert is_number(logp)
      assert logp < 0
    end
  end
end
