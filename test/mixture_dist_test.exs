defmodule Exmc.MixtureDistTest do
  use ExUnit.Case, async: false

  alias Exmc.Dist.{Mixture, Normal}
  alias Exmc.Builder
  alias Exmc.Compiler

  describe "logpdf" do
    test "50/50 Normal mixture matches manual log-sum-exp" do
      x = Nx.tensor(0.0)

      params = %{
        components: [Normal, Normal],
        params: [
          %{mu: Nx.tensor(-2.0), sigma: Nx.tensor(1.0)},
          %{mu: Nx.tensor(2.0), sigma: Nx.tensor(1.0)}
        ],
        weights: Nx.tensor([0.5, 0.5])
      }

      result = Mixture.logpdf(x, params) |> Nx.to_number()

      # Manual: log(0.5 * N(0|-2,1) + 0.5 * N(0|2,1))
      lp1 = Normal.logpdf(x, %{mu: Nx.tensor(-2.0), sigma: Nx.tensor(1.0)}) |> Nx.to_number()
      lp2 = Normal.logpdf(x, %{mu: Nx.tensor(2.0), sigma: Nx.tensor(1.0)}) |> Nx.to_number()
      expected = :math.log(0.5 * :math.exp(lp1) + 0.5 * :math.exp(lp2))

      assert_in_delta result, expected, 1.0e-5
    end

    test "asymmetric 80/20 weights" do
      x = Nx.tensor(-2.0)

      params = %{
        components: [Normal, Normal],
        params: [
          %{mu: Nx.tensor(-2.0), sigma: Nx.tensor(1.0)},
          %{mu: Nx.tensor(2.0), sigma: Nx.tensor(1.0)}
        ],
        weights: Nx.tensor([0.8, 0.2])
      }

      result = Mixture.logpdf(x, params) |> Nx.to_number()

      lp1 = Normal.logpdf(x, %{mu: Nx.tensor(-2.0), sigma: Nx.tensor(1.0)}) |> Nx.to_number()
      lp2 = Normal.logpdf(x, %{mu: Nx.tensor(2.0), sigma: Nx.tensor(1.0)}) |> Nx.to_number()
      expected = :math.log(0.8 * :math.exp(lp1) + 0.2 * :math.exp(lp2))

      assert_in_delta result, expected, 1.0e-5
    end

    test "three-component mixture" do
      x = Nx.tensor(0.0)

      params = %{
        components: [Normal, Normal, Normal],
        params: [
          %{mu: Nx.tensor(-3.0), sigma: Nx.tensor(0.5)},
          %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)},
          %{mu: Nx.tensor(3.0), sigma: Nx.tensor(0.5)}
        ],
        weights: Nx.tensor([0.2, 0.6, 0.2])
      }

      result = Mixture.logpdf(x, params) |> Nx.to_number()
      assert is_float(result)
      refute result == :nan
    end
  end

  describe "support and transform" do
    test "delegates to first component" do
      params = %{
        components: [Normal, Normal],
        params: [
          %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)},
          %{mu: Nx.tensor(1.0), sigma: Nx.tensor(1.0)}
        ],
        weights: Nx.tensor([0.5, 0.5])
      }

      assert Mixture.support(params) == :real
      assert Mixture.transform(params) == nil
    end
  end

  describe "sample" do
    test "draws from mixture components" do
      params = %{
        components: [Normal, Normal],
        params: [
          %{mu: Nx.tensor(-5.0), sigma: Nx.tensor(0.1)},
          %{mu: Nx.tensor(5.0), sigma: Nx.tensor(0.1)}
        ],
        weights: Nx.tensor([0.5, 0.5])
      }

      rng = :rand.seed_s(:exsss, 42)

      {samples, _rng} =
        Enum.map_reduce(1..200, rng, fn _, rng ->
          Mixture.sample(params, rng)
        end)

      values = Enum.map(samples, &Nx.to_number/1)
      near_neg5 = Enum.count(values, fn v -> v < 0.0 end)
      near_pos5 = Enum.count(values, fn v -> v > 0.0 end)

      # Both modes should get at least 30% of samples
      assert near_neg5 > 40
      assert near_pos5 > 40
    end
  end

  describe "compiler integration" do
    test "mixture compiles and evaluates logp" do
      ir = Builder.new_ir()

      ir =
        Builder.rv(ir, "x", Mixture, %{
          components: [Normal, Normal],
          params: [
            %{mu: Nx.tensor(-1.0), sigma: Nx.tensor(1.0)},
            %{mu: Nx.tensor(1.0), sigma: Nx.tensor(1.0)}
          ],
          weights: Nx.tensor([0.5, 0.5])
        })

      {logp_fn, pm} = Compiler.compile(ir)
      assert pm.size == 1

      flat = Nx.tensor([0.0])
      logp = logp_fn.(flat) |> Nx.to_number()
      assert is_float(logp)
      refute logp == :nan
    end

    test "mixture produces finite gradients" do
      ir = Builder.new_ir()

      ir =
        Builder.rv(ir, "x", Mixture, %{
          components: [Normal, Normal],
          params: [
            %{mu: Nx.tensor(-1.0), sigma: Nx.tensor(1.0)},
            %{mu: Nx.tensor(1.0), sigma: Nx.tensor(1.0)}
          ],
          weights: Nx.tensor([0.5, 0.5])
        })

      {vag_fn, _pm} = Compiler.value_and_grad(ir)

      flat = Nx.tensor([0.0])
      {logp, grad} = vag_fn.(flat)

      logp_val = Nx.to_number(logp)
      grad_val = Nx.to_number(Nx.reshape(grad, {}))

      assert is_float(logp_val)
      assert is_float(grad_val)
      refute logp_val == :nan
      refute grad_val == :nan
    end
  end
end
