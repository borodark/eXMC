defmodule Exmc.CustomDistTest do
  use ExUnit.Case, async: false

  alias Exmc.Builder
  alias Exmc.Compiler
  alias Exmc.Dist.Custom
  alias Exmc.NUTS.Sampler

  # A standard normal logpdf for testing (without the normalizing constant)
  defp normal_logpdf do
    fn x, params ->
      mu = params.mu
      sigma = params.sigma
      z = Nx.divide(Nx.subtract(x, mu), sigma)
      z2 = Nx.multiply(z, z)
      log_sigma = Nx.log(sigma)
      Nx.negate(Nx.add(Nx.multiply(Nx.tensor(0.5), z2), log_sigma))
    end
  end

  describe "Custom.new/2" do
    test "creates a custom dist with default support :real" do
      dist = Custom.new(normal_logpdf())
      assert dist.support == :real
      assert dist.transform == nil
    end

    test "positive support gets :log transform" do
      dist = Custom.new(normal_logpdf(), support: :positive)
      assert dist.support == :positive
      assert dist.transform == :log
    end

    test "unit_interval support gets :logit transform" do
      dist = Custom.new(normal_logpdf(), support: :unit_interval)
      assert dist.support == :unit_interval
      assert dist.transform == :logit
    end

    test "explicit transform overrides auto" do
      dist = Custom.new(normal_logpdf(), support: :positive, transform: :softplus)
      assert dist.support == :positive
      assert dist.transform == :softplus
    end

    test "stores sample_fn" do
      sample_fn = fn params, rng ->
        {z, rng} = :rand.normal_s(rng)
        mu = Nx.to_number(params.mu)
        {Nx.tensor(mu + z), rng}
      end

      dist = Custom.new(normal_logpdf(), sample_fn: sample_fn)
      assert dist.sample_fn != nil
    end
  end

  describe "logpdf evaluation" do
    test "custom logpdf matches manual computation" do
      dist = Custom.new(normal_logpdf())
      x = Nx.tensor(1.0)
      params = %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0), __dist__: dist}
      result = Custom.logpdf(x, params)
      # -0.5 * 1^2 - log(1) = -0.5
      assert_in_delta Nx.to_number(result), -0.5, 1.0e-6
    end

    test "custom logpdf with non-zero mu" do
      dist = Custom.new(normal_logpdf())
      x = Nx.tensor(3.0)
      params = %{mu: Nx.tensor(3.0), sigma: Nx.tensor(1.0), __dist__: dist}
      result = Custom.logpdf(x, params)
      # z=0, -0.5*0 - log(1) = 0.0
      assert_in_delta Nx.to_number(result), 0.0, 1.0e-6
    end
  end

  describe "behaviour callbacks" do
    test "support returns correct value" do
      dist = Custom.new(normal_logpdf(), support: :positive)
      assert Custom.support(%{__dist__: dist}) == :positive
    end

    test "transform returns correct value" do
      dist = Custom.new(normal_logpdf(), support: :positive)
      assert Custom.transform(%{__dist__: dist}) == :log
    end

    test "sample with sample_fn works" do
      sample_fn = fn params, rng ->
        mu = Nx.to_number(params.mu)
        {z, rng} = :rand.normal_s(rng)
        {Nx.tensor(mu + z), rng}
      end

      dist = Custom.new(normal_logpdf(), sample_fn: sample_fn)
      rng = :rand.seed_s(:exsss, 42)
      {value, _rng} = Custom.sample(%{mu: Nx.tensor(5.0), __dist__: dist}, rng)
      assert is_number(Nx.to_number(value))
    end

    test "sample without sample_fn raises" do
      dist = Custom.new(normal_logpdf())

      assert_raise RuntimeError, ~r/does not implement sample/, fn ->
        rng = :rand.seed_s(:exsss, 42)
        Custom.sample(%{mu: Nx.tensor(0.0), __dist__: dist}, rng)
      end
    end
  end

  describe "convenience rv helper" do
    test "Custom.rv injects __dist__ into params" do
      dist = Custom.new(normal_logpdf())
      ir = Builder.new_ir()
      ir = Custom.rv(ir, "x", dist, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

      node = ir.nodes["x"]
      assert node != nil

      {_tag, mod, params} = node.op
      assert mod == Custom
      assert Map.has_key?(params, :__dist__)
      assert params.__dist__ == dist
    end
  end

  describe "compiler integration" do
    test "custom dist compiles into a model and produces finite logp" do
      dist = Custom.new(normal_logpdf())
      ir = Builder.new_ir()
      ir = Custom.rv(ir, "x", dist, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

      {logp_fn, pm} = Compiler.compile(ir)
      assert pm.size == 1

      flat = Nx.tensor([0.5])
      logp = logp_fn.(flat) |> Nx.to_number()
      assert is_float(logp)
      refute logp == :nan
      refute logp == :infinity
      refute logp == :neg_infinity
    end

    test "custom dist compiles and produces finite gradients" do
      dist = Custom.new(normal_logpdf())
      ir = Builder.new_ir()
      ir = Custom.rv(ir, "x", dist, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

      {vag_fn, pm} = Compiler.value_and_grad(ir)
      assert pm.size == 1

      flat = Nx.tensor([0.5])
      {logp, grad} = vag_fn.(flat)

      logp_val = Nx.to_number(logp)
      grad_val = Nx.to_number(Nx.reshape(grad, {}))

      assert is_float(logp_val)
      assert is_float(grad_val)
      refute logp_val == :nan
      refute grad_val == :nan
    end

    test "custom dist with :positive support gets transform in compiled model" do
      logpdf_fn = fn x, params ->
        # Exponential-like: logpdf = log(rate) - rate * x
        rate = params.rate
        Nx.subtract(Nx.log(rate), Nx.multiply(rate, x))
      end

      dist = Custom.new(logpdf_fn, support: :positive)
      ir = Builder.new_ir()
      ir = Custom.rv(ir, "x", dist, %{rate: Nx.tensor(1.0)})

      {logp_fn, pm} = Compiler.compile(ir)
      assert pm.size == 1

      # The value is in unconstrained space (log transform applied)
      # exp(0) = 1.0 in constrained space
      flat = Nx.tensor([0.0])
      logp = logp_fn.(flat) |> Nx.to_number()
      assert is_float(logp)
      refute logp == :nan
    end
  end

  describe "NUTS sampling" do
    @tag timeout: 120_000
    test "custom dist works with NUTS sampler" do
      # Simple model: x ~ Custom Normal(0, 1)
      dist = Custom.new(normal_logpdf())
      ir = Builder.new_ir()
      ir = Custom.rv(ir, "x", dist, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

      {trace, stats} = Sampler.sample(ir, %{}, num_samples: 300, seed: 42, num_warmup: 200)

      assert Map.has_key?(trace, "x")
      samples = trace["x"]
      n = elem(Nx.shape(samples), 0)
      assert n == 300

      # Check mean is near 0 (generous tolerance for MCMC)
      mean = samples |> Nx.mean() |> Nx.to_number()
      assert_in_delta mean, 0.0, 0.5

      # Check we have reasonable step size
      assert is_number(stats.step_size)
      assert stats.step_size > 0.0
    end
  end
end
