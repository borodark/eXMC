defmodule Exmc.SMCTest do
  use ExUnit.Case, async: false

  alias Exmc.Builder
  alias Exmc.Dist.Normal
  alias Exmc.SMC

  describe "sample/2" do
    test "empty model returns empty trace" do
      ir = Builder.new_ir()
      {trace, info} = SMC.sample(ir)
      assert trace == %{}
      assert info.num_stages == 0
    end

    @tag timeout: 120_000
    test "single Normal: recovers approximate mean" do
      ir = Builder.new_ir()
      ir = Builder.rv(ir, "x", Normal, %{mu: Nx.tensor(2.0), sigma: Nx.tensor(1.0)})

      {trace, info} = SMC.sample(ir, num_particles: 200, num_mh_steps: 3, seed: 42)

      assert Map.has_key?(trace, "x")
      samples = trace["x"]
      n = elem(Nx.shape(samples), 0)
      assert n == 200

      mean = samples |> Nx.mean() |> Nx.to_number()
      assert_in_delta mean, 2.0, 2.0

      assert is_list(info.betas)
      assert length(info.betas) > 0
      assert List.last(info.betas) == 1.0
    end

    @tag timeout: 120_000
    test "betas are monotonically increasing" do
      ir = Builder.new_ir()
      ir = Builder.rv(ir, "x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

      {_trace, info} = SMC.sample(ir, num_particles: 100, num_mh_steps: 2, seed: 42)

      betas = info.betas
      pairs = Enum.zip(betas, tl(betas))
      Enum.each(pairs, fn {a, b} -> assert a <= b end)
    end

    @tag timeout: 120_000
    test "acceptance rates are between 0 and 1" do
      ir = Builder.new_ir()
      ir = Builder.rv(ir, "x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

      {_trace, info} = SMC.sample(ir, num_particles: 100, num_mh_steps: 2, seed: 42)

      Enum.each(info.acceptance_rates, fn r ->
        assert r >= 0.0
        assert r <= 1.0
      end)
    end

    @tag timeout: 120_000
    test "ESS history contains positive values" do
      ir = Builder.new_ir()
      ir = Builder.rv(ir, "x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

      {_trace, info} = SMC.sample(ir, num_particles: 100, num_mh_steps: 2, seed: 42)

      Enum.each(info.ess_history, fn ess ->
        assert ess > 0.0
      end)
    end
  end
end
