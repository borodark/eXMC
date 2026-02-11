defmodule Exmc.PathfinderTest do
  use ExUnit.Case, async: false

  alias Exmc.Builder
  alias Exmc.Dist.Normal
  alias Exmc.Pathfinder

  describe "fit/2" do
    @tag timeout: 120_000
    test "single Normal: mean near target" do
      ir = Builder.new_ir()
      ir = Builder.rv(ir, "x", Normal, %{mu: Nx.tensor(3.0), sigma: Nx.tensor(1.0)})

      {draws, info} = Pathfinder.fit(ir, num_draws: 500, max_iters: 50, seed: 42)

      assert Map.has_key?(draws, "x")
      samples = draws["x"]
      n = elem(Nx.shape(samples), 0)
      assert n == 500

      mean = samples |> Nx.mean() |> Nx.to_number()
      assert_in_delta mean, 3.0, 2.5

      assert is_number(info.elbo)
      refute info.elbo == :nan
      assert info.num_iters > 0
    end

    test "empty model returns empty draws" do
      ir = Builder.new_ir()
      {draws, info} = Pathfinder.fit(ir)
      assert draws == %{}
      assert info.num_iters == 0
    end

    @tag timeout: 120_000
    test "reproducibility: same seed gives same draws" do
      ir = Builder.new_ir()
      ir = Builder.rv(ir, "x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

      {draws1, _} = Pathfinder.fit(ir, num_draws: 50, max_iters: 20, seed: 123)
      {draws2, _} = Pathfinder.fit(ir, num_draws: 50, max_iters: 20, seed: 123)

      diff = Nx.subtract(draws1["x"], draws2["x"]) |> Nx.abs() |> Nx.sum() |> Nx.to_number()
      assert diff < 1.0e-6
    end

    @tag timeout: 120_000
    test "ELBO is finite" do
      ir = Builder.new_ir()
      ir = Builder.rv(ir, "x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

      {_draws, info} = Pathfinder.fit(ir, num_draws: 100, max_iters: 30, seed: 42)

      assert is_float(info.elbo)
      refute info.elbo == :nan
      refute info.elbo == :infinity
      refute info.elbo == :neg_infinity
    end
  end
end
