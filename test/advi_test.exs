defmodule Exmc.ADVITest do
  use ExUnit.Case, async: false

  alias Exmc.Builder
  alias Exmc.Dist.Normal
  alias Exmc.ADVI

  describe "fit/2" do
    test "empty model returns empty draws" do
      ir = Builder.new_ir()
      {draws, info} = ADVI.fit(ir)
      assert draws == %{}
      assert info.converged == true
      assert info.num_iters == 0
    end

    @tag timeout: 120_000
    test "single Normal: mean near target" do
      ir = Builder.new_ir()
      ir = Builder.rv(ir, "x", Normal, %{mu: Nx.tensor(5.0), sigma: Nx.tensor(1.0)})

      {draws, info} = ADVI.fit(ir, num_draws: 200, max_iters: 500, learning_rate: 0.05, seed: 42)

      assert Map.has_key?(draws, "x")
      samples = draws["x"]
      n = elem(Nx.shape(samples), 0)
      assert n == 200

      mean = samples |> Nx.mean() |> Nx.to_number()
      assert_in_delta mean, 5.0, 2.0

      assert is_list(info.elbo_history)
      assert length(info.elbo_history) > 0
    end

    @tag timeout: 120_000
    test "ELBO history is a list of numbers" do
      ir = Builder.new_ir()
      ir = Builder.rv(ir, "x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

      {_draws, info} = ADVI.fit(ir, num_draws: 50, max_iters: 100, seed: 42)

      assert is_list(info.elbo_history)

      Enum.each(info.elbo_history, fn e ->
        assert is_number(e)
      end)
    end

    @tag timeout: 120_000
    test "reproducibility: same seed gives same draws" do
      ir = Builder.new_ir()
      ir = Builder.rv(ir, "x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

      {draws1, _} = ADVI.fit(ir, num_draws: 50, max_iters: 100, seed: 123)
      {draws2, _} = ADVI.fit(ir, num_draws: 50, max_iters: 100, seed: 123)

      diff = Nx.subtract(draws1["x"], draws2["x"]) |> Nx.abs() |> Nx.sum() |> Nx.to_number()
      assert diff < 1.0e-6
    end
  end
end
