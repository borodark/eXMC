defmodule Exmc.NcpReconstructTest do
  @moduledoc """
  Non-centred reconstruction, `x = mu + sigma * z`, for vector RVs.

  The defect: a scalar parent's draws carry shape `{n}` and a `shape: {8}`
  child's `{n, 8}`. Nx broadcasts right-aligned, so `mu + sigma * z` raised
  "cannot broadcast tensor of dimensions {n} to {n, 8}", even though the leading
  sample axis was exactly what they shared. Any vectorised hierarchical model
  sampled with NCP on (the default) crashed after sampling, while rebuilding
  its trace. Found 2026-09-13 sampling the vectorised eight schools. The
  reconstruction lived in two private copies, in Exmc.NUTS.Sampler and
  Exmc.MCLMC (which Exmc.MAMS shares), and both had it; there is now one,
  `Exmc.Rewrite.NonCenteredParameterization.reconstruct/2`.
  """

  use ExUnit.Case, async: false

  import Exmc.TestHelper

  alias Exmc.{Builder, Dist}
  alias Exmc.Rewrite.NonCenteredParameterization, as: NCP

  @f64 [type: :f64, backend: Nx.BinaryBackend]

  describe "reconstruct/2" do
    test "a trace: scalar parents {n} against a vector child {n, 8}" do
      n = 5
      mu = Nx.tensor(Enum.map(1..n, &(&1 * 1.0)), @f64)
      tau = Nx.tensor(Enum.map(1..n, &(&1 * 0.5)), @f64)
      z = Nx.iota({n, 8}, @f64) |> Nx.divide(10.0)

      out =
        NCP.reconstruct(%{"mu" => mu, "tau" => tau, "theta" => z}, %{
          "theta" => %{mu: "mu", sigma: "tau"}
        })

      assert Nx.shape(out["theta"]) == {n, 8}

      for i <- 0..(n - 1), j <- 0..7 do
        expected = Nx.to_number(mu[i]) + Nx.to_number(tau[i]) * Nx.to_number(z[i][j])
        assert Nx.to_number(out["theta"][i][j]) == expected
      end

      # Parents are untouched.
      assert out["mu"] == mu
    end

    test "a single point, no sample axis: scalar parents {} against a vector child {8}" do
      z = Nx.iota({8}, @f64)

      out =
        NCP.reconstruct(
          %{"mu" => Nx.tensor(2.0, @f64), "tau" => Nx.tensor(3.0, @f64), "theta" => z},
          %{"theta" => %{mu: "mu", sigma: "tau"}}
        )

      assert Nx.to_flat_list(out["theta"]) == Enum.map(0..7, &(2.0 + 3.0 * &1))
    end

    test "constant parameters and a chain of NCP'd variables, in dependency order" do
      n = 3

      ncp = %{
        # b's parent is a, itself NCP'd, so a must be rebuilt first.
        "b" => %{mu: "a", sigma: 2.0},
        "a" => %{mu: Nx.tensor(10.0, @f64), sigma: "s"}
      }

      values = %{
        "s" => Nx.tensor([1.0, 2.0, 3.0], @f64),
        "a" => Nx.tensor([0.1, 0.2, 0.3], @f64),
        "b" => Nx.broadcast(Nx.tensor(1.0, @f64), {n, 4})
      }

      out = NCP.reconstruct(values, ncp)
      a = [10.0 + 1.0 * 0.1, 10.0 + 2.0 * 0.2, 10.0 + 3.0 * 0.3]
      assert Nx.to_flat_list(out["a"]) == a
      assert Nx.to_flat_list(out["b"]) == Enum.flat_map(a, &List.duplicate(&1 + 2.0, 4))
    end

    test "no NCP info: values come back unchanged" do
      values = %{"x" => Nx.tensor([1.0], @f64)}
      assert NCP.reconstruct(values, %{}) == values
    end
  end

  describe "sampling a vectorised hierarchical model with NCP on (the default)" do
    setup do
      put_env_scoped(:compiler, :none)
      :ok
    end

    defp eight_schools do
      y = Nx.tensor([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0], type: :f64)
      sigma = Nx.tensor([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0], type: :f64)

      Builder.new_ir()
      |> Builder.rv("mu", Dist.Normal, %{
        mu: Nx.tensor(0.0, type: :f64),
        sigma: Nx.tensor(5.0, type: :f64)
      })
      |> Builder.rv("tau", Dist.HalfNormal, %{sigma: Nx.tensor(5.0, type: :f64)})
      |> Builder.rv("theta", Dist.Normal, %{mu: "mu", sigma: "tau"}, shape: {8})
      |> Builder.rv("y", Dist.Normal, %{mu: "theta", sigma: sigma}, shape: {8})
      |> Builder.obs("y_obs", "y", y)
    end

    @tag timeout: 300_000
    test "NUTS: the trace is built, theta has shape {n, 8}" do
      {trace, _stats} =
        Exmc.NUTS.Sampler.sample(eight_schools(), %{}, num_warmup: 50, num_samples: 40, seed: 3)

      assert Nx.shape(trace["theta"]) == {40, 8}
      assert Nx.shape(trace["mu"]) == {40}
      assert trace["theta"] |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 0
    end

    @tag timeout: 300_000
    test "MCLMC (the copy MAMS shares): the trace is built, theta has shape {n, 8}" do
      {trace, _stats} =
        Exmc.MCLMC.sample(eight_schools(), %{}, num_warmup: 50, num_samples: 40, seed: 3)

      assert Nx.shape(trace["theta"]) == {40, 8}
    end
  end
end
