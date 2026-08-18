defmodule Exmc.SBI.DeterminismTest do
  use ExUnit.Case, async: true

  @moduledoc """
  C2.3: the parallel particle evaluation, and the property that makes it
  testable.

  ABC's cost is `N` independent simulator runs per population, which is the
  most embarrassingly parallel workload in the roadmap — and the usual price of
  taking that parallelism is that the run stops being reproducible, because the
  order in which workers consume a shared PRNG is a scheduling detail.

  `Exmc.SBI` does not pay that price. Proposals are generated sequentially from
  the parent `:rand` state, each carries a derived child state, and the batch
  size is fixed before any simulator runs. So the assertions below are
  **equalities on the full particle set**, not comparisons of moments: a
  parallel run and a sequential run over the same seed are the same run.

  That is worth more than it looks. `MISSION.md` §6.3 records two defects that
  survived a green suite because every statistical check in the repository was
  differential; an exact equality over the whole output is the one shape of
  test that cannot be satisfied by two arms being wrong together, because there
  is only one arm.
  """

  # A simulator with enough structure that a mis-threaded RNG would show up:
  # the summary depends on both parameters and on several draws.
  defp sim do
    fn %{a: a, b: b}, rng ->
      {vals, rng} =
        Enum.map_reduce(1..12, rng, fn _, rng ->
          {z, rng} = :rand.normal_s(rng)
          {a + b * z, rng}
        end)

      mean = Enum.sum(vals) / 12
      var = Enum.reduce(vals, 0.0, fn v, acc -> acc + (v - mean) * (v - mean) end) / 11
      {[mean, :math.sqrt(var)], rng}
    end
  end

  defp prior, do: [a: {:normal, 0.0, 2.0}, b: {:lognormal, 0.0, 0.5}]

  defp base_opts do
    [prior: prior(), n_particles: 60, n_populations: 4, seed: 4242]
  end

  defp strip(posterior), do: Map.drop(posterior, [:prior])

  describe "ABC-SMC" do
    test "parallel and sequential runs are bit-identical" do
      {:ok, par} = Exmc.SBI.ABCSMC.run(sim(), [1.0, 1.0], base_opts() ++ [parallel: true])
      {:ok, seq} = Exmc.SBI.ABCSMC.run(sim(), [1.0, 1.0], base_opts() ++ [parallel: false])

      assert strip(par) == strip(seq)
      assert par.n_simulations == seq.n_simulations
      assert par.stopped == :completed
    end

    test "the answer does not depend on how many workers ran it" do
      results =
        for c <- [1, 2, 7, 64] do
          {:ok, p} =
            Exmc.SBI.ABCSMC.run(sim(), [1.0, 1.0], base_opts() ++ [max_concurrency: c])

          strip(p)
        end

      assert length(Enum.uniq(results)) == 1
    end

    test "the same seed gives the same run, a different seed does not" do
      {:ok, a} = Exmc.SBI.ABCSMC.run(sim(), [1.0, 1.0], base_opts())
      {:ok, b} = Exmc.SBI.ABCSMC.run(sim(), [1.0, 1.0], base_opts())
      {:ok, c} = Exmc.SBI.ABCSMC.run(sim(), [1.0, 1.0], Keyword.put(base_opts(), :seed, 4243))

      assert strip(a) == strip(b)
      refute strip(a) == strip(c)
    end

    test "an explicit :rand state is accepted and used" do
      rng = :rand.seed_s(:exsss, {9, 9, 9})
      {:ok, a} = Exmc.SBI.ABCSMC.run(sim(), [1.0, 1.0], Keyword.put(base_opts(), :rng, rng))
      {:ok, b} = Exmc.SBI.ABCSMC.run(sim(), [1.0, 1.0], Keyword.put(base_opts(), :rng, rng))
      assert strip(a) == strip(b)
    end

    test "a module implementing the behaviour and the equivalent closure agree" do
      defmodule ModuleSim do
        @behaviour Exmc.SBI.Simulator

        @impl true
        def simulate(%{a: a, b: b}, rng) do
          {vals, rng} =
            Enum.map_reduce(1..12, rng, fn _, rng ->
              {z, rng} = :rand.normal_s(rng)
              {a + b * z, rng}
            end)

          mean = Enum.sum(vals) / 12
          var = Enum.reduce(vals, 0.0, fn v, acc -> acc + (v - mean) * (v - mean) end) / 11
          # Returned as a tensor rather than a list, to pin that both are
          # accepted and produce the same coordinate vector.
          {Nx.tensor([mean, :math.sqrt(var)], type: :f64, backend: Nx.BinaryBackend), rng}
        end
      end

      {:ok, from_mod} = Exmc.SBI.ABCSMC.run(ModuleSim, [1.0, 1.0], base_opts())
      {:ok, from_fun} = Exmc.SBI.ABCSMC.run(sim(), [1.0, 1.0], base_opts())

      assert strip(from_mod) == strip(from_fun)
    end
  end

  describe "rejection ABC" do
    test "parallel and sequential runs are bit-identical" do
      opts = [prior: prior(), n_particles: 40, epsilon: 1.5, seed: 77]
      {:ok, par} = Exmc.SBI.ABC.run(sim(), [1.0, 1.0], opts ++ [parallel: true])
      {:ok, seq} = Exmc.SBI.ABC.run(sim(), [1.0, 1.0], opts ++ [parallel: false])

      assert strip(par) == strip(seq)
    end

    test "quantile mode is bit-identical across concurrency too" do
      opts = [prior: prior(), n_particles: 40, n_draws: 2000, seed: 77]

      results =
        for c <- [1, 3, 32] do
          {:ok, p} = Exmc.SBI.ABC.run(sim(), [1.0, 1.0], opts ++ [max_concurrency: c])
          strip(p)
        end

      assert length(Enum.uniq(results)) == 1
    end
  end

  describe "argument validation" do
    test "a simulator must be a module with simulate/2 or a 2-arity function" do
      assert_raise ArgumentError, fn -> Exmc.SBI.Simulator.validate!(Enum) end
      assert_raise ArgumentError, fn -> Exmc.SBI.Simulator.validate!(fn -> :ok end) end
      assert_raise ArgumentError, fn -> Exmc.SBI.Simulator.validate!(42) end
    end

    test "a summary whose length changes is rejected rather than silently truncated" do
      wobbly = fn %{a: a}, rng ->
        if a > 0, do: {[1.0, 2.0], rng}, else: {[1.0], rng}
      end

      assert_raise ArgumentError, ~r/length/, fn ->
        Exmc.SBI.ABC.run!(wobbly, [1.0, 2.0],
          prior: [a: {:normal, 0.0, 1.0}],
          n_particles: 5,
          n_draws: 200,
          parallel: false
        )
      end
    end

    test "rejection ABC needs exactly one of :epsilon and :n_draws" do
      opts = [prior: prior(), n_particles: 5]
      assert {:error, {:missing_tolerance, _}} = Exmc.SBI.ABC.run(sim(), [1.0, 1.0], opts)

      assert {:error, {:bad_options, _}} =
               Exmc.SBI.ABC.run(sim(), [1.0, 1.0], opts ++ [epsilon: 1.0, n_draws: 100])
    end

    test "an unreachable tolerance is an error, not an empty posterior" do
      assert {:error, {:budget_exhausted, info}} =
               Exmc.SBI.ABC.run(sim(), [1.0, 1.0],
                 prior: prior(),
                 n_particles: 20,
                 epsilon: 1.0e-9,
                 max_simulations: 2000,
                 parallel: false
               )

      assert info.accepted < 20
      assert info.n_simulations >= 2000
    end

    test "Exmc.SBI.run/3 dispatches on :method" do
      {:ok, smc} = Exmc.SBI.run(sim(), [1.0, 1.0], base_opts())
      assert smc.method == :abc_smc

      {:ok, rej} =
        Exmc.SBI.run(sim(), [1.0, 1.0],
          method: :rejection,
          prior: prior(),
          n_particles: 30,
          n_draws: 1000,
          seed: 1
        )

      assert rej.method == :rejection
      assert {:error, {:bad_options, _}} = Exmc.SBI.run(sim(), [1.0, 1.0], method: :nope)
    end
  end

  describe "posterior helpers" do
    test "weighted summaries agree with hand computation" do
      posterior = %{
        names: [:x],
        particles: [%{x: 1.0}, %{x: 3.0}, %{x: 5.0}],
        weights: [0.5, 0.25, 0.25]
      }

      assert_in_delta Exmc.SBI.posterior_mean(posterior).x, 2.5, 1.0e-12
      # var = .5·2.25 + .25·.25 + .25·6.25 = 1.125 + 0.0625 + 1.5625 = 2.75
      assert_in_delta Exmc.SBI.posterior_sd(posterior).x, :math.sqrt(2.75), 1.0e-12
      assert_in_delta Exmc.SBI.weight_ess(posterior), 1.0 / 0.375, 1.0e-12

      assert_in_delta Exmc.SBI.posterior_quantile(posterior, :x, 0.4), 1.0, 1.0e-12
      assert_in_delta Exmc.SBI.posterior_quantile(posterior, :x, 0.6), 3.0, 1.0e-12
      assert_in_delta Exmc.SBI.posterior_quantile(posterior, :x, 0.99), 5.0, 1.0e-12
    end

    test "systematic resampling reproduces the weights it was given" do
      posterior = %{
        names: [:x],
        particles: [%{x: 1.0}, %{x: 2.0}, %{x: 3.0}],
        weights: [0.6, 0.3, 0.1]
      }

      {draws, _} = Exmc.SBI.resample(posterior, 1000, 5)
      counts = draws |> Enum.map(& &1.x) |> Enum.frequencies()

      # Systematic resampling puts exactly floor(n·w) or ceil(n·w) copies of
      # each particle in the sample, so this is not a statistical assertion —
      # it is a bound the method guarantees.
      assert counts[1.0] in [600, 601]
      assert counts[2.0] in [300, 301]
      assert counts[3.0] in [100, 101]
    end

    test "the SBC rank is between 0 and n_draws and moves with the truth" do
      posterior = %{
        names: [:x],
        particles: Enum.map(1..100, fn i -> %{x: i / 10.0} end),
        weights: List.duplicate(0.01, 100)
      }

      {low, _} = Exmc.SBI.rank(posterior, :x, -5.0, 99, 1)
      {high, _} = Exmc.SBI.rank(posterior, :x, 50.0, 99, 1)
      {mid, _} = Exmc.SBI.rank(posterior, :x, 5.0, 99, 1)

      assert low == 0
      assert high == 99
      assert mid > 30 and mid < 70
    end
  end
end
