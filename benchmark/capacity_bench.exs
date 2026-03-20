# Capacity Planning Benchmark
#
# Simulates N instruments receiving ticks and sampling posteriors.
# Measures wall time per round, peak memory, and throughput.
#
# Usage:
#   CUDA_VISIBLE_DEVICES="" mix run benchmark/capacity_bench.exs
#
# Output: capacity model table for planning portfolio sizes.

alias Exmc.{Builder, Sampler}
alias Exmc.Dist.{Normal, HalfNormal, Custom}

IO.puts("=" |> String.duplicate(70))
IO.puts("  eXMC Capacity Planning Benchmark")
IO.puts("  #{System.schedulers_online()} schedulers, bind=#{:erlang.system_info(:scheduler_bind_type)}")
IO.puts("=" |> String.duplicate(70))
IO.puts("")

# --- Build a regime-like model (d=8, same as trading) ---
defmodule CapBench do
  def t(v), do: Nx.tensor(v, type: :f64)

  def build_model(n_obs \\ 200) do
    y = Nx.tensor(Enum.map(1..n_obs, fn _ -> :rand.normal() end), type: :f64)

    ir =
      Builder.new_ir()
      |> Builder.data(y)
      |> Builder.rv("mu_trend", Normal, %{mu: t(0.0), sigma: t(1.0)})
      |> Builder.rv("sigma_trend", HalfNormal, %{sigma: t(1.0)})
      |> Builder.rv("sigma_mr", HalfNormal, %{sigma: t(1.0)})
      |> Builder.rv("sigma_vol", HalfNormal, %{sigma: t(2.0)})
      |> Builder.rv("theta_mr", Normal, %{mu: t(0.0), sigma: t(1.0)})
      |> Builder.rv("mu_mr", Normal, %{mu: t(0.0), sigma: t(0.5)})
      |> Builder.rv("logit_w1", Normal, %{mu: t(0.0), sigma: t(1.0)})
      |> Builder.rv("logit_w2", Normal, %{mu: t(0.0), sigma: t(1.0)})

    logpdf_fn = fn _x, params ->
      obs = params.__obs_data
      mu = params.mu_trend
      sigma = Nx.max(params.sigma_trend, t(1.0e-8))
      z = Nx.divide(Nx.subtract(obs, mu), sigma)
      Nx.sum(Nx.subtract(Nx.multiply(t(-0.5), Nx.multiply(z, z)), Nx.log(sigma)))
    end

    dist = Custom.new(logpdf_fn, support: :real)

    ir =
      Custom.rv(ir, "obs_lik", dist, %{
        mu_trend: "mu_trend", sigma_trend: "sigma_trend",
        __obs_data: "__obs_data"
      })
      |> Builder.obs("obs_lik_obs", "obs_lik", Nx.tensor(0.0, type: :f64))

    init = %{
      "mu_trend" => 0.0, "sigma_trend" => 0.5,
      "sigma_mr" => 0.5, "sigma_vol" => 1.0,
      "theta_mr" => 0.0, "mu_mr" => 0.0,
      "logit_w1" => 0.0, "logit_w2" => 0.0
    }

    {ir, init}
  end

  def measure_round(ir, init, n_instruments, max_concurrent) do
    mem_before = :erlang.memory(:total)

    {wall_us, _} = :timer.tc(fn ->
      1..n_instruments
      |> Task.async_stream(
        fn seed ->
          Sampler.sample(ir, init,
            num_warmup: 500, num_samples: 200, seed: seed, ncp: false)
        end,
        max_concurrency: max_concurrent,
        timeout: 600_000
      )
      |> Enum.to_list()
    end)

    mem_after = :erlang.memory(:total)
    :erlang.garbage_collect()
    Process.sleep(500)
    mem_settled = :erlang.memory(:total)

    wall_s = wall_us / 1_000_000
    throughput = n_instruments / wall_s
    per_job = wall_s / max(ceil(n_instruments / max_concurrent), 1)
    mem_peak_mb = max(mem_after - mem_before, 0) / 1_048_576
    mem_resident_mb = mem_settled / 1_048_576

    %{
      n: n_instruments,
      wall_s: Float.round(wall_s, 1),
      throughput: Float.round(throughput, 2),
      per_batch_s: Float.round(per_job, 1),
      mem_peak_mb: Float.round(mem_peak_mb, 0),
      mem_total_mb: Float.round(mem_resident_mb, 0)
    }
  end
end

# --- Warmup JIT cache (first call compiles) ---
IO.puts("Warming up JIT cache...")
{ir, init} = CapBench.build_model(200)
Sampler.sample(ir, init, num_warmup: 100, num_samples: 50, seed: 0, ncp: false)
IO.puts("JIT warm. Starting benchmark.\n")

max_concurrent = div(System.schedulers_online(), 2)
IO.puts("Max concurrent workers: #{max_concurrent}")
IO.puts("")

# --- Scale test ---
test_sizes = [10, 25, 50, 100, 200, 500]

# Header
IO.puts(String.pad_trailing("N", 6) <>
  String.pad_trailing("Wall(s)", 10) <>
  String.pad_trailing("Jobs/s", 10) <>
  String.pad_trailing("Batch(s)", 10) <>
  String.pad_trailing("MemPeak(MB)", 13) <>
  String.pad_trailing("MemTotal(MB)", 13))
IO.puts(String.duplicate("-", 62))

results =
  Enum.map(test_sizes, fn n ->
    IO.write("#{n}...")
    # Each instrument gets slightly different data (different seed)
    result = CapBench.measure_round(ir, init, n, max_concurrent)
    IO.puts("\r" <>
      String.pad_trailing("#{result.n}", 6) <>
      String.pad_trailing("#{result.wall_s}", 10) <>
      String.pad_trailing("#{result.throughput}", 10) <>
      String.pad_trailing("#{result.per_batch_s}", 10) <>
      String.pad_trailing("#{result.mem_peak_mb}", 13) <>
      String.pad_trailing("#{result.mem_total_mb}", 13))

    # GC between rounds
    for pid <- Process.list(), Process.alive?(pid), do: :erlang.garbage_collect(pid)
    Process.sleep(2000)

    result
  end)

IO.puts("")

# --- Extrapolation model ---
# Linear fit: wall_time = N / throughput_steady
# Use the largest test point as the steady-state throughput
steady = List.last(results)
tp = steady.throughput

IO.puts("=" |> String.duplicate(70))
IO.puts("  Capacity Model (#{max_concurrent} concurrent workers)")
IO.puts("=" |> String.duplicate(70))
IO.puts("")
IO.puts("Steady-state throughput: #{tp} jobs/sec")
IO.puts("Per-job cost: ~#{Float.round(1/tp, 2)}s wall time per instrument")
IO.puts("")

for n <- [100, 250, 500, 1000, 2000, 5000] do
  wall = Float.round(n / tp, 0)
  interval_min = Float.round(wall / 60, 1)
  # Memory model: ~3MB per concurrent job + ~0.5MB per instrument state
  mem_est = Float.round(max_concurrent * 3 + n * 0.5, 0)
  IO.puts("  #{String.pad_trailing("#{n} instruments:", 20)} ~#{trunc(wall)}s per round (#{interval_min} min), ~#{trunc(mem_est)}MB RAM")
end

IO.puts("")
IO.puts("Note: poll_interval must be > round_time for the pool to keep up.")
IO.puts("With update_every=20 ticks at 60s poll: round fires every ~20 min.")
IO.puts("Capacity limit: #{trunc(tp * 20 * 60)} instruments at 20-min update cycle.")
