# CPU Pinning Benchmark
# Run with different scheduler binding strategies:
#
#   CUDA_VISIBLE_DEVICES="" mix run benchmark/cpu_pinning_bench.exs
#   CUDA_VISIBLE_DEVICES="" elixir --erl "+sbt db" -S mix run benchmark/cpu_pinning_bench.exs
#   CUDA_VISIBLE_DEVICES="" elixir --erl "+sbt ts" -S mix run benchmark/cpu_pinning_bench.exs
#   CUDA_VISIBLE_DEVICES="" elixir --erl "+sbt tnnps" -S mix run benchmark/cpu_pinning_bench.exs
#
# +sbt strategies:
#   u      = unbound (default, OS decides)
#   db     = default bind (bind schedulers to cores)
#   ts     = thread spread (one scheduler per physical core, spread across)
#   tnnps  = thread no-node processor spread (NUMA-aware)

alias Exmc.{Builder, Sampler}
alias Exmc.Dist.{Normal, HalfNormal, Custom}

sbt = :erlang.system_info(:scheduler_bind_type)
scheds = :erlang.system_info(:schedulers_online)
IO.puts("=== CPU Pinning Benchmark ===")
IO.puts("Scheduler bind: #{sbt}")
IO.puts("Schedulers: #{scheds}")
IO.puts("")

# Build a regime-like model (d=8, similar to trading)
defmodule BenchModel do
  def t(v), do: Nx.tensor(v, type: :f64)

  def build(n_obs) do
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
        mu_trend: "mu_trend",
        sigma_trend: "sigma_trend",
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
end

{ir, init} = BenchModel.build(200)

# --- Sequential baseline ---
IO.puts("--- Sequential (1 job) ---")
{us_seq, _} = :timer.tc(fn ->
  Sampler.sample(ir, init, num_warmup: 500, num_samples: 500, seed: 42, ncp: false)
end)
ms_seq = div(us_seq, 1000)
IO.puts("  #{ms_seq}ms per sample")
IO.puts("")

# --- Parallel: 10 concurrent ---
n_parallel = 10
IO.puts("--- Parallel (#{n_parallel} concurrent) ---")
{us_par, _} = :timer.tc(fn ->
  1..n_parallel
  |> Task.async_stream(fn seed ->
    Sampler.sample(ir, init, num_warmup: 500, num_samples: 500, seed: seed, ncp: false)
  end, max_concurrency: n_parallel, timeout: 120_000)
  |> Enum.to_list()
end)
ms_par = div(us_par, 1000)
ms_per = div(ms_par, 1)  # wall time for all 10
throughput = n_parallel / (ms_par / 1000)
IO.puts("  #{ms_par}ms wall time for #{n_parallel} jobs")
IO.puts("  #{Float.round(throughput, 2)} jobs/sec")
IO.puts("")

# --- Parallel: 44 concurrent ---
n_parallel2 = 44
IO.puts("--- Parallel (#{n_parallel2} concurrent) ---")
{us_par2, _} = :timer.tc(fn ->
  1..n_parallel2
  |> Task.async_stream(fn seed ->
    Sampler.sample(ir, init, num_warmup: 500, num_samples: 500, seed: seed, ncp: false)
  end, max_concurrency: n_parallel2, timeout: 300_000)
  |> Enum.to_list()
end)
ms_par2 = div(us_par2, 1000)
throughput2 = n_parallel2 / (ms_par2 / 1000)
IO.puts("  #{ms_par2}ms wall time for #{n_parallel2} jobs")
IO.puts("  #{Float.round(throughput2, 2)} jobs/sec")
IO.puts("")

IO.puts("--- Summary ---")
IO.puts("  bind=#{sbt}  1x=#{ms_seq}ms  #{n_parallel}x=#{Float.round(throughput, 2)}j/s  #{n_parallel2}x=#{Float.round(throughput2, 2)}j/s")
