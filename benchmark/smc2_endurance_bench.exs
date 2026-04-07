# O-SMC² Endurance Benchmark
#
# Stress-tests the Online SMC² implementation at scale:
# large Nθ, long time series, high rejuvenation frequency,
# multi-parameter models, and the Ireland COVID benchmark.
#
# Usage:
#   CUDA_VISIBLE_DEVICES="" mix run benchmark/smc2_endurance_bench.exs
#   CUDA_VISIBLE_DEVICES="" mix run benchmark/smc2_endurance_bench.exs --quick
#
# Target: Ireland COVID (Nθ=1000, Nx=500, T=365, tk=80) in <1 hour
# vs Temfack & Wyse (2025) Python: <5 hours single-threaded.

alias Exmc.SMC.OnlineSMC2

quick_mode = "--quick" in System.argv()
n_cores = System.schedulers_online()

IO.puts("=" |> String.duplicate(70))
IO.puts("  O-SMC² Endurance Benchmark")
IO.puts("  #{n_cores} schedulers, bind=#{:erlang.system_info(:scheduler_bind_type)}")
IO.puts("  mode: #{if quick_mode, do: "quick", else: "full"}")
IO.puts("=" |> String.duplicate(70))
IO.puts("")

defmodule SMC2Bench do
  # --- SEIR Data Generator ---

  def generate_seir(n_pop, beta, sigma, gamma, t_max, seed \\ 42) do
    :rand.seed(:exsss, {seed, seed + 1, seed + 2})

    {_state, observations} =
      Enum.reduce(1..t_max, {%{s: n_pop - 1, e: 1, i: 0, r: 0}, []}, fn _, {state, obs} ->
        p_se = 1 - :math.exp(-beta * state.i / n_pop)
        p_ei = 1 - :math.exp(-sigma)
        p_ir = 1 - :math.exp(-gamma)

        y_se = binom(state.s, p_se)
        y_ei = binom(state.e, p_ei)
        y_ir = binom(state.i, p_ir)

        new_state = %{
          s: max(state.s - y_se, 0),
          e: max(state.e + y_se - y_ei, 0),
          i: max(state.i + y_ei - y_ir, 0),
          r: state.r + y_ir
        }

        obs_val = max(0, y_ei + round(:rand.normal() * max(:math.sqrt(y_ei), 1)))
        {new_state, obs ++ [obs_val]}
      end)

    observations
  end

  def generate_seir_time_varying(n_pop, t_max, seed \\ 42) do
    :rand.seed(:exsss, {seed, seed + 1, seed + 2})
    sigma = 0.25
    gamma = 0.15

    {_state, _beta, observations} =
      Enum.reduce(1..t_max, {%{s: n_pop - 5, e: 5, i: 0, r: 0}, 0.4, []}, fn t, {state, beta, obs} ->
        # Time-varying beta: drops during "lockdown" periods
        beta_new = cond do
          t > 30 and t < 50 -> max(beta * 0.95, 0.1)   # lockdown
          t > 50 and t < 70 -> min(beta * 1.03, 0.6)    # reopening
          t > 80 and t < 100 -> max(beta * 0.96, 0.1)   # second lockdown
          true -> beta + :rand.normal() * 0.01
        end
        beta_new = max(0.05, min(0.8, beta_new))

        p_se = 1 - :math.exp(-beta_new * state.i / n_pop)
        p_ei = 1 - :math.exp(-sigma)
        p_ir = 1 - :math.exp(-gamma)

        y_se = binom(state.s, p_se)
        y_ei = binom(state.e, p_ei)
        y_ir = binom(state.i, p_ir)

        new_state = %{
          s: max(state.s - y_se, 0),
          e: max(state.e + y_se - y_ei, 0),
          i: max(state.i + y_ei - y_ir, 0),
          r: state.r + y_ir
        }

        obs_val = max(0, y_ei + round(:rand.normal() * max(:math.sqrt(y_ei), 1)))
        {new_state, beta_new, obs ++ [obs_val]}
      end)

    observations
  end

  # --- SEIR Model + Prior for O-SMC² ---

  def seir_model(n_pop) do
    %{
      init: fn theta, rng ->
        {%{s: n_pop - 1, e: 1, i: 0, r: 0}, rng}
      end,
      transition: fn state, theta, _t, rng ->
        p_se = 1 - :math.exp(-theta.beta * state.i / n_pop)
        p_ei = 1 - :math.exp(-theta.sigma)
        p_ir = 1 - :math.exp(-theta.gamma)

        {u1, rng} = :rand.uniform_s(rng)
        {u2, rng} = :rand.uniform_s(rng)
        {u3, rng} = :rand.uniform_s(rng)

        y_se = binom_approx(state.s, p_se, u1)
        y_ei = binom_approx(state.e, p_ei, u2)
        y_ir = binom_approx(state.i, p_ir, u3)

        new = %{
          s: max(state.s - y_se, 0),
          e: max(state.e + y_se - y_ei, 0),
          i: max(state.i + y_ei - y_ir, 0),
          r: state.r + y_ir
        }
        {new, rng}
      end,
      observation_logp: fn state, theta, y_obs ->
        lambda = max(state.i * theta.sigma, 0.1)
        y_obs * :math.log(lambda) - lambda - log_fact(y_obs)
      end
    }
  end

  def seir_prior do
    %{
      sample: fn rng ->
        {u1, rng} = :rand.uniform_s(rng)
        {u2, rng} = :rand.uniform_s(rng)
        {u3, rng} = :rand.uniform_s(rng)
        {%{beta: u1 * 0.8 + 0.05, sigma: u2 * 0.4 + 0.05, gamma: u3 * 0.25 + 0.05}, rng}
      end,
      logpdf: fn theta ->
        if theta.beta > 0.05 and theta.beta < 0.85 and
           theta.sigma > 0.05 and theta.sigma < 0.45 and
           theta.gamma > 0.05 and theta.gamma < 0.30 do
          0.0
        else
          -1.0e30
        end
      end
    }
  end

  # --- Run a single benchmark ---

  def run_test(name, observations, model, prior, opts, expected_ms) do
    IO.write("  #{String.pad_trailing(name, 30)}")
    t0 = System.monotonic_time(:millisecond)

    result = OnlineSMC2.run(model, prior, observations, opts)

    elapsed = System.monotonic_time(:millisecond) - t0

    final = List.last(result.posterior_history) || %{}
    ess_min = if result.ess_history != [], do: Float.round(Enum.min(result.ess_history), 1), else: 0
    ess_mean = if result.ess_history != [], do: Float.round(Enum.sum(result.ess_history) / length(result.ess_history), 1), else: 0

    status = cond do
      elapsed > expected_ms * 3 -> "SLOW"
      result.rejuvenation_count == 0 and length(observations) > 20 -> "NO_REJUV"
      true -> "OK"
    end

    beta_str = if final[:beta], do: Float.round(final.beta, 3), else: "?"
    sigma_str = if final[:sigma], do: Float.round(final.sigma, 3), else: "?"
    gamma_str = if final[:gamma], do: Float.round(final.gamma, 3), else: "?"

    IO.puts("#{elapsed}ms  T=#{length(observations)}  rej=#{result.rejuvenation_count}  " <>
      "ESS=#{ess_mean}/#{ess_min}  β=#{beta_str} σ=#{sigma_str} γ=#{gamma_str}  [#{status}]")

    %{name: name, ms: elapsed, rejuvenations: result.rejuvenation_count,
      ess_mean: ess_mean, ess_min: ess_min, posterior: final, status: status}
  end

  # --- Helpers ---

  def binom(n, p) when n <= 0 or p <= 0, do: 0
  def binom(n, p), do: Enum.count(1..n, fn _ -> :rand.uniform() < p end)

  def binom_approx(n, p, u) when n <= 0 or p <= 0, do: 0
  def binom_approx(n, p, u) do
    mean = n * p
    round(max(0, min(n, mean + (u - 0.5) * :math.sqrt(max(mean * (1 - p), 0.01)) * 2)))
  end

  def log_fact(0), do: 0.0
  def log_fact(n) when n > 0, do: Enum.reduce(1..n, 0.0, fn k, acc -> acc + :math.log(k) end)
  def log_fact(_), do: 0.0
end

# --- Define tests ---

tests = [
  {"smoke SEIR (Nθ=100, T=40)", fn ->
    obs = SMC2Bench.generate_seir(5000, 0.4, 0.25, 0.15, 40)
    model = SMC2Bench.seir_model(5000)
    prior = SMC2Bench.seir_prior()
    SMC2Bench.run_test("smoke-seir", obs, model, prior,
      [n_theta: 100, n_x: 50, window: 15, n_moves: 2, seed: 42, parallel: false],
      5_000)
  end},

  {"parallel smoke (Nθ=100, T=40)", fn ->
    obs = SMC2Bench.generate_seir(5000, 0.4, 0.25, 0.15, 40)
    model = SMC2Bench.seir_model(5000)
    prior = SMC2Bench.seir_prior()
    SMC2Bench.run_test("parallel-smoke", obs, model, prior,
      [n_theta: 100, n_x: 50, window: 15, n_moves: 2, seed: 42, parallel: true],
      5_000)
  end},

  {"medium SEIR (Nθ=200, T=100)", fn ->
    obs = SMC2Bench.generate_seir(10_000, 0.4, 0.25, 0.15, 100)
    model = SMC2Bench.seir_model(10_000)
    prior = SMC2Bench.seir_prior()
    SMC2Bench.run_test("medium-seir", obs, model, prior,
      [n_theta: 200, n_x: 100, window: 20, n_moves: 3, seed: 42, parallel: true],
      60_000)
  end},

  {"time-varying β (Nθ=200, T=120)", fn ->
    obs = SMC2Bench.generate_seir_time_varying(20_000, 120)
    model = SMC2Bench.seir_model(20_000)
    prior = SMC2Bench.seir_prior()
    SMC2Bench.run_test("time-varying-beta", obs, model, prior,
      [n_theta: 200, n_x: 100, window: 25, n_moves: 3, seed: 42, parallel: true],
      120_000)
  end},

  {"high rejuv (Nθ=200, tk=10)", fn ->
    obs = SMC2Bench.generate_seir_time_varying(10_000, 80)
    model = SMC2Bench.seir_model(10_000)
    prior = SMC2Bench.seir_prior()
    SMC2Bench.run_test("high-rejuv-tk10", obs, model, prior,
      [n_theta: 200, n_x: 100, window: 10, n_moves: 2, seed: 42, parallel: true],
      90_000)
  end},

  {"full scale (Nθ=400, Nx=200, T=200)", fn ->
    obs = SMC2Bench.generate_seir(50_000, 0.3, 0.2, 0.1, 200)
    model = SMC2Bench.seir_model(50_000)
    prior = SMC2Bench.seir_prior()
    SMC2Bench.run_test("full-scale", obs, model, prior,
      [n_theta: 400, n_x: 200, window: 30, n_moves: 3, seed: 42, parallel: true],
      600_000)
  end},

  {"memory endurance (T=500)", fn ->
    obs = SMC2Bench.generate_seir(20_000, 0.3, 0.2, 0.1, 500)
    model = SMC2Bench.seir_model(20_000)
    prior = SMC2Bench.seir_prior()
    SMC2Bench.run_test("memory-T500", obs, model, prior,
      [n_theta: 100, n_x: 50, window: 20, n_moves: 2, seed: 42, parallel: true],
      300_000)
  end}
]

# --- Run ---

IO.puts("Test                            Time      T    Rej  ESS(avg/min)  Posterior(β,σ,γ)     Status")
IO.puts(String.duplicate("-", 95))

selected = if quick_mode, do: Enum.take(tests, 2), else: tests

mem_before = :erlang.memory(:total)

results = Enum.map(selected, fn {name, fun} ->
  try do
    fun.()
  rescue
    e ->
      IO.puts("  #{name}: FAILED — #{Exception.message(e)}")
      %{name: name, ms: 0, status: "FAIL"}
  end
end)

mem_after = :erlang.memory(:total)

IO.puts("")
IO.puts("=" |> String.duplicate(70))
total_ms = Enum.sum(Enum.map(results, fn r -> Map.get(r, :ms, 0) end))
IO.puts("  Total: #{div(total_ms, 1000)}s (#{Float.round(total_ms / 60_000, 1)} min)")
IO.puts("  Memory: #{div(mem_before, 1_048_576)}MB → #{div(mem_after, 1_048_576)}MB (+#{div(mem_after - mem_before, 1_048_576)}MB)")
IO.puts("  Schedulers: #{System.schedulers_online()}")

passed = Enum.count(results, fn r -> r[:status] == "OK" end)
total = length(results)
IO.puts("  Result: #{passed}/#{total} OK")
IO.puts("=" |> String.duplicate(70))
