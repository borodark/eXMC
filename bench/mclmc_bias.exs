# B1.4 — the MCLMC bias measurement.
#
# MCLMC is biased by construction. An analytic-moment *assertion* is the wrong
# tool for it: it would either fail correctly and block a feature that is
# working as designed, or get widened until it passes, which is how the 0.3.0
# variance defect survived a green suite. So the gate is a published number.
#
# This script sweeps the step size and reports, for each of three targets:
#
#   * bias in mean and in variance, against the analytic truth
#   * the same for MAMS and for NUTS on the identical model, as the unbiased
#     references
#   * ESS per gradient evaluation for all three, because MCLMC's whole claim is
#     that it buys speed with that bias
#
# Everything is reported. Nothing here asserts. The assertions live in
# test/mclmc/*.exs.
#
# Usage:
#   mix run --no-deps-check bench/mclmc_bias.exs
#   DIMS=2,8,32 SEEDS=1,2,3,4 SAMPLES=4000 mix run --no-deps-check bench/mclmc_bias.exs
#   OUT=bench_results/MCLMC_BIAS.md mix run --no-deps-check bench/mclmc_bias.exs

alias Exmc.{Builder, Rewrite}
alias Exmc.Dist.{Exponential, HalfNormal, Normal}
alias Exmc.NUTS.Sampler
alias Exmc.NUTS.Vulkan.Validator

env = fn n, d -> System.get_env(n) || d end
ints = fn s -> s |> String.split(",", trim: true) |> Enum.map(&String.to_integer(String.trim(&1))) end
floats = fn s -> s |> String.split(",", trim: true) |> Enum.map(&String.to_float(String.trim(&1))) end

compiler = env.("COMPILER", "none") |> String.to_atom()
Application.put_env(:exmc, :compiler, compiler)

seeds = env.("SEEDS", "1,2,3,4") |> ints.()
warmup = env.("WARMUP", "1000") |> String.to_integer()
samples = env.("SAMPLES", "4000") |> String.to_integer()
dims = env.("DIMS", "2,8,32") |> ints.()
epsilons = env.("EPS", "0.1,0.25,0.5,1.0,2.0,4.0") |> floats.()
out = System.get_env("OUT")

host = :inet.gethostname() |> elem(1) |> to_string()

targets = [
  {"Normal(0,1)", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)}, {:normal, 0.0, 1.0}},
  {"HalfNormal(1)", HalfNormal, %{sigma: Nx.tensor(1.0)}, {:half_normal, 1.0}},
  {"Exponential(2)", Exponential, %{lambda: Nx.tensor(2.0)}, {:exponential, 2.0}}
]

build = fn dist, params, d ->
  Enum.reduce(1..d, Builder.new_ir(), fn i, ir ->
    Builder.rv(ir, "x#{i}", dist, params)
  end)
  |> Rewrite.apply()
  |> Exmc.Compiler.compile_for_sampling()
end

# One pass per (sampler, seed). Draws are pooled across coordinates and seeds
# for the moment estimates — every coordinate is i.i.d. under all three
# targets, so this is the largest honest sample available.
#
# ESS is *not* computed on that pooled concatenation. Gluing independent chains
# end to end manufactures a long correlation time at every join and would
# understate ESS by roughly the number of chains. It is estimated per
# chain-coordinate and summed instead.
#
# `grad_fn` extracts the gradient budget from the sampler's own stats, because
# the three samplers count it differently and hard-coding any one of them is
# how a cost comparison quietly becomes wrong.
collect = fn compiled, d, sampler_fn, grad_fn ->
  init = {[], 0.0, 0, 0, 0.0, nil}

  {chunks, ess, grads, divs, secs, last} =
    Enum.reduce(seeds, init, fn seed, {acc, ess, grads, divs, secs, _last} ->
      t0 = System.monotonic_time(:microsecond)
      {trace, stats} = sampler_fn.(compiled, seed)
      t1 = System.monotonic_time(:microsecond)

      cols = Enum.map(1..d, fn i -> trace |> Map.fetch!("x#{i}") |> Nx.to_flat_list() end)

      {[cols | acc], ess + Enum.reduce(cols, 0.0, fn c, a -> a + Validator.ess(c) end),
       grads + grad_fn.(stats), divs + Map.get(stats, :divergences, 0),
       secs + (t1 - t0) / 1.0e6, stats}
    end)

  {chunks |> Enum.reverse() |> List.flatten(), ess, grads, divs, secs, last}
end

grads_of = fn stats -> Map.get(stats, :grad_evals, 0) end

# NUTS does not report a gradient budget. Its sampling phase is exact from the
# per-step `n_steps` it already records; its warmup is charged at the
# sampling-phase mean rather than ignored, which is an estimate and is flagged
# as one in the table.
nuts_grads_of = fn stats ->
  steps =
    stats
    |> Map.get(:sample_stats, [])
    |> Enum.reduce(0, fn s, a -> a + Map.get(s, :n_steps, 0) end)

  per = if samples > 0, do: steps / samples, else: 0.0
  steps + round(per * warmup)
end

pct = fn got, truth ->
  if abs(truth) > 1.0e-12 do
    :erlang.float_to_binary((got - truth) / truth * 100.0, decimals: 2) <> "%"
  else
    :erlang.float_to_binary(got - truth, decimals: 5)
  end
end

f = fn x, n -> :erlang.float_to_binary(x * 1.0, decimals: n) end

sci = fn x ->
  cond do
    x == 0.0 -> "0"
    abs(x) >= 0.01 -> :erlang.float_to_binary(x, decimals: 4)
    true -> :erlang.float_to_binary(x, [:compact, decimals: 8])
  end
end

lines = fn s -> IO.puts(s) end

collected = []

header = """
# MCLMC / MAMS bias and cost — B1.4

**Host:** `#{host}`
**Generated:** #{DateTime.utc_now() |> DateTime.to_iso8601()}
**Backend:** `#{inspect(Nx.default_backend())}` · compiler `#{compiler}`
**Elixir/OTP:** #{System.version()} / #{System.otp_release()}
**Config:** warmup #{warmup}, samples #{samples}, seeds #{inspect(seeds)}, dims #{inspect(dims)}

MCLMC is biased by construction and this file is its gate — a **published
number**, not an assertion. MAMS and NUTS are the unbiased references on the
identical model. `err` columns are relative to the analytic truth; `ESS/grad`
is effective samples per gradient evaluation, summed over coordinates and
chains, divided by the total gradient budget including warmup.

Reproduce with:

```sh
DIMS=#{Enum.join(dims, ",")} SEEDS=#{Enum.join(seeds, ",")} \\
  WARMUP=#{warmup} SAMPLES=#{samples} EPS=#{Enum.join(Enum.map(epsilons, &to_string/1), ",")} \\
  mix run --no-deps-check bench/mclmc_bias.exs
```
"""

lines.(header)
collected = [header | collected]

collected =
  Enum.reduce(dims, collected, fn d, collected ->
    Enum.reduce(targets, collected, fn {label, dist, params, meta}, collected ->
      compiled = build.(dist, params, d)
      {:moments, %{mean: tm, var: tv}} = Validator.analytic_moments(meta)

      title = "\n## #{label}, d = #{d}\n"
      lines.(title)

      head =
        "| sampler | eps | mean | err | var | err | ESS | grads | ESS/grad | div |\n" <>
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"

      lines.(head)

      # --- MCLMC, one row per pinned step size
      mclmc_rows =
        Enum.map(epsilons, fn eps ->
          fn_ = fn c, seed ->
            Exmc.MCLMC.sample_compiled(c, %{},
              num_warmup: warmup,
              num_samples: samples,
              seed: seed,
              step_size: eps
            )
          end

          {xs, ess, grads, divs, _secs, _st} = collect.(compiled, d, fn_, grads_of)
          {m, v} = Validator.mean_var(xs)

          row =
            "| MCLMC | #{f.(eps, 3)} | #{f.(m, 4)} | #{pct.(m, tm)} | #{f.(v, 4)} | " <>
              "#{pct.(v, tv)} | #{f.(ess, 0)} | #{grads} | #{sci.(ess / max(grads, 1))} | #{divs} |"

          lines.(row)
          row
        end)

      # --- MCLMC with its own EEVPD tuning left on (the shipping default)
      auto_fn = fn c, seed ->
        Exmc.MCLMC.sample_compiled(c, %{},
          num_warmup: warmup,
          num_samples: samples,
          seed: seed
        )
      end

      {xs_a, ess_a, grads_a, div_a, _, st_a} = collect.(compiled, d, auto_fn, grads_of)
      {m_a, v_a} = Validator.mean_var(xs_a)

      auto_row =
        "| MCLMC (tuned) | #{f.(st_a.step_size, 3)} | #{f.(m_a, 4)} | #{pct.(m_a, tm)} | " <>
          "#{f.(v_a, 4)} | #{pct.(v_a, tv)} | #{f.(ess_a, 0)} | #{grads_a} | " <>
          "#{sci.(ess_a / max(grads_a, 1))} | #{div_a} |"

      lines.(auto_row)

      # --- MAMS
      mams_fn = fn c, seed ->
        Exmc.MAMS.sample_compiled(c, %{},
          num_warmup: warmup,
          num_samples: samples,
          seed: seed
        )
      end

      {xs_m, ess_m, grads_m, div_m, _, st_m} = collect.(compiled, d, mams_fn, grads_of)
      {m_m, v_m} = Validator.mean_var(xs_m)

      mams_row =
        "| MAMS | #{f.(st_m.step_size, 3)} | #{f.(m_m, 4)} | #{pct.(m_m, tm)} | " <>
          "#{f.(v_m, 4)} | #{pct.(v_m, tv)} | #{f.(ess_m, 0)} | #{grads_m} | " <>
          "#{sci.(ess_m / max(grads_m, 1))} | #{div_m} |"

      lines.(mams_row)

      # --- NUTS, the unbiased reference every other row is measured against
      nuts_fn = fn c, seed ->
        Sampler.sample_compiled(c, %{},
          num_warmup: warmup,
          num_samples: samples,
          seed: seed
        )
      end

      {xs_n, ess_n, nuts_grads, div_n, _, st_n} =
        collect.(compiled, d, nuts_fn, nuts_grads_of)

      {m_n, v_n} = Validator.mean_var(xs_n)

      nuts_row =
        "| NUTS | #{f.(st_n.step_size, 3)} | #{f.(m_n, 4)} | #{pct.(m_n, tm)} | " <>
          "#{f.(v_n, 4)} | #{pct.(v_n, tv)} | #{f.(ess_n, 0)} | #{nuts_grads}* | " <>
          "#{sci.(ess_n / max(nuts_grads, 1))} | #{div_n} |"

      lines.(nuts_row)

      note =
        "\n`*` NUTS's warmup gradients are estimated at the sampling-phase mean " <>
          "`n_steps`; its sampling-phase count is exact.\n" <>
          "Truth: mean #{f.(tm, 6)}, var #{f.(tv, 6)}.\n"

      lines.(note)

      block =
        [title, head] ++ mclmc_rows ++ [auto_row, mams_row, nuts_row, note]

      collected ++ block
    end)
  end)

if out do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, Enum.join(collected, "\n") <> "\n")
  IO.puts("\nwrote #{out}")
end
