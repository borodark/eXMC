# Crash recovery under supervision — does a recovered run still sample?
#
# `supervised: true` catches a crashed subtree and substitutes a divergent
# placeholder. The question this measures is what that costs the posterior,
# and it has been wrong twice:
#
#   * the placeholder reported acceptance 0.0, which dual averaging read as
#     "eps is far too large" — closing a feedback loop (smaller eps -> longer
#     trajectories -> deeper trees -> more crashes) that drove eps to 2.41e-11
#     and collapsed the posterior to variance 1.45e-15 WHILE REPORTING SUCCESS
#   * supervision itself was bypassed entirely whenever the speculative
#     pre-compute buffer was live, i.e. on the default path
#
# Both are fixed. This script is how you check they stay fixed, and it prints
# the numbers that show it rather than asserting a threshold.
#
# The `placeholders` and `injector calls` counters are the vacuity guard: a run
# reporting 0 placeholders never exercised recovery at all, however green it
# looks. That is exactly how the end-to-end recovery test passed on two
# backends for months (it injected at depth 3, and a well-adapted host sampler
# never builds trees that deep).
#
# Usage:
#   COMPILER=none   DEPTH=1 mix run --no-deps-check bench/crash_recovery.exs
#   COMPILER=vulkan DEPTH=3 mix run --no-deps-check bench/crash_recovery.exs
#   COMPILER=vulkan INJECT=0 mix run --no-deps-check bench/crash_recovery.exs   # control

alias Exmc.Builder
alias Exmc.Dist.Normal
alias Exmc.NUTS.{FaultInjector, Sampler}

env = fn name, default -> System.get_env(name) || default end

compiler = env.("COMPILER", "none") |> String.to_atom()
Application.put_env(:exmc, :compiler, compiler)

inject = env.("INJECT", "1") == "1"
depth = env.("DEPTH", "1") |> String.to_integer()
warmup = env.("WARMUP", "200") |> String.to_integer()
samples = env.("SAMPLES", "300") |> String.to_integer()
seed = env.("SEED", "42") |> String.to_integer()

# Force-load before trace_pattern or it silently matches nothing (NEXT.md §5).
Code.ensure_loaded!(Exmc.NUTS.FaultInjector)
Code.ensure_loaded!(Exmc.NUTS.Tree)

for {m, f, a} <- [
      {Exmc.NUTS.FaultInjector, :maybe_fault!, 1},
      {Exmc.NUTS.Tree, :divergent_placeholder, 4},
      {Exmc.NUTS.Tree, :with_supervision, 7},
      {Exmc.NUTS.Tree, :dispatch_subtree_precomputed, 14},
      {Exmc.NUTS.Tree, :dispatch_subtree, 11},
      {Exmc.NUTS.Tree, :build_subtree, 10}
    ] do
  :erlang.trace_pattern({m, f, a}, true, [:call_count])
end

count = fn m, f, a ->
  case :erlang.trace_info({m, f, a}, :call_count) do
    {:call_count, n} when is_integer(n) -> n
    _ -> :not_traced
  end
end

# Prior-only N(0,1): no observations, so nothing here touches the observed-model
# path (docs/OPEN_VULKAN_OBSERVED_MODEL.md). Analytic mean 0, variance 1.
ir = Builder.new_ir() |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

if inject, do: FaultInjector.activate(%{depth: depth, error: :crash})

{trace, stats} =
  Sampler.sample(ir, %{},
    num_warmup: warmup,
    num_samples: samples,
    seed: seed,
    supervised: true
  )

if inject, do: FaultInjector.deactivate()

vals = Nx.to_flat_list(trace["x"])
n = length(vals)
mean = Enum.sum(vals) / n
var = Enum.reduce(vals, 0.0, fn v, a -> a + (v - mean) * (v - mean) end) / n
distinct = vals |> MapSet.new() |> MapSet.size()

eps =
  case stats[:step_size] do
    nil -> :absent
    e when is_number(e) -> e
    t -> Nx.to_number(t)
  end

recoveries = Map.get(stats, :recoveries, :absent)
placeholders = count.(Exmc.NUTS.Tree, :divergent_placeholder, 4)

IO.puts("""

=== crash recovery — #{compiler}, #{if inject, do: "crash at depth #{depth}", else: "no injection"} ===
warmup/samples    : #{warmup}/#{samples}   seed #{seed}   supervised: true

  injector calls  : #{count.(Exmc.NUTS.FaultInjector, :maybe_fault!, 1)}
  placeholders    : #{placeholders}          <- 0 means recovery never ran
  with_supervision: #{count.(Exmc.NUTS.Tree, :with_supervision, 7)}
  spec dispatch   : #{count.(Exmc.NUTS.Tree, :dispatch_subtree_precomputed, 14)}
  plain dispatch  : #{count.(Exmc.NUTS.Tree, :dispatch_subtree, 11)}
  elixir subtree  : #{count.(Exmc.NUTS.Tree, :build_subtree, 10)}

  recoveries      : #{inspect(recoveries)}   (reported in stats)
  divergences     : #{inspect(stats.divergences)}
  adapted eps     : #{inspect(eps)}
  mean / var      : #{Float.round(mean, 4)} / #{var}    (truth 0.0 / 1.0)
  distinct draws  : #{distinct}/#{n}
""")

cond do
  inject and placeholders == 0 ->
    IO.puts("VACUOUS: injection was on and nothing was ever recovered.")

  inject and recoveries != placeholders ->
    IO.puts(
      "MISREPORTED: #{placeholders} placeholders built but stats.recoveries = " <>
        "#{inspect(recoveries)}. Warmup-phase recoveries used to vanish here."
    )

  true ->
    :ok
end
