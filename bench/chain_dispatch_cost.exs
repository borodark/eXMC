# Chain-dispatch microbenchmark — per-dispatch cost of the f64 fused chain NIF.
#
# WHY THIS EXISTS
#
# The instrument it replaces was `mix test` on two small test files: ~520 chain
# dispatches inside a ~4 s wall-clock measurement whose bulk is BEAM boot, model
# compilation, GLSL synthesis, SPIR-V cache lookup and backend init. A paired
# delta from it is meaningful (fixed cost subtracts out), but the per-dispatch
# figures derived from it were not, and two of them reached a collaborating
# project and were used to size a change there. Both were withdrawn.
#
# This drives `Dispatch.chain/8` directly, in a loop, with startup outside the
# timed region. It resolves at the scale the effect lives at — tens of
# microseconds per dispatch — instead of drowning it.
#
# DESIGN, and why each part is not optional:
#
#   * Warmup discarded, and the default is large ON PURPOSE. The first
#     dispatches pay shader-module creation and first-touch allocation. They
#     are real costs but they are not per-dispatch costs, and including them
#     is how a fixed cost gets amortised into a variable one.
#
#     MEASURED, on an RTX 3060 Ti: with a 300-dispatch warmup the per-dispatch
#     cost rose 625 -> 902 -> 969 us over the first three replicates and only
#     then settled around 880. It looked exactly like a leak. It is not — it
#     is warmup, and 300 was a number picked out of the air. Raising it to
#     6000 took the spread from 38.7% of the median to 11.5%. If you shorten
#     WARMUP, expect the first replicate to read fast and the benchmark to
#     tell you a comfortable lie.
#   * Many dispatches per timed sample. A ~0.1 ms effect inside a ~0.5 ms
#     measurement does not survive; a collaborating session discarded a whole
#     run to this, having measured 34% process-to-process variance on an
#     IDENTICAL binary.
#   * Replicates, and the median rather than the mean. One slow sample from a
#     DVFS excursion should move the answer by nothing.
#   * The GPU clock is NOT observable on the FreeBSD hosts — `nvidia-smi`
#     reports [N/A] for clocks.sm — so this prints the spread and leaves the
#     reader to judge, rather than pretending to control for it.
#
# It deliberately does NOT compare two nx_vulkan revisions itself. Rebuilding
# the NIF mid-run would put a Rust compile between the arms, which is exactly
# the confound this is meant to remove. Run it, rebuild, run it again, and
# compare — with nothing else touching the box either time.
#
#   mix run bench/chain_dispatch_cost.exs
#   N=20000 REPLICATES=7 mix run bench/chain_dispatch_cost.exs
#
# Vulkan only. On any other compiler it says so and exits rather than
# reporting a number for a path that did not run.

alias Exmc.Builder
alias Exmc.Dist.Normal
alias Exmc.NUTS.Vulkan.Dispatch

n_dispatch = String.to_integer(System.get_env("N") || "5000")
replicates = String.to_integer(System.get_env("REPLICATES") || "5")
warmup = String.to_integer(System.get_env("WARMUP") || "6000")
k = String.to_integer(System.get_env("K") || "32")

# EXMC_COMPILER is honoured by `config/test.exs` ONLY, i.e. under MIX_ENV=test.
# `mix run` is MIX_ENV=dev, where the variable is inert — so this file has to
# apply it itself or its own "re-run with EXMC_COMPILER=vulkan" advice is a lie.
#
# It read as true on two hosts for unrelated reasons: super-io's EXLA is
# unusable without LD_LIBRARY_PATH and mac-248 has no EXLA at all, so
# auto_detect/0 fell through to Nx.Vulkan on both. On the Jetson, where EXLA
# genuinely works, the same command selected EXLA and the guard below caught it.
case System.get_env("EXMC_COMPILER") do
  "vulkan" -> Application.put_env(:exmc, :compiler, :vulkan)
  "exla" -> Application.put_env(:exmc, :compiler, :exla)
  "none" -> Application.put_env(:exmc, :compiler, :none)
  _ -> :ok
end

compiler = Exmc.JIT.detect_compiler()

if compiler != Nx.Vulkan do
  IO.puts("""

  SKIPPED — this benchmark measures the Vulkan f64 chain NIF and the active
  compiler is #{inspect(compiler)}.

  Re-run with EXMC_COMPILER=vulkan (honoured directly by this file, in any
  MIX_ENV). Reporting a number from another backend would be measuring a path
  this file is not about.
  """)

  System.halt(0)
end

ir =
  Builder.new_ir()
  |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(5.0)})
  |> Builder.rv("x", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
  |> Builder.obs("x_obs", "x", Nx.tensor(3.0))

meta =
  case Exmc.NUTS.ChainShaderCodegen.detect_meta(ir, []) do
    {:ok, m} ->
      m

    other ->
      IO.puts("""

      ABORTED — detect_meta/2 returned #{inspect(other)}.

      There is no synthesised chain shader for this model on this host, so
      there is nothing to time. This is a hard stop rather than a fallback:
      a per-op number reported under this file's name would be worse than
      no number.
      """)

      System.halt(1)
  end

d = 1
epsilon = 0.05
inv_mass = Nx.tensor([1.0], type: :f64)
q = Nx.tensor([0.3], type: :f64)
p = Nx.tensor([-0.7], type: :f64)

# GPU clock, where the platform will tell us. On Linux/NVIDIA nvidia-smi
# reports it; on the FreeBSD Keplers clocks.sm comes back [N/A], so this
# returns nil there and the report says "unavailable" rather than silently
# omitting the row. An unrecorded clock swung a collaborating session's
# measurement by 2.6x, which is larger than any effect either of us is
# chasing.
gpu_clock = fn ->
  try do
    args = ["--query-gpu=clocks.sm,temperature.gpu", "--format=csv,noheader,nounits"]

    case System.cmd("nvidia-smi", args, stderr_to_stdout: true) do
      {out, 0} ->
        case out |> String.trim() |> String.split(",") |> Enum.map(&String.trim/1) do
          [mhz, temp] ->
            case Integer.parse(mhz) do
              {m, _} -> {m, temp}
              _ -> nil
            end

          _ ->
            nil
        end

      _ ->
        nil
    end
  rescue
    _ -> nil
  catch
    _, _ -> nil
  end
end

drive = fn count ->
  Enum.each(1..count, fn i ->
    # Alternate direction so the run is not one degenerate trajectory the
    # driver could special-case.
    dir = if rem(i, 2) == 0, do: 1, else: -1
    Dispatch.chain(meta, d, epsilon, inv_mass, q, p, k, dir)
  end)
end

IO.puts("""
chain-dispatch cost
  host        #{:inet.gethostname() |> elem(1)}
  compiler    #{inspect(compiler)}
  nx_vulkan   #{Application.spec(:nx_vulkan, :vsn)}
  meta        #{meta |> elem(0) |> inspect()} / sha #{meta |> elem(1) |> to_string() |> String.slice(0, 12)}
  K=#{k}  d=#{d}  N=#{n_dispatch}/sample  warmup=#{warmup}  replicates=#{replicates}
""")

IO.write("  warming up (#{warmup} dispatches, discarded) ... ")
drive.(warmup)
IO.puts("done")

samples =
  Enum.map(1..replicates, fn r ->
    Dispatch.reset_dispatch_count()
    {us, _} = :timer.tc(fn -> drive.(n_dispatch) end)
    counted = Dispatch.dispatch_count()

    # Non-vacuity. If the counter disagrees with the loop, the dispatches did
    # not happen and every number below is meaningless — say so, do not divide.
    if counted != n_dispatch do
      IO.puts("""

      ABORTED — asked for #{n_dispatch} dispatches, the counter saw #{counted}.
      Something short-circuited the dispatch path; the timing is not
      measuring what this file claims.
      """)

      System.halt(1)
    end

    per = us / n_dispatch

    clk =
      case gpu_clock.() do
        {mhz, temp} -> "   sm #{mhz} MHz  #{temp}C"
        nil -> "   clock unavailable"
      end

    IO.puts(
      "  replicate #{r}:  #{Float.round(us / 1000, 1)} ms total   " <>
        "#{Float.round(per, 1)} us/dispatch#{clk}"
    )

    per
  end)

sorted = Enum.sort(samples)
median = Enum.at(sorted, div(length(sorted), 2))
lo = List.first(sorted)
hi = List.last(sorted)
spread = if median > 0, do: (hi - lo) / median * 100, else: 0.0

IO.puts("""

  median      #{Float.round(median, 1)} us/dispatch
  range       #{Float.round(lo, 1)} .. #{Float.round(hi, 1)} us   (#{Float.round(spread, 1)}% of median)

  Read the spread before the median, and read the clock column before both.
  A rising per-dispatch cost with a falling sm clock is thermal or power
  throttling, not the code — measured here on an RTX 3060 Ti, which settles
  around 1755 MHz against a 2100 MHz maximum under sustained dispatch.

  Where the clock says "unavailable" (the FreeBSD Keplers), a wide spread
  means the number above is not reproducible and no comparison should be
  built on it. Compare two revisions only when both spreads are tight and
  nothing else was running on the box either time.
""")
