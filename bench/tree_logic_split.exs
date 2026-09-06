# Where a sampling run's time actually goes: inside the GPU dispatch, or
# outside it in NUTS tree logic.
#
# WHY THIS EXISTS
#
# The working figure — ~1.2 ms GPU, ~1.0 ms in-NIF, ~1.9 ms tree logic per
# dispatch — came from subtracting an isolated benchmark median from a
# wall-clock average across a sampling run. Two different measurements on two
# different workloads, differenced. That is exactly how an upstream per-fence
# estimate came out 3x wrong, and it is worth not repeating on the term that
# now matters most.
#
# This measures the split directly: `Dispatch.dispatch_micros/0` accumulates
# time inside `chain/8`, so in-dispatch and out-of-dispatch come from one run
# rather than from two.
#
# DESIGN
#
#   * **Marginal, not total.** Each model runs at two sample counts and the
#     answer is the difference. Warmup, model compilation, shader synthesis and
#     BEAM startup are identical in both arms and cancel exactly, rather than
#     being estimated and subtracted. Same intercept/slope separation upstream
#     used to split fixed cost from per-step cost.
#
#   * **Several geometries.** The open question is whether out-of-dispatch cost
#     is per-DISPATCH or per-DRAW. A NUTS trajectory doubles until a U-turn, so
#     tree logic runs O(2^depth) per draw while dispatches run once per
#     leapfrog chain. Models with different tree depths give different
#     dispatches-per-draw ratios; whichever normalisation is stable across them
#     is the real denominator. Attributing a cost to the wrong denominator is
#     the error that made an earlier ~4500-dispatch figure wrong by 9x.
#
#   * **Run it on a quiet box.** super-io is a desktop and its GPU noise band
#     manufactured a 13x error upstream. mac-248 is headless and resolves this
#     path at 0.3%.
#
#   mix run bench/tree_logic_split.exs
#   BASE=200 TOP=800 REPS=3 mix run bench/tree_logic_split.exs

# EXMC_COMPILER is applied by config/runtime.exs in every environment. This
# file used to re-apply it because the switch was test-only and inert under
# `mix run`; that is no longer true.

alias Exmc.Builder
alias Exmc.Dist.Normal
alias Exmc.NUTS.Sampler
alias Exmc.NUTS.Vulkan.Dispatch

base = String.to_integer(System.get_env("BASE") || "200")
top = String.to_integer(System.get_env("TOP") || "800")
reps = String.to_integer(System.get_env("REPS") || "3")
warmup = String.to_integer(System.get_env("WARMUP") || "300")

compiler = Exmc.JIT.detect_compiler()

if compiler != Nx.Vulkan do
  IO.puts("""

  SKIPPED — this measures the split around the Vulkan chain dispatch and the
  active compiler is #{inspect(compiler)}. Re-run with EXMC_COMPILER=vulkan.
  """)

  System.halt(0)
end

# Independent conjugate pairs. Widening d deepens the trajectory, which changes
# dispatches-per-draw — the lever this benchmark needs.
defmodule Models do
  def conjugate(n) do
    1..n
    |> Enum.reduce(Exmc.Builder.new_ir(), fn i, acc ->
      acc
      |> Exmc.Builder.rv("mu#{i}", Exmc.Dist.Normal, %{
        mu: Nx.tensor(0.0),
        sigma: Nx.tensor(2.0)
      })
      |> Exmc.Builder.rv("y#{i}", Exmc.Dist.Normal, %{
        mu: "mu#{i}",
        sigma: Nx.tensor(1.0)
      })
      |> Exmc.Builder.obs("y#{i}_obs", "y#{i}", Nx.tensor(1.0 + i * 0.3))
    end)
  end
end

run = fn ir, n_samples ->
  Dispatch.reset_dispatch_count()

  {wall_us, {_trace, _stats}} =
    :timer.tc(fn ->
      Sampler.sample(ir, %{}, num_warmup: warmup, num_samples: n_samples, seed: 7)
    end)

  %{
    wall: wall_us,
    in_dispatch: Dispatch.dispatch_micros(),
    dispatches: Dispatch.dispatch_count(),
    draws: n_samples
  }
end

median = fn xs -> xs |> Enum.sort() |> Enum.at(div(length(xs), 2)) end

IO.puts("""
tree-logic split
  host       #{:inet.gethostname() |> elem(1)}
  compiler   #{inspect(compiler)}
  nx_vulkan  #{Application.spec(:nx_vulkan, :vsn)}
  marginal between #{base} and #{top} draws, warmup #{warmup}, #{reps} reps, median
""")

:io.format("  ~-6s ~10s ~10s ~12s ~12s ~12s ~10s~n", [
  "d",
  "disp/draw",
  "us/draw",
  "in-disp/draw",
  "out/draw",
  "out/dispatch",
  "out %"
])

for n_rv <- [1, 2, 4, 8] do
  ir = Models.conjugate(n_rv)

  # Warm the shader cache and the code paths before any timed arm.
  Sampler.sample(ir, %{}, num_warmup: 50, num_samples: 50, seed: 1)

  rows =
    for _ <- 1..reps do
      lo = run.(ir, base)
      hi = run.(ir, top)

      d_draws = hi.draws - lo.draws
      d_wall = hi.wall - lo.wall
      d_in = hi.in_dispatch - lo.in_dispatch
      d_disp = hi.dispatches - lo.dispatches

      %{
        disp_per_draw: d_disp / d_draws,
        us_per_draw: d_wall / d_draws,
        in_per_draw: d_in / d_draws,
        out_per_draw: (d_wall - d_in) / d_draws,
        out_per_disp: if(d_disp > 0, do: (d_wall - d_in) / d_disp, else: 0.0),
        out_pct: (d_wall - d_in) / max(d_wall, 1) * 100
      }
    end

  m = fn key -> median.(Enum.map(rows, & &1[key])) end

  :io.format("  ~-6B ~10.2f ~10.1f ~12.1f ~12.1f ~12.1f ~9.1f%~n", [
    n_rv,
    m.(:disp_per_draw),
    m.(:us_per_draw),
    m.(:in_per_draw),
    m.(:out_per_draw),
    m.(:out_per_disp),
    m.(:out_pct)
  ])
end

IO.puts("""

  Read the last two numeric columns against each other. If out-of-dispatch
  cost is per-DRAW it stays flat in `out/draw` while `out/dispatch` moves with
  the dispatch rate; if it is per-DISPATCH the reverse holds. Whichever is
  stable across d is the real denominator, and it decides whether the lever is
  fewer draws' worth of tree work or cheaper work per leapfrog chain.
""")
