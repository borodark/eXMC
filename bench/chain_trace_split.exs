# Where a NUTS draw's time actually goes — MEASURED, not subtracted.
#
# WHY THIS EXISTS
#
# `NEXT.md` carries a decomposition of one Jetson dispatch into ~29% GPU,
# ~25% CPU inside the NIF call and ~46% CPU outside it. It was obtained by
# subtracting a benchmark median from a wall-clock average. Differencing two
# noisy quantities amplifies the noise in both and gives the reader no way to
# tell a real split from an artefact; the same manoeuvre in this project's
# orbit produced physically impossible NEGATIVE values once, and was withdrawn
# twice. That same document says plainly that neither Kepler has ever had the
# two halves separated at all.
#
# This file separates them without a subtraction, on whatever host it is run
# on, using three instruments that are independent of one another:
#
#   wall_ms      a real `sample_compiled` run's wall clock.
#   in_chain_ms  `Dispatch.dispatch_micros/0` from that SAME run — a timer
#                around `Dispatch.chain/8`, so it is marshalling + submit +
#                fence + readback + GPU, measured directly.
#   host_ms      `ChainTrace.replay/3`'s wall clock. Every dispatch is served
#                from bytes recorded earlier, so the dispatch is free and the
#                replay's wall time IS the host-side half.
#
# None of the three is computed from the other two. That is the whole point,
# and it is what makes the closure below evidence rather than arithmetic:
#
#   CLOSURE A   host_ms + in_chain_ms  ~=  wall_ms
#
# If A fails there is a fourth term none of the three instruments contains,
# and no split derived from any of them should be quoted. A decomposition that
# cannot reconstruct the thing it decomposed has not been checked.
#
#   CLOSURE B   replay with :delay_us = in_chain_ms*1000/dispatches  ~=  wall_ms
#
# B is not redundant with A. A is a sum of two scalars; B puts the cost back
# where it was spent, one dispatch at a time, and so also tests that the cost
# is roughly uniform across dispatches rather than concentrated in a few. The
# delay comes from `dispatch_micros/0` — an instrument — never from
# `wall - host`, which would make the reconstruction circular and guaranteed.
#
# WHY REPLAY NEEDS THE RECORDED BYTES
#
# NUTS grows a trajectory until a U-turn, and the U-turn test reads the
# positions and momenta the dispatch returned. A stub handing back zeros, or
# plausible noise, sends the sampler down a different tree to a different
# depth — you would be timing a workload that never runs, and it would look
# like a clean result. Replaying the recorded bytes in order makes the
# trajectory bit-identical; the only thing that changed is that the dispatch
# is free. `ChainTrace` checks `k` and the q size at every position and RAISES
# on a mismatch, so a diverged replay aborts instead of reporting a fast,
# confident, meaningless number.
#
# SINGLE CHAIN, ONE PROCESS — NOT A PREFERENCE
#
# The record/replay state and the dispatch counters both live in the PROCESS
# DICTIONARY. `Sampler.sample_chains` fans out through `Task.async_stream` and
# `Tree.with_supervision` uses `Task.async` under `supervised: :task`; neither
# inherits it. A recording taken across either boundary under-captures without
# saying so, and an under-captured trace looks exactly like a fast host half.
# So: `sample_compiled` (never `sample_chains`), and `supervised` left unset.
#
#   mix run bench/chain_trace_split.exs
#   MODEL=hier DRAWS=400 REPLICATES=5 mix run bench/chain_trace_split.exs
#
# Vulkan only, and it aborts rather than reporting a number for a path that
# did not run.

alias Exmc.Builder
alias Exmc.Dist.{HalfNormal, Normal}
alias Exmc.NUTS.Sampler
alias Exmc.NUTS.Vulkan.Dispatch
alias Nx.Vulkan.ChainTrace

draws = String.to_integer(System.get_env("DRAWS") || "300")
warmup_draws = String.to_integer(System.get_env("WARMUP_DRAWS") || "100")
replicates = String.to_integer(System.get_env("REPLICATES") || "5")
seed = String.to_integer(System.get_env("SEED") || "42")
model = System.get_env("MODEL") || "simple"
n_obs = String.to_integer(System.get_env("N_OBS") || "64")

halt = fn msg ->
  IO.puts("\n" <> msg)
  System.halt(1)
end

# ---------------------------------------------------------------------------
# Guard: the resolved compiler, not the requested one.

compiler = Exmc.JIT.detect_compiler()

if compiler != Nx.Vulkan do
  IO.puts("""

  SKIPPED — this benchmark decomposes a Vulkan chain-dispatch sampling run and
  the active compiler is #{inspect(compiler)}.

  Re-run with EXMC_COMPILER=vulkan (config/runtime.exs honours it in every
  MIX_ENV). A split reported from another backend would be measuring a path
  this file is not about.
  """)

  System.halt(0)
end

unless Code.ensure_loaded?(ChainTrace) and function_exported?(ChainTrace, :record, 1) do
  halt.("""
  ABORTED — Nx.Vulkan.ChainTrace is not available in this nx_vulkan build.

  The host half cannot be measured without record/replay, and this file will
  not fall back to subtracting one number from another. Bump mix.lock to an
  nx_vulkan that carries ChainTrace and rebuild the NIF.
  """)
end

# ---------------------------------------------------------------------------
# Model.

ir =
  case model do
    "simple" ->
      # The same model as bench/chain_dispatch_cost.exs, so the per-dispatch
      # cost this file measures in-band is comparable with the isolated
      # figure that file reports on the same host.
      Builder.new_ir()
      |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(5.0)})
      |> Builder.rv("x", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
      |> Builder.obs("x_obs", "x", Nx.tensor(3.0))

    "hier" ->
      y =
        Nx.tensor(Enum.map(1..n_obs, fn i -> 1.0 + 0.5 * :math.sin(i / 3.0) end), type: :f64)

      Builder.new_ir()
      |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(5.0)})
      |> Builder.rv("sigma", HalfNormal, %{sigma: Nx.tensor(2.0)})
      |> Builder.rv("y", Normal, %{mu: "mu", sigma: "sigma"})
      |> Builder.obs("y_obs", "y", y)

    other ->
      halt.("ABORTED — unknown MODEL=#{other}. Use \"simple\" or \"hier\".")
  end

# Non-vacuity, before anything is timed: if there is no synthesised chain
# shader for this model on this host there is nothing to decompose, and a
# per-op number reported under this file's name would be worse than none.
meta =
  case Exmc.NUTS.ChainShaderCodegen.detect_meta(ir, []) do
    {:ok, m} ->
      m

    other ->
      halt.("""
      ABORTED — detect_meta/2 returned #{inspect(other)} for MODEL=#{model}.

      This model does not synthesise to a fused chain shader here, so the run
      would spend its time on the per-op path and the split would describe a
      workload this file does not name.
      """)
  end

sample_opts = [
  seed: seed,
  num_samples: draws,
  num_warmup: warmup_draws
]

compiled = Sampler.compile(ir, [])

run = fn -> Sampler.sample_compiled(compiled, %{}, sample_opts) end

IO.puts("""
chain-trace split — the host half, measured
  host        #{:inet.gethostname() |> elem(1)}
  compiler    #{inspect(compiler)}
  nx_vulkan   #{Application.spec(:nx_vulkan, :vsn)}
  model       #{model}#{if model == "hier", do: "  n_obs=#{n_obs}", else: ""}
  meta        #{meta |> elem(0) |> inspect()} / sha #{meta |> elem(1) |> to_string() |> String.slice(0, 12)}
  draws       #{draws} (+#{warmup_draws} warmup)   seed=#{seed}   replicates=#{replicates}
""")

# ---------------------------------------------------------------------------
# Phase 1 — warm the pipeline, discarded.
#
# The first run pays SPIR-V synthesis, shader-module creation, pipeline
# creation and first-touch allocation. Those are real costs but they are not
# per-draw costs, and folding them in is how a fixed cost gets amortised into
# a variable one and a benchmark tells you a comfortable lie.

IO.write("  warming up (1 discarded sampling run) ... ")
run.()
IO.puts("done")

# ---------------------------------------------------------------------------
# Phase 2 — wall_ms and in_chain_ms, from the SAME runs.

measure_real = fn ->
  Dispatch.reset_dispatch_count()
  {us, _} = :timer.tc(run)
  {us / 1000, Dispatch.dispatch_count(), Dispatch.dispatch_micros() / 1000}
end

real = Enum.map(1..replicates, fn r ->
  {wall, n, in_chain} = measure_real.()

  if n == 0 do
    halt.("""
    ABORTED — replicate #{r} made ZERO chain dispatches.

    The run went down the per-op or host path, so there is no dispatch cost to
    separate and the "host half" would be the whole thing. This is the check
    that stops a green-looking 100%/0% split from being reported.
    """)
  end

  IO.puts(
    "  real #{r}:  #{Float.round(wall, 1)} ms wall   " <>
      "#{Float.round(in_chain, 1)} ms in chain/8   #{n} dispatches   " <>
      "#{Float.round(in_chain * 1000 / n, 1)} us/dispatch"
  )

  {wall, n, in_chain}
end)

median_by = fn list, f ->
  sorted = list |> Enum.map(f) |> Enum.sort()
  Enum.at(sorted, div(length(sorted), 2))
end

wall_ms = median_by.(real, &elem(&1, 0))
in_chain_ms = median_by.(real, &elem(&1, 2))
dispatch_counts = real |> Enum.map(&elem(&1, 1)) |> Enum.uniq()

# A fixed seed must give a fixed trajectory. If it does not, the replay below
# cannot be trusted either, and neither can any paired comparison built on
# this file.
if length(dispatch_counts) != 1 do
  halt.("""
  ABORTED — the dispatch count varied across replicates: #{inspect(dispatch_counts)}.

  Same seed, same options, same compiled model, different trajectories. The
  run is not deterministic, so replay cannot reproduce it and the split would
  be comparing two different workloads.
  """)
end

[n_dispatch] = dispatch_counts

# ---------------------------------------------------------------------------
# Phase 3 — record.
#
# Recording accumulates the returned bytes, so THIS run's wall clock is not
# the clean one and is deliberately not used above.

IO.write("\n  recording ... ")
Dispatch.reset_dispatch_count()
{_result, trace} = ChainTrace.record(run)
recorded_dispatches = Dispatch.dispatch_count()
info = ChainTrace.info(trace)
IO.puts("#{info.dispatches} dispatches, #{Float.round(info.bytes / 1_048_576, 2)} MiB")

if info.dispatches != recorded_dispatches or info.dispatches != n_dispatch do
  halt.("""
  ABORTED — the trace holds #{info.dispatches} dispatches; Dispatch counted
  #{recorded_dispatches} in the recording run and #{n_dispatch} in the timed runs.

  A trace shorter than the run under-captures, which looks EXACTLY like a fast
  host half — the failure this check exists for. The usual cause is a process
  boundary: record/replay state lives in the process dictionary and does not
  cross Task.async_stream or Task.async.
  """)
end

# ---------------------------------------------------------------------------
# Phase 3b — the control arm, and why Closure A needs one.
#
# `host_ms` below is a real run's wall clock with the dispatch removed, but it
# is also a run taken LATER, on a hotter CPU, in a process now holding the
# whole trace live on its heap. Both of those cost time that has nothing to do
# with the host half, and both inflate `host_ms` in the same direction — so a
# Closure A that OVERSHOOTS is exactly what they would produce, and cannot be
# told apart from a genuine fourth term without measuring them.
#
# So: repeat Phase 2 verbatim, here, with the trace held. The difference from
# the clean wall is what the position in the run and the live trace cost a
# workload whose host half is already known. Subtracting it from `host_ms` is
# not a fudge — it is the same measurement under the same conditions, which is
# the only kind of subtraction this file allows.
#
# It does NOT separate the two mechanisms from each other. A multi-MiB live
# binary set that every minor GC must scan, and a CPU that has dropped off
# turbo after a minute of sustained load, are both in this number, and telling
# them apart needs an instrument this file does not have.

held =
  Enum.map(1..replicates, fn r ->
    {wall, n, in_chain} = measure_real.()

    if n != n_dispatch do
      halt.(
        "ABORTED — control run #{r} made #{n} dispatches, the clean runs made #{n_dispatch}."
      )
    end

    IO.puts(
      "  held #{r}:  #{Float.round(wall, 1)} ms wall   " <>
        "#{Float.round(in_chain, 1)} ms in chain/8"
    )

    {wall, in_chain}
  end)

wall_held_ms = median_by.(held, &elem(&1, 0))
in_chain_held_ms = median_by.(held, &elem(&1, 1))

# MEASURED, and it is why this is computed on the host portion and not on the
# wall: holding the trace slows THE DISPATCH DOWN TOO. On mac-248, in-chain
# went 509.6 -> 637.2 ms between the clean and held runs, 25% on the NIF call
# itself, presumably allocator pressure from a multi-MiB live refc-binary set.
#
# So about a third of the control's wall delta lands inside chain/8, where
# `in_chain_ms` already accounts for it. Subtracting the whole wall delta from
# `host_ms` double-counts that third and drives Closure A to -11%, which is
# how this was found: the correction overshot by more than the raw error it
# was correcting. Take the host portion of each control run instead.
host_tax_ms = wall_held_ms - in_chain_held_ms - (wall_ms - in_chain_ms)

# ---------------------------------------------------------------------------
# Phase 4 — host_ms, by replay.

replay_samples =
  Enum.map(1..replicates, fn r ->
    Dispatch.reset_dispatch_count()
    {_res, stats} = ChainTrace.replay(trace, run)

    if stats.dispatches != n_dispatch do
      halt.("""
      ABORTED — replay #{r} made #{stats.dispatches} dispatches, the real run made
      #{n_dispatch}. Control flow diverged and the elapsed time means nothing.
      """)
    end

    IO.puts(
      "  replay #{r}: #{Float.round(stats.elapsed_ms, 1)} ms host   " <>
        "(replay overhead #{Float.round(stats.replay_overhead_ms, 2)} ms, " <>
        "in-band #{Float.round(Dispatch.dispatch_micros() / 1000, 2)} ms)"
    )

    stats.elapsed_ms
  end)

host_ms = Enum.at(Enum.sort(replay_samples), div(length(replay_samples), 2))

# ---------------------------------------------------------------------------
# Phase 5 — CLOSURE B: put the cost back where it was spent.

delay_us = in_chain_ms * 1000 / n_dispatch

recon_samples =
  Enum.map(1..replicates, fn _r ->
    {_res, stats} = ChainTrace.replay(trace, [delay_us: delay_us], run)
    stats.elapsed_ms
  end)

recon_ms = Enum.at(Enum.sort(recon_samples), div(length(recon_samples), 2))

# The two replay arms differ in ONE thing: the delay. So their difference is
# the delay the busy-wait actually delivered, and comparing it with the delay
# that was asked for splits any Closure B error into two named parts instead
# of one unexplained one. Without this, "B is 11% low" is indistinguishable
# between "the spin loop undershoots" and "there is a fourth term in the wall
# clock" — and those call for completely different next steps.
achieved_us = (recon_ms - host_ms) * 1000 / n_dispatch

fidelity =
  if delay_us > 0, do: Float.round((achieved_us - delay_us) / delay_us * 100, 1), else: 0.0

# ---------------------------------------------------------------------------
# Report.

pct = fn part -> Float.round(part / wall_ms * 100, 1) end
err = fn got -> Float.round((got - wall_ms) / wall_ms * 100, 1) end
spread = fn list ->
  s = Enum.sort(list)
  m = Enum.at(s, div(length(s), 2))
  if m > 0, do: Float.round((List.last(s) - List.first(s)) / m * 100, 1), else: 0.0
end

IO.puts("""

  MEASURED (medians of #{replicates})

    wall            #{Float.round(wall_ms, 1)} ms          spread #{spread.(Enum.map(real, &elem(&1, 0)))}%
    in chain/8      #{Float.round(in_chain_ms, 1)} ms   #{pct.(in_chain_ms)}%   spread #{spread.(Enum.map(real, &elem(&1, 2)))}%
    host (replay)   #{Float.round(host_ms, 1)} ms   #{pct.(host_ms)}%   spread #{spread.(replay_samples)}%
    host corrected  #{Float.round(host_ms - host_tax_ms, 1)} ms   #{pct.(host_ms - host_tax_ms)}%

    #{n_dispatch} dispatches   #{Float.round(delay_us, 1)} us per dispatch inside chain/8

  CONTROL — what do the position in the run, and the live trace, cost?

    wall      #{Float.round(wall_ms, 1)} -> #{Float.round(wall_held_ms, 1)} ms
    in chain  #{Float.round(in_chain_ms, 1)} -> #{Float.round(in_chain_held_ms, 1)} ms   \
(#{Float.round((in_chain_held_ms - in_chain_ms) / in_chain_ms * 100, 1)}% — the NIF call itself)
    host      #{Float.round(wall_ms - in_chain_ms, 1)} -> \
#{Float.round(wall_held_ms - in_chain_held_ms, 1)} ms   \
= #{Float.round(host_tax_ms, 1)} ms  (#{pct.(host_tax_ms)}%)

    Same run, same dispatch count, taken after the recording with the trace
    live. Only the host line is subtracted from `host`: the in-chain line is
    already inside `in_chain_ms`, and taking the wall delta instead would
    count it twice. Read the in-chain line as a result in its own right — a
    large live binary set makes the dispatch measurably slower, which is a
    fact about the instrument that anyone recording a long run should know.

  CLOSURE A — do the two independent halves add up to the wall clock?

    raw        host + in-chain = #{Float.round(host_ms + in_chain_ms, 1)} ms   \
vs wall #{Float.round(wall_ms, 1)} ms   error #{err.(host_ms + in_chain_ms)}%
    corrected  (host - control) + in-chain = #{Float.round(host_ms - host_tax_ms + in_chain_ms, 1)} ms   \
vs wall #{Float.round(wall_ms, 1)} ms   error #{err.(host_ms - host_tax_ms + in_chain_ms)}%

  CLOSURE B — does replaying with the cost put back reconstruct the run?

    replay + #{Float.round(delay_us, 1)} us/dispatch = #{Float.round(recon_ms, 1)} ms   \
vs wall #{Float.round(wall_ms, 1)} ms   error #{err.(recon_ms)}%   spread #{spread.(recon_samples)}%

    of which — delay asked for #{Float.round(delay_us, 1)} us/dispatch, \
delivered #{Float.round(achieved_us, 1)} us  (#{fidelity}%)

    The two replay arms differ only in the delay, so the second line is the
    busy-wait's own fidelity, measured. A large B error with a small fidelity
    error is a real fourth term; a large B error that the fidelity line fully
    accounts for is the spin loop, and says nothing about the split.

  HOW TO READ THIS

  Read the closures BEFORE the split, and read CORRECTED before RAW. A raw
  Closure A that overshoots by about the control IS the control, and the
  corrected line is the answer. A corrected Closure A still outside the wall
  spread means there is a term none of the four instruments contains, and no
  split should be quoted until that term is named.

  Closure B has no corrected line on purpose: the control measures a whole
  run and there is no principled way to attribute it per dispatch. Read B for
  the fidelity line, and for whether the cost is uniform across dispatches —
  the sign and rough size of its error should track A's RAW line.

  A large `in chain/8` share is NOT a GPU-bound run. That number is
  marshalling, allocation, submit, fence and readback as well as GPU compute,
  and on this project's measurements the CPU inside the call has been the same
  order as the compute. Separating THAT pair needs an instrument inside the
  NIF, which this file does not have and does not pretend to.
""")
