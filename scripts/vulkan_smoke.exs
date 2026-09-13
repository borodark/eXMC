# The Vulkan arm as a CONSUMER reaches it: `mix run`, not `mix test`.
#
#   EXMC_COMPILER=vulkan MIX_ENV=test mix run --no-compile scripts/vulkan_smoke.exs
#
# WHY THIS EXISTS. On 2026-09-13 the whole fleet was green under `mix test`
# while `mix run` on every FreeBSD host could not sample on the GPU at all:
# exmc did not declare `:crypto`, CustomSynth hashes every shader with it, and
# the test environment loads crypto regardless (xla declares it on Linux, which
# hid it there too). A release, a consumer project and every bench take the
# `mix run` path. The suite measured a path users do not take. Fixed in
# 7aae323a6; this is the gate that would have caught it.
#
# `MIX_ENV=test` only so it reuses the build fleet_verify.sh just made. What
# differs from `mix test` is what matters: no ExUnit, no test-only deps, only
# the applications exmc itself declares on the code path.
#
# Two models, both through the fused chain shader, each checked three ways:
# the compile produced a {:synthesised, ...} meta (Plan-B' raises otherwise);
# sampling made chain dispatches (a count, not a timing -- counts are exact and
# host-independent); and the draws are finite and loosely right. The moment
# bounds are deliberately wide: this is a gate on "the path runs", and
# statistical correctness is the suite's and bench/nuts_truth.exs's job.
#
# Exit 0 and a `SMOKE ok` line per model, or exit 1 naming what failed.

alias Exmc.{Builder, Compiler, Dist, IR}
alias Exmc.NUTS.Sampler
alias Exmc.NUTS.Vulkan.Dispatch

Application.put_env(:exmc, :compiler, :vulkan)

f64 = [type: :f64]

normal_ir =
  Builder.new_ir()
  |> Builder.rv("x", Dist.Normal, %{mu: Nx.tensor(0.0, f64), sigma: Nx.tensor(1.0, f64)})

# The regression idiom: a vector RV, a Custom likelihood capturing its data, and
# Nx.dot over a captured design matrix -- captures, the extras buffer and the
# obs loop, which the single Normal does not touch. y = 1 + 2x exactly, so the
# posterior mean of beta sits near {1, 2} with a tiny sd.
xs = Enum.map(1..40, &(&1 / 4))
xmat = Nx.stack([Nx.broadcast(Nx.tensor(1.0, f64), {40}), Nx.tensor(xs, f64)], axis: 1)
y = Nx.tensor(Enum.map(xs, fn v -> 1.0 + 2.0 * v end), f64)
sigma = 0.3

lik =
  Dist.Custom.new(fn _x, p ->
    r = Nx.subtract(y, Nx.dot(xmat, p.beta))
    Nx.sum(Nx.divide(Nx.multiply(r, r), -2 * sigma * sigma))
  end)

regression_ir =
  IR.new()
  |> Builder.rv("beta", Dist.Normal, %{mu: Nx.tensor(0.0, f64), sigma: Nx.tensor(5.0, f64)},
    shape: {2}
  )
  |> Dist.Custom.rv("Y", lik, %{beta: "beta"})
  |> Builder.obs("Y_obs", "Y", y)

check = fn label, ir, var, ok? ->
  t0 = System.monotonic_time(:millisecond)

  result =
    try do
      compiled = Compiler.compile_for_sampling(ir)

      case elem(compiled, 5) do
        {:synthesised, _, _, _, _, _, _} -> :ok
        other -> throw({:not_synthesised, other})
      end

      Dispatch.reset_dispatch_count()

      {trace, _stats} =
        Sampler.sample_compiled(compiled, %{}, num_warmup: 100, num_samples: 200, seed: 7)

      dispatches = Dispatch.dispatch_count()
      if dispatches == 0, do: throw(:zero_chain_dispatches)

      draws = trace |> Map.fetch!(var) |> Nx.to_flat_list()

      unless Enum.all?(draws, &is_float/1) and Enum.all?(draws, &(abs(&1) < 1.0e6)),
        do: throw(:non_finite_draws)

      case ok?.(trace) do
        :ok -> {:ok, dispatches}
        {:error, why} -> throw({:implausible, why})
      end
    rescue
      e -> {:error, Exception.format(:error, e, __STACKTRACE__) |> String.slice(0, 1500)}
    catch
      kind, why -> {:error, inspect({kind, why})}
    end

  ms = System.monotonic_time(:millisecond) - t0

  case result do
    {:ok, n} ->
      IO.puts("SMOKE ok    #{label}  chain_dispatches=#{n}  #{ms} ms")
      true

    {:error, why} ->
      IO.puts("SMOKE FAIL  #{label}  #{ms} ms\n#{why}")
      false
  end
end

mean = fn xs -> Enum.sum(xs) / length(xs) end

results = [
  check.("normal(0,1)", normal_ir, "x", fn trace ->
    m = trace |> Map.fetch!("x") |> Nx.to_flat_list() |> mean.()
    if abs(m) < 0.75, do: :ok, else: {:error, "mean #{m}"}
  end),
  check.("regression d=2 (captures, Nx.dot)", regression_ir, "beta", fn trace ->
    [b0, b1] =
      trace |> Map.fetch!("beta") |> Nx.mean(axes: [0]) |> Nx.to_flat_list()

    if abs(b0 - 1.0) < 0.5 and abs(b1 - 2.0) < 0.2,
      do: :ok,
      else: {:error, "beta mean #{inspect([b0, b1])}"}
  end)
]

IO.puts("exmc: #{Exmc.JIT.describe()}")
System.halt(if Enum.all?(results), do: 0, else: 1)
