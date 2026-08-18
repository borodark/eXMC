# Re-runs the evidence table in docs/OPEN_VULKAN_OBSERVED_MODEL.md.
#
# The defect: `compose_logp_defn/1` handed EVERY observed node the WHOLE
# observation buffer, so a model with N separate `Builder.obs` nodes counted
# the likelihood N times over. Fixed in 6c1589a via per-marker {offset, count}
# spans. That fix was confirmed on ONE seed of ONE model; this sweeps the
# variants the defect was documented across, on both arms, on several seeds.
#
# Every row is measured against the CLOSED FORM, not against the other arm —
# a differential is blind to anything both arms share.
#
# Usage:
#   COMPILER=none   SEEDS=42,1,2,3 mix run --no-deps-check bench/observed_model_evidence.exs
#   COMPILER=vulkan SEEDS=42,1,2,3 mix run --no-deps-check bench/observed_model_evidence.exs

alias Exmc.Builder
alias Exmc.Dist.Normal
alias Exmc.NUTS.Sampler

env = fn name, default -> System.get_env(name) || default end

warmup = env.("WARMUP", "300") |> String.to_integer()
samples = env.("SAMPLES", "500") |> String.to_integer()

seeds =
  env.("SEEDS", "42,1,2,3")
  |> String.split(",", trim: true)
  |> Enum.map(&(&1 |> String.trim() |> String.to_integer()))

compiler = env.("COMPILER", "none") |> String.to_atom()
Application.put_env(:exmc, :compiler, compiler)

only = env.("ONLY", "") |> String.split(";", trim: true) |> Enum.map(&String.trim/1)

# Conjugate normal-normal posterior for `mu`:
#   mu ~ N(m0, s0), x_i ~ N(mu, sigma_i)  =>  precision = 1/s0^2 + sum 1/sigma_i^2
truth = fn m0, s0, obs ->
  prec = 1.0 / (s0 * s0) + Enum.reduce(obs, 0.0, fn {_x, s}, a -> a + 1.0 / (s * s) end)
  num = m0 / (s0 * s0) + Enum.reduce(obs, 0.0, fn {x, s}, a -> a + x / (s * s) end)
  {num / prec, :math.sqrt(1.0 / prec)}
end

# One `Builder.rv` + `Builder.obs` pair per observation — the defective shape.
scalar_ir = fn m0, s0, obs ->
  obs
  |> Enum.with_index(1)
  |> Enum.reduce(
    Builder.new_ir()
    |> Builder.rv("mu", Normal, %{mu: Nx.tensor(m0), sigma: Nx.tensor(s0)}),
    fn {{x, s}, i}, ir ->
      name = "x#{i}"

      ir
      |> Builder.rv(name, Normal, %{mu: "mu", sigma: Nx.tensor(s)})
      |> Builder.obs(name <> "_obs", name, Nx.tensor(x))
    end
  )
end

# One observed node owning the whole buffer — the shape that was always correct.
vector_ir = fn m0, s0, obs ->
  {xs, [sigma | _]} = Enum.unzip(obs)

  Builder.new_ir()
  |> Builder.rv("mu", Normal, %{mu: Nx.tensor(m0), sigma: Nx.tensor(s0)})
  |> Builder.rv("x", Normal, %{mu: "mu", sigma: Nx.tensor(sigma)})
  |> Builder.obs("x_obs", "x", Nx.tensor(xs))
end

variants = [
  # {label, ir, m0, s0, [{obs_value, sigma}]}
  {"scalar 3 obs", scalar_ir, 0.0, 10.0, [{4.0, 1.0}, {3.8, 1.0}, {4.2, 1.0}]},
  {"vector 3 obs", vector_ir, 0.0, 10.0, [{4.0, 1.0}, {3.8, 1.0}, {4.2, 1.0}]},
  # distinct sigmas: any permutation of the marker->slice assignment changes
  # the answer, so this arm can see the mirrored-gradient trap the equal-sigma
  # model is bit-identical under.
  {"scalar 3 obs, sigmas 1/2/3", scalar_ir, 0.0, 10.0, [{4.0, 1.0}, {3.8, 2.0}, {4.2, 3.0}]},
  {"scalar 1 obs", scalar_ir, 0.0, 10.0, [{4.0, 1.0}]},
  {"scalar 2 obs", scalar_ir, 0.0, 10.0, [{4.0, 1.0}, {5.0, 1.0}]},
  {"scalar 5 obs", scalar_ir, 0.0, 5.0,
   [{1.0, 1.0}, {2.0, 1.0}, {3.0, 1.0}, {4.0, 1.0}, {5.0, 1.0}]},
  {"vector 5 obs", vector_ir, 0.0, 5.0,
   [{1.0, 1.0}, {2.0, 1.0}, {3.0, 1.0}, {4.0, 1.0}, {5.0, 1.0}]},
  # Equal sigmas make the scalar and vector arms mathematically identical, so
  # a mis-assigned span or a colliding shader-cache key is invisible in them.
  # These two are not permutation-invariant.
  {"scalar 5 obs, sigmas 1..5", scalar_ir, 0.0, 5.0,
   [{1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}, {5.0, 5.0}]},
  {"scalar 4 obs, sigmas .5/1/2/4", scalar_ir, 0.0, 10.0,
   [{4.0, 0.5}, {3.8, 1.0}, {4.2, 2.0}, {3.5, 4.0}]}
]

variants = if only == [], do: variants, else: Enum.filter(variants, &(elem(&1, 0) in only))

IO.puts("""
=== observed-model evidence — docs/OPEN_VULKAN_OBSERVED_MODEL.md ===
compiler       : #{compiler}
warmup/samples : #{warmup}/#{samples}
seeds          : #{Enum.join(seeds, ",")} (each reported separately)
backend        : #{inspect(Nx.default_backend())}
""")

hdr =
  String.pad_trailing("variant", 32) <>
    String.pad_trailing("seed", 6) <>
    String.pad_trailing("mean", 10) <>
    String.pad_trailing("truth", 10) <>
    String.pad_trailing("sd", 10) <>
    String.pad_trailing("truth", 10) <>
    String.pad_trailing("distinct", 10) <>
    String.pad_trailing("gpu", 10) <>
    String.pad_trailing("eps", 12) <> "verdict"

IO.puts(hdr)
IO.puts(String.duplicate("-", String.length(hdr)))

# Vacuity guard. A vulkan row that silently fell back to the host path reads
# exactly like a vulkan row that passed, which is how a green differential
# suite hid two defects for weeks. Count the actual GPU dispatches per row and
# print them; force-load first or trace_pattern matches nothing (NEXT.md §5).
Code.ensure_loaded!(Exmc.NUTS.Vulkan.Dispatch)
:erlang.trace_pattern({Exmc.NUTS.Vulkan.Dispatch, :chain, 8}, true, [:call_count])

dispatches = fn ->
  case :erlang.trace_info({Exmc.NUTS.Vulkan.Dispatch, :chain, 8}, :call_count) do
    {:call_count, n} when is_integer(n) -> n
    _ -> 0
  end
end

f = fn x, d -> :erlang.float_to_binary(x * 1.0, decimals: d) end

rows =
  for {label, build, m0, s0, obs} <- variants, seed <- seeds do
    {tm, tsd} = truth.(m0, s0, obs)
    ir = build.(m0, s0, obs)
    before_calls = dispatches.()

    {draws, eps, err} =
      try do
        {trace, stats} =
          Sampler.sample(ir, %{}, num_warmup: warmup, num_samples: samples, seed: seed)

        d = trace |> Map.fetch!("mu") |> Nx.to_flat_list()

        eps =
          case stats[:step_size] do
            nil -> :nan
            t when is_number(t) -> t * 1.0
            t -> t |> Nx.to_number() |> Kernel.*(1.0)
          end

        {d, eps, nil}
      rescue
        e -> {[], :nan, Exception.message(e) |> String.split("\n") |> hd()}
      end

    if err do
      IO.puts(String.pad_trailing(label, 32) <> String.pad_trailing("#{seed}", 6) <> "ERROR — " <> err)
      {label, seed, :error}
    else
      n = length(draws)
      mean = Enum.sum(draws) / n
      var = Enum.reduce(draws, 0.0, fn x, a -> a + (x - mean) * (x - mean) end) / (n - 1)
      sd = :math.sqrt(var)
      distinct = draws |> MapSet.new() |> MapSet.size()

      # Frozen chain (the defect's signature) vs merely inaccurate.
      verdict =
        cond do
          distinct < div(n, 10) -> "FROZEN #{distinct}/#{n}"
          abs(mean - tm) > 0.5 -> "mean off"
          abs(sd - tsd) / tsd > 0.25 -> "sd off"
          true -> "ok"
        end

      IO.puts(
        String.pad_trailing(label, 32) <>
          String.pad_trailing("#{seed}", 6) <>
          String.pad_trailing(f.(mean, 4), 10) <>
          String.pad_trailing(f.(tm, 4), 10) <>
          String.pad_trailing(f.(sd, 4), 10) <>
          String.pad_trailing(f.(tsd, 4), 10) <>
          String.pad_trailing("#{distinct}/#{n}", 10) <>
          String.pad_trailing("#{dispatches.() - before_calls}", 10) <>
          String.pad_trailing(
            (if eps == :nan, do: "-", else: f.(eps, 6)),
            12
          ) <> verdict
      )

      {label, seed, verdict}
    end
  end

IO.puts("")
bad = Enum.reject(rows, fn {_l, _s, v} -> v == "ok" end)

if bad == [] do
  IO.puts("#{length(rows)} rows, all within tolerance of the closed form")
else
  IO.puts("#{length(bad)} of #{length(rows)} rows BAD:")
  for {l, s, v} <- bad, do: IO.puts("  #{l} seed=#{s}: #{v}")
  System.at_exit(fn _ -> exit({:shutdown, 1}) end)
end
