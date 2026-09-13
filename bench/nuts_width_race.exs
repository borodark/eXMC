# The chain shader's envelope, end to end: NUTS wall time, ESS per second and
# error against the CLOSED-FORM posterior, over model width, on the chain
# shader and on the host tree. MISSION.md P4 asks "keep the chain shader or
# retire it"; this is the measurement that decides it.
#
#   COMPILER is not read here; ARMS is.
#   ARMS=vulkan,none DIMS=1,4,16,64 NOBS=20,200,2000 \
#     MIX_ENV=test mix run --no-deps-check --no-compile bench/nuts_width_race.exs
#
# THE MODEL is the regression idiom the chain shader was extended for
# (test/exmc/nuts/custom_synth/vector_rv_test.exs):
#
#   beta ~ Normal(0, 5), shape {d}
#   y    ~ Normal(X * beta, 0.3)        X is {n_obs, d}, captured by the closure
#
# With known sigma the posterior is Gaussian in closed form, so every cell is
# also a correctness check, not only a stopwatch:
#
#   precision  L = X'X / sigma^2 + I / 25
#   mean       m = L^-1 X'y / sigma^2,   sd_j = sqrt((L^-1)_jj)
#
# Reported per coordinate as z_j = (mean_j - m_j) / (sd_j / sqrt(ess_j)), the
# error in units of that coordinate's own Monte-Carlo standard error. `max_z`
# is the worst over d coordinates. For a correct sampler |z| is ~N(0,1), so
# with d=64 a max near 3 is ordinary; a max of 6+ is a defect.
#
# DATA is generated with :rand under an explicit exsss seed, so every host gets
# the same X, beta and y bit-for-bit. Sampling is NOT bit-reproducible across
# hosts (docs/ARMS.md), which is why the comparison is statistical.
#
# WHAT IS TIMED: `compile_s` is Compiler.compile_for_sampling (on vulkan,
# including GLSL synthesis and SPIR-V compile or cache hit), and `sample_s` is
# Sampler.sample_compiled alone. Both are wall time in this process, after one
# untimed warm-up sample on a tiny model per arm (first-client Vulkan startup
# and module loading are not per-model costs).
#
# BUDGET: a cell whose sample_s exceeds CELL_BUDGET_S marks every larger cell
# (d' >= d and n' >= n) for that arm as `skipped`, so a slow arm cannot turn the
# sweep into a day. Cells are run smallest first.

alias Exmc.{Builder, Compiler, Dist, IR}
alias Exmc.NUTS.Sampler
alias Exmc.NUTS.Vulkan.{Dispatch, Validator}

env = fn k, d -> System.get_env(k) || d end

ints = fn k, d ->
  env.(k, d) |> String.split(",", trim: true) |> Enum.map(&String.to_integer/1)
end

arms = env.("ARMS", "vulkan,none") |> String.split(",", trim: true) |> Enum.map(&String.to_atom/1)
dims = ints.("DIMS", "1,4,16,64")
nobs = ints.("NOBS", "20,200,2000")
warmup = String.to_integer(env.("WARMUP", "300"))
samples = String.to_integer(env.("SAMPLES", "500"))
seed = String.to_integer(env.("SEED", "1"))
budget_s = String.to_integer(env.("CELL_BUDGET_S", "600"))
sigma = 0.3
prior_sd = 5.0
f64 = [type: :f64, backend: Nx.BinaryBackend]

defmodule WidthRace do
  # {n, d} standard normals, deterministic in `seed`.
  def normals(seed, n, d) do
    s0 = :rand.seed_s(:exsss, {seed, seed * 7919, seed * 104_729})

    {vals, _} =
      Enum.map_reduce(1..(n * d), s0, fn _, s ->
        :rand.normal_s(s)
      end)

    Nx.tensor(vals, type: :f64, backend: Nx.BinaryBackend) |> Nx.reshape({n, d})
  end

  def fmt(x) when is_float(x), do: :erlang.float_to_binary(x, decimals: 3)
  def fmt(x), do: to_string(x)
end

data = fn d, n ->
  x = WidthRace.normals(seed, n, d)
  beta_true = WidthRace.normals(seed + 1, d, 1) |> Nx.reshape({d})
  noise = WidthRace.normals(seed + 2, n, 1) |> Nx.reshape({n}) |> Nx.multiply(sigma)
  y = Nx.add(Nx.dot(x, beta_true), noise)

  xtx = Nx.dot(Nx.transpose(x), x)

  precision =
    Nx.add(
      Nx.divide(xtx, sigma * sigma),
      Nx.multiply(Nx.eye(d, type: :f64, backend: Nx.BinaryBackend), 1.0 / (prior_sd * prior_sd))
    )

  cov = Nx.LinAlg.invert(precision)
  mean = Nx.dot(cov, Nx.divide(Nx.dot(Nx.transpose(x), y), sigma * sigma))
  sd = cov |> Nx.take_diagonal() |> Nx.sqrt()
  {x, y, Nx.to_flat_list(mean), Nx.to_flat_list(sd)}
end

model = fn x, y, d ->
  lik =
    Dist.Custom.new(fn _obs, p ->
      r = Nx.subtract(y, Nx.dot(x, p.beta))
      Nx.sum(Nx.divide(Nx.multiply(r, r), -2 * sigma * sigma))
    end)

  IR.new()
  |> Builder.rv("beta", Dist.Normal, %{mu: Nx.tensor(0.0, f64), sigma: Nx.tensor(prior_sd, f64)},
    shape: {d}
  )
  |> Dist.Custom.rv("Y", lik, %{beta: "beta"})
  |> Builder.obs("Y_obs", "Y", y)
end

IO.puts("""
=== NUTS width race: chain shader vs host tree, against the closed-form posterior ===
host     : #{:inet.gethostname() |> elem(1)}  #{:erlang.system_info(:system_architecture)}
arms     : #{Enum.join(arms, ",")}
dims     : #{Enum.join(dims, ",")}    n_obs: #{Enum.join(nobs, ",")}
draws    : #{warmup} warmup / #{samples} samples, seed #{seed}, cell budget #{budget_s} s
device   : #{System.get_env("NXV_DEVICE") || "<unpinned>"}
""")

cells = for d <- dims, n <- nobs, do: {d, n}

results =
  for arm <- arms do
    Application.put_env(:exmc, :compiler, arm)

    # Untimed warm-up: module loading, and on vulkan the first-client startup.
    # d=2, not d=1: a `shape: {1}` vector RV does not synthesise
    # ({:unsupported_op, :param_vec}, measured 2026-09-13), and a d=1 cell
    # should report that as its own row rather than abort the sweep here.
    {wx, wy, _, _} = data.(2, 5)

    try do
      Sampler.sample_compiled(Compiler.compile_for_sampling(model.(wx, wy, 2)), %{},
        num_warmup: 20,
        num_samples: 20,
        seed: 1
      )
    rescue
      e ->
        IO.puts("warm-up failed on #{arm}: #{Exception.message(e) |> String.split("\n") |> hd()}")
    end

    {rows, _over} =
      Enum.map_reduce(cells, [], fn {d, n}, over ->
        if Enum.any?(over, fn {od, on} -> d >= od and n >= on end) do
          row = %{arm: arm, d: d, n: n, status: "skipped (budget)"}
          IO.puts("#W arm=#{arm} d=#{d} n_obs=#{n} status=skipped")
          {row, over}
        else
          {x, y, truth_mean, truth_sd} = data.(d, n)

          row =
            try do
              {compile_us, compiled} =
                :timer.tc(fn -> Compiler.compile_for_sampling(model.(x, y, d)) end)

              chain =
                case elem(compiled, 5) do
                  {:synthesised, _, _, _, _, _, _} -> "synthesised"
                  nil -> "none"
                  other -> inspect(elem(other, 0))
                end

              Dispatch.reset_dispatch_count()

              {sample_us, {trace, stats}} =
                :timer.tc(fn ->
                  Sampler.sample_compiled(compiled, %{},
                    num_warmup: warmup,
                    num_samples: samples,
                    seed: seed
                  )
                end)

              dispatches = Dispatch.dispatch_count()
              draws = trace |> Map.fetch!("beta") |> Nx.backend_transfer(Nx.BinaryBackend)
              draws = if Nx.rank(draws) == 1, do: Nx.reshape(draws, {:auto, 1}), else: draws

              per_coord =
                for j <- 0..(d - 1) do
                  col = draws[[.., j]] |> Nx.to_flat_list()
                  ess = Validator.ess(col)
                  m = Enum.sum(col) / length(col)
                  se = Enum.at(truth_sd, j) / :math.sqrt(max(ess, 1.0))
                  {ess, abs(m - Enum.at(truth_mean, j)) / se}
                end

              min_ess = per_coord |> Enum.map(&elem(&1, 0)) |> Enum.min()
              max_z = per_coord |> Enum.map(&elem(&1, 1)) |> Enum.max()
              sample_s = sample_us / 1.0e6

              %{
                arm: arm,
                d: d,
                n: n,
                status: if(max_z > 6.0, do: "WRONG (max_z > 6)", else: "ok"),
                chain: chain,
                compile_s: compile_us / 1.0e6,
                sample_s: sample_s,
                dispatches: dispatches,
                min_ess: min_ess,
                ess_per_s: min_ess / sample_s,
                max_z: max_z,
                divergences: stats.divergences,
                step: stats.step_size
              }
            rescue
              e ->
                %{
                  arm: arm,
                  d: d,
                  n: n,
                  status:
                    "error: " <>
                      (Exception.message(e) |> String.split("\n") |> hd() |> String.slice(0, 160))
                }
            end

          IO.puts(
            "#W " <>
              Enum.map_join(row, " ", fn {k, v} -> "#{k}=#{WidthRace.fmt(v)}" end)
          )

          over =
            if is_number(row[:sample_s]) and row.sample_s > budget_s,
              do: [{d, n} | over],
              else: over

          {row, over}
        end
      end)

    rows
  end
  |> List.flatten()

IO.puts(
  "\n| d | n_obs | arm | status | chain dispatches | compile s | sample s | min ESS | ESS/s | max z | div |"
)

IO.puts("|---|---|---|---|---|---|---|---|---|---|---|")

for r <- Enum.sort_by(results, &{&1.d, &1.n, &1.arm}) do
  IO.puts(
    "| #{r.d} | #{r.n} | #{r.arm} | #{r.status} | #{r[:dispatches] || ""} | " <>
      Enum.map_join(
        [:compile_s, :sample_s, :min_ess, :ess_per_s, :max_z, :divergences],
        " | ",
        fn k ->
          if Map.has_key?(r, k), do: WidthRace.fmt(r[k]), else: ""
        end
      ) <> " |"
  )
end

IO.puts("\nESS/s ratio vulkan / none, per cell (>1 means the chain shader wins):")

for d <- dims, n <- nobs do
  v = Enum.find(results, &(&1.d == d and &1.n == n and &1.arm == :vulkan))
  h = Enum.find(results, &(&1.d == d and &1.n == n and &1.arm == :none))

  if v && h && is_number(v[:ess_per_s]) && is_number(h[:ess_per_s]) do
    IO.puts("  d=#{d} n_obs=#{n}: #{WidthRace.fmt(v.ess_per_s / h.ess_per_s)}x")
  end
end
