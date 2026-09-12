# Is the Validator's KS check failing on the shader, or on its own statistic?
#
# Samples Cauchy(0, 1) on both arms — the Evaluator reference and the f64
# chain shader — with the Validator's own sizes (300 warmup, 800 draws) across
# eight seeds, and prints the two-sample KS statistic beside two critical
# values: the one the Validator used until 2026-09-12 (raw n = m = 800) and the
# one it uses now (ESS-sized, the D92 rule the KS had been left out of).
#
# MEASURED 2026-09-12, super-io (RTX 3060 Ti, 580.178.04), nx_vulkan 16d13f3:
# 3 of 8 seeds reject with raw n at a nominal α = 0.001; 1 of 8 with ESS-sized
# n. Seed 42 is the suite's standing failure on this host (d 0.1000 vs 0.0975).
# Seed 46 (d 0.30, GPU ESS 80) is a real outlier worth its own look.
#
#     EXMC_COMPILER=vulkan MIX_ENV=test mix run bench/validator_ks_seeds.exs
alias Exmc.{Builder, Dist}
alias Exmc.NUTS.Sampler
alias Exmc.NUTS.Vulkan.Validator

ir =
  Builder.new_ir() |> Builder.rv("x", Dist.Cauchy, %{loc: Nx.tensor(0.0), scale: Nx.tensor(1.0)})

opts = [num_warmup: 300, num_samples: 800]
c = 1.9495

meta = {:cauchy, 0.0, 1.0, -:math.log(:math.pi())}

run = fn compiler, seed ->
  prev = Application.get_env(:exmc, :compiler)
  prev_meta = Application.get_env(:exmc, :fused_leapfrog_meta)
  Application.put_env(:exmc, :compiler, compiler)

  if compiler == :vulkan,
    do: Application.put_env(:exmc, :fused_leapfrog_meta, meta),
    else: Application.delete_env(:exmc, :fused_leapfrog_meta)

  {trace, stats} =
    try do
      Sampler.sample(ir, %{}, Keyword.put(opts, :seed, seed))
    after
      if prev,
        do: Application.put_env(:exmc, :compiler, prev),
        else: Application.delete_env(:exmc, :compiler)

      if prev_meta,
        do: Application.put_env(:exmc, :fused_leapfrog_meta, prev_meta),
        else: Application.delete_env(:exmc, :fused_leapfrog_meta)
    end

  {Nx.to_flat_list(trace["x"]), stats}
end

IO.puts("seed | d      crit_raw  ->  raw | ess_host ess_gpu  crit_eff  ->  eff | gpu eps")

for seed <- [42, 43, 44, 45, 46, 47, 48, 49] do
  {a, _} = run.(:none, seed)
  {b, sb} = run.(:vulkan, seed)
  ea = Validator.ess(a)
  eb = Validator.ess(b)
  n = length(a)
  m = length(b)

  d =
    case Validator.check_ks(a, b) do
      :ok -> nil
      {:error, e} -> e.d
    end

  # recompute d the same way regardless of pass/fail
  sa = Enum.sort(a)
  sb2 = Enum.sort(b)

  d2 =
    Enum.reduce(Enum.uniq(sa ++ sb2), 0.0, fn x, acc ->
      fa = Enum.count(sa, &(&1 <= x)) / n
      fb = Enum.count(sb2, &(&1 <= x)) / m
      max(acc, abs(fa - fb))
    end)

  crit_raw = c * :math.sqrt((n + m) / (n * m))
  crit_eff = c * :math.sqrt((ea + eb) / (ea * eb))

  IO.puts(
    :io_lib.format("~4B | ~.4f  ~.4f  ->  ~s | ~7.1f ~7.1f  ~.4f  ->  ~s | ~.3f", [
      seed,
      d2,
      crit_raw,
      if(d2 <= crit_raw, do: "pass", else: "FAIL"),
      ea,
      eb,
      crit_eff,
      if(d2 <= crit_eff, do: "pass", else: "FAIL"),
      Map.get(sb, :step_size) || 0.0
    ])
  )
end
