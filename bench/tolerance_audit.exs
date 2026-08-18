# What do the suite's statistical tolerances actually admit?
#
# NEXT.md item 2: "find every tolerance that would accept a 20% variance
# error". This runs each model the suite asserts on, at the sample size that
# test uses, and reports three numbers per row:
#
#   admits(mean)  the largest mean error the test's own delta lets through,
#                 as a multiple of the posterior sd. 1.0 sd is a big miss.
#   admits(var)   the largest variance error it lets through, as a percentage.
#                 A test that asserts no dispersion at all admits ANY variance
#                 error, including a frozen chain, and is printed as `no gate`.
#   4sd floor     what a 4-sigma analytic gate would admit at THIS chain's
#                 effective sample size. This is the floor: no tolerance can be
#                 tightened below it without going flaky, so where the floor is
#                 above 20% the fix is more draws, not a smaller number.
#
# Usage:
#   mix run --no-deps-check bench/tolerance_audit.exs
#   SEEDS=1,2,3 mix run --no-deps-check bench/tolerance_audit.exs

alias Exmc.Builder
alias Exmc.Dist.{Normal, Exponential, Beta, Gamma}
alias Exmc.NUTS.{Sampler, Vulkan.Validator}

env = fn n, d -> System.get_env(n) || d end
compiler = env.("COMPILER", "none") |> String.to_atom()
Application.put_env(:exmc, :compiler, compiler)

# {label, ir_fn, meta_for_analytic_moments, warmup, samples, seed,
#  current_mean_delta | nil, current_sd_delta | nil}
# {label, ir_fn, var_name, meta, warmup, samples, seed, mean_delta | nil, sd_delta | nil}
#
# Each test appears twice: `was` is the configuration and assertion this suite
# carried before 2026-08-18, `now` is what it carries after. The point of the
# pair is that the `4sd floor` column moves — the tolerance rewrite alone would
# have been cosmetic.
normal_prior = fn -> Builder.new_ir() |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)}) end

conjugate = fn ->
  Builder.new_ir()
  |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(10.0)})
  |> Builder.rv("x", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
  |> Builder.obs("x_obs", "x", Nx.tensor(5.0))
end

gamma_prior = fn -> Builder.new_ir() |> Builder.rv("alpha", Gamma, %{alpha: Nx.tensor(2.0), beta: Nx.tensor(1.0)}) end
exp_prior = fn -> Builder.new_ir() |> Builder.rv("rate", Exponential, %{lambda: Nx.tensor(2.0)}) end
beta_prior = fn -> Builder.new_ir() |> Builder.rv("p", Beta, %{alpha: Nx.tensor(2.0), beta: Nx.tensor(5.0)}) end
lognormal_prior = fn -> Builder.new_ir() |> Builder.rv("x", Exmc.Dist.Lognormal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(0.5)}) end

conj_meta = {:normal, 5.0 / 1.01, :math.sqrt(1.0 / 1.01)}

cases = [
  {"integration:15 conjugate N-N      was", conjugate, "mu", conj_meta, 300, 500, 42, nil, nil},
  {"integration:15 conjugate N-N      now", conjugate, "mu", conj_meta, 300, 4000, 42, nil, nil},
  {"integration:88 Gamma(2,1)         was", gamma_prior, "alpha", {:gamma, 2.0, 1.0}, 200, 200, 99, 1.0, nil},
  {"integration:88 Gamma(2,1)         now", gamma_prior, "alpha", {:gamma, 2.0, 1.0}, 500, 8500, 99, nil, nil},
  {"integration:113 Exponential(2)    was", exp_prior, "rate", {:exponential, 2.0}, 200, 300, 77, 0.3, nil},
  {"integration:113 Exponential(2)    now", exp_prior, "rate", {:exponential, 2.0}, 500, 12000, 77, nil, nil},
  {"integration:270 Beta(2,5)         was", beta_prior, "p", {:beta, 2.0, 5.0}, 300, 400, 88, 0.15, nil},
  {"integration:270 Beta(2,5)         now", beta_prior, "p", {:beta, 2.0, 5.0}, 500, 5000, 88, nil, nil},
  {"new_dist:230 Lognormal(0,0.5)     was", lognormal_prior, "x", {:lognormal, 0.0, 0.5}, 200, 300, 42, 0.5, nil},
  {"new_dist:230 Lognormal(0,0.5)     now", lognormal_prior, "x", {:lognormal, 0.0, 0.5}, 500, 10000, 42, nil, nil},
  {"stan_test:8 conjugate N-N         was", conjugate, "mu", conj_meta, 300, 500, 42, 0.5, nil},
  {"stan_test:8 conjugate N-N         now", conjugate, "mu", conj_meta, 300, 4000, 42, nil, nil},
  # `assert abs(var - 1.0) < 1.0` is an absolute band on the VARIANCE, not the
  # sd: it admits any variance in [0, 2], i.e. 100% either way, and it passes
  # for a frozen chain.
  {"nuts_test:308 Normal(0,1)         was", normal_prior, "mu", {:normal, 0.0, 1.0}, 500, 500, 42, 0.3, {:var_pct, 100.0}},
  {"nuts_test:308 Normal(0,1)         now", normal_prior, "mu", {:normal, 0.0, 1.0}, 500, 4000, 42, nil, nil}
]

seeds =
  env.("SEEDS", "")
  |> String.split(",", trim: true)
  |> Enum.map(&String.to_integer(String.trim(&1)))

IO.puts("""
=== what the suite's statistical tolerances admit ===
compiler : #{compiler}
seeds    : #{if seeds == [], do: "each test's own", else: Enum.join(seeds, ",")}
""")

hdr =
  String.pad_trailing("model / test", 44) <>
    String.pad_trailing("n", 6) <>
    String.pad_trailing("ESS", 7) <>
    String.pad_trailing("admits(mean)", 14) <>
    String.pad_trailing("admits(var)", 13) <>
    String.pad_trailing("4sd floor", 11) <> "verdict"

IO.puts(hdr)
IO.puts(String.duplicate("-", String.length(hdr)))

f2 = fn x -> :erlang.float_to_binary(x * 1.0, decimals: 2) end

for {label, build, var_name, meta, warmup, samples, own_seed, mean_delta, sd_delta} <- cases do
  seed_list = if seeds == [], do: [own_seed], else: seeds

  draws =
    Enum.flat_map(seed_list, fn seed ->
      {trace, _} =
        Sampler.sample(build.(), %{}, num_warmup: warmup, num_samples: samples, seed: seed)

      trace |> Map.fetch!(var_name) |> Nx.to_flat_list()
    end)

  {:moments, %{mean: tm, var: tv}} = Validator.analytic_moments(meta)
  tsd = :math.sqrt(tv)
  ess = Validator.ess(draws)

  # What the current deltas admit.
  analytic? = String.ends_with?(label, "now")

  admits_mean =
    cond do
      mean_delta -> f2.(mean_delta / tsd) <> " sd"
      analytic? -> "analytic"
      true -> "no gate"
    end

  admits_var =
    case sd_delta do
      nil -> if analytic?, do: "analytic", else: "no gate"
      {:var_pct, pct} -> f2.(pct) <> "%"
      d -> f2.((:math.pow((tsd + d) / tsd, 2) - 1.0) * 100.0) <> "%"
    end

  # The floor: a 4-sigma analytic gate at this chain's own ESS.
  # se(var) for a normal target is var * sqrt(2/n_eff); use the sample's own
  # fourth moment so heavy-tailed targets are not flattered.
  {m, v} = Validator.mean_var(draws)
  n = length(draws)
  mu4 = Enum.reduce(draws, 0.0, fn x, a -> a + :math.pow(x - m, 4) end) / n
  se_v = :math.sqrt(max(mu4 - v * v, 0.0) / ess)
  floor_pct = 4.0 * se_v / tv * 100.0

  # After the sweep every `now` row uses assert_posterior!, whose gate IS the
  # 4-sigma analytic one — so for those the floor is the tolerance, and the
  # only question is whether it meets the 20% bar.
  verdict =
    cond do
      String.ends_with?(label, "was") and is_nil(sd_delta) -> "NO DISPERSION GATE"
      String.ends_with?(label, "was") -> "gate far wider than the defect"
      floor_pct > 20.0 -> "STILL too few draws"
      floor_pct > 16.0 -> "resolves 20%, thin"
      true -> "resolves 20%"
    end

  IO.puts(
    String.pad_trailing(label, 44) <>
      String.pad_trailing("#{length(draws)}", 6) <>
      String.pad_trailing("#{round(ess)}", 7) <>
      String.pad_trailing(admits_mean, 14) <>
      String.pad_trailing(admits_var, 13) <>
      String.pad_trailing(f2.(floor_pct) <> "%", 11) <> verdict
  )
end

IO.puts("""

`4sd floor` is the smallest variance error a 4-sigma analytic gate could
detect at that chain's ESS. Where it exceeds 20%, no tolerance rewrite makes
the test able to see a 20% variance error — it needs more effective draws.
""")
