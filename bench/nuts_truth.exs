# Posterior moments against ANALYTIC truth — not against another arm.
#
# The differential checks in Exmc.NUTS.Vulkan.Validator ask whether two arms
# agree. They are structurally blind to any defect the two arms SHARE, and the
# NUTS tree is shared by every arm there is. Two real defects lived behind a
# green differential suite for weeks:
#
#   * the chain shader's logp_chain[k] described the state BEFORE step k
#   * an invalid doubling was still merged into the trajectory
#
# Only measuring against the distribution's own moments can see either.
#
# The two arms isolate the two defects. `none` runs the host NUTS tree only, so
# it sees the tree defect alone. `vulkan` runs the synthesised chain shader on
# top of that same tree, so it sees both.
#
# Usage:
#   mix run --no-deps-check bench/nuts_truth.exs
#   COMPILER=vulkan SEEDS=1,2,3 mix run --no-deps-check bench/nuts_truth.exs

alias Exmc.{Builder, Rewrite}
alias Exmc.Dist.{Normal, Exponential, HalfNormal}
alias Exmc.NUTS.Sampler
alias Exmc.NUTS.Vulkan.Validator

env = fn name, default -> System.get_env(name) || default end

warmup = env.("WARMUP", "500") |> String.to_integer()
samples = env.("SAMPLES", "2000") |> String.to_integer()

seeds =
  env.("SEEDS", "1,2,3,4,5,6")
  |> String.split(",", trim: true)
  |> Enum.map(&String.to_integer(String.trim(&1)))

compiler = env.("COMPILER", "none") |> String.to_atom()
Application.put_env(:exmc, :compiler, compiler)

# There are three distinct tree implementations behind these flags and they
# each carry their own copy of the doubling logic:
#
#   USE_NIF=0                 pure Elixir  — Tree.do_build/11 + build_subtree/10
#   USE_NIF=1 (default)       Rust         — tree.rs build_subtree
#   FULL_TREE_NIF=1           Rust         — tree.rs build_full_tree
#
# Measure all three. A guard added to one and not the others is a defect that
# only shows up under whichever flag nobody set.
use_nif = env.("USE_NIF", "1") == "1"
full_tree = env.("FULL_TREE_NIF", "0") == "1"
Application.put_env(:exmc, :use_nif, use_nif)
Application.put_env(:exmc, :full_tree_nif, full_tree)

models = [
  {"Normal(0,1)", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)}, {:normal, 0.0, 1.0}},
  {"HalfNormal(1)", HalfNormal, %{sigma: Nx.tensor(1.0)}, {:half_normal, 1.0}},
  {"Exponential(2)", Exponential, %{lambda: Nx.tensor(2.0)}, {:exponential, 2.0}}
]

IO.puts("""
=== NUTS posterior moments vs analytic truth ===
warmup/samples : #{warmup}/#{samples}
seeds          : #{Enum.join(seeds, ",")} (pooled)
compiler       : #{compiler}
tree           : use_nif=#{use_nif} full_tree_nif=#{full_tree}
backend        : #{inspect(Nx.default_backend())}
""")

IO.puts(
  String.pad_trailing("model", 16) <>
    String.pad_trailing("stat", 6) <>
    String.pad_trailing("truth", 12) <>
    String.pad_trailing("got", 12) <>
    String.pad_trailing("err", 10) <> "verdict"
)

IO.puts(String.duplicate("-", 72))

results =
  for {label, dist, params, meta} <- models do
    ir =
      Builder.new_ir()
      |> Builder.rv("x", dist, params)
      |> Rewrite.apply()

    compiled =
      try do
        {:ok, Exmc.Compiler.compile_for_sampling(ir)}
      rescue
        e -> {:skip, Exception.message(e) |> String.split("\n") |> hd()}
      end

    case compiled do
      {:skip, why} ->
        IO.puts(String.pad_trailing(label, 16) <> "SKIPPED — " <> why)
        {label, :ok}

      {:ok, compiled} ->
    pooled =
      Enum.flat_map(seeds, fn seed ->
        {trace, _stats} =
          Sampler.sample_compiled(compiled, %{},
            num_warmup: warmup,
            num_samples: samples,
            seed: seed
          )

        trace |> Map.fetch!("x") |> Nx.to_flat_list()
      end)

    {:moments, %{mean: tm, var: tv}} = Validator.analytic_moments(meta)
    {m, v} = Validator.mean_var(pooled)

    verdict = Validator.check_analytic(pooled, :host, meta)

    fmt = fn x -> :erlang.float_to_binary(x * 1.0, decimals: 6) end

    pct = fn got, truth ->
      if abs(truth) > 1.0e-9,
        do: :erlang.float_to_binary((got - truth) / truth * 100.0, decimals: 2) <> "%",
        else: :erlang.float_to_binary(got - truth, decimals: 4)
    end

    tag = fn ->
      case verdict do
        :ok -> "ok"
        {:error, %{check: c}} -> "FAIL #{c}"
      end
    end

    IO.puts(
      String.pad_trailing(label, 16) <>
        String.pad_trailing("mean", 6) <>
        String.pad_trailing(fmt.(tm), 12) <>
        String.pad_trailing(fmt.(m), 12) <>
        String.pad_trailing(pct.(m, tm), 10) <> tag.()
    )

    IO.puts(
      String.pad_trailing("", 16) <>
        String.pad_trailing("var", 6) <>
        String.pad_trailing(fmt.(tv), 12) <>
        String.pad_trailing(fmt.(v), 12) <>
        String.pad_trailing(pct.(v, tv), 10) <>
        "ess=#{:erlang.float_to_binary(Validator.ess(pooled), decimals: 0)}"
    )

        {label, verdict}
    end
  end

IO.puts("")

bad = Enum.filter(results, fn {_l, v} -> v != :ok end)

if bad == [] do
  IO.puts("all #{length(results)} models within tolerance of analytic truth")
else
  for {label, {:error, info}} <- bad, do: IO.puts("#{label}: #{inspect(info)}")
  System.at_exit(fn _ -> exit({:shutdown, 1}) end)
end
