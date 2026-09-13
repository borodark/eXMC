# Step 4, eXMC half: long reference runs (docs/PYMC_RACE_PLAN.md, gate 2).
#
#   MIX_ENV=test mix run --no-deps-check --no-compile bench/pymc_race/reference_exmc.exs [model ...]
#
# EXLA arm, 4 chains x 10,000 draws after 2,000 warmup, target_accept 0.9, the
# `:vector` builders from models.exs with NCP off. Draws are written as raw
# little-endian f64 to reference/exmc_<model>__<var>.bin, shape
# (chains, draws, k), with reference/exmc_<model>.json naming shapes, the wall
# time, divergences and stats.provenance.
#
# Eight schools and the funnel use explicitly NON-CENTERED builders defined here,
# mirroring reference_pymc.py, and theta / x are computed from the draws.
# Explicit rather than exmc's automatic NCP, for symmetry with the PyMC
# reference, which cannot rely on an automatic rewrite. (Automatic NCP on a
# vector RV used to crash while rebuilding the trace; fixed in 134f9d3aa.)

Code.require_file("models.exs", __DIR__)

alias Exmc.{Builder, Dist}
alias Exmc.NUTS.Sampler

Application.put_env(:exmc, :compiler, :exla)

here = __DIR__
out_dir = Path.join(here, "reference")
File.mkdir_p!(out_dir)
data = here |> Path.join("data.json") |> File.read!() |> Jason.decode!()
t = fn v -> Nx.tensor(v, type: :f64) end
opts = [num_warmup: 2_000, num_samples: 10_000, seed: 2026, target_accept: 0.9]

eight_schools_nc = fn ->
  es = data["eight_schools"]
  y = t.(Enum.map(es["y"], &(&1 * 1.0)))
  sigma = t.(Enum.map(es["sigma"], &(&1 * 1.0)))

  lik =
    Dist.Custom.new(fn _x, p ->
      r = Nx.divide(Nx.subtract(y, Nx.add(p.mu, Nx.multiply(p.tau, p.z))), sigma)
      Nx.sum(Nx.multiply(-0.5, Nx.multiply(r, r)))
    end)

  ir =
    Builder.new_ir()
    |> Builder.rv("mu", Dist.Normal, %{mu: t.(0.0), sigma: t.(5.0)})
    |> Builder.rv("tau", Dist.HalfNormal, %{sigma: t.(5.0)})
    |> Builder.rv("z", Dist.Normal, %{mu: t.(0.0), sigma: t.(1.0)}, shape: {8})
    |> Dist.Custom.rv("Y", lik, %{mu: "mu", tau: "tau", z: "z"})
    |> Builder.obs("Y_obs", "Y", y)

  derive = fn tr ->
    %{
      "mu" => tr["mu"],
      "tau" => tr["tau"],
      "theta" => Nx.add(Nx.new_axis(tr["mu"], 1), Nx.multiply(Nx.new_axis(tr["tau"], 1), tr["z"]))
    }
  end

  {ir, derive}
end

funnel_nc = fn ->
  ir =
    Builder.new_ir()
    |> Builder.rv("y", Dist.Normal, %{mu: t.(0.0), sigma: t.(3.0)})
    |> Builder.rv("z", Dist.Normal, %{mu: t.(0.0), sigma: t.(1.0)}, shape: {9})

  derive = fn tr ->
    %{
      "y" => tr["y"],
      "x" => Nx.multiply(Nx.exp(Nx.new_axis(Nx.divide(tr["y"], 2.0), 1)), tr["z"])
    }
  end

  {ir, derive}
end

plain = fn name ->
  fn ->
    %{ir: ir, map: map} = PymcRace.Models.build(name, :vector, data)
    ids = Map.keys(map)
    {ir, fn tr -> Map.new(ids, &{&1, tr[&1]}) end}
  end
end

builders = %{
  "simple" => plain.("simple"),
  "medium" => plain.("medium"),
  "stress" => plain.("stress"),
  "eight_schools" => eight_schools_nc,
  "funnel" => funnel_nc,
  "logistic" => plain.("logistic"),
  "sv" => plain.("sv")
}

names =
  if System.argv() == [],
    do: ~w(simple medium stress eight_schools funnel logistic sv),
    else: System.argv()

for name <- names do
  {ir, derive} = Map.fetch!(builders, name).()
  t0 = System.monotonic_time(:millisecond)
  compiled = Sampler.compile(ir, ncp: false)
  {traces, stats} = Sampler.sample_chains_compiled(compiled, 4, opts)
  wall = (System.monotonic_time(:millisecond) - t0) / 1000

  per_chain = Enum.map(traces, derive)

  shapes =
    for var <- Map.keys(hd(per_chain)), into: %{} do
      stacked =
        per_chain
        |> Enum.map(fn tr ->
          tr[var] |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.as_type(:f64)
        end)
        |> Enum.map(fn x -> if Nx.rank(x) == 1, do: Nx.new_axis(x, 1), else: x end)
        |> Nx.stack()

      File.write!(Path.join(out_dir, "exmc_#{name}__#{var}.bin"), Nx.to_binary(stacked))
      {var, Tuple.to_list(Nx.shape(stacked))}
    end

  divergences = stats |> Enum.map(& &1.divergences) |> Enum.sum()

  meta = %{
    model: name,
    wall_s: wall,
    divergences: divergences,
    shapes: shapes,
    opts: Map.new(opts),
    provenance: hd(stats).provenance |> Map.drop([:host])
  }

  File.write!(Path.join(out_dir, "exmc_#{name}.json"), Jason.encode!(meta, pretty: true))

  IO.puts(
    "#{String.pad_trailing(name, 14)} wall #{:erlang.float_to_binary(wall, decimals: 1)} s  divergences #{divergences}  vars #{inspect(Map.keys(shapes))}"
  )
end
