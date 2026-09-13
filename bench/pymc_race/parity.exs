# Gate 1, eXMC half: are the two sides the same model?
#
#   bench/pymc_race/.venv/bin/python bench/pymc_race/parity.py     # writes the points
#   MIX_ENV=test mix run --no-deps-check --no-compile bench/pymc_race/parity.exs
#
# For every model and both exmc variants: take PyMC's 200 points (constrained
# values), map each to exmc's unconstrained layout through exmc's own inverse
# transforms, evaluate exmc's compiled log density on the CPU arm with NCP
# off (the race models are centered on both sides), subtract exmc's Jacobian,
# and compare with PyMC's density without Jacobian.
#
# A constant offset is allowed and reported: normalising constants may be kept
# by one side and dropped by the other (exmc's funnel Custom density drops
# -0.5 log 2*pi per coordinate). What must match is everything that depends on
# the parameters. PASS: max |diff_i - median(diff)| / max(1, |logp_i|) < 1e-9.
#
# Exit 0 if every model passes, 1 otherwise. A model that fails does not race.

Code.require_file("models.exs", __DIR__)

alias PymcRace.Models

Application.put_env(:exmc, :compiler, :none)

data = __DIR__ |> Path.join("data.json") |> File.read!() |> Jason.decode!()
# PARITY_POINTS overrides the points file, for negative controls.
parity =
  (System.get_env("PARITY_POINTS") || Path.join(__DIR__, "parity_points.json"))
  |> File.read!()
  |> Jason.decode!()

tolerance = 1.0e-9

# exmc's Transform has forward maps and Jacobians only; the inverses live here.
inverse = fn
  nil, x ->
    x

  :log, x ->
    :math.log(x)

  # softplus^-1(x) = log(expm1(x)) = x + log(-expm1(-x)). OTP has neither
  # :math.expm1 nor :math.log1p, so Nx's, in f64; the second form does not
  # overflow for large x.
  :softplus, x ->
    xt = Nx.tensor(x, type: :f64)
    xt |> Nx.negate() |> Nx.expm1() |> Nx.negate() |> Nx.log() |> Nx.add(xt) |> Nx.to_number()
end

median = fn xs ->
  s = Enum.sort(xs)
  Enum.at(s, div(length(s), 2))
end

variants_for = fn
  name when name in ["simple", "medium", "stress"] -> [:vector]
  _ -> [:vector, :scalar]
end

results =
  for name <- Models.models(), variant <- variants_for.(name) do
    %{"points" => points, "logp_nojac" => pymc_logps} = parity["models"][name]
    %{ir: ir, map: map} = Models.build(name, variant, data)

    try do
      {logp_fn, pm} = Exmc.Compiler.compile(ir, ncp: false)

      unmapped = Enum.reject(pm.entries, &Map.has_key?(map, &1.id))
      if unmapped != [], do: throw({:unmapped, Enum.map(unmapped, & &1.id)})

      diffs =
        Enum.zip(points, pymc_logps)
        |> Enum.map(fn {point, pymc_lp} ->
          {us, jac} =
            Enum.reduce(pm.entries, {[], 0.0}, fn entry, {acc, jac} ->
              {pname, idx} = Map.fetch!(map, entry.id)
              vals = Map.fetch!(point, pname)
              vals = if idx == :all, do: vals, else: [Enum.at(vals, idx)]

              if length(vals) != entry.length,
                do: throw({:length, entry.id, entry.length, length(vals)})

              u = Enum.map(vals, &inverse.(entry.transform, &1))

              j =
                Exmc.Transform.log_abs_det_jacobian(entry.transform, Nx.tensor(u, type: :f64))
                |> Nx.sum()
                |> Nx.to_number()

              {acc ++ u, jac + j}
            end)

          lp = logp_fn.(Nx.tensor(us, type: :f64)) |> Nx.to_number()
          {lp - jac - pymc_lp, pymc_lp}
        end)

      raw = Enum.map(diffs, &elem(&1, 0))
      offset = median.(raw)

      worst =
        diffs
        |> Enum.map(fn {d, lp} -> abs(d - offset) / max(1.0, abs(lp)) end)
        |> Enum.max()

      status = if is_number(worst) and worst < tolerance, do: "PASS", else: "FAIL"

      transforms =
        pm.entries |> Enum.map(&"#{&1.id}:#{&1.transform || "id"}") |> Enum.uniq() |> Enum.take(4)

      %{
        model: name,
        variant: variant,
        status: status,
        offset: offset,
        worst: worst,
        d: pm.size,
        transforms: transforms
      }
    rescue
      e ->
        %{
          model: name,
          variant: variant,
          status: "ERROR",
          error: Exception.message(e) |> String.slice(0, 300)
        }
    catch
      kind, why -> %{model: name, variant: variant, status: "ERROR", error: inspect({kind, why})}
    end
  end

IO.puts(
  "\n| model | variant | d | status | constant offset | worst rel. residual | exmc transforms (first 4) |"
)

IO.puts("|---|---|---|---|---|---|---|")

for r <- results do
  if r.status == "ERROR" do
    IO.puts("| #{r.model} | #{r.variant} | | **ERROR** | | | #{r.error} |")
  else
    IO.puts(
      "| #{r.model} | #{r.variant} | #{r.d} | #{if r.status == "PASS", do: "PASS", else: "**FAIL**"} | " <>
        "#{:erlang.float_to_binary(r.offset * 1.0, [{:scientific, 6}])} | " <>
        "#{:erlang.float_to_binary(r.worst * 1.0, [{:scientific, 2}])} | #{Enum.join(r.transforms, " ")} |"
    )
  end
end

System.halt(if Enum.all?(results, &(&1.status == "PASS")), do: 0, else: 1)
