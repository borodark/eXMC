# race.exs — diff two posteriordb result artifacts.
#
#   mix run benchmark/posteriordb/race.exs A.json B.json
#
# A and B are JSON artifacts written by validate_posteriordb.exs (schema
# exmc.posteriordb.v1). Reports per-model paired ratios and their spread.
#
# ## Read the noise floor before you read the result
#
# The first thing to do with this tool is race a build against ITSELF: two runs,
# same commit, same protocol. Whatever spread that produces is the detection
# threshold of the whole apparatus, and any later "gain" smaller than it is not
# a result. A tool that has never been shown to report zero on a no-op change
# has not been shown to report anything.

defmodule PDB.Race do
  # Ratios are ratio data, so the centre is the geometric mean and the spread is
  # in log space. An arithmetic mean of ratios is asymmetric under swapping A
  # and B, which is exactly the property a race must not have.
  @metrics [
    {:wall_ms, :lower_better, "wall"},
    {:leapfrog_per_sec_apparent, :higher_better, "leapfrog/s"},
    {:ess_per_sec, :higher_better, "ESS/s"},
    {:min_ess, :higher_better, "min ESS"},
    {:leapfrog, :neutral, "leapfrog"}
  ]

  def main([a_path, b_path]) do
    a = load(a_path)
    b = load(b_path)

    check_comparable!(a, b)
    header(a, b)

    a_models = index(a)
    b_models = index(b)
    names = a_models |> Map.keys() |> Enum.filter(&Map.has_key?(b_models, &1)) |> Enum.sort()

    only_a = Map.keys(a_models) -- Map.keys(b_models)
    only_b = Map.keys(b_models) -- Map.keys(a_models)
    if only_a != [], do: warn("only in A: #{Enum.join(only_a, ", ")}")
    if only_b != [], do: warn("only in B: #{Enum.join(only_b, ", ")}")

    if names == [], do: abort("no models in common")

    IO.puts("\n#{length(names)} models in common. Ratios are B/A.\n")

    for {key, dir, label} <- @metrics do
      ratios =
        names
        |> Enum.map(fn n -> {n, ratio(a_models[n][to_string(key)], b_models[n][to_string(key)])} end)
        |> Enum.reject(fn {_, r} -> r == nil end)

      report_metric(label, dir, ratios)
    end

    correctness(names, a_models, b_models)
  end

  def main(_), do: abort("usage: mix run benchmark/posteriordb/race.exs A.json B.json")

  defp report_metric(label, _dir, []), do: IO.puts("#{pad(label)} (no data)")

  defp report_metric(label, dir, ratios) do
    values = Enum.map(ratios, fn {_, r} -> r end)
    g = geomean(values)
    {lo_name, lo} = Enum.min_by(ratios, fn {_, r} -> r end)
    {hi_name, hi} = Enum.max_by(ratios, fn {_, r} -> r end)

    # Spread in log space, reported back as a symmetric +/- percentage.
    spread = :math.exp(logsd(values)) - 1.0

    IO.puts(
      "#{pad(label)} #{pct(g)}  #{verdict(dir, g)}    " <>
        "spread +/-#{Float.round(spread * 100, 1)}%   " <>
        "range #{pct(lo)} (#{short(lo_name)}) .. #{pct(hi)} (#{short(hi_name)})"
    )
  end

  # A geometric mean is only meaningful on positive data.
  defp ratio(a, b) when is_number(a) and is_number(b) and a > 0 and b > 0, do: b / a
  defp ratio(_, _), do: nil

  defp geomean(vs), do: :math.exp(Enum.sum(Enum.map(vs, &:math.log/1)) / length(vs))

  defp logsd([_]), do: 0.0

  defp logsd(vs) do
    logs = Enum.map(vs, &:math.log/1)
    n = length(logs)
    m = Enum.sum(logs) / n
    :math.sqrt(Enum.reduce(logs, 0.0, fn x, a -> a + (x - m) * (x - m) end) / (n - 1))
  end

  defp verdict(:neutral, _), do: ""

  defp verdict(dir, g) do
    delta = (g - 1.0) * 100

    cond do
      abs(delta) < 0.05 -> "(identical)"
      dir == :lower_better and delta < 0 -> "(B faster)"
      dir == :lower_better -> "(B slower)"
      delta > 0 -> "(B better)"
      true -> "(B worse)"
    end
  end

  defp correctness(names, a_models, b_models) do
    IO.puts("")

    flipped =
      Enum.filter(names, fn n -> a_models[n]["status"] != b_models[n]["status"] end)

    if flipped == [] do
      IO.puts("Status: unchanged on all #{length(names)} models.")
    else
      IO.puts("Status CHANGED on #{length(flipped)} model(s):")

      for n <- flipped do
        IO.puts("  #{n}: #{a_models[n]["status"]} -> #{b_models[n]["status"]}")
      end
    end

    # Divergence rate is recorded but never gated (that is step (d)). Surface
    # a material move here regardless, since it is the cheapest early warning
    # that a "speedup" was bought with sampler quality.
    moved =
      names
      |> Enum.map(fn n -> {n, a_models[n]["div_rate"], b_models[n]["div_rate"]} end)
      |> Enum.filter(fn {_, x, y} ->
        is_number(x) and is_number(y) and abs(y - x) > 0.02
      end)

    if moved != [] do
      IO.puts("\nDivergence rate moved by >2pp on #{length(moved)} model(s):")

      for {n, x, y} <- moved do
        IO.puts("  #{n}: #{Float.round(x * 100, 1)}% -> #{Float.round(y * 100, 1)}%")
      end
    end
  end

  # --- provenance ------------------------------------------------------------

  # Refuse to compare runs whose PROTOCOL differs; a different seed or draw
  # count is a different experiment, not a slower one. The compiler and the
  # commits are allowed to differ — those are the dimensions a race exists to
  # vary — but they are printed so the reader knows which one this is.
  @must_match ~w(num_warmup num_samples seed ncp)

  defp check_comparable!(a, b) do
    pa = a["provenance"]
    pb = b["provenance"]

    for k <- @must_match do
      if pa[k] != pb[k] do
        abort("protocol differs on #{k}: #{inspect(pa[k])} vs #{inspect(pb[k])}. " <>
                "Different protocol is a different experiment, not a slower one.")
      end
    end

    if pa["mode"] != "race" or pb["mode"] != "race" do
      warn(
        "one or both artifacts were produced in :validate mode (parallel). Their wall " <>
          "times measure scheduler contention, not the sampler. Timing ratios below are " <>
          "not usable; correctness and ESS still are."
      )
    end

    if pa["host"] != pb["host"] do
      warn("different hosts (#{pa["host"]} vs #{pb["host"]}) — hardware is confounded with the change.")
    end

    if pa["exmc_dirty"] or pb["exmc_dirty"] do
      warn("a tree was DIRTY; the recorded sha does not fully describe what ran.")
    end
  end

  defp header(a, b) do
    pa = a["provenance"]
    pb = b["provenance"]

    IO.puts("\n=== posteriordb race ===")
    IO.puts("  A  #{describe(pa)}")
    IO.puts("  B  #{describe(pb)}")

    if pa["exmc_sha"] == pb["exmc_sha"] and pa["nx_vulkan_sha"] == pb["nx_vulkan_sha"] and
         pa["compiler_resolved"] == pb["compiler_resolved"] do
      IO.puts("""

        SELF-RACE: identical commits, identical compiler. There is no change
        under test, so everything below is the noise floor of the apparatus.
        Read the spread as the detection threshold: a later result smaller than
        it is not a result.\
      """)
    end
  end

  defp describe(p) do
    "#{String.slice(p["exmc_sha"] || "", 0, 9)}#{if p["exmc_dirty"], do: "+dirty", else: ""} " <>
      "nxv=#{String.slice(p["nx_vulkan_sha"] || "", 0, 7)} " <>
      "#{p["compiler_resolved"]} #{p["precision"]} " <>
      "mode=#{p["mode"]} par=#{p["parallel"]} " <>
      "#{p["num_warmup"]}w+#{p["num_samples"]}s seed=#{p["seed"]} " <>
      "#{p["timestamp"]}"
  end

  # --- helpers ---------------------------------------------------------------

  defp index(doc), do: Map.new(doc["models"], fn m -> {m["name"], m} end)

  defp load(path) do
    doc = path |> File.read!() |> Jason.decode!()

    case doc["schema"] do
      "exmc.posteriordb.v1" -> doc
      other -> abort("#{path}: unknown schema #{inspect(other)}")
    end
  end

  defp pct(r), do: "#{if r >= 1.0, do: "+", else: ""}#{Float.round((r - 1.0) * 100, 1)}%"
  defp pad(s), do: String.pad_trailing(s, 14)
  defp short(n), do: n |> String.split("-") |> List.last()
  defp warn(msg), do: IO.puts(:stderr, "  WARNING: #{msg}")

  defp abort(msg) do
    IO.puts(:stderr, "ERROR: #{msg}")
    System.halt(1)
  end
end

PDB.Race.main(System.argv())
