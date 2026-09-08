# harness.exs — provenance, metrics and report emission for the posteriordb suite.
#
# Loaded by validate_posteriordb.exs via Code.require_file/2. Split out because
# the racer (step (b)) needs the same provenance and the same JSON schema; two
# copies of a result schema is how two result files stop being comparable.

defmodule PDB.Provenance do
  @moduledoc """
  Everything a result file needs in order to be comparable to another one.

  The old report recorded none of this. It named neither the backend nor any
  commit, and `Exmc.Sampler.sample/3` was called with no compiler set — under
  `mix run` that is `:dev`, where `config/test.exs` is not imported, so
  `EXMC_COMPILER` did nothing and `JIT.detect_compiler/0` silently picked EXLA.
  A results file that cannot say what produced it cannot be raced against
  anything, which is why the 2026-02-19 baseline is unattributable.

  `compiler_resolved` is read back from `Exmc.JIT` AFTER the request is applied,
  never assumed from the request. `JIT.demand/2` raises when a named compiler is
  unusable, so a mismatch here should be impossible — recording it is how we
  find out if that ever stops being true.
  """

  def collect(opts) do
    %{
      timestamp: DateTime.utc_now() |> DateTime.to_iso8601(),
      exmc_sha: git_sha("."),
      exmc_dirty: dirty?("."),
      nx_vulkan_sha: git_sha("deps/nx_vulkan"),
      host: hostname(),
      schedulers_online: System.schedulers_online(),
      otp_release: System.otp_release(),
      elixir_version: System.version(),
      compiler_requested: inspect(Keyword.fetch!(opts, :compiler)),
      compiler_resolved: inspect(Exmc.JIT.detect_compiler()),
      backend: inspect(Exmc.JIT.backend()),
      precision: inspect(Exmc.JIT.precision()),
      jit_describe: Exmc.JIT.describe(),
      mode: to_string(Keyword.fetch!(opts, :mode)),
      parallel: Keyword.fetch!(opts, :parallel),
      num_warmup: Keyword.fetch!(opts, :num_warmup),
      num_samples: Keyword.fetch!(opts, :num_samples),
      seed: Keyword.fetch!(opts, :seed),
      ncp: Keyword.fetch!(opts, :ncp),
      transcendentals: inspect(Keyword.fetch!(opts, :transcendentals)),
      chains: Keyword.fetch!(opts, :chains),
      tier: to_string(Keyword.fetch!(opts, :tier))
    }
  end

  @doc """
  One line, printed at start and stored in the artifact. Read it before
  believing any number underneath it.
  """
  def banner(p) do
    """
    === posteriordb #{p.mode} ===
      exmc          #{p.exmc_sha}#{if p.exmc_dirty, do: " (DIRTY)", else: ""}
      nx_vulkan     #{p.nx_vulkan_sha}
      host          #{p.host}  (#{p.schedulers_online} schedulers, OTP #{p.otp_release}, Elixir #{p.elixir_version})
      compiler      requested #{p.compiler_requested} -> resolved #{p.compiler_resolved}
      #{p.jit_describe}
      protocol      #{p.num_warmup} warmup + #{p.num_samples} sampling, seed=#{p.seed}, ncp=#{p.ncp}
      transcend.    #{p.transcendentals}
      tier          #{p.tier}
      chains        #{p.chains}#{if p.chains < 2, do: " (R-hat CANNOT be computed -- needs >= 2)", else: ", seeds #{p.seed}..#{p.seed + p.chains - 1}"}
      parallel      #{p.parallel}#{if p.mode == "race" and p.parallel == 1, do: " (serialized — timings are meaningful)", else: ""}
    """
  end

  defp git_sha(dir) do
    case System.cmd("git", ["-C", dir, "rev-parse", "HEAD"], stderr_to_stdout: true) do
      {out, 0} -> String.trim(out)
      _ -> "unknown"
    end
  end

  # TRACKED changes only. `--porcelain` alone counts untracked files, so a
  # stray directory that no build reads -- `.claude/`, an editor scratch dir,
  # a downloaded fixture -- stamped every report DIRTY and invalidated the
  # comparability the field exists to certify. Observed 2026-09-07: a full
  # posteriordb run recorded `(DIRTY)` for `?? .claude/`.
  #
  # `-uno` is the fix rather than a `.gitignore` entry, because the question
  # this field answers is "does the code that ran match the sha", and an
  # untracked file is not that code.
  defp dirty?(dir) do
    case System.cmd("git", ["-C", dir, "status", "--porcelain", "-uno"], stderr_to_stdout: true) do
      {out, 0} -> String.trim(out) != ""
      _ -> nil
    end
  end

  defp hostname do
    case :inet.gethostname() do
      {:ok, h} -> to_string(h)
      _ -> "unknown"
    end
  end
end

defmodule PDB.Metrics do
  @moduledoc """
  The numbers the old report threw away.

  `stats.sample_stats` has always carried per-draw `:n_steps`, `:tree_depth`,
  `:divergent` and `:accept_prob`; the validator read only `divergences` and
  `step_size` and discarded the rest. ESS was never computed at all, which is
  why the pass criteria could not see a performance regression: a model going
  from ESS 400 to ESS 50 loses 8x efficiency and still passes a fixed
  "mean within 0.5 SD" gate comfortably.

  ## The warmup dilution, named so it cannot silently corrupt a race

  `sample_stats` is produced by `run_sampling/8` and covers the SAMPLING phase
  only (`sampler.ex:286`). `wall_ms` spans warmup AND sampling. So
  `leapfrog / wall` understates true throughput by roughly the warmup fraction.

  That bias is a constant factor of the protocol, not of the code under test, so
  it CANCELS in an A/B ratio and is safe for racing. It is NOT absolute
  throughput and must never be quoted as such. The field is named
  `leapfrog_per_sec_apparent` for that reason.
  """

  @doc "Total leapfrog steps taken during the sampling phase."
  def leapfrog(stats) do
    stats
    |> Map.get(:sample_stats, [])
    |> Enum.reduce(0, fn s, acc -> acc + Map.get(s, :n_steps, 0) end)
  end

  @doc "Mean accepted-proposal probability over the sampling phase."
  def mean_accept(stats) do
    case Map.get(stats, :sample_stats, []) do
      [] -> nil
      ss -> Enum.reduce(ss, 0.0, &(&2 + Map.get(&1, :accept_prob, 0.0))) / length(ss)
    end
  end

  @doc "Mean NUTS tree depth over the sampling phase."
  def mean_tree_depth(stats) do
    case Map.get(stats, :sample_stats, []) do
      [] -> nil
      ss -> Enum.reduce(ss, 0.0, &(&2 + Map.get(&1, :tree_depth, 0))) / length(ss)
    end
  end

  @doc """
  Per-parameter ESS and MCSE for one chain.

  `ess_bulk/1` is the rank-normalised estimator (Vehtari 2021), matching
  ArviZ's `az.ess(method="bulk")`. MCSE = sd / sqrt(ESS) — the unit the moment
  checks SHOULD be expressed in, rather than a fixed fraction of the reference
  SD.
  """
  def ess_table(trace, exmc_names) do
    exmc_names
    |> Enum.filter(&Map.has_key?(trace, &1))
    |> Map.new(fn name ->
      values = trace[name] |> Nx.to_flat_list()
      ess = Exmc.Diagnostics.ess_bulk(values)
      s = sd(values)

      {name,
       %{
         ess: Float.round(ess, 1),
         sd: s,
         mcse: if(ess > 0, do: s / :math.sqrt(ess), else: nil)
       }}
    end)
  end

  @doc """
  Per-parameter convergence statistics across N chains.

  Returns `%{param => %{rhat:, ess_total:, ess_min_per_chain:, mean:, sd:,
  mcse:}}`, where `mean`/`sd` pool every draw from every chain and `mcse` is
  `sd / sqrt(ess_total)`.

  `rhat` is `nil` when there is only one chain. `Diagnostics.rhat/1` requires
  at least two, and a gate that cannot be evaluated must say so rather than
  pass by default -- the whole reason the old criteria were misleading is that
  they could not fail for the reason anyone cared about.
  """
  def param_stats(chain_traces, exmc_names) do
    exmc_names
    |> Enum.filter(fn n -> Enum.all?(chain_traces, &Map.has_key?(&1, n)) end)
    |> Map.new(fn name ->
      per_chain = Enum.map(chain_traces, fn t -> t[name] |> Nx.to_flat_list() end)
      pooled = List.flatten(per_chain)

      ess_each = Enum.map(per_chain, &Exmc.Diagnostics.ess_bulk/1)
      ess_total = Enum.sum(ess_each)
      s = sd(pooled)

      {name,
       %{
         rhat: if(length(per_chain) >= 2, do: Exmc.Diagnostics.rhat(per_chain), else: nil),
         ess_total: Float.round(ess_total, 1),
         ess_min_per_chain: Float.round(Enum.min(ess_each), 1),
         mean: mean(pooled),
         sd: s,
         mcse: if(ess_total > 0, do: s / :math.sqrt(ess_total), else: nil)
       }}
    end)
  end

  defp mean([]), do: 0.0
  defp mean(list), do: Enum.sum(list) / length(list)

  defp sd([]), do: 0.0
  defp sd([_]), do: 0.0

  defp sd(list) do
    n = length(list)
    m = Enum.sum(list) / n
    :math.sqrt(Enum.reduce(list, 0.0, fn x, a -> a + (x - m) * (x - m) end) / (n - 1))
  end
end

defmodule PDB.Report do
  @moduledoc """
  Emits JSON beside the markdown.

  The markdown is for a human reading one run. The JSON is so that "race" is a
  diff of two artifacts rather than eyeballing two tables six months apart.
  """

  def write(results, provenance, dir) do
    stamp = provenance.timestamp |> String.replace(~r/[:.]/, "-")
    json_path = Path.join(dir, "results_#{provenance.mode}_#{stamp}.json")

    payload = %{
      schema: "exmc.posteriordb.v1",
      provenance: provenance,
      models: Enum.map(results, &jsonable/1)
    }

    File.write!(json_path, Jason.encode!(payload, pretty: true))
    {json_path, latest_link(json_path, provenance, dir)}
  end

  # Drop tensors and anything Jason cannot encode; keep every scalar metric.
  defp jsonable(r) do
    r
    |> Map.drop([:comparisons])
    |> Map.put(:comparisons, Enum.map(Map.get(r, :comparisons, []), &Map.drop(&1, [])))
    |> Map.update(:status, "unknown", &to_string/1)
  end

  # A stable filename so tooling can find the most recent run without globbing.
  defp latest_link(json_path, provenance, dir) do
    latest = Path.join(dir, "results_#{provenance.mode}_latest.json")
    File.cp!(json_path, latest)
    latest
  end
end
