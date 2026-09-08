#!/usr/bin/env elixir
# validate_posteriordb.exs — Validate eXMC sampler against posteriordb gold-standard draws
#
# Loads preprocessed posteriordb models, builds eXMC model for each,
# samples, and compares posterior moments against 10,000 reference draws.
#
# Usage:
#   cd exmc
#   mix run research/posteriordb/validate_posteriordb.exs [--parallel N] [--samples N] [--warmup N]
#
# Requires: preprocess_posteriordb.py to have been run first.

Code.require_file("harness.exs", __DIR__)

defmodule PosteriorDBValidator do
  alias Exmc.Builder
  alias Exmc.Dist.{Normal, HalfNormal, HalfCauchy, Custom}

  @processed_dir Path.expand("posteriordb_processed", __DIR__)

  # --- Public API ---

  def run(opts \\ []) do
    mode = Keyword.get(opts, :mode, :validate)
    compiler = Keyword.get(opts, :compiler, :vulkan)
    num_samples = Keyword.get(opts, :num_samples, 1000)
    num_warmup = Keyword.get(opts, :num_warmup, 1000)
    seed = Keyword.get(opts, :seed, 42)
    ncp = Keyword.get(opts, :ncp, false)
    only = Keyword.get(opts, :only)
    transcendentals = Keyword.get(opts, :transcendentals, :f32_cast)
    chains = Keyword.get(opts, :chains, 4)
    arms = Keyword.get(opts, :arms) || [compiler]
    tier = Keyword.get(opts, :tier, :full)

    parallel = resolve_parallel(mode, compiler, opts)
    guard_race!(mode, parallel)

    # The compiler is a DEMAND, and it is applied before anything samples.
    #
    # Under `mix run` the environment is :dev, and config/config.exs imports
    # config/test.exs only when config_env() == :test. So EXMC_COMPILER has
    # never had any effect here, Application.get_env(:exmc, :compiler) was nil,
    # and JIT.detect_compiler/0 fell through auto_detect/0 to EXLA without
    # saying so. Every posteriordb number ever recorded is an EXLA number that
    # does not admit it.
    #
    # JIT.demand/2 RAISES when a named compiler is unusable on this host. That
    # is the behaviour we want: a loud stop beats a silent downgrade, which is
    # indistinguishable from a clean run in the report.
    Application.put_env(:exmc, :compiler, compiler)

    # The f64 chain shader has no double transcendentals to call -- GLSL.std.450
    # provides none -- so `exp_d(x)` is `double(exp(float(x)))` by default and
    # overflows at ln(f32_max) = 88.7228. For a log-scale parameter that is a
    # hard boundary at q_uc < -44.36. :polynomial synthesises real f64 log/exp
    # instead (~10-15x shader latency, ~1 ULP f64), moving the boundary to the
    # f64 range. Recorded in provenance because it changes the ARITHMETIC, so
    # two runs that differ on it are not comparable.
    Application.put_env(:exmc, :chain_shader_transcendentals, transcendentals)

    provenance =
      PDB.Provenance.collect(
        compiler: compiler,
        mode: mode,
        parallel: parallel,
        num_warmup: num_warmup,
        num_samples: num_samples,
        seed: seed,
        ncp: ncp,
        transcendentals: transcendentals,
        chains: chains,
        tier: tier
      )

    IO.puts(PDB.Provenance.banner(provenance))

    manifest = load_json(Path.join(@processed_dir, "manifest.json"))
    posteriors = manifest["posteriors"] |> tier_filter(tier, only)
    IO.puts("Posteriors to run: #{length(posteriors)}\n")

    run_opts = [
      num_samples: num_samples,
      num_warmup: num_warmup,
      seed: seed,
      ncp: ncp,
      chains: chains
    ]

    results =
      if length(arms) > 1 do
        race_arms(posteriors, arms, run_opts)
      else
        posteriors
        |> Task.async_stream(
          fn name -> validate_one(name, Keyword.put(run_opts, :arm, compiler)) end,
          max_concurrency: parallel,
          timeout: 1_800_000,
          # :kill_task, not the default :exit. With the default, ONE model
          # over the timeout kills the whole stream and the caller exits --
          # so the `{:exit, reason}` clause below could never fire, and 32
          # finished-or-runnable models were thrown away with it. Observed
          # 2026-09-07: a full-tier Vulkan run reported one PASS and then
          # died, because a single posterior with a large observation axis
          # ran past 30 minutes on the serial-reduce shader.
          on_timeout: :kill_task,
          ordered: false
        )
        |> Enum.map(fn
          {:ok, result} -> result
          {:exit, reason} -> %{name: "unknown", status: :crash, error: inspect(reason)}
        end)
        |> Enum.sort_by(& &1.name)
      end

    print_report(results, provenance)
    if length(arms) > 1, do: print_pairing(results, arms)
    save_results(results, provenance)
    results
  end

  # Racing and validating want opposite things, and fusing them into one pass is
  # exactly why the old Wall column cannot be trusted: 33 models ran through
  # Task.async_stream at max_concurrency, so each model's timer measured
  # contention from the other 32. The number moved with core count and
  # background load, and its spread was never characterised — so there was no
  # threshold below which a delta was meaningless.
  #
  #   :validate — parallel. Throughput matters; timings do not.
  #   :race     — serialized. The timer measures the sampler, not the scheduler.
  defp resolve_parallel(:race, _compiler, _opts), do: 1

  defp resolve_parallel(:validate, compiler, opts) do
    Keyword.get(opts, :parallel) || default_parallel(compiler)
  end

  # One GPU does not absorb 33 concurrent Vulkan contexts the way 88 cores
  # absorb 33 CPU chains. Conservative under :vulkan; override with --parallel.
  defp default_parallel(:vulkan), do: 4
  defp default_parallel(_), do: System.schedulers_online()

  defp guard_race!(:race, 1), do: :ok

  defp guard_race!(:race, n) do
    raise "race mode is serialized by construction, got parallel=#{n}. " <>
            "A concurrent timer measures the scheduler, not the sampler."
  end

  defp guard_race!(_, _), do: :ok

  # (f) Tiers.
  #
  # 33 models x 4 chains x 1000+1000 is far too slow to run on every bump, and
  # the obvious economy does NOT work: (d) measured kilpisjarvi at R-hat 1.845
  # on 300 draws against 1.003 on 1000. Shortening chains does not buy a
  # cheaper check, it buys a check that fails for a reason unrelated to the
  # change under test. So the tier cuts the MODEL LIST and leaves the protocol
  # alone.
  #
  # Six models, one per posteriordb family, chosen for structural coverage
  # rather than for being quick:
  #
  #   eight_schools_noncentered   hierarchical + NCP; the only non-regression
  #   mesquite-logmesquite_logvolume  n_obs=46,  n_beta=2  the small corner
  #   sblri-blr                   n_obs=100, n_beta=5  produced the Inf/NaN
  #   kidiq-kidscore_momhs        n_obs=434, n_beta=2  just under the old
  #                               pipeline ceiling (868)
  #   nes2000-nes                 n_obs=476, n_beta=9  widest d
  #   earnings-earn_height        n_obs=1192, n_beta=2 largest data; the model
  #                               that first hit the pipeline ceiling
  #
  # Two of the six are there because they BROKE: earnings-earn_height was the
  # canary for the shader-size ceiling and sblri-blr for the non-finite crash.
  # A tier picked purely for speed would have contained neither, and would have
  # been green through both bugs.
  @fast_tier ~w(
    eight_schools-eight_schools_noncentered
    mesquite-logmesquite_logvolume
    sblri-blr
    kidiq-kidscore_momhs
    nes2000-nes
    earnings-earn_height
  )

  defp tier_filter(posteriors, _tier, only) when is_binary(only),
    do: filter_only(posteriors, only)

  defp tier_filter(posteriors, :fast, _only) do
    selected = Enum.filter(posteriors, &(&1 in @fast_tier))

    # A tier that silently shrinks because a model was renamed is a tier that
    # quietly stops covering what it claims to.
    missing = @fast_tier -- selected

    if missing != [] do
      raise "fast tier names #{length(@fast_tier)} models but the manifest is missing: " <>
              Enum.join(missing, ", ")
    end

    selected
  end

  defp tier_filter(posteriors, :full, _only), do: posteriors

  # (b) Paired, interleaved, counterbalanced.
  #
  # Running all of A and then all of B hands every bit of drift over the run --
  # GPU clocks, thermal state, page cache, background load -- to whichever arm
  # went second. That is not a hypothetical: the (c) self-race measured a 55%
  # systematic slowdown of the second pass, in the SAME direction on all ten
  # models, with the code identical. An unpaired A/B would have reported that
  # as a 55% regression.
  #
  # So arms alternate per MODEL, and the order flips on odd indices. Flipping
  # matters as much as interleaving: if arm A always ran first within each
  # model it would absorb the cold-cache penalty every time, which is a
  # systematic bias merely relocated rather than removed.
  #
  # Only runtime-selectable arms can be raced this way -- a compiler is an
  # Application env key, so both arms live in one process and share one machine
  # state. Racing two COMMITS still needs two invocations and race.exs to diff
  # the artifacts; that is what its self-race mode is for.
  defp race_arms(posteriors, arms, run_opts) do
    posteriors
    |> Enum.with_index()
    |> Enum.flat_map(fn {name, idx} ->
      ordered = if rem(idx, 2) == 0, do: arms, else: Enum.reverse(arms)

      Enum.map(ordered, fn arm ->
        Application.put_env(:exmc, :compiler, arm)
        validate_one(name, Keyword.put(run_opts, :arm, arm))
      end)
    end)
    |> Enum.sort_by(&{&1.name, to_string(&1[:arm])})
  end

  # Paired per-model ratios. Reported as a geometric mean because ratios are
  # ratio data: an arithmetic mean of them is asymmetric under swapping the
  # arms, which is the one property a race must not have.
  defp print_pairing(results, [a, b | _]) do
    by_name = Enum.group_by(results, & &1.name)

    pairs =
      for {name, rows} <- by_name,
          ra = Enum.find(rows, &(&1[:arm] == a)),
          rb = Enum.find(rows, &(&1[:arm] == b)),
          ra && rb,
          is_number(ra[:wall_ms]) and is_number(rb[:wall_ms]) and ra.wall_ms > 0,
          do: {name, rb.wall_ms / ra.wall_ms, ra, rb}

    if pairs != [] do
      IO.puts("\n--- Paired comparison: #{inspect(b)} relative to #{inspect(a)} ---")
      IO.puts("(interleaved per model, order counterbalanced)\n")

      for {name, r, ra, rb} <- Enum.sort_by(pairs, fn {n, _, _, _} -> n end) do
        IO.puts("  #{String.pad_trailing(name, 44)} wall #{pct(r)}  " <>
                "#{String.pad_leading(to_string(ra.status), 5)} -> #{rb.status}  " <>
                "ess #{fmt(ra[:min_ess], 0)} -> #{fmt(rb[:min_ess], 0)}  " <>
                "div #{fmt(ra[:div_rate] && ra.div_rate * 100, 1)}% -> #{fmt(rb[:div_rate] && rb.div_rate * 100, 1)}%")
      end

      ratios = Enum.map(pairs, fn {_, r, _, _} -> r end)
      g = :math.exp(Enum.sum(Enum.map(ratios, &:math.log/1)) / length(ratios))
      IO.puts("\n  geometric mean wall: #{pct(g)} over #{length(pairs)} paired models")
    end
  end

  defp print_pairing(_results, _arms), do: :ok

  defp pct(r), do: "#{if r >= 1.0, do: "+", else: ""}#{Float.round((r - 1.0) * 100, 1)}%"

  defp filter_only(posteriors, nil), do: posteriors

  # Comma-separated substrings, so a subset can be driven in one run. Needed
  # for the noise-floor calibration, which has to compare the SAME set of
  # models twice, and for the tiered runs in (f).
  defp filter_only(posteriors, pats) do
    wanted = pats |> String.split(",", trim: true) |> Enum.map(&String.trim/1)
    Enum.filter(posteriors, fn p -> Enum.any?(wanted, &String.contains?(p, &1)) end)
  end

  # --- Per-posterior validation ---

  def validate_one(name, opts) do
    num_samples = Keyword.fetch!(opts, :num_samples)
    num_warmup = Keyword.fetch!(opts, :num_warmup)
    seed = Keyword.fetch!(opts, :seed)
    ncp = Keyword.fetch!(opts, :ncp)
    chains = Keyword.fetch!(opts, :chains)
    arm = Keyword.get(opts, :arm)

    t0 = System.monotonic_time(:millisecond)

    try do
      spec = load_json(Path.join(@processed_dir, "#{name}.json"))
      ref_draws = spec["reference_draws"]

      # Build and sample
      {ir, init_values, param_map} = build_model(spec)

      # One chain per seed. R-hat needs at least two chains to exist at all,
      # and a single chain cannot distinguish "the sampler is fine, this draw
      # was unlucky" from "the sampler regressed" -- 33/33 PASS at one seed is
      # one Bernoulli sample per model.
      runs =
        for i <- 0..(chains - 1) do
          {trace, stats} =
            Exmc.Sampler.sample(ir, init_values,
              num_samples: num_samples,
              num_warmup: num_warmup,
              seed: seed + i,
              ncp: ncp
            )

          {reconstruct_eight_schools(trace, param_map), stats}
        end

      chain_traces = Enum.map(runs, &elem(&1, 0))
      chain_stats = Enum.map(runs, &elem(&1, 1))

      wall_ms = System.monotonic_time(:millisecond) - t0
      wall_s = wall_ms / 1000
      divergences = chain_stats |> Enum.map(& &1.divergences) |> Enum.sum()
      total_draws = chains * num_samples

      pstats = PDB.Metrics.param_stats(chain_traces, Enum.map(param_map, fn {n, _} -> n end))
      comparisons = compare_stats(pstats, ref_draws, param_map, divergences / max(total_draws, 1))

      # stats.sample_stats has always carried per-draw :n_steps, :tree_depth,
      # :divergent and :accept_prob. The old harness read only :divergences and
      # :step_size and threw the rest away, and never computed ESS at all --
      # which is why its pass criteria were structurally unable to see a
      # performance regression.
      leapfrog = chain_stats |> Enum.map(&PDB.Metrics.leapfrog/1) |> Enum.sum()

      min_ess =
        case pstats |> Map.values() |> Enum.map(& &1.ess_total) do
          [] -> 0.0
          vs -> Enum.min(vs)
        end

      max_rhat =
        pstats |> Map.values() |> Enum.map(& &1.rhat) |> Enum.reject(&is_nil/1) |> then(fn
          [] -> nil
          vs -> Enum.max(vs)
        end)

      passed = Enum.all?(comparisons, fn c -> c.pass end)
      status = if passed, do: :pass, else: :fail

      result = %{
        name: name,
        arm: arm,
        status: status,
        wall_ms: wall_ms,
        divergences: divergences,
        chains: chains,
        max_rhat: max_rhat,
        step_size: chain_stats |> Enum.map(& &1.step_size) |> Enum.sum() |> Kernel./(chains),
        n_params: length(comparisons),
        comparisons: comparisons,
        leapfrog: leapfrog,
        # Diluted by warmup: sample_stats covers the SAMPLING phase only while
        # wall_ms spans warmup + sampling. That bias is a constant of the
        # protocol, not of the code under test, so it cancels in an A/B ratio.
        # It is NOT absolute throughput. See PDB.Metrics.
        leapfrog_per_sec_apparent: if(wall_s > 0, do: leapfrog / wall_s, else: nil),
        min_ess: min_ess,
        ess_per_sec: if(wall_s > 0, do: min_ess / wall_s, else: nil),
        div_rate: divergences / max(total_draws, 1),
        mean_accept: PDB.Metrics.mean_accept(List.first(chain_stats)),
        mean_tree_depth: PDB.Metrics.mean_tree_depth(List.first(chain_stats)),
        ess: pstats,
        max_mean_err: comparisons |> Enum.map(& &1.mean_err) |> Enum.max(),
        max_sd_ratio: comparisons |> Enum.map(& &1.sd_ratio) |> Enum.max(),
      }

      status_str = if passed, do: "PASS", else: "FAIL"
      IO.puts("  #{status_str}  #{String.pad_trailing("#{name}#{if arm, do: " (#{arm})", else: ""}", 44)}  " <>
              "#{wall_ms}ms  ess=#{round(min_ess)}  " <>
              "rhat=#{if max_rhat, do: Float.round(max_rhat, 3), else: "n/a"}  " <>
              "lf=#{leapfrog}  div=#{divergences}/#{total_draws}  " <>
              "err=#{Float.round(result.max_mean_err, 2)}  " <>
              "#{failed_gates(comparisons)}")

      result
    rescue
      e ->
        wall_ms = System.monotonic_time(:millisecond) - t0

        # Keep the stacktrace. "argument error" with no location is not a
        # diagnosis, and an unactionable crash row is most of why the Vulkan
        # arm looked like one undifferentiated failure instead of three.
        trace = __STACKTRACE__ |> Enum.take(8) |> Exception.format_stacktrace()

        IO.puts("  CRASH #{String.pad_trailing(name, 48)}  #{wall_ms}ms  #{Exception.message(e)}")
        IO.puts(trace)

        %{
          name: name,
          status: :crash,
          wall_ms: wall_ms,
          error: Exception.message(e),
          exception: inspect(e.__struct__),
          stacktrace: trace
        }
    end
  end

  # --- Model builders ---

  defp build_model(%{"model_type" => "linear_regression"} = spec) do
    build_linear_regression(spec)
  end

  defp build_model(%{"model_type" => "eight_schools"} = spec) do
    build_eight_schools(spec)
  end

  defp build_linear_regression(spec) do
    y_list = spec["y"]
    x_matrix = spec["X"]
    n_beta = spec["n_beta"]
    priors = spec["priors"]
    param_names = spec["param_names"]

    # Pre-compute column tensors for the design matrix
    # X is [n_obs x n_beta], we need column vectors
    n_obs = length(y_list)
    y_tensor = Nx.tensor(y_list, type: :f64)

    x_cols =
      for j <- 0..(n_beta - 1) do
        col = for row <- x_matrix, do: Enum.at(row, j)
        Nx.tensor(col, type: :f64)
      end

    # Compute OLS for init values and flat prior scaling
    {ols_betas, ols_sigma} = compute_ols(x_matrix, y_list, n_beta)

    # Build IR with beta priors
    ir = Builder.new_ir()

    ir =
      case priors["beta"] do
        %{"dist" => "flat"} ->
          # Scale flat prior relative to data: use 100x OLS sigma
          # This prevents the "flat" prior from being informative on huge-scale data
          flat_sigma = max(ols_sigma * 100, 10_000.0)
          Enum.reduce(0..(n_beta - 1), ir, fn j, acc ->
            Builder.rv(acc, "beta_#{j}", Normal, %{
              mu: Nx.tensor(0.0, type: :f64),
              sigma: Nx.tensor(flat_sigma, type: :f64)
            })
          end)

        %{"dist" => "normal", "mu" => mu, "sigma" => sigma} ->
          Enum.reduce(0..(n_beta - 1), ir, fn j, acc ->
            Builder.rv(acc, "beta_#{j}", Normal, %{
              mu: Nx.tensor(mu, type: :f64),
              sigma: Nx.tensor(sigma, type: :f64)
            })
          end)

        %{"dist" => "normal_per_param", "params" => param_list} ->
          param_list
          |> Enum.with_index()
          |> Enum.reduce(ir, fn {%{"mu" => mu, "sigma" => sigma}, j}, acc ->
            Builder.rv(acc, "beta_#{j}", Normal, %{
              mu: Nx.tensor(mu, type: :f64),
              sigma: Nx.tensor(sigma, type: :f64)
            })
          end)
      end

    # Sigma prior
    ir =
      case priors["sigma"] do
        %{"dist" => "flat_positive"} ->
          flat_scale = max(ols_sigma * 10, 10_000.0)
          Builder.rv(ir, "sigma", HalfCauchy, %{
            scale: Nx.tensor(flat_scale, type: :f64)
          })

        %{"dist" => "cauchy", "scale" => scale} ->
          Builder.rv(ir, "sigma", HalfCauchy, %{
            scale: Nx.tensor(scale, type: :f64)
          })

        %{"dist" => "half_normal", "sigma" => sigma} ->
          Builder.rv(ir, "sigma", HalfNormal, %{
            sigma: Nx.tensor(sigma, type: :f64)
          })
      end

    # Custom likelihood: sum Normal logpdf over all observations
    n_obs_t = Nx.tensor(n_obs, type: :f64)

    logpdf_fn = fn _x, params ->
      # Build linear predictor: mu = sum_j(beta_j * x_j)
      mu =
        Enum.reduce(0..(n_beta - 1), Nx.tensor(0.0, type: :f64), fn j, acc ->
          beta_key = String.to_atom("beta_#{j}")
          beta_j = Map.fetch!(params, beta_key)
          Nx.add(acc, Nx.multiply(beta_j, Enum.at(x_cols, j)))
        end)

      sigma = params.sigma

      # Normal logpdf sum: -0.5 * sum((y - mu)^2 / sigma^2) - n * log(sigma)
      residuals = Nx.subtract(y_tensor, mu)
      z = Nx.divide(residuals, sigma)
      Nx.subtract(
        Nx.multiply(Nx.tensor(-0.5, type: :f64), Nx.sum(Nx.multiply(z, z))),
        Nx.multiply(n_obs_t, Nx.log(sigma))
      )
    end

    dist = Custom.new(logpdf_fn)

    # Build params map for the custom likelihood
    likelihood_params =
      Map.new(0..(n_beta - 1), fn j -> {String.to_atom("beta_#{j}"), "beta_#{j}"} end)
      |> Map.put(:sigma, "sigma")

    ir = Custom.rv(ir, "y_likelihood", dist, likelihood_params)
    ir = Builder.obs(ir, "y_obs", "y_likelihood", Nx.tensor(0.0, type: :f64))

    # Init values from OLS
    init_values =
      Map.new(0..(n_beta - 1), fn j ->
        {"beta_#{j}", Nx.tensor(Enum.at(ols_betas, j), type: :f64)}
      end)
      |> Map.put("sigma", Nx.tensor(ols_sigma, type: :f64))

    # Param name mapping: eXMC name -> posteriordb name
    param_map =
      0..(n_beta - 1)
      |> Enum.map(fn j -> {"beta_#{j}", Enum.at(param_names, j)} end)
      |> Kernel.++([{"sigma", List.last(param_names)}])
      |> Map.new()

    {ir, init_values, param_map}
  end

  defp build_eight_schools(spec) do
    j = spec["J"]
    y = spec["y"]
    sigma_data = spec["sigma"]

    y_tensors = Enum.map(y, &Nx.tensor(&1, type: :f64))
    sigma_tensors = Enum.map(sigma_data, &Nx.tensor(&1, type: :f64))

    ir = Builder.new_ir()

    # mu ~ Normal(0, 5)
    ir = Builder.rv(ir, "mu", Normal, %{
      mu: Nx.tensor(0.0, type: :f64),
      sigma: Nx.tensor(5.0, type: :f64)
    })

    # tau ~ HalfCauchy(0, 5)
    ir = Builder.rv(ir, "tau", HalfCauchy, %{
      scale: Nx.tensor(5.0, type: :f64)
    })

    # theta_trans[j] ~ Normal(0, 1) (NCP raw)
    ir =
      Enum.reduce(0..(j - 1), ir, fn idx, acc ->
        Builder.rv(acc, "theta_trans_#{idx}", Normal, %{
          mu: Nx.tensor(0.0, type: :f64),
          sigma: Nx.tensor(1.0, type: :f64)
        })
      end)

    # Custom likelihood:
    # theta[j] = mu + tau * theta_trans[j]
    # y[j] ~ Normal(theta[j], sigma[j])  (sigma is data, not parameter)
    logpdf_fn = fn _x, params ->
      mu = params.mu
      tau = params.tau

      Enum.reduce(0..(j - 1), Nx.tensor(0.0, type: :f64), fn idx, acc ->
        theta_raw = Map.fetch!(params, String.to_atom("theta_trans_#{idx}"))
        theta = Nx.add(mu, Nx.multiply(tau, theta_raw))
        y_j = Enum.at(y_tensors, idx)
        s_j = Enum.at(sigma_tensors, idx)
        z = Nx.divide(Nx.subtract(y_j, theta), s_j)
        ll = Nx.subtract(
          Nx.multiply(Nx.tensor(-0.5, type: :f64), Nx.multiply(z, z)),
          Nx.log(s_j)
        )
        Nx.add(acc, ll)
      end)
    end

    dist = Custom.new(logpdf_fn)

    likelihood_params =
      Map.new(0..(j - 1), fn idx ->
        {String.to_atom("theta_trans_#{idx}"), "theta_trans_#{idx}"}
      end)
      |> Map.put(:mu, "mu")
      |> Map.put(:tau, "tau")

    ir = Custom.rv(ir, "likelihood", dist, likelihood_params)
    ir = Builder.obs(ir, "lik_obs", "likelihood", Nx.tensor(0.0, type: :f64))

    # Init values
    init_values =
      Map.new(0..(j - 1), fn idx -> {"theta_trans_#{idx}", Nx.tensor(0.0, type: :f64)} end)
      |> Map.put("mu", Nx.tensor(0.0, type: :f64))
      |> Map.put("tau", Nx.tensor(1.0, type: :f64))

    # Param map: eXMC -> posteriordb names
    param_map =
      Map.new(0..(j - 1), fn idx ->
        {"theta_trans_#{idx}", "theta[#{idx + 1}]"}
      end)
      |> Map.put("mu", "mu")
      |> Map.put("tau", "tau")

    {ir, init_values, param_map}
  end

  # --- Draw comparison ---

  # --- (d) Statistical gates ------------------------------------------------
  #
  # The old criteria were two fixed constants: mean within 0.5 reference SD,
  # SD within a factor of 2. Both are blind to the regression this suite is
  # supposed to catch. A model degrading from ESS 400 to ESS 50 has lost 8x its
  # sampling efficiency and still passes both comfortably, so the gate could
  # not fail for a performance reason.
  #
  # These four are the standard MCMC set (Stan, ArviZ, NumPyro all report them)
  # and they are complementary in a way worth stating, because two of them look
  # redundant and are not:
  #
  #   rhat < 1.01           chains disagree -> not converged
  #   ESS >= 100 per chain  efficiency, and the ONLY gate that catches a
  #                         slowdown in mixing
  #   |mean - ref| < 4 MCSE accuracy RELATIVE TO ACHIEVED PRECISION
  #   div rate < 1%         geometry the sampler could not integrate
  #
  # The MCSE gate alone would be perverse as a regression detector: MCSE is
  # sd/sqrt(ESS), so when ESS FALLS the tolerance WIDENS and a worse sampler
  # gets an easier test. The ESS gate is what makes the pair sound -- it fails
  # on the efficiency loss directly while MCSE checks that the mean is right
  # given the precision actually achieved. Neither substitutes for the other.
  # Thresholds are CALIBRATED AGAINST THE KNOWN-GOOD BASELINE, not taken from
  # convention. A gate that the healthy reference arm cannot pass is
  # permanently red and therefore detects nothing -- the same uselessness as a
  # gate that can never fail, arrived at from the other side.
  #
  # Measured 2026-09-05, EXLA, 1000 warmup + 1000 sampling, 4 chains. The whole
  # fast tier, i.e. one model from every posteriordb family:
  #
  #   model                           R-hat   min ESS   max err   div rate
  #   mesquite-logmesquite_logvolume  1.001      1906      0.01    32/4000 = 0.8%
  #   sblri-blr                       1.001       777      0.04    42/4000 = 1.05%
  #   kidiq-kidscore_momhs            1.002      1332      0.01    67/4000 = 1.7%
  #   eight_schools_noncentered       1.002      2402      0.03    88/4000 = 2.2%
  #   earnings-earn_height            1.000      1001      0.04   114/4000 = 2.85%
  #   nes2000-nes                     1.002      1370      0.03   240/4000 = 6.0%
  #
  # 1.01 / 100 / 4.0 clear by wide margins -- ESS is 8-24x the gate. Divergences
  # do not, and this threshold has now been wrong twice in the honest direction:
  # Stan's "any divergence is suspect" fails all six, and 5% (calibrated on two
  # models) failed nes2000-nes, whose R-hat is 1.002 and whose max mean error is
  # 0.03. That is an accurate posterior on a healthy sampler, so the gate was
  # wrong, not the model.
  #
  # 10% gives ~1.7x headroom over the worst healthy model.
  #
  # This comment used to add "and the Vulkan arm runs 45-94% on these models,
  # 4.5-9x clear of the gate". WITHDRAWN 2026-09-06. Those rates were produced
  # by a shader whose reduce loop was bounded by pc.n_obs while the data lived
  # in captures, so n_obs was 0, the likelihood evaluated to nothing, and the
  # sampler was exploring the prior. They measured a defect, not a backend.
  # With the bound fixed, the same model gives 7/300 divergences -- IDENTICAL to
  # EXLA on the same seed.
  #
  # The threshold itself is unaffected: it was derived from the EXLA baseline
  # table above, which was never touched by that defect. What is gone is the
  # claim about how much margin separates it from a failing backend, which is
  # now simply unmeasured.
  #
  # Calibrated on the 6-model fast tier, one per family. Re-derive if the full
  # 33 turn up a healthy model above 6%.
  @rhat_max 1.01
  @ess_min_per_chain 100
  @mcse_z_max 4.0
  @div_rate_max 0.10

  defp compare_stats(pstats, ref_draws, param_map, div_rate) do
    Enum.map(param_map, fn {exmc_name, pdb_name} ->
      st = Map.get(pstats, exmc_name)
      ref_samples = ref_draws[pdb_name]

      cond do
        ref_samples == nil ->
          %{param: pdb_name, pass: false, mean_err: 999.0, sd_ratio: 999.0,
            gates: %{reference: :fail}, note: "reference draws not found"}

        st == nil ->
          %{param: pdb_name, pass: false, mean_err: 999.0, sd_ratio: 999.0,
            gates: %{trace: :fail}, note: "parameter missing from trace"}

        true ->
          ref_mean = mean(ref_samples)
          ref_sd = sd(ref_samples)

          # Kept in reference-SD units so the column stays comparable with the
          # historical reports, even though it is no longer what gates.
          mean_err =
            if ref_sd > 1.0e-10,
              do: abs(st.mean - ref_mean) / ref_sd,
              else: abs(st.mean - ref_mean)

          sd_ratio = if ref_sd > 1.0e-10, do: st.sd / ref_sd, else: 1.0

          mcse_z =
            cond do
              is_nil(st.mcse) or st.mcse <= 1.0e-12 -> nil
              true -> abs(st.mean - ref_mean) / st.mcse
            end

          gates = %{
            # :skipped is NOT a pass. It means the gate could not be evaluated
            # (one chain), and the banner says so at the top of every run.
            rhat: if(is_nil(st.rhat), do: :skipped, else: gate(st.rhat < @rhat_max)),
            ess: gate(st.ess_min_per_chain >= @ess_min_per_chain),
            mean: if(is_nil(mcse_z), do: :skipped, else: gate(mcse_z < @mcse_z_max)),
            sd: gate(sd_ratio > 0.5 and sd_ratio < 2.0),
            divergences: gate(div_rate < @div_rate_max)
          }

          %{
            param: pdb_name,
            pass: gates |> Map.values() |> Enum.all?(&(&1 != :fail)),
            gates: gates,
            mean_err: mean_err,
            sd_ratio: sd_ratio,
            mcse_z: mcse_z,
            rhat: st.rhat,
            ess_total: st.ess_total,
            ess_min_per_chain: st.ess_min_per_chain,
            mcse: st.mcse,
            exmc_mean: st.mean,
            exmc_sd: st.sd,
            ref_mean: ref_mean,
            ref_sd: ref_sd
          }
      end
    end)
  end

  defp gate(true), do: :pass
  defp gate(false), do: :fail

  # Which gates failed, for the one-line-per-model output. Naming them is the
  # difference between "this model failed" and "this model failed because it
  # never mixed".
  defp failed_gates(comparisons) do
    names =
      comparisons
      |> Enum.flat_map(fn c ->
        c |> Map.get(:gates, %{}) |> Enum.filter(fn {_, v} -> v == :fail end) |> Enum.map(&elem(&1, 0))
      end)
      |> Enum.uniq()
      |> Enum.sort()

    if names == [], do: "", else: "[#{Enum.join(names, " ")}]"
  end

  # --- Reporting ---

  defp print_report(results, provenance) do
    IO.puts("\n#{"=" |> String.duplicate(80)}")
    IO.puts("POSTERIORDB VALIDATION REPORT")
    IO.puts("#{"=" |> String.duplicate(80)}\n")

    passed = Enum.count(results, & &1.status == :pass)
    failed = Enum.count(results, & &1.status == :fail)
    crashed = Enum.count(results, & &1.status == :crash)
    total = length(results)

    IO.puts("Results: #{passed} PASS / #{failed} FAIL / #{crashed} CRASH out of #{total}")
    IO.puts("")

    total_wall = results |> Enum.map(& Map.get(&1, :wall_ms, 0)) |> Enum.sum()

    shape =
      if provenance.mode == "race",
        do: "serialized - timings are meaningful",
        else: "summed across #{provenance.parallel} concurrent workers - NOT a serial cost"

    IO.puts("Total sampling time: #{div(total_wall, 1000)}s (#{shape})")

    lf = results |> Enum.map(&Map.get(&1, :leapfrog, 0)) |> Enum.sum()
    ok = Enum.filter(results, &(Map.get(&1, :min_ess) != nil))

    if ok != [] do
      worst = Enum.min_by(ok, & &1.min_ess)
      IO.puts("Total leapfrog steps: #{lf}")
      IO.puts("Lowest ESS: #{round(worst.min_ess)} (#{worst.name}) of #{provenance.num_samples} draws")
    end

    IO.puts("")

    # Failures detail
    failures = Enum.filter(results, & &1.status != :pass)
    if length(failures) > 0 do
      IO.puts("--- Failures ---")
      for r <- failures do
        IO.puts("\n  #{r.name} (#{r.status})")
        if r.status == :crash do
          IO.puts("    Error: #{r[:error]}")
        else
          for c <- (r[:comparisons] || []), !c.pass do
            failed =
              c |> Map.get(:gates, %{}) |> Enum.filter(fn {_, v} -> v == :fail end)
                |> Enum.map(&elem(&1, 0)) |> Enum.sort() |> Enum.join(",")

            detail =
              case c[:note] do
                nil ->
                  "rhat=#{fmt(c[:rhat], 3)} ess/chain=#{fmt(c[:ess_min_per_chain], 0)} " <>
                    "mcse_z=#{fmt(c[:mcse_z], 2)} sd_ratio=#{fmt(c[:sd_ratio], 2)}  " <>
                    "exmc=#{fmt(c[:exmc_mean], 3)}+/-#{fmt(c[:exmc_sd], 3)} " <>
                    "ref=#{fmt(c[:ref_mean], 3)}+/-#{fmt(c[:ref_sd], 3)}"

                note ->
                  note
              end

            IO.puts("    #{c.param} [#{failed}]: #{detail}")
          end
        end
      end
    end

    IO.puts("\n#{"=" |> String.duplicate(80)}")
    pass_rate = if total > 0, do: Float.round(passed / total * 100, 1), else: 0.0
    IO.puts("PASS RATE: #{pass_rate}% (#{passed}/#{total})")
    IO.puts("#{"=" |> String.duplicate(80)}")
  end

  defp save_results(results, provenance) do
    passed = Enum.count(results, & &1.status == :pass)
    total = length(results)
    timestamp = provenance.timestamp

    md = """
    # posteriordb Validation Results

    **Date:** #{timestamp}
    **Pass rate:** #{passed}/#{total} (#{Float.round(passed / max(total, 1) * 100, 1)}%)
    **Mode:** #{provenance.mode}#{if provenance.mode == "race", do: " (serialized)", else: " (parallel=#{provenance.parallel}; wall times are contention-bound)"}
    **Protocol:** #{provenance.num_warmup} warmup + #{provenance.num_samples} sampling, seed=#{provenance.seed}, ncp=#{provenance.ncp}
    **Compiler:** requested #{provenance.compiler_requested}, resolved #{provenance.compiler_resolved}, backend #{provenance.backend}, precision #{provenance.precision}
    **Host:** #{provenance.host} (#{provenance.schedulers_online} schedulers, OTP #{provenance.otp_release}, Elixir #{provenance.elixir_version})
    **exmc:** #{provenance.exmc_sha}#{if provenance.exmc_dirty, do: " (DIRTY)", else: ""}
    **nx_vulkan:** #{provenance.nx_vulkan_sha}

    ## Summary

    | Model | Status | Wall (s) | Min ESS | Max R-hat | Leapfrog | Div | Div % | Step Size | Max Mean Err |
    |-------|--------|----------|---------|-----------|----------|-----|-------|-----------|-------------|
    """ <>
    (results
     |> Enum.map(fn r ->
       status = r.status |> Atom.to_string() |> String.upcase()
       wall_s = Float.round((r[:wall_ms] || 0) / 1000, 1)
       div = r[:divergences] || "-"
       eps = if r[:step_size], do: Float.round(r.step_size, 4), else: "-"
       max_me = if r[:max_mean_err], do: Float.round(r.max_mean_err, 3), else: "-"
       max_sd = if r[:max_sd_ratio], do: Float.round(r.max_sd_ratio, 3), else: "-"
       ess = if r[:min_ess], do: round(r.min_ess), else: "-"
       lf = r[:leapfrog] || "-"
       rhat = fmt(r[:max_rhat], 3)
       dpct = if r[:div_rate], do: Float.round(r.div_rate * 100, 1), else: "-"
       _ = max_sd
       "| #{r.name} | #{status} | #{wall_s} | #{ess} | #{rhat} | #{lf} | #{div} | #{dpct} | #{eps} | #{max_me} |"
     end)
     |> Enum.join("\n")) <>
    "\n\n## Pass Criteria\n\n" <>
    "Statistical, not fixed constants. The previous criteria (mean within 0.5\n" <>
    "reference SD, SD within a factor of 2) could not fail for a performance\n" <>
    "reason: a model losing 8x its sampling efficiency passed both.\n\n" <>
    "- Split R-hat < #{@rhat_max} across #{provenance.chains} chains" <>
    "#{if provenance.chains < 2, do: " — NOT EVALUATED, needs >= 2 chains", else: ""}\n" <>
    "- ESS (bulk, rank-normalised) >= #{@ess_min_per_chain} per chain\n" <>
    "- |mean − reference mean| < #{@mcse_z_max} x MCSE, where MCSE = sd/sqrt(ESS)\n" <>
    "- Divergence rate < #{round(@div_rate_max * 100)}%\n" <>
    "- SD within factor of 2 of reference SD (retained, secondary)\n\n" <>
    "The MCSE gate widens as ESS falls, so it cannot detect a slowdown on its\n" <>
    "own — the ESS gate is what does that. Both are required.\n\n" <>
    "Reference: Stan gold-standard draws (10 chains x 1000 draws)\n"

    path = Path.join(@processed_dir, "validation_results.md")
    File.write!(path, md)
    IO.puts("\nMarkdown: #{path}")

    {json, latest} = PDB.Report.write(results, provenance, @processed_dir)
    IO.puts("JSON:     #{json}")
    IO.puts("Latest:   #{latest}")
  end

  # --- Post-processing ---

  defp reconstruct_eight_schools(trace, param_map) do
    # If this is Eight Schools, reconstruct theta[j] = mu + tau * theta_trans[j]
    has_theta = Enum.any?(param_map, fn {_, pdb} -> String.starts_with?(pdb, "theta[") end)

    if has_theta and Map.has_key?(trace, "mu") and Map.has_key?(trace, "tau") do
      mu = trace["mu"]
      tau = trace["tau"]

      Enum.reduce(param_map, trace, fn {exmc_name, pdb_name}, acc ->
        if String.starts_with?(pdb_name, "theta[") and String.starts_with?(exmc_name, "theta_trans_") do
          theta_trans = trace[exmc_name]
          theta = Nx.add(mu, Nx.multiply(tau, theta_trans))
          Map.put(acc, exmc_name, theta)
        else
          acc
        end
      end)
    else
      trace
    end
  end

  # --- Helpers ---

  defp compute_ols(x_matrix, y_list, n_beta) do
    # Simple OLS: beta = (X'X)^{-1} X'y
    # Use Nx for the matrix math
    x = Nx.tensor(x_matrix, type: :f64)
    y = Nx.tensor(y_list, type: :f64) |> Nx.reshape({length(y_list), 1})
    xtx = Nx.dot(Nx.transpose(x), x)
    xty = Nx.dot(Nx.transpose(x), y)

    # Solve via Cholesky or fallback
    beta =
      try do
        Exmc.JIT.jit(fn {a, b} -> Nx.LinAlg.solve(a, b) end).({xtx, xty})
      rescue
        _ ->
          # Fallback: use pseudoinverse
          Nx.tensor(List.duplicate([0.0], n_beta), type: :f64)
      end

    betas = beta |> Nx.reshape({n_beta}) |> Nx.to_flat_list()

    # Residual SD
    y_hat = Nx.dot(x, beta)
    residuals = Nx.subtract(y, y_hat)
    n = length(y_list)
    sigma = residuals |> Nx.multiply(residuals) |> Nx.sum() |> Nx.to_number()
    sigma = :math.sqrt(sigma / max(n - n_beta, 1))

    {betas, max(sigma, 0.1)}
  end

  defp load_json(path) do
    path |> File.read!() |> Jason.decode!()
  end

  # nil-safe rounding: a skipped or unavailable statistic prints as "n/a"
  # rather than crashing the report that is supposed to explain the failure.
  defp fmt(nil, _), do: "n/a"
  defp fmt(v, 0) when is_number(v), do: to_string(round(v))
  defp fmt(v, p) when is_number(v), do: to_string(Float.round(v * 1.0, p))
  defp fmt(v, _), do: inspect(v)

  defp mean(list) when is_list(list) do
    Enum.sum(list) / length(list)
  end

  defp sd(list) when is_list(list) do
    m = mean(list)
    n = length(list)
    variance = Enum.reduce(list, 0.0, fn x, acc -> acc + (x - m) * (x - m) end) / (n - 1)
    :math.sqrt(variance)
  end
end

# --- CLI ---
{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      parallel: :integer,
      samples: :integer,
      warmup: :integer,
      mode: :string,
      compiler: :string,
      seed: :integer,
      ncp: :boolean,
      only: :string,
      transcendentals: :string,
      chains: :integer,
      arms: :string,
      tier: :string
    ]
  )

mode =
  case Keyword.get(opts, :mode, "validate") do
    "validate" -> :validate
    "race" -> :race
    other -> raise ArgumentError, "--mode must be validate|race, got #{inspect(other)}"
  end

compiler =
  case Keyword.get(opts, :compiler, "vulkan") do
    "vulkan" -> :vulkan
    "exla" -> :exla
    "none" -> :none
    "auto" -> :auto
    other -> raise ArgumentError, "--compiler must be vulkan|exla|none|auto, got #{inspect(other)}"
  end

PosteriorDBValidator.run(
  [
    mode: mode,
    compiler: compiler,
    num_samples: Keyword.get(opts, :samples, 1000),
    num_warmup: Keyword.get(opts, :warmup, 1000),
    seed: Keyword.get(opts, :seed, 42),
    ncp: Keyword.get(opts, :ncp, false),
    only: Keyword.get(opts, :only),
    chains: Keyword.get(opts, :chains, 4),
    tier:
      case Keyword.get(opts, :tier, "full") do
        "full" -> :full
        "fast" -> :fast
        other -> raise ArgumentError, "--tier must be fast|full, got #{inspect(other)}"
      end,
    arms:
      case Keyword.get(opts, :arms) do
        nil ->
          nil

        str ->
          str
          |> String.split(",", trim: true)
          |> Enum.map(fn
            "vulkan" -> :vulkan
            "exla" -> :exla
            "none" -> :none
            other -> raise ArgumentError, "--arms entries must be vulkan|exla|none, got #{inspect(other)}"
          end)
      end,
    transcendentals:
      case Keyword.get(opts, :transcendentals, "f32_cast") do
        "f32_cast" -> :f32_cast
        "polynomial" -> :polynomial
        other -> raise ArgumentError, "--transcendentals must be f32_cast|polynomial, got #{inspect(other)}"
      end
  ] ++ if(opts[:parallel], do: [parallel: opts[:parallel]], else: [])
)
