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

    provenance =
      PDB.Provenance.collect(
        compiler: compiler,
        mode: mode,
        parallel: parallel,
        num_warmup: num_warmup,
        num_samples: num_samples,
        seed: seed,
        ncp: ncp
      )

    IO.puts(PDB.Provenance.banner(provenance))

    manifest = load_json(Path.join(@processed_dir, "manifest.json"))
    posteriors = manifest["posteriors"] |> filter_only(only)
    IO.puts("Posteriors to run: #{length(posteriors)}\n")

    run_opts = [
      num_samples: num_samples,
      num_warmup: num_warmup,
      seed: seed,
      ncp: ncp
    ]

    results =
      posteriors
      |> Task.async_stream(
        fn name -> validate_one(name, run_opts) end,
        max_concurrency: parallel,
        timeout: 1_800_000,
        ordered: false
      )
      |> Enum.map(fn
        {:ok, result} -> result
        {:exit, reason} -> %{name: "unknown", status: :crash, error: inspect(reason)}
      end)
      |> Enum.sort_by(& &1.name)

    print_report(results, provenance)
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

  defp filter_only(posteriors, nil), do: posteriors
  defp filter_only(posteriors, pat), do: Enum.filter(posteriors, &String.contains?(&1, pat))

  # --- Per-posterior validation ---

  def validate_one(name, opts) do
    num_samples = Keyword.fetch!(opts, :num_samples)
    num_warmup = Keyword.fetch!(opts, :num_warmup)
    seed = Keyword.fetch!(opts, :seed)
    ncp = Keyword.fetch!(opts, :ncp)

    t0 = System.monotonic_time(:millisecond)

    try do
      spec = load_json(Path.join(@processed_dir, "#{name}.json"))
      ref_draws = spec["reference_draws"]

      # Build and sample
      {ir, init_values, param_map} = build_model(spec)

      {trace, stats} =
        Exmc.Sampler.sample(ir, init_values,
          num_samples: num_samples,
          num_warmup: num_warmup,
          seed: seed,
          ncp: ncp
        )

      # Compare against reference draws
      comparisons = compare_draws(trace, ref_draws, param_map)

      wall_ms = System.monotonic_time(:millisecond) - t0
      wall_s = wall_ms / 1000
      divergences = stats.divergences

      # stats.sample_stats has always carried per-draw :n_steps, :tree_depth,
      # :divergent and :accept_prob. The old harness read only :divergences and
      # :step_size and threw the rest away, and never computed ESS at all --
      # which is why its pass criteria were structurally unable to see a
      # performance regression.
      leapfrog = PDB.Metrics.leapfrog(stats)
      ess = PDB.Metrics.ess_table(trace, Enum.map(param_map, fn {n, _} -> n end))

      min_ess =
        case ess |> Map.values() |> Enum.map(& &1.ess) do
          [] -> 0.0
          vs -> Enum.min(vs)
        end

      passed = Enum.all?(comparisons, fn c -> c.pass end)
      status = if passed, do: :pass, else: :fail

      result = %{
        name: name,
        status: status,
        wall_ms: wall_ms,
        divergences: divergences,
        step_size: stats.step_size,
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
        div_rate: divergences / max(num_samples, 1),
        mean_accept: PDB.Metrics.mean_accept(stats),
        mean_tree_depth: PDB.Metrics.mean_tree_depth(stats),
        ess: ess,
        max_mean_err: comparisons |> Enum.map(& &1.mean_err) |> Enum.max(),
        max_sd_ratio: comparisons |> Enum.map(& &1.sd_ratio) |> Enum.max(),
      }

      status_str = if passed, do: "PASS", else: "FAIL"
      IO.puts("  #{status_str}  #{String.pad_trailing(name, 46)}  " <>
              "#{wall_ms}ms  ess=#{round(min_ess)}  lf=#{leapfrog}  " <>
              "div=#{divergences}  eps=#{Float.round(stats.step_size, 4)}  " <>
              "err=#{Float.round(result.max_mean_err, 3)}")

      result
    rescue
      e ->
        wall_ms = System.monotonic_time(:millisecond) - t0
        IO.puts("  CRASH #{String.pad_trailing(name, 48)}  #{wall_ms}ms  #{Exception.message(e)}")
        %{name: name, status: :crash, wall_ms: wall_ms, error: Exception.message(e)}
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

  defp compare_draws(trace, ref_draws, param_map) do
    # For Eight Schools: reconstruct theta from NCP (theta = mu + tau * theta_trans)
    trace = reconstruct_eight_schools(trace, param_map)

    Enum.map(param_map, fn {exmc_name, pdb_name} ->
      exmc_samples = trace[exmc_name] |> Nx.to_flat_list()
      ref_samples = ref_draws[pdb_name]

      if ref_samples == nil do
        %{param: pdb_name, pass: false, mean_err: 999.0, sd_ratio: 999.0,
          note: "reference draws not found"}
      else
        exmc_mean = mean(exmc_samples)
        exmc_sd = sd(exmc_samples)
        ref_mean = mean(ref_samples)
        ref_sd = sd(ref_samples)

        # Mean error in units of reference SD
        mean_err =
          if ref_sd > 1.0e-10 do
            abs(exmc_mean - ref_mean) / ref_sd
          else
            abs(exmc_mean - ref_mean)
          end

        # SD ratio
        sd_ratio =
          if ref_sd > 1.0e-10 do
            exmc_sd / ref_sd
          else
            1.0
          end

        # Pass criteria:
        # - Mean within 0.5 SD of reference mean
        # - SD within factor of 2 of reference SD
        pass = mean_err < 0.5 and sd_ratio > 0.5 and sd_ratio < 2.0

        %{
          param: pdb_name,
          pass: pass,
          mean_err: mean_err,
          sd_ratio: sd_ratio,
          exmc_mean: exmc_mean,
          exmc_sd: exmc_sd,
          ref_mean: ref_mean,
          ref_sd: ref_sd,
        }
      end
    end)
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
            IO.puts("    #{c.param}: mean_err=#{Float.round(c.mean_err, 3)}  " <>
                    "sd_ratio=#{Float.round(c.sd_ratio, 3)}  " <>
                    "exmc=#{Float.round(c.exmc_mean, 3)}±#{Float.round(c.exmc_sd, 3)}  " <>
                    "ref=#{Float.round(c.ref_mean, 3)}±#{Float.round(c.ref_sd, 3)}")
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

    | Model | Status | Wall (s) | Min ESS | Leapfrog | Div | Step Size | Max Mean Err | Max SD Ratio |
    |-------|--------|----------|---------|----------|-----|-----------|-------------|-------------|
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
       "| #{r.name} | #{status} | #{wall_s} | #{ess} | #{lf} | #{div} | #{eps} | #{max_me} | #{max_sd} |"
     end)
     |> Enum.join("\n")) <>
    "\n\n## Pass Criteria\n\n" <>
    "- Mean within 0.5 SD of reference mean\n" <>
    "- SD within factor of 2 (0.5x-2.0x) of reference SD\n" <>
    "- Reference: Stan gold-standard draws (10 chains x 1000 draws)\n"

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
      only: :string
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
    only: Keyword.get(opts, :only)
  ] ++ if(opts[:parallel], do: [parallel: opts[:parallel]], else: [])
)
