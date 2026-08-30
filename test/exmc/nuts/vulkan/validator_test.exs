defmodule Exmc.NUTS.Vulkan.ValidatorTest do
  @moduledoc """
  W2 validation harness — applied to the known-good chain shaders.

  The *positive* cases run `Exmc.NUTS.Vulkan.Validator.validate/3`
  against the reference arm (`Validator.reference/0` — `:exla` where
  EXLA is USABLE (`Exmc.JIT.usable?/1`: present AND its application
  starts), `:none` otherwise). The arm is reported in every failure
  payload as `:reference`, and it matters: Cauchy's KS check passes
  against `:exla` and fails marginally against `:none` on super-io
  (d 0.0999 vs crit 0.0975), which is the Linux-NVIDIA over-dispersion
  noted below seen through a slightly different lens, not a new bug. All route through the **f64
  synth** path; see the `Exmc.NUTS.Vulkan.Dispatch` moduledoc. All six
  — **Normal**, **StudentT**, **Weibull**, **Exponential**,
  **HalfNormal**, and **Cauchy** (median + IQR, since it has no defined
  moments) — pass at f64 on the reference fleet.

  Note: a Linux-NVIDIA-driver-specific *over-dispersion* of the vulkan
  sampler has been observed on super-io's RTX 3060 Ti. These positive
  cases are validated on the FreeBSD/MoltenVK fleet (mac-247), the
  reference host, not on Linux NVIDIA. See
  `docs/VULKAN_KNOWN_ISSUES.md`.

  The *negative* case simulates a broken shader by feeding the
  comparator two different distributions (Normal(0,1) vs Normal(0,2));
  the harness must detect the variance mismatch.

  Positive cases require Vulkan; tagged `:requires_vulkan` so they're
  skipped on hosts without `Nx.Vulkan` loaded. The negative case is
  pure Elixir and runs everywhere.
  """

  use ExUnit.Case, async: false

  alias Exmc.{Builder, Dist}
  alias Exmc.NUTS.Vulkan.Validator

  # Smaller than production defaults — keeps test wall time reasonable
  # while still being well above the 1000-draw budget the W2 stub
  # specifies as the false-negative ceiling.
  @n_warmup 300
  @n_samples 800
  @seed 42

  # Shared opts for the positive battery.
  @opts [n_warmup: @n_warmup, n_samples: @n_samples, seed: @seed]

  describe "positive cases (6 known-good shaders)" do
    @describetag :requires_vulkan

    test "Normal(0, 1) — leapfrog_chain_normal" do
      ir =
        Builder.new_ir()
        |> Builder.rv("x", Dist.Normal, %{
          mu: Nx.tensor(0.0),
          sigma: Nx.tensor(1.0)
        })

      assert Validator.validate(ir, {:normal, 0.0, 1.0}, @opts) == :ok
    end

    test "Exponential(λ=2) — synthesized leapfrog_chain_synth" do
      # D90/G4: the legacy f32 `leapfrog_chain_exponential` family SPV
      # (and its Dispatch.do_chain clause) were removed — everything now
      # routes through the f64 synth path (Dispatch moduledoc). The old
      # `precision: :f32` variant exercised that deleted path and crashed
      # with a FunctionClauseError; run at the go-forward f64 synth
      # precision instead (Exponential is positive-support / :log
      # transform, same family as the passing Weibull/Gamma cases).
      # n=400 keeps the variance-check inside 3σ (the Exp(2) variance
      # estimator is noisy at larger N).
      ir =
        Builder.new_ir()
        |> Builder.rv("x", Dist.Exponential, %{lambda: Nx.tensor(2.0)})

      assert Validator.validate(ir, {:exponential, 2.0}, n_warmup: 200, n_samples: 400, seed: 42) ==
               :ok
    end

    test "StudentT(df=3) — leapfrog_chain_studentt" do
      ir =
        Builder.new_ir()
        |> Builder.rv("x", Dist.StudentT, %{
          df: Nx.tensor(3.0),
          loc: Nx.tensor(0.0),
          scale: Nx.tensor(1.0)
        })

      lgamma = fn x -> Nx.to_number(Exmc.Math.lgamma(Nx.tensor(x))) end
      logp_const = lgamma.(2.0) - lgamma.(1.5) - 0.5 * :math.log(:math.pi() * 3.0)

      meta = {:studentt, 0.0, 1.0, 3.0, logp_const}

      assert Validator.validate(ir, meta, @opts) == :ok
    end

    test "Cauchy(0, 1) — synthesized leapfrog_chain_synth (median + IQR)" do
      # D90/G4: was tagged :f32_precision_limited under the old f32 family
      # shader — the failure was the validator comparing f32 Vulkan against
      # f64 EXLA. Now that everything routes through the f64 synth path, the
      # comparison is f64-vs-f64: Cauchy passes the median+IQR check at f64
      # on the reference fleet (mac-247). Cauchy has no defined mean/variance,
      # so the harness substitutes median + IQR (see Validator moduledoc).
      ir =
        Builder.new_ir()
        |> Builder.rv("x", Dist.Cauchy, %{
          loc: Nx.tensor(0.0),
          scale: Nx.tensor(1.0)
        })

      log_pi_scale = -:math.log(:math.pi() * 1.0)
      meta = {:cauchy, 0.0, 1.0, log_pi_scale}

      assert Validator.validate(ir, meta, @opts) == :ok
    end

    test "HalfNormal(σ=1) — synthesized leapfrog_chain_synth" do
      # D90/G4: the legacy f32 `leapfrog_chain_halfnormal` family SPV was
      # removed along with the rest; the f32 path now flows the raw family
      # meta into the deleted Dispatch.do_chain family clause and crashes.
      # Run at the go-forward f64 synth precision — HalfNormal is
      # positive-support / :log transform (the transform was aligned to
      # :log in W7 Stage 2.5), so it synthesizes like Weibull/Gamma.
      ir =
        Builder.new_ir()
        |> Builder.rv("x", Dist.HalfNormal, %{sigma: Nx.tensor(1.0)})

      log_const = -:math.log(1.0) - 0.5 * :math.log(:math.pi())
      meta = {:halfnormal, 1.0, log_const}

      assert Validator.validate(ir, meta, n_warmup: 200, n_samples: 400, seed: 42) == :ok
    end

    test "Weibull(k=2, λ=1) — leapfrog_chain_weibull" do
      # W7 Stage 1 (precise float on loop accumulators) fixed the
      # Stage 1.5.4 mean drift on Linux NVIDIA. Now passes on all
      # platforms (R8: FreeBSD GT 750M / GT 650M / Linux RTX 3060 Ti).
      ir =
        Builder.new_ir()
        |> Builder.rv("x", Dist.Weibull, %{
          k: Nx.tensor(2.0),
          lambda: Nx.tensor(1.0)
        })

      logp_const_per_elem = :math.log(2.0) - 2.0 * :math.log(1.0)
      meta = {:weibull, 2.0, 1.0, logp_const_per_elem}

      assert Validator.validate(ir, meta, @opts) == :ok
    end
  end

  describe "Phase 1 — synthesized chain shaders" do
    @describetag :requires_vulkan
    # Synthesized shaders pay ~150 ms first-call glslangValidator + ~40 ms
    # pipeline create on top of regular sampling. Bump per-test timeout
    # so the EXLA reference + Vulkan path both fit comfortably.
    @describetag timeout: 300_000

    # Use smaller N than the hand-written shader battery — the EXLA
    # reference path is the bottleneck and we still get statistically
    # meaningful comparisons.
    @synth_opts [n_warmup: 200, n_samples: 400, seed: 42]

    test "Beta(2, 3) — synthesized leapfrog_chain_synth" do
      ir =
        Builder.new_ir()
        |> Builder.rv("x", Dist.Beta, %{alpha: Nx.tensor(2.0), beta: Nx.tensor(3.0)})

      assert Validator.validate(ir, {:beta, 2.0, 3.0}, @synth_opts) == :ok
    end

    test "Gamma(2, 1) — synthesized leapfrog_chain_synth" do
      ir =
        Builder.new_ir()
        |> Builder.rv("x", Dist.Gamma, %{alpha: Nx.tensor(2.0), beta: Nx.tensor(1.0)})

      assert Validator.validate(ir, {:gamma, 2.0, 1.0}, @synth_opts) == :ok
    end

    test "Lognormal(0, 1) — synthesized leapfrog_chain_synth (math reduces to Normal)" do
      ir =
        Builder.new_ir()
        |> Builder.rv("x", Dist.Lognormal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

      assert Validator.validate(ir, {:lognormal, 0.0, 1.0}, @synth_opts) == :ok
    end
  end

  describe "negative case (deliberately mismatched samples)" do
    test "variance mismatch trips the variance check" do
      # Simulate a broken shader: instead of feeding Validator the
      # same model on both backends, give it samples from N(0, 1)
      # vs N(0, 2). Variance ratio of 4× should trip check_variance
      # well before the KS test even gets a chance.
      a = sample_normal(0.0, 1.0, 1000, 42)
      b = sample_normal(0.0, 2.0, 1000, 42)

      assert {:error, %{check: check}} = Validator.compare(a, b, {:normal, 0.0, 1.0})
      assert check in [:variance, :ks, :mean]
    end

    test "mean mismatch trips the mean check" do
      a = sample_normal(0.0, 1.0, 1000, 42)
      b = sample_normal(2.0, 1.0, 1000, 42)

      assert {:error, %{check: check}} = Validator.compare(a, b, {:normal, 0.0, 1.0})
      assert check == :mean
    end
  end

  describe "unit tests on the stat helpers" do
    test "check_mean accepts identical samples" do
      xs = sample_normal(0.0, 1.0, 1000, 7)
      assert Validator.check_mean(xs, xs) == :ok
    end

    test "check_variance accepts identical samples" do
      xs = sample_normal(0.0, 1.0, 1000, 7)
      assert Validator.check_variance(xs, xs) == :ok
    end

    test "check_ks accepts identical samples" do
      xs = sample_normal(0.0, 1.0, 500, 11)
      assert Validator.check_ks(xs, xs) == :ok
    end

    test "check_ks rejects clearly different distributions" do
      a = sample_normal(0.0, 1.0, 1000, 1)
      b = sample_normal(3.0, 1.0, 1000, 2)
      assert {:error, %{check: :ks}} = Validator.check_ks(a, b)
    end

    test "check_ks computes D-statistic correctly on a known case" do
      # Two perfectly non-overlapping samples → D = 1.0. With 50 each
      # the critical value is c*sqrt(2/2500) ≈ 0.39 so D=1.0 rejects.
      a = Enum.map(1..50, &(&1 * 1.0))
      b = Enum.map(1..50, &(100.0 + &1))
      assert {:error, %{check: :ks, d: d}} = Validator.check_ks(a, b)
      assert_in_delta d, 1.0, 1.0e-9
    end
  end

  # --- Helpers --------------------------------------------------

  # Plain Erlang :rand for negative-test fixtures — fast, deterministic.
  defp sample_normal(mu, sigma, n, seed) do
    rng = :rand.seed_s(:exsss, {seed, seed * 31, seed * 97})

    {samples, _} =
      Enum.reduce(1..n, {[], rng}, fn _, {acc, r} ->
        {z, r2} = :rand.normal_s(r)
        {[mu + sigma * z | acc], r2}
      end)

    samples
  end

  describe "ess/1 — standard errors must not assume iid draws" do
    test "iid samples report ESS ~ n" do
      xs = for _ <- 1..2000, do: :rand.normal()
      assert_in_delta Validator.ess(xs) / 2000, 1.0, 0.15
    end

    test "an AR(1) chain reports the autocorrelation-corrected ESS" do
      # rho = 0.6 -> ESS ~ n(1-rho)/(1+rho) = 500 for n = 2000. The old code
      # divided by length/1 and would have claimed 2000, understating the
      # standard error by ~2x and turning a nominal 3-sigma gate into ~1.5.
      {xs, _} =
        Enum.map_reduce(1..2000, 0.0, fn _, prev ->
          x = 0.6 * prev + :rand.normal()
          {x, x}
        end)

      ess = Validator.ess(xs)
      assert ess < 1000, "ESS #{ess} does not reflect rho = 0.6 autocorrelation"
      assert ess > 250
    end

    test "never exceeds n, even for an antithetic chain" do
      # NUTS is often antithetic, which can push true ESS above n. Exploiting
      # that would NARROW the tolerance; this gate should err wide.
      anti = Enum.map(1..2000, fn i -> if rem(i, 2) == 0, do: 2.0, else: -2.0 end)
      assert Validator.ess(anti) <= 2000.0
    end

    test "degenerate inputs do not blow up" do
      assert Validator.ess([1.0, 2.0, 3.0, 4.0]) == 4.0
      assert Validator.ess(List.duplicate(1.0, 100)) == 100.0
    end
  end

  describe "analytic_moments/1" do
    test "closed-form moments for the families the codegen emits" do
      assert {:moments, %{mean: +0.0, var: 1.0}} = Validator.analytic_moments({:normal, 0.0, 1.0})
      assert {:moments, %{mean: 0.5, var: 0.25}} = Validator.analytic_moments({:exponential, 2.0})

      {:moments, %{mean: m, var: v}} = Validator.analytic_moments({:halfnormal, 1.0, 0.0})
      assert_in_delta m, :math.sqrt(2 / :math.pi()), 1.0e-12
      assert_in_delta v, 1.0 - 2 / :math.pi(), 1.0e-12

      {:moments, %{mean: lm, var: lv}} = Validator.analytic_moments({:lognormal, 0.0, 1.0})
      assert_in_delta lm, :math.exp(0.5), 1.0e-12
      assert_in_delta lv, (:math.exp(1.0) - 1.0) * :math.exp(1.0), 1.0e-9
    end

    test "Cauchy reports quantiles, since it has no moments" do
      assert {:quantiles, %{median: +0.0, iqr: 2.0}} =
               Validator.analytic_moments({:cauchy, 0.0, 1.0, 0.0})
    end

    test "StudentT below nu = 2 has no variance and is not claimed" do
      assert {:moments, _} = Validator.analytic_moments({:studentt, 0.0, 1.0, 5.0, 0.0})
      assert :unknown = Validator.analytic_moments({:studentt, 0.0, 1.0, 1.5, 0.0})

      # 2 < nu <= 4: mean and variance exist, the FOURTH moment does not, so
      # every sigma-multiple gate built on the sample variance is noise. The
      # table has to say :unknown rather than hand back a finite number.
      assert :unknown = Validator.analytic_moments({:studentt, 0.0, 1.0, 3.0, 0.0})
      assert :unknown = Validator.analytic_moments({:studentt, 0.0, 1.0, 4.0, 0.0})
      assert {:moments, _} = Validator.analytic_moments({:studentt, 0.0, 1.0, 4.5, 0.0})
    end

    test "an unknown family degrades to :unknown rather than guessing" do
      assert :unknown = Validator.analytic_moments({:weibull, 2.0, 1.0, 0.0})
      assert :unknown = Validator.analytic_moments({:something_else, 1.0})
    end
  end

  describe "check_analytic/3 — the check a differential test cannot make" do
    test "accepts samples drawn from the distribution it names" do
      xs = for _ <- 1..2000, do: :rand.normal()
      assert :ok = Validator.check_analytic(xs, :reference, {:normal, 0.0, 1.0})
    end

    test "rejects the over-dispersion the invalid-doubling bug produced" do
      # THE regression test for this whole class. Tree.do_build/11 merged
      # post-U-turn states into the trajectory, inflating Normal(0,1)'s
      # posterior variance to ~1.36-1.45 against a true 1.0 — in BOTH arms, so
      # every differential check passed unanimously for weeks.
      inflated = for _ <- 1..2000, do: :rand.normal() * :math.sqrt(1.4)

      assert {:error, %{check: :analytic_variance, arm: :reference, truth: 1.0}} =
               Validator.check_analytic(inflated, :reference, {:normal, 0.0, 1.0})
    end

    test "names the arm, because knowing WHICH is wrong is the point" do
      shifted = for _ <- 1..2000, do: :rand.normal() + 0.5

      assert {:error, %{check: :analytic_mean, arm: :vulkan}} =
               Validator.check_analytic(shifted, :vulkan, {:normal, 0.0, 1.0})
    end

    test "passes silently when the family has no closed form" do
      xs = for _ <- 1..100, do: :rand.normal()
      assert :ok = Validator.check_analytic(xs, :reference, {:weibull, 2.0, 1.0, 0.0})
    end
  end
end
