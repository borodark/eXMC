defmodule Exmc.NUTS.ConjugateOracleTest do
  @moduledoc """
  Recovery against a posterior that is known in closed form.

  For `y = Xb + e` with `e ~ N(0, sigma^2 I)`, `b ~ N(m0, S0)` and sigma
  **known**, the posterior is exactly normal:

      Sn = (S0^-1 + X'X / sigma^2)^-1
      mn = Sn (S0^-1 m0 + X'y / sigma^2)

  There is no Monte Carlo error on the reference side at all — the sampler is
  compared against a matrix expression, not against another sampler. That is
  what makes this sharp where the rest of the suite is not: most of the
  statistical harness here is DIFFERENTIAL, running two backends and asking
  whether they agree, which a defect in the shared NUTS tree moves identically
  in both arms.

  It also catches a specific failure this project has already shipped once. In
  `f8d430a78` a reduce loop was bounded by `pc.n_obs` while the data lived in
  closure captures, so the likelihood evaluated to exactly nothing and the
  sampler returned the PRIOR. A prior is a plausible-looking posterior; nothing
  in the suite noticed. Here it would move the mean by many analytic standard
  deviations.

  Contributed by the pathmc_ex session, which uses exmc as its inference
  engine, and adapted. Pinning sigma is what buys exactness — sampling it
  jointly gives a normal-inverse-gamma whose beta marginal is multivariate-t,
  also closed form but a heavier reference than this needs.

  ## What is gated, and what is not

  The BIAS — mean error in units of the analytic sd — is the sharp number and
  is what these tests assert. The sd ratio is reported but bounded loosely: at
  these draw counts it carries several percent of sampling error of its own,
  and a standalone run reads ~1.05 against a true value of 1.0. Gating it
  tightly would buy flakes, not coverage.

  There is a stronger reason than sampling error, contributed by the pathmc_ex
  session after they mutation-checked their copy of this oracle. The posterior
  covariance

      Sn = (S0^-1 + X'X / sigma^2)^-1

  contains no `y` at all. So **any error that only moves the mean leaves the
  analytic sd exactly unchanged, and the sd-ratio check has zero detection
  power against that entire class.** All of the discrimination here lives in
  the bias assertion; the ratio is a scale sanity check and must not be read
  as a gate.

  The consequence for anyone extending these tests: a mutation that shifts the
  response validates the bias assertion and says nothing about the sd one.
  Covering the sd side needs a perturbation that changes `X'X` or sigma —
  scaling a predictor column would do it — which is a separate mutation, not a
  stronger version of the same one.
  """
  use ExUnit.Case, async: false

  alias Exmc.{Builder, Dist, IR}

  @f64 [type: :f64]

  # Loose enough not to flake on Monte Carlo error at 400 draws, tight enough
  # that returning the prior (or a frozen chain) fails by a wide margin: the
  # prior here is N(0, 5) against a posterior at 1.01 +/- 0.064, which is ~16
  # analytic sds away.
  @max_bias_sds 0.25
  @sd_ratio_bounds {0.75, 1.3}

  defp fixture do
    xs = Enum.map(1..40, &(&1 / 4))
    noise = [0.12, -0.31, 0.05, 0.22, -0.18, 0.31, -0.07, 0.14, -0.25, 0.09]

    ys =
      xs
      |> Enum.with_index()
      |> Enum.map(fn {x, i} -> 1.0 + 2.0 * x + Enum.at(noise, rem(i, 10)) end)

    x = Nx.stack([Nx.broadcast(Nx.tensor(1.0, @f64), {40}), Nx.tensor(xs, @f64)], axis: 1)
    y = Nx.tensor(ys, @f64)
    sigma = 0.2
    prior_sd = 5.0

    s0_inv = Nx.multiply(Nx.eye(2, @f64), 1.0 / (prior_sd * prior_sd))
    precision = Nx.add(s0_inv, Nx.divide(Nx.dot(Nx.transpose(x), x), sigma * sigma))
    cov_n = Nx.LinAlg.invert(precision)
    mean_n = Nx.dot(cov_n, Nx.divide(Nx.dot(Nx.transpose(x), y), sigma * sigma))
    sd_n = cov_n |> Nx.take_diagonal() |> Nx.sqrt()

    loglik =
      Dist.Custom.new(fn observed, params ->
        resid = Nx.subtract(observed, Nx.dot(x, params.beta))

        Nx.sum(
          Nx.subtract(
            Nx.negate(Nx.log(Nx.multiply(sigma, :math.sqrt(2 * :math.pi())))),
            Nx.divide(Nx.pow(resid, 2), 2 * sigma * sigma)
          )
        )
      end)

    ir =
      IR.new()
      |> Builder.rv(
        "beta",
        Dist.Normal,
        %{mu: Nx.tensor(0.0, @f64), sigma: Nx.tensor(prior_sd, @f64)},
        shape: {2}
      )
      |> Dist.Custom.rv("Y", loglik, %{beta: "beta"})
      |> Builder.obs("Y_obs", "Y", y)

    %{ir: ir, mean_n: mean_n, sd_n: sd_n}
  end

  defp check!(traces, %{mean_n: mean_n, sd_n: sd_n}, label) do
    draws = Nx.concatenate(Enum.map(traces, & &1["beta"]), axis: 0)
    sampled_mean = Nx.mean(draws, axes: [0])
    sampled_sd = Nx.standard_deviation(draws, axes: [0])

    bias = Nx.divide(Nx.abs(Nx.subtract(sampled_mean, mean_n)), sd_n)
    ratio = Nx.divide(sampled_sd, sd_n)

    max_bias = Nx.to_number(Nx.reduce_max(bias))
    {lo, hi} = @sd_ratio_bounds
    ratios = Nx.to_flat_list(ratio)

    detail =
      "#{label}: analytic #{inspect(Nx.to_flat_list(mean_n))} sd #{inspect(Nx.to_flat_list(sd_n))} | " <>
        "sampled #{inspect(Nx.to_flat_list(sampled_mean))} sd #{inspect(Nx.to_flat_list(sampled_sd))} | " <>
        "bias #{inspect(Nx.to_flat_list(bias))} sds, ratio #{inspect(ratios)}"

    # A frozen chain has sd exactly 0. Assert it separately from the bias so
    # the failure names the right thing: a freeze is not a biased kernel.
    assert Enum.all?(Nx.to_flat_list(sampled_sd), &(&1 > 0.0)),
           "every draw identical — chain never moved. #{detail}"

    assert max_bias <= @max_bias_sds,
           "mean off by #{Float.round(max_bias, 4)} analytic sds (max #{@max_bias_sds}). #{detail}"

    assert Enum.all?(ratios, &(&1 >= lo and &1 <= hi)),
           "sd ratio outside #{inspect({lo, hi})}. #{detail}"
  end

  describe "closed-form Gaussian posterior" do
    @tag timeout: 600_000
    test "the parallel path recovers the analytic posterior" do
      f = fixture()

      {traces, _stats} =
        Exmc.NUTS.Sampler.sample_chains(f.ir, 2,
          init_values: %{"beta" => Nx.broadcast(Nx.tensor(0.0, @f64), {2})},
          num_samples: 400,
          num_warmup: 400,
          vectorized: false
        )

      check!(traces, f, "parallel")
    end

    # REGRESSION for the frozen-chain defect: sample_chains_vectorized_compiled/3
    # warmed up chain 0, then re-initialised every chain back at `init_values`
    # while sampling with the metric warmup had adapted to the typical set.
    # Every proposal was rejected and every draw equalled the init value.
    #
    # THE INIT HERE MUST STAY FAR FROM THE MODE. The predicate is init distance
    # relative to posterior width, not the likelihood type: this same model
    # started at its mode sampled correctly even with the bug present, so a
    # test written that way passes vacuously. beta = [0, 0] against a posterior
    # at [1.01, 2.00] with sds [0.064, 0.011] is ~180 posterior sds out on the
    # second coordinate, which is what exposed it.
    @tag timeout: 600_000
    test "the vectorized path recovers it too, starting far from the mode" do
      f = fixture()

      {traces, _stats} =
        Exmc.NUTS.Sampler.sample_chains(f.ir, 2,
          init_values: %{"beta" => Nx.broadcast(Nx.tensor(0.0, @f64), {2})},
          num_samples: 400,
          num_warmup: 400,
          vectorized: true
        )

      check!(traces, f, "vectorized")
    end

    # `vectorized` used to default to `num_chains > 1`, so the frozen path was
    # what every multi-chain caller got without asking for it. Pin the default
    # rather than the behaviour, because the behaviour is now correct either
    # way and the reason for the default is the R-hat one.
    test "multi-chain sampling does not opt into vectorized by default" do
      f = fixture()

      {traces, _stats} =
        Exmc.NUTS.Sampler.sample_chains(f.ir, 2,
          init_values: %{"beta" => Nx.broadcast(Nx.tensor(0.0, @f64), {2})},
          num_samples: 60,
          num_warmup: 60
        )

      # The parallel path warms each chain separately from its own seed, so the
      # two chains differ. The vectorized path starts both from one warmup
      # endpoint, so their first draws are identical.
      [a, b] = Enum.map(traces, fn t -> t["beta"] |> Nx.take(Nx.tensor(0), axis: 0) end)

      refute Nx.to_flat_list(a) == Nx.to_flat_list(b),
             "both chains started at the same point — sample_chains/3 took the " <>
               "vectorized path by default, which gives up the over-dispersed " <>
               "starts R-hat depends on"
    end
  end
end
