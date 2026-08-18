defmodule Exmc.MCLMCTest do
  use ExUnit.Case, async: false

  @moduledoc """
  B1.2's tests.

  **There is deliberately no analytic-moment assertion here.** MCLMC is biased
  by construction; `check_analytic/3` on its output would either fail correctly
  and block a feature that is working as designed, or get widened until it
  passed — which is exactly how the 0.3.0 variance defect survived a green
  suite (`MISSION.md` §6.3). Its gate is the published number in
  `bench_results/MCLMC_BIAS.md`, produced by `bench/mclmc_bias.exs`.

  What *is* asserted here is everything about MCLMC that is not a moment:

    * the sampler produces a well-formed, finite trace of the right shape
    * the EEVPD tuner hits the target it is aimed at, and moving the target
      moves the step size in the right direction
    * the diagonal preconditioner recovers known coordinate scales
    * **the bias behaves as the theory says**: `check_analytic/3` *must* flag
      MCLMC at a large ε and *must not* flag it at a small one, with MAMS on
      the same model as the control that the detector is not simply
      trigger-happy

  That last group is the honest gate for a biased method. "The answer is right"
  is the wrong question; "the error is controlled by the knob it is supposed to
  be controlled by" is the right one — and phrasing it through the repository's
  own ESS-derived detector means it cannot be satisfied by widening anything.
  """

  alias Exmc.{Builder, Rewrite}
  alias Exmc.Dist.Normal
  alias Exmc.NUTS.Vulkan.Validator

  setup do
    prev = Application.get_env(:exmc, :compiler)
    Application.put_env(:exmc, :compiler, :none)

    on_exit(fn ->
      if prev,
        do: Application.put_env(:exmc, :compiler, prev),
        else: Application.delete_env(:exmc, :compiler)
    end)

    :ok
  end

  defp iid_normals(d, sigma \\ 1.0) do
    Enum.reduce(1..d, Builder.new_ir(), fn i, ir ->
      Builder.rv(ir, "x#{i}", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(sigma)})
    end)
    |> Rewrite.apply()
    |> Exmc.Compiler.compile_for_sampling()
  end

  defp pooled(compiled, d, opts, seeds) do
    Enum.flat_map(seeds, fn seed ->
      {trace, _stats} = Exmc.MCLMC.sample_compiled(compiled, %{}, Keyword.put(opts, :seed, seed))
      Enum.flat_map(1..d, fn i -> trace |> Map.fetch!("x#{i}") |> Nx.to_flat_list() end)
    end)
  end

  # ------------------------------------------------------------- the mechanics

  test "produces a trace of the right shape with no non-finite draws" do
    compiled = iid_normals(4)

    {trace, stats} =
      Exmc.MCLMC.sample_compiled(compiled, %{},
        num_warmup: 300,
        num_samples: 500,
        seed: 7
      )

    assert map_size(trace) == 4

    for i <- 1..4 do
      t = Map.fetch!(trace, "x#{i}")
      assert Nx.shape(t) == {500}
      assert Enum.all?(Nx.to_flat_list(t), fn v -> is_float(v) and v == v end)
    end

    assert stats.step_size > 0.0
    assert stats.l > 0.0
    assert stats.grads_per_draw == 2
    assert stats.grad_evals == (300 + 500) * 2
    assert stats.divergences == 0
  end

  test "the leapfrog integrator costs one gradient per step, not two" do
    compiled = iid_normals(3)

    {_trace, stats} =
      Exmc.MCLMC.sample_compiled(compiled, %{},
        num_warmup: 100,
        num_samples: 200,
        seed: 2,
        integrator: :leapfrog
      )

    assert stats.grads_per_draw == 1
    assert stats.grad_evals == 300
  end

  test "d < 2 is refused with a message that says what to use instead" do
    compiled = iid_normals(1)

    assert_raise ArgumentError, ~r/Exmc\.NUTS\.Sampler/, fn ->
      Exmc.MCLMC.sample_compiled(compiled, %{}, num_warmup: 10, num_samples: 10)
    end
  end

  # ----------------------------------------------------------------- the tuner

  test "the tuned run lands near the EEVPD target it was aimed at" do
    compiled = iid_normals(8)

    {_trace, stats} =
      Exmc.MCLMC.sample_compiled(compiled, %{},
        num_warmup: 1500,
        num_samples: 2000,
        seed: 4
      )

    target = Exmc.MCLMC.Tuning.desired_energy_var()

    # The tuner is a stochastic-approximation scheme with a Gaussian weight on
    # log(xi); it aims at the target, it does not land on it. An order of
    # magnitude either way is a real check on whether it is aiming at all —
    # a broken tuner misses by four or five, not by two.
    assert stats.eevpd > target / 10.0 and stats.eevpd < target * 10.0,
           "measured EEVPD #{stats.eevpd} is nowhere near the target #{target}"
  end

  test "raising the EEVPD target raises the tuned step size" do
    compiled = iid_normals(8)

    eps = fn target ->
      {_t, stats} =
        Exmc.MCLMC.sample_compiled(compiled, %{},
          num_warmup: 1200,
          num_samples: 200,
          seed: 11,
          desired_energy_var: target
        )

      stats.step_size
    end

    tight = eps.(1.0e-6)
    loose = eps.(1.0e-1)

    # Five orders of magnitude in the target is ~10^(5/6) ≈ 7x in eps if the
    # sixth-power relation holds. Asserting only the ordering, with a factor of
    # 2 of headroom, keeps this insensitive to how well the chain mixed.
    assert loose > 2.0 * tight,
           "eps did not track the EEVPD target: #{tight} at 1e-6 vs #{loose} at 1e-1"
  end

  test "the diagonal preconditioner recovers known coordinate scales" do
    # Two coordinates at sd 1 and two at sd 10; sigma must come back ~[1,1,10,10].
    ir =
      Builder.new_ir()
      |> Builder.rv("a", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
      |> Builder.rv("b", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
      |> Builder.rv("c", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(10.0)})
      |> Builder.rv("d", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(10.0)})
      |> Rewrite.apply()

    compiled = Exmc.Compiler.compile_for_sampling(ir)

    {_trace, stats} =
      Exmc.MCLMC.sample_compiled(compiled, %{},
        num_warmup: 3000,
        num_samples: 100,
        seed: 13
      )

    [s1, s2, s3, s4] = Nx.to_flat_list(stats.sigma)

    # The preconditioner is estimated from a short, correlated warmup, so this
    # asserts the *separation* — the 10x coordinates must come back clearly
    # larger than the 1x ones — rather than the values.
    assert min(s3, s4) > 3.0 * max(s1, s2),
           "preconditioner did not separate the scales: #{inspect([s1, s2, s3, s4])}"
  end

  # ---------------------------------------------- the bias, behaving as claimed

  # The right question for a biased sampler is not "is the answer right" but
  # "is the error controlled by the knob it is supposed to be controlled by".
  #
  # Both tests below phrase that through `check_analytic/3` — the same
  # ESS-derived detector every other gate in this repository uses — and assert
  # its *verdict* rather than a hand-chosen effect size. At a large step size
  # the bias must be detectable; at a small one it must not be. Neither
  # assertion can be satisfied by widening anything, because the tolerance is
  # computed from the chain's own effective sample size and tightens as the
  # sampler improves.

  @tag timeout: 900_000
  test "at a large step size the bias is detectable against analytic truth" do
    d = 4
    compiled = iid_normals(d)

    xs = pooled(compiled, d, [num_warmup: 800, num_samples: 4000, step_size: 6.0], [1, 2, 3])
    {_m, v} = Validator.mean_var(xs)

    assert {:error, %{check: :analytic_variance}} =
             Validator.check_analytic(xs, :mclmc_large_eps, {:normal, 0.0, 1.0}),
           """
           MCLMC at eps = 6.0 produced variance #{v} against a true 1.0 and
           check_analytic/3 did not flag it. Either the bias vanished — in
           which case bench_results/MCLMC_BIAS.md is wrong and this method is
           not what the papers describe — or the detector is not detecting,
           which is worse.
           """
  end

  @tag timeout: 900_000
  test "at a small step size the bias falls below the detector" do
    d = 4
    compiled = iid_normals(d)

    xs = pooled(compiled, d, [num_warmup: 800, num_samples: 4000, step_size: 0.3], [1, 2, 3])

    assert Validator.check_analytic(xs, :mclmc_small_eps, {:normal, 0.0, 1.0}) == :ok,
           "MCLMC at eps = 0.3 should be within its own Monte-Carlo error of the truth"
  end

  # There was a third test here and it was deleted rather than tuned. It tried
  # to assert that the bias *ratio* between two step sizes exceeds some factor,
  # and it went through two wrong statistics before the reason became clear:
  #
  #   * `|var - 1|` alone is not monotone in eps. Measured at d = 4 over three
  #     seeds and 4000 draws: 0.0317 at eps = 0.3 against 0.0185 at eps = 8.0.
  #     Not because the bias shrank but because at eps = 0.3 the chain is so
  #     autocorrelated that Monte-Carlo error swamps it.
  #   * the deviation in units of its own standard error *is* ordered —
  #     z = 1.10 at eps = 0.3 against z = 2.18 at eps = 8.0 — but only by a
  #     factor of two, and any assertion of the form `large > K * small` then
  #     comes down to choosing K so the test passes. That is the move
  #     `MISSION.md` §8 forbids, whether the number lives in a tolerance or in
  #     a ratio.
  #
  # The two tests above already assert the substance, through the repository's
  # own detector rather than through a number chosen here: at eps = 6.0 the
  # bias is detectable and at eps = 0.3 it is not. The *shape* of the curve
  # between them is reported, not asserted — bench_results/MCLMC_BIAS.md.

  @tag timeout: 900_000
  test "MAMS on the same model is not flagged, at the step size it chooses itself" do
    # The control for the two tests above: the detector fires on MCLMC at
    # eps = 6.0 and does not fire on the adjusted sampler running the same
    # dynamics on the same model. Without this, "check_analytic returned an
    # error" only shows the detector is trigger-happy.
    d = 4
    compiled = iid_normals(d)

    xs =
      Enum.flat_map([1, 2, 3], fn seed ->
        {trace, _s} =
          Exmc.MAMS.sample_compiled(compiled, %{},
            num_warmup: 800,
            num_samples: 4000,
            seed: seed
          )

        Enum.flat_map(1..d, fn i -> trace |> Map.fetch!("x#{i}") |> Nx.to_flat_list() end)
      end)

    assert Validator.check_analytic(xs, :mams_control, {:normal, 0.0, 1.0}) == :ok
  end
end
