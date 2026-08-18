defmodule Exmc.MAMSTest do
  use ExUnit.Case, async: false

  @moduledoc """
  B1.3's gate.

  MAMS is asymptotically unbiased, so unlike `Exmc.MCLMC` it gets an
  analytic-moment battery. Every tolerance comes from
  `Exmc.NUTS.Vulkan.Validator.check_analytic/3`, which derives it from the
  sampler's **own ESS** — no round numbers appear anywhere in this file, and
  none should ever be added. `MISSION.md` §8: if one of these goes flaky, fix
  the statistic, not the threshold.

  What is asserted:

    * `check_analytic/3` on `Normal(0,1)`, `HalfNormal(1)`, `Exponential(2)`,
      pooled over seeds so the ESS is large enough for the tolerance to bite.
    * the conjugate Normal–Normal exact posterior, against the closed form.
    * the **involution property** the accept step's correctness rests on:
      `Φ^N` followed by a velocity flip and another `Φ^N` returns the start,
      and the energy error changes sign. If that fails, `min(1, e^{−ΔE})` is
      not a valid Metropolis ratio and every moment check above is luck.

  What is **not** here, and should be: a Geweke joint-distribution run. It
  needs `simulate_from_prior/2` and a public single-transition entry point,
  neither of which exists in this tree yet (`MISSION.md` §7 P1 item 9).
  """

  alias Exmc.{Builder, Rewrite}
  alias Exmc.Dist.{Exponential, HalfNormal, Normal}
  alias Exmc.MCLMC.Integrator
  alias Exmc.NUTS.Vulkan.Validator

  @seeds [1, 2, 3, 4]
  @warmup 500
  @samples 1500

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

  # MCLMC/MAMS are undefined at d = 1, so every target is declared twice. The
  # two coordinates are independent and identically distributed, so both are
  # checked and both must pass.
  defp two_of(dist, params) do
    Builder.new_ir()
    |> Builder.rv("a", dist, params)
    |> Builder.rv("b", dist, params)
    |> Rewrite.apply()
    |> Exmc.Compiler.compile_for_sampling()
  end

  defp pooled(compiled, name) do
    Enum.flat_map(@seeds, fn seed ->
      {trace, _stats} =
        Exmc.MAMS.sample_compiled(compiled, %{},
          num_warmup: @warmup,
          num_samples: @samples,
          seed: seed
        )

      trace |> Map.fetch!(name) |> Nx.to_flat_list()
    end)
  end

  # ------------------------------------------------- the analytic-moment battery

  @tag timeout: 900_000
  test "check_analytic on Normal(0,1)" do
    compiled = two_of(Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

    for name <- ["a", "b"] do
      xs = pooled(compiled, name)
      assert Validator.check_analytic(xs, :mams, {:normal, 0.0, 1.0}) == :ok
    end
  end

  @tag timeout: 900_000
  test "check_analytic on HalfNormal(1)" do
    compiled = two_of(HalfNormal, %{sigma: Nx.tensor(1.0)})

    for name <- ["a", "b"] do
      xs = pooled(compiled, name)
      assert Validator.check_analytic(xs, :mams, {:half_normal, 1.0}) == :ok
    end
  end

  @tag timeout: 900_000
  test "check_analytic on Exponential(2)" do
    compiled = two_of(Exponential, %{lambda: Nx.tensor(2.0)})

    for name <- ["a", "b"] do
      xs = pooled(compiled, name)
      assert Validator.check_analytic(xs, :mams, {:exponential, 2.0}) == :ok
    end
  end

  @tag timeout: 900_000
  test "conjugate Normal–Normal: the posterior matches the closed form" do
    # mu ~ N(0, 10); x|mu ~ N(mu, 1); observe x = 5.0.
    # Posterior precision 1/100 + 1, so var = 1/1.01 and mean = 5/1.01.
    # The same model `test/integration_test.exs:13` checks NUTS against.
    post_var = 1.0 / (1.0 / 100.0 + 1.0)
    post_mean = 5.0 / (1.0 / 100.0 + 1.0)

    ir =
      Builder.new_ir()
      |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(10.0)})
      |> Builder.rv("x", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
      |> Builder.obs("x_obs", "x", Nx.tensor(5.0))
      # a second free RV so d >= 2; it is independent of the data and must come
      # back as its own prior, which is a check in its own right
      |> Builder.rv("nuisance", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
      |> Rewrite.apply()

    compiled = Exmc.Compiler.compile_for_sampling(ir)

    mus = pooled(compiled, "mu")
    nuis = pooled(compiled, "nuisance")

    assert Validator.check_analytic(
             mus,
             :mams_conjugate,
             {:normal, post_mean, :math.sqrt(post_var)}
           ) == :ok

    assert Validator.check_analytic(nuis, :mams_nuisance, {:normal, 0.0, 1.0}) == :ok
  end

  # ------------------------------------------- the property the accept rests on

  test "the proposal is an involution up to a velocity flip, and ΔE changes sign" do
    d = 6

    inv2 =
      Nx.tensor(Enum.map(1..d, fn i -> 1.0 / (1.0 + 0.4 * i) end),
        type: :f64,
        backend: Nx.BinaryBackend
      )

    cubic = Nx.tensor(-0.15, type: :f64, backend: Nx.BinaryBackend)
    half = Nx.tensor(-0.5, type: :f64, backend: Nx.BinaryBackend)

    vag = fn x ->
      x2 = Nx.multiply(x, x)

      logp =
        Nx.add(
          Nx.multiply(half, Nx.sum(Nx.multiply(x2, inv2))),
          Nx.multiply(cubic, Nx.sum(Nx.multiply(x2, x2)))
        )

      grad =
        Nx.add(
          Nx.negate(Nx.multiply(x, inv2)),
          Nx.multiply(
            Nx.multiply(cubic, Nx.tensor(4.0, type: :f64, backend: Nx.BinaryBackend)),
            Nx.multiply(x2, x)
          )
        )

      {logp, grad}
    end

    sigma = Nx.broadcast(Nx.tensor(1.0, type: :f64, backend: Nx.BinaryBackend), {d})
    eps = 0.23
    n = 17

    x0 =
      Nx.tensor(Enum.map(1..d, fn i -> 0.2 * i - 0.5 end), type: :f64, backend: Nx.BinaryBackend)

    u0 =
      Integrator.normalize(
        Nx.tensor(Enum.map(1..d, fn i -> :math.sin(1.3 * i) end),
          type: :f64,
          backend: Nx.BinaryBackend
        )
      )

    {logp0, grad0} = vag.(x0)
    s0 = %{x: x0, u: u0, logp: logp0, grad: grad0}

    {s1, de} = Integrator.run(s0, n, eps, sigma, d, vag, :minimal_norm)

    back0 = %{s1 | u: Nx.negate(s1.u)}
    {back, de_back} = Integrator.run(back0, n, eps, sigma, d, vag, :minimal_norm)

    dx = Nx.subtract(back.x, x0) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    du = Nx.add(back.u, u0) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

    assert dx < 1.0e-9, "proposal is not an involution: max |Δx| = #{dx}"
    assert du < 1.0e-9, "velocity did not retrace: max |u + u₀| = #{du}"

    assert abs(de + de_back) < 1.0e-9 * max(1.0, abs(de)),
           "ΔE did not change sign under the reverse trajectory: #{de} vs #{de_back}"
  end

  test "the jittered step count has the requested mean" do
    for mu <- [1.5, 4.0, 17.3, 100.0] do
      rng = :rand.seed_s(:exsss, 11)

      {sum, _rng} =
        Enum.map_reduce(1..20_000, rng, fn _i, r ->
          {n, r} = Exmc.MAMS.draw_num_steps(mu, 100_000, r)
          {n, r}
        end)
        |> then(fn {ns, r} -> {Enum.sum(ns), r} end)

      got = sum / 20_000

      # 20k draws of a bounded integer: the standard error is well under 1%
      # of mu for every mu here.
      assert abs(got - mu) < 0.05 * mu, "E[N] = #{got}, wanted #{mu}"
    end
  end

  test "d < 2 is refused with a message that says what to use instead" do
    ir =
      Builder.new_ir()
      |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
      |> Rewrite.apply()

    compiled = Exmc.Compiler.compile_for_sampling(ir)

    assert_raise ArgumentError, ~r/Exmc\.NUTS\.Sampler/, fn ->
      Exmc.MAMS.sample_compiled(compiled, %{}, num_warmup: 10, num_samples: 10)
    end
  end

  test "stats report the gradient budget the ESS/gradient comparison needs" do
    compiled = two_of(Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

    {_trace, stats} =
      Exmc.MAMS.sample_compiled(compiled, %{}, num_warmup: 100, num_samples: 200, seed: 9)

    assert stats.grads_per_step == 2
    assert stats.integrator_steps > 0
    assert stats.grad_evals == stats.integrator_steps * 2
    assert stats.accept_rate > 0.0 and stats.accept_rate <= 1.0
    assert stats.mean_integration_steps >= 1.0
  end
end
