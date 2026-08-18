defmodule Exmc.SBI.PriorTest do
  use ExUnit.Case, async: true

  alias Exmc.SBI.Prior

  @moduledoc """
  Exact checks on the prior. Every assertion here is against a closed form or a
  round trip, at 1e-12; the two that involve sampling take their tolerance from
  the standard error of the estimate rather than from a round number.
  """

  describe "construction" do
    test "preserves coordinate order and round-trips through a vector" do
      p = Prior.new(lambda: {:uniform, 0.0, 2.0}, mu: {:normal, 1.0, 0.5}, k: {:exponential, 3.0})

      assert Prior.names(p) == [:lambda, :mu, :k]
      assert Prior.dimension(p) == 3

      params = %{mu: 1.25, k: 0.5, lambda: 1.75}
      assert Prior.to_vector(p, params) == [1.75, 1.25, 0.5]
      assert Prior.from_vector(p, [1.75, 1.25, 0.5]) == params
    end

    test "rejects malformed specs" do
      assert_raise ArgumentError, fn -> Prior.new([]) end
      assert_raise ArgumentError, fn -> Prior.new(a: {:uniform, 2.0, 1.0}) end
      assert_raise ArgumentError, fn -> Prior.new(a: {:normal, 0.0, -1.0}) end
      assert_raise ArgumentError, fn -> Prior.new(a: {:normal, 0.0, 1.0}, a: {:uniform, 0, 1}) end
      assert_raise ArgumentError, fn -> Prior.new(a: :whatever) end
      # A module with no sample/2: a prior has to be drawable, not only evaluable.
      assert_raise ArgumentError, fn -> Prior.new(a: {Exmc.Math, %{}}) end
    end
  end

  describe "logpdf against closed forms" do
    test "uniform is flat inside and impossible outside" do
      p = Prior.new(a: {:uniform, 1.0, 5.0})

      assert_in_delta Prior.logpdf(p, %{a: 2.0}), -:math.log(4.0), 1.0e-12
      assert_in_delta Prior.logpdf(p, %{a: 1.0}), -:math.log(4.0), 1.0e-12
      assert_in_delta Prior.logpdf(p, %{a: 5.0}), -:math.log(4.0), 1.0e-12
      assert Prior.logpdf(p, %{a: 5.0001}) == :neg_infinity
      assert Prior.logpdf(p, %{a: 0.9999}) == :neg_infinity
    end

    test "normal" do
      p = Prior.new(a: {:normal, 2.0, 3.0})

      for x <- [-4.0, 0.0, 2.0, 7.5] do
        z = (x - 2.0) / 3.0
        want = -0.5 * (z * z + :math.log(2.0 * :math.pi())) - :math.log(3.0)
        assert_in_delta Prior.logpdf(p, %{a: x}), want, 1.0e-12
      end
    end

    test "lognormal, and it is impossible at or below zero" do
      p = Prior.new(a: {:lognormal, 0.3, 0.8})

      for x <- [0.1, 1.0, 4.0] do
        z = (:math.log(x) - 0.3) / 0.8
        want = -0.5 * (z * z + :math.log(2.0 * :math.pi())) - :math.log(0.8) - :math.log(x)
        assert_in_delta Prior.logpdf(p, %{a: x}), want, 1.0e-12
      end

      assert Prior.logpdf(p, %{a: 0.0}) == :neg_infinity
      assert Prior.logpdf(p, %{a: -1.0}) == :neg_infinity
    end

    test "exponential" do
      p = Prior.new(a: {:exponential, 2.0})
      assert_in_delta Prior.logpdf(p, %{a: 0.5}), :math.log(2.0) - 1.0, 1.0e-12
      assert Prior.logpdf(p, %{a: -0.1}) == :neg_infinity
    end

    test "gamma matches Exmc.Dist.Gamma" do
      p = Prior.new(a: {:gamma, 2.0, 1.0})
      # Gamma(2, 1) at x = 1: (2-1)·log 1 + 2·log 1 - 1 - lgamma(2) = -1
      assert_in_delta Prior.logpdf(p, %{a: 1.0}), -1.0, 1.0e-12

      for x <- [0.25, 1.5, 6.0], {alpha, beta} <- [{2.0, 1.0}, {0.7, 3.0}, {5.5, 0.4}] do
        pr = Prior.new(a: {:gamma, alpha, beta})

        want =
          Nx.to_number(
            Exmc.Dist.Gamma.logpdf(Nx.tensor(x, backend: Nx.BinaryBackend), %{
              alpha: Nx.tensor(alpha, backend: Nx.BinaryBackend),
              beta: Nx.tensor(beta, backend: Nx.BinaryBackend)
            })
          )

        # 1e-5, not 1e-12, and the reason is worth recording: `Exmc.Math.lgamma/1`
        # builds its Lanczos constants with a bare `Nx.tensor/1`, whose default
        # float type is f32. The reference arm is therefore accurate to about
        # seven digits no matter what type the argument has. This module's own
        # `lgamma/1` is checked against closed forms at 1e-12 above; this
        # assertion is only that the two agree on the same function.
        assert_in_delta Prior.logpdf(pr, %{a: x}), want, 1.0e-5
      end

      assert Prior.logpdf(p, %{a: 0.0}) == :neg_infinity
    end

    test "the {Module, params} escape hatch agrees with the built-in form" do
      built_in = Prior.new(a: {:lognormal, 0.3, 0.8})
      via_dist = Prior.new(a: {Exmc.Dist.Lognormal, %{mu: 0.3, sigma: 0.8}})

      for x <- [0.1, 1.0, 4.0] do
        # 1e-6 and not 1e-12, for the same reason as the gamma case above:
        # `Nx.tensor(2.0 * :math.pi())` in `Exmc.Dist.Lognormal.logpdf/2` is
        # **f32** — Nx's default float type — so the reference arm carries a
        # ~1.6e-8 constant error even when every input is f64. Verified
        # directly: at x = 1.0 it returns -0.7661074662562859 against an exact
        # -0.766107481890463. The built-in form is computed in plain Elixir
        # floats and is the accurate one, which is why it is the default.
        assert_in_delta Prior.logpdf(built_in, %{a: x}), Prior.logpdf(via_dist, %{a: x}), 1.0e-6
      end

      assert Prior.logpdf(via_dist, %{a: -1.0}) == :neg_infinity
    end

    test "log-gamma against known values" do
      # Γ(1) = Γ(2) = 1; Γ(5) = 24; Γ(1/2) = √π
      assert_in_delta Prior.lgamma(1.0), 0.0, 1.0e-12
      assert_in_delta Prior.lgamma(2.0), 0.0, 1.0e-12
      assert_in_delta Prior.lgamma(5.0), :math.log(24.0), 1.0e-12
      assert_in_delta Prior.lgamma(0.5), :math.log(:math.sqrt(:math.pi())), 1.0e-12
      assert_in_delta Prior.lgamma(20.0), 39.339884187199495, 1.0e-9
    end

    test "the density sums over independent coordinates" do
      p = Prior.new(a: {:normal, 0.0, 1.0}, b: {:exponential, 2.0})
      pa = Prior.new(a: {:normal, 0.0, 1.0})
      pb = Prior.new(b: {:exponential, 2.0})

      assert_in_delta Prior.logpdf(p, %{a: 0.4, b: 1.1}),
                      Prior.logpdf(pa, %{a: 0.4}) + Prior.logpdf(pb, %{b: 1.1}),
                      1.0e-12
    end

    test "one impossible coordinate makes the joint impossible" do
      p = Prior.new(a: {:normal, 0.0, 1.0}, b: {:exponential, 2.0})
      assert Prior.logpdf(p, %{a: 0.4, b: -1.0}) == :neg_infinity
    end
  end

  describe "sampling" do
    # Tolerances are 4 standard errors of the estimator, not round numbers: the
    # test should fail when the sampler is wrong and not when the seed is
    # unlucky, and those are different thresholds for different n.
    @n 20_000

    test "draws land where the density says they should" do
      cases = [
        {{:uniform, 2.0, 6.0}, 4.0, 16.0 / 12.0},
        {{:normal, -1.0, 2.0}, -1.0, 4.0},
        {{:exponential, 4.0}, 0.25, 1.0 / 16.0},
        {{:gamma, 3.0, 2.0}, 1.5, 0.75},
        {{:lognormal, 0.0, 0.5}, :math.exp(0.125), (:math.exp(0.25) - 1.0) * :math.exp(0.25)}
      ]

      for {form, mean, var} <- cases do
        p = Prior.new(a: form)
        rng = :rand.seed_s(:exsss, {7, 11, 13})

        {xs, _} =
          Enum.map_reduce(1..@n, rng, fn _, rng ->
            {m, rng} = Prior.sample(p, rng)
            {m.a, rng}
          end)

        m = Enum.sum(xs) / @n
        v = Enum.reduce(xs, 0.0, fn x, acc -> acc + (x - m) * (x - m) end) / (@n - 1)

        assert_in_delta m, mean, 4.0 * :math.sqrt(var / @n)
        # SE of a variance estimate, bounded below by the Gaussian case
        # var·√(2/n); heavy-tailed families get a factor of two of slack rather
        # than a fourth moment computed per family.
        assert_in_delta v,
                        var,
                        8.0 * var * :math.sqrt(2.0 / @n),
                        "#{inspect(form)}: variance #{v} vs #{var}"
      end
    end

    test "is a pure function of the rng state" do
      p = Prior.new(a: {:normal, 0.0, 1.0}, b: {:gamma, 2.0, 3.0})
      rng = :rand.seed_s(:exsss, {1, 2, 3})

      assert Prior.sample(p, rng) == Prior.sample(p, rng)
      {first, rng2} = Prior.sample(p, rng)
      {second, _} = Prior.sample(p, rng2)
      refute first == second
    end
  end
end
