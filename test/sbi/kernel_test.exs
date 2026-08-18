defmodule Exmc.SBI.KernelTest do
  use ExUnit.Case, async: true

  alias Exmc.SBI.ABCSMC

  @moduledoc """
  The pieces of ABC-SMC that have a closed form, checked against it.

  This is the cheapest and least flaky class of gate available here: the
  Cholesky factor, the weighted covariance, the multivariate normal density and
  the importance weight are all exactly computable by hand at `d = 2`, so every
  assertion below is at 1e-12 and none of them can be made to pass by widening
  a tolerance. An algebra error in any of them produces a posterior that is
  wrong in a way no amount of sampling would identify as *this* function's
  fault, which is precisely why they are tested here rather than only through
  the end-to-end gates.
  """

  # Σ = [[4, 2], [2, 3]];  L = [[2, 0], [1, √2]]
  @sigma {{4.0, 2.0}, {2.0, 3.0}}

  defp at(m, i, j), do: m |> elem(i) |> elem(j)

  defp mat_mul_transpose(l, d) do
    for i <- 0..(d - 1) do
      for j <- 0..(d - 1) do
        Enum.reduce(0..(d - 1), 0.0, fn k, acc -> acc + at(l, i, k) * at(l, j, k) end)
      end
      |> List.to_tuple()
    end
    |> List.to_tuple()
  end

  describe "cholesky!/2" do
    test "reproduces the hand-computed factor" do
      l = ABCSMC.cholesky!(@sigma, 2)

      assert_in_delta at(l, 0, 0), 2.0, 1.0e-12
      assert_in_delta at(l, 0, 1), 0.0, 1.0e-12
      assert_in_delta at(l, 1, 0), 1.0, 1.0e-12
      assert_in_delta at(l, 1, 1), :math.sqrt(2.0), 1.0e-12
    end

    test "L·Lᵀ = Σ for a 4×4 built to be positive definite" do
      rng = :rand.seed_s(:exsss, {3, 5, 7})
      d = 4

      # A = random; Σ = AᵀA + 4I is symmetric positive definite by construction.
      {a, _} =
        Enum.map_reduce(1..(d * d), rng, fn _, rng -> :rand.normal_s(rng) end)

      a = a |> Enum.chunk_every(d) |> Enum.map(&List.to_tuple/1) |> List.to_tuple()

      sigma =
        for i <- 0..(d - 1) do
          for j <- 0..(d - 1) do
            s = Enum.reduce(0..(d - 1), 0.0, fn k, acc -> acc + at(a, k, i) * at(a, k, j) end)
            if i == j, do: s + 4.0, else: s
          end
          |> List.to_tuple()
        end
        |> List.to_tuple()

      l = ABCSMC.cholesky!(sigma, d)
      back = mat_mul_transpose(l, d)

      for i <- 0..(d - 1), j <- 0..(d - 1) do
        assert_in_delta at(back, i, j), at(sigma, i, j), 1.0e-12
      end
    end

    test "raises rather than factorising something that is not a covariance" do
      # Every particle identical: the covariance is exactly zero, and the
      # jitter ladder tops out without producing a usable kernel.
      assert_raise RuntimeError, ~r/collapsed/, fn ->
        ABCSMC.cholesky!({{0.0, 0.0}, {0.0, 0.0}}, 2)
      end
    end
  end

  describe "weighted_cov/3" do
    test "with equal weights it is the second central moment" do
      vectors = [{1.0, 2.0}, {3.0, 5.0}, {5.0, 4.0}, {7.0, 9.0}]
      w = List.duplicate(0.25, 4)

      cov = ABCSMC.weighted_cov(vectors, w, 2)

      # means 4.0 and 5.0
      # var x  = (9 + 1 + 1 + 9)/4 = 5
      # var y  = (9 + 0 + 1 + 16)/4 = 6.5
      # cov xy = (9 + 0 + (-1) + 12)/4 = 5
      assert_in_delta at(cov, 0, 0), 5.0, 1.0e-12
      assert_in_delta at(cov, 1, 1), 6.5, 1.0e-12
      assert_in_delta at(cov, 0, 1), 5.0, 1.0e-12
      assert_in_delta at(cov, 1, 0), 5.0, 1.0e-12
    end

    test "weights are honoured, and normalisation is internal" do
      vectors = [{0.0, 0.0}, {2.0, 4.0}]

      # w = (3/4, 1/4): mean = (0.5, 1.0)
      # var x = 0.75·0.25 + 0.25·2.25 = 0.75
      # var y = 0.75·1.0  + 0.25·9.0  = 3.0
      # cov   = 0.75·(-0.5)(-1.0) + 0.25·(1.5)(3.0) = 0.375 + 1.125 = 1.5
      for scale <- [1.0, 40.0] do
        cov = ABCSMC.weighted_cov(vectors, [0.75 * scale, 0.25 * scale], 2)
        assert_in_delta at(cov, 0, 0), 0.75, 1.0e-12
        assert_in_delta at(cov, 1, 1), 3.0, 1.0e-12
        assert_in_delta at(cov, 0, 1), 1.5, 1.0e-12
        assert_in_delta at(cov, 1, 0), 1.5, 1.0e-12
      end
    end
  end

  describe "mvn_logpdf/4" do
    test "d = 1 reduces to the univariate normal" do
      chol = ABCSMC.cholesky!({{9.0}}, 1)
      log_det = 2.0 * :math.log(at(chol, 0, 0))

      for x <- [-2.0, 0.0, 1.5, 7.0] do
        z = (x - 1.0) / 3.0
        want = -0.5 * (z * z + :math.log(2.0 * :math.pi())) - :math.log(3.0)
        assert_in_delta ABCSMC.mvn_logpdf({x}, {1.0}, chol, log_det), want, 1.0e-12
      end
    end

    test "d = 2 against the explicit quadratic form" do
      chol = ABCSMC.cholesky!(@sigma, 2)
      log_det = 2.0 * (:math.log(at(chol, 0, 0)) + :math.log(at(chol, 1, 1)))

      # det Σ = 12 - 4 = 8;  Σ⁻¹ = (1/8)[[3, -2], [-2, 4]]
      assert_in_delta :math.exp(log_det), 8.0, 1.0e-12

      mu = {0.5, -1.0}

      for x <- [{0.5, -1.0}, {2.0, 1.0}, {-3.0, 4.0}] do
        dx = elem(x, 0) - elem(mu, 0)
        dy = elem(x, 1) - elem(mu, 1)
        quad = (3.0 * dx * dx - 4.0 * dx * dy + 4.0 * dy * dy) / 8.0
        want = -0.5 * (2.0 * :math.log(2.0 * :math.pi()) + :math.log(8.0) + quad)

        assert_in_delta ABCSMC.mvn_logpdf(x, mu, chol, log_det), want, 1.0e-12
      end
    end

    test "it integrates to one, checked by importance sampling against itself" do
      # Sampling from the kernel and averaging exp(logpdf - logpdf) is vacuous;
      # instead draw from a WIDER Gaussian and check E[p(x)/q(x)] = 1.
      chol = ABCSMC.cholesky!(@sigma, 2)
      log_det = 2.0 * (:math.log(at(chol, 0, 0)) + :math.log(at(chol, 1, 1)))

      wide = {{16.0, 4.0}, {4.0, 12.0}}
      chol_q = ABCSMC.cholesky!(wide, 2)
      log_det_q = 2.0 * (:math.log(at(chol_q, 0, 0)) + :math.log(at(chol_q, 1, 1)))

      mu = {0.0, 0.0}
      n = 200_000

      {sum, sum_sq, _} =
        Enum.reduce(1..n, {0.0, 0.0, :rand.seed_s(:exsss, {2, 4, 8})}, fn _, {s, s2, rng} ->
          {x, rng} = ABCSMC.mvn_sample(mu, chol_q, rng)

          r =
            :math.exp(
              ABCSMC.mvn_logpdf(x, mu, chol, log_det) -
                ABCSMC.mvn_logpdf(x, mu, chol_q, log_det_q)
            )

          {s + r, s2 + r * r, rng}
        end)

      m = sum / n
      se = :math.sqrt(max(sum_sq / n - m * m, 0.0) / n)

      # 4 standard errors of the importance-sampling estimator, not a round
      # number: a wrong normalising constant moves this by O(1).
      assert_in_delta m, 1.0, 4.0 * se
    end
  end

  describe "mvn_sample/3" do
    test "draws have the covariance they were given" do
      chol = ABCSMC.cholesky!(@sigma, 2)
      mu = {-1.0, 3.0}
      n = 100_000

      {xs, _} =
        Enum.map_reduce(1..n, :rand.seed_s(:exsss, {11, 13, 17}), fn _, rng ->
          ABCSMC.mvn_sample(mu, chol, rng)
        end)

      mx = Enum.reduce(xs, 0.0, &(elem(&1, 0) + &2)) / n
      my = Enum.reduce(xs, 0.0, &(elem(&1, 1) + &2)) / n

      cxx = Enum.reduce(xs, 0.0, fn v, a -> a + (elem(v, 0) - mx) * (elem(v, 0) - mx) end) / n
      cyy = Enum.reduce(xs, 0.0, fn v, a -> a + (elem(v, 1) - my) * (elem(v, 1) - my) end) / n
      cxy = Enum.reduce(xs, 0.0, fn v, a -> a + (elem(v, 0) - mx) * (elem(v, 1) - my) end) / n

      assert_in_delta mx, -1.0, 4.0 * :math.sqrt(4.0 / n)
      assert_in_delta my, 3.0, 4.0 * :math.sqrt(3.0 / n)
      assert_in_delta cxx, 4.0, 4.0 * 4.0 * :math.sqrt(2.0 / n)
      assert_in_delta cyy, 3.0, 4.0 * 3.0 * :math.sqrt(2.0 / n)
      # SE of a covariance estimate: √((σxx·σyy + σxy²)/n)
      assert_in_delta cxy, 2.0, 4.0 * :math.sqrt((4.0 * 3.0 + 4.0) / n)
    end
  end

  describe "importance_weights/6" do
    test "one previous particle: w ∝ π(θ) / K(θ | θ₁)" do
      chol = ABCSMC.cholesky!({{1.0}}, 1)
      log_det = 0.0
      prev = [{0.0}]
      prev_w = [1.0]

      thetas = [{0.5}, {-1.0}, {2.0}]
      # any prior; use log π(θ) = -θ² so the expected answer is explicit
      log_priors = Enum.map(thetas, fn {t} -> -t * t end)

      got = ABCSMC.importance_weights(log_priors, thetas, prev_w, prev, chol, log_det)

      raw =
        Enum.zip_with(log_priors, thetas, fn lp, {t} ->
          k = -0.5 * (t * t + :math.log(2.0 * :math.pi()))
          :math.exp(lp - k)
        end)

      total = Enum.sum(raw)
      want = Enum.map(raw, &(&1 / total))

      assert length(got) == 3
      assert_in_delta Enum.sum(got), 1.0, 1.0e-12

      Enum.zip(got, want)
      |> Enum.each(fn {g, w} -> assert_in_delta g, w, 1.0e-12 end)
    end

    test "the denominator really is the weighted mixture over the whole previous population" do
      chol = ABCSMC.cholesky!({{2.0}}, 1)
      log_det = :math.log(2.0)
      prev = [{-1.0}, {1.0}, {4.0}]
      prev_w = [0.5, 0.3, 0.2]
      thetas = [{0.0}, {3.0}]
      log_priors = [-0.1, -0.9]

      got = ABCSMC.importance_weights(log_priors, thetas, prev_w, prev, chol, log_det)

      raw =
        Enum.zip_with(log_priors, thetas, fn lp, {t} ->
          denom =
            [prev_w, prev]
            |> Enum.zip_reduce(0.0, fn [w, {mu}], acc ->
              acc +
                w * :math.exp(-0.5 * ((t - mu) * (t - mu) / 2.0 + :math.log(4.0 * :math.pi())))
            end)

          :math.exp(lp) / denom
        end)

      total = Enum.sum(raw)

      Enum.zip(got, Enum.map(raw, &(&1 / total)))
      |> Enum.each(fn {g, w} -> assert_in_delta g, w, 1.0e-12 end)
    end

    test "weights are invariant to the scale of the previous population's weights" do
      chol = ABCSMC.cholesky!({{1.5}}, 1)
      log_det = :math.log(1.5)
      prev = [{-1.0}, {1.0}]
      thetas = [{0.0}, {0.7}, {-2.0}]
      log_priors = [-0.1, -0.4, -1.2]

      a = ABCSMC.importance_weights(log_priors, thetas, [0.5, 0.5], prev, chol, log_det)
      b = ABCSMC.importance_weights(log_priors, thetas, [50.0, 50.0], prev, chol, log_det)

      Enum.zip(a, b) |> Enum.each(fn {x, y} -> assert_in_delta x, y, 1.0e-12 end)
    end
  end
end
