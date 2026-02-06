defmodule Exmc.DistTest do
  use ExUnit.Case, async: true

  alias Exmc.Math
  alias Exmc.Dist.{Gamma, Beta, StudentT, Laplace, Cauchy}

  # ── lgamma ───────────────────────────────────────────────────

  test "lgamma(1) = 0" do
    result = Math.lgamma(Nx.tensor(1.0)) |> Nx.to_number()
    assert_in_delta result, 0.0, 1.0e-6
  end

  test "lgamma(0.5) = log(sqrt(pi))" do
    expected = 0.5 * :math.log(:math.pi())
    result = Math.lgamma(Nx.tensor(0.5)) |> Nx.to_number()
    assert_in_delta result, expected, 1.0e-6
  end

  test "lgamma(5) = log(24)" do
    expected = :math.log(24.0)
    result = Math.lgamma(Nx.tensor(5.0)) |> Nx.to_number()
    assert_in_delta result, expected, 1.0e-5
  end

  test "lgamma(10) = log(362880)" do
    expected = :math.log(362_880.0)
    result = Math.lgamma(Nx.tensor(10.0)) |> Nx.to_number()
    assert_in_delta result, expected, 1.0e-5
  end

  # ── lbeta ────────────────────────────────────────────────────

  test "lbeta(1,1) = 0" do
    result = Math.lbeta(Nx.tensor(1.0), Nx.tensor(1.0)) |> Nx.to_number()
    assert_in_delta result, 0.0, 1.0e-6
  end

  test "lbeta(2,3) = log(1/12)" do
    # B(2,3) = 1!*2!/4! = 2/24 = 1/12
    expected = :math.log(1.0 / 12.0)
    result = Math.lbeta(Nx.tensor(2.0), Nx.tensor(3.0)) |> Nx.to_number()
    assert_in_delta result, expected, 1.0e-6
  end

  # ── Gamma distribution ──────────────────────────────────────

  test "Gamma logpdf at known point" do
    # Gamma(alpha=2, beta=1) at x=1: (2-1)*log(1) - 1*1 + 2*log(1) - lgamma(2) = 0 - 1 + 0 - 0 = -1
    params = %{alpha: Nx.tensor(2.0), beta: Nx.tensor(1.0)}
    result = Gamma.logpdf(Nx.tensor(1.0), params) |> Nx.to_number()
    assert_in_delta result, -1.0, 1.0e-6
  end

  test "Gamma logpdf Exp(1) specialization" do
    # Gamma(alpha=1, beta=1) at x=2 should equal Exp(1) at x=2: log(1) - 1*2 = -2
    params = %{alpha: Nx.tensor(1.0), beta: Nx.tensor(1.0)}
    result = Gamma.logpdf(Nx.tensor(2.0), params) |> Nx.to_number()
    assert_in_delta result, -2.0, 1.0e-6
  end

  test "Gamma support and transform" do
    params = %{alpha: Nx.tensor(2.0), beta: Nx.tensor(1.0)}
    assert Gamma.support(params) == :positive
    assert Gamma.transform(params) == :log
  end

  # ── Beta distribution ───────────────────────────────────────

  test "Beta(1,1) is uniform -> logpdf = 0" do
    params = %{alpha: Nx.tensor(1.0), beta: Nx.tensor(1.0)}
    result = Beta.logpdf(Nx.tensor(0.5), params) |> Nx.to_number()
    assert_in_delta result, 0.0, 1.0e-6
  end

  test "Beta(2,2) logpdf at x=0.5" do
    # Beta(2,2) at 0.5: (2-1)*log(0.5) + (2-1)*log(0.5) - lbeta(2,2)
    # lbeta(2,2) = lgamma(2)+lgamma(2)-lgamma(4) = 0+0-log(6) = -log(6)
    # logpdf = -log(2) - log(2) + log(6) = log(6/4) = log(1.5)
    expected = :math.log(1.5)
    params = %{alpha: Nx.tensor(2.0), beta: Nx.tensor(2.0)}
    result = Beta.logpdf(Nx.tensor(0.5), params) |> Nx.to_number()
    assert_in_delta result, expected, 1.0e-5
  end

  test "Beta support and transform" do
    params = %{alpha: Nx.tensor(2.0), beta: Nx.tensor(2.0)}
    assert Beta.support(params) == :unit
    assert Beta.transform(params) == :logit
  end

  # ── StudentT distribution ───────────────────────────────────

  test "StudentT with large df approaches Normal" do
    # With df=1000, loc=0, scale=1 at x=0 should be close to Normal(0,1) logpdf
    normal_logpdf_0 = -0.5 * :math.log(2.0 * :math.pi())
    params = %{df: Nx.tensor(1000.0), loc: Nx.tensor(0.0), scale: Nx.tensor(1.0)}
    result = StudentT.logpdf(Nx.tensor(0.0), params) |> Nx.to_number()
    assert_in_delta result, normal_logpdf_0, 0.01
  end

  test "StudentT(df=1) is Cauchy" do
    # t(df=1, loc=0, scale=1) at x=0 = lgamma(1) - lgamma(0.5) - 0.5*log(pi) - log(1) - 1*log(1+0)
    # = 0 - 0.5*log(pi) - 0.5*log(pi) - 0 = -log(pi)
    expected = -:math.log(:math.pi())
    params = %{df: Nx.tensor(1.0), loc: Nx.tensor(0.0), scale: Nx.tensor(1.0)}
    result = StudentT.logpdf(Nx.tensor(0.0), params) |> Nx.to_number()
    assert_in_delta result, expected, 1.0e-5
  end

  test "StudentT support and transform" do
    params = %{df: Nx.tensor(3.0), loc: Nx.tensor(0.0), scale: Nx.tensor(1.0)}
    assert StudentT.support(params) == :real
    assert StudentT.transform(params) == nil
  end

  # ── Laplace distribution ────────────────────────────────────

  test "Laplace logpdf at mu" do
    # At x=mu: -log(2*b) - 0 = -log(2*1) = -log(2)
    expected = -:math.log(2.0)
    params = %{mu: Nx.tensor(0.0), b: Nx.tensor(1.0)}
    result = Laplace.logpdf(Nx.tensor(0.0), params) |> Nx.to_number()
    assert_in_delta result, expected, 1.0e-6
  end

  test "Laplace logpdf off-center" do
    # At x=1, mu=0, b=2: -log(4) - 1/2 = -log(4) - 0.5
    expected = -:math.log(4.0) - 0.5
    params = %{mu: Nx.tensor(0.0), b: Nx.tensor(2.0)}
    result = Laplace.logpdf(Nx.tensor(1.0), params) |> Nx.to_number()
    assert_in_delta result, expected, 1.0e-6
  end

  test "Laplace support and transform" do
    params = %{mu: Nx.tensor(0.0), b: Nx.tensor(1.0)}
    assert Laplace.support(params) == :real
    assert Laplace.transform(params) == nil
  end

  # ── Cauchy distribution ─────────────────────────────────────

  test "Cauchy logpdf at loc" do
    # At x=loc: -log(pi) - log(scale) - log(1) = -log(pi) - log(1)
    expected = -:math.log(:math.pi())
    params = %{loc: Nx.tensor(0.0), scale: Nx.tensor(1.0)}
    result = Cauchy.logpdf(Nx.tensor(0.0), params) |> Nx.to_number()
    assert_in_delta result, expected, 1.0e-6
  end

  test "Cauchy logpdf at x=1" do
    # At x=1, loc=0, scale=1: -log(pi) - log(1) - log(1+1) = -log(pi) - log(2)
    expected = -:math.log(:math.pi()) - :math.log(2.0)
    params = %{loc: Nx.tensor(0.0), scale: Nx.tensor(1.0)}
    result = Cauchy.logpdf(Nx.tensor(1.0), params) |> Nx.to_number()
    assert_in_delta result, expected, 1.0e-6
  end

  test "Cauchy support and transform" do
    params = %{loc: Nx.tensor(0.0), scale: Nx.tensor(1.0)}
    assert Cauchy.support(params) == :real
    assert Cauchy.transform(params) == nil
  end
end
