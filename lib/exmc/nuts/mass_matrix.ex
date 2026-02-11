defmodule Exmc.NUTS.MassMatrix do
  @moduledoc """
  Mass matrix adaptation via Welford online algorithm.

  Supports diagonal (`M^{-1} = Var(q)`) and dense (`M^{-1} = Cov(q)`) modes.
  Dense mode captures off-diagonal correlations critical for hierarchical models.
  """

  # ── Diagonal mode ────────────────────────────────────────────

  @doc """
  Initialize diagonal Welford state for dimension `d`.
  """
  def init(d) when is_integer(d) and d > 0 do
    %{
      mode: :diagonal,
      n: 0,
      mean: Nx.broadcast(Nx.tensor(0.0, type: :f64, backend: Nx.BinaryBackend), {d}),
      m2: Nx.broadcast(Nx.tensor(0.0, type: :f64, backend: Nx.BinaryBackend), {d})
    }
  end

  @doc """
  Initialize dense Welford state for dimension `d`.
  Tracks full covariance matrix.
  """
  def init_dense(d) when is_integer(d) and d > 0 do
    %{
      mode: :dense,
      n: 0,
      mean: Nx.broadcast(Nx.tensor(0.0, type: :f64, backend: Nx.BinaryBackend), {d}),
      m2: Nx.broadcast(Nx.tensor(0.0, type: :f64, backend: Nx.BinaryBackend), {d, d})
    }
  end

  @doc """
  Welford online update with a new sample `q` (flat f64 tensor).
  Works for both diagonal and dense modes.
  """
  def update(%{mode: :diagonal, n: n, mean: mean, m2: m2}, q) do
    new_n = n + 1
    delta = Nx.subtract(q, mean)

    new_mean =
      Nx.add(
        mean,
        Nx.divide(delta, Nx.tensor(new_n * 1.0, type: :f64, backend: Nx.BinaryBackend))
      )

    delta2 = Nx.subtract(q, new_mean)
    new_m2 = Nx.add(m2, Nx.multiply(delta, delta2))

    %{mode: :diagonal, n: new_n, mean: new_mean, m2: new_m2}
  end

  def update(%{mode: :dense, n: n, mean: mean, m2: m2}, q) do
    new_n = n + 1
    delta = Nx.subtract(q, mean)

    new_mean =
      Nx.add(
        mean,
        Nx.divide(delta, Nx.tensor(new_n * 1.0, type: :f64, backend: Nx.BinaryBackend))
      )

    delta2 = Nx.subtract(q, new_mean)
    # Outer product for covariance accumulation
    new_m2 = Nx.add(m2, outer(delta, delta2))

    %{mode: :dense, n: new_n, mean: new_mean, m2: new_m2}
  end

  @doc """
  Finalize diagonal Welford: return `inv_mass_diag` (posterior variance).
  Returns identity if fewer than 3 samples.
  """
  def finalize(%{mode: :diagonal, n: n, mean: mean}) when n < 3 do
    {d} = Nx.shape(mean)
    Nx.broadcast(Nx.tensor(1.0, type: :f64, backend: Nx.BinaryBackend), {d})
  end

  def finalize(%{mode: :diagonal, n: n, m2: m2}) do
    variance = Nx.divide(m2, Nx.tensor((n - 1) * 1.0, type: :f64, backend: Nx.BinaryBackend))
    floor = Nx.tensor(1.0e-6, type: :f64, backend: Nx.BinaryBackend)
    variance = Nx.max(variance, floor)

    # Stan-style regularization: shrink toward 1e-3 * I
    alpha = 5.0 / (n + 5.0)

    Nx.add(
      Nx.multiply(Nx.tensor(1.0 - alpha, type: :f64, backend: Nx.BinaryBackend), variance),
      Nx.multiply(
        Nx.tensor(alpha, type: :f64, backend: Nx.BinaryBackend),
        Nx.tensor(1.0e-3, type: :f64, backend: Nx.BinaryBackend)
      )
    )
  end

  @doc """
  Finalize dense Welford: return `%{cov: Nx.t(), chol_cov: Nx.t()}`.

  `cov` is the covariance matrix (= M^{-1}). `chol_cov` is its lower Cholesky factor.
  Returns identity if fewer than 3 samples.
  """
  def finalize_dense(%{mode: :dense, n: n, mean: mean}) when n < 3 do
    {d} = Nx.shape(mean)
    eye = Nx.eye(d, type: :f64)
    %{cov: eye, chol_cov: eye}
  end

  def finalize_dense(%{mode: :dense, n: n, m2: m2}) do
    cov = Nx.divide(m2, Nx.tensor((n - 1) * 1.0, type: :f64, backend: Nx.BinaryBackend))
    {d, _} = Nx.shape(cov)

    # Shrink toward sample diagonal — preserves marginal variances,
    # reduces noisy off-diagonal correlation estimates.
    # Stronger than scalar regularization: ensures positive definiteness
    # even with few samples relative to dimension.
    alpha = 5.0 / (n + 5.0)
    # Extract diagonal as a diagonal matrix
    diag_cov = Nx.multiply(Nx.take_diagonal(cov), Nx.eye(d, type: :f64))
    # Floor the diagonal at 1e-6
    diag_floored =
      Nx.max(
        diag_cov,
        Nx.multiply(
          Nx.tensor(1.0e-6, type: :f64, backend: Nx.BinaryBackend),
          Nx.eye(d, type: :f64)
        )
      )

    cov =
      Nx.add(
        Nx.multiply(Nx.tensor(1.0 - alpha, type: :f64, backend: Nx.BinaryBackend), cov),
        Nx.multiply(Nx.tensor(alpha, type: :f64, backend: Nx.BinaryBackend), diag_floored)
      )

    chol_cov = Nx.LinAlg.cholesky(cov)
    %{cov: cov, chol_cov: chol_cov}
  end

  @doc """
  Reset Welford state (same as init, preserves mode).
  """
  def reset(%{mode: :diagonal, mean: mean}) do
    {d} = Nx.shape(mean)
    init(d)
  end

  def reset(%{mode: :dense, mean: mean}) do
    {d} = Nx.shape(mean)
    init_dense(d)
  end

  def reset(d) when is_integer(d) and d > 0, do: init(d)

  # ── Helpers ──────────────────────────────────────────────────

  # Outer product: a (d,) x b (d,) -> (d,d)
  defp outer(a, b) do
    a_col = Nx.reshape(a, {:auto, 1})
    b_row = Nx.reshape(b, {1, :auto})
    Nx.dot(a_col, b_row)
  end
end
