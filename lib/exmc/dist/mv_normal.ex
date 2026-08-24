defmodule Exmc.Dist.MvNormal do
  import Exmc.Math, only: [c: 2]

  @moduledoc """
  Multivariate Normal distribution.

  Parameterized by mean vector `mu` ({d}) and covariance matrix `cov` ({d, d}).
  Internally pre-computes precision matrix and log-determinant for gradient-safe logpdf.

  ## Examples

      iex> mu = Nx.tensor([0.0, 0.0])
      iex> cov = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
      iex> x = Nx.tensor([0.0, 0.0])
      iex> Exmc.Dist.MvNormal.logpdf(x, %{mu: mu, cov: cov}) |> Nx.to_number() |> Float.round(4)
      -1.8379
  """

  @behaviour Exmc.Dist

  @impl true
  def logpdf(x, %{mu: mu, prec: prec, log_det_cov: log_det_cov}) do
    # Gradient-friendly path: only uses Nx.dot, Nx.subtract, Nx.multiply, Nx.add
    d = elem(Nx.shape(mu), 0)
    diff = Nx.subtract(x, mu)
    mahal = Nx.dot(diff, Nx.dot(prec, diff))
    log_2pi = c(:math.log(2.0 * :math.pi()), x)

    Nx.multiply(
      c(-0.5, x),
      Nx.add(Nx.add(Nx.multiply(Nx.tensor(d * 1.0), log_2pi), log_det_cov), mahal)
    )
  end

  def logpdf(x, %{mu: _mu, cov: _cov} = params) do
    logpdf(x, prepare_params(params))
  end

  @doc """
  Pre-compute precision matrix and log-determinant from covariance.
  Call this eagerly before gradient tracing to avoid LinAlg ops in the traced function.
  """
  def prepare_params(%{mu: mu, cov: cov}) do
    # `Nx.LinAlg.cholesky`, `Nx.take_diagonal`, and `Nx.LinAlg.invert`
    # all create internal scratch tensors on the default backend
    # (Vulkano under D88 f64 defaults); mixed with BinaryBackend
    # inputs they crash in `BinaryBackend.gather` / `to_binary`.
    # Force BinaryBackend for the eager linalg pre-compute.
    Nx.with_default_backend(Nx.BinaryBackend, fn ->
      l = Nx.LinAlg.cholesky(cov)
      log_det_cov = Nx.multiply(Nx.tensor(2.0), Nx.sum(Nx.log(Nx.take_diagonal(l))))
      prec = Nx.LinAlg.invert(cov)
      %{mu: mu, prec: prec, log_det_cov: log_det_cov}
    end)
  end

  def prepare_params(%{mu: _, prec: _, log_det_cov: _} = params), do: params

  @impl true
  def support(_params), do: :real

  @impl true
  def transform(_params), do: nil

  @impl true
  def sample(%{mu: mu, cov: cov}, rng) do
    Nx.with_default_backend(Nx.BinaryBackend, fn ->
      d = elem(Nx.shape(mu), 0)
      l = Nx.LinAlg.cholesky(cov)

      {z_list, rng} =
        Enum.map_reduce(1..d, rng, fn _i, r ->
          :rand.normal_s(r)
        end)

      z = Nx.tensor(z_list, type: :f64)
      x = Nx.add(mu, Nx.dot(l, z))
      {x, rng}
    end)
  end

  def sample(%{mu: _, prec: _, log_det_cov: _}, _rng) do
    raise ArgumentError, "MvNormal.sample requires :cov in params (not pre-computed :prec)"
  end
end
