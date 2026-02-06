defmodule Exmc.Transform do
  @moduledoc """
  Minimal transform support with log-abs-det Jacobian.

  ## Examples

      iex> z = Nx.tensor(0.0)
      iex> Exmc.Transform.apply(:log, z) |> Nx.to_number() |> Float.round(6)
      1.0
      iex> Exmc.Transform.log_abs_det_jacobian(:log, z) |> Nx.to_number() |> Float.round(6)
      0.0
  """

  @doc "Forward transform: unconstrained `z` -> constrained `x`."
  def apply(nil, z), do: z

  def apply(:log, z) do
    Nx.exp(z)
  end

  def apply(:softplus, z) do
    softplus(z)
  end

  def apply(:logit, z) do
    # Numerically stable sigmoid: avoid exp overflow on BinaryBackend
    # sigmoid(z) = exp(-softplus(-z))
    Nx.exp(Nx.negate(softplus(Nx.negate(z))))
  end

  @doc "Log absolute determinant of the Jacobian of the forward transform at `z`."
  def log_abs_det_jacobian(nil, _z), do: Nx.tensor(0.0, backend: Nx.BinaryBackend)

  def log_abs_det_jacobian(:log, z) do
    # x = exp(z), |dx/dz| = exp(z), log|dx/dz| = z
    z
  end

  def log_abs_det_jacobian(:softplus, z) do
    # x = softplus(z), dx/dz = sigmoid(z)
    # log(sigmoid(z)) = -softplus(-z)
    Nx.negate(softplus(Nx.negate(z)))
  end

  def log_abs_det_jacobian(:logit, z) do
    # x = sigmoid(z), dx/dz = sigmoid(z) * (1 - sigmoid(z))
    # log|J| = log(sigmoid(z)) + log(1 - sigmoid(z))
    #        = -softplus(-z) + -softplus(z)
    Nx.add(Nx.negate(softplus(Nx.negate(z))), Nx.negate(softplus(z)))
  end

  # Numerically stable softplus: softplus(x) = log(1 + exp(x))
  # Rewritten as: x + log(1 + exp(-|x|)) — never overflows.
  defp softplus(x) do
    abs_x = Nx.abs(x)
    Nx.add(Nx.max(x, Nx.tensor(0.0)), Nx.log1p(Nx.exp(Nx.negate(abs_x))))
  end
end
