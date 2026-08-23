defmodule Exmc.Transform do
  import Exmc.Math, only: [c: 2]

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
    # Clamp z to prevent overflow/underflow in gradient computation.
    # When EMLX falls back to Evaluator/BinaryBackend, Erlang arithmetic
    # throws ArithmeticError on Inf/NaN (unlike GPU which silently handles
    # IEEE 754 specials). The clamp range is chosen so that exp(z) AND its
    # gradients through Normal logpdf stay within float range.
    # f32: [-20, 20] → sigma ∈ [2e-9, 5e8], gradient safe for |x-mu| < ~1e6
    # f64: [-200, 200] → sigma ∈ [1e-87, 7e86]
    {lo, hi} = exp_safe_range()
    # Use max/min instead of Nx.clip — clip has broken gradient in Evaluator autodiff.
    safe_z = Nx.max(c(lo, z), Nx.min(z, c(hi, z)))
    Nx.exp(safe_z)
  end

  def apply(:softplus, z) do
    softplus(z)
  end

  def apply(:logit, z) do
    # Numerically stable sigmoid: sigmoid(z) = exp(-softplus(-z))
    Nx.exp(Nx.negate(softplus(Nx.negate(z))))
  end

  def apply(:stick_breaking, z) do
    # z is {K-1} unconstrained, returns {K} on simplex
    # Also handles batched {N, K-1} -> {N, K}
    stick_breaking_forward(z)
  end

  @doc "Log absolute determinant of the Jacobian of the forward transform at `z`."
  def log_abs_det_jacobian(nil, z), do: c(0.0, z)

  def log_abs_det_jacobian(:log, z) do
    # x = exp(z), |dx/dz| = exp(z), log|dx/dz| = z
    # Clamp consistently with apply(:log, z) to keep Jacobian finite.
    {lo, hi} = exp_safe_range()
    Nx.max(c(lo, z), Nx.min(z, c(hi, z)))
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

  def log_abs_det_jacobian(:stick_breaking, z) do
    stick_breaking_log_det_jacobian(z)
  end

  @doc """
  Compute the unconstrained dimension for a given transform and constrained shape.

  For most transforms, unconstrained and constrained have the same number of elements.
  For :stick_breaking, the simplex {K} maps to unconstrained {K-1}.
  """
  def unconstrained_length(nil, shape), do: shape_product(shape)
  def unconstrained_length(:log, shape), do: shape_product(shape)
  def unconstrained_length(:softplus, shape), do: shape_product(shape)
  def unconstrained_length(:logit, shape), do: shape_product(shape)

  def unconstrained_length(:stick_breaking, shape) do
    k = elem(shape, 0)
    k - 1
  end

  @doc """
  Compute the unconstrained shape for a given transform and constrained shape.
  """
  def unconstrained_shape(nil, shape), do: shape
  def unconstrained_shape(:log, shape), do: shape
  def unconstrained_shape(:softplus, shape), do: shape
  def unconstrained_shape(:logit, shape), do: shape

  def unconstrained_shape(:stick_breaking, shape) do
    k = elem(shape, 0)
    {k - 1}
  end

  # --- Stick-breaking implementation ---

  # Forward: z ∈ R^{K-1} → x ∈ Δ^K (simplex)
  # y_i = sigmoid(z_i + log(1/(K-1-i)))  [offset for better init]
  # Actually, standard stick-breaking:
  # y_i = sigmoid(z_i), x_i = y_i * remaining, remaining *= (1 - y_i)
  # x_K = remaining
  defp stick_breaking_forward(z) do
    shape = Nx.shape(z)

    case tuple_size(shape) do
      1 ->
        k_minus_1 = elem(shape, 0)
        stick_breaking_1d(z, k_minus_1)

      2 ->
        # Batched: {N, K-1} -> {N, K}
        k_minus_1 = elem(shape, 1)
        n = elem(shape, 0)
        stick_breaking_batched(z, n, k_minus_1)
    end
  end

  defp stick_breaking_1d(z, k_minus_1) do
    # Compute sigmoid of each z_i
    y = sigmoid(z)

    # Iteratively compute x_i = y_i * remaining
    {x_list, remaining} =
      Enum.reduce(0..(k_minus_1 - 1), {[], Nx.tensor(1.0)}, fn i, {acc, rem} ->
        y_i = Nx.slice(y, [i], [1]) |> Nx.reshape({})
        x_i = Nx.multiply(y_i, rem)
        new_rem = Nx.multiply(rem, Nx.subtract(Nx.tensor(1.0), y_i))
        {[x_i | acc], new_rem}
      end)

    # Last element is the remaining probability
    x_list = [remaining | x_list]
    x_list = Enum.reverse(x_list)
    Nx.stack(x_list)
  end

  defp stick_breaking_batched(z, n, k_minus_1) do
    y = sigmoid(z)

    {x_list, remaining} =
      Enum.reduce(0..(k_minus_1 - 1), {[], Nx.broadcast(Nx.tensor(1.0), {n})}, fn i, {acc, rem} ->
        y_i = Nx.slice_along_axis(y, i, 1, axis: 1) |> Nx.reshape({n})
        x_i = Nx.multiply(y_i, rem)
        new_rem = Nx.multiply(rem, Nx.subtract(Nx.tensor(1.0), y_i))
        {[x_i | acc], new_rem}
      end)

    x_list = [remaining | x_list]
    x_list = Enum.reverse(x_list)

    # Stack along axis 1: each x_i is {n}, stack to {n, K}
    x_list
    |> Enum.map(&Nx.reshape(&1, {n, 1}))
    |> Nx.concatenate(axis: 1)
  end

  # log|det J| for stick-breaking
  # log|J| = sum_i(log(y_i) + log(1-y_i) + log(remaining_i))
  # where remaining_i = prod_{j<i}(1-y_j)
  # Since y_i = sigmoid(z_i), log(y_i) = -softplus(-z_i), log(1-y_i) = -softplus(z_i)
  defp stick_breaking_log_det_jacobian(z) do
    shape = Nx.shape(z)
    k_minus_1 = elem(shape, tuple_size(shape) - 1)

    case tuple_size(shape) do
      1 ->
        stick_breaking_ladj_1d(z, k_minus_1)

      2 ->
        stick_breaking_ladj_batched(z, k_minus_1)
    end
  end

  defp stick_breaking_ladj_1d(z, k_minus_1) do
    y = sigmoid(z)

    {log_jac, _remaining} =
      Enum.reduce(0..(k_minus_1 - 1), {Nx.tensor(0.0), Nx.tensor(1.0)}, fn i, {lj, rem} ->
        z_i = Nx.slice(z, [i], [1]) |> Nx.reshape({})
        y_i = Nx.slice(y, [i], [1]) |> Nx.reshape({})
        # log|dx_i/dz_i| = log(remaining_i) + log(sigmoid(z_i)) + log(1-sigmoid(z_i))
        #                 = log(remaining_i) - softplus(-z_i) - softplus(z_i)
        log_dy = Nx.add(Nx.negate(softplus(Nx.negate(z_i))), Nx.negate(softplus(z_i)))
        contrib = Nx.add(Nx.log(rem), log_dy)
        new_rem = Nx.multiply(rem, Nx.subtract(Nx.tensor(1.0), y_i))
        {Nx.add(lj, contrib), new_rem}
      end)

    log_jac
  end

  defp stick_breaking_ladj_batched(z, k_minus_1) do
    n = elem(Nx.shape(z), 0)
    y = sigmoid(z)

    {log_jac, _remaining} =
      Enum.reduce(
        0..(k_minus_1 - 1),
        {Nx.broadcast(Nx.tensor(0.0), {n}), Nx.broadcast(Nx.tensor(1.0), {n})},
        fn i, {lj, rem} ->
          z_i = Nx.slice_along_axis(z, i, 1, axis: 1) |> Nx.reshape({n})
          y_i = Nx.slice_along_axis(y, i, 1, axis: 1) |> Nx.reshape({n})
          log_dy = Nx.add(Nx.negate(softplus(Nx.negate(z_i))), Nx.negate(softplus(z_i)))
          contrib = Nx.add(Nx.log(rem), log_dy)
          new_rem = Nx.multiply(rem, Nx.subtract(Nx.tensor(1.0), y_i))
          {Nx.add(lj, contrib), new_rem}
        end
      )

    # Sum over K-1 dimensions already done via accumulation
    # But log_jac is {n} — each element is the total ladj for that sample
    log_jac
  end

  # Inverse: x ∈ Δ^K → z ∈ R^{K-1}
  # Given x on simplex, recover z via: y_i = x_i / remaining, z_i = logit(y_i)
  def inverse_stick_breaking(x) do
    shape = Nx.shape(x)

    case tuple_size(shape) do
      1 ->
        k = elem(shape, 0)
        inverse_stick_breaking_1d(x, k)

      2 ->
        k = elem(shape, 1)
        inverse_stick_breaking_batched(x, k)
    end
  end

  defp inverse_stick_breaking_1d(x, k) do
    {z_list, _remaining} =
      Enum.reduce(0..(k - 2), {[], Nx.tensor(1.0)}, fn i, {acc, rem} ->
        x_i = Nx.slice(x, [i], [1]) |> Nx.reshape({})
        y_i = Nx.divide(x_i, rem)
        # Clamp y_i to (eps, 1-eps) for numerical stability
        eps = Nx.tensor(1.0e-10)
        y_i = Nx.max(eps, Nx.min(Nx.subtract(Nx.tensor(1.0), eps), y_i))
        z_i = Nx.subtract(Nx.log(y_i), Nx.log(Nx.subtract(Nx.tensor(1.0), y_i)))
        new_rem = Nx.subtract(rem, x_i)
        {[z_i | acc], new_rem}
      end)

    z_list = Enum.reverse(z_list)
    Nx.stack(z_list)
  end

  defp inverse_stick_breaking_batched(x, k) do
    n = elem(Nx.shape(x), 0)

    {z_list, _remaining} =
      Enum.reduce(0..(k - 2), {[], Nx.broadcast(Nx.tensor(1.0), {n})}, fn i, {acc, rem} ->
        x_i = Nx.slice_along_axis(x, i, 1, axis: 1) |> Nx.reshape({n})
        y_i = Nx.divide(x_i, rem)
        eps = Nx.tensor(1.0e-10)
        y_i = Nx.max(eps, Nx.min(Nx.subtract(Nx.tensor(1.0), eps), y_i))
        z_i = Nx.subtract(Nx.log(y_i), Nx.log(Nx.subtract(Nx.tensor(1.0), y_i)))
        new_rem = Nx.subtract(rem, x_i)
        {[z_i | acc], new_rem}
      end)

    z_list = Enum.reverse(z_list)

    z_list
    |> Enum.map(&Nx.reshape(&1, {n, 1}))
    |> Nx.concatenate(axis: 1)
  end

  # --- Helpers ---

  defp sigmoid(z) do
    Nx.exp(Nx.negate(softplus(Nx.negate(z))))
  end

  defp shape_product({}), do: 1
  defp shape_product(shape), do: Tuple.product(shape)

  # Safe clamp range for exp() so that exp(z) and its gradients through
  # logpdf computations (which involve 1/sigma and diff^2/sigma^3) stay
  # within float range. Prevents ArithmeticError on BinaryBackend.
  defp exp_safe_range do
    case Exmc.JIT.precision() do
      :f32 -> {-20.0, 20.0}
      _ -> {-200.0, 200.0}
    end
  end

  # Numerically stable softplus: softplus(x) = log(1 + exp(x))
  # Rewritten as: x + log(1 + exp(-|x|)) — never overflows.
  defp softplus(x) do
    abs_x = Nx.abs(x)
    Nx.add(Nx.max(x, c(0.0, x)), Nx.log1p(Nx.exp(Nx.negate(abs_x))))
  end
end
