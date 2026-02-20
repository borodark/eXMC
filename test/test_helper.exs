Nx.default_backend(Exmc.JIT.backend())

case Exmc.JIT.detect_compiler() do
  nil -> :ok
  compiler -> Nx.Defn.default_options(compiler: compiler)
end

ExUnit.start(exclude: [:distributed])

# Numeric compare helper for Nx tensors

defmodule Exmc.TestHelper do
  def assert_close(a, b, tol \\ 1.0e-6) do
    a_vals = to_vals(a)
    b_vals = to_vals(b)

    if length(a_vals) != length(b_vals) do
      raise ExUnit.AssertionError,
        message: "Tensor sizes differ: #{length(a_vals)} vs #{length(b_vals)}"
    end

    max_diff =
      Enum.zip(a_vals, b_vals)
      |> Enum.map(fn {x, y} -> abs(x - y) end)
      |> Enum.max(fn -> 0.0 end)

    if max_diff > tol do
      raise ExUnit.AssertionError,
        message: "Expected max diff #{max_diff} to be within #{tol}"
    end

    :ok
  end

  defp to_vals(%Nx.Tensor{} = t), do: Nx.to_flat_list(t)
  defp to_vals(n) when is_number(n), do: [n * 1.0]
end
