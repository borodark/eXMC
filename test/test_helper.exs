Nx.default_backend(Exmc.JIT.backend())

case Exmc.JIT.detect_compiler() do
  nil ->
    :ok

  Nx.Vulkan ->
    # Nx.Vulkan.Compiler auto-detects fusable elementwise chains and
    # dispatches Nx.Vulkan.fused_chain/3 instead of N separate shader
    # calls. Non-chain bodies fall through to Nx.Defn.Evaluator.
    Nx.Defn.default_options(compiler: Nx.Vulkan.Compiler)

  compiler ->
    Nx.Defn.default_options(compiler: compiler)
end

Application.ensure_all_started(:propcheck)
# Diagnostic suites tagged :diag are excluded by default — they're
# observational comparisons (e.g., fused-chain variance vs reference)
# meant to be run on demand via `mix test --include diag`. Same for
# :slow tests inside diagnostic suites that take many minutes.
ExUnit.start(exclude: [:diag, :slow])

# Backend-conditional test exclusions.
#
# When forcing precision to f32 (`config :exmc, :force_precision, :f32`),
# tests tagged :requires_f64 are skipped. Their precision-tolerance
# assertions (typically `assert_close` against finite-difference
# gradients with tol ≤ 1e-2) are tuned for f64 reverse-mode
# autodiff and reliably fail under f32.
#
# When running specifically under :vulkan, tests tagged
# :vulkan_known_failure are also skipped. These are tracked real
# bugs (not precision noise) documented in
# docs/VULKAN_KNOWN_ISSUES.md. Remove the tag once the
# corresponding issue is fixed.
#
# (EMLX/Apple Metal is postponed until real hardware is available —
# see the `Exmc.JIT` moduledoc. Restore its `:emlx` branch here then.)
#
# Note: each ExUnit.configure(exclude: ...) call replaces the exclude
# list. Carry forward the [:diag, :slow] base from ExUnit.start above.
base_excludes = [:diag, :slow]

f32? = Application.get_env(:exmc, :force_precision) == :f32

case Application.get_env(:exmc, :compiler) do
  :vulkan ->
    f64_excludes = if f32?, do: [:requires_f64], else: []
    ExUnit.configure(exclude: base_excludes ++ f64_excludes ++ [:vulkan_known_failure])

  _ ->
    ExUnit.configure(exclude: base_excludes ++ [:requires_vulkan])
end

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
