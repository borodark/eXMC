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

# --- application-env leak tripwire -------------------------------------------
#
# These keys decide what the sampler actually runs: which backend, and which of
# the three tree implementations. `Application.put_env/3` is global and
# VM-lifetime, so a test that sets one and does not put it back silently
# redirects every test that happens to run after it — by a different amount on
# every run, since ExUnit orders files by a random seed.
#
# That is not a hypothetical failure mode here. It cost a full backend sweep:
# `p0_correctness_test.exs` leaked `compiler: :none` and converted an arbitrary
# fraction of `EXMC_COMPILER=vulkan mix test` into another run of the
# pure-Elixir path, which reported a pass. `nuts_test.exs` "reset"
# `full_tree_nif` to `true` when its default is `false`.
#
# Ordinary assertions cannot catch this: a test that checks the keys only sees
# leaks from files that happened to run *before* it. This runs after everything
# and is therefore order-independent. Use `Exmc.TestHelper.put_env_scoped/3`
# rather than `Application.put_env/3` and it will stay quiet.
# Every `:exmc` key that gates behaviour, not a hand-picked subset — a
# tripwire that watches some of them is the same kind of half-check it exists
# to catch. Regenerate with:
#
#     grep -rho 'get_env(:exmc, :[a-z_]*' lib/ | sed 's/.*, //' | sort -u
#
# `Exmc.NUTS.Vulkan.Validator` mutates :compiler, :fused_leapfrog_meta,
# :fused_leapfrog_normal_meta and :force_precision from *library* code during
# sampling. It restores them in an `after`, so it will not trip this — but it
# is why the list cannot be "keys the tests touch".
watched_env_keys = [
  :allow_vulkan_perop_sampling,
  :chain_shader_transcendentals,
  :compiler,
  :force_precision,
  :full_tree_nif,
  :fused_leapfrog_meta,
  :fused_leapfrog_normal_meta,
  :glsl_cse,
  :gpu_node,
  :nif_depth_threshold,
  :speculative_precompute,
  :use_nif
]

env_baseline = Map.new(watched_env_keys, &{&1, Application.fetch_env(:exmc, &1)})

ExUnit.after_suite(fn _results ->
  leaked =
    Enum.filter(watched_env_keys, fn key ->
      Application.fetch_env(:exmc, key) != Map.fetch!(env_baseline, key)
    end)

  if leaked != [] do
    describe_env = fn
      {:ok, value} -> inspect(value)
      :error -> "unset"
    end

    detail =
      Enum.map_join(leaked, "\n", fn key ->
        "  :#{key}  started #{describe_env.(Map.fetch!(env_baseline, key))}, " <>
          "ended #{describe_env.(Application.fetch_env(:exmc, key))}"
      end)

    IO.puts(:stderr, """

    ================================================================
    APPLICATION ENV LEAKED OUT OF THE TEST SUITE

    #{detail}

    Some test set one of these and did not restore it, so every test
    that ran after it exercised a different backend or tree
    implementation than it claims to. The passes above are not
    trustworthy for those keys.

    Fix: use Exmc.TestHelper.put_env_scoped/3 (or delete_env_scoped/2)
    instead of Application.put_env/3.
    ================================================================
    """)

    System.at_exit(fn _ -> exit({:shutdown, 1}) end)
  end
end)

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
  @doc """
  Set an application env key for the duration of the current test, restoring
  whatever was there before when the test ends.

  `Application.put_env/3` is global and lives for the rest of the VM, and ExUnit
  orders files by a random seed. A test that sets one of these keys and does not
  put it back changes what *every test that happens to run after it* is
  exercising, and by a different amount on every run.

  That is not hypothetical here. `p0_correctness_test.exs` leaked
  `compiler: :none` and silently converted an arbitrary fraction of the backend
  sweep into another run of the pure-Elixir path. `nuts_test.exs` "reset"
  `full_tree_nif` to `true` when its default is `false`, turning the Rust
  `build_full_tree` path on globally — and per NEXT.md §5 the three tree
  implementations each carry their own copy of the doubling logic, so this
  decides which one the rest of the suite tests.

  Restoring by hand is what went wrong: it requires knowing the default
  (`speculative_precompute` is `true`, `full_tree_nif` is `false`, `use_nif` is
  `true` — not uniform, and not written down next to the tests that assume
  them) and it silently does the wrong thing when the key was never set at all.
  This reads the previous value instead of assuming it.

  Safe to call repeatedly for the same key within one test: `on_exit` callbacks
  run last-registered-first, so the nested restores unwind to the original.

      import Exmc.TestHelper

      put_env_scoped(:full_tree_nif, true)
  """
  def put_env_scoped(app \\ :exmc, key, value) do
    restore_on_exit(app, key)
    Application.put_env(app, key, value)
    :ok
  end

  @doc """
  Delete an application env key for the duration of the current test, restoring
  whatever was there before when the test ends.

  The unsetting counterpart to `put_env_scoped/3`, and it leaks the same way:
  deleting a key the next test expected to be set is as disruptive as setting
  one it expected to be absent.
  """
  def delete_env_scoped(app \\ :exmc, key) do
    restore_on_exit(app, key)
    Application.delete_env(app, key)
    :ok
  end

  defp restore_on_exit(app, key) do
    previous = Application.fetch_env(app, key)

    ExUnit.Callbacks.on_exit(fn ->
      case previous do
        {:ok, restored} -> Application.put_env(app, key, restored)
        :error -> Application.delete_env(app, key)
      end
    end)
  end

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
