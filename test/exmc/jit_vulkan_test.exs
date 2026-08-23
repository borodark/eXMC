defmodule Exmc.JITVulkanTest do
  @moduledoc """
  Phase 2 sanity: with `config :exmc, :compiler, :vulkan` set, the JIT
  layer dispatches to `Nx.Vulkan.jit/2`, runs each Nx op on the GPU
  backend, and produces results that match BinaryBackend.

  Default precision on the Vulkan path is `:f64` (D87 update / #175):
  `VulkanoBackend` supports f64 on the tested NVIDIA GPUs (Kepler
  GT 650M/750M, Ampere RTX 3060 Ti), and defaulting to f64 avoids
  the class of silent sampler collapse the regime model triggered
  at f32. Operators on genuinely f64-lacking devices flip the
  escape hatch: `config :exmc, :force_precision, :f32`.

  Tagged `:vulkan` so `mix test` skips it on hosts without Nx.Vulkan
  loaded; run with `mix test --include vulkan` after configuring the
  compiler.
  """

  use ExUnit.Case, async: false

  @moduletag :vulkan

  setup_all do
    if Code.ensure_loaded?(Nx.Vulkan) do
      previous_backend = Nx.default_backend()
      previous_opts = Application.get_env(:exmc, :compiler)

      Application.put_env(:exmc, :compiler, :vulkan)
      Nx.global_default_backend(Nx.Vulkan.VulkanoBackend)

      on_exit(fn ->
        Nx.global_default_backend(previous_backend)

        case previous_opts do
          nil -> Application.delete_env(:exmc, :compiler)
          v -> Application.put_env(:exmc, :compiler, v)
        end
      end)

      :ok
    else
      {:skip, "Nx.Vulkan not loaded"}
    end
  end

  test "Exmc.JIT.detect_compiler/0 picks Nx.Vulkan when configured" do
    assert Exmc.JIT.detect_compiler() == Nx.Vulkan

    # Post-851155b5a (D90): spirit is gone, VulkanoBackend is the only path.
    assert Exmc.JIT.backend() == Nx.Vulkan.VulkanoBackend

    assert Exmc.JIT.precision() == :f64
  end

  test "force_precision: :f32 override still works as the escape hatch" do
    Application.put_env(:exmc, :force_precision, :f32)

    try do
      assert Exmc.JIT.precision() == :f32
    after
      Application.delete_env(:exmc, :force_precision)
    end

    assert Exmc.JIT.precision() == :f64
  end

  test "Exmc.JIT.jit/2 runs a defn through the Vulkan backend" do
    f = fn a, b -> Nx.add(Nx.multiply(a, b), b) end
    a = Nx.tensor([1.0, 2.0, 3.0])
    b = Nx.tensor([4.0, 5.0, 6.0])

    out = Exmc.JIT.jit(f).(a, b)

    assert Nx.to_flat_list(out) == [8.0, 15.0, 24.0]
  end

  test "ensure_precision/1 downcasts f64 -> f32 when force_precision: :f32 is set" do
    Application.put_env(:exmc, :force_precision, :f32)

    try do
      t64 = Nx.tensor([1.5, 2.5], type: :f64, backend: Nx.Vulkan.VulkanoBackend)
      t = Exmc.JIT.ensure_precision(t64)
      assert Nx.type(t) == {:f, 32}
      assert Nx.to_flat_list(t) == [1.5, 2.5]
    after
      Application.delete_env(:exmc, :force_precision)
    end
  end

  test "ensure_precision/1 is a no-op on f64 tensors at default precision" do
    t64 = Nx.tensor([1.5, 2.5], type: :f64, backend: Nx.Vulkan.VulkanoBackend)
    t = Exmc.JIT.ensure_precision(t64)
    assert Nx.type(t) == {:f, 64}
    assert Nx.to_flat_list(t) == [1.5, 2.5]
  end
end
