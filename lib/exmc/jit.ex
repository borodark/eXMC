defmodule Exmc.JIT do
  @moduledoc """
  Runtime JIT backend abstraction.

  Auto-detects available compilers in priority order: EXLA > Vulkan > Evaluator.

  - **EXLA**: CUDA/ROCm/CPU acceleration, f64 supported. Default on Linux.
  - **Vulkan**: Cross-platform GPU compute (FreeBSD NVIDIA, Linux NVIDIA/AMD/Intel,
    macOS via MoltenVK). f64 compute throughout. No kernel fusion in v0.1.
    Opt-in via `config :exmc, :compiler, :vulkan`.
  - **Evaluator**: Pure Elixir fallback (BinaryBackend). Very slow but always works.

  > #### EMLX (Apple Metal) is postponed {: .info}
  >
  > EMLX/MLX (Metal GPU on Apple Silicon, f32-only) support was removed for now.
  > It is postponed until real Apple hardware is available to develop and test
  > against — there is no point carrying dead conditionals for a backend we
  > cannot exercise. When such a machine is on hand, re-introduce EMLX as a
  > detected compiler here (f32 precision, f64→f32 downcast via
  > `ensure_precision/1`) and restore its `:emlx` branch in `test/test_helper.exs`.

  ## Configuration

  Override auto-detection via application config:

      config :exmc, :compiler, :exla     # force EXLA
      config :exmc, :compiler, :vulkan   # force Vulkan (GPU compute path)
      config :exmc, :compiler, :none     # disable JIT (pure Evaluator)
  """

  @doc """
  JIT-compile a function using the best available compiler.

  Accepts the same opts as `EXLA.jit/2`. When no compiler is available,
  returns the function unchanged (Evaluator path).
  """
  def jit(fun, opts \\ []) do
    case detect_compiler() do
      nil ->
        fun

      Nx.Vulkan ->
        # VulkanoBackend implements compute callbacks (binary/unary
        # SPV ops + host fallbacks). Evaluator dispatches each defn
        # op through the default backend, which is set globally to
        # VulkanoBackend at application boot.
        Nx.Defn.jit(fun, [{:compiler, Nx.Defn.Evaluator} | opts])

      compiler ->
        opts = force_host_if_no_gpu(compiler, opts)
        Nx.Defn.jit(fun, [{:compiler, compiler} | opts])
    end
  end

  @doc """
  Detect the best available JIT compiler module.

  Returns `EXLA`, `Nx.Vulkan`, or `nil`. Respects `config :exmc, :compiler` override.
  """
  def detect_compiler do
    case Application.get_env(:exmc, :compiler) do
      nil -> auto_detect()
      :exla -> if loaded?(EXLA), do: EXLA, else: auto_detect()
      :vulkan -> if loaded?(Nx.Vulkan), do: Nx.Vulkan, else: auto_detect()
      :none -> nil
    end
  end

  @doc """
  Return the Nx backend module for the detected compiler.
  """
  def backend do
    case detect_compiler() do
      EXLA -> EXLA.Backend
      Nx.Vulkan -> Nx.Vulkan.VulkanoBackend
      nil -> Nx.BinaryBackend
    end
  end

  @doc """
  Working float precision for the detected compiler.

  Returns `:f64` for EXLA/Vulkan/Evaluator. Override via
  `config :exmc, :force_precision, :f32` for the validator's
  matched-precision mode (otherwise it compares f32 Vulkan against f64 EXLA,
  masking shader correctness behind precision-gap artifacts for fat-tailed
  distributions).
  """
  def precision do
    case Application.get_env(:exmc, :force_precision) do
      :f32 -> :f32
      :f64 -> :f64
      _ -> detected_precision()
    end
  end

  defp detected_precision do
    case detect_compiler() do
      # VulkanoBackend supports f64 on every NVIDIA GPU we have tested
      # (Kepler GT 650M/750M, Ampere RTX 3060 Ti — three-host bit-exact
      # confirmation, per D86/D87). Defaulting to f64 avoids the class of
      # silent sampler collapse the regime model triggered at f32. Override
      # with `config :exmc, :force_precision, :f32` on hardware without f64
      # support or for f32 throughput.
      #
      # We intentionally do NOT gate on `Nx.Vulkan.has_f64?/0`. That NIF
      # probes a legacy context that returns false on hardware where vulkano
      # f64 works end-to-end; gating here would silently keep the f32 default
      # on working hardware. A genuinely f64-lacking device is handled by the
      # operator setting `force_precision: :f32`.
      Nx.Vulkan -> :f64
      _ -> :f64
    end
  end

  @doc """
  Downcast tensor to working precision if needed.

  When precision is forced to f32 (e.g. `config :exmc, :force_precision, :f32`),
  f64 tensors are cast to f32. Otherwise returns unchanged.
  """
  def ensure_precision(%Nx.Tensor{} = t) do
    if precision() == :f32 and Nx.type(t) == {:f, 64} do
      Nx.as_type(t, :f32)
    else
      t
    end
  end

  def ensure_precision(other), do: other

  # --- Private ---

  defp auto_detect do
    cond do
      loaded?(EXLA) -> EXLA
      # Vulkan auto-picks when EXLA is absent — the FreeBSD / non-CUDA GPU
      # path. EXLA still wins on hosts that have it (a CUDA-equipped Linux
      # box won't accidentally drop down to Vulkan).
      loaded?(Nx.Vulkan) -> Nx.Vulkan
      true -> nil
    end
  end

  defp loaded?(mod) do
    Code.ensure_loaded?(mod) and function_exported?(mod, :__info__, 1)
  end

  # When CUDA_VISIBLE_DEVICES="" (GPU hidden), force EXLA to use host client.
  # Without this, EXLA still attempts a CUDA client init which crashes
  # the EXLA.Client GenServer and cascades to all subsequent JIT calls.
  defp force_host_if_no_gpu(EXLA, opts) do
    if System.get_env("CUDA_VISIBLE_DEVICES") == "" and not Keyword.has_key?(opts, :client) do
      Keyword.put(opts, :client, :host)
    else
      opts
    end
  end

  defp force_host_if_no_gpu(_compiler, opts), do: opts
end
