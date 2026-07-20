defmodule Exmc.JIT do
  @moduledoc """
  Runtime JIT backend abstraction.

  Auto-detects available compilers in priority order: EXLA > EMLX > Evaluator.

  - **EXLA**: CUDA/ROCm/CPU acceleration, f64 supported. Default on Linux.
  - **EMLX**: Metal GPU acceleration on macOS via MLX, f32 only. Default on macOS
    when EXLA is not installed.
  - **Vulkan**: Cross-platform GPU compute (FreeBSD NVIDIA, Linux NVIDIA/AMD/Intel,
    macOS via MoltenVK). f64 compute throughout. No kernel fusion in v0.1.
    Opt-in via `config :exmc, :compiler, :vulkan`.
  - **Evaluator**: Pure Elixir fallback (BinaryBackend). Very slow but always works.

  ## Configuration

  Override auto-detection via application config:

      config :exmc, :compiler, :emlx     # force EMLX
      config :exmc, :compiler, :exla     # force EXLA
      config :exmc, :compiler, :vulkan   # force Vulkan (FreeBSD GPU path)
      config :exmc, :compiler, :none     # disable JIT

  ## EMLX Precision

  EMLX (MLX) operates in f32 only — Metal GPU has no f64 support.
  When EMLX is active, all model tensors are automatically downcast to f32.
  This is sufficient for most models but may cause numerical issues with
  very steep log-density gradients or long chains.
  """

  @doc """
  JIT-compile a function using the best available compiler.

  Accepts the same opts as `EXLA.jit/2`. When EMLX is the active compiler,
  EXLA-specific options (like `client:`) are translated to EMLX equivalents.
  When no compiler is available, returns the function unchanged (Evaluator path).
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
        opts = translate_opts(compiler, opts)
        opts = force_host_if_no_gpu(compiler, opts)
        Nx.Defn.jit(fun, [{:compiler, compiler} | opts])
    end
  end

  @doc """
  Detect the best available JIT compiler module.

  Returns `EXLA`, `EMLX`, or `nil`. Respects `config :exmc, :compiler` override.
  """
  def detect_compiler do
    case Application.get_env(:exmc, :compiler) do
      nil -> auto_detect()
      :exla -> if loaded?(EXLA), do: EXLA, else: auto_detect()
      :emlx -> if loaded?(EMLX), do: EMLX, else: auto_detect()
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
      EMLX -> EMLX.Backend
      Nx.Vulkan -> Nx.Vulkan.VulkanoBackend

      nil -> Nx.BinaryBackend
    end
  end

  @doc """
  Working float precision for the detected compiler.

  Returns `:f64` for EXLA/Evaluator, `:f32` for EMLX (Metal limitation).
  Override via `Application.put_env(:exmc, :force_precision, :f32)`
  for the W2 validator's matched-precision mode (otherwise the
  validator compares f32 Vulkan against f64 EXLA, masking shader
  correctness behind precision-gap artifacts for fat-tailed
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
      EMLX -> :f32
      # VulkanoBackend supports f64 on every NVIDIA GPU we have tested
      # (Kepler GT 650M/750M, Ampere RTX 3060 Ti — three-host bit-exact
      # confirmation via research/regime_grad_diff_mac.exs, per D87 update).
      # Defaulting to f64 avoids the class of silent sampler collapse the
      # regime model triggered at f32. Override with
      # `config :exmc, :force_precision, :f32` on hardware without f64
      # support or for f32 throughput.
      #
      # We intentionally do NOT gate on `Nx.Vulkan.has_f64?/0`. That
      # NIF probes the legacy spirit C++ path (`g_vk_ctx.has_float64`,
      # only populated by `Nx.Vulkan.init/0`) and returns false on
      # super-io — where vulkano f64 works end-to-end — because vulkano
      # never touches g_vk_ctx. Gating here would silently keep the
      # f32 default on working hardware. If a genuinely f64-lacking
      # Vulkan device shows up on the fleet, the operator sets
      # `config :exmc, :force_precision, :f32`. Fixing has_f64? to
      # route through vulkano is filed against nx_vulkan.
      Nx.Vulkan -> :f64
      _ -> :f64
    end
  end

  @doc """
  Downcast tensor to working precision if needed.

  When EMLX is active, f64 tensors are cast to f32. Otherwise returns unchanged.
  """
  def ensure_precision(%Nx.Tensor{} = t) do
    if precision() == :f32 and Nx.type(t) == {:f, 64} do
      Nx.as_type(t, :f32)
    else
      t
    end
  end

  def ensure_precision(other), do: other

  @doc """
  Check if the purpose-built MLX NIF is loaded and available.

  When true, `Exmc.MLX.Compiler` can bypass EMLX's Evaluator fallback
  for models without Custom distributions.
  """
  def mlx_nif_available? do
    Code.ensure_loaded?(Exmc.MLX.Native) and Exmc.MLX.Native.available?()
  end

  # --- Private ---

  defp auto_detect do
    cond do
      loaded?(EXLA) -> EXLA
      loaded?(EMLX) -> EMLX
      # Vulkan auto-picks when EXLA and EMLX are both absent — the FreeBSD
      # GPU path. EXLA still wins on hosts that have it (a CUDA-equipped
      # Linux box won't accidentally drop down to Vulkan).
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
  defp force_host_if_no_gpu(compiler, opts) when compiler == EXLA do
    if System.get_env("CUDA_VISIBLE_DEVICES") == "" and not Keyword.has_key?(opts, :client) do
      Keyword.put(opts, :client, :host)
    else
      opts
    end
  end

  defp force_host_if_no_gpu(_compiler, opts), do: opts

  # Translate EXLA-style opts to EMLX equivalents
  defp translate_opts(compiler, opts) when compiler == EMLX do
    case Keyword.pop(opts, :client) do
      {nil, opts} -> opts
      {:cuda, opts} -> [{:device, :gpu} | opts]
      {:host, opts} -> [{:device, :cpu} | opts]
      {_, opts} -> opts
    end
  end

  defp translate_opts(_compiler, opts), do: opts
end
