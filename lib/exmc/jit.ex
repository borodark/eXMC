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
  >
  > **Apple GPU is not blocked on that, though.** The forward plan for
  > Metal-class hardware is `nx_vulkan` (VulkanoBackend) through MoltenVK, which
  > already runs f64 on the FreeBSD fleet. If vulkano holds up on Darwin then
  > that is the Apple GPU path and EMLX does not need to come back at all.
  > Dropping EMLX also collapses the compiler matrix to **EXLA > Vulkan >
  > Evaluator**, all f64-capable, which is what took the f32/f64 fork out of the
  > hot path.

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
      :auto -> auto_detect()
      :exla -> demand(EXLA, :exla)
      :vulkan -> demand(Nx.Vulkan, :vulkan)
      :none -> nil
    end
  end

  @doc """
  Is `mod` actually usable on this host?

  "Usable" means the modules are present AND the application starts — the
  same predicate `auto_detect/0` and `demand/2` use, memoised alongside them.
  Public because callers outside this module were reaching for
  `Code.ensure_loaded?/1` instead, which is the weaker check and disagrees
  exactly when it matters: a CUDA `exla` whose NIF cannot find
  `libnvshmem_host.so.3` ships every module and answers `true`, then raises on
  first use.

  `Exmc.NUTS.Vulkan.Validator.reference/0` was one such caller. It picked
  `:exla` on the weak check and then leaked `:compiler` when the strong one
  raised — see the comment in `run_reference/2`.
  """
  @spec usable?(module()) :: boolean()
  def usable?(mod), do: loaded?(mod)

  # A named backend is a demand, not a preference.
  #
  # These clauses used to be `if loaded?(mod), do: mod, else: auto_detect()`:
  # ask for EXLA, get Vulkan, hear nothing about it. That is not a convenience,
  # it is the failure mode that makes a degraded run look like a healthy one.
  #
  # Measured, 2026-08-23. This host's CUDA `exla` could not resolve
  # `libnvshmem_host.so.3` because a non-interactive shell lacks the
  # LD_LIBRARY_PATH the pip nvshmem/nvrtc wheels need. `loaded?/1` correctly
  # judged EXLA unusable and the old clause fell through to Vulkan, so the
  # suite went from 546/1 to 549/10 — seven SynthUnsupportedError plus the
  # distributed pair — and read exactly like a code regression. It cost about
  # an hour to find, and the whole of it was that nothing said "you asked for
  # EXLA and you are not getting it".
  #
  # `config :exmc, :compiler, :auto` (or leaving it unset) is the fall-through
  # behaviour, still available and now named.
  defp demand(mod, requested) do
    if loaded?(mod) do
      mod
    else
      raise """
      Requested compiler #{inspect(requested)}, but #{inspect(mod)} is not usable on this host.

      "Not usable" means more than "not loaded": #{inspect(mod)} must be present
      AND its application must start. A dependency that ships every module and
      then fails in `Application.ensure_all_started/1` — a CUDA exla whose NIF
      cannot find libnvshmem_host.so.3 is the standard case — lands here.

      Detected alternative: #{inspect(auto_detect())}

      Either fix the backend, or ask for what you actually want:

          config :exmc, :compiler, :auto      # EXLA -> Nx.Vulkan -> Evaluator
          config :exmc, :compiler, :none      # pure-Elixir Nx.BinaryBackend

      This used to fall through silently. It does not any more, because a
      silently substituted backend produces a plausible-looking run whose
      numbers belong to a different machine than the one you think you are on.
      """
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
  @doc """
  A one-line description of what this process will actually compute with.

  Everything here is resolved at call time from global state, which is why it
  is worth printing rather than assuming: the compiler comes from
  `detect_compiler/0`, the precision from `precision/0` (which reads the
  VM-global `:exmc, :force_precision`), and the backend from `backend/0`.

  `test/test_helper.exs` prints this once at suite start. That line would have
  saved most of a day on 2026-08-23, when a missing LD_LIBRARY_PATH made this
  host silently Vulkan-only and nine extra test failures looked like a code
  regression.
  """
  @spec describe() :: String.t()
  def describe do
    configured = Application.get_env(:exmc, :compiler)

    "compiler=#{inspect(detect_compiler())} " <>
      "(configured: #{inspect(configured)}) " <>
      "backend=#{inspect(backend())} " <>
      "precision=#{inspect(precision())}"
  end

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

  # "Available" means usable, not merely on the code path. The two come apart
  # more often than you would expect: a CUDA `exla` whose NIF cannot find
  # `libnvshmem_host.so.3` compiles cleanly and ships every module, so
  # `Code.ensure_loaded?/1` says yes — and then `EXLA.Application.start/2`
  # fails and the first `jit/2` raises. Checking that the application actually
  # starts is what makes an optional backend genuinely optional.
  #
  # `exla` is declared `runtime: false` in mix.exs precisely so that this is
  # the code that starts it. A broken optional dep must not take the whole VM
  # down at boot; it must make this function return false and let
  # `auto_detect/0` fall through to the next backend.
  #
  # Memoised: a failed start logs a NIF stack trace, and once is enough.
  defp loaded?(mod) do
    key = {__MODULE__, :usable?, mod}

    case :persistent_term.get(key, :unknown) do
      :unknown ->
        usable? = probe(mod)
        :persistent_term.put(key, usable?)
        usable?

      usable? ->
        usable?
    end
  end

  defp probe(mod) do
    Code.ensure_loaded?(mod) and function_exported?(mod, :__info__, 1) and
      started?(app_for(mod))
  end

  defp app_for(EXLA), do: :exla
  defp app_for(Nx.Vulkan), do: :nx_vulkan
  defp app_for(_mod), do: nil

  defp started?(nil), do: true

  defp started?(app) do
    # Already-running apps return {:ok, []}, so this is a no-op for a backend
    # the boot sequence started normally.
    match?({:ok, _}, Application.ensure_all_started(app))
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
