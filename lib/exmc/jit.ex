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

  ## The Nx default backend is deliberately unset

  exmc sets neither `:nx, :default_backend` nor `:nx, :default_defn_options`,
  in any environment. Eager tensors allocate on `Nx.BinaryBackend` -- nx's own
  default -- and the hot path says so out loud rather than relying on it:
  `Exmc.NUTS.Leapfrog` and `Exmc.NUTS.MassMatrix` pass
  `backend: Nx.BinaryBackend` on every scalar they build.

  Device work is reached two ways, both explicit, neither through a default:
  `Exmc.JIT.jit/2`, which names a compiler on every call, and direct NIF
  dispatch (`Exmc.NUTS.Vulkan.Dispatch`) for the fused f64 chain shader.

  This is a choice and it needs writing down because it looks exactly like an
  omission. Upstream `Nx.Vulkan.jit/2` sets the global default as a side
  effect; this module's Vulkan clause is that function with the side effect
  removed. Someone reading the two side by side will assume a line was lost.
  It was not.

  Why: a global eager backend sends every scalar constant, every mass-matrix
  accumulator and every adaptation counter across the device boundary, one
  round trip each, for values three floats wide. And the tempting setter is
  the wrong one -- `Nx.default_backend/1` writes the PROCESS DICTIONARY, so
  setting it and then fanning out over chains leaves workers on the global
  default while the log claims otherwise (docs/NX_BACKEND_HANDLING.md P5).

  What it costs, stated so it is never quoted as something else: the per-op
  Vulkan arm is `Nx.Defn.Evaluator` over `Nx.BinaryBackend`. An interpreter,
  on the CPU. Correct, slow, and not a GPU number. `describe/0` prints the
  observed Nx axes beside the derived ones so this stays observed rather than
  inferred.
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
        # This arm is an INTERPRETER over whatever backend the argument
        # tensors already carry, and in this project that is
        # `Nx.BinaryBackend` -- the CPU. It is not per-op GPU dispatch.
        #
        # The comment that stood here said the default backend "is set
        # globally to VulkanoBackend at application boot". That described the
        # function this one was copied FROM, not this one.
        # `Nx.Vulkan.jit/2` is this exact line preceded by
        # `ensure_default_backend!()`, which flips `:nx, :default_backend` to
        # VulkanoBackend so tensors created inside the traced function land on
        # the device. We copied the body and dropped that call -- deliberately,
        # see the moduledoc -- and the comment kept describing the version
        # with it.
        #
        # What is actually true: `:exmc` declares no `mod:` in mix.exs, so
        # there is no application boot to set anything; nothing in config/
        # sets `:nx, :default_backend`; and the only
        # `Nx.global_default_backend/1` call in the tree is one `setup_all` in
        # test/exmc/jit_vulkan_test.exs. So `Nx.default_backend()` is nx's own
        # default. `Nx.Defn.Evaluator` takes creation ops and constants from
        # `Nx.default_backend()` and dispatches every other op on the
        # argument's own backend, so VulkanoBackend is never asked for
        # anything on this path.
        #
        # MEASURED on super-io, 2026-09-06, `MIX_ENV=test EXMC_COMPILER=vulkan`:
        #
        #   compiler=Nx.Vulkan (configured: :vulkan) backend=Nx.Vulkan.VulkanoBackend
        #   precision=:f64 perop_fallback=true
        #   nx_default_backend={Nx.BinaryBackend, []} nx_defn_options=[]
        #   out_backend=Nx.BinaryBackend
        #
        # `out_backend` is the decisive read, not the banner: the jitted
        # result is a BinaryBackend tensor, which can only happen if every op
        # went through BinaryBackend's callbacks. The control arm
        # (`EXMC_COMPILER=none`) has derived and observed AGREEING, so the
        # disagreement above is signal and not an artifact of the fields.
        #
        # The cost of believing the old comment is on record: the Vulkan
        # benchmark arm ran two models 40x and 140x slower than EXLA and it
        # read as "the GPU is slow", when there was no fusing compiler and no
        # GPU on the path at all. See docs/NX_BACKEND_HANDLING.md P0.
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
      other -> raise ArgumentError, unknown_compiler_message(other)
    end
  end

  # config/runtime.exs validates EXMC_COMPILER and raises on a typo. This
  # clause covers the other door: `Application.put_env(:exmc, :compiler, ...)`
  # at runtime, which several harnesses and `Exmc.NUTS.Vulkan.Validator` do
  # legitimately, and which no config file can police.
  #
  # Without it an unrecognised value fell through to a bare CaseClauseError
  # naming only the value -- so `EXMC_COMPILER=vulcan` reported a pattern-match
  # failure in a case statement rather than a misspelled backend.
  defp unknown_compiler_message(other) do
    """
    Unknown :exmc, :compiler setting: #{inspect(other)}

    Expected one of :vulkan, :exla, :none, :auto, or nil for auto-detection.

    Set via `config :exmc, :compiler`, the EXMC_COMPILER environment variable
    (handled in config/runtime.exs), or Application.put_env/3.
    """
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
  A one-line description of what this process will actually compute with.

  Everything here is resolved at call time from global state, which is why it
  is worth printing rather than assuming: the compiler comes from
  `detect_compiler/0`, the precision from `precision/0` (which reads the
  VM-global `:exmc, :force_precision`), and the backend from `backend/0`.

  `test/test_helper.exs` prints this once at suite start. That line would have
  saved most of a day on 2026-08-23, when a missing LD_LIBRARY_PATH made this
  host silently Vulkan-only and nine extra test failures looked like a code
  regression.

  ## The line has two halves, and they are different kinds of claim

  `compiler=`, `backend=` and `precision=` are DERIVED: `backend/0` maps the
  detected compiler to the backend that compiler would imply, and never reads
  Nx. `nx_default_backend=` and `nx_defn_options=` are OBSERVED --
  `Nx.default_backend/0` and `Nx.Defn.default_options/0`, i.e. what this
  process will actually allocate on and compile with.

  They sit side by side because on the Vulkan arm they DISAGREE, and the
  banner used to print only the half that was wrong. Derived says
  `Nx.Vulkan.VulkanoBackend`; observed says `{Nx.BinaryBackend, []}`, because
  nothing in this project sets `:nx, :default_backend`. Config says what you
  asked for; these two say what you have, and when they disagree the second is
  what ran.

  ## `perop_fallback=` predicts whether an unsupported model raises

  Nothing in `config/` sets `:allow_vulkan_perop_sampling` any more. Until
  2026-09-12 `config/runtime.exs` set it under an explicit
  `EXMC_COMPILER=vulkan` and not under auto-detection, so on a Vulkan host
  the same suite reported one more failure depending on how it was invoked
  (measured on mac-248 and on super-io: `mix test test/custom_dist_test.exs`
  16 tests / 1 failure under auto-detect, 16 / 0 under the explicit form).
  The one test that needs the fallback now sets the flag for itself, scoped;
  a consumer that wants a refused model to degrade to per-op sampling sets
  it explicitly. So `perop_fallback=false` in this banner is the normal
  state, and a `SynthUnsupportedError` under it means what it says: the
  model has no chain-shader form, and the Plan B' guard refused it.

  ## Both observed reads are process-local

  `Nx.default_backend/1` and `Nx.Defn.default_options/1` write the PROCESS
  DICTIONARY, so this line describes the process that called it and no other.
  `test/test_helper.exs` sets both and then prints this from the same process,
  so under `mix test` the observed fields report what the HELPER process has,
  not what an ExUnit test process, a `Task` worker, or a sampling chain gets.
  Read it at face value under `mix run`; read it with that caveat under
  `mix test`.
  """
  @spec describe() :: String.t()
  def describe do
    configured = Application.get_env(:exmc, :compiler)
    perop = Application.get_env(:exmc, :allow_vulkan_perop_sampling, false)

    "compiler=#{inspect(detect_compiler())} " <>
      "(configured: #{inspect(configured)}) " <>
      "backend=#{inspect(backend())} " <>
      "precision=#{inspect(precision())} " <>
      "perop_fallback=#{inspect(perop)} " <>
      "nx_default_backend=#{inspect(Nx.default_backend())} " <>
      "nx_defn_options=#{inspect(Nx.Defn.default_options())}"
  end

  @doc """
  Working float precision for the detected compiler.

  Returns `:f64` for EXLA/Vulkan/Evaluator. Override via
  `config :exmc, :force_precision, :f32` for the validator's
  matched-precision mode (otherwise it compares f32 Vulkan against f64 EXLA,
  masking shader correctness behind precision-gap artifacts for fat-tailed
  distributions).

  (This `@doc` used to sit ABOVE `describe/0`'s, where two consecutive `@doc`
  attributes meant the first was discarded and `precision/0` shipped
  undocumented.)
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
