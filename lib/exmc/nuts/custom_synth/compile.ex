defmodule Exmc.NUTS.CustomSynth.Compile do
  @moduledoc """
  GLSL → SPIR-V via `glslangValidator`, content-addressed on disk.

  This is R2.4's piece in the synthesis pipeline.  Sibling to
  `Nx.Vulkan.Synthesis.compile/1` (which takes a `FamilySpec` and
  renders before compiling): we take a fully-rendered GLSL string
  produced by `Exmc.NUTS.CustomSynth` and skip the template-render
  step.

  Same cache directory as `Nx.Vulkan.Synthesis` — content hashes
  collide cleanly because they're SHA-256 of the final GLSL text.
  """

  @cache_dir Path.expand("~/.exmc/gpu_node/spv")

  @doc """
  Compile a GLSL source string to a cached SPIR-V file.

  Returns `{:ok, spv_path}` on success or
  `{:error, %{exit: code, stderr: output}}` on glslangValidator
  failure.
  """
  @spec compile_glsl(binary()) :: {:ok, Path.t()} | {:error, map()}
  def compile_glsl(glsl) when is_binary(glsl) do
    hash = :crypto.hash(:sha256, glsl) |> Base.encode16(case: :lower)
    spv_path = Path.join(@cache_dir, "synth_#{hash}.spv")

    if File.exists?(spv_path) do
      {:ok, spv_path}
    else
      File.mkdir_p!(@cache_dir)
      compile_fresh(glsl, spv_path)
    end
  end

  # Compiles to caller-unique temporary paths and renames the finished module
  # into place. `File.rename/2` is atomic within a filesystem, so a concurrent
  # `File.exists?/1` above sees either no file or a complete one.
  #
  # Writing `glslangValidator -o` straight to `spv_path` was a race, and not a
  # theoretical one. Measured at 24 concurrent compiles of the same shader over
  # 40 rounds: **50 of 960 callers received `{:ok, spv_path}` for a zero-byte
  # file**. `glslangValidator` creates its output before it writes it, so the
  # existence check above returned true for a file with no contents yet, and
  # the caller handed that path to the NIF. Ten-plus test modules are
  # `async: true` and sample, so this is reachable from an ordinary `mix test`.
  #
  # The temp *source* path was shared for the same reason, which is worse: one
  # caller's `File.rm(glsl_tmp)` could delete the source out from under another
  # caller's still-running validator, and a validator that fails partway can
  # take its `-o` target with it — the likely origin of the intermittent
  # `read spv: No such file or directory` seen under `EXMC_COMPILER=vulkan`.
  # Both paths are now per-caller.
  defp compile_fresh(glsl, spv_path) do
    unique = "#{System.unique_integer([:positive])}"
    glsl_tmp = "#{spv_path}.#{unique}.comp"
    spv_tmp = "#{spv_path}.#{unique}.tmp"

    File.write!(glsl_tmp, glsl)

    case System.cmd("glslangValidator", ["-V", glsl_tmp, "-o", spv_tmp], stderr_to_stdout: true) do
      {_out, 0} ->
        # Last writer wins, and every writer produces identical bytes — the
        # path is a SHA-256 of the source this module just compiled.
        File.rename!(spv_tmp, spv_path)
        File.rm(glsl_tmp)
        {:ok, spv_path}

      {out, code} ->
        # Keep the .comp file on failure so the operator can inspect. Drop the
        # partial output: leaving it would publish a truncated module under a
        # name that claims to be a complete one.
        File.rm(spv_tmp)
        {:error, %{exit: code, stderr: out, glsl_path: glsl_tmp}}
    end
  end
end
