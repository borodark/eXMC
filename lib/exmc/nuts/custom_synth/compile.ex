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

  defp compile_fresh(glsl, spv_path) do
    glsl_tmp = spv_path <> ".comp"
    File.write!(glsl_tmp, glsl)

    case System.cmd("glslangValidator", ["-V", glsl_tmp, "-o", spv_path],
           stderr_to_stdout: true
         ) do
      {_out, 0} ->
        File.rm(glsl_tmp)
        {:ok, spv_path}

      {out, code} ->
        # Keep the .comp file on failure so the operator can inspect.
        {:error, %{exit: code, stderr: out, glsl_path: glsl_tmp}}
    end
  end
end
