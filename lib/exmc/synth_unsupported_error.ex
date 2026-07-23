defmodule Exmc.SynthUnsupportedError do
  @moduledoc """
  Raised by `Exmc.Compiler.compile_for_sampling/2` when the vulkan compiler
  is active but the IR cannot be synthesised to a chain shader.

  Plan B' (see `docs/PLAN_B_PRIME_ONE_NIF.md`): the vulkan compiler path
  is now strictly synth-only. The silent fallback to per-op CPU through
  `Nx.Defn.Evaluator` (which left the production trial generating zero
  posterior updates for 68 hours, 2026-05-22 through 2026-05-24) is
  removed. If the IR doesn't match `ChainShaderCodegen.detect_meta/1`,
  this exception fires at compile time instead of failing silently at
  sample time.

  The exception carries the original IR so callers can introspect and
  decide whether to: rewrite the model into a synth-supported family,
  fall back to `:compiler, :exla`, or fall back to `:compiler, :none`.
  """
  defexception [:ir, :message]
end
