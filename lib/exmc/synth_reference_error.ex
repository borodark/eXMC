defmodule Exmc.SynthReferenceError do
  @moduledoc """
  Raised when a model's parameter references cannot be resolved for the
  chain-shader path — an unknown reference, or a cycle in the hierarchy.

  Distinct from `ArgumentError` on purpose. `ChainShaderCodegen.try_synthesise/2`
  converts most failures to `:unsupported`, because synthesis is best-effort by
  contract and an emitter gap should fall back rather than crash. But a bad
  reference is a fact about the MODEL, and the message names the offending id or
  prints the cycle path — information the Plan-B' guard's generic "reshape the
  model" advice cannot reconstruct.

  It has its own type because matching on `ArgumentError` was too broad: Nx
  raises that too, and `normalize_params/1` slicing a rank-2 covariance matrix
  produced one that then escaped synthesis entirely and broke the Plan-B' guard
  for `MvNormal`. Catching a type is not the same as catching a cause.
  """
  defexception [:message]
end
