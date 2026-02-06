defmodule Exmc.Rewrite do
  @moduledoc """
  Rewrite pipeline for probabilistic IR.

  ## Examples

      iex> Exmc.Rewrite.pass_names() |> Enum.member?("attach_default_transforms")
      true
  """

  alias Exmc.IR

  @passes [
    Exmc.Rewrite.AttachDefaultTransforms,
    Exmc.Rewrite.LiftMeasurableMatmul,
    Exmc.Rewrite.LiftMeasurableAffine,
    Exmc.Rewrite.NormalizeObs,
    Exmc.Rewrite.PopulateObsMetadata,
    Exmc.Rewrite.NonCenteredParameterization
  ]

  @doc "Run all rewrite passes on the IR in order."
  def apply(%IR{} = ir) do
    Enum.reduce(@passes, ir, fn pass, acc -> pass.run(acc) end)
  end

  @doc "Return the ordered list of pass modules."
  def passes, do: @passes

  @doc "Return the ordered list of pass names as strings."
  def pass_names do
    Enum.map(@passes, & &1.name())
  end
end
