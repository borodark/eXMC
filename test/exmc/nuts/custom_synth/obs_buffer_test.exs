defmodule Exmc.NUTS.CustomSynth.ObsBufferTest do
  @moduledoc """
  The observation buffer for a Custom likelihood, and the two refusals that
  populating it made necessary.

  A Custom likelihood can get its data two ways, and they are not equivalent:

    * CAPTURED by the closure (`fn _x, params -> ... y ...`), which is the
      convention in this repo's fixtures and throughout
      `benchmark/posteriordb`; or
    * read from the closure's first argument (`fn observed, params -> ...`),
      which is what `Exmc.Compiler` supplies on the host path.

  The second used to be silently wrong on the synth path — `compose_custom_term/3`
  passed `Nx.tensor(0.0)` — and then, once that was fixed, silently wrong for a
  second reason: the observation buffer was EMPTY, because `observed_obs_bin/1`
  walks only standard-family observed nodes and `Builder.obs` against a Custom
  RV contributes none. `obs_j = obs_inv_mass[j]` read into the inv-mass and
  capture regions instead. The conjugate oracle caught it as a frozen chain.

  The buffer is now populated from the Custom RV's own observed value. These
  tests pin that, and the two hazards it creates.
  """
  use ExUnit.Case, async: false

  alias Exmc.{Builder, Dist, IR}
  alias Exmc.NUTS.ChainShaderCodegen

  @f64 [type: :f64]

  defp xs(n), do: Enum.map(1..n, &(&1 / 4))

  defp design(n) do
    Nx.stack([Nx.broadcast(Nx.tensor(1.0, @f64), {n}), Nx.tensor(xs(n), @f64)], axis: 1)
  end

  defp response(n), do: Nx.tensor(Enum.map(xs(n), fn v -> 1.0 + 2.0 * v end), @f64)

  defp beta_prior(ir) do
    Builder.rv(ir, "beta", Dist.Normal, %{mu: Nx.tensor(0.0, @f64), sigma: Nx.tensor(5.0, @f64)},
      shape: {2}
    )
  end

  # Reads its FIRST ARGUMENT. That is the whole point of these fixtures: the
  # captured form never touches the obs region and cannot exercise any of this.
  defp reads_obs_lik(xmat) do
    Dist.Custom.new(fn observed, p ->
      r = Nx.subtract(observed, Nx.dot(xmat, p.beta))
      Nx.sum(Nx.multiply(r, r))
    end)
  end

  describe "a Custom that reads its observations now synthesises" do
    test "the response reaches the shader instead of being refused" do
      ir =
        IR.new()
        |> beta_prior()
        |> Dist.Custom.rv("Y", reads_obs_lik(design(40)), %{beta: "beta"})
        |> Builder.obs("Y_obs", "Y", response(40))

      assert {:ok, {:synthesised, _sha, _layout, _push, _spv, obs_bin, caps}} =
               ChainShaderCodegen.detect_meta(ir, [])

      # 40 doubles of response, and 80 of design matrix (two columns). Before
      # this work obs_bin was empty and the loop read the captures as if they
      # were observations.
      assert div(byte_size(obs_bin), 8) == 40
      assert div(byte_size(caps), 8) == 80
    end

    # `Builder.obs(ir, "y", "lik", Nx.tensor(0.0))` is the idiom for "this
    # Custom has no observation axis of its own". Giving that scalar a slot
    # would put a stray 0.0 at index 0 and shift every capture offset by one,
    # so it must contribute nothing.
    test "a SCALAR custom observation contributes no slot" do
      xmat = design(40)
      y = response(40)

      ir =
        IR.new()
        |> beta_prior()
        |> Dist.Custom.rv(
          "Y",
          # captures its data; the scalar obs is a placeholder
          Dist.Custom.new(fn _x, p ->
            r = Nx.subtract(y, Nx.dot(xmat, p.beta))
            Nx.sum(Nx.multiply(r, r))
          end),
          %{beta: "beta"}
        )
        |> Builder.obs("Y_obs", "Y", Nx.tensor(0.0, @f64))

      assert {:ok, {:synthesised, _s, _l, _p, _spv, obs_bin, _caps}} =
               ChainShaderCodegen.detect_meta(ir, [])

      assert obs_bin == <<>>, "a scalar placeholder must not occupy an obs slot"
    end
  end

  describe "refusals that populating the buffer made necessary" do
    # A marker that reads obs_j AND captures has two candidate trip counts:
    # `pc.n_obs`, and the capture length that `reduce_bounds/4` bakes in as a
    # literal because captures carry no runtime length. They must agree, or the
    # loop runs off one region and into the next — silently, since both live in
    # the same flat extras buffer.
    #
    # MUTATION-CHECKED by hand: with the guard disabled this model returns
    # {:ok, synthesised}.
    test "a response shorter than the design columns is refused, not truncated" do
      ir =
        IR.new()
        |> beta_prior()
        |> Dist.Custom.rv("Y", reads_obs_lik(design(40)), %{beta: "beta"})
        |> Builder.obs("Y_obs", "Y", response(20))

      assert {:unsupported, :obs_capture_length_mismatch} =
               ChainShaderCodegen.detect_meta(ir, [])
    end

    test "matching lengths are accepted, so the guard is not refusing everything" do
      ir =
        IR.new()
        |> beta_prior()
        |> Dist.Custom.rv("Y", reads_obs_lik(design(20)), %{beta: "beta"})
        |> Builder.obs("Y_obs", "Y", response(20))

      assert {:ok, _} = ChainShaderCodegen.detect_meta(ir, [])
    end

    # Two observation regions concatenated into one flat buffer, while
    # `obs_spans/1` returns `:full` for any model carrying a Custom — so the
    # Custom's marker would loop the whole thing and read the standard node's
    # values as its own.
    test "a standard observed node alongside a Custom's own observations is refused" do
      y = response(40)

      ir =
        IR.new()
        |> Builder.rv("mu", Dist.Normal, %{mu: Nx.tensor(0.0, @f64), sigma: Nx.tensor(5.0, @f64)})
        |> Builder.rv("yv", Dist.Normal, %{mu: "mu", sigma: Nx.tensor(1.0, @f64)})
        |> Builder.obs("yv_obs", "yv", y)
        |> Dist.Custom.rv(
          "Y",
          Dist.Custom.new(fn observed, p -> Nx.sum(Nx.multiply(observed, p.mu)) end),
          %{mu: "mu"}
        )
        |> Builder.obs("Y_obs", "Y", y)

      assert {:unsupported, :both_standard_and_custom_observations} =
               ChainShaderCodegen.detect_meta(ir, [])
    end
  end

  describe "the empty-obs-axis guard still fires where it should" do
    # Populating the buffer removed the case this guard was written for only
    # when the Custom HAS a vector observation. A closure that reads its first
    # argument while its obs value is a scalar placeholder still has nothing
    # behind those reads, and must still be refused.
    test "reading the obs axis with only a scalar placeholder is still refused" do
      ir =
        IR.new()
        |> beta_prior()
        |> Dist.Custom.rv("Y", reads_obs_lik(design(40)), %{beta: "beta"})
        |> Builder.obs("Y_obs", "Y", Nx.tensor(0.0, @f64))

      assert {:unsupported, :custom_reads_empty_obs_axis} =
               ChainShaderCodegen.detect_meta(ir, [])
    end
  end
end
