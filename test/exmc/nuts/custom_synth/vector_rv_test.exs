defmodule Exmc.NUTS.CustomSynth.VectorRvTest do
  @moduledoc """
  Vector-valued RVs (`shape: {p}`) and `Nx.dot` over a captured design matrix.

  This is the regression idiom — `y ~ N(X*beta, sigma)` with one vector of
  coefficients — and until now it could not reach the fused chain shader for
  three separate reasons, none of which was the one originally reported:

    * the synth layout had one entry per RV NAME, so a `shape: {2}` RV got one
      coordinate and arrived at the closure as a scalar;
    * `Nx.dot` had no emitter clause;
    * a rank-2 capture was refused, because the extras SSBO is flat.

  The last two dissolve together: `p` is known at synthesis, so the
  contraction unrolls and each COLUMN of X is registered as an ordinary rank-1
  capture. Nothing rank-2 ever enters the buffer.

  ## What these tests actually check

  Not "it compiles". The failure mode for a mis-attributed column or a
  mis-sliced coordinate is a finite, plausible, WRONG log-density — the shape
  of defect this project has shipped more than once. So the gate is the
  composed synth density against `Exmc.Compiler`'s on random draws, which is
  the same comparison the prior-only synth path was opened on.

  ## The Custom likelihood convention these fixtures follow

  The data is CAPTURED by the closure and the first argument is ignored
  (`fn _x, params ->`). That is not a stylistic choice: `compose_custom_term/3`
  passes `Nx.tensor(0.0)` as that argument on the synth path, while
  `Exmc.Compiler` passes the real observations on the host. A closure that
  reads its first argument therefore means two different things on the two
  paths. `benchmark/posteriordb/validate_posteriordb.exs` follows the capture
  convention throughout; a fixture written the other way silently compares a
  model against a different model, which cost an afternoon here.
  """
  use ExUnit.Case, async: false

  alias Exmc.{Builder, Dist, IR}
  alias Exmc.NUTS.{ChainShaderCodegen, CustomSynth}

  @f64 [type: :f64]
  @sigma 0.3

  defp data do
    xs = Enum.map(1..40, &(&1 / 4))

    %{
      xmat: Nx.stack([Nx.broadcast(Nx.tensor(1.0, @f64), {40}), Nx.tensor(xs, @f64)], axis: 1),
      xcol: Nx.tensor(xs, @f64),
      y: Nx.tensor(Enum.map(xs, fn v -> 1.0 + 2.0 * v end), @f64)
    }
  end

  defp normal_prior(ir, id, opts \\ []) do
    Builder.rv(
      ir,
      id,
      Dist.Normal,
      %{mu: Nx.tensor(0.0, @f64), sigma: Nx.tensor(5.0, @f64)},
      opts
    )
  end

  # beta as ONE vector RV, mu via Nx.dot over the {40,2} design matrix.
  defp vector_ir(%{xmat: xmat, y: y}) do
    lik =
      Dist.Custom.new(fn _x, p ->
        r = Nx.subtract(y, Nx.dot(xmat, p.beta))
        Nx.sum(Nx.divide(Nx.multiply(r, r), -2 * @sigma * @sigma))
      end)

    IR.new()
    |> normal_prior("beta", shape: {2})
    |> Dist.Custom.rv("Y", lik, %{beta: "beta"})
    |> Builder.obs("Y_obs", "Y", y)
  end

  # The SAME posterior as two scalar RVs and rank-1 captures — the formulation
  # that already worked. Any disagreement between the two is this feature's
  # bug, not the model's.
  defp scalar_ir(%{xcol: xcol, y: y}) do
    lik =
      Dist.Custom.new(fn _x, p ->
        r = Nx.subtract(y, Nx.add(p.beta0, Nx.multiply(xcol, p.beta1)))
        Nx.sum(Nx.divide(Nx.multiply(r, r), -2 * @sigma * @sigma))
      end)

    IR.new()
    |> normal_prior("beta0")
    |> normal_prior("beta1")
    |> Dist.Custom.rv("Y", lik, %{beta0: "beta0", beta1: "beta1"})
    |> Builder.obs("Y_obs", "Y", y)
  end

  defp synth_fun(ir) do
    {:ok, comps} = CustomSynth.extract_components(Exmc.Rewrite.apply(ir, []))
    CustomSynth.MultiRvCustomSpec.compose_logp_defn(comps)
  end

  describe "a vector RV reaches the fused shader" do
    test "detect_meta synthesises, with one layout entry per coordinate" do
      assert {:ok, {:synthesised, _sha, layout, _push, _spv, _obs, _caps}} =
               ChainShaderCodegen.detect_meta(vector_ir(data()), [])

      # `d = length(layout)` is the shader's thread count, so a shape: {2} RV
      # must contribute two entries or the shader runs one thread short.
      assert layout == ["beta[0]", "beta[1]"]
    end
  end

  describe "a shape: {1} vector RV (one coefficient)" do
    # Nx builds no :stack node for a one-operand stack, so a shape: {1} RV
    # reached the emitter as a reshape of a scalar read and was refused as
    # {:unsupported_op, :param_vec} (bench/nuts_width_race.exs, d=1,
    # 2026-09-13). A vector of one is still a vector.
    defp one_column_ir(%{xcol: xcol, y: y}) do
      xmat = Nx.reshape(xcol, {40, 1})

      lik =
        Dist.Custom.new(fn _x, p ->
          r = Nx.subtract(y, Nx.dot(xmat, p.beta))
          Nx.sum(Nx.divide(Nx.multiply(r, r), -2 * @sigma * @sigma))
        end)

      IR.new()
      |> normal_prior("beta", shape: {1})
      |> Dist.Custom.rv("Y", lik, %{beta: "beta"})
      |> Builder.obs("Y_obs", "Y", y)
    end

    test "detect_meta synthesises, with one layout entry" do
      # "beta", not "beta[0]": the layout suffixes an index only when an RV
      # has more than one coordinate. d = length(layout) = 1 either way.
      assert {:ok, {:synthesised, _sha, ["beta"], _push, _spv, _obs, _caps}} =
               ChainShaderCodegen.detect_meta(one_column_ir(data()), [])
    end

    # Synthesis succeeding is not the shader working. This samples through the
    # chain shader and checks it dispatched and landed on the least-squares
    # slope (the Normal(0, 5) prior is negligible against 40 points at sigma 0.3).
    @tag :requires_vulkan
    @tag timeout: 300_000
    test "samples through the chain shader, to the right slope" do
      import Exmc.TestHelper
      put_env_scoped(:compiler, :vulkan)
      %{xcol: xcol, y: y} = d = data()

      Exmc.NUTS.Vulkan.Dispatch.reset_dispatch_count()

      {trace, _stats} =
        Exmc.NUTS.Sampler.sample(one_column_ir(d), %{},
          num_warmup: 200,
          num_samples: 400,
          seed: 5
        )

      assert Exmc.NUTS.Vulkan.Dispatch.dispatch_count() > 0

      slope =
        Nx.to_number(Nx.divide(Nx.sum(Nx.multiply(xcol, y)), Nx.sum(Nx.multiply(xcol, xcol))))

      mean = trace["beta"] |> Nx.mean() |> Nx.to_number()
      assert abs(mean - slope) < 0.02, "posterior mean #{mean}, least squares #{slope}"
    end

    test "logp and gradient agree with the host compiler" do
      d = data()
      assert_matches_host(one_column_ir(d), d.y)
    end
  end

  describe "the composed density matches the host compiler" do
    # 200 random draws, because a wrong column attribution or a mis-sliced
    # coordinate is finite and plausible rather than a crash. The gradient is
    # checked as well as the value: it is what actually drives the sampler,
    # and a density can be right where its gradient is not.
    test "vector-RV formulation: logp and gradient agree" do
      d = data()
      assert_matches_host(vector_ir(d), d.y)
    end

    test "scalar-RV control: the same comparison on the path that already worked" do
      d = data()
      assert_matches_host(scalar_ir(d), d.y)
    end
  end

  defp assert_matches_host(ir, obs) do
    synth = synth_fun(ir)
    {host_fun, pm} = Exmc.Compiler.compile(ir)

    :rand.seed(:exsss, 20_260_908)

    {worst_lp, worst_grad} =
      Enum.reduce(1..200, {0.0, 0.0}, fn _, {wl, wg} ->
        q = Nx.tensor(Enum.map(1..pm.size, fn _ -> :rand.normal() * 1.5 end), type: :f64)

        s = Nx.to_number(Nx.Defn.jit_apply(fn a, b -> synth.(a, b) end, [q, obs]))
        h = Nx.to_number(host_fun.(q))

        sg =
          Nx.Defn.jit_apply(fn a, b -> Nx.Defn.grad(a, fn aa -> synth.(aa, b) end) end, [q, obs])

        hg = Nx.Defn.jit_apply(fn a -> Nx.Defn.grad(a, host_fun) end, [q])

        {max(wl, abs(s - h) / max(abs(h), 1.0)),
         max(wg, Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(sg, hg)))))}
      end)

    # Measured 3.97e-8 on BOTH formulations, so this bound describes the
    # pre-existing synth-vs-host agreement rather than anything the vector
    # path introduces. The gradient agrees exactly.
    assert worst_lp < 1.0e-6, "worst relative logp difference #{worst_lp}"
    assert worst_grad == 0.0, "worst absolute gradient difference #{worst_grad}"
  end

  describe "the two formulations agree once dispatched" do
    @tag :requires_vulkan
    test "same posterior, vector and scalar, agree to f64 round-off" do
      d = data()
      im = Nx.tensor([1.0, 1.0], type: :f64)
      q = Nx.tensor([0.41, 1.77], type: :f64)
      p = Nx.tensor([-0.23, 0.31], type: :f64)

      buffers =
        for ir <- [vector_ir(d), scalar_ir(d)] do
          {:ok, meta} = ChainShaderCodegen.detect_meta(ir, [])
          {qc, pc, lpc, gc} = Exmc.NUTS.Vulkan.Dispatch.chain(meta, 2, 0.011, im, q, p, 8, 1)
          [qc, pc, lpc, gc]
        end

      [a, b] = buffers

      # NOT bit-identical, and it should not be: `cap0[j]*b0 + cap1[j]*b1` and
      # `b0 + xcol[j]*b1` are the same value in a different association order.
      # Measured worst 1.98e-11 on the gradient buffer.
      Enum.zip(a, b)
      |> Enum.with_index()
      |> Enum.each(fn {{x, z}, i} ->
        diff = Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(x, z))))
        assert diff < 1.0e-9, "buffer #{i} differs by #{diff}"
      end)
    end
  end
end
