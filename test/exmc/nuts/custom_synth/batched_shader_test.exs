defmodule Exmc.NUTS.CustomSynth.BatchedShaderTest do
  @moduledoc """
  The batched chain shader computes the same thing the single-instance one
  does, per instance, bit for bit.

  ## What this replaces

  `MultiRvCustomSpec.render_batched/1` emitted an f32 shader — `float eps`,
  `float q_init[]` — while its own reduce-sum rewriter emitted `double`
  accumulators into it. GLSL has no implicit double-to-float conversion, so
  any model with a vector observation failed in `glslangValidator`. The
  narrower models that did compile would have been bound to buffers the f64
  batch NIF writes at 8 bytes an element and read at 4.

  None of that surfaced, because `leapfrog_chain_synth_batch_f64/6` did not
  exist and nothing ever dispatched the result. The batched renderer was also
  a copy of `render_with_custom/1` that had drifted from it: no obs spans (so
  a model with several observed nodes would have counted its whole likelihood
  once per node — the defect `docs/OPEN_VULKAN_OBSERVED_MODEL.md` describes),
  no common-subexpression pass, and no f64 transcendental rewrite.

  Both variants now render from one emitter. The template and the observation
  index expression are the only things that differ, which is what makes the
  bit-identity assertions below a reasonable thing to demand rather than a
  hope about two independent implementations agreeing.

  ## Why bit-identity and not a tolerance

  A tolerance would pass for a shader that reads a neighbouring instance's
  inverse mass, or sums an observation slice one element off, or drops the
  last leapfrog step. Those are the failures this file exists to catch, and
  every one of them produces numbers that are individually plausible. The
  batched instance and the lone chain execute the same operations in the same
  order on the same inputs, so anything short of identical output is a bug in
  the indexing, and the comparison should say so.

  Confirmed non-vacuous by mutation: changing the shader's per-instance
  extras stride from `inst * (n_obs + d)` to `inst * d` leaves instance 0
  matching (its offset is 0 either way) and breaks instances 1 and 2 on all
  four output buffers.
  """
  use ExUnit.Case, async: false

  alias Exmc.Builder
  alias Exmc.Dist.Normal
  alias Exmc.NUTS.CustomSynth
  alias Exmc.NUTS.CustomSynth.MultiRvCustomSpec
  alias Exmc.NUTS.Vulkan.Dispatch

  @k 8
  @eps 0.031

  # mu_i ~ Normal(0, 2); y_i ~ Normal(mu_i, 1); observe y_i = v_i.
  defp conjugate_ir(vs) do
    vs
    |> Enum.with_index(1)
    |> Enum.reduce(Builder.new_ir(), fn {v, i}, acc ->
      acc
      |> Builder.rv("mu#{i}", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(2.0)})
      |> Builder.rv("y#{i}", Normal, %{mu: "mu#{i}", sigma: Nx.tensor(1.0)})
      |> Builder.obs("y#{i}_obs", "y#{i}", Nx.tensor(v))
    end)
  end

  # One RV, one vector observation: the shader keeps the runtime `pc.n_obs`
  # bound rather than baking per-node spans, so it exercises the other arm of
  # `obs_spans/1`.
  defp vector_obs_ir(values) do
    Builder.new_ir()
    |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(2.0)})
    |> Builder.rv("y", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
    |> Builder.obs("y_obs", "y", Nx.tensor(values))
  end

  # No observations at all. n_obs is 0 and the extras buffer is inverse mass
  # only, so `extras_off` degenerates to `inst * d`.
  defp prior_only_ir do
    Builder.new_ir()
    |> Builder.rv("a", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
    |> Builder.rv("b", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(2.0)})
  end

  defp components(ir) do
    {:ok, comps} = CustomSynth.extract_components(Exmc.Rewrite.apply(ir, []))
    comps
  end

  # Deliberately divergent per instance: different position, different
  # momentum, different inverse mass. Two instances that happened to agree
  # would make an isolation failure invisible.
  defp instance_inputs(i, d, obs) do
    q = Nx.tensor(for(j <- 1..d, do: 0.1 * i + 0.037 * j), type: :f64)
    p = Nx.tensor(for(j <- 1..d, do: -0.2 * i + 0.011 * j), type: :f64)
    im = Nx.tensor(for(j <- 1..d, do: 0.5 + 0.25 * rem(i + j, 3)), type: :f64)
    {q, p, im, obs}
  end

  # Everything after the shader's entry point. The f64 transcendental
  # helpers sit above it and legitimately contain f32 casts, so assertions
  # about precision inside the leapfrog have to be scoped past them.
  defp main_body(glsl) do
    [_, body] = String.split(glsl, "void main() {", parts: 2)
    body
  end

  # The lines the emitter produced, as opposed to the template's own. Every
  # one of them is a gradient or log-density assignment, a reduce-sum loop,
  # an accumulator, or a hoisted common subexpression; the templates contain
  # no line that starts this way.
  defp emitted_lines(glsl) do
    glsl
    |> String.split("\n")
    |> Enum.map(&String.trim/1)
    |> Enum.filter(
      &(&1 =~
          ~r/^(grad_q = |grad_qn = |lp_i = |double _|for \(uint j = |_lpacc|_gacc|if \(tid == \d+u\) \{ )/)
    )
  end

  defp obs_tensor(<<>>), do: nil
  defp obs_tensor(bin), do: Nx.from_binary(bin, :f64, backend: Nx.BinaryBackend)

  defp assert_all_four_identical({sq, sp, sl, sg}, {bq, bp, bl, bg}, label) do
    for {single, batched, name} <- [
          {sq, bq, "q_chain"},
          {sp, bp, "p_chain"},
          {sl, bl, "logp_chain"},
          {sg, bg, "grad_chain"}
        ] do
      assert Nx.shape(single) == Nx.shape(batched),
             "#{label}: #{name} shape #{inspect(Nx.shape(batched))} batched vs " <>
               "#{inspect(Nx.shape(single))} single"

      assert Nx.to_binary(single) == Nx.to_binary(batched),
             "#{label}: #{name} differs. max|delta| = " <>
               "#{Nx.subtract(single, batched) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()}"
    end
  end

  describe "the emitted GLSL" do
    test "declares f64 buffers and an f64 eps, not f32" do
      {:ok, glsl} = MultiRvCustomSpec.render_batched(components(conjugate_ir([3.0, -2.0])))

      # The f32 template's exact declarations. Each one would have been bound
      # to a buffer the f64 batch NIF writes at 8 bytes an element.
      refute glsl =~ "float eps;"
      refute glsl =~ "buffer In_q     { float q_init[]; }"
      refute glsl =~ "shared float"

      assert glsl =~ "double eps;"
      assert glsl =~ "buffer In_q     { double q_init[]; }"
      assert glsl =~ "buffer Out_logp { double logp_chain[]; }"
      assert glsl =~ "shared double q_shared[256];"

      # GLSL.std.450 has no f64 log/exp, so an f64 shader that called `log(`
      # on a double would not compile. The batched renderer used to skip the
      # rewrite that turns those calls into the helpers, and to omit the
      # helpers themselves.
      assert glsl =~ "double log_d(double x)"
      assert glsl =~ "double exp_d(double x)"

      # Only inside the helpers' own f32 boundary casts, never in main.
      body = main_body(glsl)
      refute body =~ ~r/(?<![_a-zA-Z])log\(/
      refute body =~ ~r/(?<![_a-zA-Z])exp\(/

      # And no f32 anywhere in the leapfrog itself. The f32 template declared
      # `float qi`, `float grad_q` and multiplied by a `0.5` with no suffix.
      refute body =~ ~r/(?<![_a-zA-Z])float[ (]/
    end

    test "offsets every buffer read and write by the workgroup's instance" do
      {:ok, glsl} = MultiRvCustomSpec.render_batched(components(conjugate_ir([3.0, -2.0])))

      assert glsl =~ "uint inst = gl_WorkGroupID.x;"
      assert glsl =~ "uint extras_off = inst * (pc.n_obs + pc.d);"
      assert glsl =~ "q_init[q_off + tid]"
      assert glsl =~ "obs_inv_mass[extras_off + pc.n_obs + tid]"
      assert glsl =~ "q_chain[chain_off + k * pc.d + tid]"
      assert glsl =~ "logp_chain[logp_off + k]"

      # A scalar observation is inlined as a constant, so a conjugate model
      # emits no reduce-sum loop at all and this says nothing about it. The
      # loops appear for a VECTOR observation, and they are what a duplicated
      # emitter got wrong.
      {:ok, vec} = MultiRvCustomSpec.render_batched(components(vector_obs_ir([1.0, 2.0, 3.0])))

      assert vec =~ "double obs_j = obs_inv_mass[extras_off + j];"
      refute vec =~ "double obs_j = obs_inv_mass[j];"
    end

    test "emits the same bodies as the single-instance renderer" do
      # This is the property the whole design rests on, asserted directly:
      # the two variants differ in their template and in one index
      # expression, and in nothing else. Everything the emitter produces —
      # per-tid gradient and log-density assignments, reduce-sum loops, their
      # accumulators, and the hoisted common subexpressions — must come out
      # character for character the same.
      #
      # It is what the bit-identity dispatch tests below rest on too, and it
      # is cheap enough to run for every model shape. The renderer that used
      # to live here passed nothing like it: it dropped the obs spans and the
      # CSE pass, both of which change these lines.
      for ir <- [
            prior_only_ir(),
            conjugate_ir([3.0]),
            conjugate_ir([3.0, -2.0, 1.5]),
            vector_obs_ir([1.0, 2.0, 3.0])
          ] do
        comps = components(ir)
        {:ok, single, _captures} = MultiRvCustomSpec.render(comps)
        {:ok, batched} = MultiRvCustomSpec.render_batched(comps)

        expected =
          single
          |> emitted_lines()
          |> Enum.map(&String.replace(&1, "obs_inv_mass[j]", "obs_inv_mass[extras_off + j]"))

        assert emitted_lines(batched) == expected

        assert expected != [],
               "no emitted body lines to compare for #{inspect(ir.nodes |> Map.keys())}"
      end
    end

    test "a single vector-observed node keeps the runtime obs bound" do
      # It owns the whole buffer by definition, so the loop stays bounded by
      # pc.n_obs and one SPV serves every dataset size.
      {:ok, vec} = MultiRvCustomSpec.render_batched(components(vector_obs_ir([1.0, 2.0, 3.0])))
      assert vec =~ "for (uint j = 0u; j < pc.n_obs; j++)"
    end

    test "a prior-only model renders instead of being refused" do
      # This used to return {:error, :prior_only_batched_not_supported} for
      # every model with `custom: nil` — which is every conjugate model built
      # through Builder.obs, i.e. the entire class the batch coordinator
      # exists to serve.
      assert {:ok, glsl} = MultiRvCustomSpec.render_batched(components(prior_only_ir()))
      assert glsl =~ "uint inst = gl_WorkGroupID.x;"

      assert {:ok, _} = MultiRvCustomSpec.render_batched(components(conjugate_ir([3.0])))
    end
  end

  describe "compiling the batched shader" do
    @describetag :requires_vulkan

    test "glslangValidator accepts it for every model shape" do
      for {name, ir} <- [
            {"prior-only", prior_only_ir()},
            {"one scalar obs", conjugate_ir([3.0])},
            {"three scalar obs", conjugate_ir([3.0, -2.0, 1.5])},
            {"vector obs", vector_obs_ir([1.0, 2.0, 3.0, 4.0])},
            {"eight scalar obs", conjugate_ir([3.0, -2.0, 1.5, 0.5, -1.0, 2.5, -0.5, 4.0])}
          ] do
        {:ok, glsl} = MultiRvCustomSpec.render_batched(components(ir))

        assert {:ok, spv} = CustomSynth.Compile.compile_glsl(glsl),
               "the batched shader for #{name} does not compile"

        assert File.exists?(spv)
      end
    end

    test "synthesise_batched/1 sizes n_obs from the observed nodes" do
      # It used to read `ir.data` and default to 1 otherwise, which is every
      # Builder.obs model. n_obs is the per-instance extras stride
      # (`inst * (n_obs + d)`), so a wrong value does not fail — it points
      # instance 1 into the middle of instance 0's slice.
      {:ok, {:synthesised, _, layout, push_spec, _, obs_bin, _capt}} =
        CustomSynth.synthesise_batched(conjugate_ir([3.0, -2.0, 1.5]))

      assert push_spec.n_obs == 3
      assert length(layout) == 3

      # Each instance brings its own observations at dispatch time, so the
      # batched meta carries none.
      assert obs_bin == <<>>

      {:ok, {:synthesised, _, _, vec_spec, _, _, _}} =
        CustomSynth.synthesise_batched(vector_obs_ir([1.0, 2.0, 3.0, 4.0]))

      assert vec_spec.n_obs == 4
    end
  end

  describe "a batched instance equals the same chain dispatched alone" do
    @describetag :requires_vulkan

    setup do
      # Non-vacuity: if this VM cannot dispatch a batch at all, every
      # comparison below would be comparing nothing. The NIF is present in
      # nx_vulkan, but a host can carry a stale priv/native that
      # `mix deps.compile` will not replace.
      assert Code.ensure_loaded?(Nx.Vulkan.NativeV) and
               function_exported?(Nx.Vulkan.NativeV, :leapfrog_chain_synth_batch_f64, 6),
             "leapfrog_chain_synth_batch_f64/6 is not exported by this build of " <>
               "nx_vulkan. Rebuild with `mix deps.clean nx_vulkan --build && " <>
               "mix deps.compile nx_vulkan` — `mix deps.compile` alone will not " <>
               "replace a stale artifact."

      :ok
    end

    for {name, ir_fun, n_inst} <- [
          {"prior-only, 4 instances", {:prior_only, []}, 4},
          {"one RV, 4 instances", {:conjugate, [3.0]}, 4},
          {"three RVs, 5 instances", {:conjugate, [3.0, -2.0, 1.5]}, 5},
          {"vector obs, 3 instances", {:vector, [1.0, 2.0, 3.0, 4.0]}, 3},
          {"eight RVs, 6 instances", {:conjugate, [3.0, -2.0, 1.5, 0.5, -1.0, 2.5, -0.5, 4.0]}, 6}
        ] do
      test "bit-identical on all four buffers: #{name}" do
        ir =
          case unquote(Macro.escape(ir_fun)) do
            {:prior_only, _} -> prior_only_ir()
            {:conjugate, vs} -> conjugate_ir(vs)
            {:vector, vs} -> vector_obs_ir(vs)
          end

        n_inst = unquote(n_inst)

        {:ok, {:synthesised, _, layout, _, _, obs_bin, _capt} = single} = CustomSynth.synthesise(ir)
        {:ok, batched} = CustomSynth.synthesise_batched(ir)

        d = length(layout)
        obs = obs_tensor(obs_bin)
        instances = for i <- 1..n_inst, do: instance_inputs(i, d, obs)

        singles =
          for {q, p, im, _} <- instances do
            Dispatch.chain(single, d, @eps, im, q, p, @k, 1)
          end

        batch = Dispatch.chain_batch(batched, instances, @k, 1, @eps)

        assert length(batch) == n_inst

        for {{s, b}, i} <- Enum.zip(singles, batch) |> Enum.with_index() do
          assert_all_four_identical(s, b, "instance #{i}")
        end

        # Non-vacuity for the comparison itself: identical inputs would make
        # every instance agree with every other, and a shader that ignored
        # `inst` entirely would pass. The inputs above differ per instance,
        # so the outputs must too.
        first_q = singles |> Enum.map(fn {q, _, _, _} -> Nx.to_binary(q) end)

        assert length(Enum.uniq(first_q)) == n_inst,
               "the #{n_inst} instances produced fewer than #{n_inst} distinct " <>
                 "chains, so agreeing with each other proves nothing"
      end
    end

    test "a model too wide for the old push tail still batches" do
      # `chain_batch/5` used to append one f64 per prior parameter to its
      # header and then check the total against the NIF's 128-byte bound.
      # Sixteen Normal priors is 32 floats — 256 B of tail on a 24 B header —
      # so this model raised before it ever reached the GPU, and the
      # coordinator's rescue turned that into an unbatched fallback.
      #
      # Nothing read the tail. The batched shader bakes prior parameters into
      # its SPIR-V as literals exactly as the single-instance one does, and
      # the NIF pushes sizeof(PushBlockBatchF64) = 24 bytes and drops the
      # rest. It was the same defect that cost the single-instance path 13.1x
      # on an 8-RV model; see push_width_test.exs.
      ir = conjugate_ir(Enum.map(1..16, &(&1 * 0.5 - 4.0)))

      {:ok, {:synthesised, _, layout, _, _, obs_bin, _capt} = single} = CustomSynth.synthesise(ir)
      {:ok, batched} = CustomSynth.synthesise_batched(ir)

      d = length(layout)
      assert d == 16

      obs = obs_tensor(obs_bin)
      instances = for i <- 1..3, do: instance_inputs(i, d, obs)

      singles =
        for {q, p, im, _} <- instances,
            do: Dispatch.chain(single, d, @eps, im, q, p, @k, 1)

      batch = Dispatch.chain_batch(batched, instances, @k, 1, @eps)

      for {{s, b}, i} <- Enum.zip(singles, batch) |> Enum.with_index() do
        assert_all_four_identical(s, b, "16-RV instance #{i}")
      end
    end

    test "n_instances = 1 matches the single-instance path exactly" do
      ir = conjugate_ir([3.0, -2.0])
      {:ok, {:synthesised, _, layout, _, _, obs_bin, _capt} = single} = CustomSynth.synthesise(ir)
      {:ok, batched} = CustomSynth.synthesise_batched(ir)

      d = length(layout)
      inst = instance_inputs(1, d, obs_tensor(obs_bin))
      {q, p, im, _} = inst

      expected = Dispatch.chain(single, d, @eps, im, q, p, @k, 1)
      [actual] = Dispatch.chain_batch(batched, [inst], @k, 1, @eps)

      assert_all_four_identical(expected, actual, "n_instances = 1")
    end

    test "the sign of dir_sign carries through a batch" do
      # eps is signed by the caller's direction on both paths; a batch that
      # dropped it would still return finite, plausible trajectories.
      ir = conjugate_ir([3.0, -2.0])
      {:ok, {:synthesised, _, layout, _, _, obs_bin, _capt} = single} = CustomSynth.synthesise(ir)
      {:ok, batched} = CustomSynth.synthesise_batched(ir)

      d = length(layout)
      instances = for i <- 1..3, do: instance_inputs(i, d, obs_tensor(obs_bin))

      for dir <- [1, -1] do
        singles =
          for {q, p, im, _} <- instances,
              do: Dispatch.chain(single, d, @eps, im, q, p, @k, dir)

        batch = Dispatch.chain_batch(batched, instances, @k, dir, @eps)

        for {{s, b}, i} <- Enum.zip(singles, batch) |> Enum.with_index() do
          assert_all_four_identical(s, b, "dir #{dir}, instance #{i}")
        end
      end

      fwd = Dispatch.chain_batch(batched, instances, @k, 1, @eps)
      rev = Dispatch.chain_batch(batched, instances, @k, -1, @eps)

      refute Nx.to_binary(elem(hd(fwd), 0)) == Nx.to_binary(elem(hd(rev), 0)),
             "forward and reverse batches produced identical chains, so dir_sign " <>
               "is not reaching the shader"
    end
  end

  describe "instances do not read each other's data" do
    @describetag :requires_vulkan

    test "each instance sees its own observations, not its neighbour's" do
      # Three models identical but for their observations, and the values are
      # far apart on purpose: if instance 1 read from instance 0's slice its
      # log-density would be wrong by thousands, not by rounding.
      #
      # Observations reach a vector-observed shader through the buffer at
      # runtime (only scalar ones are inlined as constants), so one batched
      # SPV serves all three while each single-instance meta bakes nothing
      # that would differ.
      vs = [3.0, -50.0, 1000.0]
      d = 1

      pairs =
        for {v, i} <- Enum.with_index(vs) do
          ir = vector_obs_ir([v, v + 1.0])
          {:ok, {:synthesised, _, _, _, _, obs_bin, _capt} = meta} = CustomSynth.synthesise(ir)
          inst = instance_inputs(i + 1, d, obs_tensor(obs_bin))
          {q, p, im, _} = inst
          {Dispatch.chain(meta, d, @eps, im, q, p, @k, 1), inst}
        end

      {:ok, batched} = CustomSynth.synthesise_batched(vector_obs_ir([hd(vs), hd(vs) + 1.0]))
      batch = Dispatch.chain_batch(batched, Enum.map(pairs, &elem(&1, 1)), @k, 1, @eps)

      for {{{single, _}, b}, i} <- Enum.zip(pairs, batch) |> Enum.with_index() do
        assert_all_four_identical(single, b, "divergent-obs instance #{i}")
      end

      # Non-vacuity: the three instances must actually disagree, or matching
      # them proves nothing about isolation.
      logps = Enum.map(batch, fn {_, _, l, _} -> Nx.to_flat_list(l) |> hd() end)

      assert length(Enum.uniq(logps)) == 3,
             "the three instances produced log-densities #{inspect(logps)} — they " <>
               "were supposed to be far apart"
    end

    test "one instance's inverse mass does not leak into another's step" do
      # inv_mass sits after the observations in each instance's extras slice,
      # so it is the half of the stride the observation test does not reach.
      # Two instances with the same q and p but different inverse mass must
      # produce different trajectories, each matching its own lone dispatch.
      ir = conjugate_ir([3.0, -2.0])
      {:ok, {:synthesised, _, layout, _, _, obs_bin, _capt} = single} = CustomSynth.synthesise(ir)
      {:ok, batched} = CustomSynth.synthesise_batched(ir)

      d = length(layout)
      obs = obs_tensor(obs_bin)
      q = Nx.tensor([0.4, -0.7], type: :f64)
      p = Nx.tensor([0.9, 0.2], type: :f64)

      instances =
        for m <- [0.25, 4.0, 1.0] do
          {q, p, Nx.tensor([m, m], type: :f64), obs}
        end

      singles =
        for {qq, pp, im, _} <- instances,
            do: Dispatch.chain(single, d, @eps, im, qq, pp, @k, 1)

      batch = Dispatch.chain_batch(batched, instances, @k, 1, @eps)

      for {{s, b}, i} <- Enum.zip(singles, batch) |> Enum.with_index() do
        assert_all_four_identical(s, b, "inverse-mass instance #{i}")
      end

      qs = Enum.map(batch, fn {qc, _, _, _} -> Nx.to_binary(qc) end)

      assert length(Enum.uniq(qs)) == 3,
             "three different inverse masses produced fewer than three distinct " <>
               "chains from the same q and p"
    end
  end
end
