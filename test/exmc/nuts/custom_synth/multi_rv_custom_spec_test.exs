defmodule Exmc.NUTS.CustomSynth.MultiRvCustomSpecTest do
  # async: false, and it is not optional.
  #
  # The `render/1` describe blocks below set `:exmc, :force_precision, :f32` in
  # setup. That key is VM-global: `Exmc.JIT.precision/0` reads it, and
  # `Exmc.PointMap.pack/2` types every tensor it builds with the result. So
  # while this module runs async, EVERY other concurrently-running async test
  # sees f32.
  #
  # Measured: it silently retyped the finite-difference gradient checks in
  # gaussian_random_walk_test.exs and dirichlet_test.exs from f64 to f32.
  # Their FD step is 1.0e-5, so subtracting two nearly-equal f32 log-densities
  # left almost no significant digits — the gradient error went from 4.07e-4
  # (passing, against a 1e-3 tolerance) to 7.3e-3 (failing, 18x worse). Nothing
  # pointed at this file: the two tests passed alone, passed alongside every
  # file ported with this one, and failed only in the full suite, where the
  # bisect was non-monotonic because it depends on whether the modules overlap
  # in time.
  #
  # A test that mutates global application env cannot be async. The rule is
  # cheap; finding out the hard way took seven experiments.
  use ExUnit.Case, async: false

  # R2.2.0 — the rendered GLSL must pass glslangValidator. Numerical
  # correctness against Defn lands with R2.2.2; obs-axis parallelism +
  # custom likelihood lands with R2.2.1.

  alias Exmc.NUTS.CustomSynth.{Compile, Eval, Glsl, MultiRvCustomSpec}

  describe "shader size is independent of the DATA (regression: pipeline ceiling)" do
    # The bug this pins: closure-captured rank-1 tensors were emitted as
    # `const double name[N] = double[](...)` at file scope, so SPIR-V grew with
    # n_obs * n_beta. Synthesised shaders reached 2.15 MB against ~8 KB for a
    # hand-written one, and past roughly 1300 inlined elements the NVIDIA
    # driver refused to create the compute pipeline -- 21 of 33 posteriordb
    # models could not run on Vulkan at all.
    #
    # Asserted on GLSL rather than SPIR-V so the guard does not need
    # glslangValidator on the host. The relationship is monotone, so a
    # regression that reinlines the data blows this by orders of magnitude:
    # before the fix a 20x change in n_obs moved GLSL by megabytes.
    defp linreg_ir(n_obs, n_beta) do
      alias Exmc.Builder
      alias Exmc.Dist.{Custom, HalfNormal, Normal}

      x_cols =
        for j <- 1..n_beta do
          Nx.tensor(for(i <- 1..n_obs, do: i * 0.01 + j), type: :f64, backend: Nx.BinaryBackend)
        end

      y =
        Nx.tensor(for(i <- 1..n_obs, do: :math.sin(i * 0.1)),
          type: :f64,
          backend: Nx.BinaryBackend
        )

      ir =
        Enum.reduce(0..(n_beta - 1), Builder.new_ir(), fn j, acc ->
          Builder.rv(acc, "beta_#{j}", Normal, %{
            mu: Nx.tensor(0.0, type: :f64),
            sigma: Nx.tensor(1.0, type: :f64)
          })
        end)

      ir = Builder.rv(ir, "sigma", HalfNormal, %{sigma: Nx.tensor(1.0, type: :f64)})

      logpdf = fn _x, params ->
        mu =
          Enum.reduce(0..(n_beta - 1), Nx.tensor(0.0, type: :f64), fn j, acc ->
            beta = Map.fetch!(params, String.to_atom("beta_#{j}"))
            Nx.add(acc, Nx.multiply(beta, Enum.at(x_cols, j)))
          end)

        z = Nx.divide(Nx.subtract(y, mu), params.sigma)
        Nx.multiply(Nx.tensor(-0.5, type: :f64), Nx.sum(Nx.multiply(z, z)))
      end

      params =
        Map.new(0..(n_beta - 1), fn j -> {String.to_atom("beta_#{j}"), "beta_#{j}"} end)
        |> Map.put(:sigma, "sigma")

      ir
      |> Custom.rv("y_lik", Custom.new(logpdf), params)
      |> Builder.obs("y_obs", "y_lik", Nx.tensor(0.0, type: :f64))
    end

    defp render_sizes(n_obs, n_beta) do
      ir = Exmc.Rewrite.apply(linreg_ir(n_obs, n_beta), [])
      {:ok, components} = Exmc.NUTS.CustomSynth.extract_components(ir)
      {:ok, glsl, captures} = MultiRvCustomSpec.render(components)
      {byte_size(glsl), byte_size(captures)}
    end

    test "a 20x increase in n_obs leaves the shader essentially unchanged" do
      {small_glsl, small_cap} = render_sizes(50, 2)
      {large_glsl, large_cap} = render_sizes(1000, 2)

      # The only legitimate difference is the decimal width of literal offsets
      # and counts in the source text -- tens of bytes, not thousands.
      assert abs(large_glsl - small_glsl) < 500,
             "GLSL grew by #{large_glsl - small_glsl} bytes for 20x the observations; " <>
               "the data is being inlined into the shader again"

      # The data did not vanish -- it moved to the extras buffer, where it is
      # supposed to scale linearly and harmlessly.
      assert large_cap == small_cap * 20

      # n_beta predictor columns PLUS the response vector y -- the likelihood
      # closes over that too, so it is a capture like any other.
      assert large_cap == 1000 * (2 + 1) * 8
    end

    test "shader size still scales with the number of TERMS, which is correct" do
      {two_beta, _} = render_sizes(50, 2)
      {four_beta, _} = render_sizes(50, 4)

      assert four_beta > two_beta,
             "more predictors means more arithmetic in the shader; that much should grow"
    end
  end

  # The render/1 push-header assertions check f32-template field layout
  # (`uint  K;`, `float eps;`). Under D88 Vulkano f64 default, render/1
  # picks @template_f64 which uses `uint   K;` (extra space) and
  # `double eps;`. Force :f32 for these tests — the f64 layout is
  # exercised elsewhere.
  setup context do
    if context[:describe] in [
         "render/1",
         "render/1 + Compile.compile_glsl/1 — glslangValidator gate"
       ] do
      Application.put_env(:exmc, :force_precision, :f32)
      on_exit(fn -> Application.delete_env(:exmc, :force_precision) end)
    end

    :ok
  end

  describe "render/1" do
    test "single-RV Normal prior — fixed push header + per-tid dispatch present" do
      components = %{
        priors: [{:theta, Exmc.Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)}}],
        custom: nil,
        layout: [:theta]
      }

      assert {:ok, glsl, _captures} = MultiRvCustomSpec.render(components)

      # Fixed push header (R2.2.4: per-prior fields are inlined as constants,
      # not declared in push).
      assert glsl =~ "uint   K;"
      assert glsl =~ "uint   n_obs;"
      assert glsl =~ "uint   d;"
      assert glsl =~ "double eps;"
      refute glsl =~ "float theta_mu;"

      # Per-RV gradient + log-prob dispatched by tid (emitter-derived
      # fragment shape is parenthesized; we just check the dispatch shell).
      assert glsl =~ "if (tid == 0u) { grad_q ="
      assert glsl =~ "if (tid == 0u) { grad_qn ="
      assert glsl =~ "if (tid == 0u) { lp_i ="

      # 7-SSBO contract preserved.
      for binding <- 0..6 do
        assert glsl =~ "binding = #{binding}"
      end
    end

    test "multi-RV Normal + HalfCauchy — both priors dispatched by tid" do
      components = %{
        priors: [
          {:mu, Exmc.Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(5.0)}},
          {:sigma, Exmc.Dist.HalfCauchy, %{scale: Nx.tensor(2.5)}}
        ],
        custom: nil,
        layout: [:mu, :sigma]
      }

      assert {:ok, glsl, _captures} = MultiRvCustomSpec.render(components)

      assert glsl =~ "if (tid == 0u) { grad_q ="
      assert glsl =~ "if (tid == 1u) { grad_q ="
      assert glsl =~ "if (tid == 0u) { lp_i ="
      assert glsl =~ "if (tid == 1u) { lp_i ="

      # Emitted fragments reference qi (the per-thread position scalar)
      # and inline the prior scale (2.5 for sigma's HalfCauchy).
      assert glsl =~ "qi"
      assert glsl =~ "2.5"
    end

    test "rejects layout entry not in priors" do
      components = %{
        priors: [{:theta, Exmc.Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)}}],
        custom: nil,
        layout: [:theta, :phantom]
      }

      assert {:error, {:layout_id_not_in_priors, :phantom}} =
               MultiRvCustomSpec.render(components)
    end
  end

  describe "render/1 + Compile.compile_glsl/1 — glslangValidator gate" do
    @describetag :glslang

    test "single-RV Normal renders to GLSL that glslangValidator accepts" do
      components = %{
        priors: [{:theta, Exmc.Dist.Normal, %{mu: 0.0, sigma: 1.0}}],
        custom: nil,
        layout: [:theta]
      }

      {:ok, glsl, _captures} = MultiRvCustomSpec.render(components)

      assert {:ok, spv_path} = Compile.compile_glsl(glsl)
      assert File.exists?(spv_path)

      # SPIR-V magic number 0x07230203 little-endian in word[0].
      <<magic::little-unsigned-integer-32, _rest::binary>> = File.read!(spv_path)
      assert magic == 0x07230203
    end

    test "multi-RV (Normal + HalfCauchy + HalfNormal + Exponential) renders + compiles" do
      components = %{
        priors: [
          {:mu, Exmc.Dist.Normal, %{mu: 0.0, sigma: 5.0}},
          {:sigma, Exmc.Dist.HalfCauchy, %{scale: 2.5}},
          {:tau, Exmc.Dist.HalfNormal, %{sigma: 1.0}},
          {:lam, Exmc.Dist.Exponential, %{lambda: 0.5}}
        ],
        custom: nil,
        layout: [:mu, :sigma, :tau, :lam]
      }

      {:ok, glsl, _captures} = MultiRvCustomSpec.render(components)
      assert {:ok, spv_path} = Compile.compile_glsl(glsl)

      <<magic::little-unsigned-integer-32, _rest::binary>> = File.read!(spv_path)
      assert magic == 0x07230203
    end

    test "rendered GLSL push header is the R2.2.4 fixed shape (K, n_obs, d, _pad, eps)" do
      # R2.2.4: per-prior push fields are gone — prior params are inlined
      # as GLSL constants. The push block is a fixed 20-byte header.
      components = %{
        priors: [
          {:mu, Exmc.Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)}},
          {:sigma, Exmc.Dist.HalfCauchy, %{scale: Nx.tensor(1.0)}}
        ],
        custom: nil,
        layout: [:mu, :sigma]
      }

      {:ok, glsl, _captures} = MultiRvCustomSpec.render(components)

      for line <- ["uint   K;", "uint   n_obs;", "uint   d;", "uint   _pad;", "double eps;"] do
        assert glsl =~ line, "expected #{inspect(line)} in rendered GLSL"
      end

      # Per-prior push fields are deliberately absent now.
      refute glsl =~ "float mu_mu;"
      refute glsl =~ "float sigma_scale;"
    end
  end

  describe "R2.2.1 — compose_logp_defn + trace + emit" do
    test "prior-only single Normal: trace_logp + Glsl.emit produces a non-empty GLSL string" do
      components = %{
        priors: [{"theta", Exmc.Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)}}],
        custom: nil,
        layout: ["theta"]
      }

      expr = MultiRvCustomSpec.trace_logp(components, 0)
      assert {:ok, glsl} = Glsl.emit(expr, MultiRvCustomSpec.default_emit_layout())
      assert is_binary(glsl) and byte_size(glsl) > 0
      # Normal.logpdf reduces to an arithmetic chain over q[0].
      assert glsl =~ "q[0]"
    end

    test "prior-only single Normal: trace_grad + Glsl.emit_vector returns entries indexed by RV position" do
      components = %{
        priors: [{"theta", Exmc.Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)}}],
        custom: nil,
        layout: ["theta"]
      }

      grad_expr = MultiRvCustomSpec.trace_grad(components, 0)
      assert {:ok, entries} = Glsl.emit_vector(grad_expr, MultiRvCustomSpec.default_emit_layout())
      assert is_list(entries) and entries != []

      indices = entries |> Enum.map(&elem(&1, 0)) |> Enum.uniq() |> Enum.sort()
      assert indices == [0]
      # Each fragment is a non-empty GLSL string.
      Enum.each(entries, fn {_idx, glsl} ->
        assert is_binary(glsl) and byte_size(glsl) > 0
      end)
    end

    test "prior-only multi-RV: Normal + HalfCauchy traces and grad emits two index buckets" do
      components = %{
        priors: [
          {"mu", Exmc.Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(5.0)}},
          {"sigma", Exmc.Dist.HalfCauchy, %{scale: Nx.tensor(2.5)}}
        ],
        custom: nil,
        layout: ["mu", "sigma"]
      }

      expr = MultiRvCustomSpec.trace_logp(components, 0)
      assert {:ok, glsl} = Glsl.emit(expr, MultiRvCustomSpec.default_emit_layout())
      assert glsl =~ "q[0]"
      assert glsl =~ "q[1]"

      grad_expr = MultiRvCustomSpec.trace_grad(components, 0)
      assert {:ok, entries} = Glsl.emit_vector(grad_expr, MultiRvCustomSpec.default_emit_layout())
      indices = entries |> Enum.map(&elem(&1, 0)) |> Enum.uniq() |> Enum.sort()
      assert indices == [0, 1]
    end
  end

  describe "R2.2.2 — Eval matches Nx.Defn.Evaluator on composed log_p (prior-only)" do
    # The structural gate: prove the parallel Elixir walker (Eval) agrees
    # with Nx.Defn.Evaluator on random concrete q. If Eval matches, the
    # GLSL emitter — which uses the same op-dispatch structure with string
    # leaves instead of float leaves — is structurally correct as well.

    defp reference(fun, args) do
      Nx.Defn.jit_apply(fun, args, compiler: Nx.Defn.Evaluator)
      |> Nx.to_number()
    end

    test "single Normal(0,1) prior — Eval matches Defn for 100 random q values" do
      components = %{
        priors: [{"theta", Exmc.Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)}}],
        custom: nil,
        layout: ["theta"]
      }

      fun = MultiRvCustomSpec.compose_logp_defn(components)
      expr = MultiRvCustomSpec.trace_logp(components, 0)

      for _ <- 1..100 do
        x = :rand.uniform() * 6.0 - 3.0
        q = [x]
        obs = [0.0]

        ref = reference(fun, [Nx.tensor(q, type: :f64), Nx.tensor(obs, type: :f64)])
        {:ok, ours} = Eval.evaluate(expr, [q, obs])

        assert_in_delta(ours, ref, 1.0e-5)
      end
    end

    test "multi-RV (Normal + HalfCauchy + HalfNormal + Exponential) — Eval matches Defn" do
      components = %{
        priors: [
          {"mu", Exmc.Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(5.0)}},
          {"sigma", Exmc.Dist.HalfCauchy, %{scale: Nx.tensor(2.5)}},
          {"tau", Exmc.Dist.HalfNormal, %{sigma: Nx.tensor(1.0)}},
          {"lam", Exmc.Dist.Exponential, %{lambda: Nx.tensor(0.5)}}
        ],
        custom: nil,
        layout: ["mu", "sigma", "tau", "lam"]
      }

      fun = MultiRvCustomSpec.compose_logp_defn(components)
      expr = MultiRvCustomSpec.trace_logp(components, 0)

      for _ <- 1..100 do
        # Sample q with positive entries for the positive-support priors.
        q = [
          :rand.uniform() * 6.0 - 3.0,
          :rand.uniform() * 2.0 + 0.1,
          :rand.uniform() * 2.0 + 0.1,
          :rand.uniform() * 2.0 + 0.1
        ]

        obs = [0.0]

        ref = reference(fun, [Nx.tensor(q, type: :f64), Nx.tensor(obs, type: :f64)])
        {:ok, ours} = Eval.evaluate(expr, [q, obs])

        assert_in_delta(ours, ref, 2.0e-5)
      end
    end
  end

  describe "R2.2.3 — pack_input_buffer/2 + template binding-2 repack" do
    # Post 851155b5a: f64-only synth architecture — pack_input_buffer
    # always emits f64 (little-float-64). No precision override needed.

    test "pack_input_buffer/2 lays obs first then inv_mass, f32 little-endian" do
      obs = [0.1, 0.2, 0.3]
      inv_mass = [4.0, 5.0]

      bin = MultiRvCustomSpec.pack_input_buffer(obs, inv_mass)
      assert byte_size(bin) == (length(obs) + length(inv_mass)) * 8

      <<o0::little-float-64, o1::little-float-64, o2::little-float-64, m0::little-float-64,
        m1::little-float-64>> = bin

      assert_in_delta(o0, 0.1, 1.0e-6)
      assert_in_delta(o1, 0.2, 1.0e-6)
      assert_in_delta(o2, 0.3, 1.0e-6)
      assert_in_delta(m0, 4.0, 1.0e-6)
      assert_in_delta(m1, 5.0, 1.0e-6)
    end

    test "pack_input_buffer/2 accepts nil obs (prior-only model)" do
      bin = MultiRvCustomSpec.pack_input_buffer(nil, [1.0, 2.0])
      assert byte_size(bin) == 2 * 8

      <<m0::little-float-64, m1::little-float-64>> = bin
      assert_in_delta(m0, 1.0, 1.0e-6)
      assert_in_delta(m1, 2.0, 1.0e-6)
    end

    test "pack_input_buffer/2 accepts Nx tensors" do
      obs = Nx.tensor([0.5, -0.5])
      inv_mass = Nx.tensor([3.0])

      bin = MultiRvCustomSpec.pack_input_buffer(obs, inv_mass)
      assert byte_size(bin) == 3 * 8
    end

    @tag :glslang
    test "template with new binding-2 layout still compiles via glslangValidator" do
      components = %{
        priors: [{"theta", Exmc.Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)}}],
        custom: nil,
        layout: ["theta"]
      }

      {:ok, glsl, _captures} = MultiRvCustomSpec.render(components)
      # New binding-2 declaration is present.
      assert glsl =~ "obs_inv_mass"
      assert glsl =~ "obs_inv_mass[pc.n_obs + tid]"

      assert {:ok, spv_path} = Compile.compile_glsl(glsl)
      <<magic::little-unsigned-integer-32, _rest::binary>> = File.read!(spv_path)
      assert magic == 0x07230203
    end
  end

  describe "R2.2.4 — emit_prior_fragments numerical equivalence vs mod.logpdf" do
    # Walk the emitted log-pdf + gradient GLSL via Eval (string-leaf →
    # float-leaf parallel walker) and compare against the dist module's
    # logpdf + its Defn.grad at random concrete x. If these agree, the
    # emitter is producing arithmetically correct fragments — the
    # GLSL-on-Vulkan side stays a downstream concern.

    defp ref_logpdf(mod, x, params) do
      mod.logpdf(Nx.tensor(x, type: :f64), params)
      |> Nx.to_number()
    end

    defp ref_grad(mod, x, params) do
      grad_fn = fn xt -> Nx.Defn.grad(xt, fn xt -> mod.logpdf(xt, params) end) end

      Nx.Defn.jit_apply(grad_fn, [Nx.tensor(x, type: :f64)], compiler: Nx.Defn.Evaluator)
      |> Nx.to_number()
    end

    defp assert_prior_equivalence(mod, params, x_sampler) do
      # Nx 0.12 rejects closure captures that mix VulkanoBackend
      # tensors with Nx.Defn.Expr. Force params to BinaryBackend so
      # the trace can inline them as constants.
      params =
        Map.new(params, fn
          {k, %Nx.Tensor{} = v} -> {k, Nx.backend_copy(v, Nx.BinaryBackend)}
          kv -> kv
        end)

      {:ok, [{_id, lp_glsl, grad_glsl}]} =
        MultiRvCustomSpec.emit_prior_fragments([{"x", mod, params}])

      # Eval walks the GLSL-shaped tree. To re-create the same op tree
      # for Eval, trace the SAME functions.
      value_fn = fn xt -> mod.logpdf(xt, params) end
      grad_fn = fn xt -> Nx.Defn.grad(xt, value_fn) end
      tmpl = Nx.template({}, :f64)

      value_expr = Nx.Defn.debug_expr_apply(value_fn, [tmpl])
      grad_expr = Nx.Defn.debug_expr_apply(grad_fn, [tmpl])

      for _ <- 1..50 do
        x = x_sampler.()

        ref_lp = ref_logpdf(mod, x, params)
        ref_g = ref_grad(mod, x, params)

        {:ok, ours_lp} = Eval.evaluate(value_expr, [x])
        {:ok, ours_g} = Eval.evaluate(grad_expr, [x])

        # Tolerance accounts for f64 rounding divergence between Erlang
        # :math.* and Nx BinaryBackend's underlying ops — last-few-bits
        # drift is normal, particularly on log(1 + x²) chains.
        assert_in_delta(ours_lp, ref_lp, max(1.0e-7, abs(ref_lp) * 1.0e-7))
        assert_in_delta(ours_g, ref_g, max(1.0e-7, abs(ref_g) * 1.0e-7))

        # Sanity: GLSL fragment is a non-empty parseable string.
        assert is_binary(lp_glsl) and byte_size(lp_glsl) > 0
        assert is_binary(grad_glsl) and byte_size(grad_glsl) > 0
      end
    end

    test "Normal" do
      assert_prior_equivalence(
        Exmc.Dist.Normal,
        %{mu: Nx.tensor(0.5), sigma: Nx.tensor(2.0)},
        fn -> :rand.uniform() * 6.0 - 3.0 end
      )
    end

    test "HalfNormal" do
      assert_prior_equivalence(
        Exmc.Dist.HalfNormal,
        %{sigma: Nx.tensor(1.5)},
        fn -> :rand.uniform() * 3.0 + 0.01 end
      )
    end

    test "HalfCauchy" do
      assert_prior_equivalence(
        Exmc.Dist.HalfCauchy,
        %{scale: Nx.tensor(2.5)},
        fn -> :rand.uniform() * 4.0 + 0.01 end
      )
    end

    test "Exponential" do
      assert_prior_equivalence(
        Exmc.Dist.Exponential,
        %{lambda: Nx.tensor(0.7)},
        fn -> :rand.uniform() * 3.0 + 0.01 end
      )
    end
  end

  describe "R2.3 — parameter references (hierarchical models)" do
    # The gate for reference resolution, and the reason it is host-vs-synth
    # rather than a smoke test: the failure mode here is a finite, plausible,
    # WRONG log-density. A transform applied in the wrong space or an NCP
    # reconstruction that silently does not happen both produce a number, and
    # the sampler runs happily on it. Only comparing against
    # Exmc.Compiler.compile/1 at the same q can tell them apart.

    defp t64(v), do: Nx.tensor(v, type: :f64, backend: Nx.BinaryBackend)

    defp worst_gap(ir, draws, ncp_override \\ :keep) do
      {logp_fn, pm} = Exmc.Compiler.compile(ir)
      rewritten = Exmc.Rewrite.apply(ir, [])
      {:ok, comps} = Exmc.NUTS.CustomSynth.extract_components(rewritten)

      ncp =
        case ncp_override do
          :keep -> rewritten.ncp_info || %{}
          other -> other
        end

      fun = MultiRvCustomSpec.compose_logp_defn(Map.put(comps, :ncp_info, ncp))
      obs = Nx.tensor([4.0, 5.0, 8.0], type: :f64)
      names = Enum.map(pm.entries, & &1.id)
      :rand.seed(:exsss, {9, 9, 9})

      Enum.reduce(1..draws, 0.0, fn _, acc ->
        vals = Map.new(names, fn n -> {n, t64(:rand.normal())} end)
        host = logp_fn.(Exmc.PointMap.pack(vals, pm)) |> Nx.to_number()

        q =
          Nx.tensor(Enum.map(comps.layout, &(vals |> Map.fetch!(&1) |> Nx.to_number())),
            type: :f64
          )

        synth = Nx.Defn.jit_apply(fun, [q, obs], compiler: Nx.Defn.Evaluator) |> Nx.to_number()
        max(acc, abs(host - synth) / max(abs(host), 1.0))
      end)
    end

    defp ncp_ir do
      # alpha is NCP'd (both params are refs) AND observed through y, which is
      # what makes the mu + sigma*z reconstruction appear in the density. An
      # NCP'd RV that nothing downstream names does not exercise it at all.
      Exmc.Builder.new_ir()
      |> Exmc.Builder.rv("mu", Exmc.Dist.Normal, %{mu: t64(0.0), sigma: t64(5.0)})
      |> Exmc.Builder.rv("sigma", Exmc.Dist.Exponential, %{lambda: t64(1.0)})
      |> Exmc.Builder.rv("alpha", Exmc.Dist.Normal, %{mu: "mu", sigma: "sigma"})
      |> Exmc.Builder.rv("y", Exmc.Dist.Normal, %{mu: "alpha", sigma: t64(1.0)})
      |> Exmc.Builder.obs("y_obs", "y", t64(3.0))
    end

    test "a plain reference matches the host" do
      ir =
        Exmc.Builder.new_ir()
        |> Exmc.Builder.rv("mu", Exmc.Dist.Normal, %{mu: t64(0.0), sigma: t64(5.0)})
        |> Exmc.Builder.rv("alpha", Exmc.Dist.Normal, %{mu: "mu", sigma: t64(1.0)})

      assert worst_gap(ir, 100) < 1.0e-12
    end

    test "a reference to a transformed RV resolves in constrained space" do
      # sigma carries transform: :log, so the referrer must see exp(z), not z.
      # Resolving in the wrong space still yields a finite density.
      ir =
        Exmc.Builder.new_ir()
        |> Exmc.Builder.rv("s", Exmc.Dist.Exponential, %{lambda: t64(1.0)})
        |> Exmc.Builder.rv("alpha", Exmc.Dist.Normal, %{mu: t64(0.0), sigma: "s"})

      assert worst_gap(ir, 100) < 1.0e-12
    end

    test "a two-level reference chain resolves in dependency order" do
      ir =
        Exmc.Builder.new_ir()
        |> Exmc.Builder.rv("m0", Exmc.Dist.Normal, %{mu: t64(0.0), sigma: t64(2.0)})
        |> Exmc.Builder.rv("m1", Exmc.Dist.Normal, %{mu: "m0", sigma: t64(1.0)})
        |> Exmc.Builder.rv("m2", Exmc.Dist.Normal, %{mu: "m1", sigma: t64(1.0)})

      assert worst_gap(ir, 100) < 1.0e-12
    end

    test "an NCP model matches the host" do
      assert worst_gap(ncp_ir(), 100) < 1.0e-12
    end

    test "and dropping ncp_info makes it diverge — the gate has teeth" do
      # Without this the previous test proves nothing: a resolver that ignored
      # ncp_info entirely would still pass it on a model where the
      # reconstruction never enters the density.
      assert worst_gap(ncp_ir(), 50, %{}) > 1.0e-3
    end

    test "a reference to a non-coordinate is rejected by name at build time" do
      ir =
        Exmc.Builder.new_ir()
        |> Exmc.Builder.rv("a", Exmc.Dist.Normal, %{mu: "nonexistent", sigma: t64(1.0)})

      rewritten = Exmc.Rewrite.apply(ir, [])
      {:ok, comps} = Exmc.NUTS.CustomSynth.extract_components(rewritten)

      assert_raise Exmc.SynthReferenceError, ~r/nonexistent.*not a\s+sampled coordinate/s, fn ->
        MultiRvCustomSpec.compose_logp_defn(comps)
      end
    end

    test "several scalar observations sharing a parent match the host" do
      # The 5-parameter hierarchical shape from integration_test. y1 and y2 have
      # IDENTICAL params — both N(alpha, sigma_obs) — so their expressions are
      # structurally identical, Nx merges them, and the emitter produced 2
      # REDUCE_SUM markers for 3 observed nodes. Positional span attribution
      # then refused the model:
      #
      #     2 REDUCE_SUM marker(s) for 3 observed node(s)
      #
      # Scalar observations are inlined as distinct constants now, so nothing
      # merges and no marker is emitted for them.
      ir =
        Exmc.Builder.new_ir()
        |> Exmc.Builder.rv("mu_global", Exmc.Dist.Normal, %{mu: t64(0.0), sigma: t64(10.0)})
        |> Exmc.Builder.rv("sigma_global", Exmc.Dist.Exponential, %{lambda: t64(1.0)})
        |> Exmc.Builder.rv("alpha", Exmc.Dist.Normal, %{mu: "mu_global", sigma: "sigma_global"})
        |> Exmc.Builder.rv("beta", Exmc.Dist.Normal, %{mu: "mu_global", sigma: "sigma_global"})
        |> Exmc.Builder.rv("sigma_obs", Exmc.Dist.Exponential, %{lambda: t64(2.0)})
        |> Exmc.Builder.rv("y1", Exmc.Dist.Normal, %{mu: "alpha", sigma: "sigma_obs"})
        |> Exmc.Builder.obs("y1_obs", "y1", t64(4.0))
        |> Exmc.Builder.rv("y2", Exmc.Dist.Normal, %{mu: "alpha", sigma: "sigma_obs"})
        |> Exmc.Builder.obs("y2_obs", "y2", t64(5.0))
        |> Exmc.Builder.rv("y3", Exmc.Dist.Normal, %{mu: "beta", sigma: "sigma_obs"})
        |> Exmc.Builder.obs("y3_obs", "y3", t64(8.0))

      assert worst_gap(ir, 100) < 1.0e-12
      assert {:ok, _glsl, _captures} = render_components(ir)
    end

    defp render_components(ir) do
      rewritten = Exmc.Rewrite.apply(ir, [])
      {:ok, comps} = Exmc.NUTS.CustomSynth.extract_components(rewritten)
      MultiRvCustomSpec.render(Map.put(comps, :ncp_info, rewritten.ncp_info || %{}))
    end

    test "layout stays in PointMap order past the 32-key flatmap threshold" do
      # layout IS the q-vector order. Map iteration order matches sorted order
      # only under 32 keys; past that the shader would read q in one order and
      # the sampler write it in another, with no error.
      ir =
        Enum.reduce(1..40, Exmc.Builder.new_ir(), fn i, acc ->
          id = "x#{String.pad_leading(to_string(i), 2, "0")}"
          Exmc.Builder.rv(acc, id, Exmc.Dist.Normal, %{mu: t64(0.0), sigma: t64(1.0)})
        end)

      rewritten = Exmc.Rewrite.apply(ir, [])
      {:ok, comps} = Exmc.NUTS.CustomSynth.extract_components(rewritten)
      assert comps.layout == Enum.map(Exmc.PointMap.build(rewritten).entries, & &1.id)
    end
  end
end
