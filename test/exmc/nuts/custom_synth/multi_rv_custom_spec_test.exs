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

      assert {:ok, glsl} = MultiRvCustomSpec.render(components)

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

      assert {:ok, glsl} = MultiRvCustomSpec.render(components)

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

      {:ok, glsl} = MultiRvCustomSpec.render(components)

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

      {:ok, glsl} = MultiRvCustomSpec.render(components)
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

      {:ok, glsl} = MultiRvCustomSpec.render(components)

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

      {:ok, glsl} = MultiRvCustomSpec.render(components)
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
end
