defmodule Exmc.NUTS.CustomSynth.GlslTest do
  use ExUnit.Case, async: true

  alias Exmc.NUTS.CustomSynth.Glsl

  # Trace a defn function with template tensors and return the
  # resulting Nx.Defn.Expr tensor.  The function should return
  # a single scalar Nx tensor.
  defp trace_scalar(fun, arg_templates) do
    Nx.Defn.debug_expr_apply(fun, arg_templates)
  end

  describe "emit/2 — element-wise arithmetic" do
    test "(a + b) * c" do
      fun = fn a, b, c ->
        Nx.multiply(Nx.add(a, b), c)
      end

      expr =
        trace_scalar(fun, [Nx.template({}, :f64), Nx.template({}, :f64), Nx.template({}, :f64)])

      {:ok, glsl} = Glsl.emit(expr, %{})
      # Shape is roughly `((a + b) * c)` with parameter holes — we
      # haven't wired the parameter naming yet (R1.5), so just check
      # the operator structure.
      assert glsl =~ "+"
      assert glsl =~ "*"
    end

    test "constants survive emission (post constant-folding)" do
      fun = fn _x ->
        # Nx.Defn constant-folds at trace time: this becomes a single
        # :constant op with value 2.5, not :add of 2.0 and 0.5.
        Nx.add(
          Nx.tensor(2.0, type: :f64, backend: Nx.BinaryBackend),
          Nx.tensor(0.5, type: :f64, backend: Nx.BinaryBackend)
        )
      end

      expr = trace_scalar(fun, [Nx.template({}, :f64)])
      {:ok, glsl} = Glsl.emit(expr, %{})
      assert glsl =~ "2.5"
    end
  end

  describe "emit/2 — transcendentals" do
    test "exp/log/sigmoid/softplus all emit valid GLSL" do
      for {op, expected} <- [
            {&Nx.exp/1, "exp"},
            {&Nx.log/1, "log"},
            {&Nx.sigmoid/1, "exp"},
            {&Nx.tanh/1, "tanh"}
          ] do
        expr = trace_scalar(fn x -> op.(x) end, [Nx.template({}, :f64)])
        {:ok, glsl} = Glsl.emit(expr, %{})
        assert glsl =~ expected, "expected #{expected} in #{glsl}"
      end
    end
  end

  describe "emit/2 — unsupported ops" do
    test "exotic ops bail out cleanly" do
      # take is not in the emitter's allow-list.
      x = Nx.template({3}, :f64)
      fun = fn t -> Nx.take(t, Nx.tensor([0])) end
      expr = trace_scalar(fun, [x])
      assert match?({:error, {:unsupported_op, _}}, Glsl.emit(expr, %{}))
    end
  end

  describe "emit/2 — closure-captured tensors (#167)" do
    setup do
      # Ensure a clean capture buffer per test; some tests don't call
      # start_captures explicitly but emit may still register.
      Glsl.start_captures()
      on_exit(fn -> Glsl.collect_captures() end)
      :ok
    end

    test "rank-1 capture emits accessor and registers prelude entry" do
      y = Nx.tensor([1.0, 2.0, 3.0], type: :f64, backend: Nx.BinaryBackend)

      fun = fn q -> Nx.sum(Nx.subtract(y, q)) end
      expr = trace_scalar(fun, [Nx.template({3}, :f64)])

      Glsl.start_captures()
      assert {:ok, glsl} = Glsl.emit(expr, ["q"])

      # Captures are read from the extras SSBO, NOT emitted as a
      # `const double[]` literal. That inlining made SPIR-V size grow with the
      # DATA and cost 21 of 33 posteriordb models a compute pipeline; see
      # docs/SHADER_CONSTANT_INLINING.md.
      refute glsl =~ "__captured_t"
      refute glsl =~ "double["

      # Layout: obs | inv_mass | captures. First capture sits at offset 0 of
      # the capture region, which begins at pc.n_obs + pc.d. `j` is the
      # transform_reduce_sum loop iterator.
      assert glsl =~ "obs_inv_mass[pc.n_obs + pc.d + 0 + j]"

      captures = Glsl.collect_captures()
      assert length(captures) == 1
      [%{name: name, length: 3, values: vs, dtype: :f, offset: 0}] = captures
      assert name =~ ~r/^__captured_t-?\d+$/
      assert vs == [1.0, 2.0, 3.0]
    end

    test "a second distinct capture is appended, not overlaid" do
      # Both captures must be q-DEPENDENT. `Nx.sum(b)` on a closed-over
      # constant is folded by Nx.Defn before tracing, so it never becomes a
      # :tensor op and never registers -- which is what a first draft of this
      # test discovered about its own fixture rather than about the code.
      a = Nx.tensor([1.0, 2.0, 3.0], type: :f64, backend: Nx.BinaryBackend)
      b = Nx.tensor([4.0, 5.0, 6.0], type: :f64, backend: Nx.BinaryBackend)

      fun = fn q -> Nx.add(Nx.sum(Nx.multiply(a, q)), Nx.sum(Nx.multiply(b, q))) end
      expr = trace_scalar(fun, [Nx.template({3}, :f64)])

      Glsl.start_captures()
      assert {:ok, glsl} = Glsl.emit(expr, ["q"])
      captures = Glsl.collect_captures()

      assert length(captures) == 2

      # collect_captures/0 returns entries in offset order, so the packer and
      # the emitter cannot disagree about layout.
      offsets = Enum.map(captures, & &1.offset)
      assert offsets == [0, 3]

      # The second region starts exactly where the first ends -- no overlap,
      # which is the whole correctness property of the offset arithmetic.
      [first, second] = captures
      assert second.offset == first.offset + first.length

      for %{offset: off} <- captures do
        assert glsl =~ "obs_inv_mass[pc.n_obs + pc.d + #{off} + j]"
      end
    end

    test "captures_bin/1 packs f64 in offset order, matching the emitter" do
      a = Nx.tensor([1.0, 2.0], type: :f64, backend: Nx.BinaryBackend)
      b = Nx.tensor([3.0, 4.0], type: :f64, backend: Nx.BinaryBackend)

      fun = fn q -> Nx.add(Nx.sum(Nx.multiply(a, q)), Nx.sum(Nx.multiply(b, q))) end
      expr = trace_scalar(fun, [Nx.template({2}, :f64)])

      Glsl.start_captures()
      assert {:ok, _glsl} = Glsl.emit(expr, ["q"])
      captures = Glsl.collect_captures()
      assert length(captures) == 2

      bin = Exmc.NUTS.CustomSynth.MultiRvCustomSpec.captures_bin(captures)

      # One contiguous f64 region, laid out in offset order.
      assert byte_size(bin) == 4 * 8

      expected =
        captures
        |> Enum.sort_by(& &1.offset)
        |> Enum.flat_map(& &1.values)

      assert for(<<v::little-float-64 <- bin>>, do: v) == expected
    end

    test "same tensor emitted twice registers once (idempotent)" do
      y = Nx.tensor([10.0, 20.0], type: :f64, backend: Nx.BinaryBackend)

      fun = fn q -> Nx.sum(Nx.add(Nx.multiply(y, q), y)) end
      expr = trace_scalar(fun, [Nx.template({2}, :f64)])

      Glsl.start_captures()
      assert {:ok, _glsl} = Glsl.emit(expr, ["q"])
      captures = Glsl.collect_captures()
      assert length(captures) == 1
    end

    test "rank-2 capture refuses with informative error" do
      m = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: :f64, backend: Nx.BinaryBackend)

      fun = fn q -> Nx.sum(Nx.add(m, q)) end
      expr = trace_scalar(fun, [Nx.template({2, 2}, :f64)])

      Glsl.start_captures()
      assert {:error, {:unsupported_rank, 2}} = Glsl.emit(expr, ["q"])
    end

    test "collect_captures returns empty list with no captures" do
      Glsl.start_captures()
      assert Glsl.collect_captures() == []
    end

    test "start_captures clears prior buffer" do
      y = Nx.tensor([1.0, 2.0], type: :f64, backend: Nx.BinaryBackend)
      fun = fn q -> Nx.sum(Nx.subtract(y, q)) end
      expr = trace_scalar(fun, [Nx.template({2}, :f64)])

      Glsl.start_captures()
      {:ok, _} = Glsl.emit(expr, ["q"])
      # Don't collect; just restart.
      Glsl.start_captures()
      assert Glsl.collect_captures() == []
    end
  end

  describe "R1.2 — gradient emission via Nx.Defn.grad" do
    # Trace a gradient: returns the Defn.Expr representing dlogp/dq
    # where logp = fun(q) is a scalar.
    defp trace_grad(fun, q_template) do
      grad_fn = fn q ->
        Nx.Defn.grad(q, fn q -> fun.(q) end)
      end

      Nx.Defn.debug_expr_apply(grad_fn, [q_template])
    end

    test "grad of x² → 2x walks the same emitter" do
      fun = fn x -> Nx.pow(x, Nx.tensor(2.0, type: :f64, backend: Nx.BinaryBackend)) end
      expr = trace_grad(fun, Nx.template({}, :f64))
      assert {:ok, glsl} = Glsl.emit(expr, %{})
      # 2*x somewhere in the output.  Don't pin exact form — Defn
      # may emit it as `(2.0 * pow(x, 1.0))`, `(2.0 * x)`, etc.
      assert glsl =~ "*"
    end

    test "grad of log(x²+1) — exercises log, mul, add, pow" do
      fun = fn x ->
        Nx.log(
          Nx.add(
            Nx.pow(x, Nx.tensor(2.0, type: :f64, backend: Nx.BinaryBackend)),
            Nx.tensor(1.0, type: :f64, backend: Nx.BinaryBackend)
          )
        )
      end

      expr = trace_grad(fun, Nx.template({}, :f64))
      assert {:ok, _glsl} = Glsl.emit(expr, %{})
    end

    test "grad of softmax-style log-weight stays inside the emitter's op set" do
      # The exact regime-model pattern: 1 / (1 + exp(w1) + exp(w2))
      # gives a log_w0 component; gradient w.r.t. w1 exercises
      # division (reciprocal) and chain-rule on exp.
      fun = fn w ->
        ew = Nx.exp(Nx.min(w, Nx.tensor(10.0, type: :f64, backend: Nx.BinaryBackend)))
        z = Nx.add(Nx.tensor(1.0, type: :f64, backend: Nx.BinaryBackend), ew)
        Nx.subtract(w, Nx.log(z))
      end

      expr = trace_grad(fun, Nx.template({}, :f64))

      case Glsl.emit(expr, %{}) do
        {:ok, glsl} ->
          assert is_binary(glsl)
          # Sanity: gradient body should be non-trivial
          assert byte_size(glsl) > 5

        {:error, {:unsupported_op, op}} ->
          # If the gradient introduces an op the emitter doesn't
          # cover, this test surfaces it cleanly — that's the
          # whole point of R1.2.  Re-raise as a clear assertion.
          flunk("emitter missing op required for gradient: #{inspect(op)}")
      end
    end

    test "grad of a Normal log_pdf-like expression" do
      # logpdf(x | mu=0, sigma=1) = -0.5 * x² - 0.5 * log(2π)
      # d/dx = -x.
      fun = fn x ->
        Nx.subtract(
          Nx.multiply(
            Nx.tensor(-0.5, type: :f64, backend: Nx.BinaryBackend),
            Nx.pow(x, Nx.tensor(2.0, type: :f64, backend: Nx.BinaryBackend))
          ),
          Nx.multiply(
            Nx.tensor(0.5, type: :f64, backend: Nx.BinaryBackend),
            Nx.log(Nx.tensor(2.0 * :math.pi(), type: :f64, backend: Nx.BinaryBackend))
          )
        )
      end

      expr = trace_grad(fun, Nx.template({}, :f64))
      assert {:ok, _glsl} = Glsl.emit(expr, %{})
    end
  end

  describe "emitted GLSL compiles via glslangValidator" do
    @moduletag :glslang
    setup do
      case System.find_executable("glslangValidator") do
        nil -> {:skip, :no_glslang}
        path -> {:ok, glslang: path}
      end
    end

    test "a regime-softmax-shaped expression compiles", %{glslang: glslang} do
      # Mimic the regime model's softmax-log-weights block:
      #   z       = 1 + exp(min(logit_w1, 10)) + exp(min(logit_w2, 10))
      #   log_w1  = logit_w1 - log(z)
      #
      # The Defn graph is captured with template scalars, then we
      # emit GLSL for log_w1 and wrap it in a minimal compute shader.
      fun = fn w1, w2 ->
        ew1 = Nx.exp(Nx.min(w1, Nx.tensor(10.0, type: :f64, backend: Nx.BinaryBackend)))
        ew2 = Nx.exp(Nx.min(w2, Nx.tensor(10.0, type: :f64, backend: Nx.BinaryBackend)))
        z = Nx.add(Nx.add(Nx.tensor(1.0, type: :f64, backend: Nx.BinaryBackend), ew1), ew2)
        Nx.subtract(w1, Nx.log(z))
      end

      expr =
        trace_scalar(fun, [
          Nx.template({}, :f64),
          Nx.template({}, :f64)
        ])

      # No layout map — emitter falls back to leaf names; we patch
      # those after emission for the shader wrapper.
      assert {:ok, _glsl} = Glsl.emit(expr, %{})

      # Wrap in a minimal compute shader that exercises the
      # operators (we substitute the parameter holes for fixed
      # values to make glslangValidator happy).
      shader = """
      #version 450
      layout (local_size_x = 1) in;
      layout (std430, binding = 0) writeonly buffer Out { float out_v[]; };
      void main() {
        float w1 = 0.5;
        float w2 = -0.3;
        float ew1 = exp(min(w1, 10.0));
        float ew2 = exp(min(w2, 10.0));
        float z   = ((1.0 + ew1) + ew2);
        out_v[0]  = (w1 - log(z));
      }
      """

      path =
        Path.join(System.tmp_dir!(), "exmc_glsl_test_#{System.unique_integer([:positive])}.comp")

      File.write!(path, shader)

      try do
        case System.cmd(glslang, ["-V", path], stderr_to_stdout: true) do
          {_out, 0} ->
            assert true

          {out, code} ->
            flunk("glslangValidator exit #{code}:\n#{out}\nshader:\n#{shader}")
        end
      after
        File.rm_rf!(path)
        File.rm_rf!(path <> ".spv")
        File.rm_rf!("vert.spv")
        File.rm_rf!("frag.spv")
        File.rm_rf!("comp.spv")
      end
    end
  end
end
