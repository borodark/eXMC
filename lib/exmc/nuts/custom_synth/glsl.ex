defmodule Exmc.NUTS.CustomSynth.Glsl do
  @moduledoc """
  Emit GLSL fragments from `Nx.Defn.Expr` trees.

  This is the load-bearing piece of M-II R1
  (`specs/vulkan-custom-synthesis.md`).  Walk an `Nx.Defn.Expr`
  tree and produce a single GLSL scalar expression string.

  The emitter is **scalar-first**: it treats input tensors of
  shape `{}` as `float` and tensors of shape `{n}` as already-
  indexed scalars (the leapfrog template's outer loop binds the
  thread index `i`).  Larger-rank tensors are out of scope for
  R1 — those land in R2 / Mission III Layer 2.

  ## Coverage (R1 — sufficient for the regime model)

  Element-wise:  add, subtract, multiply, divide, negate, abs,
                 max, min, exp, log, log1p, expm1,
                 sigmoid (1/(1+exp(-x))), tanh,
                 power (a^b → pow(a,b)).

  Reductions:    sum (axis-0) — emitted as a parallel-reduction
                 stub in the leapfrog template; the emitter
                 marks reductions for the template renderer.

  Constants:     `:constant` op → GLSL literal.

  Parameters:    `:parameter` op → reference to a named SSBO
                 read or push-constant load, per the
                 layout map passed in by the caller.

  Anything outside this set returns `{:error, {:unsupported_op, op}}`
  so the synthesis pipeline can short-circuit to per-op dispatch
  (the existing slow path).

  ## Closure-captured tensors (`:tensor` op)

  When a defn-traced closure captures a non-scalar `Nx.tensor` (e.g.
  a linear-regression validator's `y_tensor` or `x_cols`), Nx.Defn
  lowers it to a `:tensor` op node carrying the original tensor as
  its arg (`Nx.Defn.Expr.to_expr/1`). The emitter handles these via
  a process-local capture channel: each unique tensor reference
  gets a deterministic GLSL accessor `__captured_<hash>[j]` (or
  `[i]` — index variable picked by the surrounding loop the
  orchestrator wraps the expression in) AND is recorded in the
  current capture buffer. The orchestrator (typically
  `MultiRvCustomSpec.render_batched/1`) is expected to bracket its
  `emit/2` calls with `start_captures/0` and `collect_captures/0`,
  then emit `const float __captured_<hash>[N] = float[](...);`
  declarations in the shader prelude.

  Inline `const float[]` works for the synth-coverage probe and
  for small obs tensors. Per-instance batched dispatch requires
  the captures to move to SSBO bindings, with the obs registered
  via `Builder.data/2` at the model-builder layer — that is
  follow-up work, not this clause.

  Scalar (`shape: {}`) tensor captures are folded to `:constant`
  by Nx.Defn before reaching this emitter (see
  `Nx.Defn.Expr.to_expr/1`); the `:tensor` clause never sees
  rank-0 tensors in practice, but handles them defensively as
  inline float literals.

  Rank ≥ 2 captures return `{:error, {:unsupported_rank, rank}}`.
  """

  alias Nx.Defn.Expr
  alias Nx.Tensor, as: T

  @captures_key :__exmc_glsl_captures__
  @loop_index_var "j"

  @type layout :: %{(atom() | binary()) => binary()} | [binary()]
  @type emit_result :: {:ok, binary()} | {:error, term()}
  @type capture_entry :: %{
          name: binary(),
          values: [number()],
          length: non_neg_integer(),
          dtype: atom()
        }

  @doc """
  Start a per-process capture buffer for closure-captured tensors.

  Call this once before a sequence of `emit/2` / `emit_vector/2`
  invocations that may encounter `:tensor` ops. Tensors registered
  during emission are accumulated in process dictionary; retrieve
  them with `collect_captures/0`.

  Clears any prior buffer in the current process.
  """
  @spec start_captures() :: :ok
  def start_captures do
    Process.put(@captures_key, %{})
    :ok
  end

  @doc """
  Drain the per-process capture buffer.

  Returns a list of `capture_entry` maps in deterministic
  insertion order (by hash, since they are keyed by
  `:erlang.phash2/1` of the tensor). The orchestrator emits one
  `const float __captured_<name>[N] = float[](...);` declaration
  per entry in the shader prelude.

  Clears the buffer.
  """
  @spec collect_captures() :: [capture_entry()]
  def collect_captures do
    captures = Process.get(@captures_key, %{})
    Process.delete(@captures_key)

    captures
    |> Map.values()
    |> Enum.sort_by(& &1.name)
  end

  @typedoc "Vector-output emission result: list of (index, scalar GLSL expr)."
  @type vector_emit_result :: {:ok, [{non_neg_integer(), binary()}]} | {:error, term()}

  @doc """
  Emit a vector-output expression as a list of per-position scalar
  assignments.

  Vector gradients in `Nx.Defn` are built as right-leaning `:add`
  trees of `:put_slice(zeros, [idx], pad(broadcast(scalar)))` leaves.
  This function walks that pattern and returns `[{idx, glsl}, ...]`
  where each entry is one position's scalar contribution.  Multiple
  entries with the same idx are valid (Defn lowering sometimes
  splits a partial across branches) — the leapfrog template
  composer is expected to write them as `out[idx] += <glsl>`.

  Walks (`:add`, `:put_slice`, `:pad`, `:broadcast`) are unwrapped
  in-emitter; once we reach the scalar contribution, `emit/2` does
  the rest.

  Returns `{:error, {:unsupported_op, op}}` if a non-scatter-pattern
  op shows up.
  """
  @spec emit_vector(T.t(), layout) :: vector_emit_result()
  def emit_vector(%T{data: %Expr{op: :add, args: [a, b]}}, layout) do
    with {:ok, l1} <- emit_vector(a, layout),
         {:ok, l2} <- emit_vector(b, layout) do
      {:ok, l1 ++ l2}
    end
  end

  def emit_vector(
        %T{data: %Expr{op: :put_slice, args: [_zeros, [idx_tensor], slice_expr]}},
        layout
      ) do
    with {:ok, idx} <- extract_index(idx_tensor),
         {:ok, glsl} <- emit_scatter_value(slice_expr, layout) do
      {:ok, [{idx, glsl}]}
    end
  end

  # The :constant-zero base of a put_slice chain contributes nothing.
  def emit_vector(%T{data: %Expr{op: :constant, args: [n]}}, _layout)
      when is_number(n) and n == 0 do
    {:ok, []}
  end

  def emit_vector(%T{data: %Expr{op: op}}, _layout) do
    {:error, {:unsupported_vec_op, op}}
  end

  # Unwrap shape-only ops that wrap a scalar contribution.  Defn
  # synthesises `pad(broadcast(squeeze(scalar)))` or similar chains
  # to coerce a scalar into the {1}-shape required by put_slice.
  # All of these are no-ops on the underlying scalar value.
  defp emit_scatter_value(%T{data: %Expr{op: :pad, args: [inner, _pad_val, _config]}}, layout) do
    emit_scatter_value(inner, layout)
  end

  defp emit_scatter_value(%T{data: %Expr{op: :broadcast, args: [inner | _]}}, layout) do
    emit_scatter_value(inner, layout)
  end

  defp emit_scatter_value(%T{data: %Expr{op: :squeeze, args: [inner | _]}}, layout) do
    emit_scatter_value(inner, layout)
  end

  defp emit_scatter_value(%T{data: %Expr{op: :reshape, args: [inner | _]}}, layout) do
    emit_scatter_value(inner, layout)
  end

  defp emit_scatter_value(%T{data: %Expr{op: :as_type, args: [inner | _]}}, layout) do
    emit_scatter_value(inner, layout)
  end

  defp emit_scatter_value(scalar, layout) do
    emit(scalar, layout)
  end

  # Pull the integer index out of a constant tensor.
  defp extract_index(%T{data: %Expr{op: :constant, args: [n]}}) when is_integer(n), do: {:ok, n}

  defp extract_index(other) do
    {:error, {:non_constant_index, other}}
  end

  @doc """
  Emit a GLSL scalar expression for an `Nx.Defn.Expr` tree.

  - `expr` is a tensor with an `Expr`-shaped data slot (from
    `Nx.Defn.debug_expr_apply/3`).
  - `layout` is either:
    - a list of GLSL accessor strings indexed by Defn parameter
      position (positional, matching the order of args passed to
      `debug_expr_apply`), or
    - a map from named ids to accessors (used after R1.5 wires
      Composite-aware param flattening).

  Returns `{:ok, glsl_expr_string}` on success.
  """
  @spec emit(T.t(), layout) :: emit_result()
  def emit(%T{data: %Expr{op: op, args: args}} = _expr, layout) do
    do_emit(op, args, layout)
  end

  def emit(other, _layout), do: {:error, {:not_an_expr, other}}

  # --- Constants & parameters ---

  defp do_emit(:constant, [number], _layout) when is_number(number) do
    {:ok, format_float(number)}
  end

  defp do_emit(:parameter, [pos], layout) when is_integer(pos) do
    cond do
      is_list(layout) and pos < length(layout) ->
        {:ok, Enum.at(layout, pos)}

      is_map(layout) and map_size(layout) == 0 ->
        # No layout supplied — default to a generic q[pos] accessor.
        # Sufficient for syntactic emit tests; real use must supply
        # one.
        {:ok, "q[#{pos}]"}

      true ->
        {:error, {:no_accessor_for_position, pos}}
    end
  end

  defp do_emit(:metadata, [inner, meta], layout) do
    case meta do
      %{name: name} when is_atom(name) or is_binary(name) ->
        case Map.fetch(layout, name) do
          {:ok, accessor} -> {:ok, accessor}
          :error -> emit(inner, layout)
        end

      _ ->
        emit(inner, layout)
    end
  end

  # --- Element-wise binary ops ---

  binary_ops_arith = %{
    add: "+",
    subtract: "-",
    multiply: "*",
    divide: "/"
  }

  for {op, glyph} <- binary_ops_arith do
    defp do_emit(unquote(op), [a, b], layout) do
      with {:ok, a_s} <- emit(a, layout),
           {:ok, b_s} <- emit(b, layout) do
        {:ok, "(#{a_s} #{unquote(glyph)} #{b_s})"}
      end
    end
  end

  binary_ops_fn = [:min, :max, :pow, :atan2]

  for op <- binary_ops_fn do
    glsl_fn = if op == :pow, do: "pow", else: to_string(op)

    defp do_emit(unquote(op), [a, b], layout) do
      with {:ok, a_s} <- emit(a, layout),
           {:ok, b_s} <- emit(b, layout) do
        {:ok, "#{unquote(glsl_fn)}(#{a_s}, #{b_s})"}
      end
    end
  end

  defp do_emit(:remainder, [a, b], layout) do
    with {:ok, a_s} <- emit(a, layout),
         {:ok, b_s} <- emit(b, layout) do
      {:ok, "mod(#{a_s}, #{b_s})"}
    end
  end

  # Nx.clip(x, lo, hi) → min(max(x, lo), hi). Bernoulli (and other
  # bounded-support) likelihoods clip probabilities away from 0/1 so
  # log() stays finite. lo/hi are typically constant tensors.
  defp do_emit(:clip, [a, lo, hi], layout) do
    with {:ok, a_s} <- emit(a, layout),
         {:ok, lo_s} <- emit(lo, layout),
         {:ok, hi_s} <- emit(hi, layout) do
      {:ok, "min(max(#{a_s}, #{lo_s}), #{hi_s})"}
    end
  end

  # --- Element-wise unary ops ---

  unary_ops_fn = [
    :exp, :log, :log1p, :expm1, :sqrt, :rsqrt,
    :sin, :cos, :tan, :asin, :acos, :atan,
    :sinh, :cosh, :tanh, :abs, :floor, :ceil
  ]

  for op <- unary_ops_fn do
    glsl_fn =
      case op do
        :log1p -> "log_1p_safe"
        :expm1 -> "expm1_safe"
        :abs -> "abs"
        :rsqrt -> "inversesqrt"
        _ -> to_string(op)
      end

    defp do_emit(unquote(op), [a], layout) do
      with {:ok, a_s} <- emit(a, layout) do
        {:ok, "#{unquote(glsl_fn)}(#{a_s})"}
      end
    end
  end

  defp do_emit(:negate, [a], layout) do
    with {:ok, a_s} <- emit(a, layout), do: {:ok, "(-(#{a_s}))"}
  end

  defp do_emit(:sigmoid, [a], layout) do
    with {:ok, a_s} <- emit(a, layout) do
      {:ok, "(1.0 / (1.0 + exp(-(#{a_s}))))"}
    end
  end

  defp do_emit(:softplus, [a], layout) do
    # Numerically stable softplus: log1p(exp(-|x|)) + max(x, 0).
    # See Honkela ch. on "Computing with probabilities".
    with {:ok, a_s} <- emit(a, layout) do
      {:ok, "(log(1.0 + exp(-abs(#{a_s}))) + max(#{a_s}, 0.0))"}
    end
  end

  # --- Comparison + select (the gradient-of-min/max pattern needs these) ---

  comparison_ops_glyph = %{
    less: "<",
    less_equal: "<=",
    greater: ">",
    greater_equal: ">=",
    equal: "==",
    not_equal: "!="
  }

  for {op, glyph} <- comparison_ops_glyph do
    defp do_emit(unquote(op), [a, b], layout) do
      with {:ok, a_s} <- emit(a, layout),
           {:ok, b_s} <- emit(b, layout) do
        # GLSL bool — wrap in parens so the caller (typically :select)
        # can use it as a ternary condition.
        {:ok, "(#{a_s} #{unquote(glyph)} #{b_s})"}
      end
    end
  end

  # Boolean combinators for compound select conditions. The gradient of
  # Nx.clip(x, lo, hi) is the indicator `and(x > lo, x < hi)`, so the mask
  # for its :select is a bitwise_and of two comparison bools. On GLSL bool
  # operands these map to logical && / ||.
  defp do_emit(:bitwise_and, [a, b], layout) do
    with {:ok, a_s} <- emit(a, layout),
         {:ok, b_s} <- emit(b, layout) do
      {:ok, "(#{a_s} && #{b_s})"}
    end
  end

  defp do_emit(:bitwise_or, [a, b], layout) do
    with {:ok, a_s} <- emit(a, layout),
         {:ok, b_s} <- emit(b, layout) do
      {:ok, "(#{a_s} || #{b_s})"}
    end
  end

  # Nx.Defn emits :select(cond, on_true, on_false) for indicator-shaped
  # gradients (e.g. ∂min(x,c)/∂x = 1 if x < c else 0).  GLSL has no
  # ternary on scalar floats but the standard idiom is `(cond ? a : b)`
  # which glsl 4.50 accepts on numeric types; alternatively use
  # `mix(b, a, float(cond))` but the ternary is clearer.
  defp do_emit(:select, [cond_expr, on_true, on_false], layout) do
    with {:ok, c_s} <- emit(cond_expr, layout),
         {:ok, t_s} <- emit(on_true, layout),
         {:ok, f_s} <- emit(on_false, layout) do
      {:ok, "(#{c_s} ? #{t_s} : #{f_s})"}
    end
  end

  # --- Shape-only passthroughs ---

  # In scalar position, these are no-ops on the underlying value.
  # Real broadcasts that change rank get caught by emit_vector's
  # scatter unwrap; here we just forward.
  defp do_emit(:squeeze, [inner | _], layout), do: emit(inner, layout)
  defp do_emit(:reshape, [inner | _], layout), do: emit(inner, layout)
  defp do_emit(:as_type, [inner | _], layout), do: emit(inner, layout)

  # Length-1 slices on a tensor parameter are how Defn lowers
  # `q[i]`.  Emit the parameter accessor with an integer subscript.
  # For other slice patterns (arbitrary stride / length / rank>1
  # source), bail — those need real array emission, which is
  # Mission III Layer 2.
  defp do_emit(:slice, [tensor, start_indices, lengths, strides], layout) do
    with true <- all_ones?(strides) || {:error, :strided_slice},
         true <- all_ones?(lengths) || {:error, :multi_element_slice},
         {:ok, start_idx} <- single_start_idx(start_indices),
         {:ok, base} <- emit(tensor, layout) do
      # Strip trailing [n] if base already has one, else append.
      # Typical case: base is "q[0]" wait no — base could be a
      # parameter accessor like "q" or "obs".  For "q" we want
      # "q[start_idx]".
      {:ok, "#{base}[#{start_idx}]"}
    else
      {:error, _} = e -> e
      _ -> {:error, {:unsupported_slice_shape, lengths, strides}}
    end
  end

  # --- Closure-captured tensors ---

  # Scalar `Nx.tensor(...)` captures never reach here in practice —
  # Nx.Defn.Expr.to_expr folds shape-{} BinaryBackend tensors to
  # `:constant` before tracing. Handle defensively anyway.
  defp do_emit(:tensor, [%T{shape: {}} = t], _layout) do
    {:ok, format_float(Nx.to_number(t))}
  end

  defp do_emit(:tensor, [%T{shape: {_n}} = t], _layout) do
    {:ok, register_capture(t)}
  end

  defp do_emit(:tensor, [%T{shape: shape}], _layout) do
    {:error, {:unsupported_rank, tuple_size(shape)}}
  end

  # --- Reductions (R1 marker, full handling in R2 template) ---

  defp do_emit(:sum, [a, _axes_or_opts], layout) do
    # Reductions can't be inlined into a scalar GLSL expression
    # — they need a parallel reduction loop in the kernel body.
    # The emitter marks the inner expression so the leapfrog
    # template renderer knows to wrap it.
    with {:ok, inner} <- emit(a, layout) do
      {:ok, "/*REDUCE_SUM*/(#{inner})"}
    end
  end

  # --- Catch-all ---

  defp do_emit(op, _args, _layout) do
    {:error, {:unsupported_op, op}}
  end

  # --- Helpers ---

  defp all_ones?(list) when is_list(list), do: Enum.all?(list, &(&1 == 1))
  defp all_ones?(_), do: false

  defp single_start_idx([%T{data: %Expr{op: :constant, args: [n]}}]) when is_integer(n), do: {:ok, n}
  defp single_start_idx([n]) when is_integer(n), do: {:ok, n}
  defp single_start_idx(other), do: {:error, {:non_constant_start_idx, other}}

  defp format_float(n) when is_integer(n), do: "#{n}.0"

  defp format_float(n) when is_float(n) do
    s = Float.to_string(n)

    cond do
      s == "Inf" or s == "-Inf" -> raise "infinity in GLSL emission: #{s}"
      s == "NaN" -> raise "NaN in GLSL emission"
      String.contains?(s, ".") or String.contains?(s, "e") -> s
      true -> s <> ".0"
    end
  end

  # Register a rank-1 captured tensor and return its GLSL accessor.
  # Idempotent on tensor identity (same `:erlang.phash2/1` hash →
  # same accessor, no duplicate registration).
  defp register_capture(%T{shape: {n}, type: type} = t) do
    hash = :erlang.phash2(Nx.to_binary(t))
    name = "__captured_t#{hash}"
    accessor = "#{name}[#{@loop_index_var}]"

    captures = Process.get(@captures_key, %{})

    case Map.fetch(captures, hash) do
      {:ok, _existing} ->
        accessor

      :error ->
        entry = %{
          name: name,
          values: Nx.to_flat_list(t),
          length: n,
          dtype: elem(type, 0)
        }

        Process.put(@captures_key, Map.put(captures, hash, entry))
        accessor
    end
  end
end
