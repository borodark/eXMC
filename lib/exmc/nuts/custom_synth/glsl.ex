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
          dtype: atom(),
          offset: non_neg_integer()
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
    |> Enum.sort_by(& &1.offset)
  end

  # --- Common-subexpression elimination (CSE) -----------------------
  #
  # The Defn graph is a DAG, but this emitter walks it as a tree, so a
  # subexpression shared by K parents is re-emitted K times. In a
  # softmax-mixture gradient a single denominator can recur thousands of
  # times, producing a multi-hundred-KB shader that recomputes exp/log on
  # the GPU per obs per leapfrog step. start_cse/1 marks the shared node
  # ids; emit/2 hoists each to a `double _cseN = <expr>;` binding the first
  # time it's seen and returns `_cseN` after. collect_cse/0 drains the
  # bindings in dependency order (children bound before parents).

  @cse_key :exmc_glsl_cse

  @doc "Begin a CSE scope. `shared` = MapSet of Expr ids referenced >1×."
  @spec start_cse(MapSet.t()) :: :ok
  def start_cse(shared) do
    Process.put(@cse_key, %{shared: shared, vars: %{}, binds: [], n: 0})
    :ok
  end

  @doc "Drain CSE bindings (dependency order) and end the scope."
  @spec collect_cse() :: [binary()]
  def collect_cse do
    st = Process.get(@cse_key)
    Process.delete(@cse_key)

    case st do
      %{binds: b} -> Enum.reverse(b)
      _ -> []
    end
  end

  @doc "Set of Expr node ids referenced more than once in `expr`."
  @spec shared_ids(T.t()) :: MapSet.t()
  def shared_ids(expr) do
    for {id, c} <- count_refs(expr, %{}), c > 1, into: MapSet.new(), do: id
  end

  defp count_refs(%T{data: %Expr{id: id, args: args}}, acc) do
    case acc do
      %{^id => c} -> Map.put(acc, id, c + 1)
      _ -> Enum.reduce(args, Map.put(acc, id, 1), &count_refs/2)
    end
  end

  defp count_refs(list, acc) when is_list(list), do: Enum.reduce(list, acc, &count_refs/2)
  defp count_refs(_leaf, acc), do: acc

  @doc """
  Ids of nodes whose subtree references Defn parameter position `pos`.

  Used to keep obs-dependent subexpressions (`obs_j`, an obs-loop variable)
  out of the loop-invariant CSE bucket — only obs-independent shared nodes
  may be hoisted above the `for (j < n_obs)` loop.
  """
  @spec dependent_ids(T.t(), non_neg_integer()) :: MapSet.t()
  def dependent_ids(expr, pos) do
    memo = dep_memo(expr, pos, %{})
    for {id, true} <- memo, into: MapSet.new(), do: id
  end

  defp dep_memo(%T{data: %Expr{id: id, op: op, args: args}}, pos, memo) do
    if Map.has_key?(memo, id) do
      memo
    else
      memo = Enum.reduce(args, memo, &dep_memo(&1, pos, &2))
      dep = (op == :parameter and args == [pos]) or Enum.any?(args, &arg_dep?(&1, memo))
      Map.put(memo, id, dep)
    end
  end

  defp dep_memo(list, pos, memo) when is_list(list),
    do: Enum.reduce(list, memo, &dep_memo(&1, pos, &2))

  defp dep_memo(_leaf, _pos, memo), do: memo

  defp arg_dep?(%T{data: %Expr{id: id}}, memo), do: Map.get(memo, id, false)
  defp arg_dep?(_, _), do: false

  defp cse_action(id) do
    case Process.get(@cse_key) do
      %{shared: shared, vars: vars} ->
        cond do
          Map.has_key?(vars, id) -> {:bound, Map.fetch!(vars, id)}
          MapSet.member?(shared, id) -> :hoist
          true -> :inline
        end

      _ ->
        :inline
    end
  end

  defp cse_hoist(id, glsl) do
    st = Process.get(@cse_key)
    var = "_cse#{st.n}"

    Process.put(@cse_key, %{
      st
      | vars: Map.put(st.vars, id, var),
        binds: ["double #{var} = #{glsl};" | st.binds],
        n: st.n + 1
    })

    var
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
  def emit(%T{data: %Expr{id: id, op: op, args: args}} = _expr, layout) do
    case cse_action(id) do
      {:bound, var} -> {:ok, var}
      :hoist -> with {:ok, s} <- do_emit(op, args, layout), do: {:ok, cse_hoist(id, s)}
      :inline -> do_emit(op, args, layout)
    end
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

  binary_ops_fn = [:min, :max, :atan2]

  for op <- binary_ops_fn do
    defp do_emit(unquote(op), [a, b], layout) do
      with {:ok, a_s} <- emit(a, layout),
           {:ok, b_s} <- emit(b, layout) do
        {:ok, "#{unquote(to_string(op))}(#{a_s}, #{b_s})"}
      end
    end
  end

  # --- Nx.pow ---
  #
  # `pow` is NOT emitted, and this is not a style choice.
  #
  # GLSL.std.450's `Pow` is declared on `float` only; there is no `double`
  # overload, and the whole chain shader is f64. So `Nx.pow(x, 2)` produced
  # GLSL that glslangValidator rejected outright --
  #
  #     'pow' : no matching overloaded function found
  #
  # -- which killed the shader for any Gaussian log-density written the
  # obvious way, `(x - mu)^2` being how everyone writes it. Including us: the
  # `Exmc.Dist.Custom` moduledoc taught exactly that form. Found by the
  # pathmc_ex session, whose likelihoods all used it, with a five-cell probe
  # that isolated it from two other suspected causes.
  #
  # A `pow_d(x, y) = exp_d(y * log_d(x))` helper, the obvious mirror of the
  # exp_d/log_d ones next door, would be WRONG here and worse than the error
  # it replaces: `log` of a negative number is NaN, and the base is a residual
  # `(x - mu)` that is negative about half the time. That trades a compile
  # error for a silent NaN, which is the exact trade this project keeps
  # finding and undoing.
  #
  # So: a constant integer exponent unrolls to multiplication, which is exact,
  # sign-correct and needs no helper. Anything else refuses, and the model
  # takes the host path -- correct, slower -- rather than reaching a shader
  # that would be undefined for negative bases. GLSL's own `pow` is undefined
  # for x < 0 too, so refusing loses nothing that was ever well-defined.
  @max_pow_unroll 8

  defp do_emit(:pow, [a, b], layout) do
    with {:ok, a_s} <- emit(a, layout) do
      case integer_exponent(b) do
        {:ok, n} when n >= 0 and n <= @max_pow_unroll ->
          {:ok, unroll_pow(a_s, n)}

        {:ok, n} when n < 0 and -n <= @max_pow_unroll ->
          {:ok, "(1.0lf / #{unroll_pow(a_s, -n)})"}

        _ ->
          {:error, {:unsupported_op, :pow}}
      end
    end
  end

  # Matched on the EXPRESSION TREE rather than on emitted text: a constant
  # reaches `do_emit(:constant, ...)` as a number and comes back formatted, so
  # parsing the string back would be re-deriving what we already had.
  defp integer_exponent(%T{data: %Expr{op: :constant, args: [n]}}) when is_integer(n),
    do: {:ok, n}

  defp integer_exponent(%T{data: %Expr{op: :constant, args: [n]}}) when is_float(n) do
    if n == Float.round(n), do: {:ok, trunc(n)}, else: :error
  end

  defp integer_exponent(_), do: :error

  defp unroll_pow(_a_s, 0), do: "1.0lf"
  defp unroll_pow(a_s, 1), do: "(#{a_s})"

  # x^n as n factors. The text repeats, which looks wasteful and is not: the
  # CSE pass over the fused obs-loop body hoists a repeated subexpression into
  # one local, so `(r) * (r)` costs one binding and one multiply.
  defp unroll_pow(a_s, n) when n > 1 do
    "(" <> Enum.map_join(1..n, " * ", fn _ -> "(#{a_s})" end) <> ")"
  end

  # --- Nx.dot over a constant design matrix and a parameter vector ---
  #
  # `Nx.dot(X, beta)` with X a captured `{n, p}` tensor and beta a `{p}`
  # parameter vector is THE regression idiom, and it was refused twice over:
  # `:dot` had no clause at all, and a rank-2 capture returns
  # `{:error, {:unsupported_rank, 2}}` because the extras SSBO is flat.
  #
  # Both go away by unrolling the contraction. p is known at synthesis time, so
  #
  #     dot(X, beta)[j]  ==  X[j,0]*beta_0 + ... + X[j,p-1]*beta_{p-1}
  #
  # and each `X[.., k]` is a rank-1 column the existing capture machinery
  # already handles. The SSBO stays flat, no rank-2 support is needed, and the
  # result is an ordinary obs-axis scalar expression in `j` — exactly what the
  # REDUCE_SUM loop consumes.
  #
  # Only the `[1], [], ..., [0], []` axis signature is matched: contract X's
  # last axis with beta's only axis, no batching. Anything else falls through
  # to the catch-all and is refused rather than silently contracted along the
  # wrong axis.
  defp do_emit(:dot, [a, [1], [], b, [0], []], layout) do
    with {:ok, m} <- const_matrix(a),
         {:ok, elems} <- emit_param_vec(b, layout) do
      {_n, p} = m.shape

      if length(elems) == p do
        terms =
          elems
          |> Enum.with_index()
          |> Enum.map(fn {e, k} -> "(#{register_capture(m[[.., k]])} * #{e})" end)

        {:ok, "(" <> Enum.join(terms, " + ") <> ")"}
      else
        {:error, {:dot_shape_mismatch, p, length(elems)}}
      end
    end
  end

  # --- inner product over the OBSERVATION axis ---
  #
  # Reverse-mode AD of `dot(X, beta)` produces, for each parameter k, the
  # contraction `sum_j X[j,k] * r_j` — two rank-1 operands contracted to a
  # scalar. That is an obs-axis reduction, so it emits as the same
  # `/*REDUCE_SUM*/` marker `Nx.sum` does, over the product of the two
  # operands' per-`j` expressions.
  #
  # Both operands must be rank 1. A rank-2 operand here would be a contraction
  # this clause has no right to guess the axis order of, and falls through to
  # the catch-all.
  defp do_emit(:dot, [%T{shape: {n}} = a, [0], [], %T{shape: {n}} = b, [0], []], layout) do
    with {:ok, a_s} <- emit(a, layout),
         {:ok, b_s} <- emit(b, layout) do
      {:ok, "/*REDUCE_SUM*/((#{a_s}) * (#{b_s}))"}
    end
  end

  defp const_matrix(%T{data: %Expr{op: :tensor, args: [%T{shape: {_n, _p}} = t]}}), do: {:ok, t}
  defp const_matrix(_other), do: {:error, {:unsupported_op, :dot}}

  # A PARAMETER-AXIS vector: statically many scalar GLSL expressions, one per
  # free coordinate. Distinct from the observation axis, which stays implicit —
  # an obs-axis value is a single expression in `j`.
  #
  # This is how a `shape: {2}` RV reaches the emitter at all.
  # `MultiRvCustomSpec.slot_slice/2` builds it as `Nx.stack([q[off], q[off+1]])`
  # precisely because the emitter is otherwise scalar-valued and a
  # multi-element `Nx.slice` has no representation in it.
  defp emit_param_vec(%T{data: %Expr{op: :stack, args: [elems, _axis]}}, layout)
       when is_list(elems) do
    elems
    |> Enum.reduce_while({:ok, []}, fn e, {:ok, acc} ->
      case emit(e, layout) do
        {:ok, str} -> {:cont, {:ok, [str | acc]}}
        err -> {:halt, err}
      end
    end)
    |> case do
      {:ok, acc} -> {:ok, Enum.reverse(acc)}
      err -> err
    end
  end

  defp emit_param_vec(%T{data: %Expr{op: :reshape, args: [t | _]}}, layout),
    do: emit_param_vec(t, layout)

  defp emit_param_vec(_other, _layout), do: {:error, {:unsupported_op, :param_vec}}

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
    :exp,
    :log,
    :log1p,
    :expm1,
    :sqrt,
    :rsqrt,
    :sin,
    :cos,
    :tan,
    :asin,
    :acos,
    :atan,
    :sinh,
    :cosh,
    :tanh,
    :abs,
    :floor,
    :ceil
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
  # --- one component of a parameter-axis contraction ---
  #
  # This is the gradient half of `dot(X, beta)`, and the shape was MEASURED
  # rather than assumed after two guesses missed. Reverse-mode AD of a
  # likelihood in `dot(X, beta)` produces, at the leaf:
  #
  #   add{2} <- put_slice{2} <- pad{1} <- broadcast{1} <- squeeze{} <- slice{1}
  #                                                                     |
  #                            dot{2} [negate{40}, [0], [], tensor{40,2}, [0], []]
  #
  # `emit_scatter_value/2` strips pad/broadcast/squeeze and lands on the slice,
  # whose source is a rank-1 contracted with a rank-2 along axis 0 — the whole
  # {p} parameter vector at once, in a position where the emitter expects a
  # scalar. The generic `:slice` clause below then called `emit/2` on the dot
  # and got `{:unsupported_op, :dot}`.
  #
  # Taking the slice and the contraction TOGETHER is what makes it emittable:
  # component k is `sum_j v[j] * X[j,k]`, an ordinary observation-axis
  # reduction over the product of an obs-axis expression and one rank-1 column
  # of the captured matrix. No parameter-vector value ever has to exist.
  #
  # Both operand orders are matched because the contraction is symmetric and
  # unambiguous: whichever side is rank 2 supplies the columns.
  defp do_emit(
         :slice,
         [%T{data: %Expr{op: :dot, args: dot_args}} = _src, starts, lens, strides],
         layout
       )
       when is_list(dot_args) do
    with true <- all_ones?(strides) || {:error, :strided_slice},
         true <- all_ones?(lens) || {:error, :multi_element_slice},
         {:ok, k} <- single_start_idx(starts),
         {:ok, vec, mat} <- param_contraction(dot_args),
         {:ok, m} <- const_matrix(mat),
         {:ok, v_s} <- emit(vec, layout) do
      {:ok, "/*REDUCE_SUM*/((#{v_s}) * (#{register_capture(m[[.., k]])}))"}
    else
      {:error, _} = e -> e
      _ -> {:error, {:unsupported_slice_shape, lens, strides}}
    end
  end

  # A rank-1 contracted with a rank-2 on axis 0 of each, in either order.
  # Returns {vector_expr, matrix_expr}.
  defp param_contraction([%T{shape: {n}} = v, [0], [], %T{shape: {n, _p}} = m, [0], []]),
    do: {:ok, v, m}

  defp param_contraction([%T{shape: {n, _p}} = m, [0], [], %T{shape: {n}} = v, [0], []]),
    do: {:ok, v, m}

  defp param_contraction(_other), do: {:error, {:unsupported_op, :dot}}

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

  defp single_start_idx([%T{data: %Expr{op: :constant, args: [n]}}]) when is_integer(n),
    do: {:ok, n}

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
    captures = Process.get(@captures_key, %{})

    entry =
      case Map.fetch(captures, hash) do
        {:ok, existing} ->
          existing

        :error ->
          entry = %{
            name: "__captured_t#{hash}",
            values: Nx.to_flat_list(t),
            length: n,
            dtype: elem(type, 0),
            offset: next_capture_offset(captures)
          }

          Process.put(@captures_key, Map.put(captures, hash, entry))
          entry
      end

    capture_accessor(entry)
  end

  # Captures are a THIRD region of the extras SSBO, after obs and inv_mass:
  #
  #     obs[0, n_obs)  inv_mass[n_obs, n_obs+d)  captures[n_obs+d, ...)
  #
  # Appended after inv_mass rather than inserted before it so that every
  # pre-existing index expression -- `obs_inv_mass[j]` and
  # `obs_inv_mass[pc.n_obs + tid]` -- stays byte-identical. A change that
  # cannot perturb the paths it was not meant to touch is worth an extra
  # addend in the index.
  #
  # `j` is the reduce-loop variable and is a GLOBAL index into the
  # concatenated observation buffer in both span modes (`loop_lo/2` starts
  # marker n at its global offset), so a capture aligned to the whole obs
  # buffer is read correctly. `MultiRvCustomSpec` refuses the one shape where
  # that is not sound -- per-node spans WITH captures -- rather than emitting
  # a plausible wrong density.
  defp capture_accessor(%{offset: off}) do
    "obs_inv_mass[pc.n_obs + pc.d + #{off} + #{@loop_index_var}]"
  end

  # Offsets are assigned at first registration and never recomputed, so the
  # emitter is the single source of truth for the packing order. The binary
  # MUST be built from these same entries -- see
  # `MultiRvCustomSpec.captures_bin/1`. Two encoders that agree by
  # construction today and drift later is defect D1 in
  # docs/BATCHED_CHAIN_DISPATCH.md, and this is the same shape of trap.
  defp next_capture_offset(captures) do
    captures |> Map.values() |> Enum.reduce(0, fn e, acc -> acc + e.length end)
  end
end
