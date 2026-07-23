defmodule Exmc.NUTS.CustomSynth.Eval do
  @moduledoc """
  Parallel walker to `Exmc.NUTS.CustomSynth.Glsl` that evaluates an
  `Nx.Defn.Expr` tree directly on Elixir floats.

  Same op-handling structure as the GLSL emitter, just emitting
  numbers instead of GLSL strings.  Used by R1.5 to validate the
  walker is faithful to Defn's graph:

      tree = trace(fun, [templates])              # symbolic Expr
      reference = run_defn(fun, [concrete_args])  # Nx.Defn.Evaluator
      ours      = Eval.evaluate(tree, layout)     # our walker
      assert abs(reference - ours) < 1.0e-12

  If `Eval` agrees with the reference on random inputs, the GLSL
  walker (which uses the same dispatch structure, just with string
  leaves) is structurally correct — leaving only GLSL-vs-Vulkan
  numerical agreement for R2 to validate.
  """

  alias Nx.Defn.Expr
  alias Nx.Tensor, as: T

  @type layout :: [number()] | %{(atom() | binary() | non_neg_integer()) => number()}

  @doc """
  Walk the Expr tree, computing a scalar Elixir float using
  `layout` to resolve parameter accessors.

  - For positional params (`:parameter` op): `layout` is a list of
    floats indexed by position, OR a map from position-int to float.
  """
  @spec evaluate(T.t(), layout) :: {:ok, number()} | {:error, term()}
  def evaluate(%T{data: %Expr{op: op, args: args}}, layout) do
    do_eval(op, args, layout)
  end

  def evaluate(other, _layout), do: {:error, {:not_an_expr, other}}

  defp do_eval(:constant, [n], _layout) when is_number(n), do: {:ok, n * 1.0}

  defp do_eval(:parameter, [pos], layout) when is_integer(pos) do
    value =
      cond do
        is_list(layout) and pos < length(layout) -> Enum.at(layout, pos)
        is_map(layout) -> Map.get(layout, pos)
        true -> nil
      end

    cast_param_value(value, pos)
  end

  defp do_eval(:metadata, [inner, _meta], layout), do: evaluate(inner, layout)

  # Vector parameters (q, obs) come through as lists or Nx tensors;
  # `:slice` will index into them on the next walker step. Scalars
  # are normalised to f64 to match Nx.Defn.Evaluator's numeric path.
  defp cast_param_value(nil, pos), do: {:error, {:no_accessor_for_position, pos}}
  defp cast_param_value(n, _pos) when is_number(n), do: {:ok, n * 1.0}
  defp cast_param_value(list, _pos) when is_list(list), do: {:ok, list}
  defp cast_param_value(%Nx.Tensor{} = t, _pos), do: {:ok, t}
  defp cast_param_value(other, _pos), do: {:error, {:bad_layout_value, other}}

  # --- Arithmetic ---
  #
  # Every binary op routes through broadcast2/3 — a scalar+scalar
  # input produces a scalar (R1.5's contract); a scalar/vector or
  # vector/vector input produces an element-wise list. R2.2.2's
  # prior-only tests stay scalar end-to-end; R2.2.2-extended adds
  # vector parameters (obs) so the regime model's per-obs
  # likelihood can be reduced through :sum.

  for {op, kf} <- [add: &+/2, subtract: &-/2, multiply: &*/2, divide: &//2] do
    defp do_eval(unquote(op), [a, b], layout) do
      with {:ok, av} <- evaluate(a, layout),
           {:ok, bv} <- evaluate(b, layout) do
        {:ok, broadcast2(av, bv, unquote(kf))}
      end
    end
  end

  defp do_eval(:negate, [a], layout) do
    with {:ok, av} <- evaluate(a, layout), do: {:ok, broadcast1(av, &(-&1))}
  end

  defp do_eval(:abs, [a], layout) do
    with {:ok, av} <- evaluate(a, layout), do: {:ok, broadcast1(av, &abs/1)}
  end

  defp do_eval(:pow, [a, b], layout) do
    with {:ok, av} <- evaluate(a, layout),
         {:ok, bv} <- evaluate(b, layout) do
      {:ok, broadcast2(av, bv, &:math.pow/2)}
    end
  end

  defp do_eval(:min, [a, b], layout) do
    with {:ok, av} <- evaluate(a, layout),
         {:ok, bv} <- evaluate(b, layout), do: {:ok, broadcast2(av, bv, &min/2)}
  end

  defp do_eval(:max, [a, b], layout) do
    with {:ok, av} <- evaluate(a, layout),
         {:ok, bv} <- evaluate(b, layout), do: {:ok, broadcast2(av, bv, &max/2)}
  end

  defp do_eval(:remainder, [a, b], layout) do
    with {:ok, av} <- evaluate(a, layout),
         {:ok, bv} <- evaluate(b, layout) do
      {:ok, broadcast2(av, bv, &:math.fmod/2)}
    end
  end

  defp do_eval(:atan2, [a, b], layout) do
    with {:ok, av} <- evaluate(a, layout),
         {:ok, bv} <- evaluate(b, layout), do: {:ok, broadcast2(av, bv, &:math.atan2/2)}
  end

  # --- Transcendentals ---

  for {op, fun} <- [
        exp: &:math.exp/1,
        log: &:math.log/1,
        log1p: &:math.log1p/1,
        expm1: &:math.expm1/1,
        sqrt: &:math.sqrt/1,
        sin: &:math.sin/1,
        cos: &:math.cos/1,
        tan: &:math.tan/1,
        asin: &:math.asin/1,
        acos: &:math.acos/1,
        atan: &:math.atan/1,
        sinh: &:math.sinh/1,
        cosh: &:math.cosh/1,
        tanh: &:math.tanh/1,
        floor: &:math.floor/1,
        ceil: &:math.ceil/1
      ] do
    defp do_eval(unquote(op), [a], layout) do
      with {:ok, av} <- evaluate(a, layout), do: {:ok, broadcast1(av, unquote(fun))}
    end
  end

  defp do_eval(:rsqrt, [a], layout) do
    with {:ok, av} <- evaluate(a, layout),
         do: {:ok, broadcast1(av, &(1.0 / :math.sqrt(&1)))}
  end

  defp do_eval(:sigmoid, [a], layout) do
    with {:ok, av} <- evaluate(a, layout),
         do: {:ok, broadcast1(av, &(1.0 / (1.0 + :math.exp(-&1))))}
  end

  defp do_eval(:softplus, [a], layout) do
    with {:ok, av} <- evaluate(a, layout) do
      sp = fn x -> :math.log(1.0 + :math.exp(-abs(x))) + max(x, 0.0) end
      {:ok, broadcast1(av, sp)}
    end
  end

  # --- Comparison + select ---

  for {op, kf} <- [
        less: &</2,
        less_equal: &<=/2,
        greater: &>/2,
        greater_equal: &>=/2,
        equal: &==/2,
        not_equal: &!=/2
      ] do
    defp do_eval(unquote(op), [a, b], layout) do
      with {:ok, av} <- evaluate(a, layout),
           {:ok, bv} <- evaluate(b, layout) do
        cmp = fn x, y -> if(unquote(kf).(x, y), do: 1, else: 0) end
        {:ok, broadcast2(av, bv, cmp)}
      end
    end
  end

  defp do_eval(:select, [cond_expr, on_true, on_false], layout) do
    with {:ok, c} <- evaluate(cond_expr, layout),
         {:ok, t} <- evaluate(on_true, layout),
         {:ok, f} <- evaluate(on_false, layout) do
      {:ok, select_broadcast(c, t, f)}
    end
  end

  # --- Shape-only passthroughs (scalar context) ---

  defp do_eval(:squeeze, [inner | _], layout), do: evaluate(inner, layout)
  defp do_eval(:reshape, [inner | _], layout), do: evaluate(inner, layout)
  defp do_eval(:as_type, [inner | _], layout), do: evaluate(inner, layout)
  defp do_eval(:broadcast, [inner | _], layout), do: evaluate(inner, layout)

  # --- Slice (vector parameter indexing) ---

  # Parallels CustomSynth.Glsl's :slice clause: only handles the
  # length-1 single-index slice produced by `q[i]` access against a
  # vector parameter. Returns the indexed scalar as an f64 float.
  defp do_eval(:slice, [tensor, start_indices, lengths, strides], layout) do
    with true <- all_ones?(strides) || {:error, :strided_slice},
         true <- all_ones?(lengths) || {:error, :multi_element_slice},
         {:ok, start_idx} <- single_start_idx(start_indices),
         {:ok, base} <- evaluate(tensor, layout) do
      index_into(base, start_idx)
    else
      {:error, _} = e -> e
      _ -> {:error, {:unsupported_slice_shape, lengths, strides}}
    end
  end

  defp all_ones?(list) when is_list(list), do: Enum.all?(list, &(&1 == 1))
  defp all_ones?(_), do: false

  defp single_start_idx([%Nx.Tensor{data: %Nx.Defn.Expr{op: :constant, args: [n]}}])
       when is_integer(n),
       do: {:ok, n}

  defp single_start_idx([n]) when is_integer(n), do: {:ok, n}
  defp single_start_idx(other), do: {:error, {:non_constant_start_idx, other}}

  defp index_into(list, idx) when is_list(list) and idx < length(list) do
    {:ok, Enum.at(list, idx) * 1.0}
  end

  defp index_into(%Nx.Tensor{} = t, idx) do
    {:ok, t[idx] |> Nx.to_number()}
  end

  defp index_into(other, idx), do: {:error, {:bad_slice_base, other, idx}}

  # --- Reduction ---

  # `:sum` reduces an axis-0 vector to a scalar. R2.2.2-extended:
  # the regime model's per-obs likelihood is summed across the obs
  # axis to produce the joint log-density.
  defp do_eval(:sum, [a, _axes_or_opts], layout) do
    with {:ok, av} <- evaluate(a, layout) do
      cond do
        is_number(av) -> {:ok, av * 1.0}
        is_list(av) -> {:ok, Enum.sum(av) * 1.0}
        true -> {:error, {:bad_sum_arg, av}}
      end
    end
  end

  # --- Broadcast helpers ---

  defp broadcast1(v, fun) when is_number(v), do: fun.(v)
  defp broadcast1(v, fun) when is_list(v), do: Enum.map(v, fun)

  defp broadcast2(a, b, fun) when is_number(a) and is_number(b), do: fun.(a, b)

  defp broadcast2(a, b, fun) when is_list(a) and is_list(b) and length(a) == length(b),
    do: Enum.zip(a, b) |> Enum.map(fn {x, y} -> fun.(x, y) end)

  defp broadcast2(a, b, fun) when is_number(a) and is_list(b),
    do: Enum.map(b, fn y -> fun.(a, y) end)

  defp broadcast2(a, b, fun) when is_list(a) and is_number(b),
    do: Enum.map(a, fn x -> fun.(x, b) end)

  defp broadcast2(a, b, _fun),
    do: raise("Eval.broadcast2 shape mismatch: #{inspect(a)} vs #{inspect(b)}")

  defp select_broadcast(c, t, f) when is_number(c),
    do: if(c != 0 and c != 0.0, do: t, else: f)

  defp select_broadcast(c, t, f) when is_list(c) and is_list(t) and is_list(f) do
    [c, t, f]
    |> Enum.zip()
    |> Enum.map(fn {cv, tv, fv} -> if(cv != 0 and cv != 0.0, do: tv, else: fv) end)
  end

  defp select_broadcast(c, t, f) when is_list(c) and is_number(t) and is_number(f) do
    Enum.map(c, fn cv -> if(cv != 0 and cv != 0.0, do: t, else: f) end)
  end

  # --- Catch-all ---

  defp do_eval(op, _args, _layout) do
    {:error, {:unsupported_op, op}}
  end
end
