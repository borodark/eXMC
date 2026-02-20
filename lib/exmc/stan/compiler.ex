defmodule Exmc.Stan.Compiler do
  @moduledoc """
  Compiles a Stan AST into an `Exmc.IR`.

  Walks the parsed AST and emits `Exmc.Builder` calls to construct the IR.
  Data variables become tensor lookups; parameters become free RVs;
  sampling statements on data variables become observations.
  """

  alias Exmc.Builder
  alias Exmc.Stan.{AST, DistMap}

  @doc """
  Compile a parsed Stan AST with a data map into an Exmc IR.

  ## Parameters

    * `ast` - parsed program tuple from yecc: `{:program, data_decls, param_decls, model_stmts}`
    * `data` - map of data variable name (string) to Nx tensor values

  ## Returns

    `{:ok, %IR{}}` on success, `{:error, reason}` on failure.
  """
  def compile(ast, data \\ %{}) when is_map(data) do
    data_decls = AST.data_decls(ast)
    param_decls = AST.param_decls(ast)
    model_stmts = AST.model_stmts(ast)

    data_vars = MapSet.new(data_decls, &AST.decl_name/1)
    param_vars = MapSet.new(param_decls, &AST.decl_name/1)

    # Build constraint map: param_name -> %{lower: ..., upper: ...}
    param_constraints = build_constraint_map(param_decls)

    # Validate data map has all declared data vars
    with :ok <- validate_data(data_vars, data) do
      ir = Builder.new_ir()

      ir =
        Enum.reduce(model_stmts, ir, fn stmt, acc ->
          compile_stmt(stmt, acc, data, data_vars, param_vars, param_constraints)
        end)

      {:ok, ir}
    end
  end

  # --- Statement compilation ---

  defp compile_stmt({:sample, target_expr, dist_name, arg_exprs}, ir, data, data_vars, param_vars, param_constraints) do
    {dist_mod, param_names} = DistMap.lookup!(dist_name)

    # Resolve distribution arguments to params map
    params = resolve_dist_args(param_names, arg_exprs, data, data_vars, param_vars)

    case target_expr do
      {:var, name} when is_binary(name) ->
        cond do
          MapSet.member?(param_vars, name) ->
            # Parameter: emit as free RV with optional transform from constraints
            opts = transform_opts(name, param_constraints)
            Builder.rv(ir, name, dist_mod, params, opts)

          MapSet.member?(data_vars, name) ->
            # Data variable: emit RV + observation
            rv_id = name <> "_rv"
            obs_id = name <> "_obs"
            value = Map.fetch!(data, name)

            ir
            |> Builder.rv(rv_id, dist_mod, params)
            |> Builder.obs(obs_id, rv_id, value)

          true ->
            raise ArgumentError, "undeclared variable in sampling statement: #{name}"
        end

      _ ->
        raise ArgumentError, "complex sampling targets not supported in Phase 1"
    end
  end

  defp compile_stmt({:target_incr, _expr}, _ir, _data, _data_vars, _param_vars, _param_constraints) do
    raise ArgumentError,
          "target += is not yet supported. Use sampling statements (x ~ dist(...)) instead. " <>
            "Custom log-probability increments will be added in a future version."
  end

  # --- Expression resolution ---

  defp resolve_dist_args(param_names, arg_exprs, data, data_vars, param_vars) do
    if length(param_names) != length(arg_exprs) do
      raise ArgumentError,
            "distribution expects #{length(param_names)} args (#{inspect(param_names)}), got #{length(arg_exprs)}"
    end

    param_names
    |> Enum.zip(arg_exprs)
    |> Map.new(fn {pname, expr} ->
      {pname, resolve_expr(expr, data, data_vars, param_vars)}
    end)
  end

  defp resolve_expr({:lit, n}, _data, _data_vars, _param_vars) when is_number(n) do
    Nx.tensor(n / 1, type: :f64)
  end

  defp resolve_expr({:var, name}, data, data_vars, param_vars) do
    cond do
      MapSet.member?(param_vars, name) ->
        # String reference — creates dependency edge in IR
        name

      MapSet.member?(data_vars, name) ->
        Map.fetch!(data, name)

      true ->
        raise ArgumentError, "undefined variable in expression: #{name}"
    end
  end

  defp resolve_expr({:neg, inner}, data, data_vars, param_vars) do
    val = resolve_expr(inner, data, data_vars, param_vars)

    case val do
      %Nx.Tensor{} = t -> Nx.negate(t)
      _ ->
        raise ArgumentError,
              "negation of parameter references is not yet supported (e.g., normal(-mu, sigma)). " <>
                "Define a separate parameter with the negated value."
    end
  end

  defp resolve_expr({:binop, _op, _l, _r}, _data, _data_vars, _param_vars) do
    raise ArgumentError,
          "arithmetic expressions in distribution arguments are not yet supported " <>
            "(e.g., normal(mu, sigma * 2)). Use an intermediate parameter instead."
  end

  defp resolve_expr({:call, _name, _args}, _data, _data_vars, _param_vars) do
    raise ArgumentError,
          "function calls in distribution arguments are not yet supported " <>
            "(e.g., normal(0, sqrt(tau))). Define a parameter with the transformed value."
  end

  # --- Constraints → transforms ---

  defp build_constraint_map(param_decls) do
    Map.new(param_decls, fn {:var_decl, name, type_spec, _} ->
      {name, AST.constraints(type_spec)}
    end)
  end

  defp transform_opts(name, param_constraints) do
    case Map.get(param_constraints, name) do
      %{lower: {:lit, lo}, upper: {:lit, hi}} when lo == 0 and hi == 1 -> [transform: :logit]
      %{lower: {:lit, lo}} when lo == 0 -> [transform: :log]
      _ -> []
    end
  end

  # --- Validation ---

  defp validate_data(data_vars, data) do
    missing =
      data_vars
      |> MapSet.to_list()
      |> Enum.reject(&Map.has_key?(data, &1))

    case missing do
      [] -> :ok
      vars -> {:error, {:missing_data, vars}}
    end
  end
end
