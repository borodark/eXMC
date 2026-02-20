defmodule Exmc.Stan.AST do
  @moduledoc """
  AST node structs for parsed Stan programs.

  The parser (yecc) produces plain Erlang tuples. This module provides
  helpers to work with the AST from Elixir.

  ## AST node shapes (from parser)

      {program, data_decls, param_decls, model_stmts}
      {var_decl, name, type_spec, nil}
      {sample, target_expr, dist_name, arg_exprs}
      {target_incr, expr}

  ## Type specs

      {real, nil | constraints_map}
      {int_type, nil | constraints_map}
      {vector_type, size_expr, nil | constraints_map}

  ## Expressions

      {var, name}
      {lit, number}
      {neg, expr}
      {binop, op, left, right}
      {call, fname, args}
  """

  @doc "Extract data declarations from a program AST."
  def data_decls({:program, data, _params, _model}), do: data

  @doc "Extract parameter declarations from a program AST."
  def param_decls({:program, _data, params, _model}), do: params

  @doc "Extract model statements from a program AST."
  def model_stmts({:program, _data, _params, model}), do: model

  @doc "Extract the name from a var_decl."
  def decl_name({:var_decl, name, _type, _extra}), do: name

  @doc "Extract constraints from a type spec, or nil."
  def constraints({_type, constraints}), do: constraints
  def constraints({_type, _dim, constraints}), do: constraints
  def constraints({_type, _d1, _d2, constraints}), do: constraints
end
