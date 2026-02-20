defmodule Exmc.Stan.LexerTest do
  use ExUnit.Case, async: true

  defp lex!(code) do
    {:ok, tokens, _} = :exmc_stan_lexer.string(String.to_charlist(code))
    tokens
  end

  defp token_types(code) do
    lex!(code) |> Enum.map(&elem(&1, 0))
  end

  test "keywords" do
    assert token_types("data parameters model") == [:data, :parameters, :model]
  end

  test "real with constraints" do
    types = token_types("real<lower=0>")
    assert types == [:real, :langle, :lower, :eq, :int_lit, :rangle]
  end

  test "identifiers" do
    tokens = lex!("mu sigma y_obs")
    names = for {:ident, _, name} <- tokens, do: name
    assert names == ["mu", "sigma", "y_obs"]
  end

  test "integer and float literals" do
    tokens = lex!("42 3.14 1e5")
    values = Enum.map(tokens, &elem(&1, 2))
    assert values == [42, 3.14, 1.0e5]
  end

  test "negative integer" do
    types = token_types("-1")
    assert types == [:minus, :int_lit]
  end

  test "sampling statement tokens" do
    types = token_types("mu ~ normal(0, 10);")

    assert types == [
             :ident,
             :tilde,
             :ident,
             :lparen,
             :int_lit,
             :comma,
             :int_lit,
             :rparen,
             :semicolon
           ]
  end

  test "line comments are skipped" do
    tokens = lex!("real // this is a comment\nmu")
    types = Enum.map(tokens, &elem(&1, 0))
    assert types == [:real, :ident]
  end

  test "target += tokens" do
    types = token_types("target += x;")
    assert types == [:target, :plus_eq, :ident, :semicolon]
  end

  test "braces and blocks" do
    types = token_types("data { }")
    assert types == [:data, :lbrace, :rbrace]
  end

  test "arithmetic operators" do
    types = token_types("a + b * c - d / e")
    assert types == [:ident, :plus, :ident, :star, :ident, :minus, :ident, :slash, :ident]
  end
end
