defmodule Exmc.Stan.ParserTest do
  use ExUnit.Case, async: true

  defp parse!(code) do
    {:ok, tokens, _} = :exmc_stan_lexer.string(String.to_charlist(code))
    {:ok, ast} = :exmc_stan_parser.parse(tokens)
    ast
  end

  test "empty model block" do
    ast = parse!("model { }")
    assert {:program, [], [], []} = ast
  end

  test "data and parameters blocks" do
    ast =
      parse!("""
      data { real y; }
      parameters { real mu; }
      model { }
      """)

    assert {:program, [data_decl], [param_decl], []} = ast
    assert {:var_decl, "y", {:real, nil}, nil} = data_decl
    assert {:var_decl, "mu", {:real, nil}, nil} = param_decl
  end

  test "constrained parameter" do
    ast =
      parse!("""
      parameters { real<lower=0> sigma; }
      model { }
      """)

    assert {:program, [], [{:var_decl, "sigma", {:real, constraints}, nil}], []} = ast
    assert %{lower: {:lit, 0}} = constraints
  end

  test "double-bounded parameter" do
    ast =
      parse!("""
      parameters { real<lower=0, upper=1> p; }
      model { }
      """)

    assert {:program, [], [{:var_decl, "p", {:real, constraints}, nil}], []} = ast
    assert %{lower: {:lit, 0}, upper: {:lit, 1}} = constraints
  end

  test "sampling statement" do
    ast =
      parse!("""
      model {
        mu ~ normal(0, 10);
      }
      """)

    assert {:program, [], [], [stmt]} = ast
    assert {:sample, {:var, "mu"}, "normal", [{:lit, 0}, {:lit, 10}]} = stmt
  end

  test "multiple sampling statements" do
    ast =
      parse!("""
      model {
        mu ~ normal(0, 10);
        sigma ~ exponential(1);
        y ~ normal(mu, sigma);
      }
      """)

    assert {:program, [], [], [s1, s2, s3]} = ast
    assert {:sample, {:var, "mu"}, "normal", _} = s1
    assert {:sample, {:var, "sigma"}, "exponential", [{:lit, 1}]} = s2
    assert {:sample, {:var, "y"}, "normal", [{:var, "mu"}, {:var, "sigma"}]} = s3
  end

  test "negative literal in args" do
    ast =
      parse!("""
      model {
        x ~ normal(-1, 2);
      }
      """)

    assert {:program, [], [], [{:sample, _, _, [{:neg, {:lit, 1}}, {:lit, 2}]}]} = ast
  end

  test "arithmetic expressions" do
    ast =
      parse!("""
      model {
        x ~ normal(0, 2 + 3);
      }
      """)

    assert {:program, [], [], [{:sample, _, _, [{:lit, 0}, {:binop, :+, {:lit, 2}, {:lit, 3}}]}]} =
             ast
  end

  test "float literals in args" do
    ast =
      parse!("""
      model {
        x ~ normal(0.0, 10.5);
      }
      """)

    assert {:program, [], [], [{:sample, _, _, [{:lit, 0.0}, {:lit, 10.5}]}]} = ast
  end

  test "full program" do
    ast =
      parse!("""
      data {
        real y;
      }
      parameters {
        real mu;
        real<lower=0> sigma;
      }
      model {
        mu ~ normal(0, 10);
        sigma ~ exponential(1);
        y ~ normal(mu, sigma);
      }
      """)

    assert {:program, [_], [_, _], [_, _, _]} = ast
  end

  test "parse error on invalid syntax" do
    {:ok, tokens, _} = :exmc_stan_lexer.string(~c"model { mu ~ ; }")
    assert {:error, _} = :exmc_stan_parser.parse(tokens)
  end
end
