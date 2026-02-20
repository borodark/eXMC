defmodule Exmc.Stan.ErrorTest do
  use ExUnit.Case, async: true

  describe "block comments" do
    test "strips block comments" do
      code = """
      data { /* this is a comment */ real y; }
      parameters { real mu; }
      model { mu ~ normal(0, 10); y ~ normal(mu, 1); }
      """

      assert {:ok, _ir} = Exmc.Stan.compile(code, %{"y" => Nx.tensor(5.0)})
    end

    test "strips multi-line block comments preserving line numbers" do
      code = """
      data {
        /* multi
           line
           comment */
        real y;
      }
      parameters { real mu; }
      model { mu ~ normal(0, 10); y ~ normal(mu, 1); }
      """

      assert {:ok, _ir} = Exmc.Stan.compile(code, %{"y" => Nx.tensor(5.0)})
    end

    test "block comment preserves line count for error reporting" do
      cleaned = Exmc.Stan.strip_block_comments("a\n/* b\nc\nd */\ne")
      assert cleaned == "a\n\n\n\ne"
    end
  end

  describe "lexer errors" do
    test "invalid character" do
      assert {:error, {:lexer, _line, _msg}} = Exmc.Stan.compile("data { @ }")
    end
  end

  describe "parser errors" do
    test "missing semicolon" do
      code = """
      data { real y }
      parameters { real mu; }
      model { mu ~ normal(0, 10); }
      """

      assert {:error, {:parser, _line, _msg, _lines}} = Exmc.Stan.compile(code, %{})
    end

    test "unclosed brace" do
      code = """
      data { real y;
      parameters { real mu; }
      model { mu ~ normal(0, 10); }
      """

      assert {:error, {:parser, _line, _msg, _lines}} = Exmc.Stan.compile(code, %{})
    end

    test "error message includes source line" do
      code = "data { real y }\nparameters { real mu; }\nmodel { mu ~ normal(0, 10); }"

      assert_raise RuntimeError, ~r/\d+ \|/, fn ->
        Exmc.Stan.compile!(code, %{})
      end
    end
  end

  describe "unknown distribution" do
    test "raises with list of supported distributions" do
      code = """
      parameters { real x; }
      model { x ~ foobar(1, 2); }
      """

      assert_raise ArgumentError, ~r/unknown distribution 'foobar'.*Supported/, fn ->
        Exmc.Stan.compile!(code)
      end
    end
  end

  describe "wrong argument count" do
    test "too few arguments" do
      code = """
      parameters { real x; }
      model { x ~ normal(0); }
      """

      assert_raise ArgumentError, ~r/expects 2 args/, fn ->
        Exmc.Stan.compile!(code)
      end
    end

    test "too many arguments" do
      code = """
      parameters { real x; }
      model { x ~ normal(0, 1, 2); }
      """

      assert_raise ArgumentError, ~r/expects 2 args/, fn ->
        Exmc.Stan.compile!(code)
      end
    end
  end

  describe "missing data" do
    test "missing data variable" do
      code = """
      data { real y; real z; }
      parameters { real mu; }
      model { mu ~ normal(0, 10); y ~ normal(mu, 1); z ~ normal(mu, 1); }
      """

      assert {:error, {:missing_data, missing}} = Exmc.Stan.compile(code, %{"y" => Nx.tensor(1.0)})
      assert "z" in missing
    end
  end

  describe "unsupported features" do
    test "target += gives clear error" do
      code = """
      parameters { real x; }
      model { target += x; }
      """

      # target += parses but compiler rejects
      assert_raise ArgumentError, ~r/target \+= is not yet supported/, fn ->
        Exmc.Stan.compile!(code)
      end
    end

    test "arithmetic in dist args gives clear error" do
      code = """
      parameters { real mu; real sigma; }
      model { mu ~ normal(0, sigma * 2); }
      """

      assert_raise ArgumentError, ~r/arithmetic expressions.*not yet supported/, fn ->
        Exmc.Stan.compile!(code)
      end
    end

    test "function calls in dist args gives clear error" do
      code = """
      parameters { real mu; real tau; }
      model { mu ~ normal(0, sqrt(tau)); }
      """

      assert_raise ArgumentError, ~r/function calls.*not yet supported/, fn ->
        Exmc.Stan.compile!(code)
      end
    end

    test "undeclared variable gives clear error" do
      code = """
      parameters { real x; }
      model { x ~ normal(y, 1); }
      """

      assert_raise ArgumentError, ~r/undefined variable/, fn ->
        Exmc.Stan.compile!(code)
      end
    end
  end
end
