defmodule Exmc.Stan do
  @moduledoc """
  Compile Stan model code to Exmc IR for MCMC sampling.

  Supports a subset of the Stan modeling language: `data`, `parameters`,
  and `model` blocks with sampling statements (`~`), type constraints
  (`<lower=0>`, `<lower=0, upper=1>`), and distribution calls.

  ## Supported Distributions

  #{Exmc.Stan.DistMap.supported() |> Enum.sort() |> Enum.map(&"  - `#{&1}`") |> Enum.join("\n")}

  ## Examples

      code = \"""
      data { real y; }
      parameters { real mu; }
      model {
        mu ~ normal(0, 10);
        y ~ normal(mu, 1);
      }
      \"""

      ir = Exmc.Stan.compile!(code, %{"y" => Nx.tensor(5.0)})
      {trace, stats} = Exmc.Stan.sample(code, %{"y" => Nx.tensor(5.0)}, seed: 42)

  ## Constraints

  Parameters with `<lower=0>` get a log transform; `<lower=0, upper=1>` gets
  a logit transform. This ensures the sampler explores unconstrained space.

  ## Limitations (Phase 1)

  - No arithmetic expressions in distribution arguments (e.g., `normal(mu, sigma * 2)`)
  - No function calls in distribution arguments (e.g., `normal(0, sqrt(tau))`)
  - No `target +=` custom log-probability increments
  - No `for` loops, `transformed parameters`, or `generated quantities` blocks
  """

  @doc """
  Compile Stan code to an Exmc IR.

  ## Parameters

    * `stan_code` - Stan model code as a string
    * `data` - map of data variable names to Nx tensors (default `%{}`)

  ## Returns

    `{:ok, %Exmc.IR{}}` on success, `{:error, reason}` on failure.
  """
  def compile(stan_code, data \\ %{}) when is_binary(stan_code) and is_map(data) do
    source_lines = String.split(stan_code, "\n")
    cleaned = strip_block_comments(stan_code)

    with {:ok, tokens} <- tokenize(cleaned),
         {:ok, ast} <- parse(tokens, source_lines),
         {:ok, ir} <- Exmc.Stan.Compiler.compile(ast, data) do
      {:ok, ir}
    end
  end

  @doc "Like `compile/2` but raises on error."
  def compile!(stan_code, data \\ %{}) do
    case compile(stan_code, data) do
      {:ok, ir} -> ir
      {:error, reason} -> raise "Stan compilation failed: #{format_error(reason)}"
    end
  end

  @doc """
  Compile and sample a Stan model in one step.

  Accepts all options that `Exmc.Sampler.sample/3` accepts, plus:
    * `:init_values` - initial parameter values (default `%{}`)
  """
  def sample(stan_code, data \\ %{}, opts \\ []) do
    ir = compile!(stan_code, data)
    init = Keyword.get(opts, :init_values, %{})
    Exmc.Sampler.sample(ir, init, opts)
  end

  # --- Internal ---

  @doc false
  def strip_block_comments(code) do
    Regex.replace(~r|/\*.*?\*/|s, code, fn match ->
      # Preserve line count for accurate error reporting
      newlines = match |> String.graphemes() |> Enum.count(&(&1 == "\n"))
      String.duplicate("\n", newlines)
    end)
  end

  defp tokenize(code) do
    case :exmc_stan_lexer.string(String.to_charlist(code)) do
      {:ok, tokens, _line} -> {:ok, tokens}
      {:error, {line, _mod, msg}, _} -> {:error, {:lexer, line, msg}}
    end
  end

  defp parse(tokens, source_lines) do
    case :exmc_stan_parser.parse(tokens) do
      {:ok, ast} -> {:ok, ast}
      {:error, {line, _mod, msg}} -> {:error, {:parser, line, msg, source_lines}}
    end
  end

  defp format_error({:lexer, line, msg}), do: "lexer error at line #{line}: #{inspect(msg)}"

  defp format_error({:parser, line, msg, source_lines}) do
    context = Enum.at(source_lines, line - 1, "")
    "parse error at line #{line}: #{inspect(msg)}\n    #{line} | #{context}"
  end

  defp format_error({:parser, line, msg}), do: "parse error at line #{line}: #{inspect(msg)}"
  defp format_error({:missing_data, vars}), do: "missing data variables: #{inspect(vars)}"
  defp format_error(other), do: inspect(other)
end
