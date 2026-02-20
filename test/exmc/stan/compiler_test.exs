defmodule Exmc.Stan.CompilerTest do
  use ExUnit.Case, async: true

  alias Exmc.Stan

  test "compiles simple prior-only model" do
    code = """
    parameters { real mu; }
    model {
      mu ~ normal(0, 10);
    }
    """

    {:ok, ir} = Stan.compile(code)
    assert map_size(ir.nodes) == 1
    assert Map.has_key?(ir.nodes, "mu")

    node = ir.nodes["mu"]
    assert {:rv, Exmc.Dist.Normal, %{mu: _, sigma: _}} = node.op
  end

  test "compiles data + observation" do
    code = """
    data { real y; }
    parameters { real mu; }
    model {
      mu ~ normal(0, 10);
      y ~ normal(mu, 1);
    }
    """

    {:ok, ir} = Stan.compile(code, %{"y" => Nx.tensor(5.0)})
    assert map_size(ir.nodes) == 3
    assert Map.has_key?(ir.nodes, "mu")
    assert Map.has_key?(ir.nodes, "y_rv")
    assert Map.has_key?(ir.nodes, "y_obs")

    # mu is free RV
    assert {:rv, Exmc.Dist.Normal, %{mu: _, sigma: _}} = ir.nodes["mu"].op

    # y_rv has mu as string dep
    {:rv, Exmc.Dist.Normal, params} = ir.nodes["y_rv"].op
    assert params.mu == "mu"
    assert %Nx.Tensor{} = params.sigma

    # y_obs is observation of y_rv
    assert {:obs, "y_rv", _, _meta} = ir.nodes["y_obs"].op
  end

  test "constrained parameter gets log transform" do
    code = """
    parameters { real<lower=0> sigma; }
    model {
      sigma ~ exponential(1);
    }
    """

    {:ok, ir} = Stan.compile(code)
    node = ir.nodes["sigma"]
    # Explicit transform from constraint
    assert {:rv, Exmc.Dist.Exponential, %{lambda: _}, :log} = node.op
  end

  test "unit-bounded parameter gets logit transform" do
    code = """
    parameters { real<lower=0, upper=1> p; }
    model {
      p ~ beta(2, 5);
    }
    """

    {:ok, ir} = Stan.compile(code)
    node = ir.nodes["p"]
    assert {:rv, Exmc.Dist.Beta, %{alpha: _, beta: _}, :logit} = node.op
  end

  test "missing data variable returns error" do
    code = """
    data { real y; }
    model { }
    """

    assert {:error, {:missing_data, ["y"]}} = Stan.compile(code, %{})
  end

  test "parameter references create string deps" do
    code = """
    parameters {
      real mu;
      real<lower=0> sigma;
    }
    model {
      mu ~ normal(0, 10);
      sigma ~ exponential(1);
    }
    """

    {:ok, ir} = Stan.compile(code)
    assert map_size(ir.nodes) == 2

    mu_node = ir.nodes["mu"]
    assert mu_node.deps == []

    sigma_node = ir.nodes["sigma"]
    assert sigma_node.deps == []
  end

  test "hierarchical model with param dependencies" do
    code = """
    data { real y; }
    parameters {
      real mu;
      real<lower=0> sigma;
    }
    model {
      mu ~ normal(0, 10);
      sigma ~ exponential(1);
      y ~ normal(mu, sigma);
    }
    """

    {:ok, ir} = Stan.compile(code, %{"y" => Nx.tensor(3.0)})
    assert map_size(ir.nodes) == 4

    y_rv = ir.nodes["y_rv"]
    {:rv, Exmc.Dist.Normal, params} = y_rv.op
    assert params.mu == "mu"
    assert params.sigma == "sigma"
    assert "mu" in y_rv.deps
    assert "sigma" in y_rv.deps
  end

  test "compile! raises on error" do
    assert_raise RuntimeError, ~r/missing data/, fn ->
      Stan.compile!("data { real y; } model { }", %{})
    end
  end

  test "negative literal in distribution args" do
    code = """
    parameters { real mu; }
    model {
      mu ~ normal(-5, 10);
    }
    """

    {:ok, ir} = Stan.compile(code)
    {:rv, Exmc.Dist.Normal, params} = ir.nodes["mu"].op
    assert Nx.to_number(params.mu) == -5.0
  end

  test "multiple data variables" do
    code = """
    data {
      real x;
      real y;
    }
    parameters { real mu; }
    model {
      mu ~ normal(0, 10);
      x ~ normal(mu, 1);
      y ~ normal(mu, 1);
    }
    """

    {:ok, ir} =
      Stan.compile(code, %{
        "x" => Nx.tensor(3.0),
        "y" => Nx.tensor(4.0)
      })

    assert map_size(ir.nodes) == 5
    assert Map.has_key?(ir.nodes, "x_rv")
    assert Map.has_key?(ir.nodes, "x_obs")
    assert Map.has_key?(ir.nodes, "y_rv")
    assert Map.has_key?(ir.nodes, "y_obs")
  end
end
