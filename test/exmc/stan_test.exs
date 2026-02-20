defmodule Exmc.StanTest do
  use ExUnit.Case

  @moduletag :stan

  test "end-to-end: normal conjugate posterior" do
    code = """
    data { real y; }
    parameters { real mu; }
    model {
      mu ~ normal(0, 10);
      y ~ normal(mu, 1);
    }
    """

    ir = Exmc.Stan.compile!(code, %{"y" => Nx.tensor(5.0)})
    {trace, _stats} = Exmc.Sampler.sample(ir, %{}, num_warmup: 300, num_samples: 500, seed: 42)

    mu_samples = trace["mu"]
    mean = Nx.mean(mu_samples) |> Nx.to_number()

    # Posterior mean should be near 5.0 (strong data, weak prior)
    # With prior N(0,10) and obs y=5 with sigma=1:
    # posterior mean = (0/100 + 5/1) / (1/100 + 1/1) ≈ 4.95
    assert_in_delta mean, 4.95, 0.5
  end

  test "end-to-end: constrained parameter stays positive" do
    code = """
    parameters { real<lower=0> sigma; }
    model {
      sigma ~ exponential(1);
    }
    """

    ir = Exmc.Stan.compile!(code)
    {trace, _stats} = Exmc.Sampler.sample(ir, %{}, num_warmup: 200, num_samples: 300, seed: 42)

    sigma_samples = trace["sigma"] |> Nx.to_flat_list()
    assert Enum.all?(sigma_samples, fn v -> is_number(v) and v > 0.0 end)

    mean = Enum.sum(sigma_samples) / length(sigma_samples)
    # Exponential(1) has mean 1.0
    assert_in_delta mean, 1.0, 0.5
  end

  test "end-to-end: two-parameter model" do
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

    ir = Exmc.Stan.compile!(code, %{"y" => Nx.tensor(3.0)})

    {trace, _stats} =
      Exmc.Sampler.sample(ir, %{}, num_warmup: 500, num_samples: 500, seed: 42)

    mu_mean = trace["mu"] |> Nx.mean() |> Nx.to_number()
    # With observation y=3, mu should be pulled toward 3
    assert_in_delta mu_mean, 3.0, 1.5

    sigma_samples = trace["sigma"] |> Nx.to_flat_list()
    assert Enum.all?(sigma_samples, fn v -> is_number(v) and v > 0.0 end)
  end
end
