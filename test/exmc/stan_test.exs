defmodule Exmc.StanTest do
  use ExUnit.Case

  @moduletag :stan

  import Exmc.TestHelper, only: [assert_posterior!: 3]

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
    {trace, _stats} = Exmc.Sampler.sample(ir, %{}, num_warmup: 300, num_samples: 4000, seed: 42)

    samples = trace["mu"] |> Nx.to_flat_list()

    # The closed-form conjugate posterior, not the rounded 4.95:
    #   precision = 1/100 + 1/1, var = 1/1.01, mean = 5/1.01
    #
    # Was `assert_in_delta mean, 4.95, 0.5` on 500 draws — half a posterior sd
    # of slack on the mean and no gate at all on the spread. This is a Stan
    # front-end test, but what it claims is that the sampler recovers the
    # posterior, so it should be able to see a wrong one.
    assert_posterior!(samples, {:normal, 5.0 / 1.01, :math.sqrt(1.0 / 1.01)}, resolution: 0.20)
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
