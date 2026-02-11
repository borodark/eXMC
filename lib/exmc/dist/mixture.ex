defmodule Exmc.Dist.Mixture do
  @moduledoc """
  Finite mixture distribution.

  logpdf(x, %{components: [dist1, dist2], params: [params1, params2], weights: Nx.tensor([0.3, 0.7])})
  = log(sum_k w_k * exp(logpdf_k(x, params_k)))
  = logsumexp(log(w_k) + logpdf_k(x, params_k))
  """
  @behaviour Exmc.Dist

  @impl true
  def logpdf(x, %{components: components, params: component_params, weights: weights}) do
    log_weights = Nx.log(weights)
    k = length(components)

    log_probs =
      Enum.zip([components, component_params, 0..(k - 1)])
      |> Enum.map(fn {dist, params, i} ->
        lp = dist.logpdf(x, params)
        lw = Nx.reshape(log_weights[i], {})
        Nx.add(lp, lw)
      end)

    stacked = Nx.stack(log_probs)
    Nx.logsumexp(stacked)
  end

  @impl true
  def support(%{components: [first | _], params: [first_params | _]}) do
    first.support(first_params)
  end

  @impl true
  def transform(%{components: [first | _], params: [first_params | _]}) do
    first.transform(first_params)
  end

  @impl true
  def sample(%{components: components, params: component_params, weights: weights}, rng) do
    {u, rng} = :rand.uniform_s(rng)
    weight_list = Nx.to_flat_list(weights)
    k = choose_component(weight_list, u, 0)
    dist = Enum.at(components, k)
    params = Enum.at(component_params, k)
    dist.sample(params, rng)
  end

  defp choose_component([w | rest], u, k) do
    if u < w, do: k, else: choose_component(rest, u - w, k + 1)
  end

  defp choose_component([], _u, k), do: k - 1
end
