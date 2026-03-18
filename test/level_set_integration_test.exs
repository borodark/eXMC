defmodule Exmc.LevelSetIntegrationTest do
  use ExUnit.Case, async: false

  @moduletag :level_set_integration
  @moduletag timeout: 300_000

  alias Exmc.{Builder, NUTS.Sampler}
  alias Exmc.Dist
  alias Exmc.Physics.{LevelSet, Heat2D}

  test "recovers circular inclusion on 6x6 grid" do
    ny = 6
    nx = 6
    n = ny * nx

    # True level set: circle at center, radius 1.5
    true_phi =
      for i <- 0..(ny - 1), j <- 0..(nx - 1) do
        r = :math.sqrt(:math.pow(i - 2.5, 2) + :math.pow(j - 2.5, 2))
        1.5 - r
      end
      |> Nx.tensor()
      |> Nx.reshape({ny, nx})

    kappa_a = 5.0
    kappa_b = 1.0

    # Generate synthetic observations
    true_kappa = LevelSet.material_field(true_phi, Nx.tensor(kappa_a), Nx.tensor(kappa_b))
    true_temp = Heat2D.solve(true_kappa, bc_top: 1.0, bc_bottom: 0.0, iterations: 50)
    true_sensors = Heat2D.read_sensors(true_temp, :bottom_row)

    # Add observation noise
    sigma_obs = 0.02
    rng = :rand.seed_s(:exsss, 42)

    {noise_list, _rng} =
      Enum.map_reduce(1..nx, rng, fn _, rng ->
        {n, rng} = :rand.normal_s(rng)
        {sigma_obs * n, rng}
      end)

    sensor_data = Nx.add(true_sensors, Nx.tensor(noise_list))

    # Build model manually (turbine notebook pattern)
    ir = Builder.new_ir()

    # Phi: level set values with combined N(0,2) + Laplacian smoothness prior
    # Custom dist logpdf receives x = phi_flat {n}, returns scalar logp.
    # The `x` argument IS the free RV value that the sampler explores.
    laplacian_fn = LevelSet.laplacian_prior_logpdf(ny, nx)

    phi_prior_logpdf = fn x, params ->
      lambda = params.lambda
      sigma_prior = params.sigma_prior

      # Isotropic Normal(0, sigma_prior) prior: sum of element-wise logpdfs
      z2 = Nx.divide(Nx.pow(x, 2), Nx.multiply(2.0, Nx.pow(sigma_prior, 2)))
      normal_logp = Nx.negate(Nx.sum(z2))

      # Laplacian smoothness penalty
      laplacian_logp = laplacian_fn.(Nx.tensor(0.0), %{phi: x, lambda: lambda})

      Nx.add(normal_logp, laplacian_logp)
    end

    phi_dist = Dist.Custom.new(phi_prior_logpdf, support: :real)

    ir =
      Dist.Custom.rv(ir, "phi", phi_dist, %{
        lambda: Nx.tensor(0.5),
        sigma_prior: Nx.tensor(2.0)
      }, shape: {n})

    # Observation noise
    ir = Builder.rv(ir, "sigma_obs", Dist.HalfCauchy, %{scale: Nx.tensor(0.05)}, transform: :log)

    # Forward model likelihood
    forward_fn = fn kappa_field ->
      Heat2D.solve(kappa_field, bc_top: 1.0, bc_bottom: 0.0, iterations: 30)
      |> Heat2D.read_sensors(:bottom_row)
    end

    likelihood_logpdf = fn _x, params ->
      phi_flat = params.phi
      sigma = params.sigma_obs
      obs = params.data

      phi_2d = Nx.reshape(phi_flat, {ny, nx})
      kappa = LevelSet.material_field(phi_2d, kappa_a, kappa_b, eps: 1.0)
      predictions = forward_fn.(kappa)

      resid = Nx.subtract(obs, predictions)
      n_obs = Nx.tensor(Nx.size(obs) * 1.0)

      Nx.subtract(
        Nx.sum(Nx.negate(Nx.divide(Nx.pow(resid, 2), Nx.multiply(2.0, Nx.pow(sigma, 2))))),
        Nx.multiply(n_obs, Nx.log(sigma))
      )
    end

    ll_dist = Dist.Custom.new(likelihood_logpdf, support: :real)

    ir =
      Dist.Custom.rv(ir, "ll", ll_dist, %{
        phi: "phi",
        sigma_obs: "sigma_obs",
        data: sensor_data
      })

    ir = Builder.obs(ir, "ll_obs", "ll", Nx.tensor(0.0))

    # Init: flat phi at zero (prior mode), sigma_obs near true
    init = %{"phi" => Nx.broadcast(0.0, {n}), "sigma_obs" => 0.03}

    {trace, stats} =
      Sampler.sample(ir, init,
        num_warmup: 500,
        num_samples: 300,
        seed: 42,
        ncp: false
      )


    # Verify: posterior mean phi should have correct sign pattern
    # (positive at center, negative at corners)
    phi_samples = trace["phi"]
    mean_phi = Nx.mean(phi_samples, axes: [0]) |> Nx.reshape({ny, nx})

    # Center region (rows 2-3, cols 2-3) should be more positive than corners
    center_vals =
      for i <- 2..3, j <- 2..3 do
        Nx.to_number(mean_phi[i][j])
      end

    corner_vals =
      for {i, j} <- [{0, 0}, {0, nx - 1}, {ny - 1, 0}, {ny - 1, nx - 1}] do
        Nx.to_number(mean_phi[i][j])
      end

    center_mean = Enum.sum(center_vals) / length(center_vals)
    corner_mean = Enum.sum(corner_vals) / length(corner_vals)

    # The center should be more positive (inclusion detected)
    assert center_mean > corner_mean,
           "Center mean phi (#{center_mean}) should be > corner mean (#{corner_mean})"

    # Sanity checks
    assert stats.divergences < 150, "Too many divergences: #{stats.divergences}"
    assert length(Nx.to_flat_list(trace["phi"][0])) == n
  end
end
