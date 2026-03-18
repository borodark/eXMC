defmodule Exmc.Physics.LevelSet do
  @moduledoc """
  Level set utilities for Bayesian geometric inverse problems.

  A level set function phi(x,y) on a 2D grid classifies space into regions:
  - phi > 0 -> material A (e.g., rock, intact structure)
  - phi < 0 -> material B (e.g., void, crack, reservoir)

  Uses smooth Heaviside approximation for differentiability through EXLA JIT.

  ## Industrial Applications
  - Subsurface reservoir characterization (oil & gas)
  - Non-destructive testing (crack/void detection in turbine blades)
  - Electrical impedance tomography (geophysical imaging)
  - Medical imaging (tumor boundary detection)
  """

  alias Exmc.{Builder, Dist}

  @doc """
  Smooth Heaviside function: H_eps(phi) = 0.5 * (1 + tanh(phi / eps))

  Maps level set values to [0, 1] with a differentiable transition.
  Larger eps gives a smoother (more blurred) boundary.

  ## Parameters
  - `phi`: tensor of level set values (any shape)
  - `eps`: smoothing width (default 1.0). For unit grid spacing, 0.5-1.0 is typical.
  """
  def smooth_heaviside(phi, eps \\ 1.0) do
    Nx.multiply(0.5, Nx.add(1.0, Nx.tanh(Nx.divide(phi, eps))))
  end

  @doc """
  Map level set to material properties via smooth Heaviside interpolation.

  kappa(x,y) = kappa_a * H(phi) + kappa_b * (1 - H(phi))

  Where phi > 0 corresponds to material A (kappa_a) and phi < 0 to material B (kappa_b).

  ## Parameters
  - `phi`: tensor of level set values
  - `kappa_a`: material A property (scalar or tensor)
  - `kappa_b`: material B property (scalar or tensor)
  - `opts`: keyword list with `:eps` (default 1.0)
  """
  def material_field(phi, kappa_a, kappa_b, opts \\ []) do
    eps = Keyword.get(opts, :eps, 1.0)
    h = smooth_heaviside(phi, eps)
    Nx.add(Nx.multiply(kappa_a, h), Nx.multiply(kappa_b, Nx.subtract(1.0, h)))
  end

  @doc """
  Build a Laplacian smoothness prior logpdf closure.

  Returns a function `fn(_x, params) -> scalar logp` for use with `Exmc.Dist.Custom.new/2`.

  The prior penalizes roughness via the discrete Laplacian:
    logp(phi) = -lambda/2 * sum((nabla^2 phi)^2)

  where nabla^2 phi at interior point (i,j) is the 5-point stencil:
    phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1] - 4*phi[i,j]

  This is the GMRF (Gaussian Markov Random Field) approximation of a Matern
  kernel (Rue & Held 2005), with O(n) evaluation cost.

  ## Parameters
  - `ny`: grid rows
  - `nx`: grid cols

  The returned closure expects `params.lambda` (prior strength) and
  `params.phi` (level set values as flat vector of length ny*nx, resolved
  from a string param ref).
  """
  def laplacian_prior_logpdf(ny, nx) do
    fn _x, params ->
      lambda = params.lambda
      phi_flat = params.phi
      phi = Nx.reshape(phi_flat, {ny, nx})

      # 5-point Laplacian on interior (ny-2) x (nx-2)
      center = Nx.slice(phi, [1, 1], [ny - 2, nx - 2])
      up = Nx.slice(phi, [0, 1], [ny - 2, nx - 2])
      down = Nx.slice(phi, [2, 1], [ny - 2, nx - 2])
      left = Nx.slice(phi, [1, 0], [ny - 2, nx - 2])
      right = Nx.slice(phi, [1, 2], [ny - 2, nx - 2])

      laplacian = Nx.subtract(
        Nx.add(Nx.add(up, down), Nx.add(left, right)),
        Nx.multiply(4.0, center)
      )

      Nx.multiply(-0.5, Nx.multiply(lambda, Nx.sum(Nx.pow(laplacian, 2))))
    end
  end

  @doc """
  Build a complete level set inverse problem model IR.

  Creates an IR with:
  - `"phi"` — free RV with Laplacian smoothness prior (shape {ny*nx})
  - `"lambda"` — prior strength (free, with Exponential prior)
  - `"sigma_obs"` — observation noise (free, HalfCauchy prior)
  - `"ll"` — forward model likelihood (Custom dist)
  - `"ll_obs"` — observation node

  ## Parameters
  - `forward_fn`: `fn(kappa_field :: {ny,nx}) -> predictions :: {n_obs}` (pure Nx)
  - `data`: `{n_obs}` tensor of observed values
  - `opts`:
    - `ny`: grid rows (default 8)
    - `nx`: grid cols (default 8)
    - `kappa_a`: material A property (default 5.0)
    - `kappa_b`: material B property (default 1.0)
    - `sigma_obs_scale`: HalfCauchy scale for obs noise (default 0.1)
    - `lambda_rate`: Exponential rate for prior strength (default 0.1)
    - `eps`: Heaviside smoothing (default 1.0)
  """
  def build_model(forward_fn, data, opts \\ []) do
    ny = Keyword.get(opts, :ny, 8)
    nx = Keyword.get(opts, :nx, 8)
    kappa_a = Keyword.get(opts, :kappa_a, 5.0)
    kappa_b = Keyword.get(opts, :kappa_b, 1.0)
    sigma_obs_scale = Keyword.get(opts, :sigma_obs_scale, 0.1)
    lambda_rate = Keyword.get(opts, :lambda_rate, 0.1)
    eps = Keyword.get(opts, :eps, 1.0)
    n = ny * nx

    ir = Builder.new_ir()

    # Lambda: Laplacian smoothness strength (free, Exponential prior)
    ir = Builder.rv(ir, "lambda", Dist.Exponential, %{rate: lambda_rate}, transform: :log)

    # Phi: level set values (unconstrained, flat vector)
    # Combined prior: isotropic Normal(0,1) + Laplacian smoothness penalty.
    # Custom dist logpdf receives x = phi_flat {n}, returns scalar logp.
    laplacian_fn = laplacian_prior_logpdf(ny, nx)

    phi_prior_logpdf = fn x, params ->
      lambda = params.lambda

      # Isotropic Normal(0, 1): sum of element-wise -0.5 * x^2
      normal_logp = Nx.negate(Nx.sum(Nx.multiply(0.5, Nx.pow(x, 2))))

      # Laplacian smoothness penalty
      laplacian_logp = laplacian_fn.(Nx.tensor(0.0), %{phi: x, lambda: lambda})

      Nx.add(normal_logp, laplacian_logp)
    end

    phi_dist = Dist.Custom.new(phi_prior_logpdf, support: :real)

    ir = Dist.Custom.rv(ir, "phi", phi_dist, %{
      lambda: "lambda"
    }, shape: {n})

    # Observation noise
    ir = Builder.rv(ir, "sigma_obs", Dist.HalfCauchy, %{scale: sigma_obs_scale}, transform: :log)

    # Forward model likelihood
    likelihood_logpdf = fn _x, params ->
      phi_flat = params.phi
      sigma = params.sigma_obs
      obs = params.data

      phi_2d = Nx.reshape(phi_flat, {ny, nx})
      kappa = material_field(phi_2d, kappa_a, kappa_b, eps: eps)
      predictions = forward_fn.(kappa)

      resid = Nx.subtract(obs, predictions)
      n_obs = Nx.size(obs)
      Nx.subtract(
        Nx.sum(Nx.negate(Nx.divide(Nx.pow(resid, 2), Nx.multiply(2.0, Nx.pow(sigma, 2))))),
        Nx.multiply(n_obs, Nx.log(sigma))
      )
    end

    ll_dist = Dist.Custom.new(likelihood_logpdf, support: :real)
    ir = Dist.Custom.rv(ir, "ll", ll_dist, %{
      phi: "phi",
      sigma_obs: "sigma_obs",
      data: data
    })
    ir = Builder.obs(ir, "ll_obs", "ll", Nx.tensor(0.0))

    ir
  end
end
