defmodule Exmc.NUTS.Leapfrog do
  @moduledoc """
  Leapfrog integrator and momentum utilities for HMC/NUTS.

  All position/momentum/gradient tensors are flat f64 vectors in unconstrained space.
  Supports both diagonal (`{d}`) and dense (`{d,d}`) mass matrices.
  """

  @doc """
  One leapfrog step: half-momentum, full-position, evaluate, half-momentum.

  Returns `{q_new, p_new, logp_new, grad_new}`.
  """
  def step(vag_fn, q, p, grad, epsilon, inv_mass) do
    eps = Nx.tensor(epsilon, type: :f64, backend: Nx.BinaryBackend)
    half_eps = Nx.divide(eps, Nx.tensor(2.0, type: :f64, backend: Nx.BinaryBackend))

    # half step for momentum
    p_half = Nx.add(p, Nx.multiply(half_eps, grad))

    # full step for position: q + eps * M^{-1} @ p
    q_new = Nx.add(q, Nx.multiply(eps, mass_times_p(inv_mass, p_half)))

    # evaluate logp and gradient at new position
    {logp_new, grad_new} = vag_fn.(q_new)

    # half step for momentum
    p_new = Nx.add(p_half, Nx.multiply(half_eps, grad_new))

    {q_new, p_new, logp_new, grad_new}
  end

  @doc """
  Kinetic energy: `0.5 * p^T M^{-1} p`.

  Accepts diagonal `{d}` or dense `{d,d}` inverse mass matrix.
  Returns a scalar Nx tensor.
  """
  def kinetic_energy(p, inv_mass) do
    half = Nx.tensor(0.5, type: :f64, backend: Nx.BinaryBackend)
    Nx.multiply(half, Nx.sum(Nx.multiply(p, mass_times_p(inv_mass, p))))
  end

  @doc """
  Joint log probability (negative Hamiltonian): `logp - kinetic_energy(p, inv_mass)`.

  Higher is better. Returns a scalar Nx tensor.
  """
  def joint_logp(logp, p, inv_mass) do
    Nx.subtract(logp, kinetic_energy(p, inv_mass))
  end

  @doc """
  Compute `M^{-1} @ p` — element-wise for diagonal, matrix-vector for dense.
  """
  def mass_times_p(inv_mass, p) do
    case Nx.rank(inv_mass) do
      1 -> Nx.multiply(inv_mass, p)
      2 -> Nx.dot(inv_mass, p)
    end
  end

  @doc """
  Sample momentum from the mass matrix.

  `p = z / sqrt(inv_mass_diag)` where `z ~ N(0, I)`.

  Returns `{p, new_key}`.
  """
  def sample_momentum(key, inv_mass_diag) do
    shape = Nx.shape(inv_mass_diag)
    {z, key} = Nx.Random.normal(key, shape: shape, type: :f64)
    p = Nx.divide(z, Nx.sqrt(inv_mass_diag))
    {p, key}
  end
end
