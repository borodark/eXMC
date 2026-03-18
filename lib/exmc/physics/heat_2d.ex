defmodule Exmc.Physics.Heat2D do
  @moduledoc """
  2D steady-state heat equation solver via Jacobi iteration.

  Solves: -div(kappa * grad(T)) = 0 on a rectangular grid
  with Dirichlet boundary conditions.

  All operations are pure Nx for EXLA JIT compatibility and autodiff.
  The Jacobi iteration unrolls at trace time via Enum.reduce,
  producing a fixed-length XLA computation graph.
  """

  @doc """
  Solve the 2D steady-state heat equation.

  ## Parameters
  - `kappa`: `{ny, nx}` tensor of thermal conductivity values
  - `opts`:
    - `iterations`: number of Jacobi iterations (default 50)
    - `bc_top`: temperature at top boundary (default 1.0)
    - `bc_bottom`: temperature at bottom boundary (default 0.0)
    - `bc_left`: temperature at left boundary (default: linear interpolation)
    - `bc_right`: temperature at right boundary (default: linear interpolation)

  ## Returns
  `{ny, nx}` temperature field tensor.
  """
  def solve(kappa, opts \\ []) do
    iterations = Keyword.get(opts, :iterations, 50)
    bc_top = Keyword.get(opts, :bc_top, 1.0)
    bc_bottom = Keyword.get(opts, :bc_bottom, 0.0)

    {ny, nx} = Nx.shape(kappa)

    # Initialize temperature with linear interpolation between top and bottom BCs
    t_init = init_temperature(ny, nx, bc_top, bc_bottom, opts)

    # Build interior mask: 1 for interior, 0 for boundary
    mask = interior_mask(ny, nx)

    # Jacobi iteration — unrolls at Nx trace time
    Enum.reduce(1..iterations, t_init, fn _i, t ->
      jacobi_step(t, kappa, mask, t_init)
    end)
  end

  @doc """
  Extract sensor readings from the bottom row of a temperature field.

  Returns a 1D tensor of temperatures at the bottom interior points.
  """
  def read_sensors(temperature, :bottom_row) do
    {ny, _nx} = Nx.shape(temperature)
    Nx.slice(temperature, [ny - 1, 0], [1, elem(Nx.shape(temperature), 1)])
    |> Nx.reshape({elem(Nx.shape(temperature), 1)})
  end

  def read_sensors(temperature, positions) when is_list(positions) do
    values = Enum.map(positions, fn {row, col} ->
      temperature[row][col] |> Nx.to_number()
    end)
    Nx.tensor(values)
  end

  # --- Private helpers ---

  defp init_temperature(ny, nx, bc_top, bc_bottom, opts) do
    bc_left = Keyword.get(opts, :bc_left, nil)
    bc_right = Keyword.get(opts, :bc_right, nil)

    # Linear interpolation from top to bottom
    rows = Nx.iota({ny, 1})
    frac = Nx.divide(rows, ny - 1)
    # frac=0 at row 0 (top), frac=1 at row ny-1 (bottom)
    t = Nx.add(Nx.multiply(bc_top, Nx.subtract(1.0, frac)), Nx.multiply(bc_bottom, frac))
    t = Nx.broadcast(t, {ny, nx})

    # Override left/right boundaries if specified
    t = if bc_left do
      col_vals = Nx.add(
        Nx.multiply(bc_top, Nx.subtract(1.0, frac)),
        Nx.multiply(bc_bottom, frac)
      ) |> Nx.reshape({ny})
      # Set left column — use put_slice-free approach
      left_col = Nx.reshape(col_vals, {ny, 1})
      left_mask = Nx.concatenate([Nx.broadcast(1.0, {ny, 1}), Nx.broadcast(0.0, {ny, nx - 1})], axis: 1)
      Nx.add(Nx.multiply(t, Nx.subtract(1.0, left_mask)), Nx.multiply(Nx.broadcast(left_col, {ny, nx}), left_mask))
    else
      t
    end

    t = if bc_right do
      col_vals = Nx.add(
        Nx.multiply(bc_top, Nx.subtract(1.0, frac)),
        Nx.multiply(bc_bottom, frac)
      ) |> Nx.reshape({ny})
      right_col = Nx.reshape(col_vals, {ny, 1})
      right_mask = Nx.concatenate([Nx.broadcast(0.0, {ny, nx - 1}), Nx.broadcast(1.0, {ny, 1})], axis: 1)
      Nx.add(Nx.multiply(t, Nx.subtract(1.0, right_mask)), Nx.multiply(Nx.broadcast(right_col, {ny, nx}), right_mask))
    else
      t
    end

    t
  end

  defp interior_mask(ny, nx) do
    # 1 for interior points, 0 for boundary
    row_mask = Nx.concatenate([
      Nx.broadcast(0.0, {1, nx}),
      Nx.broadcast(1.0, {ny - 2, nx}),
      Nx.broadcast(0.0, {1, nx})
    ])

    col_mask = Nx.concatenate([
      Nx.broadcast(0.0, {ny, 1}),
      Nx.broadcast(1.0, {ny, nx - 2}),
      Nx.broadcast(0.0, {ny, 1})
    ], axis: 1)

    Nx.multiply(row_mask, col_mask)
  end

  defp jacobi_step(t, kappa, mask, t_bc) do
    {ny, nx} = Nx.shape(t)

    # Shifted versions for 5-point stencil via concatenation with zero padding
    zeros_row = Nx.broadcast(Nx.tensor(0.0), {1, nx})
    zeros_col = Nx.broadcast(Nx.tensor(0.0), {ny, 1})

    # T shifted: up means T[i-1,j] at position (i,j)
    up = Nx.concatenate([zeros_row, Nx.slice(t, [0, 0], [ny - 1, nx])])
    down = Nx.concatenate([Nx.slice(t, [1, 0], [ny - 1, nx]), zeros_row])
    left = Nx.concatenate([zeros_col, Nx.slice(t, [0, 0], [ny, nx - 1])], axis: 1)
    right = Nx.concatenate([Nx.slice(t, [0, 1], [ny, nx - 1]), zeros_col], axis: 1)

    # Kappa at neighbor positions
    k_up = Nx.concatenate([zeros_row, Nx.slice(kappa, [0, 0], [ny - 1, nx])])
    k_down = Nx.concatenate([Nx.slice(kappa, [1, 0], [ny - 1, nx]), zeros_row])
    k_left = Nx.concatenate([zeros_col, Nx.slice(kappa, [0, 0], [ny, nx - 1])], axis: 1)
    k_right = Nx.concatenate([Nx.slice(kappa, [0, 1], [ny, nx - 1]), zeros_col], axis: 1)

    # Weighted average: T_new = sum(k_neighbor * T_neighbor) / sum(k_neighbor)
    numerator = Nx.add(
      Nx.add(Nx.multiply(k_up, up), Nx.multiply(k_down, down)),
      Nx.add(Nx.multiply(k_left, left), Nx.multiply(k_right, right))
    )

    denominator = Nx.add(Nx.add(k_up, k_down), Nx.add(k_left, k_right))
    safe_denom = Nx.max(denominator, 1.0e-10)

    t_update = Nx.divide(numerator, safe_denom)

    # Apply mask: interior uses update, boundary keeps original BC values
    Nx.add(Nx.multiply(mask, t_update), Nx.multiply(Nx.subtract(1.0, mask), t_bc))
  end
end
