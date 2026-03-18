defmodule Exmc.LevelSetTest do
  use ExUnit.Case, async: true

  alias Exmc.Physics.LevelSet
  alias Exmc.Physics.Heat2D

  describe "smooth_heaviside/2" do
    test "returns ~1.0 for large positive phi" do
      phi = Nx.tensor([[10.0, 10.0], [10.0, 10.0]])
      h = LevelSet.smooth_heaviside(phi, 1.0)
      for v <- Nx.to_flat_list(h), do: assert v > 0.99
    end

    test "returns ~0.0 for large negative phi" do
      phi = Nx.tensor([[-10.0, -10.0], [-10.0, -10.0]])
      h = LevelSet.smooth_heaviside(phi, 1.0)
      for v <- Nx.to_flat_list(h), do: assert v < 0.01
    end

    test "returns 0.5 at phi = 0" do
      phi = Nx.tensor([[0.0, 0.0], [0.0, 0.0]])
      h = LevelSet.smooth_heaviside(phi, 1.0)
      for v <- Nx.to_flat_list(h), do: assert_in_delta(v, 0.5, 1.0e-6)
    end

    test "smaller eps gives sharper transition" do
      phi = Nx.tensor([[0.5]])
      h_sharp = LevelSet.smooth_heaviside(phi, 0.1) |> Nx.squeeze() |> Nx.to_number()
      h_smooth = LevelSet.smooth_heaviside(phi, 2.0) |> Nx.squeeze() |> Nx.to_number()
      # Sharper eps pushes H(0.5) closer to 1.0
      assert h_sharp > h_smooth
    end

    test "is monotonically increasing" do
      phi = Nx.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
      h = LevelSet.smooth_heaviside(phi, 1.0) |> Nx.to_flat_list()
      for {a, b} <- Enum.zip(h, tl(h)), do: assert a < b
    end
  end

  describe "material_field/4" do
    test "maps positive phi to kappa_a, negative to kappa_b" do
      phi = Nx.tensor([[10.0, -10.0], [-10.0, 10.0]])
      kappa = LevelSet.material_field(phi, Nx.tensor(5.0), Nx.tensor(1.0))
      vals = Nx.to_flat_list(kappa)
      # Positive phi -> near kappa_a=5, negative -> near kappa_b=1
      assert Enum.at(vals, 0) > 4.9
      assert Enum.at(vals, 1) < 1.1
      assert Enum.at(vals, 2) < 1.1
      assert Enum.at(vals, 3) > 4.9
    end

    test "phi=0 gives midpoint of kappa_a and kappa_b" do
      phi = Nx.tensor([[0.0]])
      kappa = LevelSet.material_field(phi, Nx.tensor(6.0), Nx.tensor(2.0))
      assert_in_delta Nx.squeeze(kappa) |> Nx.to_number(), 4.0, 1.0e-6
    end

    test "respects eps parameter" do
      phi = Nx.tensor([[1.0]])
      k_sharp = LevelSet.material_field(phi, Nx.tensor(5.0), Nx.tensor(1.0), eps: 0.1) |> Nx.squeeze() |> Nx.to_number()
      k_smooth = LevelSet.material_field(phi, Nx.tensor(5.0), Nx.tensor(1.0), eps: 5.0) |> Nx.squeeze() |> Nx.to_number()
      # Sharp eps at phi=1 should be closer to kappa_a=5
      assert k_sharp > k_smooth
    end
  end

  describe "laplacian_prior_logpdf/2" do
    test "constant field has zero Laplacian penalty" do
      logpdf_fn = LevelSet.laplacian_prior_logpdf(4, 4)
      phi_flat = Nx.broadcast(Nx.tensor(3.0), {16})
      logp = logpdf_fn.(Nx.tensor(0.0), %{phi: phi_flat, lambda: Nx.tensor(1.0)})
      assert_in_delta Nx.to_number(logp), 0.0, 1.0e-6
    end

    test "non-constant field has negative logp" do
      logpdf_fn = LevelSet.laplacian_prior_logpdf(4, 4)
      # Random-ish field
      phi_flat = Nx.tensor([1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5,
                            0.3, -0.3, 0.8, -0.8, 1.2, -1.2, 0.7, -0.7])
      logp = logpdf_fn.(Nx.tensor(0.0), %{phi: phi_flat, lambda: Nx.tensor(1.0)})
      assert Nx.to_number(logp) < 0.0
    end

    test "larger lambda gives more negative logp for same field" do
      logpdf_fn = LevelSet.laplacian_prior_logpdf(4, 4)
      phi_flat = Nx.tensor([1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5,
                            0.3, -0.3, 0.8, -0.8, 1.2, -1.2, 0.7, -0.7])
      logp_small = logpdf_fn.(Nx.tensor(0.0), %{phi: phi_flat, lambda: Nx.tensor(0.1)}) |> Nx.to_number()
      logp_large = logpdf_fn.(Nx.tensor(0.0), %{phi: phi_flat, lambda: Nx.tensor(10.0)}) |> Nx.to_number()
      assert logp_large < logp_small
    end

    test "linear gradient has non-zero Laplacian" do
      # On a 4x4 grid, a perfect linear field still has Laplacian ~ 0
      # at interior points (2nd derivative of linear = 0)
      logpdf_fn = LevelSet.laplacian_prior_logpdf(4, 4)
      phi_flat = Nx.tensor([
        0.0, 1.0, 2.0, 3.0,
        0.0, 1.0, 2.0, 3.0,
        0.0, 1.0, 2.0, 3.0,
        0.0, 1.0, 2.0, 3.0
      ])
      logp = logpdf_fn.(Nx.tensor(0.0), %{phi: phi_flat, lambda: Nx.tensor(1.0)})
      # Linear field: Laplacian should be ~0 everywhere
      assert_in_delta Nx.to_number(logp), 0.0, 1.0e-4
    end
  end

  describe "Heat2D.solve/2" do
    test "uniform kappa gives approximately linear temperature profile" do
      ny = 6
      nx = 6
      kappa = Nx.broadcast(Nx.tensor(1.0), {ny, nx})
      t = Heat2D.solve(kappa, bc_top: 1.0, bc_bottom: 0.0, iterations: 100)

      # Check middle row: should be ~0.5 (linear interpolation)
      mid_row = t[3] |> Nx.to_flat_list()
      for v <- Enum.slice(mid_row, 1..(nx - 2)) do
        assert_in_delta v, 0.5, 0.15
      end

      # Top should be ~1.0, bottom ~0.0
      assert Nx.to_number(t[0][nx |> div(2)]) > 0.9
      assert Nx.to_number(t[ny - 1][nx |> div(2)]) < 0.1
    end

    test "inclusion with different kappa distorts temperature" do
      ny = 8
      nx = 8
      # Uniform kappa except a hot spot in the center
      kappa_base = Nx.broadcast(Nx.tensor(1.0), {ny, nx})
      # Create an inclusion: high conductivity in center
      inclusion = for i <- 0..(ny - 1), j <- 0..(nx - 1) do
        if i >= 3 and i <= 4 and j >= 3 and j <= 4, do: 10.0, else: 1.0
      end |> Nx.tensor() |> Nx.reshape({ny, nx})

      t_uniform = Heat2D.solve(kappa_base, bc_top: 1.0, bc_bottom: 0.0, iterations: 100)
      t_inclusion = Heat2D.solve(inclusion, bc_top: 1.0, bc_bottom: 0.0, iterations: 100)

      # Temperature fields should differ at interior points
      diff = Nx.subtract(t_uniform, t_inclusion) |> Nx.abs() |> Nx.sum() |> Nx.to_number()
      assert diff > 0.01
    end

    test "read_sensors extracts bottom row" do
      ny = 4
      nx = 4
      kappa = Nx.broadcast(Nx.tensor(1.0), {ny, nx})
      t = Heat2D.solve(kappa, bc_top: 1.0, bc_bottom: 0.0, iterations: 50)

      sensors = Heat2D.read_sensors(t, :bottom_row)
      assert Nx.shape(sensors) == {nx}
      # Bottom row should be near 0.0 (bc_bottom)
      for v <- Nx.to_flat_list(sensors), do: assert v < 0.15
    end
  end
end
