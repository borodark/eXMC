defmodule Exmc.NUTS.LeapfrogLeafDiffTest do
  @moduledoc """
  Element-wise differential: the synthesised chain shader against the host's own
  leapfrog, along a whole trajectory.

  Promoted from `bench/leapfrog_leaf_diff.exs`, which computed
  `{ok_q, ok_p, ok_g, ok_lp}` and returned them into nothing. It was the only
  harness in this repository that dispatches a shader and compares `q`, `p`,
  `grad` and `logp` element-wise — everything else numerical here is either
  differential between two backends, which a defect in the shared NUTS tree
  moves identically in both arms, or end-to-end statistical, which sees only
  what survives hundreds of draws.

  Its own header records the cost of that: it destructured the meta 6-tuple for
  a day after captures moved to the extras SSBO, so the one instrument that
  could have caught the zero-likelihood reduce-bound defect was itself
  un-runnable, on the commit that made that defect reachable. Nothing said so,
  because `mix test` never ran it.

  ## Tolerances, and why they are not the bench's

  The bench flagged at 1e-6 relative. That was NOT slack when written — the
  emitted GLSL then carried its distribution constants at f32 (`log(2*pi)` as
  1.8378770351409912) and packed the observation buffer from f32 tensors, so
  ~1e-7 was the floor the path could reach. The f64 migration removed that
  floor and nobody revisited the number.

  MEASURED on all four hosts at `994305de4`, worst relative Δ over the three
  parameter sets:

      host       q         p         grad      logp      offset spread
      super-io   1.92e-15  2.22e-15  3.07e-15  9.20e-16  1.78e-14
      mac-247    4.77e-15  4.22e-15  6.72e-15  1.53e-15  2.84e-14
      mac-248    4.77e-15  4.22e-15  6.72e-15  1.53e-15  2.84e-14
      jetson     4.77e-15  4.22e-15  6.72e-15  1.53e-15  2.84e-14

  So 1e-13 is ~15x the fleet worst and 1e-12 ~35x the worst offset spread.
  Derived from four machines on purpose: a bound taken from one describes that
  machine's agreement with its own CPU, and three of those hosts agree with each
  other to the last digit while the fourth does not.

  **These bounds are FIXTURE-calibrated, and the headroom is not generous.**
  Setting all three sigmas to 1.0 — still a perfectly legitimate model — takes
  the adapted-eps arm to `grad` 1.212e-13 on super-io alone, over the bound.
  That is trajectory geometry, not a defect: a tighter posterior at eps=1.139
  simply travels further per step and accumulates more rounding.

  So before ADDING a fixture here, measure it. A new model that overshoots by a
  small factor is telling you about its geometry; only a gross overshoot is
  telling you about the shader. The bounds were not loosened to accommodate the
  hypothetical, because a bound widened for a model that is not in the file buys
  nothing and costs sharpness for the models that are.

  ## The distinct sigmas are load-bearing

  The fixture uses 1.0 / 2.0 / 3.0, not three identical values. Marker *i* is
  attributed to observed node *i* positionally, and the gradient's markers
  arrive from `Nx.Defn.grad`, which owes nobody forward order. With identical
  sigmas a permuted attribution is bit-for-bit the correct answer and this test
  passes while the code is wrong. Do not "simplify" them.
  """
  use ExUnit.Case, async: false

  @moduletag :requires_vulkan

  alias Exmc.{Builder, Compiler, Dist, IR}
  alias Exmc.NUTS.{ChainShaderCodegen, Leapfrog}
  alias Exmc.NUTS.Vulkan.Dispatch

  @f64 [type: :f64, backend: Nx.BinaryBackend]

  # ~15x and ~35x the fleet worst; see the moduledoc for the measurements.
  @tol 1.0e-13
  @offset_tol 1.0e-12

  @k 32

  defp three_observed_ir do
    IR.new()
    |> Builder.rv("mu", Dist.Normal, %{mu: Nx.tensor(0.0, @f64), sigma: Nx.tensor(10.0, @f64)})
    |> Builder.rv("x1", Dist.Normal, %{mu: "mu", sigma: Nx.tensor(1.0, @f64)})
    |> Builder.obs("x1_obs", "x1", Nx.tensor(4.0, @f64))
    |> Builder.rv("x2", Dist.Normal, %{mu: "mu", sigma: Nx.tensor(2.0, @f64)})
    |> Builder.obs("x2_obs", "x2", Nx.tensor(3.8, @f64))
    |> Builder.rv("x3", Dist.Normal, %{mu: "mu", sigma: Nx.tensor(3.0, @f64)})
    |> Builder.obs("x3_obs", "x3", Nx.tensor(4.2, @f64))
  end

  # beta as ONE vector RV with Nx.dot over a captured design matrix — the path
  # that only began synthesising at 371785ff5 and has the least element-wise
  # coverage.
  defp vector_rv_ir do
    xs = Enum.map(1..12, &(&1 / 4))
    xmat = Nx.stack([Nx.broadcast(Nx.tensor(1.0, @f64), {12}), Nx.tensor(xs, @f64)], axis: 1)
    y = Nx.tensor(Enum.map(xs, fn v -> 1.0 + 2.0 * v end), @f64)

    lik =
      Dist.Custom.new(fn _x, p ->
        r = Nx.subtract(y, Nx.dot(xmat, p.beta))
        Nx.sum(Nx.divide(Nx.multiply(r, r), -0.5))
      end)

    IR.new()
    |> Builder.rv("beta", Dist.Normal, %{mu: Nx.tensor(0.0, @f64), sigma: Nx.tensor(5.0, @f64)},
      shape: {2}
    )
    |> Dist.Custom.rv("Y", lik, %{beta: "beta"})
    |> Builder.obs("Y_obs", "Y", y)
  end

  # A Custom that READS its observations rather than capturing them — the path
  # the obs-buffer work opened at d299f4fc4, and the one that was a frozen chain
  # before it.
  defp custom_reads_obs_ir do
    xs = Enum.map(1..12, &(&1 / 4))
    xcol = Nx.tensor(xs, @f64)
    y = Nx.tensor(Enum.map(xs, fn v -> 1.0 + 2.0 * v end), @f64)

    lik =
      Dist.Custom.new(fn observed, p ->
        r = Nx.subtract(observed, Nx.multiply(xcol, p.slope))
        Nx.sum(Nx.divide(Nx.multiply(r, r), -0.5))
      end)

    IR.new()
    |> Builder.rv("slope", Dist.Normal, %{mu: Nx.tensor(0.0, @f64), sigma: Nx.tensor(5.0, @f64)})
    |> Dist.Custom.rv("Y", lik, %{slope: "slope"})
    |> Builder.obs("Y_obs", "Y", y)
  end

  defp host_chain(vag_fn, q0, p0, eps, inv_mass) do
    {lp0, g0} = vag_fn.(q0)

    {rows, _} =
      Enum.map_reduce(1..@k, {q0, p0, g0, lp0}, fn _i, {q, p, g, _lp} ->
        {qn, pn, lpn, gn} = Leapfrog.step(vag_fn, q, p, g, eps, inv_mass)
        {{qn, pn, gn, lpn}, {qn, pn, gn, lpn}}
      end)

    rows
  end

  defp worst_rel(gpu_t, host_vals) do
    gpu = Nx.to_flat_list(gpu_t)
    host = host_vals |> Enum.flat_map(&Nx.to_flat_list/1)

    Enum.zip(gpu, host)
    |> Enum.map(fn {a, b} -> abs(a - b) / max(1.0, abs(b)) end)
    |> Enum.max()
  end

  defp assert_agrees(ir, d, eps, q0v, p0v, label) do
    {:ok, meta} = ChainShaderCodegen.detect_meta(ir, [])
    {vag_fn, _pm} = Compiler.value_and_grad(ir)

    q0 = Nx.tensor(q0v, @f64)
    p0 = Nx.tensor(p0v, @f64)
    inv_mass = Nx.tensor(List.duplicate(1.0, d), @f64)

    {q_c, p_c, lp_c, g_c} = Dispatch.chain(meta, d, eps, inv_mass, q0, p0, @k, 1)
    host = host_chain(vag_fn, q0, p0, eps, inv_mass)

    for {name, gpu, pick} <- [
          {"q", q_c, fn {q, _, _, _} -> q end},
          {"p", p_c, fn {_, p, _, _} -> p end},
          {"grad", g_c, fn {_, _, g, _} -> g end},
          {"logp", lp_c, fn {_, _, _, lp} -> lp end}
        ] do
      w = worst_rel(gpu, Enum.map(host, pick))

      assert w <= @tol,
             "#{label}: #{name} worst relative Δ #{w} exceeds #{@tol}"
    end

    # A STANDBY, not the sharpest assertion here — I claimed the latter in the
    # plan and the mutation said otherwise.
    #
    # The design intent is real: logp may legitimately differ from the host by a
    # constant normaliser, and what the Metropolis ratio consumes is the SHAPE
    # along the trajectory, so a constant offset is ratio-equivalent and
    # harmless while a varying one is a defect.
    #
    # But as written this is IMPLIED by the element-wise logp assertion above:
    # if every element agrees to 1e-13 then the offset is within 2e-13 by
    # arithmetic. Making `logp_chain[k]` lag its position by one — the historical
    # defect that paired every leaf with the density of the PREVIOUS position,
    # produced no divergences because it was mis-scaled rather than mis-signed,
    # and read as "Ampere over-dispersion" for three weeks — fails all five
    # tests on the element-wise check, never reaching here.
    #
    # It is kept because it becomes load-bearing the moment the element-wise
    # logp bound is relaxed to permit a normaliser, which is the situation the
    # bench anticipated. Measured today, the normaliser is 0.0 on every host and
    # arm, so that relaxation is not needed and this assertion is a guard
    # against a future in which it is.
    offs =
      Enum.zip(
        Nx.to_flat_list(lp_c),
        host |> Enum.flat_map(fn {_, _, _, lp} -> Nx.to_flat_list(lp) end)
      )
      |> Enum.map(fn {a, b} -> a - b end)

    spread = Enum.max(offs) - Enum.min(offs)

    assert spread <= @offset_tol,
           "#{label}: logp offset is NOT constant along the trajectory " <>
             "(spread #{spread} > #{@offset_tol}) — the densities do not describe " <>
             "the same states as the positions beside them"
  end

  describe "three observed nodes with DISTINCT sigmas" do
    # Small eps: if the arithmetic is right this must agree to ~1e-15.
    test "small step size" do
      assert_agrees(three_observed_ir(), 1, 0.05, [0.5], [1.0], "eps=0.05")
    end

    # The adapted eps from the original bug report, where the trajectory moves
    # ~2 sigma per step and any real discrepancy is amplified.
    test "adapted step size, from the centre" do
      assert_agrees(three_observed_ir(), 1, 1.1391216000810296, [0.5], [1.0], "eps=1.139 q0=0.5")
    end

    test "adapted step size, from the tail" do
      assert_agrees(
        three_observed_ir(),
        1,
        1.1391216000810296,
        [3.99],
        [0.1],
        "eps=1.139 q0=3.99"
      )
    end
  end

  describe "paths with the least element-wise coverage" do
    test "a vector RV with Nx.dot over a captured design matrix" do
      assert_agrees(vector_rv_ir(), 2, 0.02, [0.4, 1.8], [-0.2, 0.3], "vector RV + dot")
    end

    test "a Custom likelihood that reads its observations" do
      assert_agrees(custom_reads_obs_ir(), 1, 0.02, [1.7], [0.2], "Custom reads obs")
    end
  end
end
