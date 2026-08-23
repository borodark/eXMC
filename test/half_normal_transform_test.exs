defmodule Exmc.HalfNormalTransformTest do
  @moduledoc """
  HalfNormal's default transform is `:softplus`, and this pins the choice.

  It is pinned because the two trees disagreed about it and the disagreement
  survived for months in silence. The applications tree carried `:log`, with a
  comment citing "W7 Stage 2.5 — to match the chain shader"; that shader has
  since been removed, and the synthesised path emits softplus correctly. This
  tree carried `:softplus` and had no test for it at all, which is why nothing
  noticed.

  Measured before choosing, 4000 draws, seed 7, both arms, HalfNormal(1):

      transform   arm      mean err   var err   ESS    divergences
      :softplus   EXLA      0.0101    0.0260    1417        8
      :log        EXLA      0.0192    0.0157    1030      203
      :softplus   vulkan    0.0075    0.0225    1327        7
      :log        vulkan    0.0054    0.0081    1111      139

  Both pass `Validator.check_analytic/3` on both arms, so correctness does not
  separate them — only geometry does, and `:log` costs 17-20x the divergences
  for it. Either transform is a valid R -> R+ bijection; this is a conditioning
  decision, not a correctness one, which is exactly the kind that drifts
  unnoticed when nothing asserts it.
  """

  use ExUnit.Case, async: true

  alias Exmc.{Builder, LogProb, Transform}
  alias Exmc.Dist.HalfNormal

  # 1e-7, and the number is the finding rather than a shrug.
  #
  # With every literal in this test built at f64 the residual disagreement is
  # ~1.75e-8, which is f32-scale, not f64-scale. It comes from the library, not
  # from here: `Exmc.Dist.HalfNormal.logpdf/2` builds `Nx.tensor(-0.5)`,
  # `Nx.tensor(2.0)` and `Nx.tensor(2.0 * :math.pi())`, and
  # `Exmc.Transform.softplus/1` builds `Nx.tensor(0.0)` — all of which are f32,
  # because f32 is Nx's default float type. A bare f32 constant in an f64
  # expression makes the arithmetic f32-accurate while leaving the result
  # f64-typed, which is TODO.md §5's unaudited class, here with a site and a
  # magnitude attached.
  #
  # Tighten this to 1e-12 once those literals are built at the tensor's own
  # type; it should pass. Until then a tighter bound would be measuring Nx's
  # default dtype rather than the transform.
  defp close!(a, b, tol \\ 1.0e-7) do
    assert_in_delta Nx.to_number(a), Nx.to_number(b), tol
  end

  test "the default transform is :softplus" do
    assert HalfNormal.transform(%{sigma: Nx.tensor(1.0)}) == :softplus
  end

  test "it is the only positive-support family that could reasonably differ, and it matches its own Jacobian" do
    # x = softplus(z); log|dx/dz| = log(sigmoid(z)) = -softplus(-z).
    # Hand-computed rather than taken from Transform, so this fails if the
    # transform and its Jacobian ever stop agreeing.
    # Every literal is built at f64 on purpose. `Nx.tensor(0.2)` is f32 — Nx's
    # default float type — and an f32 leaf anywhere in this chain drags the
    # comparison to f32 epsilon: the first version of this test failed by
    # 1.19e-7, which is exactly that and nothing to do with the transform.
    # Same trap as TODO.md §5.
    f = fn v -> Nx.tensor(v, type: :f64) end

    ir =
      Builder.new_ir()
      |> Builder.rv("z", HalfNormal, %{sigma: f.(1.0)})

    z = f.(0.2)
    x = Nx.log(Nx.add(f.(1.0), Nx.exp(z)))

    log_p_x =
      f.(-0.5)
      |> Nx.multiply(Nx.add(Nx.multiply(x, x), Nx.log(f.(2.0 * :math.pi()))))
      |> Nx.add(Nx.log(f.(2.0)))

    jac = Nx.negate(Nx.log(Nx.add(f.(1.0), Nx.exp(Nx.negate(z)))))

    close!(LogProb.eval(ir, %{"z" => z}), Nx.add(log_p_x, jac))
    close!(Transform.log_abs_det_jacobian(:softplus, z), jac)
  end

  test "the sampler still recovers HalfNormal(1)'s analytic moments under it" do
    # The conditioning claim above is only worth anything if the posterior is
    # right, so the moment gate runs too. Sized loosely: this pins correctness,
    # not the ESS numbers in the moduledoc.
    ir =
      Builder.new_ir()
      |> Builder.rv("x", HalfNormal, %{sigma: Nx.tensor(1.0)})

    {trace, stats} =
      Exmc.NUTS.Sampler.sample(ir, %{}, num_warmup: 500, num_samples: 1500, seed: 7)

    xs = trace["x"] |> Nx.to_flat_list()

    assert :ok = Exmc.NUTS.Vulkan.Validator.check_analytic(xs, :softplus, {:halfnormal, 1.0, 0.0})

    # The divergence gap between the two transforms is 17-20x, so a generous
    # bound still catches a silent revert to :log.
    assert stats.divergences < 60,
           "#{stats.divergences} divergences on HalfNormal(1) — :log measured " <>
             "139-203 here and :softplus 7-8, so this looks like the transform " <>
             "changed back"
  end
end
