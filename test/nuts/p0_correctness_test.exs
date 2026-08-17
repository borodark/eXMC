defmodule Exmc.NUTS.P0CorrectnessTest do
  @moduledoc """
  Regression tests for the two defects fixed in 0.3.1.

  Both defects were invisible to every check this repo had, for the same
  reason: the existing statistical harness is DIFFERENTIAL. It runs the model
  under two backends and asks whether they agree. A defect in the shared NUTS
  tree moves both arms identically, so the comparison passes; a defect in the
  chain shader shifts the arm that the other arm is being compared against.
  The 33/33 posteriordb gate could not see them either — its window is a
  factor of 2 on the standard deviation, which is a factor of 4 on the
  variance, and the worst of these was 1.38.

  So these tests deliberately do NOT compare arms. Two are structural
  assertions about generated code and tree bookkeeping, and one measures
  against the distribution's own analytic moments.
  """
  use ExUnit.Case, async: false

  alias Exmc.{Builder, Rewrite}
  alias Exmc.Dist.{Normal, HalfNormal}
  alias Exmc.NUTS.CustomSynth.MultiRvCustomSpec
  alias Exmc.NUTS.Sampler
  alias Exmc.NUTS.Vulkan.Validator

  # ------------------------------------------------------------------
  # Defect 2 — logp_chain[k] described the state BEFORE step k
  # ------------------------------------------------------------------

  describe "chain shader: logp_chain[k] describes the state q_chain[k] describes" do
    setup do
      components = %{
        priors: [{"x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)}}],
        custom: nil,
        layout: ["x"]
      }

      {:ok, glsl} = MultiRvCustomSpec.render(components)
      {:ok, glsl: glsl}
    end

    test "the log-density is evaluated after the position update, not before", %{glsl: glsl} do
      # `qi = qn;` is the position update. Everything written into the
      # per-step chain buffers must be computed after it, because the host
      # reads logp_chain[k] as the density AT q_chain[k].
      update = index_of!(glsl, ~r/\bqi\s*=\s*qn\s*;/, "position update `qi = qn;`")

      lp = index_of!(glsl, ~r/\blp_i\s*=\s*(?!0\.0)/, "assignment to lp_i from the emitted body")

      assert lp > update, """
      The prior log-density is assigned to lp_i BEFORE the position update.

      logp_chain[k] is stored alongside q_chain[k] / p_chain[k], so the host
      believes it is the density at that position. Evaluating it pre-update
      lags it by one leapfrog step, and the host feeds that lagged value
      straight into the Metropolis ratio and the U-turn test.

        `qi = qn;`  at offset #{update}
        `lp_i = ...` at offset #{lp}
      """
    end

    test "the gradient written to grad_chain is also post-update", %{glsl: glsl} do
      # Same invariant, and it already held — asserted so that a future edit
      # which moves the logp body back cannot quietly take the gradient with
      # it.
      update = index_of!(glsl, ~r/\bqi\s*=\s*qn\s*;/, "position update")
      grad_qn = index_of!(glsl, ~r/\bgrad_qn\s*=\s*(?!0\.0)/, "assignment to grad_qn")

      assert grad_qn > update
    end

    test "lp_i and grad_qn are computed in the same block", %{glsl: glsl} do
      # The whole defect was these two drifting apart. If they are adjacent,
      # neither can lag the other.
      lp = index_of!(glsl, ~r/\blp_i\s*=\s*(?!0\.0)/, "assignment to lp_i")
      grad_qn = index_of!(glsl, ~r/\bgrad_qn\s*=\s*(?!0\.0)/, "assignment to grad_qn")

      between = String.slice(glsl, min(lp, grad_qn), abs(lp - grad_qn))

      refute between =~ ~r/\bqi\s*=\s*qn\s*;/,
             "a position update now sits between the gradient and the log-density"
    end
  end

  # ------------------------------------------------------------------
  # Defect 1 — an invalid doubling was still merged into the trajectory
  # ------------------------------------------------------------------

  describe "tree: an invalid doubling contributes no proposal" do
    @tag :statistical
    @tag timeout: 600_000
    test "Normal(0,1) recovers its analytic variance" do
      # This is the check that could actually see the defect. With the
      # invalid doubling merged, states beyond the U-turn enter
      # combined_log_weight and can be drawn as the proposal, which breaks
      # detailed balance outward and over-disperses the posterior:
      # measured variance was 1.378 against a true 1.0.
      #
      # The effect is roughly 12 standard errors at this sample size, so a
      # 4-sigma gate is not a close call.
      samples = draw(Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})

      assert_analytic(samples, {:normal, 0.0, 1.0})
    end

    @tag :statistical
    @tag timeout: 600_000
    test "HalfNormal(1) recovers its analytic mean" do
      # A bounded support turns over-dispersion into a visible mean shift:
      # the posterior drifts away from the boundary. Measured mean was
      # 0.863 against an analytic 0.7979.
      samples = draw(HalfNormal, %{sigma: Nx.tensor(1.0)})

      assert_analytic(samples, {:half_normal, 1.0})
    end
  end

  # ------------------------------------------------------------------
  # The documented push-constants cap must be the real one
  # ------------------------------------------------------------------

  describe "chain shader: the push block caps model width at 13 prior floats" do
    # The moduledoc used to claim 28 prior floats, and eleven dispatch guards
    # read `d <= 256`, so the cap was reasoned about as 256 free RVs. The real
    # number is 20x smaller and it changes what the synthesis path is FOR.
    # Pinned here so nobody has to re-derive it from a byte count again.

    test "the fixed header is 24 bytes, not 16" do
      {:ok, bin, n} = pack(Exmc.Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)}, 0)
      assert byte_size(bin) == 24
      assert n == 24
    end

    test "one-parameter priors cap at d = 13" do
      assert max_d(Exmc.Dist.HalfNormal, %{sigma: Nx.tensor(1.0)}) == 13
      assert max_d(Exmc.Dist.Exponential, %{lambda: Nx.tensor(1.0)}) == 13
    end

    test "Normal caps at d = 6" do
      assert max_d(Exmc.Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)}) == 6
    end

    test "TruncatedNormal caps at d = 3" do
      params = %{
        mu: Nx.tensor(0.0),
        sigma: Nx.tensor(1.0),
        lower: Nx.tensor(0.0),
        upper: Nx.tensor(1.0)
      }

      assert max_d(Exmc.Dist.TruncatedNormal, params) == 3
    end

    test "one RV past the cap is refused, not truncated" do
      params = %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)}
      assert {:ok, _, _} = pack(Exmc.Dist.Normal, params, 6)
      assert {:error, :push_too_large} = pack(Exmc.Dist.Normal, params, 7)
    end
  end

  defp pack(mod, params, d) do
    priors = for i <- 1..d//1, do: {"x#{i}", mod, params}
    Exmc.NUTS.CustomSynth.Push.pack(%{K: 32, n_obs: 0, d: d, eps: 0.1, priors: priors})
  end

  defp max_d(mod, params) do
    Enum.reduce_while(1..40, 0, fn d, _acc ->
      case pack(mod, params, d) do
        {:ok, _, _} -> {:cont, d}
        {:error, :push_too_large} -> {:halt, d - 1}
      end
    end)
  end

  # ------------------------------------------------------------------
  # Helpers
  # ------------------------------------------------------------------

  # These checks are deliberately run on the host path, so they pin the
  # compiler to :none — but `Application.put_env/3` is global and lives for the
  # rest of the VM. Setting it without restoring it leaked `:none` into every
  # test that happened to run after this file, which silently converted the
  # whole backend sweep into another run of the pure-Elixir path. Because
  # ExUnit orders files by a random seed, *how much* of the suite it swallowed
  # varied run to run: `EXMC_COMPILER=vulkan mix test` reported 3 failures for
  # the full suite while `EXMC_COMPILER=vulkan mix test test/integration_test.exs`
  # reported 2, the extra one being the known Vulkan defect at
  # integration_test.exs:639.
  #
  # Restore it. A test that mutates application env owes the next test the
  # value it found.
  defp draw(dist, params, seeds \\ [1, 2]) do
    Exmc.TestHelper.put_env_scoped(:compiler, :none)

    ir =
      Builder.new_ir()
      |> Builder.rv("x", dist, params)
      |> Rewrite.apply()

    compiled = Exmc.Compiler.compile_for_sampling(ir)

    Enum.flat_map(seeds, fn seed ->
      {trace, _stats} =
        Sampler.sample_compiled(compiled, %{},
          num_warmup: 300,
          num_samples: 1200,
          seed: seed
        )

      trace |> Map.fetch!("x") |> Nx.to_flat_list()
    end)
  end

  defp assert_analytic(samples, meta) do
    case Validator.check_analytic(samples, :host, meta) do
      :ok ->
        :ok

      {:error, info} ->
        {m, v} = Validator.mean_var(samples)
        {:moments, %{mean: tm, var: tv}} = Validator.analytic_moments(meta)

        flunk("""
        Posterior does not match #{inspect(meta)}'s analytic moments.

          mean  truth #{tm}   got #{m}
          var   truth #{tv}   got #{v}
          ess   #{Map.get(info, :ess)}

        Failing check: #{inspect(info)}

        If the variance is inflated and any bounded support has drifted away
        from its boundary, suspect the !valid_subtree guards in
        Tree.do_build/11, Tree.build_subtree/10, and their two counterparts
        in native/exmc_tree/src/tree.rs. Note there are THREE tree
        implementations behind config flags (use_nif, full_tree_nif) and each
        carries its own copy — see bench/nuts_truth.exs, which sweeps them.
        """)
    end
  end

  # Regex index with a readable failure when the pattern is missing, so that a
  # template rename shows up as "pattern not found" rather than as a passing
  # test that asserted nothing.
  defp index_of!(glsl, regex, what) do
    case Regex.run(regex, glsl, return: :index) do
      [{start, _} | _] ->
        start

      nil ->
        flunk("""
        Could not find #{what} in the generated GLSL.

        This test cannot verify the ordering invariant if it cannot locate
        the statements, so it fails rather than passing vacuously. If the
        template was legitimately renamed, update the pattern: #{inspect(regex)}
        """)
    end
  end
end
