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
  # The push block does NOT cap model width
  # ------------------------------------------------------------------

  describe "chain shader: the push block is a fixed header and caps nothing" do
    # This describe block used to be called "the push block caps model width at
    # 13 prior floats", and it asserted d <= 13 for one-parameter priors, 6 for
    # Normal, 4 for StudentT, 3 for TruncatedNormal, with a comment saying the
    # number "changes what the synthesis path is FOR".
    #
    # Every one of those numbers was real arithmetic on a quantity that did not
    # matter. `pack/1` appended one f64 per prior parameter; the NIF forwards
    # only the 24-byte header and drops the rest; and `MultiRvCustomSpec` bakes
    # prior parameters into the GLSL as literals, so the shader never read them
    # either. The tail's only effect was to be counted against the NIF's
    # `push.len() > 128` check — a budget it never spent.
    #
    # So these four tests were not measuring a limit. They were pinning a
    # defect, and pinning it with a table precise enough to look authoritative.
    # An 8-RV model was being pushed onto per-op sampling for no reason: 0 chain
    # dispatches and 160.9 s, against 2564 and 12.3 s once the tail was removed.
    #
    # Kept as a regression: if anyone reinstates the tail, these fail.
    # See `test/exmc/nuts/custom_synth/push_width_test.exs` for the end-to-end
    # dispatch and posterior checks.

    test "the fixed header is 24 bytes, not 16" do
      {:ok, bin, n} = pack(Exmc.Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)}, 0)
      assert byte_size(bin) == 24
      assert n == 24
    end

    test "the block is 24 bytes at every width, for every prior arity" do
      cases = [
        {Exmc.Dist.HalfNormal, %{sigma: Nx.tensor(1.0)}},
        {Exmc.Dist.Exponential, %{lambda: Nx.tensor(1.0)}},
        {Exmc.Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)}},
        {Exmc.Dist.TruncatedNormal,
         %{
           mu: Nx.tensor(0.0),
           sigma: Nx.tensor(1.0),
           lower: Nx.tensor(0.0),
           upper: Nx.tensor(1.0)
         }}
      ]

      for {mod, params} <- cases, d <- [1, 7, 14, 40] do
        assert {:ok, bin, 24} = pack(mod, params, d),
               "#{inspect(mod)} at d=#{d} must still pack to a 24-byte header"

        assert byte_size(bin) == 24
      end
    end

    test "the widths this used to refuse now pack fine" do
      # The exact boundaries the old cap enforced: 7 Normals (was 6 max),
      # 14 one-parameter priors (was 13), 4 TruncatedNormals (was 3).
      assert {:ok, _, 24} = pack(Exmc.Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)}, 7)
      assert {:ok, _, 24} = pack(Exmc.Dist.HalfNormal, %{sigma: Nx.tensor(1.0)}, 14)

      tn = %{
        mu: Nx.tensor(0.0),
        sigma: Nx.tensor(1.0),
        lower: Nx.tensor(0.0),
        upper: Nx.tensor(1.0)
      }

      assert {:ok, _, 24} = pack(Exmc.Dist.TruncatedNormal, tn, 4)
    end
  end

  defp pack(mod, params, d) do
    priors = for i <- 1..d//1, do: {"x#{i}", mod, params}
    Exmc.NUTS.CustomSynth.Push.pack(%{K: 32, n_obs: 0, d: d, eps: 0.1, priors: priors})
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
