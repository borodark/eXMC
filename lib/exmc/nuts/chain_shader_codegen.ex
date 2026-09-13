defmodule Exmc.NUTS.ChainShaderCodegen do
  @moduledoc """
  Generate a fused-leapfrog dispatcher from an `Exmc.IR` by
  routing to (Phase A) hand-written chain shaders or (Phase C+)
  generated GLSL.

  Phases — see `docs/PLAN_DSL_SHADER_CODEGEN.md` for the full
  rationale:

  * Phase A — single-RV unconstrained models (Normal, StudentT,
    Cauchy). Auto-detects the IR shape and constructs the right
    `fused_leapfrog_meta` tagged tuple, eliminating the manual
    `Application.put_env` step.
  * Phase B — single-RV constrained models (HalfNormal,
    Exponential, Weibull). Same as A but verifies the transform
    matches what the chain shader expects.
  * Phase C — hierarchical (multi-RV) models. Composes
    per-distribution gradient + logp templates into a generated
    GLSL leapfrog shader.
  * Phase D — observed-data models. Bakes likelihood data into
    the generated shader as constants.

  Phases A + B originally reused nx_vulkan's hand-written
  `leapfrog_chain_*` shaders; nx_vulkan deleted those (`8006a4d`,
  2026-09-01) and every chain shader is now synthesised here and
  compiled via `Nx.Vulkan.Codegen.compile_cached/1`, which is what
  C + D always required.
  """

  alias Exmc.{IR, Node}

  @typedoc "Tagged-tuple meta consumed by `Tree.do_dispatch/10`."
  @type meta ::
          {:normal, mu :: number(), sigma :: number()}
          | {:exponential, lambda :: number()}
          | {:studentt, mu :: number(), sigma :: number(), nu :: number(), logp_const :: number()}
          | {:cauchy, loc :: number(), scale :: number(), log_pi_scale :: number()}
          | {:halfnormal, sigma :: number(), log_const :: number()}
          | {:weibull, k :: number(), lambda :: number(), logp_const :: number()}

  @doc """
  Inspect an IR. If it's a recognized single-RV-model shape,
  return `{:ok, meta}` for the right chain shader. Otherwise
  return `:unsupported`.

  Phase A + B implementations dispatch on the IR's nodes here.
  """
  @spec detect_meta(IR.t(), keyword()) ::
          {:ok, meta() | tuple()} | :unsupported | {:unsupported, atom()}
  def detect_meta(ir, opts \\ [])

  def detect_meta(%IR{nodes: nodes} = ir, opts) when map_size(nodes) == 1 do
    # UNDER VULKAN, SYNTHESIS OR REFUSAL, NEVER A FAMILY META.
    #
    # Every chain shader is synthesised now: nx_vulkan deleted the hand-written
    # family SPVs (`8006a4d`), and `Exmc.NUTS.Vulkan.Dispatch.do_chain/8` has
    # exactly one clause, for `{:synthesised, ...}`. The family tuples
    # detect_family/1 builds still reach it -- Tree.do_dispatch/10 routes
    # `{:normal, ...}` et al. to it under Nx.Vulkan -- and die there with a
    # FunctionClauseError at the first chain.
    #
    # This clause used to fall back to detect_family/1 whenever synthesis did
    # not return {:ok, _}, and at :f32 without trying synthesis at all. On
    # 2026-09-13 that turned an environment defect into a mystery: on FreeBSD,
    # `:crypto` was not on the code path (fixed in mix.exs), CustomSynth raised,
    # try_synthesise/2 swallowed it, the family fallback handed Dispatch a
    # `{:normal, 0.0, 1.0}`, and bench/nuts_truth.exs died after one second on
    # every FreeBSD host with an error that named neither crypto nor synthesis.
    # A refusal here reaches the Plan-B' guard in Exmc.Compiler, which says why.
    #
    # Under EXLA or the Evaluator nothing changes: the family meta is how
    # Sampler seeds the initial mass matrix (prior_inv_mass_per_rv/2), and no
    # chain dispatch happens off Vulkan. The multi-RV clause below already
    # synthesised unconditionally; this makes the single-RV one agree with it.
    if Exmc.JIT.detect_compiler() == Nx.Vulkan do
      try_synthesise(ir, opts)
    else
      detect_family(nodes)
    end
  end

  defp detect_family(nodes) do
    [{_id, node}] = Map.to_list(nodes)
    detect_from_node(node)
  end

  # Synthesis is best-effort by contract: a model whose Defn graph contains ops
  # the emitter does not cover returns :unsupported, and the Plan-B' guard turns
  # that into a loud compile-time refusal rather than a silent 100x slower
  # per-op fallback.
  #
  # `opts` is threaded because synthesise/2 rewrites the IR itself, and
  # Rewrite.apply/2 drops the non-centred-parameterisation pass on `ncp: false`.
  # Both arms have to be given the same flag or the shader and the PointMap
  # describe different coordinate systems — see the comment at its call site.
  #
  # Exmc.SynthReferenceError is deliberately NOT swallowed: it names the
  # offending id or prints the cycle path, and converting that to a bare
  # :unsupported replaces a message about the actual problem with the Plan-B'
  # guard's generic "reshape the model" advice.
  #
  # It has its own type rather than being an ArgumentError, because rescuing
  # THAT was too broad — Nx raises ArgumentError too, and normalize_params/1
  # slicing MvNormal's rank-2 covariance produced one ("invalid start indices
  # rank for shape of rank 2") that escaped synthesis and broke the Plan-B'
  # guard for every MvNormal model. Catching a type is not catching a cause.
  #
  # Anything else that RAISES is still turned into a refusal, but a tagged,
  # logged one: `{:unsupported, :synthesis_raised}`, with the exception and the
  # top of its stacktrace in a warning. A controlled refusal is a return value
  # from CustomSynth; a raise is a defect or an environment problem, and the
  # bare `:unsupported` it used to become was indistinguishable from "this
  # model has no chain-shader form". The `:crypto` hunt is what that cost.
  #
  # `:synthesiser` in opts replaces CustomSynth -- a seam for tests, which need
  # a synthesis that raises and have no ordinary model that does. It is popped
  # before the rest of opts reach synthesise/2.
  defp try_synthesise(%IR{} = ir, opts) do
    {synthesiser, opts} = Keyword.pop(opts, :synthesiser, Exmc.NUTS.CustomSynth)

    try do
      synthesiser.synthesise(ir, opts)
    rescue
      e in Exmc.SynthReferenceError ->
        reraise e, __STACKTRACE__

      e ->
        log_synthesis_raised(:error, e, __STACKTRACE__)
        {:unsupported, :synthesis_raised}
    catch
      kind, value ->
        log_synthesis_raised(kind, value, __STACKTRACE__)
        {:unsupported, :synthesis_raised}
    end
  end

  defp log_synthesis_raised(kind, value, stacktrace) do
    require Logger

    Logger.warning(
      "[ChainShaderCodegen] chain-shader synthesis raised, so this model is refused " <>
        "as {:unsupported, :synthesis_raised}. A raise is not a model-shape refusal; " <>
        "it is a defect or an environment problem:\n" <>
        Exception.format(kind, value, Enum.take(stacktrace, 8))
    )
  end

  # Multi-RV IRs with at least one Custom-likelihood node: hand off
  # to the custom-distribution synthesis pipeline (M-II R0/R1; see
  # specs/vulkan-custom-synthesis.md and Exmc.NUTS.CustomSynth).
  # The synthesis path either returns `{:ok, synthesised_meta}`
  # when all log_prob/grad expressions emit cleanly, or
  # `:unsupported` when the model's Defn graph contains ops the
  # emitter doesn't yet cover.
  def detect_meta(%IR{nodes: nodes} = ir, opts) when map_size(nodes) > 1 do
    cond do
      # Task #153 (Option A, 2026-05-26): MultiRvCustomSpec.compose_logp_defn
      # now applies Transform.apply + log_abs_det_jacobian per-RV (matches
      # Compiler.node_term/4). The Task #150 guard for any_transformed_rv?
      # is no longer needed — synth path handles transformed RVs correctly.
      # Differential probe confirms: at q=[0.05]*8, both paths give
      # logp ≈ -42.53 (within 1.58e-8 of each other).
      #
      # If a future model uses a transform Exmc.Transform.apply doesn't
      # cover (currently: nil/:log/:softplus/:logit/:stick_breaking), the
      # try/rescue below catches the synthesise failure and falls through.

      # synth P1: multi-RV with an observed standard-family likelihood
      # (an {:obs, rv_id, ...} node and no Custom node). CustomSynth's
      # compose path sums the observed logpdf into the joint. If the
      # emitter can't cover the shape, synthesise returns :unsupported and
      # the Plan-B' guard raises SynthUnsupportedError (same as any other
      # unsynthesisable model under Vulkan).
      has_custom_likelihood?(nodes) or has_observed_likelihood?(nodes) ->
        try_synthesise(ir, opts)

      # Prior-only multi-RV, and this clause used to be `:unsupported`.
      #
      # It was a gate, not a capability limit. CustomSynth.synthesise/1 handles
      # these models — probed directly on a two-Normal IR it returns
      # {:ok, {:synthesised, ...}} — but detect_meta/1 never called it unless
      # the model had a Custom or observed likelihood, so every prior-only
      # multi-RV model raised SynthUnsupportedError on a Vulkan-only host.
      # That is two of the seven such failures on the FreeBSD fleet, from
      # models as ordinary as two independent Normals.
      #
      # Checked before opening it, because "it emits" and "it is correct" are
      # different claims and only the second one matters. The composed
      # log-density was compared against Compiler.compile/1's on the same q,
      # 200 random draws each:
      #
      #     2 independent Normals              worst rel. diff 0.0
      #     Normal + Exponential               worst rel. diff 0.0
      #     Normal + HalfCauchy(:log) + Exp    worst rel. diff 1.69e-9
      #
      # Exact where no transform is involved, f64 rounding where one is. The
      # transformed case matters most: a wrong Jacobian would give a finite,
      # plausible log-density and a silently wrong posterior, which is the
      # shape of the bug that read as "Ampere over-dispersion" for three weeks.
      true ->
        try_synthesise(ir, opts)
    end
  end

  def detect_meta(%IR{}, _opts), do: :unsupported

  defp has_custom_likelihood?(nodes) do
    Enum.any?(nodes, fn
      {_id, %Node{op: {:rv, Exmc.Dist.Custom, _params}}} -> true
      _ -> false
    end)
  end

  defp has_observed_likelihood?(nodes) do
    Enum.any?(nodes, fn
      {_id, %Node{op: {:obs, _rv_id, _value, _meta}}} -> true
      _ -> false
    end)
  end

  # Task #150 guard — returns true if any RV in the IR has a non-trivial
  # unconstrained-space transform (log/logit/stick_breaking/...). Custom
  # likelihood RVs are skipped (their transform is meaningless — they
  # contribute to log-likelihood, not log-prior). 3-tuple RV ops carry
  # no explicit transform; query the dist module's transform/1 directly.
  # 4-tuple RV ops include the resolved transform (from AttachDefaultTransforms).
  defp any_transformed_rv?(nodes) do
    Enum.any?(nodes, fn
      {_id, %Node{op: {:rv, Exmc.Dist.Custom, _params}}} -> false
      {_id, %Node{op: {:rv, dist, params}}} -> not is_nil(dist.transform(params))
      {_id, %Node{op: {:rv, _dist, _params, transform}}} -> not is_nil(transform)
      _ -> false
    end)
  end

  # --- Phase A: single-RV unconstrained ---

  defp detect_from_node(%Node{op: {:rv, Exmc.Dist.Normal, params}}) do
    with {:ok, mu} <- scalar_param(params, :mu),
         {:ok, sigma} <- scalar_param(params, :sigma) do
      {:ok, {:normal, mu, sigma}}
    end
  end

  defp detect_from_node(%Node{op: {:rv, Exmc.Dist.StudentT, params}}) do
    with {:ok, df} <- scalar_param(params, :df),
         {:ok, loc} <- scalar_param(params, :loc),
         {:ok, scale} <- scalar_param(params, :scale) do
      logp_const = student_t_logp_const(df, loc, scale)
      {:ok, {:studentt, loc, scale, df, logp_const}}
    end
  end

  defp detect_from_node(%Node{op: {:rv, Exmc.Dist.Cauchy, params}}) do
    with {:ok, loc} <- scalar_param(params, :loc),
         {:ok, scale} <- scalar_param(params, :scale) do
      log_pi_scale = -:math.log(:math.pi() * scale)
      {:ok, {:cauchy, loc, scale, log_pi_scale}}
    end
  end

  # --- Phase B: single-RV constrained (transform present) ---

  defp detect_from_node(%Node{op: {:rv, Exmc.Dist.Exponential, params}}) do
    with {:ok, lambda} <- scalar_param(params, :lambda) do
      {:ok, {:exponential, lambda}}
    end
  end

  defp detect_from_node(%Node{op: {:rv, Exmc.Dist.HalfNormal, params}}) do
    with {:ok, sigma} <- scalar_param(params, :sigma) do
      log_const = -:math.log(sigma) - 0.5 * :math.log(:math.pi())
      {:ok, {:halfnormal, sigma, log_const}}
    end
  end

  defp detect_from_node(%Node{op: {:rv, Exmc.Dist.Weibull, params}}) do
    with {:ok, k} <- scalar_param(params, :k),
         {:ok, lambda} <- scalar_param(params, :lambda) do
      # logp_const = n * (log(k) - k * log(lambda))
      # n is determined at dispatch time from the actual q tensor;
      # the chain shader takes the *per-element constant* and
      # the dispatch site multiplies by n. For now, emit the
      # per-element value (log(k) - k*log(lambda)); Tree.do_dispatch
      # multiplies by d at dispatch time. (TODO: wire the d
      # multiplication; for now caller must pre-multiply.)
      logp_const_per_elem = :math.log(k) - k * :math.log(lambda)
      {:ok, {:weibull, k, lambda, logp_const_per_elem}}
    end
  end

  # Phase 1 — synthesized chain shaders (templated GLSL).
  defp detect_from_node(%Node{op: {:rv, Exmc.Dist.Beta, params}}) do
    with {:ok, alpha} <- scalar_param(params, :alpha),
         {:ok, beta} <- scalar_param(params, :beta) do
      {:ok, {:beta, alpha, beta}}
    end
  end

  defp detect_from_node(%Node{op: {:rv, Exmc.Dist.Gamma, params}}) do
    with {:ok, alpha} <- scalar_param(params, :alpha),
         {:ok, beta} <- scalar_param(params, :beta) do
      {:ok, {:gamma, alpha, beta}}
    end
  end

  defp detect_from_node(%Node{op: {:rv, Exmc.Dist.Lognormal, params}}) do
    with {:ok, mu} <- scalar_param(params, :mu),
         {:ok, sigma} <- scalar_param(params, :sigma) do
      {:ok, {:lognormal, mu, sigma}}
    end
  end

  # Catch-all: unrecognized RV distribution, observed RV, hierarchical model.
  defp detect_from_node(_), do: :unsupported

  # --- Helpers ---

  # Extract a scalar number from a parameter map. Returns
  # :unsupported if the parameter is a string (parameter reference
  # → hierarchical), a non-scalar tensor, or missing.
  defp scalar_param(params, key) do
    case Map.fetch(params, key) do
      {:ok, %Nx.Tensor{shape: {}} = t} -> {:ok, Nx.to_number(t)}
      {:ok, n} when is_number(n) -> {:ok, n}
      _ -> :unsupported
    end
  end

  defp student_t_logp_const(df, _loc, scale) do
    lgamma = fn x -> Nx.to_number(Exmc.Math.lgamma(Nx.tensor(x))) end

    lgamma.((df + 1) / 2) -
      lgamma.(df / 2) -
      0.5 * :math.log(:math.pi() * df) -
      :math.log(scale)
  end
end
