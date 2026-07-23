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

  Phases A + B require zero new shader code; the existing
  `leapfrog_chain_*` shaders in `nx_vulkan/priv/shaders/` are
  reused. C + D require GLSL emission via
  `Nx.Vulkan.Codegen.compile_cached/1`.
  """

  alias Exmc.{IR, Node}

  @typedoc "Tagged-tuple meta consumed by `Tree.do_dispatch/10`."
  @type meta ::
          {:normal, mu :: number(), sigma :: number()}
          | {:exponential, lambda :: number()}
          | {:studentt, mu :: number(), sigma :: number(),
             nu :: number(), logp_const :: number()}
          | {:cauchy, loc :: number(), scale :: number(),
             log_pi_scale :: number()}
          | {:halfnormal, sigma :: number(), log_const :: number()}
          | {:weibull, k :: number(), lambda :: number(),
             logp_const :: number()}

  @doc """
  Inspect an IR. If it's a recognized single-RV-model shape,
  return `{:ok, meta}` for the right chain shader. Otherwise
  return `:unsupported`.

  Phase A + B implementations dispatch on the IR's nodes here.
  """
  @spec detect_meta(IR.t()) :: {:ok, meta()} | :unsupported | {:unsupported, :push_too_large}
  def detect_meta(%IR{nodes: nodes} = ir) when map_size(nodes) == 1 do
    # Surface A of PLAN_F64_CHAIN_SHADER (Option B): under D88's f64
    # Vulkano default, route single-family models to the vulkano synth
    # path (Surface 7) instead of the spirit C++ family fused SPVs.
    # Spirit family SPVs are f32-only; under f64 they trigger the D87
    # silent-collapse pathology (the whole reason #175 exists). Synth
    # emits precision-portable GLSL and dispatch.ex's chain_synth_vulkano
    # routes to leapfrog_chain_synth_f64 automatically.
    #
    # At :f32 (the pre-D88 default or an explicit force_precision: :f32
    # override) keep the family fast path — it's a few % faster and works
    # correctly at f32. Under EXLA compiler, always keep the family
    # fast path — synth returns a `{:synthesised, ...}` meta for the
    # Vulkan chain-shader pipeline, which EXLA has no notion of; the
    # synth SPV also can't be compiled for a multi-D single-RV shape
    # (regression seen in bench/nx_0_12_race_results.md at d=8 / d=50).
    with :f64 <- Exmc.JIT.precision(),
         Nx.Vulkan <- Exmc.JIT.detect_compiler(),
         {:ok, meta} <- try_synthesise(ir) do
      {:ok, meta}
    else
      _ -> detect_family(nodes)
    end
  end

  defp detect_family(nodes) do
    [{_id, node}] = Map.to_list(nodes)
    detect_from_node(node)
  end

  defp try_synthesise(%IR{} = ir) do
    try do
      Exmc.NUTS.CustomSynth.synthesise(ir)
    rescue
      _ -> :unsupported
    catch
      _, _ -> :unsupported
    end
  end

  # Multi-RV IRs with at least one Custom-likelihood node: hand off
  # to the custom-distribution synthesis pipeline (M-II R0/R1; see
  # specs/vulkan-custom-synthesis.md and Exmc.NUTS.CustomSynth).
  # The synthesis path either returns `{:ok, synthesised_meta}`
  # when all log_prob/grad expressions emit cleanly, or
  # `:unsupported` when the model's Defn graph contains ops the
  # emitter doesn't yet cover.
  def detect_meta(%IR{nodes: nodes} = ir) when map_size(nodes) > 1 do
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
        try do
          Exmc.NUTS.CustomSynth.synthesise(ir)
        rescue
          _ -> :unsupported
        catch
          _, _ -> :unsupported
        end

      true ->
        :unsupported
    end
  end

  def detect_meta(%IR{}), do: :unsupported

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
