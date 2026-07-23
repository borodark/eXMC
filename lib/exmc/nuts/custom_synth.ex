defmodule Exmc.NUTS.CustomSynth do
  @moduledoc """
  Synthesise a fused leapfrog chain shader from a multi-RV IR
  containing an `Exmc.Dist.Custom` likelihood.

  This is Mission-II R1 territory (`specs/vulkan-custom-synthesis.md`).
  The single entry point is `synthesise/1`, which is called from
  `ChainShaderCodegen.detect_meta/1` when the IR contains a Custom
  RV.  Return contract is the same as the rest of
  `ChainShaderCodegen`:

      {:ok, meta} | :unsupported

  where `meta` is consumed by `Exmc.NUTS.Tree.do_dispatch/10`.

  ## Current state — R0/R1 in progress

  - **R0 (this module's bare existence)**: routing from
    `ChainShaderCodegen.detect_meta/1` lands here; returns
    `:unsupported` until R1 lands a working emitter.
  - **R1 (in progress)**: walk the regime model's Custom logpdf_fn
    by tracing it with symbolic Nx tensors, render the resulting
    `Nx.Defn.Expr` tree to GLSL fragments for both `log_prob` and
    `dlog_prob/dq` (via `Nx.Defn.Grad`), then hand the fragments
    to the templated leapfrog chain shader.

  ## Walkthrough of the synthesis pipeline (R1+)

  ```
  IR (multi-RV + Custom node)
    │
    ├── extract priors → list of {name, dist, params}
    │     used to build the prior log_prob expression and the
    │     position-vector layout (one component per free RV)
    │
    ├── extract Custom.logpdf_fn → trace with symbolic inputs
    │     → Nx.Defn.Expr.t() representing the user's log-likelihood
    │
    ├── compose: full_logp = sum(prior_logps) + custom_logp
    │     Nx.Defn.Expr representing the joint log-posterior
    │
    ├── Nx.Defn.Grad on `full_logp` w.r.t. position vector
    │     → Nx.Defn.Expr for ∂logp/∂q
    │
    ├── walk both expressions to GLSL via `Exmc.NUTS.CustomSynth.Glsl`
    │     produces {log_prob_body, grad_body, n_inputs, n_outputs}
    │
    ├── render leapfrog template with the bodies plugged in
    │     uses `Nx.Vulkan.ShaderTemplate`
    │
    ├── glslangValidator + content-addressed cache
    │     existing `Nx.Vulkan.Synthesis.compile/1`
    │
    └── return {:ok, {:synthesised, sha256, layout, push_const_spec}}
          Tree.do_dispatch routes through Nx.Vulkan.Native
          for the synthesised shader's leaf op.
  ```

  ## Acceptance criteria (R1)

  Fuzz test in `test/exmc/nuts/custom_synth_test.exs`:

  1. Build the regime model IR from a 200-element synthetic returns
     vector.
  2. Walk it through `synthesise/1`; expect `{:ok, _meta}`.
  3. Generate 100 random position vectors q ∈ ℝ^8.
  4. For each q, compare:
     - `log_prob`(q) emitted GLSL vs BinaryBackend evaluation: rel
       error ≤ 1e-6.
     - `grad log_prob`(q) emitted GLSL vs `Nx.Defn.Grad` →
       BinaryBackend: rel error ≤ 1e-6.

  Until that passes, this module stays at `:unsupported`.
  """

  alias Exmc.IR

  @typedoc "Meta returned when synthesis succeeds; consumed by Tree.do_dispatch/10."
  @type synth_meta ::
          {:synthesised,
           sha256 :: binary(),
           layout :: [atom()],
           push_spec :: map(),
           spv_path :: Path.t(),
           obs_bin :: binary()}

  @doc """
  Entry point.  Walks the IR + Custom dist, renders a fused
  multi-RV chain shader, compiles it to SPIR-V (cached), and
  returns the meta tuple `Tree.do_dispatch/10` consumes.

  Pipeline:

      IR → extract_components/1   (priors + custom + layout)
         → render_template/2      (R2.2 — GLSL template fill;
                                    PLACEHOLDER until R2.2 lands)
         → Compile.compile_glsl/1 (R2.4 — glslang + content-addressed cache)
         → Push.build/2           (R2.3 — push-constants spec)
         → return {:ok, {:synthesised, sha, layout, push_spec, spv_path}}

  Today this returns `:unsupported` because `render_template/2`
  is the R2.2 stub.  Every OTHER pipeline stage is wired and
  exercised by `synthesise_with_template_glsl/2` (used by tests
  passing a hand-written shader).
  """
  @spec synthesise(IR.t()) :: {:ok, synth_meta()} | :unsupported | {:unsupported, :push_too_large}
  def synthesise(%IR{} = ir) do
    with {:ok, components} <- extract_components(ir),
         {:ok, glsl} <- render_template(components, ir) do
      synthesise_with_template_glsl(components, glsl, ir)
    else
      _ -> :unsupported
    end
  end

  @doc """
  Task #154 Phase 3 — synthesise the BATCHED multi-instance shader.

  Same return shape as `synthesise/1` but uses
  `MultiRvCustomSpec.render_batched/1` to emit the per-instance-offset
  variant. The returned `obs_bin` is empty — each instance brings its
  own obs at dispatch time via `Dispatch.chain_batch/4`.

  Returns `{:ok, {:synthesised, sha, layout, push_spec, spv_path, <<>>}}`.
  """
  @spec synthesise_batched(IR.t()) :: {:ok, synth_meta()} | :unsupported | {:unsupported, :push_too_large}
  def synthesise_batched(%IR{} = ir) do
    with {:ok, components} <- extract_components(ir),
         {:ok, glsl} <- Exmc.NUTS.CustomSynth.MultiRvCustomSpec.render_batched(components) do
      n_obs =
        case ir.data do
          %Nx.Tensor{shape: {n}} -> n
          _ -> 1
        end

      push_spec =
        Exmc.NUTS.CustomSynth.Push.build(components, K: 32, eps: 0.05, n_obs: n_obs)

      # Same 128-byte push-constants cap as synthesise_with_template_glsl —
      # reject (signalling :push_too_large) rather than crash at dispatch.
      case Exmc.NUTS.CustomSynth.Push.pack(push_spec) do
        {:error, :push_too_large} ->
          {:unsupported, :push_too_large}

        {:ok, _bin, _n} ->
          with {:ok, spv_path} <- Exmc.NUTS.CustomSynth.Compile.compile_glsl(glsl) do
            sha = :crypto.hash(:sha256, glsl) |> Base.encode16(case: :lower)
            {:ok, {:synthesised, sha, components.layout, push_spec, spv_path, <<>>}}
          else
            _ -> :unsupported
          end
      end
    else
      _ -> :unsupported
    end
  end

  @doc """
  Test/dev entry that takes a pre-rendered GLSL string and runs
  it through the R2.3 + R2.4 wiring (push spec + glslang compile)
  to produce the synthesised meta.  Exists so the rest of the
  pipeline can be exercised end-to-end before R2.2's template
  renderer lands.

  `opts` must include `:K` and `:eps` (the runtime leapfrog
  parameters) and may include `:n_obs` (otherwise inferred from
  the IR's `:data` field when present).
  """
  @spec synthesise_with_template_glsl(map(), binary(), IR.t(), keyword()) ::
          {:ok, synth_meta()} | {:unsupported, :push_too_large} | {:error, term()}
  def synthesise_with_template_glsl(components, glsl, %IR{} = ir, opts \\ []) do
    # Obs data has two sources: `ir.data` (Custom / regime models via
    # Builder.data) and the `observed` list (synth P1 — RVs carrying an
    # {:obs, ...} node). When ir.data is absent, size + pack from the
    # observed entries, concatenating obs binaries in iteration order.
    observed = Map.get(components, :observed, [])

    n_obs =
      Keyword.get_lazy(opts, :n_obs, fn ->
        case ir.data do
          %Nx.Tensor{shape: {n}} -> n
          _ -> observed_n_obs(observed)
        end
      end)

    k = Keyword.get(opts, :K, 32)
    eps = Keyword.get(opts, :eps, 0.05)

    push_spec =
      Exmc.NUTS.CustomSynth.Push.build(components, K: k, eps: eps, n_obs: n_obs)

    # Push-constants block is capped at 128 bytes; under f64 (8 B/float)
    # only ~14 prior floats fit after the 16 B header, so higher-dimensional
    # models overflow. Reject synthesis here — the model is valid, just too
    # wide for the fused chain path — and signal the reason so the Plan B'
    # guard degrades to per-op sampling instead of crashing later at
    # Dispatch's `{:ok, _} = Push.pack(...)`.
    case Exmc.NUTS.CustomSynth.Push.pack(push_spec) do
      {:error, :push_too_large} ->
        {:unsupported, :push_too_large}

      {:ok, _bin, _n} ->
        obs_bin =
          case ir.data do
            %Nx.Tensor{} = t ->
              t |> Nx.as_type(:f64) |> Nx.to_binary()

            _ ->
              observed_obs_bin(observed)
          end

        with {:ok, spv_path} <- Exmc.NUTS.CustomSynth.Compile.compile_glsl(glsl) do
          sha = :crypto.hash(:sha256, glsl) |> Base.encode16(case: :lower)
          {:ok, {:synthesised, sha, components.layout, push_spec, spv_path, obs_bin}}
        end
    end
  end

  # R2.2.0 wires this to MultiRvCustomSpec.render, which covers
  # prior-only models (Custom likelihood contributes 0.0). Obs-axis
  # parallelism + Defn-emitted custom likelihood bodies land in R2.2.1
  # — at that point this delegate either grows a second clause or
  # MultiRvCustomSpec.render becomes IR-aware.
  defp render_template(components, _ir) do
    Exmc.NUTS.CustomSynth.MultiRvCustomSpec.render(components)
  end

  @doc false
  # Used by tests + the R1 emitter.  Splits the IR into:
  #   priors:     list of {name, dist_mod, params_map} for the
  #               standard-family RVs
  #   custom:     {name, %Exmc.Dist.Custom{}, params_map}
  #   layout:     ordered list of free-RV names matching the
  #               position vector q's component order
  @spec extract_components(IR.t()) ::
          {:ok,
           %{priors: list(), observed: list(), custom: tuple() | nil, layout: [atom()]}}
          | {:error, atom()}
  def extract_components(%IR{nodes: nodes}) do
    # An RV `rv_id` is *observed* when some node carries
    # `{:obs, rv_id, value, meta}`.  Observed RVs are not free
    # parameters — they contribute a likelihood term but stay out
    # of the sampled position vector (layout).
    observed_ids =
      nodes
      |> Enum.flat_map(fn
        {_id, %{op: {:obs, rv_id, value, meta}}} -> [{rv_id, {value, meta}}]
        _ -> []
      end)
      |> Map.new()

    {observed_rvs, latent_rvs} =
      nodes
      |> Enum.filter(&standard_rv_node?/1)
      |> Enum.split_with(fn {id, _} -> Map.has_key?(observed_ids, id) end)

    priors = latent_rvs

    observed =
      Enum.map(observed_rvs, fn {id, node} ->
        {mod, params} = rv_mod_params(node)
        {value, meta} = Map.fetch!(observed_ids, id)
        {id, mod, params, value, meta}
      end)

    customs =
      Enum.filter(nodes, fn
        {_id, %{op: {:rv, Exmc.Dist.Custom, _}}} -> true
        _ -> false
      end)

    case customs do
      [{_id, %{op: {:rv, Exmc.Dist.Custom, custom_params}} = node}] ->
        custom_struct = Map.get(custom_params, :__dist__)

        cond do
          is_nil(custom_struct) ->
            {:error, :custom_missing_dist_struct}

          # No latent (free) RVs → layout would be empty, which
          # produces d=0 and vulkano panics uploading a zero-byte
          # chain buffer. `compose_custom_term/5` hardcodes
          # `Nx.tensor(0.0)` as the x argument to the user's logpdf,
          # so the Custom RV is not a free parameter to sample; and
          # observed RVs are likewise not free. That semantic means a
          # model with no latent priors has no free RVs — Plan-B'
          # catches this at compile time with a clearer error than a
          # Rust panic. Filed against nx_vulkan/248_TODO.md.
          priors == [] ->
            {:error, :no_free_rvs_in_custom_only_model}

          true ->
            {:ok, build_components(priors, observed, {node_id(node), custom_struct, custom_params})}
        end

      [] ->
        # No Custom likelihood.  Two sub-cases now share this branch:
        #   * prior-only IR (observed == []) — Surface A of
        #     PLAN_F64_CHAIN_SHADER routes single-family models here.
        #   * multi-RV with an observed standard-family likelihood
        #     (observed != []) — synth P1.  The observed RVs' logpdfs
        #     are summed into the compose joint at trace time.
        if priors == [] do
          {:error, :no_rvs}
        else
          {:ok, build_components(priors, observed, nil)}
        end

      _ ->
        # More than one Custom node — out of scope for R1.  A real
        # multi-likelihood model needs separate handling.
        {:error, :multiple_custom_nodes}
    end
  end

  defp build_components(priors, observed, custom) do
    %{
      priors:
        Enum.map(priors, fn {id, node} ->
          {mod, params} = rv_mod_params(node)
          {id, mod, params}
        end),
      observed: observed,
      custom: custom,
      # Only latents go in the layout — observed RVs are not sampled.
      layout: Enum.map(priors, fn {id, _} -> id end)
    }
  end

  # A standard-family RV node (not a Custom likelihood). Matches both the
  # bare 3-tuple `{:rv, mod, params}` and the 4-tuple
  # `{:rv, mod, params, transform}` produced by AttachDefaultTransforms
  # for constrained RVs (e.g. `sigma` with a :log transform).
  defp standard_rv_node?({_id, %{op: {:rv, Exmc.Dist.Custom, _}}}), do: false
  defp standard_rv_node?({_id, %{op: {:rv, Exmc.Dist.Custom, _, _}}}), do: false
  defp standard_rv_node?({_id, %{op: {:rv, _mod, _params}}}), do: true
  defp standard_rv_node?({_id, %{op: {:rv, _mod, _params, _transform}}}), do: true
  defp standard_rv_node?(_), do: false

  # Extract {module, params} from an RV node op of either arity. The
  # explicit 4-tuple transform is ignored here — MultiRvCustomSpec derives
  # it from `mod.transform(params)`, matching the 3-tuple prior path.
  defp rv_mod_params(%{op: {:rv, mod, params}}), do: {mod, params}
  defp rv_mod_params(%{op: {:rv, mod, params, _transform}}), do: {mod, params}

  defp node_id(%{id: id}), do: id
  defp node_id(_), do: nil

  # Total observation count across all observed RVs — sums the flat
  # length of each entry's obs value (scalar obs counts as 1).
  defp observed_n_obs([]), do: 0

  defp observed_n_obs(observed) do
    observed |> Enum.map(fn {_id, _mod, _params, value, _meta} -> obs_size(value) end) |> Enum.sum()
  end

  # Concatenate every observed RV's obs value as an f64 binary in
  # iteration order — matches the order compose_logp_defn reads them.
  defp observed_obs_bin([]), do: <<>>

  defp observed_obs_bin(observed) do
    observed
    |> Enum.map(fn {_id, _mod, _params, value, _meta} ->
      value |> Nx.as_type(:f64) |> Nx.to_binary()
    end)
    |> IO.iodata_to_binary()
  end

  defp obs_size(%Nx.Tensor{} = t), do: max(Nx.size(t), 1)
  defp obs_size(_), do: 1
end
