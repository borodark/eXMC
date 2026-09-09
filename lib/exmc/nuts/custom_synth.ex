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
          {:synthesised, sha256 :: binary(), layout :: [atom()], push_spec :: map(),
           spv_path :: Path.t(), obs_bin :: binary(), captures_bin :: binary()}

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
  @spec synthesise(IR.t(), keyword()) ::
          {:ok, synth_meta()} | :unsupported | {:unsupported, :push_too_large}
  def synthesise(%IR{} = ir, opts \\ []) do
    # Rewrite FIRST, or the shader describes different coordinates than the
    # sampler does.
    #
    # Compiler.compile_for_sampling/2 called detect_meta/1 with the caller's raw
    # IR, while do_compile/2 rebinds `ir = Rewrite.apply(ir, opts)` locally
    # before building the PointMap. So for a model with a non-centred
    # parameterisation, `pm` described the NCP'd coordinates (alpha is the
    # standardised z) while the synthesised shader was built from the centred
    # model — two coordinate systems for one q vector, silently.
    #
    # `opts` is threaded rather than defaulted because Rewrite.apply/2 drops the
    # NCP pass entirely on `ncp: false`. Both arms must be handed the same flag
    # or the mismatch simply reverses.
    #
    # Rewriting twice is safe: NonCenteredParameterization merges into
    # ncp_info rather than replacing it, and the other four passes are
    # idempotent (AttachDefaultTransforms and NormalizeObs fall through on
    # 4-tuple ops, PopulateObsMetadata is all Map.put_new).
    ir = Exmc.Rewrite.apply(ir, opts)

    with {:ok, components} <- extract_components(ir),
         {:ok, glsl, captures_bin} <- render_template(components, ir) do
      synthesise_with_template_glsl(components, glsl, ir, captures_bin: captures_bin)
    else
      _ -> :unsupported
    end
  end

  @doc """
  Task #154 Phase 3 — synthesise the BATCHED multi-instance shader.

  Same return shape as `synthesise/1` but uses
  `MultiRvCustomSpec.render_batched/1` to emit the per-instance-offset
  variant. The returned `obs_bin` is empty — each instance brings its
  own obs at dispatch time via `Dispatch.chain_batch/5`.

  `opts` is threaded into `Exmc.Rewrite.apply/2` for the same reason
  `synthesise/1` threads it: the two arms must be handed the same `:ncp`
  flag or the shader and the PointMap describe different coordinates.

  Every model class `synthesise/1` accepts is accepted here — prior-only,
  observed, and Custom alike. It used to refuse anything with `custom: nil`,
  which is every conjugate model built through `Builder.obs` and so the
  entire class the batch coordinator exists to serve.

  Returns `{:ok, {:synthesised, sha, layout, push_spec, spv_path, <<>>, <<>>}}`.
  """
  @spec synthesise_batched(IR.t(), keyword()) ::
          {:ok, synth_meta()} | :unsupported | {:unsupported, :push_too_large}
  def synthesise_batched(%IR{} = ir, opts \\ []) do
    # Rewrite FIRST, and with the caller's opts, for exactly the reason
    # synthesise/1 documents: skipping it gives the shader one coordinate
    # system and the sampler another. This path used to skip it, so a
    # non-centred model would have batched a shader built from the centred
    # parameterisation while the PointMap described the standardised one.
    ir = Exmc.Rewrite.apply(ir, opts)

    with {:ok, components} <- extract_components(ir),
         {:ok, glsl} <- Exmc.NUTS.CustomSynth.MultiRvCustomSpec.render_batched(components) do
      # n_obs sizes the batched shader's per-instance stride:
      # `extras_off = inst * (n_obs + d)`. Getting it wrong does not fail —
      # it silently points every instance past index 0 at another instance's
      # observations and inverse mass. This used to default to 1 whenever
      # `ir.data` was absent, which is every `Builder.obs` model, so a
      # two-observation conjugate model would have had instance 1 reading
      # from the middle of instance 0's slice.
      n_obs =
        case ir.data do
          %Nx.Tensor{shape: {n}} -> n
          _ -> observed_n_obs(Map.get(components, :observed, []))
        end

      push_spec =
        Exmc.NUTS.CustomSynth.Push.build(components, K: 32, eps: 0.05, n_obs: n_obs)

      # `Push.pack/1` cannot fail: it emits the fixed 24-byte header, and the
      # prior floats it used to append are baked into `glsl` as literals. This
      # used to branch on {:error, :push_too_large} and reject models wider
      # than ~6 free Normal RVs. See Exmc.NUTS.CustomSynth.Push.
      {:ok, _bin, _n} = Exmc.NUTS.CustomSynth.Push.pack(push_spec)

      # The real width bound, now that the push tail is gone: the shader is a
      # single workgroup with `local_size_x = 256` and a `q_shared[256]` tile,
      # one thread per free RV. Refuse here so a too-wide model degrades to
      # per-op sampling, the way an over-wide push block used to — rather than
      # reaching Dispatch's `d <= 256` guard, which has no other clause and
      # would raise FunctionClauseError naming nothing useful.
      if length(components.layout) > 256 do
        {:unsupported, :d_exceeds_tile}
      else
        with {:ok, spv_path} <- Exmc.NUTS.CustomSynth.Compile.compile_glsl(glsl) do
          sha = :crypto.hash(:sha256, glsl) |> Base.encode16(case: :lower)
          # Batched synthesis refuses any non-empty capture set in
          # `capture_guard/3`, so this arm can only ever carry an empty one.
          {:ok, {:synthesised, sha, components.layout, push_spec, spv_path, <<>>, <<>>}}
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
    # The packed closure-capture region of the extras buffer, produced by the
    # same emitter pass that assigned its offsets. Defaults to empty so the
    # hand-written-GLSL callers (tests, the R1 emitter) are unaffected.
    captures_bin = Keyword.get(opts, :captures_bin, <<>>)

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

    # A capture that cannot be read is a likelihood that silently vanishes.
    #
    # `Glsl.register_capture/1` always emits `obs_inv_mass[pc.n_obs + pc.d +
    # <off> + j]`, and `j` exists ONLY inside a `/*REDUCE_SUM*/` loop bounded
    # by `pc.n_obs`. So captures with `n_obs == 0` are provably unreachable:
    # the loop runs zero times, every captured vector is ignored, and the
    # model reports a log-density with no likelihood term at all -- sampling
    # the prior while claiming to sample the posterior.
    #
    # Found 2026-09-05 on the poker model. It carries its observations as
    # closure captures rather than as observed nodes or `Builder.data/2`, so
    # `n_obs` is sized to 0 while four captures of 100 elements each sit in
    # the extras buffer at offsets 0/100/200/300. Before the likelihood was
    # rewritten to rank-1 the emitter refused it outright over `:stack` and
    # the model fell back to the host -- slow, but right. Making it emittable
    # turned a slow-and-correct model into a fast-and-wrong one, which is the
    # worse trade.
    #
    # Refusing sends it back to the host exactly as before. The real fix is to
    # bound each REDUCE_SUM marker by the length of the vectors it actually
    # reduces rather than by the observation-buffer size -- `obs_spans/1`
    # already does per-marker ranges for multiple observed nodes, so the
    # machinery exists. Recorded in docs/SHADER_CONSTANT_INLINING.md.
    # Narrowed once MultiRvCustomSpec learned to bound a marker by the
    # captures it reads: a capture-driven reduction with n_obs == 0 is now
    # CORRECT and must not be refused. What remains indefensible is a loop
    # still bounded by `pc.n_obs` when there are no observations to iterate --
    # that reduction is provably empty and its term silently vanishes.
    if n_obs == 0 and String.contains?(glsl, "j < pc.n_obs") do
      {:unsupported, :empty_obs_axis_reduction}
    else
      # SIBLING GUARD, same defect class from the other direction: a loop that
      # READS the observation region when there are no observations behind it.
      #
      # `compose_custom_term/3` hands the Custom likelihood the traced `obs`
      # parameter, so a closure written `fn observed, params -> ...` emits
      # `double obs_j = obs_inv_mass[j];` and uses it. But a Custom-likelihood
      # model has an EMPTY obs buffer: `observed_obs_bin/1` walks only
      # standard-family observed nodes, and `Builder.obs(ir, "Y_obs", "Y", y)`
      # against a Custom RV contributes none. The extras buffer is then
      # `inv_mass | captures` with nothing at index j, so those reads land in
      # the inv-mass and capture regions and the residual is computed against
      # garbage.
      #
      # MEASURED on the conjugate oracle, whose fixture reads its first
      # argument: obs_bin = 0 doubles, captures = 80, loop `j < 40u`, and the
      # sampled chain came out FROZEN (sd exactly 0.0) where the host path
      # gives 0.045. The GLSL itself was correct; the buffer under it was not.
      #
      # This only became reachable when vector RVs and `dot` started
      # synthesising models that had previously been refused earlier and fell
      # back to the host. Refusing here puts them back on that path -- slower
      # and right -- rather than shipping a fast wrong answer.
      #
      # The real fix is to populate the obs buffer from the Custom RV's own
      # observed value, which also needs obs_size/1, the spans and the push
      # `n_obs` to agree. That is a larger change than this guard and is not
      # attempted here.
      #
      # Models that CAPTURE their data (the convention in this repo's fixtures
      # and throughout benchmark/posteriordb, written `fn _x, params ->`) emit
      # no such read and are unaffected.
      # Checks for a USE of obs_j, not for the read itself. Every reduce loop
      # emits `double obs_j = obs_inv_mass[...];` unconditionally whether the
      # body needs it or not, so testing for that substring flagged
      # capture-style models too -- it fired on this file's own passing
      # fixtures the first time. Strip the declarations, then look.
      obs_axis_used? =
        glsl
        |> String.replace(~r/double obs_j = obs_inv_mass\[[^\]]*\];/, "")
        |> String.contains?("obs_j")

      if n_obs == 0 and obs_axis_used? do
        {:unsupported, :custom_reads_empty_obs_axis}
      else
    k = Keyword.get(opts, :K, 32)
    eps = Keyword.get(opts, :eps, 0.05)

    push_spec =
      Exmc.NUTS.CustomSynth.Push.build(components, K: k, eps: eps, n_obs: n_obs)

    # No width rejection here any more. The push block is the fixed 24-byte
    # header; prior parameters reach the shader baked in as literals. This
    # used to reject models past ~14 prior floats and degrade them to per-op
    # sampling, which cost an 8-RV model 13.1x. See Exmc.NUTS.CustomSynth.Push.
    {:ok, _bin, _n} = Exmc.NUTS.CustomSynth.Push.pack(push_spec)

    obs_bin =
      case ir.data do
        %Nx.Tensor{} = t ->
          t |> Nx.as_type(:f64) |> Nx.to_binary()

        _ ->
          observed_obs_bin(observed)
      end

    if length(components.layout) > 256 do
      # See the sibling guard above: 256 is the shader's thread tile.
      {:unsupported, :d_exceeds_tile}
    else
      with {:ok, spv_path} <- Exmc.NUTS.CustomSynth.Compile.compile_glsl(glsl) do
        sha = :crypto.hash(:sha256, glsl) |> Base.encode16(case: :lower)
        {:ok, {:synthesised, sha, components.layout, push_spec, spv_path, obs_bin, captures_bin}}
      end
      end
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
          {:ok, %{priors: list(), observed: list(), custom: tuple() | nil, layout: [atom()]}}
          | {:error, atom()}
  def extract_components(%IR{nodes: nodes} = ir) do
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

    # Sorted by id, because `layout` IS the q-vector order and it has to agree
    # with Exmc.PointMap.build/1, which sorts free RVs by id and assigns offsets
    # in that order.
    #
    # This relied on `Map` iteration order, which coincides with sorted order
    # only while the node map is a :flatmap — under 32 keys. Past that Erlang
    # switches to a hashmap and the two silently permute, so a model with more
    # than 32 RVs would have had the shader read q in one order and the sampler
    # write it in another. No error, just a wrong posterior.
    #
    # Note this is NOT dependency order. Reference resolution needs its own
    # evaluation order, computed separately; reordering `layout` to match it
    # would be the same bug from the other direction.
    {observed_rvs, latent_rvs} =
      nodes
      |> Enum.filter(&standard_rv_node?/1)
      |> Enum.sort_by(fn {id, _} -> id end)
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
            {:ok,
             build_components(ir, priors, observed, {node_id(node), custom_struct, custom_params})}
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
          {:ok, build_components(ir, priors, observed, nil)}
        end

      _ ->
        # More than one Custom node — out of scope for R1.  A real
        # multi-likelihood model needs separate handling.
        {:error, :multiple_custom_nodes}
    end
  end

  defp build_components(ir, priors, observed, custom) do
    slots = build_slots(ir)

    %{
      priors:
        Enum.map(priors, fn {id, node} ->
          {mod, params} = rv_mod_params(node)
          {id, mod, params}
        end),
      observed: observed,
      custom: custom,
      # ONE ENTRY PER q SLOT, not per RV name.
      #
      # `layout` is the q-vector order and `d = length(layout)` is the shader's
      # thread count, so a `shape: {2}` RV must contribute two entries. It used
      # to contribute one: the trace template was
      # `Nx.template({length(layout)}, :f64)`, so a vector RV arrived at the
      # closure as a SCALAR and `p.beta[0]` raised "cannot use the tensor[index]
      # syntax on scalar tensor" — while `Exmc.PointMap` had correctly given it
      # two slots. Two coordinate systems for one q vector, which is the same
      # class of defect as the NCP/centred mismatch fixed in compiler.ex.
      #
      # Names are `id` for a scalar RV and `id[k]` for element k of a vector
      # one. They are diagnostic labels; the AUTHORITY is `slots`.
      layout: Enum.flat_map(slots, &slot_names/1),
      slots: slots
    }
  end

  # Derived from `Exmc.PointMap.build/1` rather than recomputed here, and that
  # is the point: the synth path and the host sampler must agree on where each
  # RV lives in q, and agreement by construction beats agreement by two
  # implementations of the same rule. PointMap also resolves the UNCONSTRAINED
  # length and shape (a `:stick_breaking` RV has a different unconstrained size
  # than constrained), which a local reimplementation would have to mirror.
  #
  # Restricted to ids this component set actually samples: PointMap is built
  # from the whole IR, and an observed RV is not a free parameter.
  defp build_slots(%IR{} = ir) do
    ir
    |> Exmc.PointMap.build()
    |> Map.fetch!(:entries)
    |> Enum.map(fn e ->
      %{id: e.id, offset: e.offset, length: e.length, shape: e.shape}
    end)
  end

  defp slot_names(%{id: id, length: 1}), do: [id]
  defp slot_names(%{id: id, length: n}), do: Enum.map(0..(n - 1), &"#{id}[#{&1}]")

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
    observed
    |> Enum.map(fn {_id, _mod, _params, value, _meta} -> obs_size(value) end)
    |> Enum.sum()
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
