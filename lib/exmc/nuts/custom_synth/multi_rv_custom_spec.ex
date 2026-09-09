defmodule Exmc.NUTS.CustomSynth.MultiRvCustomSpec do
  @moduledoc """
  R2.2 of `specs/vulkan-custom-synthesis.md` — GLSL template for
  multi-RV chain shaders with a Custom likelihood hook.

  ## R2.2.0 (this commit): prior-only path

  The R2.2.0 deliverable is the shader-design half of R2.2: a
  template that renders to valid GLSL (passes `glslangValidator`)
  for any multi-RV regime model whose Custom likelihood
  contributes zero. The leapfrog math is correct for the
  prior-only case; the Custom hook is `{{likelihood_logp_body}}` /
  `{{likelihood_grad_body}}` placeholders defaulted to `0.0`.

  Thread layout for R2.2.0: thread `tid` holds free RV index
  `tid` (for `tid < d`); threads `tid >= d` carry `lp_i = 0`
  through the workgroup reduction. This is the same pattern as
  the existing `FamilySpec` template, generalised so that
  per-thread gradient/log-prob bodies dispatch on `tid` to the
  right prior's analytical formula.

  ## R2.2.1 (next): obs-axis parallelism + Defn-emitted bodies

  Design A from the spec — `local_size_x = 256` becomes
  `n_obs`-axis parallelism, `q_shared[D_MAX]` broadcasts to all
  threads, per-RV gradient reductions, custom likelihood
  fragments emitted by `CustomSynth.Glsl` from
  `Nx.Defn.value_and_grad`. Lands when the regime model needs
  observation-conditioned sampling.

  ## Template contract

  Push block is fixed: `K, n_obs, d, _pad, eps` (20 bytes header).
  Per-prior scalars are inlined as GLSL constants because
  `emit_prior_fragments/1` traces `mod.logpdf(x, params)` with
  the concrete `params`; the cache hash captures them so different
  priors produce different SPV.

  Holes filled by `render/1`:

    * `{{prior_grad_body_q}}` / `{{prior_grad_body_qn}}` —
      `if (tid == N) { grad_q = <emitted>; }` chain dispatching
      to the emitter-derived gradient per free RV. The `_qn`
      variant is the same body assigning to `grad_qn` instead;
      the template rebinds `qi = qn` before the second
      half-step so the same fragment evaluates at the new
      position.
    * `{{prior_logp_body_qn}}` — `if (tid == N) { lp_i = <emitted>; }`
      chain; emitter-derived log-density per prior. Rendered into
      the POST-update block, beside `{{prior_grad_body_qn}}`, so
      that `logp_chain[k]` describes the same state as
      `q_chain[k]`. It used to sit in the pre-update block, which
      made every returned density lag its position by one step.

  Numerical equivalence vs `mod.logpdf` is validated in
  `multi_rv_custom_spec_test.exs` under the R2.2.4 describe.
  """

  @template ~S"""
  #version 450
  #extension GL_ARB_gpu_shader_fp64 : require

  // SYNTHESIZED by Exmc.NUTS.CustomSynth.MultiRvCustomSpec (f64)
  // Same as @template but with double-precision buffers and arithmetic.

  layout (local_size_x = 256) in;

  layout (push_constant) uniform Push {
      uint   K;
      uint   n_obs;
      uint   d;
      uint   _pad;
      double eps;
  } pc;

  layout (std430, binding = 0) readonly  buffer In_q     { double q_init[]; };
  layout (std430, binding = 1) readonly  buffer In_p     { double p_init[]; };
  layout (std430, binding = 2) readonly  buffer In_extras {
      double obs_inv_mass[];
  };
  layout (std430, binding = 3) writeonly buffer Out_q    { double q_chain[]; };
  layout (std430, binding = 4) writeonly buffer Out_p    { double p_chain[]; };
  layout (std430, binding = 5) writeonly buffer Out_grad { double grad_chain[]; };
  layout (std430, binding = 6) writeonly buffer Out_logp { double logp_chain[]; };

  {{captured_decls}}

  shared double partial[256];
  shared double q_shared[256];

  void main() {
      uint tid = gl_LocalInvocationIndex;
      bool in_bounds = (tid < pc.d);

      double qi = in_bounds ? q_init[tid]                       : 0.0lf;
      double pi = in_bounds ? p_init[tid]                       : 0.0lf;
      double mi = in_bounds ? obs_inv_mass[pc.n_obs + tid]      : 0.0lf;

      for (uint k = 0u; k < pc.K; k++) {
          if (in_bounds) q_shared[tid] = qi;
          barrier();

          double grad_q = 0.0lf;
          if (in_bounds) {
  {{prior_grad_body_q}}
          }
          double p_half = pi + 0.5lf * pc.eps * grad_q;

          double qn = qi + pc.eps * mi * p_half;
          qi = qn;

          barrier();
          if (in_bounds) q_shared[tid] = qi;
          barrier();

          // INVARIANT: the four outputs of iteration k must describe the SAME
          // state. q_chain[k], p_chain[k] and grad_chain[k] are all post-step, so
          // logp_chain[k] must be too — which is why the log-density body belongs
          // here, below the position update and reading the refreshed q_shared,
          // and not up beside prior_grad_body_q.
          //
          // It used to be up there. Tree.synth_chain_subtree/10 pairs the four by
          // index, so every NUTS leaf carried log p(q_chain[k-1]) — a stale,
          // systematically-too-high density. The multinomial then over-weighted
          // the far end of each trajectory. Mis-scaled rather than mis-signed, so
          // it produced no divergences and adaptation simply pushed eps up.
          //
          // This read as "Ampere over-dispersion" for three weeks (Normal(0,1)
          // posterior variance 8.55 against a CPU reference's 1.45) and was
          // blamed on the GPU. It is not hardware: both Keplers and the Ampere
          // produce bit-identical q/p/grad from this shader.
          double grad_qn = 0.0lf;
          double lp_i    = 0.0lf;
          if (in_bounds) {
  {{prior_grad_body_qn}}
  {{prior_logp_body_qn}}
          }
          pi = p_half + 0.5lf * pc.eps * grad_qn;

          if (in_bounds) {
              q_chain[k * pc.d + tid]    = qi;
              p_chain[k * pc.d + tid]    = pi;
              grad_chain[k * pc.d + tid] = grad_qn;
          }

          partial[tid] = lp_i;
          barrier();

          for (uint s = 128u; s > 0u; s /= 2u) {
              if (tid < s) partial[tid] += partial[tid + s];
              barrier();
          }

          if (tid == 0u) {
              logp_chain[k] = partial[0];
          }
          barrier();
      }
  }
  """

  @typedoc "Components map produced by `CustomSynth.extract_components/1`."
  @type components :: %{
          :priors => [{atom() | binary(), module(), map()}],
          optional(:observed) => [{atom() | binary(), module(), map(), Nx.Tensor.t(), map()}],
          :custom => any(),
          :layout => [atom() | binary()]
        }

  @doc """
  Build a Defn-traceable function `fn q, obs -> total_log_p end`
  that composes prior log-pdfs (`mod.logpdf/2` for each entry of
  `components.priors`) with the custom likelihood
  (`custom.logpdf_fn.(0.0, params)`).

  R2.2.1 deliverable: lets `Nx.Defn.debug_expr_apply` produce an
  Expr tree for the full log-posterior that `CustomSynth.Glsl`
  can walk into shader fragments.  Numerical correctness vs Defn
  lands in R2.2.2 (Eval validation); template integration in
  R2.2.4.

  Assumptions:
  - Prior params are concrete tensors. Hierarchical priors whose
    params reference other RVs by name are not yet supported.
  - Custom params reference prior RVs by atom or binary key
    matching `components.layout`. The `:__obs_data` key resolves
    to the `obs` argument.
  """
  @spec compose_logp_defn(components()) :: (Nx.t(), Nx.t() -> Nx.t())
  def compose_logp_defn(%{priors: priors, custom: custom_info, layout: layout} = components) do
    # Observed standard-family RVs (synth P1). Each entry is
    # {id, mod, params, value, meta}; `value`/`meta` are unused inside
    # the traced closure (the observation vector arrives via the `obs`
    # argument / binding-2 SSBO, sized by CustomSynth). `params` may
    # reference latent RVs by name, so normalize tensor literals to
    # BinaryBackend just like priors to keep the closure captures
    # inlinable (Nx 0.12 rejects mixed VulkanoBackend + Expr).
    observed =
      components
      |> Map.get(:observed, [])
      |> Enum.map(fn {id, mod, params, value, meta} ->
        {id, mod, normalize_params(params), value, meta}
      end)

    # id -> SLOT ENTRY (offset/length/shape), not id -> index. A `shape: {2}`
    # RV owns two coordinates, and `q[i]` cannot express that. `slots` comes
    # from `Exmc.PointMap.build/1` via `extract_components/1`, so the offsets
    # here are the same ones the host sampler writes.
    #
    # `components.slots` is absent only for a components map hand-built by a
    # test that predates it; fall back to one slot per layout entry, which is
    # exactly the old behaviour for the all-scalar models those tests use.
    slots =
      Map.get(components, :slots) ||
        layout |> Enum.with_index() |> Enum.map(fn {id, i} ->
          %{id: id, offset: i, length: 1, shape: {}}
        end)

    q_index = Map.new(slots, fn slot -> {slot.id, slot} end)

    # Nx 0.12 rejects closures that mix defn Expr with a non-Expr
    # backend tensor. Under the D88 f64 Vulkano default, `params`
    # values are VulkanoBackend; the returned closure below captures
    # them and later runs inside Nx.Defn.debug_expr_apply. Normalize
    # every tensor param to BinaryBackend so closure captures are
    # inlinable constants that Nx.Defn traces cleanly.
    priors =
      Enum.map(priors, fn {id, mod, params} ->
        {id, mod, normalize_params(params)}
      end)

    # Task #153 (Option A, 2026-05-26): pre-compute per-RV transforms once
    # so both the prior reduce AND custom-likelihood param resolution can
    # access them. Transform.apply(nil, z) = z and log_abs_det_jacobian(nil, _) = 0,
    # so untransformed RVs are no-ops; we apply unconditionally for uniformity.
    # Matches Compiler.node_term/4's transform + Jacobian treatment, eliminating
    # the structural divergence the Task #150 guard currently routes around.
    transforms_by_id =
      Map.new(priors, fn {id, mod, params} -> {id, mod.transform(params)} end)

    # `ncp_info` mirrors `IR.ncp_info`: an NCP'd RV is sampled as a standard
    # normal `z` and reconstructed as `mu + sigma * z` wherever another node
    # names it. Absent for models the NCP pass did not touch.
    ncp_info = Map.get(components, :ncp_info) || %{}

    # Dependency order for reference resolution, computed once at build time —
    # an NCP source may itself be a reference, so resolution is recursive and
    # the order has to be settled before tracing. Cycles raise here rather than
    # hanging inside the traced closure.
    #
    # NOT the layout order. `layout` is the q-vector order and must keep
    # matching Exmc.PointMap; this is purely an evaluation order.
    # RV IDS, not layout entries. `layout` is one name per q slot since vector
    # RVs (`beta[0]`, `beta[1]`), and reference resolution is per RV: an NCP
    # edge names an RV, and `resolve_rv_values/5` looks each id up in
    # `q_index`. Feeding it layout names raised
    # `KeyError: key "beta[0]" not found`.
    order = resolution_order!(Enum.map(slots, & &1.id), ncp_info)

    # Every reference must name a sampled coordinate, checked at build time so
    # an unresolvable one is a clear error at synthesis rather than a
    # Map.fetch! deep inside a Defn trace.
    validate_refs!(priors, observed, custom_info, q_index)

    fn q, obs ->
      # id -> CONSTRAINED value of every sampled RV. Ports
      # Compiler.resolve_ref/4 to slices of the position vector.
      resolved_rvs = resolve_rv_values(order, q, q_index, transforms_by_id, ncp_info)

      prior_lp =
        Enum.reduce(priors, Nx.tensor(0.0), fn {id, mod, params}, acc ->
          slot = Map.fetch!(q_index, id)
          transform = Map.fetch!(transforms_by_id, id)
          # An RV's own density is always evaluated at its OWN coordinate,
          # never at the NCP reconstruction: after the rewrite its params are
          # N(0,1), so this is the z-density, and the reconstruction appears
          # only where other nodes refer to it.
          # This is the line the hierarchical case turned on. The params map
          # went to logpdf raw, so `%{mu: "mu"}` handed Nx a BitString and the
          # trace died before any GLSL existed.
          resolved = resolve_params(params, obs, resolved_rvs)
          # UNROLLED over the RV's coordinates in Elixir, one scalar term per
          # slot, rather than evaluated on the vector and summed.
          #
          # The host writes `Nx.sum(dist.logpdf(vm[id], resolved))`
          # (Compiler.node_term/4) and that is right for it. Here the sum would
          # be TRACED, and the emitter turns a `sum` node into a
          # `/*REDUCE_SUM*/` marker — a loop over the OBSERVATION axis. A
          # parameter-axis sum emitted as an obs-axis loop is not slow, it is
          # wrong: it multiplies the prior by n_obs.
          #
          # An earlier attempt applied `Nx.sum` unconditionally and the shader
          # goldens caught it — q, p and grad stayed byte-identical while logp
          # moved, and only for n_obs > 1, because the gradient trace
          # simplifies the redundant sum away and the forward one does not.
          #
          # `slot.length` is known when the closure is BUILT, so the unroll is
          # an ordinary Elixir loop and every term reaching the emitter is
          # scalar. For length 1 this is exactly the old single-term code.
          Enum.reduce(0..(slot.length - 1), acc, fn k, inner_acc ->
            z_k = q[slot.offset + k]
            x_k = Exmc.Transform.apply(transform, z_k)
            logp = mod.logpdf(x_k, resolved)
            jac = Exmc.Transform.log_abs_det_jacobian(transform, z_k)
            Nx.add(inner_acc, Nx.add(logp, jac))
          end)
        end)

      observed_lp =
        Enum.reduce(observed, Nx.tensor(0.0), fn {_id, mod, params, value, _meta}, acc ->
          resolved = resolve_params(params, obs, resolved_rvs)
          Nx.add(acc, observed_logp(mod, value, obs, resolved))
        end)

      custom_lp = compose_custom_term(custom_info, obs, resolved_rvs)

      Nx.add(prior_lp, Nx.add(observed_lp, custom_lp))
    end
  end

  defp compose_custom_term(nil, _obs, _resolved_rvs), do: Nx.tensor(0.0)

  defp compose_custom_term(
         {_id, %Exmc.Dist.Custom{logpdf_fn: logpdf_fn}, params_map},
         obs,
         resolved_rvs
       ) do
    resolved = resolve_params(params_map, obs, resolved_rvs)
    # `obs`, not `Nx.tensor(0.0)`.
    #
    # `Exmc.Compiler` hands a Custom likelihood the real observations as its
    # first argument. This path handed it a constant zero, so the SAME closure
    # meant two different things on the two paths, and a closure written
    # `fn observed, params -> ... Nx.subtract(observed, mu) ...` silently
    # computed its residual against zero.
    #
    # It stayed hidden because the convention in this repo's own fixtures --
    # and in benchmark/posteriordb -- is `fn _x, params ->` with the data
    # CAPTURED, which is insensitive to what the first argument is. Vector-RV
    # support exposed it: models that read the argument had previously been
    # refused at detect and fell back to the host, where it was correct.
    #
    # Caught by test/nuts/conjugate_oracle_test.exs, whose fixture reads the
    # argument. Its composed density was off by 10.88 RELATIVE and its
    # gradient by 74376; with `obs` the gradient matches the host exactly.
    logpdf_fn.(obs, resolved)
  end

  # A SCALAR observation is inlined as a constant; only a vector one reads the
  # shared obs buffer.
  #
  # Two reasons, and the second is the one that bites. The buffer is a single
  # concatenated vector every reader sees in full, so `mod.logpdf(obs, ...)`
  # for one scalar observation folds in every OTHER observation too — the
  # defect the per-node spans exist to correct.
  #
  # But spans are positional: marker i is observed node i. Two observed nodes
  # with identical params produce structurally identical expressions, Nx merges
  # them, and the emitter then sees fewer markers than there are nodes. The
  # 5-parameter hierarchical model does exactly that — y1 and y2 are both
  # `N(alpha, sigma_obs)` — and synthesis failed with
  #
  #     2 REDUCE_SUM marker(s) for 3 observed node(s)
  #
  # Inlining makes each scalar term carry its own distinct constant, so nothing
  # merges, no marker is emitted for it, and obs_spans/1 stops counting it.
  # Mirrors the host's eager_obs_term/3.
  defp observed_logp(mod, %Nx.Tensor{shape: {}} = value, _obs, resolved) do
    mod.logpdf(Nx.backend_copy(value, Nx.BinaryBackend), resolved)
  end

  defp observed_logp(mod, value, _obs, resolved) when is_number(value) do
    mod.logpdf(Nx.tensor(value, backend: Nx.BinaryBackend), resolved)
  end

  defp observed_logp(mod, _value, obs, resolved) do
    # The observation vector is the `obs` argument (binding-2 SSBO). Nx.sum
    # collapses the obs axis into a scalar log-likelihood; the emitter rewrites
    # that sum into the per-tid GLSL reduce loop, the same wire the Custom
    # likelihood uses.
    mod.logpdf(obs, resolved) |> Nx.sum()
  end

  defp scalar_obs?(%Nx.Tensor{shape: {}}), do: true
  defp scalar_obs?(v) when is_number(v), do: true
  defp scalar_obs?(_), do: false

  # --- Reference resolution ------------------------------------------------
  #
  # Port of Exmc.Compiler.resolve_ref/4 + resolve_value/4 + entry_transform/2
  # to the position vector. The host reads a value map; here every RV's value
  # is a slice of `q`, and the transform comes from `mod.transform(params)` —
  # the same source Rewrite.AttachDefaultTransforms uses to fill the PointMap
  # entry the host reads, so the two agree by construction rather than by
  # coincidence.
  #
  # Returns %{id => constrained tensor} for every sampled coordinate.
  defp resolve_rv_values(order, q, q_index, transforms_by_id, ncp_info) do
    Enum.reduce(order, %{}, fn id, acc ->
      z = slot_slice(q, Map.fetch!(q_index, id))

      value =
        case Map.get(ncp_info, id) do
          nil ->
            Exmc.Transform.apply(Map.get(transforms_by_id, id), z)

          %{mu: mu_src, sigma: sigma_src} ->
            # NCP reconstruction. mu/sigma may themselves be references, and
            # `order` guarantees they are already in `acc`.
            mu = ref_value(mu_src, acc)
            sigma = ref_value(sigma_src, acc)
            Nx.add(mu, Nx.multiply(sigma, z))
        end

      Map.put(acc, id, value)
    end)
  end

  # A one-slot RV stays a SCALAR (`q[off]`, not a {1} tensor). That is not
  # cosmetic: every existing model is all-scalar, and slicing them to rank-1
  # would change every emitted expression and every shader hash for no reason.
  defp slot_slice(q, %{offset: off, length: 1, shape: {}}), do: q[off]

  # STACK of per-element scalar indices, not `Nx.slice`. The emitter is
  # scalar-valued -- every `emit/2` returns one GLSL expression -- and a
  # multi-element slice has no representation in it (`{:error,
  # :multi_element_slice}`). `q[off + k]` for each k is d scalar reads the
  # emitter already handles, and `Nx.stack` reassembles them into the shape the
  # user's closure expects.
  defp slot_slice(q, %{offset: off, length: n, shape: shape}) do
    0..(n - 1)
    |> Enum.map(fn k -> q[off + k] end)
    |> Nx.stack()
    |> Nx.reshape(shape)
  end

  defp ref_value(v, resolved_rvs) when is_binary(v), do: Map.fetch!(resolved_rvs, v)
  defp ref_value(%Nx.Tensor{} = t, _resolved_rvs), do: Nx.backend_copy(t, Nx.BinaryBackend)
  defp ref_value(v, _resolved_rvs) when is_number(v), do: Nx.tensor(v)

  # Dependency-ordered layout ids.
  #
  # Only an NCP'd RV depends on other RVs, through its recorded mu/sigma
  # sources; every other RV is a function of its own coordinate alone, so the
  # graph is small. That also means a future value-level dependency which is
  # NOT an NCP edge would be silently unordered — worth knowing before adding
  # one.
  @doc false
  def resolution_order!(layout, ncp_info) do
    ids = MapSet.new(layout)

    {order, _done} =
      Enum.reduce(layout, {[], MapSet.new()}, fn id, acc ->
        visit_ref(id, ncp_info, ids, acc, [])
      end)

    Enum.reverse(order)
  end

  defp visit_ref(id, ncp_info, ids, {order, done} = acc, path) do
    cond do
      MapSet.member?(done, id) ->
        acc

      # `path` is the descent stack, so this is a real cycle check rather than
      # a visit counter that would also trip on a legitimate diamond.
      id in path ->
        raise Exmc.SynthReferenceError,
              "reference cycle in the model's hierarchy: " <>
                Enum.join(Enum.reverse([id | path]), " -> ")

      true ->
        deps =
          case Map.get(ncp_info, id) do
            nil -> []
            %{mu: mu_src, sigma: sigma_src} -> Enum.filter([mu_src, sigma_src], &is_binary/1)
          end

        {order, done} =
          Enum.reduce(deps, {order, done}, fn dep, inner ->
            unless MapSet.member?(ids, dep) do
              raise Exmc.SynthReferenceError,
                    "non-centered parameterization of #{inspect(id)} refers to " <>
                      "#{inspect(dep)}, which is not a sampled coordinate"
            end

            visit_ref(dep, ncp_info, ids, inner, [id | path])
          end)

        {[id | order], MapSet.put(done, id)}
    end
  end

  # Every binary/atom param value is treated as a reference to a sampled RV —
  # that is what resolve_params/3 does with it — so check at build time that
  # each one names a layout entry. A distribution carrying a legitimate atom
  # param (a mode selector, say) would raise here; none currently does.
  defp validate_refs!(priors, observed, custom_info, q_index) do
    param_maps =
      Enum.map(priors, fn {_id, _mod, params} -> params end) ++
        Enum.map(observed, fn {_id, _mod, params, _value, _meta} -> params end) ++
        case custom_info do
          {_id, _struct, params} -> [Map.delete(params, :__dist__)]
          _ -> []
        end

    for params <- param_maps, {key, ref} <- params, key != :__obs_data do
      case ref_key(ref) do
        nil ->
          :ok

        name ->
          unless Map.has_key?(q_index, name) or Map.has_key?(q_index, ref) do
            raise Exmc.SynthReferenceError,
                  "parameter #{inspect(key)} refers to #{inspect(ref)}, which is not a " <>
                    "sampled coordinate (layout: #{inspect(Map.keys(q_index))})"
          end
      end
    end

    :ok
  end

  # The layout key a param value refers to, or nil when it is a literal. Atom
  # ids are rare (most IRs use strings) but supported: an atom matches either
  # its string form or itself.
  defp ref_key(ref) when is_binary(ref), do: ref
  defp ref_key(ref) when is_atom(ref) and ref not in [nil, true, false], do: Atom.to_string(ref)
  defp ref_key(_ref), do: nil

  # Resolve a distribution's params map to concrete traceable tensors:
  #   * `:__obs_data`         → the `obs` argument (binding-2 SSBO)
  #   * binary/atom RV name   → q[layout_index], with the RV's transform
  #                             applied so the logpdf sees the CONSTRAINED
  #                             value (e.g. sigma as positive, not log-sigma)
  #   * literal               → passthrough
  # Shared by the Custom likelihood and observed standard-family logpdfs.
  # Resolve a params map to concrete traceable tensors:
  #
  #   :__obs_data       -> the `obs` argument (binding-2 SSBO)
  #   binary/atom name  -> that RV's CONSTRAINED value, transform applied and
  #                        NCP reconstructed, from `resolved_rvs`
  #   literal           -> passthrough
  #
  # Took (q, q_index, transforms_by_id) and applied the transform itself, which
  # handled a plain reference but not one to an NCP'd RV — those need
  # `mu + sigma * z` with mu/sigma possibly references in turn. Reading a
  # pre-resolved map instead makes the recursive case fall out for free.
  defp resolve_params(params_map, obs, resolved_rvs) do
    params_map
    |> Map.delete(:__dist__)
    |> Enum.into(%{}, fn
      {:__obs_data, _} ->
        {:__obs_data, obs}

      {key, ref} ->
        case ref_key(ref) do
          nil ->
            {key, ref}

          name ->
            case Map.fetch(resolved_rvs, name) do
              {:ok, value} -> {key, value}
              :error -> {key, Map.fetch!(resolved_rvs, ref)}
            end
        end
    end)
  end

  @doc """
  Trace `compose_logp_defn/1`'s function with symbolic templates
  via `Nx.Defn.debug_expr_apply` and return the resulting Expr.

  `n_obs` sizes the `obs` template; pass `0` for prior-only
  models (the obs template is shape `{0}` but unused).
  """
  @spec trace_logp(components(), non_neg_integer()) :: Nx.Tensor.t()
  def trace_logp(components, n_obs) do
    fun = compose_logp_defn(components)
    q_template = Nx.template({length(components.layout)}, :f64)
    obs_template = Nx.template({max(n_obs, 1)}, :f64)
    Nx.Defn.debug_expr_apply(fun, [q_template, obs_template])
  end

  @doc """
  Trace the gradient of `compose_logp_defn/1`'s function w.r.t. q.
  Returns the Expr for the d-vector `∂total_log_p/∂q`.
  """
  @spec trace_grad(components(), non_neg_integer()) :: Nx.Tensor.t()
  def trace_grad(components, n_obs) do
    fun = compose_logp_defn(components)
    q_template = Nx.template({length(components.layout)}, :f64)
    obs_template = Nx.template({max(n_obs, 1)}, :f64)

    grad_fn = fn q, obs -> Nx.Defn.grad(q, fn q -> fun.(q, obs) end) end
    Nx.Defn.debug_expr_apply(grad_fn, [q_template, obs_template])
  end

  @doc """
  Default GLSL layout for emit/emit_vector: parameter 0 is `q`,
  parameter 1 is `obs`. The R1 emitter rewrites `q[i]` slices to
  `q[i]` and `obs[j]` to `obs[j]` GLSL accesses.

  R2.2.4 will swap this for template-aware accessors that match
  the rendered shader's bindings.
  """
  @spec default_emit_layout() :: [binary()]
  def default_emit_layout(), do: ["q", "obs"]

  @doc """
  R2.2.3 — pack `obs` and `inv_mass` into the f64 binary the
  template's binding-2 SSBO consumes.

  Layout: `obs[0..n_obs-1]` followed by `inv_mass[0..d-1]`. The
  shader reads `obs_inv_mass[j]` for `j < n_obs` (observation) and
  `obs_inv_mass[pc.n_obs + i]` for `i < d` (inverse mass).

  Inputs are flat lists of numbers, an Nx tensor, or `nil` (treated
  as an empty list). Output is a binary of length
  `(length(obs) + length(inv_mass)) * 8` bytes.
  """
  @spec pack_input_buffer(input(), input()) :: binary()
  def pack_input_buffer(obs, inv_mass) do
    obs_list = to_flat_list(obs)
    inv_mass_list = to_flat_list(inv_mass)

    for f <- obs_list ++ inv_mass_list, into: <<>>, do: <<f * 1.0::little-float-64>>
  end

  @typep input :: [number()] | Nx.Tensor.t() | nil

  defp to_flat_list(nil), do: []
  defp to_flat_list(list) when is_list(list), do: list

  defp to_flat_list(%Nx.Tensor{} = t) do
    case Nx.shape(t) do
      {} -> [Nx.to_number(t)]
      _ -> Nx.to_flat_list(t)
    end
  end

  @doc """
  Render a `components` map to a GLSL source string.

  Two paths:

  - **Prior-only** (`components.custom == nil`): per-prior bodies
    via `emit_prior_fragments/1`, dispatched per-tid. (R2.2.4.)
  - **With Custom likelihood**: full compose via
    `compose_logp_defn/1` traced + emitted; any `/*REDUCE_SUM*/`
    markers (the obs-axis reduction) are rewritten into serial
    GLSL for-loops over `pc.n_obs` with `obs_j` rebound per
    iteration to `obs_inv_mass[j]`. (R3.)

  Returns `{:error, {:layout_id_not_in_priors, id}}` when the
  layout references an RV the priors list doesn't know about,
  or whatever the emitter returns for an unsupported op.
  """
  @spec render(components()) :: {:ok, binary(), binary()} | {:error, term()}
  def render(%{priors: priors, custom: custom_info, layout: layout} = components) do
    observed = Map.get(components, :observed, [])

    with :ok <- validate_layout(priors, layout) do
      # The compose path (`render_with_custom`) traces priors + observed
      # + custom into one joint via `compose_logp_defn`, so it serves both
      # the Custom likelihood and the observed standard-family shape (synth
      # P1). Only truly prior-only models (no custom, no observed) take the
      # per-tid analytic path.
      if is_nil(custom_info) and observed == [] do
        render_prior_only(priors, layout, @template)
      else
        render_with_custom(components, @template, "j", :single)
      end
    end
  rescue
    # The obs-span attribution guard in do_transform_rs/5. Degrade to an
    # {:error, _} so CustomSynth.synthesise/1 reports :unsupported and the
    # model samples on the host — slower, but not wrong.
    e in RuntimeError ->
      if String.starts_with?(Exception.message(e), "transform_reduce_sum:") do
        {:error, {:obs_span_attribution, Exception.message(e)}}
      else
        reraise e, __STACKTRACE__
      end
  end

  # `template` selects the single-instance or the batched skeleton. The bodies
  # are the same either way — a prior-only model reads no observations, so
  # there is not even an index expression to vary.
  defp render_prior_only(priors, layout, template) do
    with {:ok, fragments} <- emit_prior_fragments(priors) do
      grad_q_body = build_per_tid_dispatch(fragments, layout, "grad_q", :grad)
      grad_qn_body = build_per_tid_dispatch(fragments, layout, "grad_qn", :grad)
      logp_body = build_per_tid_dispatch(fragments, layout, "lp_i", :log_p)

      helpers = f64_transcendental_helpers()

      glsl =
        template
        |> String.replace("{{prior_grad_body_q}}", indent(grad_q_body, 12))
        |> String.replace("{{prior_grad_body_qn}}", indent(grad_qn_body, 12))
        |> String.replace("{{prior_logp_body_qn}}", indent(logp_body, 12))
        |> String.replace("{{captured_decls}}", helpers)
        |> rewrite_transcendentals_f64()

      # A prior-only model reads no observations, so it can hold no closure
      # captures either. The empty binary keeps the arity uniform.
      {:ok, glsl, <<>>}
    end
  end

  # `obs_index` is the expression the emitted reduce-sum loops use to read the
  # observation buffer: `j` for the single-instance shader, `extras_off + j`
  # for the batched one, where each instance owns its own `n_obs + d` slice.
  # It is the ONLY thing that varies in the emitted bodies, and threading it
  # through here rather than duplicating the emitter is what keeps the batched
  # arithmetic identical to the single-instance arithmetic instruction for
  # instruction. A previous copy of this function drifted: it dropped the obs
  # spans, the CSE pass and the f64 transcendental rewrite, and the copy that
  # was left could not compile.
  defp render_with_custom(components, template, obs_index, variant) do
    # Trace the full compose (priors + custom) with an obs template
    # of shape {1}. The actual loop iteration count comes from
    # pc.n_obs at runtime — the trace shape just needs to be > 0 so
    # Defn lowers Nx.sum into a :sum op.
    n_obs_trace = 1
    layout = ["q_shared", "obs_j"]

    fun = compose_logp_defn(components)
    q_template = Nx.template({length(components.layout)}, :f64)
    obs_template = Nx.template({n_obs_trace}, :f64)

    value_expr = Nx.Defn.debug_expr_apply(fun, [q_template, obs_template])

    grad_fn = fn q, obs -> Nx.Defn.grad(q, fn q -> fun.(q, obs) end) end
    grad_expr = Nx.Defn.debug_expr_apply(grad_fn, [q_template, obs_template])

    Exmc.NUTS.CustomSynth.Glsl.start_captures()

    with {:ok, log_p_glsl_raw} <- Exmc.NUTS.CustomSynth.Glsl.emit(value_expr, layout),
         {:ok, grad_entries_raw} <- Exmc.NUTS.CustomSynth.Glsl.emit_vector(grad_expr, layout) do
      captures = Exmc.NUTS.CustomSynth.Glsl.collect_captures()
      spans = obs_spans(components)

      {log_p_loops, log_p_expr} =
        transform_reduce_sum(log_p_glsl_raw, "_lpacc", spans, obs_index, captures)

      grad_by_idx =
        grad_entries_raw
        |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))

      d = length(components.layout)

      {grad_loops_per_tid, grad_expr_per_tid} =
        for i <- 0..(d - 1), into: %{} do
          frags = Map.get(grad_by_idx, i, ["0.0"])

          # Sum multi-put_slice fragments into one expression.
          summed =
            case frags do
              [single] -> single
              many -> "(" <> Enum.join(many, ") + (") <> ")"
            end

          # The gradient's markers arrive MIRRORED relative to the forward
          # log-density's. `compose_logp_defn/1` folds the observed terms left
          # to right, and reverse-mode AD walks that tape backwards, so
          # `_gacc*_0` is the LAST observed node while `_lpacc0` is the first.
          # Confirmed by reading the emitted GLSL: with sigmas 1/2/3 the
          # gradient's first accumulator carries the /3.0 divisor.
          #
          # Positional attribution is therefore per-direction, and
          # bench/leapfrog_leaf_diff.exs pins it with DISTINCT per-node sigmas
          # precisely so a permutation cannot pass unnoticed — with identical
          # observations a mirrored assignment gives bit-identical answers.
          {loops, expr} =
            transform_reduce_sum(summed, "_gacc#{i}_", reverse_spans(spans), obs_index, captures)

          {i, {loops, expr}}
        end
        |> Enum.reduce({%{}, %{}}, fn {i, {loops, expr}}, {loops_acc, expr_acc} ->
          {Map.put(loops_acc, i, loops), Map.put(expr_acc, i, expr)}
        end)

      logp_body =
        build_logp_body_with_loops(
          d,
          log_p_loops,
          log_p_expr
        )

      grad_q_body = build_grad_body_with_loops(d, grad_loops_per_tid, grad_expr_per_tid, "grad_q")

      grad_qn_body =
        build_grad_body_with_loops(d, grad_loops_per_tid, grad_expr_per_tid, "grad_qn")

      glsl =
        template
        |> String.replace("{{prior_grad_body_q}}", indent(grad_q_body, 12))
        |> String.replace("{{prior_grad_body_qn}}", indent(grad_qn_body, 12))
        |> String.replace("{{prior_logp_body_qn}}", indent(logp_body, 12))
        |> String.replace("{{captured_decls}}", f64_transcendental_helpers())
        |> rewrite_transcendentals_f64()

      case capture_guard(captures, spans, variant) do
        :ok -> {:ok, glsl, captures_bin(captures)}
        {:error, _} = err -> err
      end
    else
      err ->
        # Clear capture buffer on emit failure so subsequent renders
        # don't pick up stale tensors from a partially-completed pass.
        Exmc.NUTS.CustomSynth.Glsl.collect_captures()
        err
    end
  end

  # Captures used to be emitted here as `const double name[N] = double[](...)`
  # at file scope, which made SPIR-V size grow with the DATA: a linear
  # regression closes over its design matrix as one rank-1 tensor per
  # predictor, so the shader carried n_obs * n_beta float literals. Past
  # roughly 1300 of them the NVIDIA driver refuses to create the compute
  # pipeline -- 21 of 33 posteriordb models died in `ComputePipeline::new`
  # with "a non-validation error occurred". Synthesised shaders reached
  # 2.15 MB against ~8 KB for a hand-written one.
  #
  # They now live in the extras SSBO instead; `Glsl.capture_accessor/1` has
  # the layout. See docs/SHADER_CONSTANT_INLINING.md.

  # The packed capture region, in the SAME order the emitter assigned offsets.
  # Built from the emitter's own entries and never re-derived from the
  # tensors: a second encoder is how the offsets and the bytes drift apart.
  # `render_batched/1` keeps its 2-tuple contract: `capture_guard/3` has
  # already refused any non-empty capture set on that path, so the only
  # success it can see carries an empty binary.
  defp drop_captures({:ok, glsl, <<>>}), do: {:ok, glsl}
  defp drop_captures({:ok, _glsl, _bin}), do: {:error, :captures_unsupported_batched}
  defp drop_captures(other), do: other

  @doc false
  def captures_bin([]), do: <<>>

  def captures_bin(captures) do
    captures
    |> Enum.sort_by(& &1.offset)
    |> Enum.map(fn %{values: values} ->
      for v <- values, into: <<>>, do: <<v * 1.0::little-float-64>>
    end)
    |> IO.iodata_to_binary()
  end

  # Two shapes we refuse rather than emit a plausible wrong density for.
  defp capture_guard([], _spans, _variant), do: :ok

  # `j` is a GLOBAL index into the concatenated observation buffer even under
  # per-node spans (`loop_lo/2` starts marker n at its global offset). A
  # capture aligned to one node's own slice would therefore be read at the
  # wrong offset. Every capture-bearing model today has a Custom likelihood,
  # and `obs_spans/1` returns `:full` for those, so this combination is not
  # reachable from anything that currently works -- but it is silently wrong
  # if it ever becomes reachable, so refuse it and sample on the host.
  defp capture_guard(captures, spans, _variant) when is_list(spans) do
    {:error, {:captures_with_obs_spans, length(captures), length(spans)}}
  end

  # The batched shader gives each instance its own `n_obs + d` slice of the
  # extras buffer. Captures are shared data, so they belong outside that
  # per-instance region -- an unresolved layout question. Refusing is the
  # honest answer; emitting would bake instance-0's data into a shader every
  # instance shares, which is exactly what this function's predecessor's own
  # comment predicted.
  defp capture_guard(captures, _spans, :batched) do
    {:error, {:captures_unsupported_batched, length(captures)}}
  end

  defp capture_guard(_captures, _spans, :single), do: :ok

  # Render `const float __captured_tN[K] = float[](v0, v1, ...);`
  # declarations from the per-process tensor capture buffer. Returns
  # an empty string when no captures occurred (the typical case for
  # prior-only models and the eight-schools NCP). Emitted at file
  # scope between the SSBO bindings and the `shared` declarations.
  #
  # Inline `const float[]` is correct for the synth-coverage probe
  # and for closures over fixed reference data. Per-instance batched
  # dispatch requires the obs tensors to move to SSBO bindings via
  # `Builder.data/2` at the model-builder layer — closures over
  # rank-1 obs bake instance-0's data into the shader.
  defp build_captured_decls([]), do: ""

  defp build_captured_decls(captures) do
    Enum.map_join(captures, "\n", fn %{name: name, values: values, length: n} ->
      literals =
        values
        |> Enum.map(&render_float_for_glsl/1)
        |> Enum.join(", ")

      "const double #{name}[#{n}] = double[](#{literals});"
    end)
  end

  defp render_float_for_glsl(n) when is_float(n) do
    s = Float.to_string(n)

    cond do
      s == "Inf" -> "(1.0 / 0.0)"
      s == "-Inf" -> "(-1.0 / 0.0)"
      s == "NaN" -> "(0.0 / 0.0)"
      String.contains?(s, ".") or String.contains?(s, "e") -> s
      true -> s <> ".0"
    end
  end

  defp render_float_for_glsl(n) when is_integer(n), do: "#{n}.0"

  # GLSL.std.450 defines log/exp only for 16- or 32-bit floats — the
  # SPIR-V spec explicitly lists these instructions as float-only. So
  # every "f64" chain shader must synthesise its own double-precision
  # transcendentals somehow. Two modes:
  #
  #   :f32_cast (default) — cheapest, ~1 ULP of f32 relative error
  #     (worst case ~6e-8). Ships since day one. The observed
  #     regime-tolerance drift on Kepler is NOT caused by this
  #     precision loss — that was refuted by
  #     `research/kepler_precision_p1_findings.md`. So this mode
  #     stays default until we have evidence a caller needs better.
  #
  #   :polynomial — real f64 log/exp via range-reduction +
  #     polynomial approximation. ~10-15× shader latency; ~1 ULP f64
  #     accuracy. Enable with
  #       config :exmc, :chain_shader_transcendentals, :polynomial
  #     if a downstream model surfaces where the f32 precision loss
  #     provably matters.
  #
  # The polynomial path is here to make the "clean fix" available,
  # not because we know a caller today needs it.
  defp f64_transcendental_helpers do
    case Application.get_env(:exmc, :chain_shader_transcendentals, :f32_cast) do
      :polynomial -> polynomial_f64_transcendentals()
      _ -> f32_cast_transcendentals()
    end
  end

  defp f32_cast_transcendentals do
    """
    // exp/log/log1p/expm1 boundary-cast helpers. GLSL.std.450 has no
    // double transcendentals, so we cast down to float, apply the
    // built-in, cast back. ~1 ULP of f32 relative error (~6e-8).
    // See f64_transcendental_helpers/0 for the polynomial alternative.
    double exp_d(double x) { return double(exp(float(x))); }
    double log_d(double x) { return double(log(float(x))); }
    double log_1p_safe_d(double x) { return double(log(float(1.0lf + x))); }
    double expm1_safe_d(double x) { return double(exp(float(x))) - 1.0lf; }
    """
  end

  # Cody-Waite-style range reduction + polynomial evaluation for real
  # f64 log/exp with no reliance on GLSL.std.450 built-ins on double
  # operands. Guaranteed ~1 ULP f64.
  #
  # log(x): reduce x = m * 2^e with m ∈ [1, 2), then substitute
  #   z = (m - 1)/(m + 1) so log(m) = 2 * atanh(z) =
  #     2 * (z + z^3/3 + z^5/5 + z^7/7 + z^9/9 + z^11/11).
  #   The final log(x) = log(m) + e * ln(2).
  #
  # exp(x): reduce x = k*ln(2) + r with r ∈ [-ln(2)/2, ln(2)/2],
  #   then exp(x) = 2^k * exp(r) using a Taylor series in r up to
  #   1/10! (12 terms give < 1 ULP for |r| < ln(2)/2).
  #
  # Integer exponent handling uses `int(floor(log2(float(x))))` —
  # this only requires float32 precision because we then correct
  # via `exp2(float(-e))` which is EXACT for integer e in [-1023,
  # 1023] since 2^k is exactly representable in both f32 and f64
  # for that range.
  defp polynomial_f64_transcendentals do
    """
    // Real f64 log/exp via range reduction + polynomial approximation.
    // Opt-in via config :exmc, :chain_shader_transcendentals, :polynomial.
    // ~1 ULP f64, ~10-15× the latency of the f32-cast variant.
    double log_d(double x) {
      if (x <= 0.0lf) return -1.0lf / 0.0lf;
      if (x == 1.0lf) return 0.0lf;
      // Extract binary exponent via f32 log2 — precise enough for
      // integer floor(log2(x)) as long as x is finite and > 0.
      int e = int(floor(log2(float(x))));
      // 2^-e is exact for integer e in the safe range.
      double m = x * double(exp2(float(-e)));
      if (m < 0.7071067811865475lf) { m *= 2.0lf; e -= 1; }
      // z = (m - 1) / (m + 1), log(m) = 2 * atanh(z).
      double z = (m - 1.0lf) / (m + 1.0lf);
      double z2 = z * z;
      // Horner: (((((1/11)*z2 + 1/9)*z2 + 1/7)*z2 + 1/5)*z2 + 1/3)*z2 + 1
      double s = 0.09090909090909091lf;
      s = s * z2 + 0.1111111111111111lf;
      s = s * z2 + 0.14285714285714285lf;
      s = s * z2 + 0.2lf;
      s = s * z2 + 0.3333333333333333lf;
      s = s * z2 + 1.0lf;
      return 2.0lf * z * s + double(e) * 0.6931471805599453094172321lf;
    }

    double exp_d(double x) {
      // Guard against overflow: exp(x) > f64_max at x ≈ 709.
      if (x > 709.0lf) return 1.0lf / 0.0lf;
      if (x < -745.0lf) return 0.0lf;
      // Range reduce: x = k*ln(2) + r, r ∈ [-ln(2)/2, ln(2)/2].
      double k_f = floor(x * 1.4426950408889634lf + 0.5lf);
      int k = int(k_f);
      // Two-part ln(2) split (Cody-Waite) to preserve precision in r.
      double r = (x - k_f * 0.6931471805599453lf) - k_f * 2.319046813846299e-17lf;
      // exp(r) via Taylor series (12 terms, |r| ≤ 0.347).
      double t = r;
      double s = 1.0lf + t;
      t *= r * 0.5lf;                     s += t;
      t *= r * 0.3333333333333333lf;      s += t;
      t *= r * 0.25lf;                    s += t;
      t *= r * 0.2lf;                     s += t;
      t *= r * 0.16666666666666666lf;     s += t;
      t *= r * 0.14285714285714285lf;     s += t;
      t *= r * 0.125lf;                   s += t;
      t *= r * 0.1111111111111111lf;      s += t;
      t *= r * 0.1lf;                     s += t;
      t *= r * 0.09090909090909091lf;     s += t;
      return s * double(exp2(float(k)));
    }

    // log1p(x) uses log_d(1+x) directly at f64 — no precision issue
    // because the 1+x addition is exact when |x| >= 2^-52.
    double log_1p_safe_d(double x) { return log_d(1.0lf + x); }

    // expm1(x) = exp(x) - 1 via log_d — no separate identity needed
    // at f64 precision.
    double expm1_safe_d(double x) { return exp_d(x) - 1.0lf; }
    """
  end

  # Replace f32 transcendentals with boundary-cast f64 helpers in the
  # emitted GLSL body. The helpers (exp_d, log_d, etc.) are injected
  # into {{captured_decls}} by render_with_custom. We must NOT replace
  # calls inside the helper definitions themselves (they're already in
  # the captured_decls block above the main() function), only calls
  # inside main(). Since the helpers are defined before main(), a
  # simple global replace is safe — the helper bodies use float(x)
  # which doesn't match `exp(` patterns.
  defp rewrite_transcendentals_f64(glsl) do
    glsl
    |> String.replace("exp(", "exp_d(")
    |> String.replace("log(", "log_d(")
    |> String.replace("log_1p_safe(", "log_1p_safe_d(")
    |> String.replace("expm1_safe(", "expm1_safe_d(")
    # Fix the helper definitions that got double-rewritten:
    # "double exp_d(double x) { return double(exp_d(float(x))); }"
    # should be "... exp(float(x)) ..."
    |> String.replace("double(exp_d(float(", "double(exp(float(")
    |> String.replace("double(log_d(float(", "double(log(float(")
    |> String.replace("double(log_d(float(1.0lf + x))", "double(log(float(1.0lf + x))")
    |> String.replace("double(exp_d(float(x))) - 1.0lf", "double(exp(float(x))) - 1.0lf")
  end

  # Rewrite `/*REDUCE_SUM*/(<inner>)` markers in `glsl` into serial
  # GLSL for-loops, accumulating into uniquely-named locals (`prefix0`,
  # `prefix1`, ...). Returns `{loops_block, glsl_with_accums_substituted}`.
  #
  # The inner expression references `obs_j` (the second-parameter
  # layout binding); the loop's `double obs_j = obs_inv_mass[...];`
  # binding provides it, indexed by `obs_index` below.
  #
  # `spans` decides what each loop ranges over:
  #
  #   :full          every marker loops `j < pc.n_obs` — the whole observation
  #                  buffer. Correct when ONE observed node owns the buffer
  #                  (a single `Builder.obs` carrying a vector) or when a
  #                  Custom likelihood consumes all of it.
  #
  #   [{off, cnt}]   marker i ranges over `[off, off+cnt)` — its OWN slice of
  #                  the buffer. Required when a model has several observed
  #                  nodes, because `observed_obs_bin/1` concatenates their
  #                  values in iteration order and each node's log-density is
  #                  a function of its own observations only.
  #
  # Without the second form every observed node summed the ENTIRE buffer, so a
  # model with n separate scalar observations counted its whole likelihood n
  # times over — n× the log-density AND n× the gradient. The posterior came out
  # ~sqrt(n) too narrow, which made the adapted step size far too large for it,
  # which collapsed the acceptance rate and froze the chain. Measured at
  # 3 observations: gradient 31.495 against a host/analytic 10.495, diverging
  # from leapfrog step 0. See docs/OPEN_VULKAN_OBSERVED_MODEL.md.
  # `obs_index` is the index expression for the observation buffer read. `"j"`
  # for the single-instance shader; `"extras_off + j"` for the batched one,
  # whose instances each own a contiguous `n_obs + d` slice. The loop BOUNDS
  # stay in per-instance coordinates either way, so spans compose with it
  # unchanged: a span is a property of the model's observed nodes, and the
  # offset is a property of which instance is reading them.
  defp transform_reduce_sum(glsl, prefix, spans \\ :full, obs_index \\ "j", captures \\ []) do
    {markers, glsl} = collect_rs(glsl, prefix, [], 0, spans, obs_index, captures)
    {render_reduce_loops(markers, obs_index), glsl}
  end

  defp reverse_spans(:full), do: :full
  defp reverse_spans(spans) when is_list(spans), do: Enum.reverse(spans)

  # Where each observed node's observations live in the binding-2 buffer.
  # `CustomSynth.observed_obs_bin/1` concatenates the nodes' values in
  # iteration order, and this is the matching read side — the correspondence
  # its own comment claims ("matches the order compose_logp_defn reads them")
  # but which nothing enforced until the spans below.
  #
  # Returns `:full` — every marker sums the whole buffer — in the two cases
  # where that is the correct reading and per-node attribution is not:
  #
  #   * one observed node: it owns the entire buffer by definition, and
  #     keeping the runtime `pc.n_obs` bound means a vector-obs model still
  #     compiles to one SPV regardless of how many data points it carries;
  #   * a Custom likelihood: its term contributes markers of its own that do
  #     not correspond to observed nodes positionally, so the offsets could
  #     not be attributed.
  defp obs_spans(%{custom: custom}) when not is_nil(custom), do: :full

  defp obs_spans(components) do
    observed = Map.get(components, :observed, [])
    vectors = Enum.reject(observed, fn {_id, _m, _p, value, _meta} -> scalar_obs?(value) end)

    cond do
      # Every observation is inlined as a constant, so no REDUCE_SUM marker is
      # emitted and there is nothing to attribute.
      vectors == [] ->
        []

      # A single observed node keeps the runtime bound, so it compiles to the
      # GLSL it always did and stays reusable across dataset sizes.
      length(observed) < 2 ->
        :full

      true ->
        # Offsets are computed over ALL observed nodes, because that is the
        # order pack_input_buffer/2 lays the buffer out in — but spans are
        # emitted only for the vector ones, which are the only nodes that
        # produce a marker.
        observed
        |> Enum.map_reduce(0, fn {_id, _mod, _params, value, _meta}, off ->
          cnt = obs_count(value)
          {{value, off, cnt}, off + cnt}
        end)
        |> elem(0)
        |> Enum.reject(fn {value, _off, _cnt} -> scalar_obs?(value) end)
        |> Enum.map(fn {_value, off, cnt} -> {off, cnt} end)
    end
  end

  # Mirrors CustomSynth.obs_size/1 — a scalar observation still occupies one
  # slot, so an all-scalar model yields spans {0,1}, {1,1}, {2,1}.
  defp obs_count(%Nx.Tensor{} = t), do: max(Nx.size(t), 1)
  defp obs_count(_), do: 1

  # `:full` keeps the runtime bound, so a single-observed-node model compiles to
  # the same GLSL it always did and stays reusable across dataset sizes. A span
  # bakes in constants, which is correct: the offsets ARE a property of the
  # model's observed nodes, not of the data length.
  # A reduce loop must iterate over the vectors it actually reduces.
  #
  # `pc.n_obs` is the length of the OBSERVATION BUFFER, and for a
  # likelihood whose data arrives as closure captures that is not the same
  # number -- it is zero. `Builder.obs(ir, "y", "lik", Nx.tensor(0.0))`
  # contributes a scalar, so the observation axis is empty while the captured
  # vectors are 100 or 1192 long. Every such model synthesised a loop bounded
  # by 0: the likelihood term evaluated to exactly nothing, the shader
  # returned the PRIOR, and the sampler reported it as a posterior. No error,
  # no divergence, a confident wrong answer. Measured 2026-09-05 on
  # posteriordb under `compiler: :vulkan` -- R-hat 7.5, max mean error 37-55,
  # divergence rates of 45-94% against EXLA's 0.13 and 2.3% on the same models
  # and seeds.
  #
  # So the bound comes from the captures the marker reads. Their offsets are
  # already correct (`pc.n_obs + pc.d + off` is `0 + d + off` when the
  # observation region is empty); only the iteration count was wrong. Markers
  # that read no captures keep the observation-buffer bound, which is right
  # for models whose data really does arrive through `ir.data` or observed
  # nodes.
  #
  # Emitted as a literal rather than a push field because the shader is
  # content-addressed per model, exactly as the per-node span bounds already
  # are.
  defp reduce_bounds(inner, captures, spans, n) do
    case capture_reduce_len(inner, captures) do
      nil -> {loop_lo(spans, n), loop_hi(spans, n)}
      len -> {"0u", "#{len}u"}
    end
  end

  defp capture_reduce_len(_inner, []), do: nil

  defp capture_reduce_len(inner, captures) do
    offsets =
      ~r/obs_inv_mass\[pc\.n_obs \+ pc\.d \+ (\d+) \+ j\]/
      |> Regex.scan(inner)
      |> Enum.map(fn [_, off] -> String.to_integer(off) end)
      |> Enum.uniq()

    case offsets do
      [] ->
        nil

      _ ->
        lengths =
          offsets
          |> Enum.map(fn off ->
            Enum.find_value(captures, fn c -> c.offset == off && c.length end)
          end)
          |> Enum.uniq()

        case lengths do
          [len] when is_integer(len) ->
            len

          other ->
            # One reduction cannot range over vectors of two different
            # lengths. Raising degrades to :unsupported and the host, rather
            # than picking one and being wrong on the rest.
            raise "transform_reduce_sum: captures in one marker disagree on length: " <>
                    "#{inspect(other)} at offsets #{inspect(offsets)}"
        end
    end
  end

  defp loop_lo(:full, _n), do: "0u"
  defp loop_lo(spans, n) when is_list(spans), do: "#{elem(Enum.at(spans, n), 0)}u"

  defp loop_hi(:full, _n), do: "pc.n_obs"

  defp loop_hi(spans, n) when is_list(spans) do
    {off, cnt} = Enum.at(spans, n)
    "#{off + cnt}u"
  end

  # Collect every marker as `{bounds, accum, inner}` in emission order, WITHOUT
  # rendering or CSE-ing it. Rendering is deferred to `render_reduce_loops/2`
  # so that markers sharing a trip count can be fused into one loop; CSE is
  # deferred with it, because a per-marker pass cannot see the redundancy
  # BETWEEN markers, which is where nearly all of it lives.
  defp collect_rs(glsl, prefix, acc, n, spans, obs_index, captures) do
    marker = "/*REDUCE_SUM*/("

    case :binary.match(glsl, marker) do
      :nomatch ->
        # A per-node span list is positional: marker i is observed node i. If
        # the emitter ever produces a different number of markers than there
        # are observed nodes that correspondence is broken, and the offsets
        # would silently land on the wrong observations. Raise instead — the
        # caller turns it into `:unsupported`, which costs a slower host path
        # rather than a wrong posterior.
        if is_list(spans) and n != length(spans) do
          raise "transform_reduce_sum: #{n} REDUCE_SUM marker(s) for " <>
                  "#{length(spans)} observed node(s); cannot attribute obs slices"
        end

        {Enum.reverse(acc), glsl}

      {start, marker_len} ->
        open_paren = start + marker_len - 1
        # marker_len includes the trailing '(' so open_paren points to it.

        case find_matching_paren(glsl, open_paren) do
          {:error, _} = e ->
            raise "transform_reduce_sum: unbalanced parens after REDUCE_SUM marker at offset #{open_paren}: #{inspect(e)}"

          close_paren ->
            inner = binary_part(glsl, open_paren + 1, close_paren - open_paren - 1)
            accum = "#{prefix}#{n}"

            # A nested marker was moved into `inner` wholesale and the scan
            # never revisits it, so it would reach glslangValidator as a
            # literal comment-and-paren and fail there with no reference to
            # the model. It is not known whether the emitter can produce
            # nested sums; find out from this raise rather than from a GLSL
            # syntax error.
            if String.contains?(inner, "/*REDUCE_SUM*/") do
              raise "transform_reduce_sum: nested REDUCE_SUM marker inside marker #{n} " <>
                      "(#{accum}); the outer loop body would carry an unexpanded marker"
            end

            # Once, not twice: `reduce_bounds/4` runs a regex scan over the
            # body via `capture_reduce_len/2`, and the old code called it
            # separately for the lo and the hi of one loop header.
            bounds = reduce_bounds(inner, captures, spans, n)

            before = binary_part(glsl, 0, start)
            after_ = binary_part(glsl, close_paren + 1, byte_size(glsl) - close_paren - 1)
            new_glsl = before <> accum <> after_

            collect_rs(
              new_glsl,
              prefix,
              [{bounds, accum, inner} | acc],
              n + 1,
              spans,
              obs_index,
              captures
            )
        end
    end
  end

  # --- Loop fusion ---------------------------------------------------------
  #
  # `CustomSynth.Glsl` emits one marker per `sum` node, and reverse-mode AD of
  # a multi-parameter log-density produces many. Rendering each as its own
  # `for` loop made the shader walk the observation axis once per marker and
  # load `obs_inv_mass[j]` once per marker per observation — MEASURED on
  # mac-248 at 3 loops for `y ~ N(mu, 1)`, 15 for `y ~ N(mu, sigma)` and
  # **165** for `y ~ T(df, mu, sigma)`, with bodies averaging ~4 arithmetic
  # ops. Dispatch cost tracked the loop count, not the model size:
  # `cost ~= 215 + 131*n_obs` us at d=2, and 716 us/obs at d=3.
  #
  # Markers that share a trip count are fused into ONE loop with one `obs_j`
  # load and one accumulator per marker.
  #
  # This is BIT-IDENTICAL, and that is the reason it goes in ahead of the
  # obs-axis parallelism this module's docstring has queued as "R2.2.1": each
  # accumulator still visits the same `j` in the same order over the same
  # values, so nothing is reassociated and the f64 result is unchanged to the
  # last bit. A workgroup tree reduction would change the summation order and
  # needs the equivalence tolerances re-derived first.
  #
  # Fusion is also what finally lets the CSE pass do its job. `@cse_min_len 18`
  # against ~4-op bodies found nothing to hoist; run over the fused body it
  # sees the repetition across accumulators that it was calibrated for.
  defp render_reduce_loops([], _obs_index), do: []

  defp render_reduce_loops(markers, obs_index) do
    if fusable?(markers) do
      markers
      |> group_by_bounds()
      |> Enum.map(fn {bounds, entries} -> reduce_loop(bounds, entries, obs_index) end)
    else
      # Conservative fallback: one loop per marker, exactly as before. Taken
      # when some body reads another marker's accumulator, where fusing would
      # hand it a PARTIAL sum instead of a finished one.
      Enum.map(markers, fn {bounds, accum, inner} ->
        reduce_loop(bounds, [{accum, inner}], obs_index)
      end)
    end
  end

  # Safe to fuse only if no marker's body reads another marker's accumulator.
  # `collect_rs/7` rewrites each marker to its accumulator name in the
  # SURROUNDING text, so sibling markers are independent by construction — but
  # that is a property of the emitter, not a guarantee, and reading a partial
  # sum would be a finite, plausible, wrong answer rather than a crash.
  #
  # Compares identifier sets rather than running a regex per pair: at 165
  # markers the pairwise form is 27k regex compiles, and a substring test
  # would match `_gacc0_1` inside `_gacc0_10`.
  defp fusable?(markers) do
    accums = MapSet.new(markers, fn {_bounds, accum, _inner} -> accum end)

    Enum.all?(markers, fn {_bounds, own, inner} ->
      inner
      |> identifiers()
      |> MapSet.delete(own)
      |> MapSet.disjoint?(accums)
    end)
  end

  defp identifiers(text) do
    ~r/[A-Za-z_][A-Za-z0-9_]*/
    |> Regex.scan(text)
    |> List.flatten()
    |> MapSet.new()
  end

  # Group by trip count, preserving first-appearance order. Markers with
  # different bounds — per-node obs spans, or a capture-derived length — form
  # separate groups and keep their own loops, which is correct: they iterate
  # over different data.
  defp group_by_bounds(markers) do
    {order, groups} =
      Enum.reduce(markers, {[], %{}}, fn {bounds, accum, inner}, {order, groups} ->
        seen = Map.get(groups, bounds, [])
        order = if seen == [], do: [bounds | order], else: order
        {order, Map.put(groups, bounds, [{accum, inner} | seen])}
      end)

    order
    |> Enum.reverse()
    |> Enum.map(fn bounds -> {bounds, Enum.reverse(Map.fetch!(groups, bounds))} end)
  end

  # One loop, N accumulators, one obs_j load.
  #
  # CSE runs over the joined `+=` STATEMENTS rather than over a bare
  # expression. That is safe because `group_freqs/1` only ever proposes
  # balanced `(...)` groups found by paren matching, so a hoisted candidate is
  # always a complete parenthesised subexpression and can never span a `;`.
  # With a single entry it reduces to the pre-fusion output: the extra
  # `(#{inner})` group appears once and `shortest_repeat/1` requires count > 1.
  defp reduce_loop({lo, hi}, entries, obs_index) do
    decls = Enum.map_join(entries, "\n", fn {accum, _inner} -> "double #{accum} = 0.0lf;" end)

    adds =
      Enum.map_join(entries, "\n", fn {accum, inner} -> "    #{accum} += (#{inner});" end)

    {cse_binds, adds_cse} = cse_loop_body(adds)

    """
    #{decls}
    for (uint j = #{lo}; j < #{hi}; j++) {
        double obs_j = obs_inv_mass[#{obs_index}];
    #{cse_binds}
    #{adds_cse}
    }
    """
  end

  # --- In-loop common-subexpression elimination -------------------
  #
  # Greedy shortest-first hoisting of repeated balanced subexpressions in
  # the per-obs body. Shortest-first means innermost bindings are created
  # first, so a later (larger) binding references the vars introduced by
  # earlier ones and creation order already is dependency order. Every
  # binding and use shares the single obs-loop scope, so this is a safe
  # local transform (validated numerically against the non-CSE shader).

  @cse_min_len 18
  @cse_max_binds 600

  defp cse_loop_body(inner) do
    # Toggle for benchmarking/debugging: `config :exmc, glsl_cse: false` emits
    # the raw redundant body (no hoisting). On by default.
    if Application.get_env(:exmc, :glsl_cse, true) do
      do_cse_loop_body(inner)
    else
      {"", inner}
    end
  end

  defp do_cse_loop_body(inner) do
    {binds, expr} = cse_hoist(inner, [], 0)

    lines =
      binds
      |> Enum.reverse()
      |> Enum.map_join("\n", fn {v, s} -> "        double #{v} = #{s};" end)

    {lines, expr}
  end

  defp cse_hoist(expr, binds, n) when n >= @cse_max_binds, do: {binds, expr}

  defp cse_hoist(expr, binds, n) do
    case shortest_repeat(expr) do
      nil ->
        {binds, expr}

      sub ->
        var = "_c#{n}"
        cse_hoist(String.replace(expr, sub, var), [{var, sub} | binds], n + 1)
    end
  end

  defp shortest_repeat(expr) do
    expr
    |> group_freqs()
    |> Enum.filter(fn {s, c} -> c > 1 and byte_size(s) >= @cse_min_len end)
    |> case do
      [] -> nil
      list -> list |> Enum.min_by(fn {s, _} -> byte_size(s) end) |> elem(0)
    end
  end

  # Frequency map of all balanced `(...)` groups, each with its call-name
  # prefix (so `exp_d(...)` is captured whole, not just `(...)`).
  defp group_freqs(expr) do
    scan_groups(:binary.bin_to_list(expr), 0, expr, [], %{})
  end

  defp scan_groups([], _i, _expr, _stack, freq), do: freq

  defp scan_groups([?( | rest], i, expr, stack, freq),
    do: scan_groups(rest, i + 1, expr, [{i, ident_start(expr, i)} | stack], freq)

  defp scan_groups([?) | rest], i, expr, [{_open, istart} | stack], freq) do
    group = binary_part(expr, istart, i - istart + 1)
    scan_groups(rest, i + 1, expr, stack, Map.update(freq, group, 1, &(&1 + 1)))
  end

  defp scan_groups([?) | rest], i, expr, [], freq),
    do: scan_groups(rest, i + 1, expr, [], freq)

  defp scan_groups([_c | rest], i, expr, stack, freq),
    do: scan_groups(rest, i + 1, expr, stack, freq)

  # Start index of the identifier immediately before the `(` at `i`
  # (so the call name is included), or `i` for a bare group.
  defp ident_start(expr, i), do: scan_ident(expr, i - 1)

  defp scan_ident(expr, j) when j >= 0 do
    if ident_char?(:binary.at(expr, j)), do: scan_ident(expr, j - 1), else: j + 1
  end

  defp scan_ident(_expr, _j), do: 0

  defp ident_char?(c),
    do: (c >= ?a and c <= ?z) or (c >= ?A and c <= ?Z) or (c >= ?0 and c <= ?9) or c == ?_

  # Given the open-paren index, find the matching close-paren index.
  defp find_matching_paren(glsl, open_idx) do
    do_find_paren(glsl, open_idx + 1, 1)
  end

  defp do_find_paren(glsl, idx, depth) when depth > 0 and idx < byte_size(glsl) do
    case binary_part(glsl, idx, 1) do
      "(" -> do_find_paren(glsl, idx + 1, depth + 1)
      ")" -> if depth == 1, do: idx, else: do_find_paren(glsl, idx + 1, depth - 1)
      _ -> do_find_paren(glsl, idx + 1, depth)
    end
  end

  defp do_find_paren(_glsl, _idx, _depth), do: {:error, :unbalanced}

  # log_p body — thread 0 computes everything (loops + expr), other
  # threads carry lp_i = 0 through the workgroup reduction.
  defp build_logp_body_with_loops(_d, loops, expr) do
    """
    if (tid == 0u) {
    #{indent(Enum.join(loops, "\n"), 4)}
        lp_i = #{expr};
    }
    """
    |> String.trim_trailing()
  end

  # grad body — each tid < d runs its own per-RV loop chain in
  # parallel. The threads execute the same template code with `tid`
  # branching to their assigned grad expression.
  defp build_grad_body_with_loops(d, loops_per_tid, expr_per_tid, out_var) do
    0..(d - 1)
    |> Enum.map(fn i ->
      loops = Map.get(loops_per_tid, i, [])
      expr = Map.get(expr_per_tid, i, "0.0")

      loops_str =
        if loops == [] do
          ""
        else
          indent(Enum.join(loops, "\n"), 4) <> "\n"
        end

      """
      if (tid == #{i}u) {
      #{loops_str}    #{out_var} = #{expr};
      }
      """
      |> String.trim_trailing()
    end)
    |> Enum.join("\n")
  end

  # Checks RV IDS, not layout entries. Since vector RVs, `layout` carries one
  # name per q SLOT — `beta[0]`, `beta[1]` — which are labels rather than ids
  # and are not in `priors` by construction. Strip the element suffix and check
  # the id that remains, so the guard still catches what it was written for: a
  # layout naming an RV the prior set does not contain.
  defp validate_layout(priors, layout) do
    ids = MapSet.new(priors, fn {id, _, _} -> id end)

    case Enum.find(layout, &(not MapSet.member?(ids, slot_base_id(&1)))) do
      nil -> :ok
      missing -> {:error, {:layout_id_not_in_priors, missing}}
    end
  end

  defp slot_base_id(name) when is_binary(name) do
    case String.split(name, "[", parts: 2) do
      [base, _rest] -> base
      [base] -> base
    end
  end

  defp slot_base_id(name), do: name

  @doc """
  For each prior, trace `mod.logpdf(x, params)` + its gradient
  with a scalar template and emit the resulting `Nx.Defn.Expr`
  trees through `CustomSynth.Glsl.emit`. Returns
  `[{id, log_p_glsl, grad_glsl}, ...]` on success.

  Layout `["qi"]` binds the lone parameter (the scalar `x`) to
  the GLSL local `qi` — which is what the template's per-thread
  leapfrog code uses. Prior params (mu, sigma, scale, lambda) are
  inlined as GLSL constants because they come in as concrete
  tensors at trace time.
  """
  @spec emit_prior_fragments([prior_entry()]) ::
          {:ok, [{atom() | binary(), binary(), binary()}]} | {:error, term()}
  def emit_prior_fragments(priors) do
    priors
    |> Enum.reduce_while({:ok, []}, fn {id, mod, params}, {:ok, acc} ->
      # Nx 0.12 refuses to trace a defn with closure-captured tensors
      # from a different backend implementation. Under the D88 f64
      # Vulkano default, `params` values built via `Nx.tensor/1` at
      # the caller land on VulkanoBackend; the trace boundary here
      # can't mix VulkanoBackend + Nx.Defn.Expr. Copy every tensor
      # param to BinaryBackend so the closure captures inlinable
      # constants. Same trick as the "backend_copy" hint in Nx's own
      # error message.
      params = normalize_params(params)
      transform = mod.transform(params)

      value_fn = fn z ->
        x = Exmc.Transform.apply(transform, z)
        logp = mod.logpdf(x, params)
        jac = Exmc.Transform.log_abs_det_jacobian(transform, z)
        Nx.add(logp, jac)
      end

      grad_fn = fn z -> Nx.Defn.grad(z, value_fn) end
      template = Nx.template({}, :f64)

      value_expr = Nx.Defn.debug_expr_apply(value_fn, [template])
      grad_expr = Nx.Defn.debug_expr_apply(grad_fn, [template])

      with {:ok, lp_glsl} <- Exmc.NUTS.CustomSynth.Glsl.emit(value_expr, ["qi"]),
           {:ok, grad_glsl} <- Exmc.NUTS.CustomSynth.Glsl.emit(grad_expr, ["qi"]) do
        {:cont, {:ok, [{id, lp_glsl, grad_glsl} | acc]}}
      else
        err -> {:halt, err}
      end
    end)
    |> case do
      {:ok, fragments} -> {:ok, Enum.reverse(fragments)}
      err -> err
    end
  end

  @typep prior_entry :: {atom() | binary(), module(), map()}

  # Copy every %Nx.Tensor{} value in a prior's params map to
  # Nx.BinaryBackend. Scalars and non-tensor values pass through.
  # Used so the closure passed to Nx.Defn.debug_expr_apply doesn't
  # capture VulkanoBackend tensors — Nx 0.12 rejects that mix.
  defp normalize_params(params) when is_map(params) do
    Map.new(params, fn
      {k, %Nx.Tensor{} = v} ->
        v = Nx.backend_copy(v, Nx.BinaryBackend)
        # Vectorized prior params (shape {d}) must be scalarized for
        # the per-element logpdf trace. The prior-only synth path
        # dispatches per-tid — each thread evaluates the same formula
        # with its own qi. Vector params would produce captured-array
        # references indexed by an obs-loop variable 'j' that doesn't
        # exist in the prior-only template (Surface A codegen bug).
        # Extract element 0 to get a scalar; the formula is identical
        # for all elements of a homogeneous vectorized prior.
        v = if Nx.rank(v) > 0, do: Nx.squeeze(Nx.slice(v, [0], [1])), else: v
        {k, v}

      {k, v} ->
        {k, v}
    end)
  end

  defp normalize_params(other), do: other

  defp build_per_tid_dispatch(fragments, layout, out_var, kind) do
    layout
    |> Enum.with_index()
    |> Enum.map(fn {id, idx} ->
      {^id, lp, grad} = Enum.find(fragments, fn {fid, _, _} -> fid == id end)
      expr = if kind == :grad, do: grad, else: lp
      "if (tid == #{idx}u) { #{out_var} = #{expr}; }"
    end)
    |> Enum.join("\n")
  end

  defp indent(text, spaces) do
    pad = String.duplicate(" ", spaces)

    text
    |> String.split("\n")
    |> Enum.map(fn
      "" -> ""
      line -> pad <> line
    end)
    |> Enum.join("\n")
  end

  # ============================================================
  # Task #154 — Batched multi-instance shader
  # ============================================================
  #
  # `render_batched/1` mirrors `render/1` but emits a multi-instance shader.
  # Each workgroup processes one instance; every buffer index is offset by
  # `gl_WorkGroupID.x`. Dispatch with `[n_instances, 1, 1]`.
  #
  # The leapfrog math and the prior/likelihood bodies are identical
  # per-instance, and that is load-bearing rather than incidental: both
  # variants render from the SAME emitted bodies, through the same
  # `emit_prior_fragments/1`, the same `transform_reduce_sum/4` (spans and
  # CSE included) and the same `rewrite_transcendentals_f64/1`. Only two
  # things differ — the template text, and the observation index expression
  # (`obs_inv_mass[j]` becomes `obs_inv_mass[extras_off + j]`). Anything a
  # batched instance computes is therefore the same arithmetic in the same
  # order as the single-instance shader computes for that chain, which is
  # what makes bit-identity a testable claim rather than a hope. See
  # `batched_shader_test.exs`.
  #
  # Shared memory (`partial[]`, `q_shared[]`) is per-workgroup, so it is
  # already per-instance and needs no offsetting.
  #
  # ## This was f32, and could not have run
  #
  # The first version of this template declared `float` buffers and a
  # `float eps` push field while `do_transform_rs_batched/4` emitted `double`
  # accumulators into it — GLSL has no implicit double-to-float conversion,
  # so any model reaching the reduce-sum path failed in `glslangValidator`.
  # It never surfaced because there was no f64 batch NIF to dispatch to, so
  # nothing ever rendered this template in anger.
  #
  # f32 was the wrong target regardless. `leapfrog_chain_synth_batch_f64/6`
  # writes `n_instances * K * d * 8` bytes and `Dispatch.chain_batch/5` packs
  # and slices f64 at 8 bytes an element; an f32 shader bound to those
  # buffers reads and writes at half stride and returns plausible garbage.

  @batched_template ~S"""
  #version 450
  #extension GL_ARB_gpu_shader_fp64 : require

  // SYNTHESIZED BATCHED by Exmc.NUTS.CustomSynth.MultiRvCustomSpec (f64)
  //
  // Same shader as @template, one workgroup per instance. Every difference
  // from the single-instance variant is an index offset; the arithmetic is
  // byte-for-byte the same code.
  //
  // Buffer layouts (instance-major, std430, f64) — these match
  // Dispatch.chain_batch/5's packer and the batch NIF's slicing:
  //   q_init[i*d + j]
  //   p_init[i*d + j]
  //   obs_inv_mass[i*(n_obs+d) + j]   obs 0..n_obs-1, then inv_mass 0..d-1
  //   q_chain[i*K*d + k*d + j], p_chain[...], grad_chain[...]
  //   logp_chain[i*K + k]             one scalar per step, not per dimension
  //
  // Push block is 24 bytes: the same first twelve as the single-instance
  // block, with n_instances occupying the slot _pad holds there. Matches
  // parse_push_block_batch_f64 in nx_vulkan.

  layout (local_size_x = 256) in;

  layout (push_constant) uniform Push {
      uint   K;
      uint   n_obs;
      uint   d;
      uint   n_instances;
      double eps;
  } pc;

  layout (std430, binding = 0) readonly  buffer In_q     { double q_init[]; };
  layout (std430, binding = 1) readonly  buffer In_p     { double p_init[]; };
  layout (std430, binding = 2) readonly  buffer In_extras {
      double obs_inv_mass[];
  };
  layout (std430, binding = 3) writeonly buffer Out_q    { double q_chain[]; };
  layout (std430, binding = 4) writeonly buffer Out_p    { double p_chain[]; };
  layout (std430, binding = 5) writeonly buffer Out_grad { double grad_chain[]; };
  layout (std430, binding = 6) writeonly buffer Out_logp { double logp_chain[]; };

  {{captured_decls}}

  // Per-workgroup, therefore per-instance. No offsetting needed, and none
  // wanted: two instances sharing one tile is exactly the bleed this
  // shader's tests go looking for.
  shared double partial[256];
  shared double q_shared[256];

  void main() {
      uint inst = gl_WorkGroupID.x;
      uint tid  = gl_LocalInvocationIndex;

      // `inst` is uniform across the workgroup, so every barrier below is
      // still reached by every invocation in it — the whole workgroup is
      // either in range or idle together. A per-thread in_inst would be a
      // deadlock, not a guard.
      bool in_inst   = (inst < pc.n_instances);
      bool in_bounds = in_inst && (tid < pc.d);

      uint q_off      = inst * pc.d;
      uint extras_off = inst * (pc.n_obs + pc.d);
      uint chain_off  = inst * pc.K * pc.d;
      uint logp_off   = inst * pc.K;

      double qi = in_bounds ? q_init[q_off + tid]                       : 0.0lf;
      double pi = in_bounds ? p_init[q_off + tid]                       : 0.0lf;
      double mi = in_bounds ? obs_inv_mass[extras_off + pc.n_obs + tid] : 0.0lf;

      for (uint k = 0u; k < pc.K; k++) {
          if (in_bounds) q_shared[tid] = qi;
          barrier();

          double grad_q = 0.0lf;
          if (in_bounds) {
  {{prior_grad_body_q}}
          }
          double p_half = pi + 0.5lf * pc.eps * grad_q;

          double qn = qi + pc.eps * mi * p_half;
          qi = qn;

          barrier();
          if (in_bounds) q_shared[tid] = qi;
          barrier();

          // POST-update, for the same reason as the single-instance template:
          // the four outputs of iteration k must describe the SAME state, so
          // logp_chain[k] is computed here, below the position update and
          // reading the refreshed q_shared. See @template for what a stale
          // density did to the posterior, and for three weeks of blaming it
          // on the GPU.
          double grad_qn = 0.0lf;
          double lp_i    = 0.0lf;
          if (in_bounds) {
  {{prior_grad_body_qn}}
  {{prior_logp_body_qn}}
          }
          pi = p_half + 0.5lf * pc.eps * grad_qn;

          if (in_bounds) {
              q_chain[chain_off + k * pc.d + tid]    = qi;
              p_chain[chain_off + k * pc.d + tid]    = pi;
              grad_chain[chain_off + k * pc.d + tid] = grad_qn;
          }

          // lp_i is already 0.0lf on every thread that did not compute one —
          // out of bounds, out of instance, or simply not thread 0 in the
          // custom path — so this needs no in_inst ternary, and matches
          // @template's line exactly.
          partial[tid] = lp_i;
          barrier();

          for (uint s = 128u; s > 0u; s /= 2u) {
              if (tid < s) partial[tid] += partial[tid + s];
              barrier();
          }

          if (tid == 0u && in_inst) {
              logp_chain[logp_off + k] = partial[0];
          }
          barrier();
      }
  }
  """

  @doc """
  Render the batched multi-instance GLSL shader for `components`.

  Same contract as `render/1` and the same two paths through it — prior-only
  models go through `render_prior_only/3`, everything else through
  `render_with_custom/3` — differing only in the template and in the
  observation index expression. One workgroup handles one instance; dispatch
  with `[n_instances, 1, 1]` and lay the buffers out instance-major.

  ## Why it is not its own renderer

  It was, and the copy drifted. The batched path had its own reduce-sum
  rewriter that omitted the obs spans (so a model with several observed nodes
  would have counted its whole likelihood once per node), omitted the
  common-subexpression pass, and emitted `double` accumulators into a `float`
  template — which does not compile. None of that showed up, because there
  was no f64 batch NIF to dispatch the result to.

  Sharing the emitter is also the only way the bit-identity claim in
  `batched_shader_test.exs` can hold: an instance of a batched dispatch must
  equal the same chain dispatched alone, byte for byte, on all four output
  buffers. That is a statement about the two shaders performing the same
  arithmetic in the same order, and the cheapest way to guarantee it is for
  one function to emit both.

  ## Prior-only and observed models are batchable

  This used to return `{:error, :prior_only_batched_not_supported}` for
  `custom: nil`, which excluded every conjugate model — the whole class the
  batch coordinator exists to serve, since `Builder.obs` models have no
  Custom likelihood. The guard now matches `render/1`'s: only a model with
  neither a Custom term nor observed nodes takes the prior-only path, and it
  takes it in both variants.
  """
  @spec render_batched(components()) :: {:ok, binary()} | {:error, term()}
  def render_batched(%{priors: priors, custom: custom_info, layout: layout} = components) do
    observed = Map.get(components, :observed, [])

    with :ok <- validate_layout(priors, layout) do
      if is_nil(custom_info) and observed == [] do
        drop_captures(render_prior_only(priors, layout, @batched_template))
      else
        drop_captures(
          render_with_custom(components, @batched_template, "extras_off + j", :batched)
        )
      end
    end
  rescue
    # Same degradation as render/1: an unattributable obs span becomes an
    # {:error, _} so the caller reports :unsupported and the model samples
    # unbatched rather than wrongly.
    e in RuntimeError ->
      if String.starts_with?(Exception.message(e), "transform_reduce_sum:") do
        {:error, {:obs_span_attribution, Exception.message(e)}}
      else
        reraise e, __STACKTRACE__
      end
  end
end
