defmodule Exmc.NUTS.CustomSynth.Push do
  @moduledoc """
  Push-constants packing for synthesised chain shaders.

  R2.3 of `specs/vulkan-custom-synthesis.md`.  The
  `nxv_leapfrog_chain_synth` shim accepts an opaque push-constants
  block up to 128 bytes; this module defines the layout for
  synthesised regime-shaped shaders and packs it into a binary.

  Layout (matches the `Push` UBO declared in the
  `MultiRvCustomSpec` template — the GLSL side reads these fields
  in the same order):

      uint   K           // leapfrog steps per dispatch
      uint   n_obs       // observation count
      uint   d           // free-RV dimension (= length of position vector q)
      uint   _pad        // 4-byte pad to keep double alignment to 8
      double eps         // leapfrog step size
      double prior_params[N]  // packed standard-family prior params,
                              // one entry per RV in `layout` order;
                              // shape per RV depends on its prior
                              // distribution and is encoded by
                              // `prior_param_floats/2`

  ## There is no prior-float cap. The binding limit is `d <= 256`.

  `pack/1` emits the 24-byte header and nothing else, so the block is a
  fixed size regardless of model width. **The binding constraint on the
  synthesised chain path is the shader's thread tile — `local_size_x = 256`
  with a `q_shared[256]` tile, i.e. `d <= 256`.**

  This moduledoc used to say the opposite, at length: that the 128-byte
  push block held 13 f64 prior floats and was "the binding one", capping
  models at `d <= 13` for one-parameter priors, 6 for Normal, 3 for
  TruncatedNormal, while dismissing `d <= 256` as "never the binding one".
  Both halves were backwards, and the error was load-bearing — it is why
  `chain_batch/5`, `tree.ex` and `compiler.ex` all repeated it.

  What was actually happening: `pack/1` appended prior floats that
  **nothing consumed**. `MultiRvCustomSpec` bakes prior parameters into the
  GLSL as literals at synthesis time, and `leapfrog_chain_synth_f64`
  forwards only `sizeof(PushBlockF64) = 24` bytes. But the NIF rejects
  `push.len() > 128` before dispatching, so the unread tail was counted
  against a budget it never spent, and models past ~6 free Normal RVs were
  pushed onto the per-op fallback for no reason.

  Measured on an 8-RV conjugate model: **0 chain dispatches / 160.9 s**
  with the tail, **2564 dispatches / 12.3 s** without — 13.1x, posterior
  unchanged and correct against the closed-form conjugate on both arms.

  `prior_param_floats/1` is retained: `chain_batch/5` still packs a tail for
  the batched f32 path, and `ensure_fits!/2` still guards that against the
  same 128-byte NIF limit. That path is not currently reachable (see
  `docs/BATCHED_CHAIN_DISPATCH.md`, D4) and its shader bakes priors too, so
  the tail there is likely just as dead — but it has not been measured, so
  it stays.

  A model that needs to grow past these counts needs its prior params
  moved into an SSBO; there is no headroom to tune.

  Obs data does NOT go in push constants (1600 B f64 / 800 B f32
  exceeds 128 B).  Obs is repacked into one of the existing SSBO
  bindings per R2.3.b (see specs).
  """

  alias Exmc.IR

  @type prior :: {atom() | binary(), module(), map()}
  @type spec :: %{
          K: non_neg_integer(),
          n_obs: non_neg_integer(),
          d: non_neg_integer(),
          eps: float(),
          priors: [prior()]
        }

  @max_bytes 128

  @doc """
  Build a push-constants spec from R1's extracted IR components +
  runtime parameters (K, eps).
  """
  @spec build(map(), keyword()) :: spec()
  def build(components, opts \\ []) do
    k = Keyword.get(opts, :K, 32)
    eps = Keyword.fetch!(opts, :eps)
    n_obs = Keyword.fetch!(opts, :n_obs)

    %{
      K: k,
      n_obs: n_obs,
      d: length(components.layout),
      eps: eps,
      priors: components.priors
    }
  end

  @doc """
  Pack a spec into the binary push block the shader expects.

  Returns `{:ok, binary, n_bytes}`. Always succeeds: the block is the fixed 24-byte header, well inside the
  NIF's 128-byte limit, so there is no width at which this can fail. It used
  to return `{:error, :push_too_large}` for wide models; see the comment in
  the body for why that cap was measuring the wrong thing.
  """
  @spec pack(spec()) :: {:ok, binary(), non_neg_integer()}
  def pack(%{K: k, n_obs: n_obs, d: d, eps: eps, priors: priors}) do
    header =
      <<
        k::little-unsigned-integer-32,
        n_obs::little-unsigned-integer-32,
        d::little-unsigned-integer-32,
        0::little-unsigned-integer-32,
        eps::little-float-64
      >>

    # Header only. The prior parameters are NOT sent through push constants.
    #
    # This block used to be `header <> prior_bin`, and that tail was inert in
    # every direction except one, where it was actively harmful:
    #
    #   * The synthesised shader never read it. `MultiRvCustomSpec` bakes prior
    #     parameters into the GLSL as literals at synthesis time — verified by
    #     disassembling a cached SPV for a `Normal(0.0, 7.3125)` prior, which
    #     contains `OpConstant %double 7.3125` and its precomputed
    #     normalisation term, and whose push struct is
    #     `OpTypeStruct %uint %uint %uint %uint %double` — these five members
    #     and nothing else.
    #   * The NIF never forwarded it. `leapfrog_chain_synth_f64` pushes
    #     `sizeof(PushBlockF64) = 24` bytes; anything beyond that is dropped.
    #   * But the NIF DOES reject `push.len() > 128` before dispatching. So the
    #     tail's only effect was to be counted against a limit it never
    #     consumed, and it was the sole cause of the free-RV width cap.
    #
    # Measured on the 8-RV conjugate model in
    # `test/exmc/nuts/custom_synth/push_width_test.exs`: with the tail, 0 chain
    # dispatches and 160.9 s via per-op fallback; without it, 2564 dispatches
    # and 12.3 s, posterior unchanged and correct against the closed-form
    # conjugate on both arms. 13.1x.
    #
    # `priors` stays in the spec because synthesis needs it to bake the
    # literals and because it feeds the content-addressed shader hash — a
    # different prior must produce a different shader. It just does not travel
    # through this block.
    _ = priors

    {:ok, header, byte_size(header)}
  end

  @doc """
  GLSL declarations matching the packed layout.  Used by R2.2's
  template renderer to fill in the `Push` UBO field list.
  """
  @spec glsl_fields(spec()) :: binary()
  def glsl_fields(%{priors: priors}) do
    prior_lines =
      priors
      |> Enum.flat_map(&prior_glsl_field_lines(&1, "double"))
      |> Enum.join("\n    ")

    """
        uint  K;
        uint  n_obs;
        uint  d;
        uint  _pad;
        double eps;
        #{prior_lines}
    """
  end

  @doc """
  The push-constants budget, in bytes.

  Public so the batched dispatch path can enforce the same number rather than
  hard-code a second copy of it. Vulkan guarantees at least 128 bytes of push
  constants; this is that floor, not a tunable.
  """
  @spec max_bytes() :: pos_integer()
  def max_bytes, do: @max_bytes

  @doc """
  Raise unless `bin` fits the push-constants budget.

  `pack/1` returns `{:error, :push_too_large}` because its callers degrade to
  per-op sampling. `chain_batch/5` has no such branch -- it builds its own
  header (`n_instances` in the pad slot, signed eps) so it cannot go through
  `pack/1`, and until 2026-08-30 it did no size check at all: an oversized
  block went to the NIF, came back `{:error, :bad_input}`, and failed the
  `{:ok, {...}} =` match as a `MatchError` naming nothing. Raising here is
  what the batch coordinator's `try/rescue` turns into `{:fallback, ...}`, so
  the draw degrades to unbatched instead of dying.

  As of 2026-09-02 nothing calls it: `chain_batch/5`, its only caller, now
  builds a fixed 24-byte block that cannot overflow. The bound is still the
  NIF's real one, so the function stays; see `prior_param_floats/1` for why
  the tail it guarded is gone.

  `context` names the caller in the message; the numbers are only knowable
  here.
  """
  @spec ensure_fits!(binary(), String.t()) :: binary()
  def ensure_fits!(bin, context) when is_binary(bin) do
    n = byte_size(bin)

    if n <= @max_bytes do
      bin
    else
      raise """
      #{context}: push block is #{n} B, budget is #{@max_bytes} B (over by #{n - @max_bytes} B).

      The block holds a fixed header plus 8 B per prior parameter float, so a
      model overflows it by having too many prior parameters, not by having
      large ones. Reshape the model, or route it through the per-op path.

      Note this is a DIFFERENT check from the one CustomSynth.synthesise_batched/1
      runs at synth time: that one packs placeholder K and eps values, not the
      ones dispatch actually sends. The two can disagree.
      """
    end
  end

  @doc """
  Extract the scalar parameter floats a prior contributes to the push block.

  ## No dispatch path calls this any more

  It became public on 2026-08-29 for the batched path
  (`Exmc.NUTS.Vulkan.Dispatch.chain_batch/5`), which built its own push header
  and had kept a private copy with 5 of these 12 clauses. `chain_batch/5` now
  sends the 24-byte header alone, for the same reason `pack/1` does: the
  shader reads prior parameters from SPIR-V literals, and the NIF forwards
  only `sizeof(PushBlockBatchF64)`. The tail was unread bytes charged against
  the NIF's 128-byte bound, which capped batching at about six free Normal
  RVs — an 8-RV model came to 152 B and raised.

  Kept because it is public API and the encoding is correct; a future push
  layout that genuinely carries parameters would want exactly this. Do not
  reinstate a tail on the strength of its existence — see
  `push_width_test.exs` for what that cost the single-instance path, and
  `batched_shader_test.exs` for the batched equivalent.

  Raises for a distribution with no clause — a missing encoder must not
  silently pack a short push block.
  """
  @spec prior_param_floats(prior()) :: [float()]
  def prior_param_floats({_id, Exmc.Dist.Normal, params}) do
    [scalar(params, :mu), scalar(params, :sigma)]
  end

  def prior_param_floats({_id, Exmc.Dist.HalfCauchy, params}) do
    [scalar(params, :scale)]
  end

  def prior_param_floats({_id, Exmc.Dist.HalfNormal, params}) do
    [scalar(params, :sigma)]
  end

  def prior_param_floats({_id, Exmc.Dist.Exponential, params}) do
    [scalar(params, :lambda)]
  end

  # Surface A: single-family models routed through synth under f64
  # default now include the three that were spirit-only pre-Option-B.
  def prior_param_floats({_id, Exmc.Dist.StudentT, params}) do
    [scalar(params, :df), scalar(params, :loc), scalar(params, :scale)]
  end

  def prior_param_floats({_id, Exmc.Dist.Cauchy, params}) do
    [scalar(params, :loc), scalar(params, :scale)]
  end

  def prior_param_floats({_id, Exmc.Dist.Weibull, params}) do
    [scalar(params, :k), scalar(params, :lambda)]
  end

  def prior_param_floats({_id, Exmc.Dist.Lognormal, params}) do
    [scalar(params, :mu), scalar(params, :sigma)]
  end

  def prior_param_floats({_id, Exmc.Dist.TruncatedNormal, params}) do
    [scalar(params, :mu), scalar(params, :sigma), scalar(params, :lower), scalar(params, :upper)]
  end

  def prior_param_floats({_id, Exmc.Dist.Gamma, params}) do
    [scalar(params, :alpha), scalar(params, :beta)]
  end

  def prior_param_floats({_id, Exmc.Dist.Beta, params}) do
    [scalar(params, :alpha), scalar(params, :beta)]
  end

  def prior_param_floats({id, mod, _params}) do
    raise "Push.prior_param_floats/1 has no encoder for prior #{id} (#{inspect(mod)}). " <>
            "Add a clause matching the distribution + the scalar fields it requires."
  end

  # GLSL field declarations per distribution module.
  defp prior_glsl_field_lines({id, Exmc.Dist.Normal, _}, fp) do
    ["#{fp} #{id}_mu;", "#{fp} #{id}_sigma;"]
  end

  defp prior_glsl_field_lines({id, Exmc.Dist.HalfCauchy, _}, fp) do
    ["#{fp} #{id}_scale;"]
  end

  defp prior_glsl_field_lines({id, Exmc.Dist.HalfNormal, _}, fp) do
    ["#{fp} #{id}_sigma;"]
  end

  defp prior_glsl_field_lines({id, Exmc.Dist.Exponential, _}, fp) do
    ["#{fp} #{id}_lambda;"]
  end

  defp prior_glsl_field_lines({id, mod, _}, _fp) do
    raise "Push.prior_glsl_field_lines/1 has no encoder for prior #{id} (#{inspect(mod)})."
  end

  defp scalar(params, key) do
    case Map.fetch!(params, key) do
      # A hierarchical parameter is the NAME of another RV, not a constant: it
      # is a function of the position vector, resolved inside the shader body
      # from q. It takes a slot here only so the 128-byte gate below stays
      # conservative.
      #
      # Safe because no prior float in this binary is read by any shader. The
      # template's Push block declares K/n_obs/d/_pad/eps and nothing else, and
      # Push.glsl_fields/1 — which would add per-prior fields — has no callers
      # anywhere in lib or test. Prior parameters reach the shader as constants
      # inlined into the traced expression instead.
      v when is_binary(v) ->
        0.0

      v when is_atom(v) and v not in [nil, true, false] ->
        0.0

      v when is_number(v) ->
        v * 1.0

      %Nx.Tensor{shape: {}} = t ->
        Nx.to_number(t) * 1.0

      # Vectorized prior params (shape {d}): extract element 0.
      # Homogeneous vectorized priors (Normal d=8 with uniform mu/sigma)
      # store the same value for every element; push constants hold
      # the scalar. Heterogeneous vector params are not yet supported
      # in the push-constant packing (would need per-element SSBO).
      %Nx.Tensor{} = t ->
        t |> Nx.squeeze() |> Nx.slice([0], [1]) |> Nx.squeeze() |> Nx.to_number() |> Kernel.*(1.0)

      v ->
        raise "Push.scalar/2: param #{key} is not numeric: #{inspect(v)}"
    end
  end

  # Useful for IR-side validation.
  @doc false
  @spec ir_summary(IR.t()) :: map()
  def ir_summary(%IR{nodes: nodes}) do
    by_mod =
      nodes
      |> Enum.map(fn {_id, n} -> n.op end)
      |> Enum.frequencies()

    %{n_nodes: map_size(nodes), op_counts: by_mod}
  end
end
