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

  ## The real cap: 13 prior floats, not 256 free RVs

  Maximum is 128 bytes per Vulkan spec.  The header above is **24**
  bytes (`4*4` for K/n_obs/d/_pad plus 8 for the f64 `eps`), which
  leaves 104 bytes = **13 f64 prior floats**.

  This moduledoc previously said "16 bytes of fixed header leaves
  room for 28 prior-param floats", which was wrong twice: the header
  is 24 bytes, not 16, and 104 bytes holds 13 f64 values, not 28
  (that count was for f32).  Measured with `pack/1`:

      1 float per RV  (HalfNormal, Exponential, HalfCauchy)  d <= 13
      2 floats per RV (Normal, Cauchy, Weibull, Lognormal,
                       Gamma, Beta)                          d <=  6
      3 floats per RV (StudentT)                             d <=  4
      4 floats per RV (TruncatedNormal)                      d <=  3

  Treat this as a design statement rather than a number to look up.
  The guards elsewhere in the codebase that read `d <= 256` are the
  *thread-tile* limit (`local_size_x = 256` with a `q_shared[256]`
  tile) — a real constraint, but never the binding one. The binding
  one is this block, and it caps model complexity an order of
  magnitude below the width at which GPU compute begins to beat an
  interpreter at all. Anything wider is refused here with
  `{:error, :push_too_large}` and degrades to per-op sampling.

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

  Returns `{:ok, binary, n_bytes}` on success, or
  `{:error, :push_too_large}` if the total exceeds 128 bytes.
  """
  @spec pack(spec()) :: {:ok, binary(), non_neg_integer()} | {:error, atom()}
  def pack(%{K: k, n_obs: n_obs, d: d, eps: eps, priors: priors}) do
    header =
      <<
        k::little-unsigned-integer-32,
        n_obs::little-unsigned-integer-32,
        d::little-unsigned-integer-32,
        0::little-unsigned-integer-32,
        eps::little-float-64
      >>

    prior_floats =
      priors
      |> Enum.flat_map(&prior_param_floats/1)

    prior_bin = for f <- prior_floats, into: <<>>, do: <<f::little-float-64>>

    bin = header <> prior_bin
    n = byte_size(bin)

    if n <= @max_bytes do
      {:ok, bin, n}
    else
      # The tuple stays two-wide — CustomSynth matches on it in two places and
      # degrades to per-op sampling. But "too large" on its own tells nobody by
      # how much or against what budget, and those numbers are only knowable
      # here. A silent downgrade to a 100x-slower path is the kind of thing that
      # gets diagnosed as a hardware problem three weeks later.
      require Logger

      Logger.warning(
        "[Push] push block #{n} B > #{@max_bytes} B: #{length(prior_floats)} prior " <>
          "parameter floats, budget is #{div(@max_bytes - byte_size(header), 8)} " <>
          "(#{@max_bytes} B block less a #{byte_size(header)} B header). This is the " <>
          "real width cap on the synthesised chain path — not d <= 256, which is the " <>
          "shader's workgroup width and never binds first."
      )

      {:error, :push_too_large}
    end
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

  # Extract scalar parameter floats per distribution module.
  defp prior_param_floats({_id, Exmc.Dist.Normal, params}) do
    [scalar(params, :mu), scalar(params, :sigma)]
  end

  defp prior_param_floats({_id, Exmc.Dist.HalfCauchy, params}) do
    [scalar(params, :scale)]
  end

  defp prior_param_floats({_id, Exmc.Dist.HalfNormal, params}) do
    [scalar(params, :sigma)]
  end

  defp prior_param_floats({_id, Exmc.Dist.Exponential, params}) do
    [scalar(params, :lambda)]
  end

  # Surface A: single-family models routed through synth under f64
  # default now include the three that were spirit-only pre-Option-B.
  defp prior_param_floats({_id, Exmc.Dist.StudentT, params}) do
    [scalar(params, :df), scalar(params, :loc), scalar(params, :scale)]
  end

  defp prior_param_floats({_id, Exmc.Dist.Cauchy, params}) do
    [scalar(params, :loc), scalar(params, :scale)]
  end

  defp prior_param_floats({_id, Exmc.Dist.Weibull, params}) do
    [scalar(params, :k), scalar(params, :lambda)]
  end

  defp prior_param_floats({_id, Exmc.Dist.Lognormal, params}) do
    [scalar(params, :mu), scalar(params, :sigma)]
  end

  defp prior_param_floats({_id, Exmc.Dist.TruncatedNormal, params}) do
    [scalar(params, :mu), scalar(params, :sigma), scalar(params, :lower), scalar(params, :upper)]
  end

  defp prior_param_floats({_id, Exmc.Dist.Gamma, params}) do
    [scalar(params, :alpha), scalar(params, :beta)]
  end

  defp prior_param_floats({_id, Exmc.Dist.Beta, params}) do
    [scalar(params, :alpha), scalar(params, :beta)]
  end

  defp prior_param_floats({id, mod, _params}) do
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
