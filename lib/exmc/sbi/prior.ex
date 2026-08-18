defmodule Exmc.SBI.Prior do
  @moduledoc """
  Independent, scalar-per-parameter priors for likelihood-free inference.

  ABC needs two things from a prior and only two: draw from it, and evaluate
  its density at a perturbed particle so the perturbation can be rejected when
  it lands outside the support (Toni et al. 2009, step 2.1). It does **not**
  need a gradient, a transform, or a place in `Exmc.IR` — so this is
  deliberately a small standalone thing rather than a route into the model
  compiler. A `Exmc.SBI` prior is a keyword list, ordered, one scalar per
  entry:

      prior = [
        lambda: {:lognormal, 0.0, 0.5},
        mu:     {:uniform, 0.5, 4.0}
      ]

  The order is load-bearing: it fixes the coordinate order of the particle
  vectors that `Exmc.SBI.ABCSMC`'s perturbation kernel is defined over.

  ## Forms

  | form | support | notes |
  |---|---|---|
  | `{:uniform, a, b}` | `[a, b]` | |
  | `{:normal, mu, sigma}` | ℝ | |
  | `{:lognormal, mu, sigma}` | `(0, ∞)` | `mu`/`sigma` are of the underlying normal |
  | `{:exponential, rate}` | `(0, ∞)` | |
  | `{:gamma, alpha, beta}` | `(0, ∞)` | `beta` is a **rate** |
  | `{Module, params}` | per `Module.support/1` | any `Exmc.Dist` implementing `sample/2` |

  The `{Module, params}` escape hatch takes any module implementing the
  `Exmc.Dist` behaviour with the optional `sample/2` callback; params are
  wrapped as `Nx.BinaryBackend` scalars, because a prior draw is three floats
  and pushing it through an accelerator costs more than it computes.

  ## Densities that are not densities

  `logpdf/2` returns `:neg_infinity` — the atom, not a float — when the value
  is outside the support. Erlang's `:math.log/1` raises on `0.0` rather than
  returning `-inf`, so there is no float to return, and silently substituting
  a large negative number would make an out-of-support particle merely
  improbable instead of impossible. ABC-SMC's correctness depends on that
  rejection being exact: the importance weight has `π(θ)` in the numerator, and
  a particle with `π(θ) = 0` must never enter the population at all.
  """

  alias Exmc.Dist.Gamma

  defstruct entries: []

  @type form ::
          {:uniform, number(), number()}
          | {:normal, number(), number()}
          | {:lognormal, number(), number()}
          | {:exponential, number()}
          | {:gamma, number(), number()}
          | {module(), map()}

  @type spec :: [{atom(), form()}]
  @type t :: %__MODULE__{entries: [{atom(), tuple()}]}

  @log_2pi :math.log(2.0 * :math.pi())

  @doc """
  Normalise a prior spec (or pass an already-built prior through).
  """
  @spec new(spec() | t()) :: t()
  def new(%__MODULE__{} = prior), do: prior

  def new(spec) when is_list(spec) do
    if spec == [] do
      raise ArgumentError, "Exmc.SBI.Prior: prior must have at least one parameter"
    end

    entries =
      Enum.map(spec, fn
        {name, form} when is_atom(name) ->
          {name, normalise(name, form)}

        other ->
          raise ArgumentError,
                "Exmc.SBI.Prior: expected a keyword list of {name, form}, got: #{inspect(other)}"
      end)

    names = Enum.map(entries, &elem(&1, 0))

    if length(Enum.uniq(names)) != length(names) do
      raise ArgumentError, "Exmc.SBI.Prior: duplicate parameter names in #{inspect(names)}"
    end

    %__MODULE__{entries: entries}
  end

  @doc "Parameter names, in coordinate order."
  @spec names(t()) :: [atom()]
  def names(%__MODULE__{entries: entries}), do: Enum.map(entries, &elem(&1, 0))

  @doc "Number of parameters."
  @spec dimension(t()) :: pos_integer()
  def dimension(%__MODULE__{entries: entries}), do: length(entries)

  @doc """
  Draw one parameter map from the prior.

  Threads `:rand` state functionally, so a run is reproducible from its seed
  and a particle's draw does not depend on which scheduler evaluated it.
  """
  @spec sample(t(), :rand.state()) :: {%{atom() => float()}, :rand.state()}
  def sample(%__MODULE__{entries: entries}, rng) do
    {pairs, rng} =
      Enum.map_reduce(entries, rng, fn {name, form}, rng ->
        {value, rng} = sample_form(form, rng)
        {{name, value}, rng}
      end)

    {Map.new(pairs), rng}
  end

  @doc """
  Log prior density at `params`, or `:neg_infinity` outside the support.
  """
  @spec logpdf(t(), %{atom() => number()}) :: float() | :neg_infinity
  def logpdf(%__MODULE__{entries: entries}, params) do
    Enum.reduce_while(entries, 0.0, fn {name, form}, acc ->
      value =
        case Map.fetch(params, name) do
          {:ok, v} when is_number(v) ->
            v * 1.0

          {:ok, other} ->
            raise ArgumentError,
                  "Exmc.SBI.Prior: parameter #{inspect(name)} must be a number, got #{inspect(other)}"

          :error ->
            raise ArgumentError,
                  "Exmc.SBI.Prior: missing parameter #{inspect(name)} in #{inspect(Map.keys(params))}"
        end

      case logpdf_form(form, value) do
        :neg_infinity -> {:halt, :neg_infinity}
        lp -> {:cont, acc + lp}
      end
    end)
  end

  @doc "Particle map to coordinate vector, in prior order."
  @spec to_vector(t(), %{atom() => number()}) :: [float()]
  def to_vector(%__MODULE__{entries: entries}, params) do
    Enum.map(entries, fn {name, _} -> Map.fetch!(params, name) * 1.0 end)
  end

  @doc "Coordinate vector back to a particle map."
  @spec from_vector(t(), [number()]) :: %{atom() => float()}
  def from_vector(%__MODULE__{entries: entries}, vector) do
    entries
    |> Enum.zip(vector)
    |> Map.new(fn {{name, _}, v} -> {name, v * 1.0} end)
  end

  # --- forms -------------------------------------------------------------

  defp normalise(_name, {:uniform, a, b}) when is_number(a) and is_number(b) and b > a,
    do: {:uniform, a * 1.0, b * 1.0}

  defp normalise(_name, {:normal, mu, sigma})
       when is_number(mu) and is_number(sigma) and sigma > 0,
       do: {:normal, mu * 1.0, sigma * 1.0}

  defp normalise(_name, {:lognormal, mu, sigma})
       when is_number(mu) and is_number(sigma) and sigma > 0,
       do: {:lognormal, mu * 1.0, sigma * 1.0}

  defp normalise(_name, {:exponential, rate}) when is_number(rate) and rate > 0,
    do: {:exponential, rate * 1.0}

  defp normalise(_name, {:gamma, alpha, beta})
       when is_number(alpha) and is_number(beta) and alpha > 0 and beta > 0,
       do: {:gamma, alpha * 1.0, beta * 1.0}

  defp normalise(name, {mod, params}) when is_atom(mod) and is_map(params) do
    Code.ensure_loaded!(mod)

    unless function_exported?(mod, :sample, 2) do
      raise ArgumentError,
            "Exmc.SBI.Prior: #{inspect(name)} uses #{inspect(mod)}, which does not export " <>
              "sample/2. A prior must be drawable, not only evaluable."
    end

    {:dist, mod, Map.new(params, fn {k, v} -> {k, tensorise(v)} end)}
  end

  defp normalise(name, other) do
    raise ArgumentError,
          "Exmc.SBI.Prior: unrecognised prior form for #{inspect(name)}: #{inspect(other)}"
  end

  defp tensorise(%Nx.Tensor{} = t), do: t

  defp tensorise(v) when is_number(v),
    do: Nx.tensor(v * 1.0, type: :f64, backend: Nx.BinaryBackend)

  defp sample_form({:uniform, a, b}, rng) do
    {u, rng} = :rand.uniform_s(rng)
    {a + u * (b - a), rng}
  end

  defp sample_form({:normal, mu, sigma}, rng) do
    {z, rng} = :rand.normal_s(rng)
    {mu + sigma * z, rng}
  end

  defp sample_form({:lognormal, mu, sigma}, rng) do
    {z, rng} = :rand.normal_s(rng)
    {:math.exp(mu + sigma * z), rng}
  end

  defp sample_form({:exponential, rate}, rng) do
    {u, rng} = :rand.uniform_s(rng)
    {-:math.log(u) / rate, rng}
  end

  defp sample_form({:gamma, alpha, beta}, rng), do: Gamma.sample_gamma(alpha, beta, rng)

  defp sample_form({:dist, mod, params}, rng) do
    {t, rng} = mod.sample(params, rng)
    {Nx.to_number(t) * 1.0, rng}
  end

  defp logpdf_form({:uniform, a, b}, v) when v >= a and v <= b, do: -:math.log(b - a)
  defp logpdf_form({:uniform, _, _}, _), do: :neg_infinity

  defp logpdf_form({:normal, mu, sigma}, v) do
    z = (v - mu) / sigma
    -0.5 * (z * z + @log_2pi) - :math.log(sigma)
  end

  defp logpdf_form({:lognormal, _, _}, v) when v <= 0.0, do: :neg_infinity

  defp logpdf_form({:lognormal, mu, sigma}, v) do
    log_v = :math.log(v)
    z = (log_v - mu) / sigma
    -0.5 * (z * z + @log_2pi) - :math.log(sigma) - log_v
  end

  defp logpdf_form({:exponential, _}, v) when v < 0.0, do: :neg_infinity
  defp logpdf_form({:exponential, rate}, v), do: :math.log(rate) - rate * v

  defp logpdf_form({:gamma, _, _}, v) when v <= 0.0, do: :neg_infinity

  defp logpdf_form({:gamma, alpha, beta}, v) do
    (alpha - 1.0) * :math.log(v) + alpha * :math.log(beta) - beta * v - lgamma(alpha)
  end

  defp logpdf_form({:dist, mod, params}, v) do
    if in_support?(mod.support(params), v) do
      Nx.to_number(mod.logpdf(Nx.tensor(v, type: :f64, backend: Nx.BinaryBackend), params)) * 1.0
    else
      :neg_infinity
    end
  end

  defp in_support?(:positive, v), do: v > 0.0
  defp in_support?(:unit, v), do: v >= 0.0 and v <= 1.0
  defp in_support?(_, _), do: true

  # Lanczos (g = 7, 9 terms), the same series as `Exmc.Math.lgamma/1` but in
  # plain floats. `Exmc.Math` builds its constants with `Nx.tensor/1`, which
  # picks up the *default* backend; mixing those with the BinaryBackend scalars
  # this module deliberately uses would either raise or quietly drag a prior
  # evaluation onto an accelerator. A prior density is one float.
  @lanczos_g 7.0
  @lanczos_coeffs [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
  ]

  @doc false
  @spec lgamma(float()) :: float()
  def lgamma(x) when x > 0.0 do
    [c0 | rest] = @lanczos_coeffs

    series =
      rest
      |> Enum.with_index(1)
      |> Enum.reduce(c0, fn {c, i}, acc -> acc + c / (x - 1.0 + i) end)

    t = x - 1.0 + @lanczos_g + 0.5
    0.5 * @log_2pi + (x - 0.5) * :math.log(t) - t + :math.log(series)
  end
end
