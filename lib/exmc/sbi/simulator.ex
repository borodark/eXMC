defmodule Exmc.SBI.Simulator do
  @moduledoc """
  The one thing likelihood-free inference needs from a model: the ability to
  run it forward.

      defmodule MyQueue do
        @behaviour Exmc.SBI.Simulator

        @impl true
        def simulate(%{lambda: lambda, mu: mu}, rng) do
          {stats, rng} = MySim.run(lambda, mu, rng)
          {[stats.mean_wait, stats.utilisation], rng}
        end
      end

  `simulate/2` takes a parameter map — plain floats, keyed by the prior's
  parameter names — and a functional `:rand` state, and returns a **summary
  statistic** together with the advanced state. It does not take data, it does
  not return a log-density, and nothing in `Exmc.SBI` ever differentiates it.
  That is the whole point: a discrete-event simulation has no tractable
  likelihood and no gradient, and ABC asks for neither.

  ## Why the RNG is threaded rather than global

  `:rand`'s functional state is what makes an ABC run reproducible from a
  single seed *and* parallelisable at the same time. `Exmc.SBI.ABC` and
  `Exmc.SBI.ABCSMC` derive one child state per proposal from the parent state,
  sequentially and deterministically, and only then fan the simulator calls out
  over schedulers. The draws a particle sees therefore do not depend on which
  core ran it or on how many cores there were — a property a simulator using
  the process-dictionary `:rand.uniform/0` would forfeit, and one that a
  `Task.async_stream` over a shared mutable PRNG cannot have at all.

  It is also what makes **common random numbers** across particles available
  for free: pass the same child seed to two parameter settings and the
  difference between their outputs is the parameter change rather than the
  noise.

  ## Summaries

  The return may be an `Nx.t()`, a list of numbers, or a bare number; all three
  are flattened to a coordinate vector before the distance is taken. A list is
  usually the right choice. The summary of a queueing model is three or four
  floats, allocating a tensor per simulator call costs more than it computes,
  and on a host where the default backend is an accelerator it also moves the
  hottest loop in ABC onto a device to add three numbers.

  Whatever the shape, it must be **the same length on every call**; a summary
  whose dimension depends on the parameters has no meaningful distance.
  """

  @typedoc "Parameter map, keyed by the prior's parameter names."
  @type params :: %{atom() => float()}

  @typedoc "Functional `:rand` state."
  @type rng :: :rand.state()

  @typedoc "A summary statistic: tensor, list, or scalar."
  @type summary :: Nx.t() | [number()] | number()

  @callback simulate(params(), rng()) :: {summary(), rng()}

  @typedoc """
  Anything `Exmc.SBI` will run forward: a module implementing this behaviour,
  or a plain `fn params, rng -> {summary, rng} end`.
  """
  @type t :: module() | (params(), rng() -> {summary(), rng()})

  @doc """
  Run a simulator once, returning the summary as a flat list of floats.
  """
  @spec invoke(t(), params(), rng()) :: {[float()], rng()}
  def invoke(fun, params, rng) when is_function(fun, 2) do
    {summary, rng} = fun.(params, rng)
    {to_vector(summary), rng}
  end

  def invoke(mod, params, rng) when is_atom(mod) do
    {summary, rng} = mod.simulate(params, rng)
    {to_vector(summary), rng}
  end

  @doc """
  Flatten a summary to a list of floats.
  """
  @spec to_vector(summary()) :: [float()]
  def to_vector(%Nx.Tensor{} = t), do: t |> Nx.to_flat_list() |> Enum.map(&(&1 * 1.0))
  def to_vector(list) when is_list(list), do: Enum.map(list, &(&1 * 1.0))
  def to_vector(n) when is_number(n), do: [n * 1.0]

  def to_vector(other) do
    raise ArgumentError,
          "Exmc.SBI.Simulator: a summary must be an Nx tensor, a list of numbers, " <>
            "or a number; got: #{inspect(other)}"
  end

  @doc """
  Validate that a term looks like a simulator, raising with a useful message.
  """
  @spec validate!(term()) :: t()
  def validate!(fun) when is_function(fun, 2), do: fun

  def validate!(mod) when is_atom(mod) do
    Code.ensure_loaded!(mod)

    if function_exported?(mod, :simulate, 2) do
      mod
    else
      raise ArgumentError,
            "Exmc.SBI.Simulator: #{inspect(mod)} does not export simulate/2"
    end
  end

  def validate!(other) do
    raise ArgumentError,
          "Exmc.SBI.Simulator: expected a module implementing Exmc.SBI.Simulator " <>
            "or a 2-arity function, got: #{inspect(other)}"
  end
end
