defmodule Exmc.NUTS.FaultInjector do
  @moduledoc """
  Configurable fault injection for testing fault-tolerant tree building.

  Uses the process dictionary for zero overhead when not active.
  Activate with `FaultInjector.activate(spec)` where spec describes
  when and how to raise.

  ## Spec formats

  - `%{step_count: N, error: type}` — raise on the Nth leaf evaluation
  - `%{depth: D, error: type}` — raise when entering a subtree at depth D

  ## Error types

  - `:crash` — RuntimeError
  - `:oom` — ErlangError (out of memory)
  - `:exla` — ErlangError (EXLA timeout)
  - `:arithmetic` — ArithmeticError
  """

  @doc "Activate fault injection in the current process."
  def activate(spec) when is_map(spec) do
    Process.put(:exmc_fault_inject, %{spec: spec, counter: 0})
    :ok
  end

  @doc "Deactivate fault injection and return the final state."
  def deactivate do
    Process.delete(:exmc_fault_inject)
    :ok
  end

  @doc """
  Check if a fault should fire at the given depth.
  No-op when fault injection is not active (~10ns process dict lookup).
  """
  def maybe_fault!(depth) do
    case Process.get(:exmc_fault_inject) do
      nil ->
        :ok

      state ->
        new_counter = state.counter + 1
        Process.put(:exmc_fault_inject, %{state | counter: new_counter})
        check_and_raise(state.spec, depth, new_counter)
    end
  end

  defp check_and_raise(%{depth: target_depth, error: error_type}, depth, _counter)
       when depth == target_depth do
    raise_error(error_type)
  end

  defp check_and_raise(%{step_count: target, error: error_type}, _depth, counter)
       when counter >= target do
    raise_error(error_type)
  end

  defp check_and_raise(_spec, _depth, _counter), do: :ok

  defp raise_error(:crash), do: raise("Injected crash for fault tolerance testing")
  defp raise_error(:oom), do: raise(%ErlangError{original: :enomem})
  defp raise_error(:exla), do: raise(%ErlangError{original: {:exla_error, "timeout"}})
  defp raise_error(:arithmetic), do: raise(ArithmeticError, message: "injected arithmetic error")
end
