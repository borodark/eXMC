defmodule Exmc.Dyntrace do
  @moduledoc false
  # Thin wrapper around Erlang's `:dyntrace` USDT/DTrace probes.
  #
  # The `:dyntrace` module only exists on emulators built with
  # `--with-dynamic-trace`. On a stock BEAM it is absent, and calling
  # `:dyntrace.p/8` raises `UndefinedFunctionError`. The Vulkan dispatch
  # path fires probes on the hot loop, so we route them through here:
  # real probes when the emulator supports them, no-ops otherwise.

  @available Code.ensure_loaded?(:dyntrace) and function_exported?(:dyntrace, :p, 8)

  if @available do
    defdelegate p(a, b, c, d, e, f, g, h), to: :dyntrace
    defdelegate put_tag(tag), to: :dyntrace
  else
    def p(_a, _b, _c, _d, _e, _f, _g, _h), do: false
    def put_tag(_tag), do: false
  end
end
