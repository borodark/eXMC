defmodule Exmc.OptionalDepsTest do
  use ExUnit.Case, async: true

  @moduledoc """
  Optional backends must be optional at *runtime*, not merely at resolution.

  The defect this guards against: `exla` was declared `optional: true` but
  still landed in `exmc`'s `applications` list, so the BEAM tried to start it
  at boot. A CUDA build of exla on a host missing `libnvshmem_host.so.3`
  therefore aborted the VM before ExUnit ran a single test — the whole suite
  made unrunnable by a dependency the project advertises as optional.
  """

  describe "exmc's boot path" do
    test "does not require exla to start" do
      applications = Application.spec(:exmc, :applications)

      refute :exla in applications, """
      :exla is back in exmc's applications list, which means a present-but-broken
      exla (wrong XLA_TARGET, missing CUDA runtime library, stale _build) aborts
      the VM at boot and makes the entire test suite unrunnable.

      Keep `runtime: false` on the exla dep in mix.exs. Exmc.JIT starts it
      lazily and treats a failed start as "backend unavailable".

      applications: #{inspect(applications)}
      """
    end

    if match?({:unix, :freebsd}, :os.type()) do
      test "exla is deliberately absent on FreeBSD" do
        # The inverse invariant, and it is the one that matters here. The `xla`
        # archive ships darwin and linux-gnu targets only, so on FreeBSD merely
        # *declaring* exla makes `mix compile` die in the dependency before
        # reaching a module of this library. mix.exs drops it from the dep list
        # on this platform; this is the assertion that says so out loud, so the
        # next person to "fix" that conditional finds out here rather than on
        # the fleet.
        refute Code.ensure_loaded?(EXLA),
               "exla must not be a dependency on FreeBSD — see the @freebsd? " <>
                 "conditional in mix.exs, and b536a40 for the measurement"
      end
    else
      test "exla is still on the code path when installed" do
        # `runtime: false` must not make exla invisible — only unstarted. If this
        # fails, Exmc.JIT can never select EXLA on any host.
        assert Code.ensure_loaded?(EXLA), "exla is not installed in this environment"
      end
    end
  end

  describe "Exmc.JIT.detect_compiler/0" do
    test "never returns a backend whose application failed to start" do
      case Exmc.JIT.detect_compiler() do
        nil ->
          :ok

        compiler ->
          app = Application.get_application(compiler)

          assert app in Enum.map(Application.started_applications(), &elem(&1, 0)),
                 "#{inspect(compiler)} was selected but #{inspect(app)} is not running"
      end
    end

    test "backend/0 agrees with the detected compiler" do
      expected =
        case Exmc.JIT.detect_compiler() do
          EXLA -> EXLA.Backend
          Nx.Vulkan -> Nx.Vulkan.VulkanoBackend
          nil -> Nx.BinaryBackend
        end

      assert Exmc.JIT.backend() == expected
    end
  end
end
