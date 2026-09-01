defmodule Exmc.NUTS.CustomSynth.SpvRecoveryTest do
  @moduledoc """
  A compiled shader that disappears mid-run is rebuilt, not fatal.

  ## The failure this guards

  `Dispatch.chain_synth_vulkano/8` holds an `spv_path` produced at synthesis
  and hands it to the NIF on every dispatch. If the file goes away in between,
  the NIF returns `{:error, :dispatch_failed, "read spv: No such file or
  directory"}` and the `{:ok, _} =` match raises — several frames deep inside
  `Tree.do_build`, in whatever test happened to be running, with nothing in the
  message suggesting that a file was deleted.

  It has happened for at least two unrelated reasons:

    * This module's own shared temp paths, before `compile_fresh/2` moved to
      per-caller names.
    * `Nx.Vulkan.Synthesis.clear_cache/0` doing `File.rm_rf` on
      `~/.exmc/gpu_node/spv` — the two projects shared that directory until
      nx_vulkan moved its caches under `~/.nx_vulkan/`. That one cost a
      full-suite run and an hour of suspecting an unrelated change.

  Eviction, a partial write, an operator `rm`, and a home directory restored
  from backup all produce the same state, so the recovery is written against
  the condition rather than against any one cause.

  ## Why deleting before sampling would not test anything

  `Sampler.sample/3` re-synthesises through `compile_for_sampling/2`, and
  `Compile.compile_glsl/1` rebuilds a missing artifact on its own. So removing
  the file *before* sampling is repaired before dispatch ever sees it — an
  earlier version of this test did exactly that, passed with the recovery code
  reverted, and proved nothing. The deletion has to race an in-flight run.
  """
  use ExUnit.Case, async: false

  @moduletag :requires_vulkan

  alias Exmc.Builder
  alias Exmc.Dist.Normal
  alias Exmc.NUTS.Sampler
  alias Exmc.NUTS.Vulkan.Dispatch

  # A sigma no other test uses, so this model's GLSL — and therefore its
  # content-addressed .spv filename — is unique to this file. The deletions
  # below then cannot disturb a shader another test is dispatching against.
  @sigma 3.7182818

  defp ir do
    Builder.new_ir()
    |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(@sigma)})
    |> Builder.rv("y", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
    |> Builder.obs("y_obs", "y", Nx.tensor(3.0))
  end

  test "sampling survives its shader being deleted underneath it" do
    {:ok, {:synthesised, _sha, _layout, _spec, spv, _obs}} =
      Exmc.NUTS.CustomSynth.synthesise(ir())

    assert File.exists?(spv), "synthesis did not produce #{spv}"

    # Delete repeatedly during the run, so the race is hit rather than hoped
    # for. Without the recovery path this raises on the first dispatch after
    # the first deletion — measured: 1 dispatch, then a MatchError naming
    # "read spv: No such file or directory".
    killer =
      Task.async(fn ->
        Process.sleep(400)

        for _ <- 1..40,
            do:
              (
                File.rm(spv)
                Process.sleep(25)
              )

        :done
      end)

    Dispatch.reset_dispatch_count()

    {trace, _stats} = Sampler.sample(ir(), %{}, num_warmup: 200, num_samples: 600, seed: 5)

    assert Task.await(killer, 30_000) == :done

    dispatches = Dispatch.dispatch_count()

    assert dispatches > 100,
           "only #{dispatches} chain dispatches completed — the run did not survive " <>
             "long enough for this to be testing recovery"

    vals = Nx.to_flat_list(trace["mu"])
    assert length(vals) == 600

    # Correctness, not just survival: a rebuilt shader is compiled from the
    # same GLSL under the same content hash, so the posterior must be
    # unaffected. Conjugate mean for Normal(0, 3.7182818) prior with one
    # observation at 3.0 and likelihood sd 1.
    prec = 1 / (@sigma * @sigma) + 1.0
    truth = 3.0 / prec
    mean = Enum.sum(vals) / length(vals)

    assert abs(mean - truth) < 0.5,
           "posterior mean #{Float.round(mean, 3)} vs analytic #{Float.round(truth, 3)} — " <>
             "recovery rebuilt a shader that computes something else"

    # NOT `assert File.exists?(spv)` here. The killer's last deletion can land
    # after the last dispatch, so the file's presence at the end of the run is
    # itself a race — an earlier version of this test asserted it and failed
    # for that reason while the recovery it was testing had worked perfectly.
    # Ask the recovery path directly instead; it is deterministic.
    assert Exmc.NUTS.CustomSynth.Compile.ensure!(spv) == :ok
    assert File.exists?(spv), "ensure!/1 returned :ok without producing the artifact"
  end

  test "ensure!/1 reports a path it cannot rebuild rather than pretending" do
    # A hash this VM has never synthesised: nothing remembers the source, so
    # recovery must fail explicitly. Silently returning :ok here would turn a
    # missing shader into a dispatch error one frame later.
    bogus =
      Path.join(
        Path.dirname(
          (fn ->
             {:ok, {:synthesised, _, _, _, p, _}} = Exmc.NUTS.CustomSynth.synthesise(ir())
             p
           end).()
        ),
        "synth_#{String.duplicate("ab", 32)}.spv"
      )

    refute File.exists?(bogus)

    assert {:error, {:no_remembered_source, ^bogus}} =
             Exmc.NUTS.CustomSynth.Compile.ensure!(bogus)
  end
end
