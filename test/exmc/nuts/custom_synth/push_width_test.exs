defmodule Exmc.NUTS.CustomSynth.PushWidthTest do
  @moduledoc """
  The push block carries no prior parameters, and model width is bounded by
  the shader's thread tile rather than by 128 bytes of push constants.

  ## What this replaces, and why it matters

  `Push.pack/1` used to append one f64 per prior parameter to the 24-byte
  header, and reject the result past 128 bytes. That capped the fused chain
  path at roughly 13 free RVs with one-parameter priors, 6 with Normal, 3
  with TruncatedNormal. Anything wider degraded silently to per-op sampling.

  The tail was read by nothing:

    * `MultiRvCustomSpec` bakes prior parameters into the generated GLSL as
      literals at synthesis time. Disassembling a cached SPV for a
      `Normal(0.0, 7.3125)` prior shows `OpConstant %double 7.3125` and its
      precomputed normalisation term, with a push struct of exactly
      `OpTypeStruct %uint %uint %uint %uint %double`.
    * `leapfrog_chain_synth_f64` pushes `sizeof(PushBlockF64) = 24` bytes and
      drops the rest.

  But the NIF rejects `push.len() > 128` *before* dispatching. So the unread
  tail was counted against a budget it never spent, and it was the sole cause
  of the width cap. Measured on the 8-RV conjugate model below: **0 chain
  dispatches and 160.9 s** with the tail, **2564 dispatches and 12.3 s**
  without — 13.1x, with the posterior unchanged and correct on both arms.

  ## Non-vacuity

  The dispatch-count assertions here are worthless if this host cannot
  dispatch at all — zero-versus-zero reads as a pass. Every Vulkan test below
  therefore asserts a *narrow* model dispatches first, so a box with no
  working chain path fails loudly instead of passing by being uniformly
  broken. Same discipline as `chain_meta_routing_test.exs`.
  """
  use ExUnit.Case, async: false

  alias Exmc.Builder
  alias Exmc.Dist.Normal
  alias Exmc.NUTS.CustomSynth.Push
  alias Exmc.NUTS.Sampler
  alias Exmc.NUTS.Vulkan.Dispatch

  # mu_i ~ Normal(0, 2); y_i ~ Normal(mu_i, 1); observe y_i = v_i
  # Conjugate, so the posterior is closed form:
  #   precision = 1/4 + 1/1 = 1.25;  mean = 0.8 * v_i;  sd = sqrt(1/1.25)
  @vs [3.0, -2.0, 1.5, 0.5, -1.0, 2.5, -0.5, 4.0]
  @truth_sd :math.sqrt(1 / 1.25)

  defp conjugate_ir(vs) do
    vs
    |> Enum.with_index(1)
    |> Enum.reduce(Builder.new_ir(), fn {v, i}, acc ->
      acc
      |> Builder.rv("mu#{i}", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(2.0)})
      |> Builder.rv("y#{i}", Normal, %{mu: "mu#{i}", sigma: Nx.tensor(1.0)})
      |> Builder.obs("y#{i}_obs", "y#{i}", Nx.tensor(v))
    end)
  end

  defp narrow_ir do
    Builder.new_ir()
    |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(2.0)})
    |> Builder.rv("y", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
    |> Builder.obs("y_obs", "y", Nx.tensor(3.0))
  end

  describe "pack/1 emits the header and nothing else" do
    test "block is 24 bytes regardless of how many priors the model has" do
      for n <- [1, 2, 4, 8, 16] do
        spec = %{
          K: 32,
          n_obs: 1,
          d: n,
          eps: 0.05,
          priors:
            for i <- 1..n do
              {"mu#{i}", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(2.0)}}
            end
        }

        assert {:ok, bin, 24} = Push.pack(spec),
               "pack/1 must emit the fixed 24-byte header for #{n} priors"

        assert byte_size(bin) == 24
      end
    end

    test "the header still carries K, n_obs, d and eps in NIF order" do
      spec = %{K: 7, n_obs: 3, d: 5, eps: 0.25, priors: []}
      {:ok, bin, 24} = Push.pack(spec)

      # parse_push_block_f64: k_steps@0, n_obs@4, d@8, _pad@12, eps@16
      assert <<7::little-unsigned-32, 3::little-unsigned-32, 5::little-unsigned-32,
               0::little-unsigned-32, 0.25::little-float-64>> == bin
    end

    test "16 Normal priors would have overflowed the old 128-byte cap" do
      # 16 priors x 2 floats x 8 B = 256 B of tail, plus a 24 B header.
      # This is the arithmetic that used to reject the model outright; it is
      # here so the regression is visible if the tail is ever reinstated.
      assert 24 + 16 * 2 * 8 > Push.max_bytes()
      assert {:ok, _bin, 24} = Push.pack(%{K: 32, n_obs: 1, d: 16, eps: 0.05, priors: []})
    end
  end

  describe "the real width bound is the shader's thread tile" do
    @describetag :requires_vulkan

    # Removing the push cap made this boundary reachable for the first time.
    # It matters more than it looks: past d = 256 the chain shaders returned
    # buffers with an undefined tail (handed back whole, never sliced to a
    # logical size) and the logp tree reduce summed only the first 256
    # elements — so the log-probability was WRONG rather than truncated, and
    # would have surfaced as a sampler bug rather than a dispatch error.
    #
    # That was harmless only by accident: the push-block budget kept d near
    # 13, so nothing ever got close. Removing the accident is what made the
    # guard necessary. nx_vulkan now also refuses d > 256 at the NIF
    # (:bad_input); this asserts our half, which refuses earlier and
    # degrades to per-op instead of raising.
    defp wide_ir(n) do
      1..n
      |> Enum.reduce(Builder.new_ir(), fn i, acc ->
        Builder.rv(acc, "m#{i}", Exmc.Dist.HalfNormal, %{sigma: Nx.tensor(1.0)})
      end)
      |> Builder.obs("y", "m1", Nx.tensor(1.0))
    end

    test "d = 256 synthesises and dispatches finite, correctly sized chains" do
      # n_rv = 257 because observing m1 leaves 256 free.
      assert {:ok, {:synthesised, _sha, layout, _spec, _spv, _obs} = meta} =
               Exmc.NUTS.ChainShaderCodegen.detect_meta(wide_ir(257), [])

      assert length(layout) == 256

      d = 256
      k = 32
      ones = fn v -> Nx.tensor(List.duplicate(v, d), type: :f64) end

      {qc, pc, lc, gc} = Dispatch.chain(meta, d, 0.01, ones.(1.0), ones.(0.5), ones.(-0.25), k, 1)

      # Sizes: the undefined-tail bug showed here first.
      assert Nx.size(qc) == k * d
      assert Nx.size(pc) == k * d
      assert Nx.size(gc) == k * d
      assert Nx.size(lc) == k, "logp is one scalar per step, not per dimension"

      # And the values: a wrong logp reduce produced plausible numbers, so
      # finiteness is the check that would have caught it, not the shape.
      for {t, name} <- [{qc, "q"}, {pc, "p"}, {gc, "grad"}, {lc, "logp"}] do
        assert Enum.all?(
                 Nx.to_flat_list(t),
                 &(is_float(&1) and &1 == &1 and abs(&1) != :infinity)
               ),
               "#{name}_chain contains NaN or infinity at d = 256"
      end
    end

    test "d = 257 is refused at synthesis, not left to fail at dispatch" do
      assert {:unsupported, :d_exceeds_tile} =
               Exmc.NUTS.ChainShaderCodegen.detect_meta(wide_ir(258), [])
    end
  end

  describe "an 8-RV model reaches the fused chain shader" do
    @describetag :requires_vulkan

    test "it synthesises rather than degrading to per-op" do
      assert {:ok, {:synthesised, _sha, layout, _spec, _spv, _obs}} =
               Exmc.NUTS.ChainShaderCodegen.detect_meta(conjugate_ir(@vs), []),
             "an 8 free-RV model must synthesise; it used to return " <>
               "{:unsupported, :push_too_large} and sample per-op at ~13x the cost"

      assert length(layout) == 8
    end

    test "it dispatches, and its posterior matches the closed form" do
      # Non-vacuity first: if this host cannot dispatch a narrow model, the
      # assertion below would pass at zero for the wrong reason.
      Dispatch.reset_dispatch_count()
      Sampler.sample(narrow_ir(), %{}, num_warmup: 20, num_samples: 20, seed: 1)
      narrow = Dispatch.dispatch_count()

      assert narrow > 0,
             "the 1-RV model issued no chain dispatches on this host, so the " <>
               "8-RV assertion below would be vacuous"

      Dispatch.reset_dispatch_count()

      {trace, _stats} =
        Sampler.sample(conjugate_ir(@vs), %{}, num_warmup: 300, num_samples: 800, seed: 7)

      wide = Dispatch.dispatch_count()

      assert wide > 0,
             "the 8-RV model issued #{wide} chain dispatches against #{narrow} " <>
               "for the 1-RV model — it is still falling back to per-op sampling"

      # Correctness, not just speed: a faster wrong posterior is the failure
      # mode that matters here, since the shader is now doing work it was
      # previously never asked to do at this width.
      mcse = @truth_sd / :math.sqrt(800)

      for {v, i} <- Enum.with_index(@vs, 1) do
        draws = trace["mu#{i}"] |> Nx.to_flat_list()
        mean = Enum.sum(draws) / length(draws)
        truth = v * 0.8

        assert abs(mean - truth) < 5 * mcse,
               "mu#{i}: posterior mean #{Float.round(mean, 4)} vs analytic " <>
                 "#{Float.round(truth, 4)}, off by #{Float.round(abs(mean - truth) / mcse, 2)} MCSE"
      end
    end
  end
end
