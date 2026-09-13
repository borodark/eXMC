defmodule Exmc.ReproducibilityContractTest do
  @moduledoc """
  The reproducibility contract, as tests (docs/REPRODUCIBILITY.md).

  PROMISED, and checked here on every arm the host has: the same seed, arm,
  build and host give bit-identical draws, for one chain and for several; a
  different seed gives different draws; and every result records what
  produced it.

  NOT promised, and deliberately not tested here: identical draws across
  hosts. The host libm returns different bits for log/exp/pow on glibc and
  FreeBSD's msun, and GPU vendors differ in log/exp and division, so long
  chains on different machines diverge. Across hosts the promise is
  statistical, and bench/nuts_truth.exs and bench/nuts_width_race.exs check it
  against closed-form posteriors.

  Bitwise, not `assert_close`: the older `nuts_test.exs` "seed reproducibility"
  compares at 1e-10, which a last-bit difference passes. A last-bit difference
  is exactly what flips an accept/reject decision several hundred draws later,
  so the contract is on the bits. The model has a transformed RV (HalfNormal,
  log transform) so exp/log run on the path, and the draws are long enough for
  a flipped decision to show.
  """

  use ExUnit.Case, async: false

  import Exmc.TestHelper

  alias Exmc.{Builder, Dist}
  alias Exmc.NUTS.Sampler

  @f64 [type: :f64]
  @single [num_warmup: 300, num_samples: 500]
  @chains [num_warmup: 200, num_samples: 300]

  defp ir do
    Builder.new_ir()
    |> Builder.rv("mu", Dist.Normal, %{mu: Nx.tensor(0.0, @f64), sigma: Nx.tensor(2.0, @f64)})
    |> Builder.rv("s", Dist.HalfNormal, %{sigma: Nx.tensor(1.0, @f64)})
  end

  # The trace as raw bytes per variable, so equality is on bits.
  defp bits(trace), do: trace |> Enum.sort() |> Enum.map(fn {k, t} -> {k, Nx.to_binary(t)} end)

  defp assert_contract(label) do
    {t1, s1} = Sampler.sample(ir(), %{}, [seed: 7] ++ @single)
    {t2, _} = Sampler.sample(ir(), %{}, [seed: 7] ++ @single)
    {t3, _} = Sampler.sample(ir(), %{}, [seed: 8] ++ @single)

    assert bits(t1) == bits(t2), "#{label}: same seed, different bits (single chain)"
    refute bits(t1) == bits(t3), "#{label}: a different seed gave identical draws"

    {c1, cs} = Sampler.sample_chains(ir(), 4, [seed: 7] ++ @chains)
    {c2, _} = Sampler.sample_chains(ir(), 4, [seed: 7] ++ @chains)

    assert Enum.map(c1, &bits/1) == Enum.map(c2, &bits/1),
           "#{label}: same seed, different bits (4 chains)"

    # Provenance: the facts a replay needs, on every result.
    assert %{seed: 7, arm: arm, exmc: exmc, otp: otp, arch: arch} = s1.provenance
    assert arm == Exmc.JIT.describe()
    assert is_binary(exmc) and is_binary(otp) and is_binary(arch)
    assert Enum.all?(cs, &match?(%{provenance: %{arm: ^arm}}, &1))

    s1
  end

  # 1_200_000 ms, not 300_000: the CPU arm timed out on the Jetson (two 5 W cores,
  # OTP without JIT) at 300 s during the Rustler 0.38 fleet run, 2026-09-13,
  # before reaching the comparison. The draws are long on purpose (a last-bit
  # difference needs a long chain to flip a decision), so the budget follows
  # the slowest host rather than the chains shrinking.
  describe "CPU arm" do
    setup do
      put_env_scoped(:compiler, :none)
      :ok
    end

    @tag timeout: 1_200_000
    test "same seed gives identical bits; a different seed does not; provenance recorded" do
      stats = assert_contract("cpu")
      assert stats.provenance.device == nil
    end
  end

  describe "the arm this host detects" do
    @tag timeout: 1_200_000
    test "same seed gives identical bits; a different seed does not; provenance recorded" do
      assert_contract("detected: #{Exmc.JIT.describe()}")
    end
  end

  describe "Vulkan arm, through the chain shader" do
    @describetag :requires_vulkan

    setup do
      put_env_scoped(:compiler, :vulkan)
      :ok
    end

    @tag timeout: 1_200_000
    test "same seed gives identical bits; provenance names the device actually open" do
      stats = assert_contract("vulkan")

      assert %{name: name, uuid: uuid, driver: driver, selected_by: by} = stats.provenance.device
      assert is_binary(name) and is_binary(uuid) and is_binary(driver) and is_binary(by)
    end
  end
end
