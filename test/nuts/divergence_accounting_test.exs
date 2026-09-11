defmodule Exmc.NUTS.DivergenceAccountingTest do
  @moduledoc """
  `stats.divergences` counts the SAMPLING phase only.

  It used to include warmup. `run_sampling/8` was handed warmup's final state,
  counters and all, so the field reported both phases — while every consumer
  divides it by `num_samples`. `benchmark/posteriordb/validate_posteriordb.exs`
  prints lines like `div=94/4000` where 4000 is `chains * num_samples`, so the
  numerator spanned warmup and sampling and the denominator only sampling. That
  is not a rate of anything, and the harness's `@div_rate_max` gate sat on it.

  Divergences during warmup are EXPECTED — dual averaging deliberately explores
  step sizes that diverge — so counting them in a diagnostic whose whole purpose
  is "did the sampler struggle on this posterior" answers a different question
  from the one being asked.

  Reported by the pathmc_ex session, which saw 10 and 11 against 0 flagged
  draws and could not tell from the name whether that was intended.

  ## The invariant, and which arm actually gates it

  `stats.divergences` must equal the number of kept draws flagged
  `divergent: true`.

  The two arms below do DIFFERENT jobs, and an earlier version of this comment
  had them the wrong way round — caught by running the mutation rather than by
  reasoning about it:

  * **`num_warmup: 1000` is the GATE.** With the bug present it reports 14
    against 0 flagged draws, so the assertion fails. This is the arm that turns
    red if the counters are ever inherited again.
  * **`num_warmup: 0` is the DIAGNOSTIC, and it cannot gate.** With the bug it
    reads 2 against 2 — sampling is the only phase that can contribute, so both
    sides agree and the test passes either way. Its value was evidential: it
    proved the excess in the other arms was warmup and not miscounting, which
    is what turned "appears to include warmup" into a fact.

  Keep both, for those two separate reasons. Removing the long-warmup arm
  removes the gate; removing the zero-warmup arm removes the evidence that the
  gate is measuring what it claims.
  """
  use ExUnit.Case, async: false

  alias Exmc.{Builder, Dist, IR}

  @f64 [type: :f64]

  defp unit_normal do
    IR.new()
    |> Builder.rv("mu", Dist.Normal, %{mu: Nx.tensor(0.0, @f64), sigma: Nx.tensor(1.0, @f64)})
  end

  defp run(vectorized, num_warmup, num_samples) do
    {_traces, stats} =
      Exmc.NUTS.Sampler.sample_chains(unit_normal(), 2,
        init_values: %{"mu" => 0.0},
        num_warmup: num_warmup,
        num_samples: num_samples,
        vectorized: vectorized
      )

    Enum.map(stats, fn st ->
      flagged = st |> Map.get(:sample_stats, []) |> Enum.count(& &1[:divergent])
      {st.divergences, flagged, length(Map.get(st, :sample_stats, []))}
    end)
  end

  for vectorized <- [false, true] do
    describe "vectorized: #{vectorized}" do
      @vectorized vectorized

      # The DIAGNOSTIC arm, not the gate — see the moduledoc. With no warmup,
      # sampling is the only phase that can contribute, so both sides agree
      # whether or not the counters are inherited. It is here because it is the
      # evidence that the excess seen in the long-warmup arm is warmup, and not
      # the sampler miscounting.
      test "with no warmup, the reported count equals the flagged draws" do
        for {reported, flagged, kept} <- run(@vectorized, 0, 300) do
          assert kept == 300
          assert reported == flagged,
                 "reported #{reported} divergences, #{flagged} kept draws flagged"
        end
      end

      # THE GATE. Long warmup on a unit Normal: adaptation diverges while it
      # finds a step size, and none of that may reach the field. With the
      # counters inherited this reads 14 against 0 and fails.
      test "a long warmup contributes nothing to the count" do
        for {reported, flagged, _kept} <- run(@vectorized, 1000, 300) do
          assert reported == flagged,
                 "reported #{reported} divergences, #{flagged} kept draws flagged — " <>
                   "warmup divergences are leaking into the sampling count"
        end
      end
    end
  end

  describe "the field is a rate over num_samples" do
    # What consumers actually compute. Stated as a test so that changing the
    # field's meaning again breaks something that names the consequence, rather
    # than silently re-inflating every div= line in the posteriordb reports.
    test "divergences never exceeds the number of kept draws" do
      for warmup <- [0, 500] do
        for {reported, _flagged, kept} <- run(false, warmup, 200) do
          assert reported <= kept,
                 "#{reported} divergences against #{kept} kept draws — the field " <>
                   "cannot be divided by num_samples to give a rate"
        end
      end
    end
  end
end
