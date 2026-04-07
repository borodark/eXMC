alias Exmc.Poker.{Simulator, OpponentModel}

:rand.seed(:exsss, 42)
true_params = [%{vpip: 0.25, pfr: 0.20, agg: 1.5, bluff: 0.20}]
{obs, _} = Simulator.simulate(true_params, 100)

{ir, _} = OpponentModel.build(obs)
init = OpponentModel.init_values(1)

IO.puts("Compiling model + launching 88 chains x 200 samples...")
t0 = System.monotonic_time(:millisecond)

{traces, stats_list} =
  Exmc.Sampler.sample_chains(ir, 88,
    num_samples: 200,
    num_warmup: 300,
    seed: 42,
    ncp: false,
    init_values: init,
    vectorized: false
  )

t1 = System.monotonic_time(:millisecond)
wall = (t1 - t0) / 1000

total_samples = length(traces) * 200
divs = Enum.map(stats_list, & &1.divergences) |> Enum.sum()

# Combine all chains for parameter recovery
all_profiles =
  Enum.map(traces, fn t ->
    [profile] = OpponentModel.extract_profiles(t, 1)
    profile
  end)

mean_vpip = all_profiles |> Enum.flat_map(&Nx.to_flat_list(&1.vpip)) |> then(&(Enum.sum(&1) / length(&1)))
mean_pfr = all_profiles |> Enum.flat_map(&Nx.to_flat_list(&1.pfr)) |> then(&(Enum.sum(&1) / length(&1)))
mean_agg = all_profiles |> Enum.flat_map(&Nx.to_flat_list(&1.agg)) |> then(&(Enum.sum(&1) / length(&1)))
mean_bluff = all_profiles |> Enum.flat_map(&Nx.to_flat_list(&1.bluff)) |> then(&(Enum.sum(&1) / length(&1)))

IO.puts("")
IO.puts("=== 88-chain poker stress test ===")
IO.puts("Chains:          88")
IO.puts("Samples/chain:   200")
IO.puts("Total samples:   #{total_samples}")
IO.puts("Wall time:       #{Float.round(wall, 1)}s")
IO.puts("Samples/sec:     #{round(total_samples / wall)}")
IO.puts("Total divs:      #{divs}")
IO.puts("")
IO.puts("Parameter recovery (88-chain combined):")
IO.puts("  VPIP:  #{Float.round(mean_vpip, 3)}  (true: 0.25)")
IO.puts("  PFR:   #{Float.round(mean_pfr, 3)}  (true: 0.20)")
IO.puts("  AGG:   #{Float.round(mean_agg, 3)}  (true: 1.50)")
IO.puts("  BLUFF: #{Float.round(mean_bluff, 3)}  (true: 0.20)")
