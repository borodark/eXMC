# Tolerance audit — raw output

`bench/tolerance_audit.exs`. Each test appears twice: `was` is the
configuration and assertion the suite carried before 2026-08-18, `now` is what
it carries after. The `4sd floor` column is the point — it is the smallest
variance error a 4-sigma analytic gate can detect at that chain's effective
sample size, so it is the resolution limit no tolerance rewrite can beat.

Host: super-io — Intel Xeon E5-2699 v4 (88 threads), NVIDIA GeForce RTX 3060 Ti,
Linux 6.8.0-137-generic x86_64, Erlang/OTP 27 (erts 15.2.7.2), Elixir 1.18.3.
`compiler: :none` (Nx.BinaryBackend). Measured 2026-08-18.

Analysis and the per-test verdicts: [`docs/TOLERANCE_AUDIT.md`](../docs/TOLERANCE_AUDIT.md).

```
=== what the suite's statistical tolerances admit ===
compiler : none
seeds    : each test's own
model / test                                n     ESS    admits(mean)  admits(var)  4sd floor  verdict
------------------------------------------------------------------------------------------------------
integration:15 conjugate N-N      was       500   178    no gate       no gate      37.82%     NO DISPERSION GATE
integration:15 conjugate N-N      now       4000  1280   analytic      analytic     14.46%     resolves 20%
integration:88 Gamma(2,1)         was       200   83     0.71 sd       no gate      86.16%     NO DISPERSION GATE
integration:88 Gamma(2,1)         now       8500  3206   analytic      analytic     15.45%     resolves 20%
integration:113 Exponential(2)    was       300   69     0.60 sd       no gate      53.32%     NO DISPERSION GATE
integration:113 Exponential(2)    now       12000 4916   analytic      analytic     14.72%     resolves 20%
integration:270 Beta(2,5)         was       400   159    0.94 sd       no gate      40.57%     NO DISPERSION GATE
integration:270 Beta(2,5)         now       5000  1405   analytic      analytic     14.47%     resolves 20%
new_dist:230 Lognormal(0,0.5)     was       300   98     0.83 sd       no gate      63.65%     NO DISPERSION GATE
new_dist:230 Lognormal(0,0.5)     now       10000 3640   analytic      analytic     15.64%     resolves 20%
stan_test:8 conjugate N-N         was       500   178    0.50 sd       no gate      37.82%     NO DISPERSION GATE
stan_test:8 conjugate N-N         now       4000  1280   analytic      analytic     14.46%     resolves 20%
nuts_test:308 Normal(0,1)         was       500   172    0.30 sd       100.00%      36.38%     gate far wider than the defect
nuts_test:308 Normal(0,1)         now       4000  1606   analytic      analytic     14.10%     resolves 20%
`4sd floor` is the smallest variance error a 4-sigma analytic gate could
detect at that chain's ESS. Where it exceeds 20%, no tolerance rewrite makes
the test able to see a 20% variance error — it needs more effective draws.
```

## ESS scaling, measured separately

NUTS on `Normal(0,1)`, 500 warmup, `compiler: :none`. `Validator.ess/1` (Geyer)
and `Exmc.Diagnostics.ess/1` agree to within 1%, so the low ESS is the
sampler's autocorrelation and not an artefact of one estimator.

```
draws   Validator.ess   Diagnostics.ess   ess/draw   4sd floor  wall ms
500     172             175               0.344      36.4%      7428
1000    348             349               0.348      29.5%      10387
2000    713             715               0.357      22.7%      16521
4000    1606            1608              0.401      14.1%      28721
```

A 20% resolution on a Normal target needs ESS > 800, i.e. ~2300 draws. On an
Exponential it needs ESS > 3200, because the gate's width scales with
`(mu4 - sigma^4)/sigma^4`, which is 2 for a Normal and 8 for an Exponential.
