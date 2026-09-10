# posteriordb Validation Results

**Date:** 2026-09-08T01:53:00.627722Z
**Pass rate:** 6/6 (100.0%)
**Mode:** validate (parallel=88; wall times are contention-bound)
**Protocol:** 1000 warmup + 1000 sampling, seed=42, ncp=false
**Compiler:** requested :exla, resolved EXLA, backend EXLA.Backend, precision :f64
**Host:** super-io (88 schedulers, OTP 27, Elixir 1.18.4)
**exmc:** 45c7b7566fa39afe5b511be238e5fe0d1eaf0370
**nx_vulkan:** bc54f340b9aa4c2cfa8c3b13cbbd1bf7ed06f065

## Summary

| Model | Status | Wall (s) | Min ESS | Max R-hat | Leapfrog | Div | Div % | Step Size | Max Mean Err |
|-------|--------|----------|---------|-----------|----------|-----|-------|-----------|-------------|
| earnings-earn_height | PASS | 338.5 | 903 | 1.002 | 234164 | 99 | 2.5 | 0.0406 | 0.047 |
| eight_schools-eight_schools_noncentered | PASS | 282.9 | 2402 | 1.002 | 28834 | 88 | 2.2 | 0.4625 | 0.029 |
| kidiq-kidscore_momhs | PASS | 287.2 | 1713 | 1.0 | 39178 | 70 | 1.8 | 0.3268 | 0.028 |
| mesquite-logmesquite_logvolume | PASS | 269.3 | 1906 | 1.001 | 24964 | 32 | 0.8 | 0.5241 | 0.012 |
| nes2000-nes | PASS | 327.7 | 1398 | 1.002 | 182692 | 256 | 6.4 | 0.078 | 0.038 |
| sblri-blr | PASS | 285.4 | 999 | 1.002 | 30190 | 43 | 1.1 | 0.2255 | 0.045 |

## Pass Criteria

Statistical, not fixed constants. The previous criteria (mean within 0.5
reference SD, SD within a factor of 2) could not fail for a performance
reason: a model losing 8x its sampling efficiency passed both.

- Split R-hat < 1.01 across 4 chains
- ESS (bulk, rank-normalised) >= 100 per chain
- |mean − reference mean| < 4.0 x MCSE, where MCSE = sd/sqrt(ESS)
- Divergence rate < 10%
- SD within factor of 2 of reference SD (retained, secondary)

The MCSE gate widens as ESS falls, so it cannot detect a slowdown on its
own — the ESS gate is what does that. Both are required.

Reference: Stan gold-standard draws (10 chains x 1000 draws)
