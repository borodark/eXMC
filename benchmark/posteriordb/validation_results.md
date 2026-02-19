# posteriordb Validation Results

**Date:** 2026-02-19T06:27:22.588173Z
**Pass rate:** 33/33 (100.0%)
**Protocol:** 1000 warmup + 1000 sampling, seed=42, ncp=false

## Summary

| Model | Status | Wall (s) | Div | Step Size | Max Mean Err | Max SD Ratio |
|-------|--------|----------|-----|-----------|-------------|-------------|
| earnings-earn_height | PASS | 200.8 | 26 | 0.0325 | 0.107 | 1.316 |
| earnings-log10earn_height | PASS | 203.6 | 22 | 0.0264 | 0.07 | 1.382 |
| earnings-logearn_height | PASS | 202.3 | 20 | 0.0259 | 0.106 | 1.381 |
| earnings-logearn_height_male | PASS | 208.0 | 33 | 0.0222 | 0.095 | 1.357 |
| earnings-logearn_interaction | PASS | 215.1 | 42 | 0.0111 | 0.203 | 1.344 |
| earnings-logearn_interaction_z | PASS | 155.3 | 21 | 0.2352 | 0.117 | 1.149 |
| earnings-logearn_logheight_male | PASS | 221.3 | 42 | 0.0058 | 0.057 | 1.34 |
| eight_schools-eight_schools_noncentered | PASS | 146.6 | 18 | 0.5052 | 0.041 | 1.076 |
| kidiq-kidscore_interaction | PASS | 203.3 | 40 | 0.0256 | 0.071 | 1.49 |
| kidiq-kidscore_momhs | PASS | 157.6 | 18 | 0.2637 | 0.079 | 1.278 |
| kidiq-kidscore_momhsiq | PASS | 185.8 | 25 | 0.069 | 0.005 | 1.192 |
| kidiq-kidscore_momiq | PASS | 176.2 | 19 | 0.0749 | 0.089 | 1.36 |
| kidiq_with_mom_work-kidscore_interaction_c | PASS | 149.6 | 11 | 0.6496 | 0.025 | 1.047 |
| kidiq_with_mom_work-kidscore_interaction_c2 | PASS | 152.5 | 13 | 0.438 | 0.074 | 1.092 |
| kidiq_with_mom_work-kidscore_interaction_z | PASS | 153.4 | 12 | 0.6546 | 0.069 | 1.038 |
| kidiq_with_mom_work-kidscore_mom_work | PASS | 161.5 | 24 | 0.2787 | 0.194 | 1.226 |
| kilpisjarvi_mod-kilpisjarvi | PASS | 221.5 | 31 | 0.0039 | 0.095 | 1.453 |
| mesquite-logmesquite | PASS | 172.7 | 27 | 0.1067 | 0.102 | 1.192 |
| mesquite-logmesquite_logva | PASS | 171.5 | 28 | 0.0917 | 0.085 | 1.145 |
| mesquite-logmesquite_logvas | PASS | 187.7 | 39 | 0.0794 | 0.085 | 1.169 |
| mesquite-logmesquite_logvash | PASS | 189.0 | 40 | 0.0717 | 0.132 | 1.27 |
| mesquite-logmesquite_logvolume | PASS | 143.6 | 8 | 0.4637 | 0.059 | 1.184 |
| mesquite-mesquite | PASS | 198.5 | 43 | 0.057 | 0.066 | 1.137 |
| nes1972-nes | PASS | 183.7 | 56 | 0.0965 | 0.04 | 1.09 |
| nes1976-nes | PASS | 187.6 | 53 | 0.0848 | 0.094 | 1.128 |
| nes1980-nes | PASS | 187.2 | 64 | 0.0876 | 0.075 | 1.11 |
| nes1984-nes | PASS | 186.2 | 61 | 0.0895 | 0.03 | 1.076 |
| nes1988-nes | PASS | 186.7 | 54 | 0.0835 | 0.046 | 1.12 |
| nes1992-nes | PASS | 183.6 | 57 | 0.1005 | 0.047 | 1.094 |
| nes1996-nes | PASS | 189.4 | 68 | 0.0796 | 0.065 | 1.164 |
| nes2000-nes | PASS | 189.4 | 58 | 0.08 | 0.061 | 1.153 |
| sblrc-blr | PASS | 161.7 | 9 | 0.1038 | 0.069 | 1.323 |
| sblri-blr | PASS | 152.5 | 10 | 0.2392 | 0.032 | 1.1 |

## Pass Criteria

- Mean within 0.5 SD of reference mean
- SD within factor of 2 (0.5x-2.0x) of reference SD
- Reference: Stan gold-standard draws (10 chains x 1000 draws)
