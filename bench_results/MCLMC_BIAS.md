
















# MCLMC / MAMS bias and cost — B1.4

**Host:** `super-io`
**Generated:** 2026-08-16T23:30:38.016402Z
**Backend:** `{Nx.BinaryBackend, []}` · compiler `none`
**Elixir/OTP:** 1.18.3 / 27
**Config:** warmup 1000, samples 3000, seeds [1, 2, 3], dims [2, 8, 32]

MCLMC is biased by construction and this file is its gate — a **published
number**, not an assertion. MAMS and NUTS are the unbiased references on the
identical model. `err` columns are relative to the analytic truth; `ESS/grad`
is effective samples per gradient evaluation, summed over coordinates and
chains, divided by the total gradient budget including warmup.

Reproduce with:

```sh
DIMS=2,8,32 SEEDS=1,2,3 \
  WARMUP=1000 SAMPLES=3000 EPS=0.1,0.25,0.5,1.0,2.0,4.0 \
  mix run --no-deps-check bench/mclmc_bias.exs
```



> **PARTIAL RUN — 2026-08-16, `super-io`.** The sweep was killed by a host
> reboot before it finished. Complete and trustworthy: **all three targets at
> `d = 2`**. Partial: **`Normal(0,1)` at `d = 8`**, which has every row except
> NUTS — that one was measured separately afterwards and is appended to the
> block below. Missing entirely: `HalfNormal`/`Exponential` at `d = 8`, and
> **all of `d = 32`**.
>
> `d = 32` is the case the roadmap's "abandon if" condition is actually about
> (`PLAN_SAMPLER_ROADMAP.md` §3). **Do not quote this file as evidence about
> high dimensions.** `NEXT.md` §6 has the command to finish it and the runtime
> to budget for.

## Normal(0,1), d = 2

| sampler | eps | mean | err | var | err | ESS | grads | ESS/grad | div |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MCLMC | 0.100 | -0.0691 | -0.06905 | 1.0144 | 1.44% | 485 | 24000 | 0.0202 | 0 |
| MCLMC | 0.250 | -0.0367 | -0.03673 | 0.9942 | -0.58% | 1013 | 24000 | 0.0422 | 0 |
| MCLMC | 0.500 | -0.0079 | -0.00789 | 0.9907 | -0.93% | 2396 | 24000 | 0.0998 | 0 |
| MCLMC | 1.000 | -0.0092 | -0.00920 | 1.0079 | 0.79% | 4471 | 24000 | 0.1863 | 0 |
| MCLMC | 2.000 | -0.0038 | -0.00379 | 0.9684 | -3.16% | 10600 | 24000 | 0.4417 | 0 |
| MCLMC | 4.000 | 0.0022 | 0.00224 | 0.9277 | -7.23% | 18000 | 24000 | 0.7500 | 0 |
| MCLMC (tuned) | 1.087 | -0.0035 | -0.00349 | 0.9718 | -2.82% | 5480 | 24000 | 0.2283 | 0 |
| MAMS | 2.191 | -0.0004 | -0.00035 | 0.9926 | -0.74% | 9184 | 22028 | 0.4169 | 0 |
| NUTS | 0.903 | 0.0006 | 0.00057 | 1.0088 | 0.88% | 16368 | 37768* | 0.4334 | 20 |

`*` NUTS's warmup gradients are estimated at the sampling-phase mean `n_steps`; its sampling-phase count is exact.
Truth: mean 0.000000, var 1.000000.


## HalfNormal(1), d = 2

| sampler | eps | mean | err | var | err | ESS | grads | ESS/grad | div |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MCLMC | 0.100 | 0.7669 | -3.88% | 0.3446 | -5.17% | 469 | 24000 | 0.0195 | 0 |
| MCLMC | 0.250 | 0.8074 | 1.19% | 0.3529 | -2.88% | 1144 | 24000 | 0.0477 | 0 |
| MCLMC | 0.500 | 0.7965 | -0.17% | 0.3617 | -0.45% | 2636 | 24000 | 0.1098 | 0 |
| MCLMC | 1.000 | 0.7946 | -0.41% | 0.3615 | -0.51% | 5578 | 24000 | 0.2324 | 0 |
| MCLMC | 2.000 | 0.7872 | -1.34% | 0.3488 | -4.01% | 13235 | 24000 | 0.5515 | 0 |
| MCLMC | 4.000 | 0.8222 | 3.05% | 0.4550 | 25.20% | 16655 | 24000 | 0.6940 | 0 |
| MCLMC (tuned) | 1.149 | 0.7993 | 0.18% | 0.3664 | 0.84% | 6162 | 24000 | 0.2568 | 0 |
| MAMS | 1.554 | 0.7913 | -0.82% | 0.3609 | -0.68% | 12207 | 29280 | 0.4169 | 0 |
| NUTS | 0.579 | 0.8016 | 0.46% | 0.3618 | -0.44% | 13016 | 48024* | 0.2710 | 16 |

`*` NUTS's warmup gradients are estimated at the sampling-phase mean `n_steps`; its sampling-phase count is exact.
Truth: mean 0.797885, var 0.363380.


## Exponential(2), d = 2

| sampler | eps | mean | err | var | err | ESS | grads | ESS/grad | div |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MCLMC | 0.100 | 0.4646 | -7.08% | 0.2117 | -15.31% | 380 | 24000 | 0.0158 | 0 |
| MCLMC | 0.250 | 0.5143 | 2.86% | 0.2607 | 4.29% | 2066 | 24000 | 0.0861 | 0 |
| MCLMC | 0.500 | 0.5007 | 0.13% | 0.2478 | -0.87% | 3732 | 24000 | 0.1555 | 0 |
| MCLMC | 1.000 | 0.4876 | -2.48% | 0.2481 | -0.77% | 7222 | 24000 | 0.3009 | 0 |
| MCLMC | 2.000 | 0.4990 | -0.20% | 0.2892 | 15.66% | 15615 | 24000 | 0.6506 | 0 |
| MCLMC | 4.000 | 0.6570 | 31.41% | 1.0215 | 308.60% | 17170 | 24000 | 0.7154 | 0 |
| MCLMC (tuned) | 0.692 | 0.4939 | -1.22% | 0.2503 | 0.10% | 5089 | 24000 | 0.2120 | 0 |
| MAMS | 0.927 | 0.4998 | -0.04% | 0.2518 | 0.72% | 13569 | 48120 | 0.2820 | 0 |
| NUTS | 0.633 | 0.4996 | -0.08% | 0.2411 | -3.58% | 11095 | 53799* | 0.2062 | 61 |

`*` NUTS's warmup gradients are estimated at the sampling-phase mean `n_steps`; its sampling-phase count is exact.
Truth: mean 0.500000, var 0.250000.


## Normal(0,1), d = 8

| sampler | eps | mean | err | var | err | ESS | grads | ESS/grad | div |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MCLMC | 0.100 | 0.0399 | 0.03992 | 1.0523 | 5.23% | 452 | 24000 | 0.0188 | 0 |
| MCLMC | 0.250 | -0.0110 | -0.01099 | 0.9793 | -2.07% | 1809 | 24000 | 0.0754 | 0 |
| MCLMC | 0.500 | -0.0030 | -0.00299 | 1.0195 | 1.95% | 4625 | 24000 | 0.1927 | 0 |
| MCLMC | 1.000 | -0.0054 | -0.00540 | 1.0172 | 1.72% | 9849 | 24000 | 0.4104 | 0 |
| MCLMC | 2.000 | -0.0034 | -0.00338 | 0.9932 | -0.68% | 21635 | 24000 | 0.9014 | 0 |
| MCLMC | 4.000 | -0.0011 | -0.00111 | 0.9653 | -3.47% | 57183 | 24000 | 2.3826 | 0 |
| MCLMC (tuned) | 2.808 | -0.0010 | -0.00103 | 0.9832 | -1.68% | 36619 | 24000 | 1.5258 | 0 |
| MAMS | 4.179 | -0.0034 | -0.00341 | 1.0045 | 0.45% | 37082 | 21860 | 1.6964 | 0 |
