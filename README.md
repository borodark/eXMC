# Exmc

**Probabilistic programming for the BEAM.** A from-scratch Elixir implementation of PyMC's architecture: declarative model specification, automatic constraint transforms, NUTS sampling, and Bayesian diagnostics -- all running on Nx tensors with optional EXLA acceleration.

![Live Streaming Dashboard](assets/live_streaming.png)

## Why Elixir?

PyMC is brilliant. It is also a Python library, which means it inherits Python's concurrency story: none. Running four MCMC chains means four OS processes, four copies of the model in memory, four separate GIL-bound interpreters.

Exmc runs on the BEAM. Four chains are four lightweight processes sharing one compiled model. `Task.async_stream` dispatches them across all cores with zero serialization overhead. The Erlang `:rand` module provides microsecond PRNG with explicit state threading -- no process dictionary, no global mutation, fully deterministic given a seed.

The result: a probabilistic programming framework where concurrency is not bolted on but falls naturally out of the runtime.

## Architecture

```
Builder.new_ir()                        # 1. Declare
|> Builder.rv("mu", Normal, params)     #    your model
|> Builder.rv("sigma", HalfNormal, ...) #    as an IR graph
|> Builder.obs("y", "x", data)          #
                                        #
Rewrite.run(ir, passes)                 # 2. Rewrite passes:
  # affine -> meas_obs                  #    NCP, measurable ops,
  # non-centered parameterization       #    constraint transforms
                                        #
Compiler.compile_for_sampling(ir)       # 3. Compile to:
  # => {vag_fn, step_fn, pm, ncp_info}  #    logp + gradient closure
                                        #    (EXLA JIT when available)
                                        #
Sampler.sample(ir, init, opts)          # 4. NUTS with Stan-style
  # => {trace, stats}                   #    three-phase warmup
```

Four layers, each a clean boundary:

| Layer | Modules | Responsibility |
|-------|---------|----------------|
| **IR** | `Builder`, `IR`, `Node`, `Dist.*` | Model as data. 9 distributions, 3 node types |
| **Compiler** | `Compiler`, `PointMap`, `Transform`, `Rewrite` | IR to differentiable closure. Transforms, Jacobians, NCP |
| **NUTS** | `Leapfrog`, `Tree`, `MassMatrix`, `StepSize` | Multinomial NUTS (Betancourt 2017) with diagonal mass |
| **Sampler** | `Sampler`, `Diagnostics`, `Predictive` | Orchestration, warmup, ESS, R-hat, prior/posterior predictive |

## Quick Start

```elixir
alias Exmc.{Builder, Dist.Normal, Dist.HalfNormal}

# Define a hierarchical model
ir =
  Builder.new_ir()
  |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(5.0)})
  |> Builder.rv("sigma", HalfNormal, %{sigma: Nx.tensor(2.0)})
  |> Builder.rv("x", Normal, %{mu: "mu", sigma: "sigma"})
  |> Builder.obs("x_obs", "x",
    Nx.tensor([2.1, 1.8, 2.5, 2.0, 1.9, 2.3, 2.2, 1.7, 2.4, 2.6])
  )

# Sample
{trace, stats} = Exmc.NUTS.Sampler.sample(ir,
  %{"mu" => 2.0, "sigma" => 1.0},
  num_samples: 1000, num_warmup: 500
)

# Posterior mean
Nx.mean(trace["mu"]) |> Nx.to_number()
# => ~2.1

# Parallel chains (compile once, run on all cores)
{traces, stats_list} = Exmc.NUTS.Sampler.sample_chains(ir, 4,
  init_values: %{"mu" => 2.0, "sigma" => 1.0}
)
```

## Distributions

| Distribution | Support | Transform | Params |
|-------------|---------|-----------|--------|
| `Normal` | R | none | `mu`, `sigma` |
| `HalfNormal` | R+ | `:log` | `sigma` |
| `Exponential` | R+ | `:log` | `rate` |
| `Gamma` | R+ | `:softplus` | `alpha`, `beta` |
| `Beta` | (0,1) | `:logit` | `alpha`, `beta` |
| `Uniform` | (a,b) | `:logit` | `low`, `high` |
| `StudentT` | R | none | `nu`, `mu`, `sigma` |
| `Cauchy` | R | none | `mu`, `sigma` |
| `LogNormal` | R+ | `:log` | `mu`, `sigma` |

## Key Features

**Automatic Non-Centered Parameterization.** Hierarchical Normals where both `mu` and `sigma` are string references are automatically rewritten to `z ~ N(0,1)` with `x = mu + sigma * z`. Eliminates funnel geometry without user intervention.

**EXLA Auto-Detection.** When EXLA is available, `value_and_grad` is JIT-compiled and the entire leapfrog step is fused into a single XLA computation. Falls back to BinaryBackend transparently.

**Vectorized Observations.** Pass `Nx.tensor([...])` to `Builder.obs` -- reduction is handled automatically. No need to create one RV+obs pair per data point.

**Model Comparison.** WAIC and LOO-CV via `Exmc.ModelComparison.compare([{model_a, trace_a}, {model_b, trace_b}])`.

**Prior & Posterior Predictive.** `Exmc.Predictive.prior_samples(ir, 500)` and `posterior_predictive(ir, trace)` for model checking.

## Test Suite

123 tests (11 doctests + 112 tests), including 25 integration tests covering:

- Conjugate posteriors (Normal-Normal, Gamma-Exponential)
- Hierarchical models up to 5 free parameters
- All 9 constrained distributions
- Non-centered parameterization
- WAIC/LOO model comparison
- Vectorized observations
- Parallel multi-chain sampling

## 35 Architectural Decisions

Every non-trivial choice is recorded in [`DECISIONS.md`](DECISIONS.md) with rationale, assumptions, and implications. From "why `:rand` instead of `Nx.Random`" (D16: 1000x faster on BinaryBackend) to "why auto-NCP" (D32: eliminates funnel geometry) to "why compile once for parallel chains" (D35: no redundant JIT compilation).

## Companion: ExmcViz

See [`../exmc_viz/`](../exmc_viz/) for native ArviZ-style diagnostics -- trace plots, histograms, ACF, pair plots, forest plots, energy diagnostics, and live streaming visualization during sampling.

![Pair Plot](assets/pair_plot_4k.png)

## License

Exmc is licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0).

You are free to use, modify, and distribute this software under AGPL terms. If you run a modified version as a network service, you must make your source code available to users of that service.

**Commercial licensing** is available for organizations that need to embed Exmc in proprietary products without AGPL obligations. Contact us for terms.
