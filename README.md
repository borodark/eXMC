# Exmc

**A new PPL environment on the BEAM, inspired by PyMC.** Exmc is a from-scratch Elixir implementation of PyMC’s architecture: declarative model specification, automatic constraint transforms, NUTS sampling, and Bayesian diagnostics — all on Nx tensors with optional EXLA acceleration.

**With deep respect:** this project builds on the ideas, rigor, and ergonomics pioneered by the PyMC community. The goal is not to replace PyMC. The goal is to preserve correctness and usability while exploring what changes when the runtime is the BEAM.

![Live Streaming Dashboard](assets/live_streaming.png)

## Why A New PPL Environment?

PyMC established a high bar for statistical correctness, extensibility, and user experience. Exmc asks a focused question:

**What happens if that architecture runs on a fault-tolerant, massively concurrent runtime?**

The BEAM gives us lightweight processes, isolation, and message passing. That changes how we think about multi-chain sampling, streaming diagnostics, and observability. Exmc keeps PyMC’s model semantics and diagnostics philosophy, while rethinking execution.

## What We Preserve From PyMC

- Model semantics and ergonomics: declarative RVs, clear constraints, sensible defaults.
- Statistical correctness: NUTS with Stan-style warmup, ESS/R-hat, WAIC/LOO.
- Composable diagnostics: traces, energy, autocorrelation, and predictive checks.



## What The BEAM Enables

- True multi-chain concurrency with one compiled model.
- Live streaming diagnostics without polling.
- A live posterior state that can be updated online during sampling.
- Isolated failure domains so one chain can fail without killing the run.

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


![Architecture](assets/architecture.svg)


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

- Automatic Non-Centered Parameterization for hierarchical Normals.
- EXLA auto-detection for `value_and_grad` JIT compilation.
- Vectorized observations via `Nx.tensor([...])`.
- Model comparison with WAIC and LOO-CV.
- Prior and posterior predictive sampling.

## Suggested Screenshots (Placeholders)

Add these images once captured:

- `assets/architecture.png` — IR → Rewrite → Compile → NUTS diagram.
- `assets/live_streaming.png` — live dashboard during sampling.
- `assets/pair_plot_4k.png` — pair plot with correlations.
- `assets/energy_plot.png` — energy marginal + transition plot.

## Architectural Decisions

Every non-trivial choice is recorded in `exmc/DECISIONS.md` with rationale, assumptions, and implications.

## Companion: ExmcViz

See `exmc_viz/` for native ArviZ-style diagnostics — trace plots, histograms, ACF, pair plots, forest plots, energy diagnostics, and live streaming visualization during sampling.

![Pair Plot](assets/pair_plot_4k.png)
