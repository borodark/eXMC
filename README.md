# eXMC

**A PPL environment on the BEAM, inspired by PyMC.** eXMC is a from-scratch Elixir implementation of PyMC's architecture: declarative model specification, automatic constraint transforms, NUTS sampling, and Bayesian diagnostics — all on Nx tensors with optional EXLA acceleration.

**With deep respect:** this project builds on the ideas, rigor, and ergonomics pioneered by the PyMC community. The goal is not to replace PyMC. The goal is to preserve correctness and usability while exploring what changes when the runtime is the BEAM.

![Live Streaming Dashboard](assets/live_streaming.png)

## Why A New PPL Environment?

PyMC established a high bar for statistical correctness, extensibility, and user experience. eXMC asks a focused question:

**What happens if that architecture runs on a fault-tolerant, massively concurrent runtime?**

The BEAM gives us lightweight processes, isolation, and message passing. That changes how we think about multi-chain sampling, streaming diagnostics, and observability. eXMC keeps PyMC's model semantics and diagnostics philosophy, while rethinking execution.

## What We Preserve From PyMC

- Model semantics and ergonomics: declarative RVs, clear constraints, sensible defaults.
- Statistical correctness: NUTS with Stan-style three-phase warmup, ESS/R-hat, WAIC/LOO.
- Composable diagnostics: traces, energy, autocorrelation, and predictive checks.

## What The BEAM Enables

- **Concurrency without copies.** Four chains are four lightweight processes sharing one compiled model. No `cloudpickle`, no `multiprocessing.Pool`, no four copies of the interpreter. `Task.async_stream` dispatches them across all cores.
- **Per-sample streaming.** `sample_stream/4` sends each posterior sample as a message to any BEAM process — a Scenic window, a Phoenix LiveView, a GenServer computing running statistics. [Nutpie](https://github.com/pymc-devs/nutpie) sets the standard for live MCMC UX with rich terminal progress bars (per-chain draws, divergences, step size, gradients/draw), `blocking=False` with pause/resume/abort, and access to incomplete traces. eXMC takes a different approach: instead of a built-in terminal display, it streams individual samples as BEAM messages, composing with whatever visualization layer you choose — Scenic for native desktop, Phoenix LiveView for browser, or a custom GenServer for online statistics.
- **Fault isolation.** A chain that hits a numerical singularity — NaN gradient, EXLA crash, memory fault — is caught and replaced with a divergent placeholder. The other chains keep running. The supervisor tree doesn't care.
- **Distribution as a language primitive.** `Distributed.sample_chains/2` sends model IR to remote `:peer` nodes via `:erpc`. Each node compiles independently (heterogeneous hardware). If a node dies, the chain retries on the coordinator automatically. Zero external infrastructure.

## Performance

Seven-model benchmark against PyMC (1-chain, 5-seed medians, 1000 warmup + 1000 draws):

```
                PyMC ESS/s   eXMC ESS/s   Ratio    Winner
                ──────────────────────────────────────────
simple (d=2)         576          469      0.81x    PyMC
medium (d=5)         157          298      1.90x    eXMC
stress (d=8)         185          215      1.16x    eXMC
eight_schools (d=10)   5           12      2.55x    eXMC
funnel (d=10)          6            2      0.40x    PyMC
logistic (d=21)      336           69      0.21x    PyMC
sv (d=102)             1            1      1.20x    eXMC
```

**eXMC wins 4 models to PyMC's 3**, including the canonical Eight Schools benchmark (2.55x) and 102-dimensional stochastic volatility (1.20x). PyMC wins on throughput-bound models where compiled C++ per-step speed dominates. eXMC wins on adaptation-bound models where posterior geometry is hard.

With 5-node distribution, eXMC achieves 2.88x average scaling:

```
                1ch ESS/s   5-node ESS/s   PyMC 4ch   Dist vs PyMC
                ─────────────────────────────────────────────────────
medium              271           841          680       1.24x eXMC
funnel              1.6           5.4          4.1       1.32x eXMC
```

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
```

### DSL Syntax

```elixir
use Exmc.DSL

ir = model do
  rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(5.0)})
  rv("sigma", HalfNormal, %{sigma: Nx.tensor(2.0)})
  rv("x", Normal, %{mu: "mu", sigma: "sigma"})
  obs("x_obs", "x", Nx.tensor([2.1, 1.8, 2.5]))
end
```

### Multi-Chain

```elixir
# Parallel chains — compile once, run on all cores
{traces, stats_list} = Exmc.NUTS.Sampler.sample_chains(ir, 4,
  init_values: %{"mu" => 2.0, "sigma" => 1.0}
)
```

### Streaming

```elixir
# Stream samples to any process — LiveView, GenServer, Scenic window
Exmc.NUTS.Sampler.sample_stream(ir, self(), %{"mu" => 2.0, "sigma" => 1.0},
  num_warmup: 500, num_samples: 1000
)

# Receive samples as they arrive
receive do
  {:exmc_sample, point, stat} -> IO.inspect(point["mu"])
end
```

### Distributed

```elixir
# Spawn peer nodes and sample across them — zero infrastructure
{:ok, _pid, node1} = :peer.start_link(%{name: :worker1})
{:ok, _pid, node2} = :peer.start_link(%{name: :worker2})

{traces, stats_list} = Exmc.NUTS.Distributed.sample_chains(ir,
  nodes: [node(), node1, node2],
  init_values: %{"mu" => 2.0, "sigma" => 1.0}
)
# Node dies? Chain retries on coordinator automatically.
```

## Inference Methods

| Method | Module | Use Case |
|--------|--------|----------|
| **NUTS** | `Exmc.NUTS.Sampler` | Gold standard. Stan-style three-phase warmup, multinomial trajectory sampling, rho-based U-turn criterion |
| **ADVI** | `Exmc.ADVI` | Fast approximate posterior. Mean-field normal in unconstrained space, stochastic gradient ELBO |
| **SMC** | `Exmc.SMC` | Multimodal posteriors. Likelihood tempering with Metropolis-Hastings transitions |
| **Pathfinder** | `Exmc.Pathfinder` | L-BFGS path toward mode with diagonal normal fit at each step. Fast initialization for NUTS |

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
| `Laplace` | R | none | `mu`, `b` |
| `MvNormal` | R^d | none | `mu` (vector), `cov` (matrix) |
| `GaussianRandomWalk` | R^T | none | `sigma` |
| `Dirichlet` | Δ^K (simplex) | `:stick_breaking` | `alpha` (vector) |
| `Custom` | any | user-defined | user-defined closure |
| `Mixture` | any | component-based | `weights`, `components` |
| `Censored` | any | wraps base dist | `dist`, `lower`, `upper` |

## Key Features

- **Automatic Non-Centered Parameterization.** Hierarchical Normals where both `mu` and `sigma` are parent references are rewritten to `z ~ N(0,1)` with `x = mu + sigma * z`. Disable with `ncp: false` when data is informative.
- **EXLA auto-detection.** When EXLA is available, `value_and_grad` is JIT-compiled. Falls back to BinaryBackend transparently. GPU via `device: :cuda`.
- **Vectorized observations.** Pass `Nx.tensor([...])` to `Builder.obs` — reduction is handled automatically. No need to create one RV per data point.
- **Model comparison.** WAIC and LOO-CV via `Exmc.ModelComparison.compare/1`.
- **Prior and posterior predictive.** `Exmc.Predictive.prior_samples/2` and `posterior_predictive/2` for model checking.
- **Custom distributions.** `Exmc.Dist.Custom` takes a `logpdf` closure — any differentiable density. Used for Bernoulli likelihoods, random walk models, and domain-specific densities.
- **Fault-tolerant tree building.** Four layers: IEEE 754 NaN/Inf detection, subtree early termination, trajectory-level divergence tracking, process-level crash recovery via `try/rescue`.
- **Deterministic seeding.** Erlang `:rand` with explicit state threading. Every chain is reproducible given `{seed, tuning_params, ir}`.

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
| **IR** | `Builder`, `DSL`, `IR`, `Node`, `Dist.*` | Model as data. 16 distributions (3 vector-valued), 3 node types |
| **Compiler** | `Compiler`, `PointMap`, `Transform`, `Rewrite` | IR to differentiable closure. Transforms, Jacobians, NCP |
| **NUTS** | `Leapfrog`, `Tree`, `MassMatrix`, `StepSize` | Multinomial NUTS (Betancourt 2017) with diagonal/dense mass |
| **Sampler** | `Sampler`, `Distributed`, `Diagnostics`, `Predictive` | Orchestration, warmup, ESS, R-hat, streaming, distribution |

![Architecture](assets/architecture.svg)

## Diagnostics

```elixir
# Summary statistics
Exmc.Diagnostics.summary(trace)
# => %{"mu" => %{mean: 2.15, std: 0.31, q5: 1.63, q50: 2.14, q95: 2.68}, ...}

# Effective sample size and R-hat
Exmc.Diagnostics.ess(trace["mu"])
Exmc.Diagnostics.rhat([trace1["mu"], trace2["mu"]])

# Model comparison
Exmc.ModelComparison.compare([
  {"model_a", Exmc.ModelComparison.waic(ll_a)},
  {"model_b", Exmc.ModelComparison.waic(ll_b)}
])
```

## Companion: ExmcViz

See [`../exmc_viz/`](../exmc_viz/) for native ArviZ-style diagnostics built on [Scenic](https://github.com/ScenicFramework/scenic) — trace plots, histograms, ACF, pair plots, forest plots, energy diagnostics, and live streaming visualization during sampling.

```elixir
ExmcViz.show(trace, stats)                    # static dashboard
ExmcViz.stream(ir, init, num_samples: 5000)   # live sampling dashboard
```

![Pair Plot](assets/pair_plot_4k.png)

## Backends

eXMC's tensor operations go through [Nx](https://github.com/elixir-nx/nx), with backend-specific acceleration:

| Backend | Platform | Status |
|---------|----------|--------|
| [EXLA](https://github.com/elixir-nx/nx/tree/main/exla) | CPU, CUDA GPU | Supported. JIT-compiled gradients, `device: :cuda` for GPU |
| [EMLX](https://github.com/elixir-nx/emlx) | Apple Silicon (Metal) | Planned. MLX backend for M-series Macs ([#1](https://github.com/borodark/exmc/issues/1)) |
| BinaryBackend | Any | Fallback. Pure Elixir, no dependencies |

## Architectural Decisions

Every non-trivial choice is recorded in [`DECISIONS.md`](DECISIONS.md) with rationale, assumptions, and implications. From "why `:rand` instead of `Nx.Random`" to "why auto-NCP" to "why compile once for parallel chains."

## License

eXMC is dual-licensed:

- **Community License:** [Apache 2.0](LICENSE_COMMUNITY.md) — free and open source
- **Commercial License:** [Proprietary](LICENSE_COMMERCIAL.md) — for enterprises, OEM embedding, and closed-source use

Choose one. If you use the Community License, you must comply with Apache 2.0. If you need proprietary use, hosted/SaaS deployment, or enterprise support, contact sales@octanix.com for a Commercial License.
