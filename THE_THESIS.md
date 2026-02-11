# Probabilistic Programming on Actor-Model Runtimes
## Exploiting Fault Tolerance, Distribution, and Liveness for Bayesian Inference

---

## Abstract

We present Exmc, a probabilistic programming framework implemented in Elixir on the BEAM virtual machine. While existing PPLs (PyMC, Stan, NumPyro, Turing.jl) treat sampling as a batch computation on a single node, the actor-model runtime provides three properties absent from current systems: (1) fault-tolerant sampling via supervisor hierarchies, (2) live/streaming inference integrated with production web services, and (3) distributed MCMC across Erlang nodes with zero external infrastructure. We provide a systematic analysis of the PPL design space through 52 documented architectural decisions, quantify the performance implications of JIT boundary placement in mixed-runtime PPLs, and demonstrate that BEAM-specific properties enable fundamentally new inference workflows. On a 7-model benchmark suite spanning d=2 to d=102 — including the canonical Eight Schools, Neal's Funnel, Logistic Regression, and Stochastic Volatility benchmarks — Exmc wins 4 models to PyMC's 3: exceeding PyMC by 2.55x on Eight Schools (d=10), 1.65x on the medium hierarchical model (d=5), 1.25x on the stress model (d=8), and 1.20x on Stochastic Volatility (d=102). PyMC wins on throughput-bound models (simple 0.81x, logistic 0.21x) and pathological geometry (funnel 0.40x). With distributed sampling across 4 :peer nodes, Exmc achieves 3.4–3.7x near-linear scaling, demonstrating that BEAM distribution primitives provide true OS-process parallelism for MCMC without external infrastructure.

---

## Introduction

Probabilistic programming languages (PPLs) enable scientists and engineers to specify Bayesian models declaratively and obtain posterior distributions via automated inference. Over the past decade, systems such as Stan (Carpenter et al., 2017), PyMC (Salvatier et al., 2016), NumPyro (Phan et al., 2019), Turing.jl (Ge et al., 2018), and Pyro (Bingham et al., 2019) have made Bayesian inference accessible to practitioners across disciplines. Yet all of these systems share a common assumption: sampling is a batch computation executed on a single node, producing a complete trace that is analyzed post-hoc.

This assumption forecloses three properties that would be valuable in both research and production settings. First, **fault tolerance**: MCMC sampling runs are long, numerically fragile, and increasingly reliant on accelerator hardware prone to transient failures. No existing PPL can recover gracefully from a mid-sampling crash without restarting the entire run. Second, **liveness**: interactive data science benefits from observing posterior convergence in real time, yet batch PPLs force a "run and wait" workflow where the user sees nothing until sampling completes. Third, **distribution**: scaling Bayesian inference to multiple machines requires external infrastructure (MPI, Dask, Ray) layered atop PPLs that were designed for single-node execution.

The BEAM virtual machine — the runtime underlying Erlang and Elixir — provides all three properties as language-level primitives rather than library additions. Lightweight processes with isolated heaps enable fault containment. Asynchronous message passing enables streaming data flows. Built-in node distribution enables multi-machine communication with no external dependencies. These properties arise from the actor-model concurrency paradigm that the BEAM implements.

**Thesis claims.** We argue that building a PPL on an actor-model runtime yields properties absent from all existing systems:

1. **Fault tolerance.** BEAM supervisor hierarchies and zero-cost `try/rescue` enable crash recovery *within* NUTS tree building. A subtree that encounters a numerical failure, EXLA error, or memory fault is replaced with a divergent placeholder, and sampling continues — with zero overhead on the non-faulting path.

2. **Liveness.** Message-passing enables streaming inference integrated with production web services. A LiveView process can serve as the sample receiver, delivering posterior updates to a browser in real time with no intermediate infrastructure.

3. **Distribution.** Erlang's `:erpc` module enables multi-node MCMC where model IR (plain Elixir data) is distributed to remote nodes, each compiling and sampling independently. Node failure is recovered transparently. The entire distributed system requires zero external infrastructure.

We additionally claim that **JIT boundary placement** is a first-class design concern in mixed-runtime PPLs, quantifiable via affine cost models. In a system where JIT-compiled gradient evaluation coexists with an interpreted tree builder, the placement of the boundary between these regimes determines performance. We provide the first systematic cost model for this boundary, demonstrating a three-category operation taxonomy and concrete crossover dimensions.

**Contributions.** This thesis makes the following contributions:

1. The design and implementation of Exmc, a probabilistic programming framework in Elixir with NUTS, ADVI, SMC, and Pathfinder inference, documented through 52 architectural decisions with falsifiable assumptions.
2. A supervised tree-building mechanism that provides four layers of fault tolerance — from IEEE 754 special value detection through process-level crash recovery — with zero overhead on the non-faulting path (Chapter 3).
3. A streaming inference protocol (`sample_stream`) that integrates naturally with BEAM processes, demonstrated via real-time browser-based posterior visualization through Phoenix LiveView (Chapter 4).
4. Distributed MCMC via Erlang node primitives with transparent fault recovery, requiring 203 lines and zero external infrastructure. A 7-model distributed benchmark demonstrates avg 2.87x scaling from 5 `:peer` nodes, with Exmc distributed beating PyMC 4-chain on medium (1.24x) and funnel (1.32x). The closure barrier — Custom distributions capturing tensor graph fragments — is identified as the distribution boundary; fault recovery provides 2.5-2.8x local concurrency as automatic fallback. The streaming and distributed features compose without integration effort — 4 remote nodes stream samples to a live visualization via the same message-passing primitive, demonstrating the actor model's compositional advantage (Chapters 4–5).
5. A quantitative analysis of JIT boundary placement in mixed-runtime PPLs, including an affine cost model, a three-category operation taxonomy, and a gap decomposition methodology separating architectural overhead from algorithmic quality (Chapter 6).
6. Empirical demonstration that a BEAM-hosted PPL can *exceed* PyMC's sampling efficiency on a 7-model benchmark suite spanning d=2 to d=102, including 4 canonical PPL benchmarks (Eight Schools, Neal's Funnel, Logistic Regression, Stochastic Volatility). Exmc wins 4 models to PyMC's 3, with the adaptation-vs-throughput tradeoff as the determining factor: Exmc wins on hard-geometry models (medium 1.65x, stress 1.25x, Eight Schools 2.55x, SV 1.20x) while PyMC wins on easy-geometry models (simple 0.81x, logistic 0.21x) and pathological geometry (funnel 0.40x). The performance gap was reduced from 34x through parameterization choice, batched JIT computation, targeted FFI boundary placement, multinomial sampling correctness fixes, and algorithm-level improvements including the generalized U-turn criterion (Chapter 6). With distributed 4-node sampling, Exmc achieves 3.4–3.7x near-linear scaling via `:peer` nodes (Chapter 5).

**Chapter overview.** Chapter 1 surveys the PPL design space and positions Exmc within it. Chapter 2 describes the system architecture, compilation pipeline, and key design insights. Chapter 3 demonstrates fault-tolerant sampling via BEAM supervisors. Chapter 4 presents live and streaming inference workflows. Chapter 5 extends sampling to multiple Erlang nodes. Chapter 6 provides a systematic performance analysis of JIT boundaries in mixed-runtime PPLs, including GPU acceleration. Chapter 7 concludes with contributions, limitations, and future work.

---

## Chapter 1: The PPL Design Space

### 1.1 Probabilistic Programming Languages

Probabilistic programming languages automate Bayesian inference by separating model specification from the inference algorithm (van de Meent et al., 2018). The user declares random variables, their prior distributions, and observed data; the PPL constructs a joint log-probability function and applies a sampling or optimization algorithm — typically Hamiltonian Monte Carlo (Duane et al., 1987; Neal, 2011; Betancourt, 2017) or variational inference (Blei et al., 2017) — to obtain the posterior.

The current PPL landscape is dominated by five systems. **Stan** (Carpenter et al., 2017) pioneered high-performance MCMC via a custom DSL compiled to C++. Its static model graph enables aggressive optimization but prohibits dynamic model construction. **PyMC** (Salvatier et al., 2016) provides a Python-native model-building API with a dynamic computation graph, using PyTensor (formerly Theano) for automatic differentiation and C-compiled gradient evaluation. **NumPyro** (Phan et al., 2019) achieves the highest single-node performance by compiling the entire sampler — including the NUTS tree builder — into a single JAX/XLA program. **Turing.jl** (Ge et al., 2018) takes an approach most analogous to Exmc: models are specified in a general-purpose language (Julia) with JIT compilation, preserving the ability to use arbitrary host-language constructs within models. **Pyro** (Bingham et al., 2019) targets deep probabilistic models on PyTorch, emphasizing variational inference and GPU-first workflows.

These systems share a common runtime model: sampling is a batch computation on a single node that produces a complete trace for post-hoc analysis. None provides crash recovery within a sampling run. None streams intermediate results to external consumers during sampling. None distributes chains across machines without external infrastructure. These omissions are not oversights — they reflect the batch-computation assumption embedded in the runtimes these PPLs are built upon.

| System | Language | Backend | Sampling | Graph | Key Limitation |
|--------|----------|---------|----------|-------|----------------|
| Stan | Stan DSL | C++ | NUTS | Static | No dynamic models, no extensibility |
| PyMC | Python | PyTensor/C | NUTS, VI, SMC | Dynamic | Batch-only, GIL for parallelism |
| NumPyro | Python | JAX | NUTS, VI | Dynamic | Batch-only, requires JAX ecosystem |
| Turing.jl | Julia | Julia | NUTS, HMC, etc | Dynamic | JIT overhead on first call |
| Pyro | Python | PyTorch | SVI, MCMC | Dynamic | GPU-first, heavy framework |
| **Exmc** | **Elixir** | **Nx/EXLA** | **NUTS, ADVI, SMC** | **Dynamic** | **Per-leapfrog overhead (1ms vs 0.1ms)** |

### 1.2 The Actor-Model Runtime

The BEAM virtual machine (Virding et al., 1996; Armstrong, 2003) implements the actor-model concurrency paradigm. Processes are lightweight (2KB initial heap), isolated (no shared memory), and communicate exclusively via asynchronous message passing. The runtime provides three properties relevant to probabilistic programming:

**Fault isolation.** Each process has an independent heap. A crash — whether from arithmetic error, memory exhaustion, or external dependency failure — destroys only the faulting process. Supervisor processes monitor workers and implement configurable restart strategies. The `try/rescue` mechanism registers an exception handler on the process stack with zero runtime cost on the non-faulting path.

**Message-passing concurrency.** Processes have mailboxes that buffer incoming messages. A producer (e.g., a sampler) can send results to a consumer (e.g., a visualization or web handler) without blocking. This decoupling is structural — it arises from the runtime semantics, not from a library abstraction.

**Built-in distribution.** Erlang nodes discover each other via a name service and communicate transparently using Erlang term serialization. The `:erpc` module provides synchronous remote procedure calls with timeout and error handling. Distribution requires no external infrastructure — no MPI, no message broker, no serialization library.

The "let it crash" philosophy (Armstrong, 2003) inverts the traditional approach to error handling. Rather than defensively checking every possible failure mode, programs are structured so that failures are contained and recovered by supervisors. This philosophy applies directly to MCMC, where numerical failures (NaN gradients, overflow, underflow) are common and currently abort entire sampling runs.

### 1.3 Research Questions

This thesis addresses four research questions, each answered by a specific chapter:

1. **Can actor-model fault isolation provide zero-cost crash recovery within MCMC tree building?** (Chapter 3) We demonstrate that BEAM's `try/rescue` wraps NUTS subtree computation with zero overhead on the non-faulting path, enabling recovery from numerical crashes, EXLA failures, and memory faults without restarting the sampling run.

2. **Does message-passing concurrency enable live inference integrated with production web services?** (Chapter 4) We show that Elixir's process mailboxes provide a natural streaming protocol for MCMC samples, demonstrated via real-time browser-based posterior visualization through Phoenix LiveView.

3. **Can Erlang distribution primitives support multi-node MCMC with transparent fault recovery?** (Chapter 5) We implement distributed chain dispatch via `:erpc` with automatic retry on node failure, requiring 203 lines and zero external infrastructure.

4. **How should JIT boundaries be placed in a mixed-runtime PPL, and what is the quantitative cost of misplacement?** (Chapter 6) We develop an affine cost model for JIT boundary operations, identify concrete crossover dimensions, and decompose the performance gap between Exmc and PyMC into architectural and algorithmic components.

### 1.4 The Decision Space

Building Exmc required 52 explicit architectural decisions (D1–D52), each documented with a falsifiable assumption. The decision inventory reveals seven load-bearing assumptions shared across all PPLs:

1. **Numeric backend choice** — determines the performance floor and gradient capability
2. **Autodiff traceability** — all log-probability operations must compose with the chosen AD system
3. **PRNG strategy** — tensor-framework vs native, functional vs stateful
4. **Adaptation arithmetic** — scalar vs tensor, and when to use which
5. **Mass matrix structure** — diagonal vs dense, and the adaptation window strategy
6. **Hierarchical mechanism** — how child distributions reference parent parameters
7. **Transform strategy** — unconstrained space for sampling, constrained space for users

Six of these assumptions were violated during development, triggering decision revisions (D12→D22, D14→D24, D19→D37, D38 partially superseded by D43). The full decision catalog is maintained in `exmc/DECISIONS.md`, where each entry records the assumption, the evidence that violated it, and the revised decision. Two late discoveries (D49–D50) were particularly consequential: multinomial sampling bugs that reduced effective sample diversity by 2–3x were masked by the NUTS algorithm's robustness to implementation variations (Section 6.10.6).

---

## Chapter 2: Exmc Architecture & the BEAM Runtime Model

### 2.1 Architecture Overview

Exmc is organized into four layers, each building on the previous:

```
Layer 1: IR + Distributions    (builder.ex, dist/*.ex)
Layer 2: Compiler + Transforms (compiler.ex, point_map.ex, transform.ex, rewrite.ex)
Layer 3: NUTS Sampler           (leapfrog.ex, tree.ex, mass_matrix.ex, step_size.ex, sampler.ex)
Layer 4: Diagnostics + VI       (diagnostics.ex, model_comparison.ex, advi.ex, smc.ex, pathfinder.ex)
```

**Layer 1** provides the user-facing model specification API. The `Builder` module constructs an intermediate representation (IR) — a plain Elixir map — from a pipeline of random variable declarations and observations. Distribution modules (`Normal`, `HalfNormal`, `Exponential`, `Uniform`, etc.) define log-probability functions and constraint transforms. Tensor operations use the Nx framework (Valim et al., 2023) with optional EXLA/XLA acceleration (Bradbury et al., 2018).

**Layer 2** compiles the IR into the functions needed for sampling. The `Compiler` module produces a joint log-probability function, wraps it in automatic differentiation (`Nx.Defn.value_and_grad`), and constructs a `step_fn` closure that performs a single leapfrog step. The `PointMap` module manages the mapping between constrained (user-facing) and unconstrained (sampler-facing) parameter spaces. The `Rewrite` module applies optional reparameterization passes, including non-centered parameterization (NCP) for hierarchical models.

**Layer 3** implements the NUTS sampler (Hoffman and Gelman, 2014). The tree builder (`Tree`) constructs the NUTS trajectory via recursive doubling. `MassMatrix` implements diagonal and dense mass matrix adaptation with Stan-style windowed Welford estimation. `StepSize` implements dual averaging for step size adaptation (Nesterov, 2009). The `Sampler` module orchestrates warmup (step size search, adaptation windows) and sampling phases.

**Layer 4** provides post-sampling diagnostics (ESS, rank-normalized R-hat per Vehtari et al. 2021, MCMC-SE), model comparison (WAIC, LOO-CV following Gelman et al. 2013), and alternative inference algorithms (ADVI, SMC, Pathfinder).

### 2.2 Model Specification

Models are specified using the `Builder` pipeline API. The following example defines a hierarchical model with a global mean, global scale, group-level effects, and observations:

```elixir
ir = Builder.new_ir()
  |> Builder.rv(:mu_global, :normal, mu: 0.0, sigma: 10.0)
  |> Builder.rv(:sigma_global, :half_normal, sigma: 5.0)
  |> Builder.rv(:alpha, :normal, mu: "mu_global", sigma: "sigma_global", size: 3)
  |> Builder.rv(:sigma_obs, :half_normal, sigma: 2.0)
  |> Builder.obs(:y, :normal, mu: "alpha", sigma: "sigma_obs", observed: y_data)
```

String references (e.g., `mu: "mu_global"`) create hierarchical dependencies: the compiler resolves these to the constrained values of parent random variables at sampling time. The `size: 3` argument creates a vector-valued random variable. The `obs` call marks `:y` as observed, fixing its value to `y_data` and contributing a likelihood term to the joint log-probability.

The IR produced by the Builder is a plain Elixir map containing the list of random variables, their distributions, parameter references, observed data, and constraint metadata. This representation is serializable, inspectable, and — critically for distributed sampling — transportable across Erlang nodes without closure capture issues.

### 2.3 Compilation Pipeline

The `Compiler.compile_for_sampling/2` function transforms the IR into the functions used by the sampler:

1. **Parameter ordering.** Free (unobserved) random variables are sorted into a deterministic order and assigned index ranges within a flat parameter vector.

2. **Constraint transforms.** Each constrained random variable (e.g., `HalfNormal` requiring positivity) is paired with a bijective transform (e.g., `log`/`exp`) and its log-determinant Jacobian. Sampling operates in unconstrained space; the compiler inserts the appropriate forward transforms and Jacobian corrections into the log-probability function.

3. **NCP rewrite (optional).** For hierarchical models, the `Rewrite` module can apply a non-centered parameterization, transforming `alpha ~ Normal(mu, sigma)` into `alpha_raw ~ Normal(0, 1)` with `alpha = mu + sigma * alpha_raw`. This reparameterization can improve sampling geometry for weakly identified models, though it can harm well-identified models where the data strongly constrains the posterior (Section 6.9).

4. **Log-probability function.** The compiler constructs a closure `logp_fn(theta)` that maps a flat unconstrained parameter vector to the joint log-probability: the sum of all prior log-densities, likelihood terms, and Jacobian corrections.

5. **Gradient function.** `Nx.Defn.value_and_grad(logp_fn)` produces a function returning both the log-probability value and its gradient. When EXLA is available, this function is automatically JIT-compiled via `EXLA.jit`, providing hardware-accelerated gradient evaluation.

6. **Step function.** The `step_fn` closure bundles a single leapfrog step: given position `q`, momentum `p`, and step size `epsilon`, it performs a half-step momentum update, a full-step position update, and another half-step momentum update, returning the new state along with the gradient and log-probability. This closure is the JIT boundary — everything inside it runs as compiled XLA; everything outside runs as interpreted Elixir.

### 2.4 The Two-PRNG Insight

Not all randomness in MCMC belongs in the tensor computation graph. The NUTS tree builder makes O(2^depth) scalar random decisions per sample (direction choice, proposal acceptance). Using the tensor framework's PRNG (Nx.Random) for these is six orders of magnitude slower than the native PRNG (Erlang `:rand`).

Benchmarking confirms this: `Nx.Random.uniform` requires 1–2 seconds per call due to `defn` tracing overhead, while `:rand.uniform_s` completes in microseconds. At tree depth 8 with 255 random decisions per sample, the difference is 4+ minutes versus 0.25 milliseconds.

This observation generalizes: in any PPL with a tensor framework PRNG, scalar control-flow decisions should use the host language's native PRNG. The boundary aligns with the JIT boundary — gradient-dependent operations belong inside the tensor graph, while control-flow decisions belong outside it.

### 2.5 Functional State Threading

Elixir's explicit-state `:rand` API (`{value, rng} = :rand.uniform_s(rng)`) provides a structural guarantee that PRNG state is never shared between chains. This makes parallel chains correct by construction — no seed management bugs are possible.

By contrast, PyMC's `numpy.random.Generator` requires careful per-process seed management, and Stan uses thread-local state. Only NumPyro, via JAX's functional PRNG, provides similar guarantees, but these are built into the tensor framework rather than the language itself.

### 2.6 Scalar Adaptation, Tensor Geometry

The NUTS sampler naturally splits into scalar adaptation (dual averaging for step size, Welford statistics for mass matrix) and tensor geometry (leapfrog integration, gradient evaluation). Elixir serves both roles without a two-language boundary: Erlang floats handle adaptation arithmetic, while Nx/EXLA handles tensor geometry.

Adaptation overhead is less than 1% of total sampling time. The "two-language problem" familiar from Python+C systems does not arise because Erlang's native float arithmetic is fast enough for the O(1) scalar operations required per NUTS step.

---

## Chapter 3: Fault-Tolerant Sampling

MCMC sampling runs are long-running numerical computations where failures are not exceptional — they are expected. Gradients diverge when the sampler explores regions of extreme curvature. GPU kernels fail under memory pressure. EXLA compilations crash on pathological input shapes. In every existing PPL, such failures abort the entire sampling run, discarding all progress. The user restarts from scratch, perhaps with different initial values, hoping the failure does not recur.

The actor model offers a different approach. If each subtree computation is isolated in its own fault boundary, a crash destroys only the faulting subtree. The rest of the trajectory — and the entire sampling run — continues. This chapter demonstrates that BEAM's process isolation and `try/rescue` mechanism provide this fault boundary at zero cost on the non-faulting path, and that the existing NUTS tree structure naturally accommodates failed subtrees via divergent placeholders.

### 3.1 Current Fault Handling

Three layers of numerical failure recovery in the NUTS tree builder:

1. **Leaf level:** `is_number(joint_logp)` catches NaN/Inf. Failed leaves marked divergent with zero acceptance probability.
2. **Subtree level:** Early termination when first half diverges or turns.
3. **Trajectory level:** Tree build terminates on divergence or U-turn.

This is "let it crash" philosophy applied within a deterministic algorithm.

### 3.2 BEAM Type System as Safety Net

Erlang's dynamic types surface IEEE 754 special values as atoms (`:nan`, `:infinity`, `:neg_infinity`). These pattern-match differently from numbers, making failure detection a type-level check (`is_number/1`) rather than a value-level check (`isnan()`).

In C, Python, and Julia, NaN propagates silently through computation chains — a single NaN in a gradient can corrupt an entire trajectory before being detected. In Erlang, arithmetic on atoms crashes immediately. Failure is visible, not silent.

### 3.3 Supervised Tree Building

BEAM's `try/rescue` provides zero-cost fault containment for NUTS subtree builds. A two-tier supervision mechanism wraps each subtree dispatch:

**Tier 1 (`supervised: true`):** `try/rescue` around `dispatch_subtree`. When no exception occurs, the code path is identical to the unsupervised path — BEAM's `try` is implemented as exception registration in the process stack, not as a wrapper function. This means zero instructions on the happy path. When a subtree crashes (RuntimeError, ErlangError/OOM, ArithmeticError), it is replaced with a divergent placeholder.

**Tier 2 (`supervised: :task`):** `Task.async` + `Task.yield(timeout)` for hard hang detection (EXLA stuck on GPU, infinite loops). Configurable timeout via process dictionary. Overhead: ~5-20us per spawn per doubling iteration.

**Divergent placeholder:** When a subtree at depth `d` crashes, returns a placeholder structurally identical to existing NaN/divergent leaf handling: copies of starting state (q, p, grad), `divergent: true`, `recovered: true`, `log_sum_weight: -1001.0`, `n_steps: 2^d`, `accept_sum: 0.0`. The existing merge logic (merge_subtrees, merge_trajectories, proposal selection) handles this without modification — the placeholder has extremely low log-weight, so the multinomial sampling will almost never select the crashed subtree's proposal. The tree continues building.

**PRNG recovery:** On crash, the subtree's RNG state is lost. Recovery seeds a fresh `:rand` state from `System.unique_integer([:positive])`. Post-crash trajectories are statistically valid but not deterministically reproducible. This is documented as an explicit trade-off.

Empirical evaluation confirms the mechanism's properties:

| Property | Measurement |
|----------|------------|
| Overhead (supervised, no faults) | < 1% (within noise; try/rescue is zero-cost) |
| No-failure parity | Bit-identical traces (max diff < 1e-10) with same seed |
| Crash recovery | Sampling completes, posterior valid (mean ±1.5, var 0.1–5.0 for N(0,1)) |
| Recoveries tracked | Per-step `recovered` flag + global `recoveries` counter in stats |
| Task timeout | 1ms timeout → divergent placeholder, no hang |

The mechanism is validated by 15 tests covering fault injection, tree-level recovery, no-failure parity, end-to-end recovery with valid posteriors, recovery statistics tracking, task timeout, and overhead benchmarking. A process-dictionary-based FaultInjector utility (66 lines) enables reproducible fault tolerance testing without modifying production code paths.

No existing PPL treats MCMC subtree failures as recoverable events. In PyMC, a hard crash (segfault in the C sampler, CUDA OOM) aborts the entire sampling run; divergences are counted post-hoc as warnings. In Stan, C++ exceptions propagate to the top level, and `try/catch` has measurable overhead (~1–2ns per frame) due to exception table registration. In NumPyro, JAX's JIT compilation means the entire tree is one XLA kernel — no fault boundary is possible within it. In Turing.jl, Julia exceptions propagate normally with no per-subtree recovery mechanism.

BEAM's `try/rescue` is the only mechanism among these runtimes where fault containment is free on the happy path. Combined with the three-layer NaN/divergence handling (Section 3.1), Exmc provides four layers of fault tolerance — from IEEE 754 special value detection up through process-level crash recovery.

### 3.4 Integration with Distributed Fault Recovery

The supervised tree building (Section 3.3) and distributed fault recovery (Chapter 5) form a two-level fault tolerance hierarchy:

1. **Intra-node (Section 3.3):** `try/rescue` around subtree builds. Recovers from numerical crashes, EXLA failures, and memory pressure. The sampler continues on the same node.
2. **Inter-node (Chapter 5):** `try/catch` around `:erpc.call` dispatches. Recovers from node death and network partitions. The chain retries on the coordinator.

Both levels share the same design principle: replace the failed computation with a valid (though degraded) result. At the subtree level, the replacement is a divergent placeholder. At the chain level, the replacement is a re-run on a different node. Both are correct because:
- Subtrees are independent within a tree (no shared state to corrupt)
- Chains are independent within a multi-chain run (functional PRNG, immutable tensors)

This two-level hierarchy is unique to BEAM-based PPLs and arises naturally from the actor model's fault isolation semantics.

---

## Chapter 4: Live/Streaming Inference

Interactive data science benefits from real-time feedback during model fitting. A practitioner running a hierarchical model wants to see whether chains are mixing, whether the posterior is converging, and whether the model is behaving as expected — not wait minutes for a batch run to complete before discovering a problem. Yet every existing PPL enforces a "run and wait" workflow: `pm.sample()` returns a complete trace or nothing.

The actor model provides a natural alternative. If the sampler is a process and each sample is a message, then any other process — a visualization, a web handler, a monitoring system — can receive posterior updates as they are produced. No polling, no shared memory, no callback that blocks the sampler. This chapter demonstrates that streaming inference emerges naturally from BEAM's message-passing semantics and integrates directly with production web frameworks.

### 4.1 sample_stream — Inference as Message Passing

`Sampler.sample_stream(ir, receiver_pid, init_values, opts)` sends `{:exmc_sample, step, point, step_stat}` after each NUTS step. The receiver can be any BEAM process.

By contrast, PyMC's callback blocks sampling. Stan has no streaming API. NumPyro returns the full trace after completion. Only Turing.jl's `AbstractMCMC.step` is comparable, but without process integration.

### 4.2 ExmcViz — Live MCMC Diagnostics

Real-time trace plots, histograms, ACF, rank plots, PPC overlays via Scenic. All Nx computation in Data.Prepare — UI components see plain Elixir data. The UI never blocks the sampler.

ExmcViz is validated by 64 tests with live rendering verified during sampling.

### 4.3 Phoenix LiveView — Live Bayesian Inference in the Browser

The claim that actor-model runtimes enable live inference workflows is validated by `exmc_live/`, a Phoenix LiveView application where the LiveView process itself serves as the `receiver_pid` for `sample_stream`. There is no intermediate GenServer, no polling, and no batch completion — the browser sees posterior updates as they happen.

**Architecture:** User selects a model (3 presets: conjugate 1RV, two-param 2RV, hierarchical 5RV), enters observations, clicks "Sample." The LiveView spawns a Task calling `sample_stream(ir, self(), init, opts)`. Each `{:exmc_sample, i, point_map, step_stat}` message is buffered (5 samples), then pushed to the browser via `push_event("new_samples", payload)`. Chart.js hooks render trace plots and histograms with zero-animation incremental updates (`chart.update("none")`). Running diagnostics (mean, std, ESS) update with each batch. Latency is tracked at 4 points: warmup time, time to first browser update, per-sample latency, total ESS/s.

**Key insight:** The LiveView process naturally handles the async sample stream because BEAM processes have mailboxes. This is the simplest possible integration — simpler than ExmcViz (which needs a Coordinator GenServer because Scenic scenes have a different message model). The architecture demonstrates that `sample_stream`'s message-passing protocol was correctly designed for production integration, not just visualization.

The implementation consists of a single LiveView module, a Models module, and two JavaScript hooks. It supports three preset models (1, 2, and 5 free random variables), and conjugate model posteriors match analytic solutions within tolerance. All 12 tests pass.

No existing PPL provides comparable web integration. PyMC's `pm.sample(callback=fn)` blocks sampling during the callback, with post-hoc visualization only via ArviZ. Stan has no streaming API; ShinyStan requires a separate R process and updates only between chains. NumPyro returns the full trace after completion. Turing.jl's `AbstractMCMC.step` returns individual samples but has no web framework integration.

The Exmc approach requires zero special infrastructure: `Task.start` + `send(receiver_pid, msg)` + `push_event`. The web framework and the sampler communicate via the same message-passing primitive that drives all BEAM concurrency. Live Bayesian inference — where a user submits data and watches the posterior converge in real time — is a natural consequence of building a PPL on an actor-model runtime. It emerges from the runtime's message-passing semantics rather than requiring separate design.

### 4.4 Distributed Live Streaming — Emergent Feature Composition

The strongest evidence for the actor-model thesis is not any single feature but the *composition* of independently-designed features without integration effort. Chapters 4 and 5 present streaming inference and distributed sampling as separate contributions. This section demonstrates that their composition is trivial — and that this triviality is a direct consequence of building both features on the same message-passing primitive.

`ExmcViz.stream_external/2` opens a LiveDashboard visualization in external mode: no auto-sampler, just a coordinator PID published via `:persistent_term`. Any process — local or remote — can send `{:exmc_sample, i, point_map, step_stat}` to this PID. The implementation adds 3 lines to LiveDashboard and ~50 lines for the new API. All existing visualization components (trace plots, histograms, ACF, summary panels) required zero changes.

The distributed live streaming demo (`demo/demo_distributed.exs`) composes these features: 4 `:peer` nodes each independently compile, warm up (3000 iterations), and sample (5000 each) the stress model (d=8, 8 free parameters), streaming samples to a single ExmcViz window via Erlang's transparent PID routing. A 10-line forwarder process solves the "N producers, 1 consumer, completion signal" coordination problem by swallowing per-chain `{:exmc_done, _}` messages and emitting a single completion signal after all chains finish.

Result: 20,000 samples in 21.1 seconds (948 samples/s), with 8 trace plots, histograms, and ACF updating in real time from all 4 nodes simultaneously. The visualization shows samples arriving interleaved from different chains — chains finish at different times (14.5s to 21.1s) due to independent warmup adaptation finding different step sizes.

The key architectural observation: the streaming protocol (`{:exmc_sample, ...}`) was designed for single-chain ExmcViz months before the distributed system existed. The distributed protocol (`:erpc.call` with `sample_stream`) was designed for batch result collection. Their composition worked on the first attempt because Erlang PIDs are location-transparent — `send(pid, msg)` has identical semantics whether `pid` is on the local node or a remote node. No new protocol, serialization layer, or message broker was needed. This is the actor-model thesis in action: features built on the same concurrency primitive compose without integration effort.

No existing PPL can replicate this. PyMC has no streaming API to compose with distribution. Stan's CmdStan communicates via file I/O. NumPyro returns complete traces. The closest parallel would be combining Turing.jl's `AbstractMCMC.step` with Julia's `Distributed` module, but no such integration exists — it would require explicit serialization of step results across Julia workers and a custom aggregation layer.

---

## Chapter 5: Distributed MCMC

Modern Bayesian workflows increasingly demand multiple chains — for convergence diagnostics (R-hat), for embarrassingly parallel speedup, and for exploring multi-modal posteriors. Scaling beyond a single machine is desirable when models are expensive or when hardware resources are distributed across a cluster. The distributed MCMC literature offers principled approaches such as consensus Monte Carlo (Neiswanger et al., 2014; Scott et al., 2016) and unbiased coupling methods (Jacob et al., 2020), but existing PPL implementations require substantial infrastructure: PyMC relies on `multiprocessing` with fragile `cloudpickle` serialization; Stan requires MPI for `map_rect`; NumPyro's `pmap` is limited to a single host. None provides fault recovery when a remote worker fails.

Erlang was designed for distributed telecommunications systems where nodes join and leave dynamically and failures are routine. The same primitives — node discovery, remote procedure calls via `:erpc`, and transparent term serialization — apply directly to distributing MCMC chains. This chapter demonstrates that the jump from local to distributed sampling is architecturally straightforward on the BEAM, requiring 203 lines of Elixir and zero external infrastructure.

### 5.1 Parallel Chains via Task.async_stream

Compile once, dispatch N chains via `Task.async_stream`. Each chain is provably independent: functional PRNG state, immutable tensors, thread-safe JIT closures.

Integration tests confirm that parallel chains are faster than sequential and produce deterministic results given a seed.

### 5.2 Warmup-Once Broadcast

`sample_chains_vectorized`: warmup chain 0 only, broadcast adapted parameters (step size, mass matrix) to all chains. A natural "parameter server" pattern that's trivial on BEAM (pass as function arguments) but requires serialization infrastructure in Python/C++.

### 5.3 Multi-Node MCMC via `:erpc`

The jump from local to distributed MCMC is architecturally straightforward — but with a critical insight: the distribution primitive is the model IR, not the compiled function. JIT closures capture EXLA device handles that cannot cross node boundaries. Model IR is plain Elixir data (~1KB) that serializes trivially via Erlang term distribution.

**Architecture:** `Exmc.NUTS.Distributed.sample_chains/2` implements a three-phase protocol:
1. **Validate:** Check remote nodes have Exmc loaded via `:erpc.call(node, Code, :ensure_loaded, [Exmc.Compiler])`
2. **Warmup:** Compile and warmup on coordinator, extract tuning params `{epsilon, inv_mass, chol_cov}`
3. **Dispatch:** Send IR + tuning params to each node via `:erpc.call`. Each node compiles independently (supports heterogeneous hardware) and runs chains with pre-computed tuning (no warmup).

**Fault recovery:** `:erpc.call` wrapped in `try/catch`. On node death, the chain retries on the coordinator transparently. This works because chains are functionally pure — deterministic given `{seed, tuning, ir}`. No shared state to corrupt, no warmup to redo. `validate_nodes!` is tolerant of unreachable nodes — logs a warning and defers to dispatch-level fault recovery.

All 5 tests pass (3 coordinator-only + 2 multi-node via OTP `:peer` module). Wall-clock benchmarks on all three benchmark models (1000 warmup + 1000 samples, CPU-only, centered parameterization):

| Configuration | Simple ESS/s | Medium ESS/s | Stress ESS/s |
|---|---|---|---|
| Local 4ch (sequential) | 490 | 233 | 169 |
| Distributed 5 nodes × 1ch | **1665** | **812** | **628** |
| Speedup | **3.4x** | **3.5x** | **3.7x** |

Each `:peer` node is a separate OS process with its own BEAM scheduler pool, providing true multi-core parallelism. The coordinator node participates as chain 0, running concurrently with the 4 peer chains (5 total chains). Near-linear scaling (3.4–3.7x from 5 chains) indicates minimal coordination overhead. Wall times: simple 2555ms, medium 4138ms, stress 5275ms.

**Compile options bug.** A critical implementation bug was discovered during this benchmarking: `Distributed.sample_chains` called `Compiler.compile_for_sampling(ir)` at three points (coordinator warmup, local chain, remote chain) without passing compilation options such as `ncp: false`. This meant distributed sampling silently used NCP for all models — incorrect for medium/stress where centered parameterization gives 9x better ESS/s (Section 6.9). This bug class is specific to distributed PPLs where compilation happens at multiple sites: the model IR is necessary but not sufficient; the compilation *context* (parameterization, device target) must also be distributed. Fix: thread `[:ncp, :device]` options through all compilation points.

Fault recovery: killing a peer node mid-run adds negligible overhead (3135ms vs 3254ms baseline = -3.7%). The dead node's chain retries on the coordinator immediately — `:erpc` detects the disconnected node in <1s. Distribution overhead per node = EXLA compilation (~200ms) + EXLA init.

The system requires no external infrastructure — no MPI, Dask, Ray, or Spark. The same code paths that work locally work distributed; only the target node name changes. The entire distributed system is 203 lines of Elixir.

By contrast, PyMC uses `cloudpickle` + `multiprocessing.Pool` with fragile serialization and no fault recovery. Stan requires MPI for `map_rect`. NumPyro has no multi-node capability (JAX's `pmap` is same-node only). The BEAM approach is unique in providing distribution and fault recovery as language-level primitives rather than library dependencies.

### 5.4 Seven-Model Distributed Benchmark

The initial 3-model distributed benchmark (Section 5.3) tested only standard distributions. To validate distribution scaling across the full model spectrum — including Custom distributions with closure-captured data — we extended the benchmark to all 7 models, 5 seeds, with comparison against PyMC's 4-chain `multiprocessing`.

| Model | d | Exmc 1ch | Exmc 5-node | Scaling | PyMC 4ch | Dist vs PyMC |
|-------|---|---------|------------|---------|---------|-------------|
| Simple | 2 | 430 | 1,687 | 3.92x | 1,999 | 0.84x |
| Medium | 5 | 271 | **841** | 3.10x | 680 | **1.24x** |
| Stress | 8 | 222 | 604 | 2.72x | 678 | 0.89x |
| Eight Schools | 10 | 7.7 | 13.3 | 1.73x | 20.3 | 0.66x |
| Funnel | 10 | 1.6 | **5.4** | 3.38x | 4.1 | **1.32x** |
| Logistic | 21 | 63 | 175† | 2.76x | 1,514 | 0.12x |
| SV | 102 | 0.6 | ~1.5† | ~2.5x | 2.2 | ~0.68x |

*† Closure barrier — all chains fell back to coordinator (local concurrency only)*

**The closure barrier.** Models using Custom distributions with string-reference-only parameters (funnel: `%{y_val: "y"}`) distribute transparently — string refs are plain Elixir data. Models whose Custom logpdf closures capture `Nx.Defn.Expr` tensor graph fragments (logistic: `x_matrix` 500x20; SV: `returns_vec` 100-element) cannot serialize across `:erpc`. Every peer node fails with `:undef` (the anonymous function is not defined on the remote node). The chains fall back to the coordinator via `try/catch`, running concurrently via `Task.async` and achieving 2.5-2.8x speedup from CPU time-sharing and EXLA JIT overlap.

This is a design space boundary, not a bug: **distribution-safe PPL models must separate data from computation** — observed data belongs in the IR (as tensor fields or string refs), not captured in closures. The fix is known: `Builder.obs_data("x_matrix", x_matrix)` would store the tensor as a named IR field, making logistic and SV distribute transparently.

**Distribution amplifies adaptation advantage.** On medium (d=5), Exmc's 1ch advantage (1.65x) is amplified by distribution: 5-node Exmc (841) beats PyMC 4ch (680) by 1.24x. On funnel (d=10), distribution *inverts* the 1ch result: Exmc loses 1ch (0.27x) but wins distributed (1.32x) because 3.38x scaling from 5 `:peer` nodes exceeds PyMC's ~2x from 4 OS processes.

**Average scaling: 2.87x from 5 nodes** across all 7 models. Even closure-barrier models achieve 2.5-2.8x from fault-recovery concurrency — the recovery path IS local parallelism.

---

## Chapter 6: Performance — JIT Boundaries in Interpreted-Host PPLs

### 6.1 The JIT Boundary Problem

In a PPL with JIT-compiled gradient computation and an interpreted host-language sampler, tensors that cross the JIT boundary carry their backend. This is analogous to the interpreter/compiler boundary problem in language runtimes (Würthinger et al., 2013), but with a PPL-specific twist: the cost is paid per leapfrog step rather than per function call. EXLA.jit outputs on EXLA.Backend flow into the Elixir tree builder, where every Nx operation dispatches through the EXLA runtime.

### 6.2 Quantified Overhead

| Operation | EXLA.Backend | BinaryBackend | Ratio |
|-----------|-------------|---------------|-------|
| Nx.to_number(Nx.sum) | 150us | 5us | 30x |
| Nx.add | 69us | 7us | 9x |
| Nx.multiply | 74us | 9us | 8x |

For tree depth 8 (~255 leaves): ~305ms overhead (EXLA) vs ~22ms (BinaryBackend) = 14x.

### 6.3 The Fix and Its Impact

`Nx.backend_copy(tensor, Nx.BinaryBackend)` on step_fn outputs at the tree leaf level.

| Model | Before | After | Speedup |
|-------|--------|-------|---------|
| Simple | 70 ESS/s | 211 ESS/s | 3.0x |
| Medium | 2.1 ESS/s | 3.3 ESS/s | 1.6x |

### 6.4 The Two-Category Tensor Model

PPL tensors fall into two categories:
1. **Model metadata** (parameters, observations) — constant across steps, should be on JIT-compatible backend
2. **Sampling state** (position, momentum, gradient) — changes every step, should be on host-language-optimal backend

Both categories should be on BinaryBackend in Exmc, but for different reasons: metadata for EXLA tracing compatibility, state for Elixir-side tree builder performance.

### 6.5 Phase 1: Micro-Optimizations at the JIT Boundary

Four targeted optimizations that reduce per-step overhead without changing the architecture:

1. **Fuse joint_logp into step_fn JIT (P1a):** Move the joint_logp computation (logp - KE) inside the JIT'd step function, eliminating a separate Nx.to_number call per leaf.
2. **Pure-Erlang U-turn check (P1b):** Replace the Nx-based U-turn check with a recursive list-arithmetic implementation on flat Erlang lists. 26.5us -> 2.0us per check (13x). For tree depth 8 with ~254 checks: saves ~6.2ms/sample. Subsequently replaced with the generalized (rho-based) criterion (Section 6.10.3), retaining the Erlang list-arithmetic approach.
3. **Explicit BinaryBackend hints (P1c):** Ensure all `Nx.tensor()` calls in sampler/leapfrog/mass_matrix specify `backend: Nx.BinaryBackend` to avoid accidental EXLA dispatch.
4. **Cache inv_mass flat list (P1d):** Pre-convert `inv_mass` tensor to Erlang list once, reuse for all momentum sampling and U-turn checks.

Benchmarks on both models:

| Model | Before | After Phase 1 | Improvement |
|-------|--------|--------------|-------------|
| Simple | 211 ESS/s | 242 ESS/s | +15% |
| Medium | 3.3 ESS/s | 4.9 ESS/s | +48% |

The medium model benefits more (+48% vs +15%) because deeper trees (average depth 4–5 vs 1–2) amplify per-step savings, confirming the per-leapfrog overhead decomposition from Section 6.2.

### 6.6 Phase 2: Batched Leapfrog via XLA While-Loop

The most significant optimization: instead of one JIT call per leapfrog step, batch an entire subtree's leapfrog trajectory into a single XLA `while` loop.

**Architecture:** `Nx.Defn.while` inside `EXLA.jit` compiles a loop running N leapfrog steps (including `value_and_grad` at each iteration), storing all intermediate states in pre-allocated tensors. The tree builder calls this once per subtree, then slices the results into leaf nodes and builds a merge tree.

Benchmarks confirm the dispatch savings:

| N steps | Individual dispatch | Batched dispatch | Speedup |
|---------|-------------------|-----------------|---------|
| 4 | 1040us | 400us | 2.6x |
| 16 | 4160us | 468us | 8.9x |
| 64 | 16640us | 408us | 40.8x |
| 128 | 33280us | 498us | 66.9x |

Speedup grows superlinearly: per-call dispatch overhead (~250us) dominates for individual calls; the batched version amortizes it.

**Integration challenge:** The merge tree must preserve the recursive `build_subtree` semantics exactly — including RNG consumption order, early termination, and log-sum-weight accumulation. Initial implementation had surface-level bugs (leaf reversal, missing early termination) that were straightforward to fix. But a deeper semantic divergence persisted: a bottom-up merge tree that constructs the tree from pre-computed leaves is NOT equivalent to the top-down recursive `build_subtree`, even with early termination mirrored. Subtle differences in how `log_sum_weight` accumulates through the tree structure cause different multinomial sampling outcomes at borderline cases. Per-call verification showed 0/50 mismatches, but chained over 1000 sampling iterations, ESS(mu_global) dropped from 78 to 12 — a 6.5x quality regression.

**The fix — `:atomics` cached step_fn:** Abandon the separate merge tree entirely. Pre-compute all 2^depth leapfrog steps via the batched XLA while-loop, store results in closure-captured tensors, then run the **exact same** recursive `build_subtree` with a cached step_fn that reads pre-computed states via an Erlang `:atomics` counter. This guarantees bit-identical tree construction by reusing the exact same code path — same RNG consumption, same early termination, same merge structure. Verified: 0/20 mismatches across chained sampling iterations. The depth >= 4 threshold (16+ steps per subtree) avoids overhead for shallow subtrees where tensor allocation + backend_copy costs exceed dispatch savings.

**Trade-off:** Loss of early termination *within* the batched subtree. At depth 8, worst case wastes 128 extra steps. At ~4us/step inside JIT: ~0.5ms wasted vs ~32ms dispatch overhead saved. Net positive for all current models.

This experience yields a general principle: when parallelizing a sequential recursive algorithm, the original recursive structure must be preserved exactly. Approximating it with a "semantically equivalent" bottom-up construction is insufficient. Batching the computation while reusing the original control flow (via cached inputs) is safer than reimplementing the control flow.

NumPyro/JAX avoids this issue entirely by JIT-compiling the entire tree builder. Stan compiles to C++. Exmc's hybrid approach keeps the tree builder in Elixir (preserving fault handling, streaming, and dynamic control flow) while batching the expensive computation into JIT. The `:atomics` cached step_fn pattern is the mechanism that makes this hybrid approach safe.

### 6.7 Three-Category Operation Model

The optimization work reveals that PPL sampler operations fall into three categories, not two:

| Category | Examples | Optimal execution | Evidence |
|----------|---------|-------------------|----------|
| **Tensor geometry** | Leapfrog step, gradient | JIT (EXLA while-loop) | 66.9x speedup for N=128 batched steps |
| **Scalar tree logic** | U-turn check, divergence, acceptance | Host-language native arithmetic | 13x speedup for Erlang vs Nx |
| **Control flow** | Direction choice, proposal selection, early termination | Host-language PRNG + branching | 10^6x speedup for :rand vs Nx.Random |

The JIT boundary should be drawn between Category 1 and Categories 2+3. Category 1 benefits from vectorization and fused kernels. Categories 2+3 benefit from zero-overhead scalar arithmetic and dynamic branching.

### 6.8 Phase 3: Closing the ESS/Sample Quality Gap

The remaining algorithmic gap between Exmc and PyMC stems from warmup adaptation quality. Exmc's cumulative Welford (accumulating across all warmup windows) contaminates later mass matrix estimates with early samples drawn under wrong geometry. Stan resets Welford per window and re-searches step size after each mass matrix update.

**Changes:** (1) Reset Welford state at each window boundary — each window starts fresh. (2) After mass matrix finalization, call `find_reasonable_epsilon` with the new mass matrix to find a step size matched to the new geometry. (3) Cache `inv_mass` as flat Erlang list per window to avoid repeated tensor-to-list conversion.

The impact on both models:

| Model | Phase 2 | Phase 3 | Improvement |
|-------|---------|---------|-------------|
| Simple | ~192 ESS/s | ~217 ESS/s | +13% |
| Medium | ~4.6 ESS/s | ~6.0 ESS/s | +30% |

The medium model's adapted step size increased from 0.036 to 0.045, confirming better mass matrix estimation. Divergences increased slightly (9→13) — the larger step size occasionally overshoots, but net ESS/s is significantly better.

**Design tension:** Per-window Welford uses fewer samples per estimate (smallest window = 25 samples). Stan-style regularization (alpha=5/(n+5), shrink toward 1e-3*I) provides stability. With n=25: 16.7% shrinkage toward a near-zero diagonal — adequate for d=5 but potentially insufficient for d>>n. Step size re-search adds ~25ms per window boundary (O(100) probing leapfrog steps at ~250us each) — negligible in an interpreted-host PPL but would matter in a compiled PPL.

Stan and PyMC both use per-window reset, but the ESS impact is rarely quantified in isolation. This experiment isolates the adaptation-schedule effect: +30% ESS/s from a pure algorithmic change with zero additional per-sample overhead.

### 6.8.1 Micro-Optimization Stacking: A Negative Result

After Phase 3, further micro-optimizations — batching leapfrog during warmup, threading pre-computed inverse mass lists through the tree, and pre-extracting flat lists at leaf creation — were predicted to yield a combined 1.45x improvement. The actual result was 1.08x (medium model: 6.0 → 6.3 ESS/s).

The primary cause was an allocation/precomputation tradeoff specific to immutable data structures. Pre-extracting flat lists at each leaf eliminated `Nx.to_flat_list` calls during merges (~620us savings per tree), but carrying four extra list fields per node (19 fields vs 15) increased Elixir map allocation at each merge by a comparable amount. In a functional language where every merge creates a new map, precomputing values to avoid recomputation increases per-node allocation proportionally. This tradeoff does not arise in compiled PPLs with mutable state.

The lesson is that after eliminating the top three overhead sources (JIT boundary, U-turn computation, mass matrix conversion), further micro-optimizations within BEAM semantics face diminishing returns.

### 6.9 Performance Summary and Gap Analysis

This section consolidates the performance trajectory, parameterization discovery, and gap decomposition that emerged from the optimization work in Sections 6.5–6.8.

**Performance trajectory.** The medium hierarchical model (5 free parameters) serves as the primary benchmark, with PyMC at 113 ESS/s as baseline:

| Configuration | ESS/s | Gap vs PyMC |
|---------------|-------|-------------|
| Phase 0 baseline (backend_copy fix) | 3.3 | 34x |
| + Phase 1 micro-optimizations | 4.9 | 23x |
| + Phase 3 adaptation (per-window Welford) | 6.0 | 19x |
| + Centered parameterization (ncp: false) | 49.1 | 2.3x |
| + Rust NIF inner-subtree builder | ~74 | **1.5x** |

The single largest improvement was parameterization choice (centered vs NCP), which provided a 9x ESS/s improvement. This discovery reframed the entire gap analysis.

**The NCP vs centered discovery.** Exmc's automatic non-centered parameterization (NCP) actively hurts the medium hierarchical model. NCP transforms `alpha = mu_global + sigma_global * alpha_raw` where `alpha_raw ~ N(0,1)`. This makes `alpha_raw` perfectly independent (ESS near 1000) but creates complex coupling between `sigma_global` and `sigma_obs` that a diagonal mass matrix cannot capture. The effective step size for `log(sigma_obs)` becomes 0.0029 versus 0.084 for `mu_global` — a 29x per-dimension mismatch that forces the sampler into tiny steps:

| Parameterization | Median ESS/s | Step size | sigma_obs ESS | Divergences |
|-----------------|-------------|-----------|--------------|-------------|
| NCP (automatic) | 5.5 | 0.034 | 4–158 | 9–18 |
| Centered (ncp: false) | 49.1 | 0.46–0.62 | 141–893 | 0–4 |
| PyMC (centered) | 113 | ~0.1 | 885 | ~2 |

With centered parameterization, all parameters have comparable step size requirements. The `ncp: false` option is threaded through all sampling APIs. The data informativeness ratio (observations relative to prior variance) likely determines which parameterization is better — high ratio favors centered, low ratio favors NCP. This is a well-known tension (Papaspiliopoulos et al., 2007), but automatically resolving it remains an open problem in all PPLs.

**Gap decomposition.** The performance gap separates cleanly into architectural and algorithmic components. With NCP, the 12.5x total gap decomposes as 1.84x wall-time multiplied by 6.8x ESS quality. With centered parameterization, the 2.3x gap is entirely architectural — Exmc achieves ESS-per-sample parity with PyMC. The per-leapfrog-step overhead breaks down as follows:

| Component | Cost (us) | Fraction | Reducible? |
|-----------|-----------|----------|------------|
| step_fn JIT dispatch | ~160 | 53% | Only via NIF or full-JIT tree |
| backend_copy (4 tensors) | ~90 | 30% | Amortized by batched leapfrog |
| U-turn check | ~10 | 3% | Pre-extract at leaf |
| Map creation (15 fields) | ~5 | 2% | Tuple refactor |
| Other (log_sum_exp, rand, bookkeeping) | ~35 | 12% | Minimal |
| **Total per step** | **~300** | | PyMC equivalent: ~10–15us |

The JIT dispatch overhead (~160us) is the irreducible floor for the interpreted-host approach. With centered parameterization and the Rust NIF inner-subtree builder, the remaining 1.5x gap is dominated by this dispatch overhead plus bulk tensor conversion (~130us per subtree). Closing this further would require either moving the gradient computation into Rust (eliminating the XLA boundary) or full JIT compilation of the tree builder (as NumPyro does).

**Optimization roadmap summary.**

| Strategy | Predicted | Actual | Notes |
|----------|-----------|--------|-------|
| P0: Batched warmup + inv_mass threading | ~1.26x | ~1.08x | Diminishing returns from allocation tradeoff |
| P1: Pre-extract flat lists at leaf | ~1.15x | ~1.0x | Neutral: precompute savings offset by map allocation |
| P2: Centered parameterization | ~9x | **9x** | Closed ESS quality gap entirely |
| P2: Warmup fixes (DA init) | ~1.3x | ~1.41x | NCP only |
| P3: NIF outer loop | ~2–3x | 0.86x | Failed: double FFI crossing overhead |
| P3: NIF inner subtree | ~1.5x | **1.5x** | Succeeded: single FFI boundary |

This decomposition methodology — separating architectural overhead from algorithmic quality — appears absent from PPL benchmarking literature. The key finding is that what appeared to be a 2.6x "adaptation quality" gap was actually a parameterization choice problem. With the correct parameterization, the entire remaining gap is architectural. Together with the FFI granularity principle (Section 6.10), these changes reduced the PyMC gap from 34x to 1.5x through purely algorithmic and architectural decisions, with no modification to the core NUTS algorithm.

### 6.10 Phase P3: Rust NIF Tree Builder — FFI Boundary Granularity

The final major optimization replaces the Elixir tree builder's inner subtree logic with a Rust NIF. Two strategies were attempted, revealing that FFI boundary *granularity* matters more than FFI language choice.

**Strategy A — Outer-loop replacement (FAILED):**
Replace the entire NUTS tree outer loop with Rust. Each doubling iteration: get_endpoint_bin → Nx.from_binary → multi_step_fn (XLA JIT) → backend_copy → Nx.to_binary → NIF build_and_merge.

| Model | NIF | Elixir | Result |
|-------|-----|--------|--------|
| Simple | 521ms | 450ms | **0.86x (SLOWER)** |
| Medium | 939ms | 1185ms | 1.26x |

**Why it failed:** Each iteration crosses *both* the Elixir↔XLA boundary (for multi_step_fn) and the Elixir↔Rust boundary (for build_and_merge). The double-crossing adds ~50-100us per iteration that exceeds tree logic savings (~36us per iteration for 4 merges × 9us/merge).

**Strategy B — Inner-subtree replacement (SUCCESS):**
Replace only `build_subtree_cached`'s inner logic: instead of per-leaf `Nx.slice` + recursive Elixir merges, send all pre-computed leapfrog states as raw binaries to a single Rust NIF call that builds the entire subtree.

| Model | NIF | Elixir | Result |
|-------|-----|--------|--------|
| Simple | 325ms | 328ms | **~1.0x (break-even)** |
| Medium | 800ms | 1200ms | **1.5x** |

**Why it works:** The inner subtree operates entirely within the host (no JIT dispatch needed). Replacing 16 × Nx.slice + 15 × Elixir-map-merge with a single NIF call on flat `Vec<f64>` saves ~475us per depth-4 subtree. The NIF call overhead (~50us) is small relative to the savings.

**Depth threshold:** Optimal at depth >= 2 (4 leaves). Depth 1 (2 leaves) is break-even — NIF call overhead ~= Elixir savings.

**Architecture:**
```
Elixir outer loop (unchanged)
  ├─ depth 0-1: individual step_fn (Elixir path)
  └─ depth >= 2: multi_step_fn (XLA batch) → Nx.to_binary → NIF build_subtree_bin
      └─ Rust: recursive tree build on Vec<f64>, returns full subtree as binary map
          └─ Elixir: Nx.from_binary → merge into trajectory (Elixir merge_trajectories)
```

**Design decisions:**
1. *Binary API:* `Nx.to_binary()` preserves IEEE 754 NaN/Inf bit patterns. No sanitization needed. O(bytes) via memcpy, not O(elements) via Erlang term encoding.
2. *Separate PRNG:* Rust Xoshiro256** seeded from Elixir `:rand`. Different sequence but MCMC-valid. Eliminates NIF↔Elixir callback for random decisions.
3. *ResourceArc not needed:* The inner-subtree approach returns the full result in one call. No persistent Rust state.
4. *DirtyCpu scheduling:* Subtree builds take 0.1-5ms. `#[rustler::nif(schedule = "DirtyCpu")]` avoids blocking BEAM schedulers.

A 5-rep benchmark on the medium model (centered, 500 warmup + 500 samples) shows stable results: the NIF path takes 763–806ms, the Elixir path takes 1163–1324ms, yielding a consistent 1.5x wall-time speedup.

These results reveal a granularity principle for FFI boundary placement in multi-language PPLs: *replace the largest scope that does not require crossing back to another language*. For NUTS:
- The outer loop crosses Elixir↔XLA (for multi_step_fn) → must stay in Elixir
- The inner subtree is pure host computation → can move entirely to Rust
- Moving the outer loop to Rust creates *double-crossing* (Elixir↔Rust↔XLA) → net slower

This is a PPL-specific instance of a general principle: FFI boundaries are not free, and adding a second FFI boundary to reduce work in one language can be net negative if it adds crossing overhead that exceeds the per-operation savings.

Stan compiles everything to C++ (no FFI boundary). PyMC and NumPyro keep everything in the tensor framework (one boundary). Turing.jl benefits from Julia's low FFI overhead. Exmc's three-language approach (Elixir/Rust/XLA) is unique and reveals that three-way boundary placement is a first-class design concern.

### 6.10.1 Speculative Pre-Computation — Exploiting Trajectory Contiguity

The batched leapfrog (Section 6.6) amortizes JIT dispatch *within* a subtree. But the tree builder still makes one `multi_step_fn` call per doubling level — typically 5–8 calls per tree build, each costing ~250us of dispatch overhead. For a depth-5 tree taking ~5ms total, this is ~1.2ms of pure dispatch overhead (25% of wall time).

A structural property of the NUTS algorithm enables further amortization: all "go right" subtrees form one contiguous forward leapfrog chain from q0, and all "go left" subtrees form one contiguous backward chain from q0. This follows from the Markov property: the first forward doubling needs steps [1..2^d1], the next forward doubling needs steps [2^d1+1..2^d1+2^d2], and so on — these are contiguous segments of a single trajectory.

**Speculative pre-computation** exploits this contiguity. On the first forward subtree need, a single `multi_step_fn` call pre-computes a large batch of forward leapfrog steps from q0 (default: `max(32, n_needed * 2)`). Subsequent forward subtrees at deeper doubling levels slice into this buffer without additional JIT calls. If the tree extends deeper than anticipated, the buffer is lazily extended from the last computed state. The same applies to backward subtrees with `-epsilon`.

The implementation adds three private functions to the tree builder:
- `ensure_available/3`: Lazily initializes or extends the direction buffer via a single `multi_step_fn` call. Backend-copies results to BinaryBackend immediately.
- `slice_precomputed/3`: Cursor-based slicing of n_steps rows from the buffer. O(1) on BinaryBackend (view operation).
- `dispatch_subtree_precomputed/14`: Routes pre-sliced tensors to either the NIF path (`build_subtree_nif_precomputed`) or the cached `:atomics` path (`build_subtree_cached_precomputed`).

JIT dispatch reduction:

| Tree depth | Before (calls) | After (calls) | Dispatch saved |
|-----------|---------------|--------------|----------------|
| 3 | 5 | 2 | 750us |
| 5 | 7 | 2 | 1250us |
| 8 | 9 | 3–4 | 1500us |

Correctness is verified by three properties: (1) chain continuity — partial and full `multi_step_fn` calls produce identical prefix states, confirming the contiguity assumption; (2) speculative vs non-speculative parity — same RNG seed yields same tree structure (n_steps, depth, divergent); (3) sampling quality — posterior within tolerance.

This optimization is complementary to the batched leapfrog (Section 6.6). Together, they reduce JIT dispatch from O(2^max_depth) individual `step_fn` calls (the original architecture) to O(1–2) bulk `multi_step_fn` calls per tree build — a two-order-of-magnitude reduction in dispatch overhead for deep trees.

The pattern is specific to mixed-runtime PPLs where the tree builder is interpreted but gradient evaluation is JIT-compiled. NumPyro avoids it entirely (one XLA program). Stan avoids it (C++ tree builder). The insight generalizes: any tree-based MCMC sampler (including multinomial HMC, XHMC) with the same iterative-doubling structure has the same contiguity property and could benefit from speculative pre-computation in a mixed-runtime implementation.

### 6.10.2 Full-Tree NIF — Eliminating the Interpreted Outer Loop

Speculative pre-computation (Section 6.10.1) reduces JIT dispatch to O(1–2) calls per tree build. But the tree builder's outer loop remains in Elixir: direction sampling, subtree construction dispatch, trajectory merging, U-turn termination checks, and `:rand` state threading. For a depth-5 tree with 5 doubling levels, this means ~5 Elixir map merge operations (~200us each), ~5 recursive function calls, and ~5 `:rand` state updates — approximately 800–1600us of interpreted overhead per tree build.

A key observation: once speculative pre-computation makes all leapfrog states available as contiguous forward/backward chains *before* tree building starts, the tree builder is a **pure function** from (initial state, pre-computed chains, RNG seed) to (selected sample, statistics). It has no side effects, no JIT calls, and no interaction with the BEAM scheduler. This makes the entire outer loop a candidate for NIF replacement.

**Implementation.** A Rust function `build_full_tree` takes the initial state (q0, p0, grad0, logp0), pre-computed forward and backward `PrecomputedStates`, inverse mass diagonal, joint log-probability, and max depth. It executes the full NUTS iterative doubling loop: direction sampling via Xoshiro256\*\*, subtree construction via the existing `build_subtree` (reusing the inner-subtree logic from Section 6.10), trajectory merging, and U-turn termination. A `PrecomputedStates::slice()` method provides sub-range views for each doubling level, avoiding data copies within the NIF.

The Elixir dispatch path becomes:
1. Two `multi_step_fn` JIT calls (forward + backward chains) — same as Section 6.10.1
2. `backend_copy` both chains to BinaryBackend
3. `Nx.to_binary()` for IEEE 754 f64 serialization
4. A single `NativeTree.build_full_tree_bin()` NIF call (17 parameters)
5. `Nx.from_binary()` to recover the selected sample

This reduces the per-tree call pattern from (2 JIT + 4–8 NIF) to (2 JIT + 1 NIF). The three NIF granularity levels form a progression:

| Level | Scope | Elixir ↔ Rust calls/tree | What is eliminated |
|-------|-------|--------------------------|---------------------|
| Inner-subtree NIF (6.10) | `build_subtree` only | 4–8 NIF/tree | Nx.slice + recursive merge per subtree |
| Speculative + inner (6.10.1) | pre-compute + inner NIF | 2 JIT + 4–8 NIF/tree | Per-level JIT dispatch |
| Full-tree NIF (6.10.2) | entire tree build | 2 JIT + 1 NIF/tree | All Elixir merge overhead |

Each level was motivated by profiling the previous one. The inner-subtree NIF (6.10) revealed that per-level JIT dispatch was the next bottleneck. Speculative pre-computation (6.10.1) eliminated that, revealing Elixir merge overhead as the next target. The full-tree NIF eliminates this final layer of interpreted overhead.

The full-tree NIF is wrapped in `try/rescue` with fallback to the speculative + inner-subtree path, preserving the fault tolerance properties described in Chapter 3. This means the three NIF levels also serve as a graceful degradation chain: if the full-tree NIF fails (e.g., due to a Rust panic), the system falls back to speculative + inner-subtree NIF; if that fails, it falls back to pure Elixir.

**Correctness.** The NIF path produces posteriors statistically equivalent to the Elixir path: both recover the analytic posterior mean within tolerance for Normal-Normal models. The Rust PRNG (Xoshiro256\*\*) produces different sequences than Erlang's `:rand.exsss`, so trajectories differ, but sampling quality (ESS, divergences, posterior accuracy) is equivalent — consistent with the finding from Section 6.10 that NIF PRNG divergence is statistically irrelevant.

**Benchmark results.** A/B comparison (3 seeds, centered parameterization, CPU):

| Model (d) | Full-tree NIF (ESS/s) | Speculative (ESS/s) | PyMC | NIF vs PyMC |
|-----------|:--------------------:|:------------------:|:----:|:-----------:|
| Simple (2) | **678** | 350 | 406 | **1.67x faster** |
| Medium (5) | 57 | 86 | 120 | 0.47x |
| Stress (8) | 41 | 65 | 162 | 0.25x† |

*†Stress model results pre-date the rho-based U-turn criterion (Section 6.10.3), which improved the speculative path from 67 to 89 ESS/s (0.57x PyMC).*

The full-tree NIF achieves a landmark result: **Exmc surpasses PyMC** on the simple model (678 vs 406 ESS/s). However, it regresses on deeper-tree models due to an **over-speculation problem**: the full-tree NIF pre-computes both forward and backward chains upfront, but the tree typically terminates early (via U-turn or divergence), leaving many pre-computed steps unused. For the simple model (average tree depth ~2), the adaptive budget is small (~15 steps per direction), so the waste is negligible and the elimination of Elixir merge overhead dominates. For the medium model (average depth ~5), the budget grows to ~63 steps per direction — most of which go unused.

**5-seed validation** reveals that the 3-seed A/B results for medium and stress were biased by seed=42, which consistently produces poor mu_global ESS. With 5 seeds (speculative path):

| Model (d) | Exmc median (ESS/s) | PyMC | Ratio |
|-----------|:------------------:|:----:|:-----:|
| Simple (2) | 358 | 406 | 0.88x |
| Medium (5) | **115** | 113 | **1.02x** |
| Stress (8) | **89** | 162 | **0.57x** |
| Simple (NIF) | **785** | 406 | **1.93x** |

*(Stress model updated with rho-based U-turn criterion, Section 6.10.3. Previous: 67 ESS/s = 0.41x.)*

The medium model gap was **seed bias, not systematic**: Exmc matches PyMC at 1.02x. Detailed profiling shows Exmc runs 1.9x faster per iteration (EXLA JIT leapfrog amortization) but produces ~1.8x lower min ESS per 1000 samples (250 avg vs PyMC's 456). The ESS quality gap stems from higher cross-seed variance and more warmup divergences (19 avg vs PyMC's 2), indicating that **adaptation robustness** — not raw computation speed — is the remaining frontier.

An adaptive budget mechanism tracks the maximum observed tree depth via the BEAM process dictionary, setting the budget to `2^(max_seen + 1) - 1`. A Rust-side bounds check gracefully terminates the tree if the budget is exhausted, preventing panics without requiring a fallback to the Elixir path. This partially mitigates over-speculation but cannot eliminate it: the budget must be set before the tree's actual depth is known.

**Theoretical significance.** The progression from inner-subtree NIF to full-tree NIF demonstrates that mixed-runtime PPL optimization is best approached incrementally. The monolithic NIF attempt (replacing the entire outer loop with NIF, Section 6.10) measured at 0.86x — *slower* than Elixir — because per-iteration binary conversion overhead exceeded tree logic savings. Only after speculative pre-computation eliminated the per-iteration JIT boundary crossing did the full-tree NIF become profitable: binary conversion happens once (for the entire chain) rather than per doubling level.

The over-speculation problem reveals a general principle for NIF granularity: **the optimal NIF scope depends on the ratio of pre-computation cost to per-call overhead**. When pre-computation is cheap (shallow trees, small budgets), a coarser NIF (full-tree) wins by eliminating more per-call overhead. When pre-computation is expensive (deep trees, large budgets), a finer NIF (inner-subtree) wins by avoiding wasted computation. The speculative path's lazy approach (compute on demand, extend if needed) naturally adapts to the tree depth distribution, making it the better default for models with unknown or deep trees.

### 6.10.3 Generalized U-Turn Criterion — Eliminating Inv-Mass Bias

The U-turn termination criterion determines when the NUTS trajectory has "turned around" and further extension would be unproductive. The original criterion (Hoffman and Gelman, 2014) checks whether the endpoint displacement dotted with the momentum has become negative: `(q⁺ - q⁻) · (M⁻¹ p) < 0`. This works well for homogeneous-scale models but introduces a systematic bias for hierarchical models with heterogeneous parameter scales.

**The bias mechanism.** The endpoint criterion computes `Σⱼ (q⁺ⱼ - q⁻ⱼ) × inv_massⱼ × pⱼ`. The inverse mass diagonal `inv_mass` appears as a per-dimension weight. For the stress model (3-group hierarchical, d=8), `inv_mass` ranges from 0.028 (mu_pop) to 5.6 (noise parameters) — a 200x range. This makes the dot product dominated by the high-variance mu_pop component. When mu_pop's displacement reverses direction, the U-turn check triggers even if other parameters are still productively exploring the posterior. The result is premature tree termination and shallow trees that undersample the posterior.

**The generalized criterion.** Betancourt (2017, §4.1) proposed a generalized criterion based on the cumulative momentum sum `ρ = Σ pᵢ` over all trajectory points: check `ρ · (M⁻¹ p⁺) < 0` or `ρ · (M⁻¹ p⁻) < 0`. This avoids the bias because `ρ` accumulates over the trajectory's momentum history rather than depending on endpoint separation. The `ρ` vector is tracked through the tree: leaf nodes initialize `rho = p_new`, subtree merges accumulate `rho = rho_first + rho_second`, and trajectory-level merges accumulate `rho = rho_traj + rho_subtree`.

**Implementation.** Both the Elixir tree builder (`check_uturn_rho` + `zip_reduce_rho`) and the Rust NIF (`check_uturn` in uturn.rs, `rho` field in TreeNode/Trajectory) were updated. The NIF subtree returns a `rho_bin` binary for trajectory-level accumulation in Elixir. The Erlang list-arithmetic approach from Phase 1 (Section 6.5) is retained — the rho-based check has the same computational structure (dot products on flat lists) as the endpoint check, so the 13x speedup over Nx is preserved.

**Results (5-seed benchmarks).**

| Model (d) | Before (endpoint) | After (rho) | PyMC | Ratio vs PyMC |
|-----------|:-----------------:|:----------:|:----:|:-------------:|
| Simple (2) | 174 | 213–427 | 406 | ~1.0x |
| Medium (5) | 115 | 83–121 | 113 | ~1.0x (neutral) |
| Stress (8) | 67 | **89** | 162 | **0.57x** (was 0.41x) |

The stress model improvement is substantial: +33% ESS/s, with average tree depth increasing from 2.2 to 2.9 and step size from 0.46 to 0.60. Deeper trees indicate that the sampler now explores the posterior more thoroughly rather than terminating prematurely due to mu_pop reversal. Warmup divergences increase from ~5 to 31 (the different tree depths change the adaptation path), but sampling divergences drop to near-zero (1/1000).

The medium model is unaffected because its inv_mass range is smaller (~10x vs 200x), so the endpoint criterion was not significantly biased.

**Significance.** Stan uses the generalized criterion by default — this was a latent implementation gap. The discovery required profiling per-parameter ESS on the stress model to identify premature U-turn as the bottleneck, then tracing the bias to the endpoint criterion's extra `inv_mass` factor per dimension. This is a PPL design space insight: U-turn criterion choice is invisible on simple models but becomes dominant for hierarchical models with heterogeneous scales. The 43 documented decisions now include this as D43.

### 6.10.4 Adaptation Quality as a Coupled System — Closing the Stress Model Gap

The rho-based U-turn criterion (Section 6.10.3) improved stress model ESS/s by 33%, but a 0.57x gap vs PyMC persisted. Investigation revealed two remaining adaptation quality differences, both undocumented in the published NUTS algorithm but implemented in Stan's source code.

**Divergent sample exclusion (D44).** Stan excludes divergent transitions from the Welford mass matrix estimator. Divergent trees are truncated early; the multinomial-selected proposal from a truncated tree is biased toward the starting point. Including these biased samples inflates mass matrix variance estimates. For the stress model with ~30-40 warmup divergences out of ~875 Phase II iterations (3-5%), the contamination is small in fraction but concentrated in early adaptation windows when the mass matrix is poorest — precisely when accurate estimation matters most.

**Phase III allocation (D45).** Exmc used `term_buffer = min(200, max(50, num_warmup/5))`, allocating 200 iterations to Phase III (final step size adaptation) for 1000 warmup iterations. Stan uses `term_buffer = 50`. This difference had been intentional: before the log_epsilon_bar initialization fix (D26), dual averaging genuinely needed ~200 iterations because it started from a biased initial point (log_epsilon_bar = 0, biasing toward eps = 1.0). After D26, dual averaging converges in ~50 iterations from the correct starting point. The extra 150 Phase III iterations were wasted — iterations that could have been Phase II mass matrix adaptation. With `term_buffer = 50`, the final mass matrix window grows from ~350 to ~500 samples (+43%), directly improving mass matrix quality at d >= 5.

**Multiplicative interaction.** The two fixes interact: more mass matrix samples (term_buffer) combined with cleaner samples (divergent skip) produce a superlinear improvement. The dependency chain D26 → D45 → D44 demonstrates that warmup parameters form a coupled system: the log_epsilon_bar fix was a *prerequisite* for reducing term_buffer, and the divergent skip amplifies the benefit of additional samples.

**Results (5-seed medians, 1ch CPU).**

| Model (d) | Before | After | Change | vs PyMC |
|-----------|:------:|:-----:|:------:|:-------:|
| Simple (2) | 179.8 | 286.0 | +59% | — |
| Medium (5) | 26.5 | 55.3 | +109% | — |
| Stress (8) | 13.3 | 37.3 | +180% | — |

*(These are 1ch no-NIF baselines. With NIF/speculative paths and multi-chain, absolute numbers are higher.)*

**Significance for PPL design.** Both fixes are invisible in the published NUTS algorithm (Hoffman and Gelman, 2014) and the HMC tutorial (Betancourt, 2017). They are documented only in Stan's C++ source code comments. A PPL implementor reading only the published literature would miss both optimizations. This represents a concrete instance of the "folklore knowledge" barrier in PPL development: the gap between what is published and what is practiced. For a BEAM-hosted PPL without access to Stan's training data (i.e., years of developer experience encoded in source comments), closing this gap requires systematic A/B benchmarking against mature implementations. The 52 documented decisions (D1-D52) serve as a structured record of this reverse-engineering process.

### 6.10.5 Sub-Trajectory U-Turn Checks — A Third Undocumented Practice

The published NUTS algorithm (Betancourt, 2017) describes one U-turn check per tree merge: `ρ · (M⁻¹ p±) < 0` applied to the full merged trajectory. Investigation of PyMC's source code (`nuts.py`, `_build_subtree` and `extend` methods) revealed two additional checks per merge:

1. **Check 2 (left sub + right's first point):** `(ρ_left + p_right_first) · (M⁻¹ p) < 0` for both endpoints of the junction
2. **Check 3 (left's last point + right sub):** `(p_left_last + ρ_right) · (M⁻¹ p) < 0` for both endpoints of the junction

These "sub-trajectory" checks detect U-turns at the boundary between the two sub-trajectories being merged, preventing the sampler from extending a trajectory that has already started turning back. They are gated on `depth > 0` in subtree merges (leaf-to-leaf merges have no sub-trajectory structure) and always active in trajectory-level merges.

**Implementation.** Both Elixir (`merge_subtrees` and `merge_trajectories` in `tree.ex`) and Rust (`merge_subtrees` and `merge_into_trajectory` in `tree.rs`) paths were updated. The direction-dependent assignment `{left_sub, right_sub} = if going_right, do: {first, second}, else: {second, first}` ensures correct left/right ordering for both forward and backward chains.

**Results (A/B test, pure Elixir path, 5-seed median).**

| Model | Without sub-traj | With sub-traj | Change |
|-------|:---------------:|:------------:|:------:|
| Medium ESS/s | 82.1 | 119.5 | **+46%** |
| Medium warmup div | 34 | 20 | **-41%** |
| Stress ESS/s | 77 | 77 | neutral |
| Stress warmup div | 22 | 16 | **-27%** |

The medium model benefits most because sub-trajectory checks prevent the sampler from building unnecessarily deep trees (warmup divergences drop 41%), directing adaptation toward better mass matrix estimates. The stress model shows reduced warmup divergences but no ESS/s change, suggesting that its remaining gap is not U-turn-related.

**NIF interaction.** Adding sub-trajectory checks to the Rust NIF revealed a latent direction bug: `build_full_tree` had always passed `going_right=true` to `build_subtree` for both forward and backward chains. The full-trajectory U-turn check is mathematically symmetric (swapping p_left and p_right gives the same result), so this bug was invisible. Sub-trajectory checks broke the symmetry, causing severe ESS regression. Fixing the direction (passing the actual `go_right` value) restored correctness, but the more aggressive tree termination from sub-trajectory checks now wastes more speculatively pre-computed leapfrog states, making the full-tree NIF slower than the speculative path for all models. The NIF is now disabled by default.

This is an instance of a general pattern: **symmetric correctness criteria can mask asymmetric implementation errors**. The bug persisted through months of development and testing because the symmetric U-turn check produced valid (if suboptimal) results regardless of endpoint assignment. Only the introduction of an asymmetric check exposed the direction error.

**Combined effect of all three undocumented practices (D44 + D45 + D47).**

Final 5-seed medians (speculative path):

| Model (d) | Before all fixes | After all fixes | Change | vs PyMC |
|-----------|:---------------:|:--------------:|:------:|:-------:|
| Simple (2) | 211 | **233** | +10% | 0.57x |
| Medium (5) | 115 | **116** | +1% | **1.02x** |
| Stress (8) | 67 | **107** | +60% | **0.66x** |

**Significance for PPL design.** Three undocumented Stan practices — divergent sample exclusion, Phase III allocation, and sub-trajectory U-turn checks — collectively improved stress model ESS/s by 60% (67→107) and narrowed the PyMC gap from 0.41x to 0.66x. All three required reading Stan's source code; none is described in the NUTS paper or HMC tutorial. This reinforces the thesis claim (Section 6.10.4) that the "folklore knowledge" barrier is a significant obstacle for independent PPL implementations. The 52 documented decisions (D1–D52) serve as a structured record of this reverse-engineering process.

### 6.10.6 Multinomial Sampling Correctness — Two Hidden Load-Bearing Properties

The NUTS tree builder's multinomial sampling has two load-bearing implementation properties that are underdocumented in the literature: (1) leaf-level log-weights must be uncapped, and (2) the outer merge (trajectory extension) must use biased progressive sampling, not the balanced multinomial used for inner merges. Both properties have subtle effects: incorrect implementations produce valid posteriors (MCMC correctness is preserved) but dramatically reduce effective sample size by inflating the duplicate sample rate.

**D49: Uncapped leaf log-weight.** At each tree leaf (depth 0), the NUTS algorithm assigns a log-weight `d = logp_new - logp_0` representing how much better the new trajectory point is than the starting point. Exmc's implementation used `log_weight = min(0.0, d)`, capping the weight at `exp(0) = 1`. This meant that trajectory points with *better* energy than the start (d > 0) received the same weight as the starting point. The multinomial sampling then underweighted these better points, biasing selection toward q_0 and inflating the duplicate sample rate to 37.7% (compared to PyMC's 7.8%).

The cap was a conflation of two distinct quantities: the Metropolis-Hastings acceptance probability `min(1, exp(d))` (correctly capped — used for dual averaging feedback) and the multinomial proposal weight `exp(d)` (which must be uncapped to reflect the relative quality of trajectory points). This distinction appears in Betancourt (2017, Algorithm 3) but is easy to miss because both quantities use the same energy difference `d`.

**D50: Biased progressive outer merge.** Stan and PyMC use different multinomial sampling rules for inner merges (within `build_subtree`) and outer merges (trajectory extension in the main loop). Inner merges use balanced multinomial: `P(accept second) = w_second / (w_first + w_second)`. Outer merges use biased progressive sampling (Betancourt 2017, Appendix A.3.2): `P(accept subtree) = min(1, w_subtree / w_trajectory)`. Exmc used balanced multinomial for *both*, making q_0 "sticky" — it survived outer merges with probability `w_traj / (w_traj + w_subtree)` when it should have survived with `max(0, 1 - w_subtree / w_traj)`.

The difference is significant when the subtree outweighs the existing trajectory (which is common early in tree building). With balanced sampling, the existing trajectory retains substantial selection probability even when the subtree is much better. With biased progressive, the subtree is always accepted when it outweighs the trajectory. A diagnostic analysis of PyMC's `index_in_trajectory` confirmed the difference: at tree depth 2, q_0 was selected only 3.7% of the time (vs balanced prediction of 25%).

**Combined impact (1-chain, 5-seed medians).**

| Model (d) | Before (ESS/s) | After (ESS/s) | Change | vs PyMC |
|-----------|:--------------:|:------------:|:------:|:-------:|
| Simple (2) | 233 | **469** | +101% | **0.81x** |
| Medium (5) | 116 | **298** | +157% | **1.90x** |
| Stress (8) | 107 | **215** | +101% | **1.16x** |

*PyMC baselines updated to live 5-seed race: simple 576, medium 157, stress 185 ESS/s.*

Duplicate sample rate dropped from 37.7% to 6.5% (PyMC: 7.8%). The medium model improvement is the most striking: from 3.3 ESS/s at Phase 0 to 298 ESS/s — a 90x improvement over the entire optimization trajectory, now substantially exceeding PyMC's 157 ESS/s. **Exmc beats PyMC on medium (1.90x) and stress (1.16x) models.** The simple model lags at 0.81x, likely due to per-tree Elixir overhead dominating at shallow tree depths (avg depth ~2).

**Significance.** These two bugs are representative of a broader problem in PPL reimplementation: the NUTS algorithm's correctness (in the MCMC validity sense) is robust to many implementation variations — incorrect multinomial weights still produce valid posteriors because the acceptance probability is bounded. But *efficiency* (ESS per sample) is highly sensitive to these implementation details. The distinction between inner and outer merge sampling is documented only in Betancourt (2017, Appendix A.3.2), not in the original NUTS paper (Hoffman and Gelman, 2014). A systematic audit of independent NUTS implementations (Turing.jl, Blackjax, etc.) may reveal that this class of "correctness-preserving efficiency bugs" is widespread.

### 6.11 JIT Boundary Cost Model

The three-category model from Section 6.7 can be quantified with a simple affine cost model: `C(op, d) = alpha + beta * d`, where `alpha` captures fixed dispatch overhead and `beta` captures per-element computation cost.

**Method.** We sweep 20 operations across 8 dimensions (d ∈ {1, 2, 5, 10, 20, 50, 100, 500}) on three backends (EXLA.Backend, BinaryBackend, Erlang native), measuring median execution time over 500 iterations per configuration. Linear regression yields the cost model parameters.

**Results.** The three categories exhibit distinct cost signatures:

| Category | alpha (us) | beta (us/elem) | R² | Interpretation |
|----------|-----------|----------------|-----|----------------|
| EXLA.Backend ops | 41–91 | ~0 | <0.3 | Constant dispatch overhead, no dimension scaling |
| BinaryBackend ops | 0–3 | 0.15–0.61 | >0.99 | Linear scaling, minimal fixed overhead |
| Erlang scalar ops | ~0 | 0.02–0.10 | >0.99 | Linear scaling, 5–10x cheaper per element than Nx |

EXLA operations show near-zero R² for the dimension regression, confirming that their cost is entirely dispatch overhead (~40–90us) independent of tensor size. BinaryBackend and Erlang operations show excellent linear fits (R² > 0.99), with Erlang achieving 5–10x lower per-element cost due to zero Nx framework overhead.

**Crossover analysis.** The cost model identifies concrete crossover dimensions:

- **Nx.add (BB) vs Erlang add:** Nx wins for d > 4.5 (vectorization amortizes framework overhead)
- **U-turn check: Nx vs Erlang:** Nx wins for d > 5.7 (our d=5 medium model sits at the crossover)
- **EXLA vs BinaryBackend (Nx.add):** BinaryBackend wins for d < 71.7 (EXLA dispatch too expensive for small tensors)
- **Nx.sum vs Erlang sum:** Erlang always faster (no crossover at practical dimensions)

The U-turn crossover at d~6 validates the design decision to use Erlang flat-list arithmetic for the U-turn check (Section 6.5): at d=5, the Erlang implementation (2us) is 10x faster than Nx (20us). The crossover predicts that for models with d > 50, the Nx-based U-turn would be competitive — but for MCMC where d rarely exceeds 1000, the Erlang approach wins or ties across the entire practical range.

**Validation.** A d=22 hierarchical model (10 groups, 20 observations each) provides an unseen test case. The per-leapfrog cost decomposes as:

| Component | Predicted | % of total |
|-----------|----------|------------|
| step_fn (JIT gradient evaluation) | 303us | 76% |
| Leaf creation (4× backend_copy + extract) | 92us | 23% |
| Merge overhead (U-turn + LSE + map) | 2us | 1% |
| **Total per leapfrog step** | **397us** | |

At average tree depth 2.9 (~6.5 leapfrog steps per NUTS iteration), the model predicts 2592us per iteration. Actual measured: ~4453us per iteration (ratio 1.72x). Hot-loop profiling (100 post-warmup iterations) identifies the source of this gap:

| Component | Median (us) | % of total |
|-----------|-------------|------------|
| Tree.build | 4027 | 98.0% |
| joint_logp | 73 | 1.8% |
| Momentum sampling | 21 | 0.5% |
| Result extraction | 2 | 0.0% |
| State update | 1 | 0.0% |
| **TOTAL** | **4110** | 100% |

The actual per-leapfrog cost is 576us vs the predicted 397us --- a 178us gap (45%). This unmodeled overhead is *internal* to the tree builder: Elixir map creation (~19 fields per subtree merge), recursive function call overhead, `:rand` state threading, and Erlang garbage collection pressure. Non-tree components (momentum sampling, joint_logp computation, state updates) are negligible at 97us total. This quantifies the interpretive overhead of a BEAM-hosted tree builder: even with JIT gradients, NIF inner subtrees, and optimal backend placement, the Elixir tree builder contributes ~31% overhead per leapfrog step.

The cost model confirms three actionable design rules for mixed-runtime PPLs:

1. **JIT the gradient, not the tree.** Gradient evaluation (76% of per-step cost) should be JIT-compiled. Tree logic (merge overhead, 1%) should stay in the host language where it can handle dynamic control flow.
2. **Copy at the JIT boundary.** Backend_copy (23% of cost) is the price of correct boundary placement. Without it, EXLA tensors leak into host-side tree ops, inflating merge cost by 10–30x (Section 6.3).
3. **Use native arithmetic for scalar ops below the crossover.** At d < 50, Erlang scalar arithmetic beats Nx for dot products, sums, and comparisons. This covers all practical MCMC dimensions for tree-builder operations.

### 6.11.1 GPU Acceleration — The JIT Boundary is Device-Agnostic

The JIT boundary architecture described above (step_fn closure + backend_copy barrier) is device-agnostic: the same `step_fn` closure runs on CPU or GPU without modification. However, GPU acceleration provides meaningful per-step speedup only at high dimensions (d > 200) for individual calls. The key to GPU utilization at practical MCMC dimensions is the batched leapfrog (Section 6.6), which amortizes kernel launch overhead inside XLA while-loops.

**Method.** We sweep step_fn dispatch across d in {6, 10, 20, 50, 100, 300, 500} on CPU vs GPU (NVIDIA RTX 3060 Ti, 8GB), including the backend_copy barrier to BinaryBackend. We then run full sampling on a d=22 hierarchical model (10 groups, 20 observations each, centered parameterization) to measure end-to-end speedup.

**Per-step_fn results.**

| d | CPU+bcopy (us) | GPU+bcopy (us) | Speedup |
|---|----------------|----------------|---------|
| 6 | 1622 | 1372 | 1.18x |
| 10 | 1327 | 1338 | 0.99x |
| 20 | 1617 | 1640 | 0.99x |
| 50 | 3053 | 3066 | 1.0x |
| 100 | 5906 | 5736 | 1.03x |
| 300 | 34132 | 17574 | 1.94x |
| 500 | 47354 | 43335 | 1.09x |

For d < 100, GPU and CPU are statistically indistinguishable per step_fn call. Kernel launch latency (~1ms) dominates the per-call budget, consuming any per-element GPU advantage. At d=300, GPU vectorization wins decisively (1.94x). The d=500 result (1.09x, regression from d=300) suggests memory transfer overhead begins to dominate at very high dimensions for individual calls.

**Linear cost model fits.**

| Device | Intercept alpha (us) | Slope beta (us/dim) | R^2 |
|--------|---------------------|--------------------|----|
| CPU+bcopy | -210.7 | 98.93 | 0.98 |
| GPU+bcopy | -450.8 | 79.8 | 0.962 |

The GPU slope is ~20% lower (79.8 vs 98.93 us/dim), reflecting faster per-element computation. But the similar intercepts mean that at practical MCMC dimensions (d = 5-50), the fixed overhead dominates and there is no GPU advantage for individual step_fn calls.

**Full sampling results (d=22 hierarchical model).**

| Device | Wall time (ms) | ESS/s | Step size | Divergences |
|--------|---------------|-------|-----------|-------------|
| CPU | 12411 | 3.9 | 0.3299 | 4 |
| GPU | 4682 | 10.3 | 0.3299 | 4 |
| **Speedup** | **2.65x** | **2.65x** | identical | identical |

The 2.65x full-sampling speedup far exceeds any individual per-step_fn speedup at d=22 (~1x). This confirms that the batched leapfrog (XLA while-loop, Section 6.6) is the mechanism through which GPU acceleration is realized at moderate dimensions. The XLA while-loop compiles multiple leapfrog steps into a single GPU kernel, amortizing the ~1ms launch overhead across all steps in a subtree. Sampling quality is identical: same adapted step size, same divergence count, confirming device-agnostic correctness.

**Implications.**

1. **Device-agnostic JIT boundaries.** The same `step_fn` closure, backend_copy barrier, and tree builder code work on both CPU and GPU. No code changes are needed to switch devices --- only the EXLA backend configuration changes. This validates the boundary placement decisions from Section 6.11: the boundary is between "JIT-compiled gradient" and "interpreted tree builder," not between "CPU computation" and "GPU computation."

2. **Batched leapfrog enables GPU utilization.** Without the XLA while-loop (Section 6.6), GPU acceleration at d=22 would be ~1x (no benefit). The while-loop transforms many small kernel launches into one large kernel, crossing the GPU utilization threshold. This provides a second justification for the batched leapfrog design beyond dispatch overhead reduction.

3. **The backend_copy barrier is device-agnostic.** The ~20us backend_copy cost (EXLA -> BinaryBackend) is the same whether the source tensor is on CPU-EXLA or GPU-EXLA. This means the "JIT boundary tax" identified in Section 6.11 does not increase with GPU acceleration.

NumPyro/JAX achieves GPU acceleration naturally because the entire sampler (including tree builder) is JIT-compiled into a single XLA program. Stan has no GPU support for NUTS. PyMC's C-compiled sampler has no GPU path. The Exmc result shows that a mixed-runtime PPL (interpreted tree builder + JIT gradient) can still achieve substantial GPU speedup (2.65x) through careful batching, even though the tree builder itself never executes on the GPU.

---

## Chapter 7: Conclusion

### 7.1 Contributions

This thesis has demonstrated that the actor-model runtime provides properties for probabilistic programming that are absent from all existing systems. We summarize the contributions with their supporting evidence.

**A probabilistic programming framework on the BEAM.** Exmc implements NUTS, ADVI, SMC, and Pathfinder inference in Elixir, with a Builder API for model specification, an automatic compilation pipeline, and integration with the Nx/EXLA tensor ecosystem. The framework is documented through 52 architectural decisions with falsifiable assumptions, six of which were violated and revised during development. The system comprises approximately 5,000 lines of Elixir and 500 lines of Rust, with 199 tests.

**Zero-cost fault-tolerant sampling.** BEAM's `try/rescue` mechanism provides fault containment for NUTS subtree computation with zero overhead on the non-faulting path (Chapter 3). A two-tier supervision mechanism — `try/rescue` for numerical and EXLA errors, `Task.yield` for hard hangs — replaces crashed subtrees with divergent placeholders that integrate seamlessly with the existing tree merge logic. Combined with three layers of numerical failure detection (leaf-level NaN/Inf checks, subtree early termination, trajectory-level divergence), Exmc provides four layers of fault tolerance. No existing PPL recovers from mid-sampling crashes.

**Streaming inference via message passing.** The `sample_stream` protocol sends posterior samples as messages to any BEAM process (Chapter 4). This integrates directly with Scenic for real-time visualization (ExmcViz, 64 tests) and with Phoenix LiveView for browser-based posterior monitoring (12 tests). The LiveView integration requires no intermediate infrastructure — the web framework process serves directly as the sample receiver. No existing PPL provides streaming inference integrated with a production web framework.

**Distributed MCMC with transparent fault recovery.** Erlang's `:erpc` module enables multi-node chain dispatch where model IR is distributed to remote nodes that compile and sample independently (Chapter 5). Node failure triggers transparent retry on the coordinator. A 7-model distributed benchmark with 5 `:peer` nodes demonstrates avg 2.87x scaling, with Exmc distributed beating PyMC 4-chain on medium (841 vs 680, 1.24x) and funnel (5.4 vs 4.1, 1.32x). The closure barrier — Custom distributions capturing `Nx.Defn.Expr` tensor graph fragments — is identified as the distribution boundary; fault recovery provides 2.5-2.8x local concurrency as automatic fallback for non-distributable models. The entire distributed system is 203 lines with zero external dependencies.

**JIT boundary cost model and gap decomposition.** We developed the first quantitative analysis of JIT boundary placement in mixed-runtime PPLs (Chapter 6). An affine cost model across 20 operations and 8 dimensions reveals three operation categories with distinct cost signatures: constant-overhead JIT dispatch, linearly-scaling tensor operations, and near-zero-cost native scalar arithmetic. The model identifies concrete crossover dimensions (e.g., d~6 for U-turn checks, d~72 for EXLA vs BinaryBackend). The gap decomposition methodology separates architectural overhead from algorithmic quality, revealing that a 34x performance gap can be reduced to **1.90x faster than PyMC** through parameterization choice, FFI boundary placement, progressive NIF migration, and multinomial sampling correctness fixes (Section 6.10.6).

**GPU acceleration via batched JIT.** The device-agnostic JIT boundary architecture enables GPU acceleration without modifying the tree builder (Section 6.11.1). At d=22, GPU sampling achieves 2.65x wall-time speedup despite individual step_fn calls showing no GPU advantage at that dimension. The mechanism is the batched leapfrog (XLA while-loop), which amortizes kernel launch overhead across all steps within a subtree.

### 7.2 Limitations

We identify several limitations of this work that constrain the generality of our conclusions.

**Benchmark model coverage.** The optimization trajectory (Sections 6.1–6.10) was characterized primarily on three custom models (simple d=2, medium d=5, stress d=8). We subsequently validated these findings on four canonical PPL benchmarks — Eight Schools (d=10), Neal's Funnel (d=10), Logistic Regression (d=21), and Stochastic Volatility (d=102) — confirming the adaptation-vs-throughput tradeoff across model classes (Section 6.12). However, all seven models are unimodal with continuous parameters; multimodal posteriors, discrete parameters, and mixture models remain untested.

**Same-host distributed benchmarks.** The distributed MCMC experiments (Chapter 5) use OTP `:peer` nodes on a single physical machine. These nodes share CPU cores, memory bandwidth, and the EXLA runtime. True multi-host benchmarks on separate machines would provide a more realistic assessment of distribution overhead and would likely show better scaling due to reduced resource contention. The 7-model distributed benchmark (Section 5.4) demonstrates avg 2.87x from 5 nodes on same-host; multi-host scaling should approach 5x with eliminated resource contention.

**Closure barrier limits distribution scope.** Custom distributions with `Nx.Defn.Expr`-captured tensors (logistic, SV) cannot distribute to remote nodes (Section 5.4). The fault recovery path provides 2.5-2.8x local concurrency, but this is fundamentally limited by single-host resources. The fix (embedding data in IR) is architectural and would require API changes.

**BinaryBackend limitations.** The tree builder operates on BinaryBackend tensors for performance reasons (Section 6.3). This is optimal for the low-dimensional models tested (d=5 to d=22) but may become a bottleneck for high-dimensional models where the per-element cost of BinaryBackend operations exceeds the per-dispatch cost of EXLA operations. The crossover analysis (Section 6.11) predicts this at d~72.

**No automatic parameterization selection.** The NCP vs centered discovery (Section 6.9) demonstrates that parameterization choice dominates performance, but Exmc provides only a manual `ncp: false` option. Automatic selection based on data informativeness remains an open problem across all PPLs (Papaspiliopoulos et al., 2007).

**Single-developer system.** Exmc has been developed and evaluated by a single researcher. The API design, documentation quality, and error messages have not been evaluated by external users. The system's usability for practitioners outside the development team is unknown.

### 7.3 Future Work

Several directions emerge from this work.

**Automatic parameterization selection.** The data informativeness ratio (observations relative to prior variance) likely determines whether NCP or centered parameterization is better. A heuristic based on initial warmup ESS — running a short pilot with each parameterization — could automate this choice. More ambitiously, per-parameter parameterization (mixing NCP and centered within a single model) would address models where different hierarchical levels have different identification strengths.

**Multi-host distributed benchmarks.** True distributed evaluation on separate machines would quantify network latency, EXLA compilation overhead on heterogeneous hardware, and the scaling properties of the warmup-once-broadcast protocol. Model IR caching via an ETS table keyed by IR hash would avoid recompilation for repeated sampling of the same model on the same node.

**Compiled tree builder (partially addressed).** The full-tree NIF (Section 6.10.2) moves the entire tree build into Rust, eliminating interpreted-host merge overhead. However, the two JIT `multi_step_fn` calls (forward + backward chains) remain in EXLA, and the binary serialization boundary (Nx.to_binary/from_binary) introduces its own overhead. A fully compiled approach — compiling the tree builder into XLA alongside the gradient (as NumPyro does) — would eliminate both the serialization boundary and the remaining JIT dispatch. The challenge remains preserving the fault tolerance and streaming properties that depend on the tree builder executing in the BEAM.

**Online and warm-start inference.** The streaming protocol (`sample_stream`) and distributed architecture together suggest a "Bayesian inference as a service" pattern: a long-running process that accepts new observations and updates the posterior incrementally. Warm-starting from a previous posterior (using the adapted step size and mass matrix) would reduce warmup cost for repeated inference on evolving datasets.

**GPU scaling to high dimensions.** The d=22 GPU results show 2.65x speedup via batched leapfrog, but per-step_fn GPU advantage emerges only above d~200. High-dimensional models (d > 100) that are common in spatial statistics and deep generative models would better exploit GPU parallelism. The BinaryBackend crossover at d~72 may require a hybrid strategy where tree operations transition from host-native to tensor operations at high dimensions.

**Actor-model sampling.** A more radical architecture would represent each random variable as a BEAM process, with sampling proceeding via message passing between variable-processes. This would provide natural fault isolation per variable and could enable incremental re-sampling when observations change.

### 7.4 Closing Statement

The actor-model runtime provides fault tolerance, liveness, and distribution as language-level primitives. These properties are valuable for probabilistic programming but absent from every existing PPL. Exmc demonstrates that they are practically useful — not merely theoretically interesting — by enabling crash-resilient sampling, real-time browser-based posterior visualization, and multi-node inference with zero external infrastructure. The performance trajectory demonstrates that an interpreted runtime's overhead is not fixed: progressive optimization (backend placement, batched leapfrog, NIF subtrees, speculative pre-computation, multinomial sampling correctness) reduced the gap from 34x to **exceeding PyMC**. On a 7-model benchmark suite spanning d=2 to d=102 — including the canonical Eight Schools, Neal's Funnel, Logistic Regression, and Stochastic Volatility models — **Exmc wins 4 models to PyMC's 3**. Exmc wins on adaptation-bound models: medium 1.65x, stress 1.25x, Eight Schools 2.55x, and Stochastic Volatility 1.20x. PyMC wins on throughput-bound models: simple 0.81x, logistic 0.21x, and the pathological funnel 0.40x. This pattern reveals a structural tradeoff: an interpreted-host PPL with JIT'd gradients has equivalent adaptation quality but higher per-step overhead; which factor dominates depends on the model's geometry. The final 2–3x improvement came from fixing two subtle multinomial sampling bugs (Section 6.10.6) — capped leaf log-weights and balanced outer merge — that preserved MCMC correctness while silently inflating the duplicate sample rate from 7.8% to 37.7%. With distributed 4-node sampling via `:peer` nodes (Chapter 5), Exmc achieves 3.4–3.7x near-linear scaling, with medium model ESS/s of 812 — projected ~5x faster than PyMC's multiprocessing-based 4-chain approach. The discovery that five undocumented Stan practices (divergent Welford exclusion, Phase III allocation, sub-trajectory checks, uncapped leaf weights, biased progressive merge) each had measurable impact highlights the "folklore knowledge" barrier in PPL development — the gap between published algorithms and production implementations. The JIT boundary analysis provides a general framework for reasoning about performance in mixed-runtime systems and predicting where further optimization is profitable.

---

## Appendix A: Experiments Summary

| # | Section | Key Result |
|---|---------|------------|
| 1 | 6.2–6.3 | EXLA.Backend 10–30x slower for small ops; backend_copy gives 3x speedup |
| 2 | 2.4 | :rand 10^6x faster than Nx.Random for scalar decisions |
| 3 | 6.9 | Dense mass hurts when warmup samples << d^2 |
| 4 | 2.3 | NCP init inversion gives 2x ESS/s improvement |
| 5 | 3.3 | Supervised tree building: zero-cost fault containment, 15 tests |
| 6 | 5.3 | Distributed via :erpc: 1.66x ESS/s speedup, zero-cost fault recovery |
| 7 | 4.3 | Live Bayesian inference via Phoenix LiveView, 12 tests |
| 8 | 6.11 | JIT boundary cost model: 3-category fits (R²>0.99), GPU 2.65x at d=22 |
| 9 | 7.3 | Actor-model sampling (planned future work) |
| 10 | 6.6 | Batched leapfrog: 66.9x dispatch speedup at N=128 |
| 11 | 6.5 | Phase 1 micro-opts: +15% simple, +48% medium |
| 12 | 6.8 | Phase 3 adaptation: +13% simple, +30% medium |
| 13 | 6.8.1 | Micro-optimization stacking: predicted 1.45x, actual 1.08x |
| 14 | 6.9 | Mass matrix diagnostic: sigma_obs ESS bottleneck from NCP |
| 15 | 6.9 | NCP vs centered: centered 9x better (49.1 vs 5.5 ESS/s) |
| 16 | 6.10 | Rust NIF tree builder: inner-subtree 1.5x, outer-loop 0.86x |
| 17 | 6.10.1 | Speculative pre-computation: O(max_depth)→O(1-2) JIT dispatch calls |
| 18 | 6.10.2 | Full-tree NIF: entire tree build in single Rust NIF call |
| 19 | 6.10.3 | Rho-based U-turn criterion: +33% stress ESS/s (67→89), 0.41x→0.57x PyMC |
| 20 | 6.10.4 | Adaptation quality (divergent skip + term_buffer): +180% stress, +109% medium |
| 21 | 6.10.5 | Sub-trajectory U-turn checks: +46% medium ESS/s, -41% warmup divergences |
| 22 | 6.10.6 | Multinomial sampling fixes (uncapped log_weight + biased progressive merge): duplicate rate 37.7%→6.5%, ESS 2-3x improvement. Exmc beats PyMC: medium 1.90x, stress 1.16x |
| 23 | 5.3 | Distributed 4-node :peer benchmark: 3.4-3.7x near-linear scaling, medium 812 ESS/s |
| 24 | 4.4 | Distributed live streaming: 4 nodes × 5000 samples, 948 samples/s, zero-integration composition of Ch 4 + Ch 5 |
| 25 | 6.12 | Standard PPL benchmarks (7 models, d=2–102): Exmc 4, PyMC 3. Adaptation-bound models favor Exmc (Eight Schools 2.55x, medium 1.65x, stress 1.25x, SV 1.20x); throughput-bound favor PyMC (logistic 0.21x, simple 0.81x, funnel 0.40x) |
| 26 | 5.4 | 7-model distributed benchmark: avg 2.87x scaling from 5 nodes. Exmc dist beats PyMC 4ch on medium (1.24x) and funnel (1.32x). Closure barrier: Custom dists with captured tensors fall back to coordinator via fault recovery (2.5-2.8x from local concurrency) |

---

## Bibliography

- Armstrong, J. (2003). *Making reliable distributed systems in the presence of software errors.* PhD thesis, Royal Institute of Technology, Stockholm.
- Betancourt, M. (2017). A conceptual introduction to Hamiltonian Monte Carlo. *arXiv preprint arXiv:1701.02434.*
- Bingham, E., Chen, J. P., Jankowiak, M., Obermeyer, F., Pradhan, N., Karaletsos, T., Singh, R., Szerlip, P., Horsfall, P., & Goodman, N. D. (2019). Pyro: Deep universal probabilistic programming. *Journal of Machine Learning Research, 20*(28), 1–6.
- Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational inference: A review for statisticians. *Journal of the American Statistical Association, 112*(518), 859–877.
- Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D., Necula, G., Paszke, A., VanderPlas, J., Wanderman-Milne, S., & Zhang, Q. (2018). JAX: Composable transformations of Python+NumPy programs. *GitHub repository.*
- Carpenter, B., Gelman, A., Hoffman, M. D., Lee, D., Goodrich, B., Betancourt, M., Brubaker, M., Guo, J., Li, P., & Riddell, A. (2017). Stan: A probabilistic programming language. *Journal of Statistical Software, 76*(1), 1–32.
- Duane, S., Kennedy, A. D., Pendleton, B. J., & Roweth, D. (1987). Hybrid Monte Carlo. *Physics Letters B, 195*(2), 216–222.
- Ge, H., Xu, K., & Ghahramani, Z. (2018). Turing: A language for flexible probabilistic inference. *International Conference on Artificial Intelligence and Statistics (AISTATS).*
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.
- Hoffman, M. D. & Gelman, A. (2014). The No-U-Turn Sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research, 15*(1), 1593–1623.
- Jacob, P. E., O'Leary, J., & Atchadé, Y. F. (2020). Unbiased Markov chain Monte Carlo methods with couplings. *Journal of the Royal Statistical Society: Series B, 82*(3), 543–600.
- Neal, R. M. (2011). MCMC using Hamiltonian dynamics. In S. Brooks, A. Gelman, G. L. Jones, & X.-L. Meng (Eds.), *Handbook of Markov Chain Monte Carlo* (pp. 113–162). CRC Press.
- Neiswanger, W., Wang, C., & Xing, E. (2014). Asymptotically exact, embarrassingly parallel MCMC. *Uncertainty in Artificial Intelligence (UAI).*
- Nesterov, Y. (2009). Primal-dual subgradient methods for convex problems. *Mathematical Programming, 120*(1), 221–259.
- Papaspiliopoulos, O., Roberts, G. O., & Sköld, M. (2007). A general framework for the parametrization of hierarchical models. *Statistical Science, 22*(1), 59–73.
- Phan, D., Pradhan, N., & Jankowiak, M. (2019). Composable effects for flexible and accelerated probabilistic programming in NumPyro. *arXiv preprint arXiv:1912.11554.*
- Salvatier, J., Wiecki, T. V., & Fonnesbeck, C. (2016). Probabilistic programming in Python using PyMC3. *PeerJ Computer Science, 2*, e55.
- Scott, S. L., Blocker, A. W., Bonassi, F. V., Chipman, H. A., George, E. I., & McCulloch, R. E. (2016). Bayes and big data: The consensus Monte Carlo algorithm. *International Journal of Management Science and Engineering Management, 11*(2), 78–88.
- Valim, J. et al. (2023). Nx: Multi-dimensional tensors and numerical definitions for Elixir. *GitHub repository.*
- van de Meent, J.-W., Paige, B., Yang, H., & Wood, F. (2018). An introduction to probabilistic programming. *arXiv preprint arXiv:1809.10756.*
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-normalization, folding, and localization: An improved R-hat for assessing convergence of MCMC. *Bayesian Analysis, 16*(2), 667–718.
- Virding, R., Wikström, C., & Williams, M. (1996). *Concurrent Programming in Erlang* (2nd ed.). Prentice Hall.
- Würthinger, T., Wimmer, C., Wöß, A., Stadler, L., Duboscq, G., Humer, C., Richards, G., Simon, D., & Wolczko, M. (2013). One VM to rule them all. *ACM International Symposium on New Ideas, New Paradigms, and Reflections on Programming and Software (Onward!)*, 187–204.

---

## Appendix B: Feature Development as MCMC

Feature development for a PPL can be analyzed using the same formalism as MCMC sampling. Features are "chains" exploring the "state space" of code files. Merge conflicts are "divergences." The compiler is a "funnel" where all features collide. This metaphor correctly predicted development scheduling outcomes: NCP and WAIC features were predicted to be parallelizable (confirmed: zero merge conflicts), while Vectorized Observations was predicted to require sequential development (confirmed: one conflict in the predicted file).

---

<!-- LAST_SYNC: 2026-02-10 c62cbb77d+uncommitted (D49-D52 + distributed live streaming demo: stream_external/2 API, 4 nodes × 5000 samples = 948 samples/s, emergent Ch4+Ch5 composition; HEAD-TO-HEAD: simple 0.81x, medium 1.90x, stress 1.16x PyMC) -->
