# Feature Parity and Speed: Exmc vs PyMC

*This is Part 2 of "What If Probabilistic Programming Were Different?" Part 1 introduced the idea that PPL architecture is downstream of runtime choice. This part puts numbers on it.*

---

The obvious objection to building a probabilistic programming framework in Elixir is performance. Python's scientific stack — NumPy, SciPy, PyTensor, JAX — is backed by decades of optimized C/Fortran/CUDA kernels. Elixir's numerical ecosystem (Nx, EXLA) is younger by an order of magnitude. So does Exmc actually work as a sampler? Can it keep up?

The short answer: yes, and on the harder models it's faster. The long answer involves understanding *where* the speed comes from and *where* it doesn't.

## What Exmc Implements

First, feature parity. Exmc isn't a toy — it's a complete probabilistic programming framework.

### Inference Algorithms

| Algorithm | Exmc | PyMC | Notes |
|-----------|:----:|:----:|-------|
| NUTS | Yes | Yes | Same algorithm, different runtimes |
| ADVI | Yes | Yes | Mean-field variational inference |
| SMC | Yes | Yes | Sequential Monte Carlo with tempering |
| Pathfinder | Yes | Yes | L-BFGS path-based variational approx |

Both frameworks offer the same four inference methods. The implementations differ in their compilation targets — PyMC compiles through PyTensor to C/Fortran, Exmc compiles through Nx to XLA — but the mathematical algorithms are identical.

### Distributions

| | Exmc | PyMC |
|---|:---:|:---:|
| Count | 12 + Custom | 30+ |
| Univariate continuous | Normal, HalfNormal, Exponential, Gamma, Beta, Uniform, StudentT, Cauchy, LogNormal, Laplace | All of the above + many more |
| Special | Censored, Mixture | Censored, Mixture, Truncated, many more |
| Multivariate | — | MvNormal, Wishart, Dirichlet, etc. |

PyMC has a larger distribution library, particularly for multivariate distributions. Exmc covers the distributions needed for common hierarchical models. The `Dist.Custom` module allows users to define new distributions with a log-probability function.

### Diagnostics and Model Comparison

| Capability | Exmc | PyMC |
|-----------|:----:|:----:|
| ESS (rank-normalized) | Yes | Yes |
| Split R-hat | Yes | Yes |
| MCMC standard error | Yes | Yes |
| Summary statistics | Yes | Yes |
| WAIC | Yes | Yes |
| LOO-CV (PSIS) | Yes | Yes |
| Prior predictive | Yes | Yes |
| Posterior predictive | Yes | Yes |

Full parity on diagnostics and model comparison. Both follow Vehtari et al. (2021) for rank-normalized ESS and R-hat.

### What Exmc Has That PyMC Doesn't

| Feature | Exmc | PyMC |
|---------|:----:|:----:|
| Real-time sample streaming | `sample_stream/4` | — |
| Live visualization during sampling | ExmcViz (Scenic) | — |
| Browser integration | Phoenix LiveView | — |
| Multi-node distributed MCMC | `:erpc` + fault recovery | — |
| Mid-tree crash recovery | `try/rescue` (zero overhead) | — |
| Distributed live streaming | 4+ nodes → single dashboard | — |

These aren't incremental features. They're architectural capabilities that emerge from the BEAM runtime. Part 1 of this series explains why.

### What PyMC Has That Exmc Doesn't

- Multivariate distributions (MvNormal, Wishart, LKJ, Dirichlet)
- Gaussian process modules
- ODE integration
- Timeseries models (AR, GARCH)
- Imputation of missing data
- `pm.Data` for mutable shared variables
- ArviZ ecosystem (rich post-hoc visualization)
- A decade of community, documentation, and tutorials

PyMC is a mature ecosystem. Exmc is a research framework proving an architectural thesis. The distribution and model coverage gap is real — but it's a library gap, not an architectural one. Adding distributions to Exmc is mechanical work (implement `log_prob`, `sample`, and optionally a constraint transform). The architectural capabilities go the other direction.

## The Speed Question

Now the numbers. All benchmarks use the same models, same data, same seeds, same machine. One chain, 1000 warmup + 1000 samples, `target_accept=0.8`. Five seeds per model, median reported. CPU only.

### Three Benchmark Models

| Model | Free Parameters | Structure | Challenge |
|-------|:-:|---|---|
| **Simple** | 2 | Normal mean + Exponential variance | Baseline: fast, well-conditioned |
| **Medium** | 5 | Hierarchical 2-group: global mean/variance, per-group intercepts, shared noise | Adaptation: hierarchical correlation structure |
| **Stress** | 8 | Hierarchical 3-group: population mean/variance, 3 group effects, 3 noise scales | Scale: 200x range in inverse mass matrix |

### Head-to-Head Results

| Model | Exmc ESS/s | PyMC ESS/s | Ratio |
|-------|--------:|--------:|:-----:|
| Simple (d=2) | 469 | 576 | **0.81x** |
| Medium (d=5) | 298 | 157 | **1.90x** |
| Stress (d=8) | 215 | 185 | **1.16x** |

Exmc beats PyMC on the two harder models. The simple model gap (0.81x) is the cost of an interpreted host language — Elixir's BEAM VM adds per-operation overhead that dominates when the model is trivial. For models where adaptation quality and algorithmic decisions matter more than raw per-step speed, Exmc wins.

### With Distribution: 4-Node Scaling

| Configuration | Simple | Medium | Stress | Scaling |
|---|---:|---:|---:|:---:|
| 1 chain (local) | 469 | 298 | 215 | 1.0x |
| 4 chains (4 `:peer` nodes) | 1665 | 812 | 628 | **3.4-3.7x** |

Near-linear scaling from 4 nodes with zero infrastructure. Each `:peer` node is a separate OS process with its own BEAM scheduler pool. The entire distributed system is 203 lines of Elixir.

## Where the Speed Comes From (and Doesn't)

The headline numbers hide a more interesting story. Exmc isn't fast because Elixir is fast — it's fast *despite* Elixir being slow at arithmetic. The speed comes from three sources.

### 1. The JIT Boundary

Every NUTS step involves two kinds of work: *gradient computation* (tensor math, heavy) and *tree logic* (scalar decisions, light). In Exmc, gradient computation is JIT-compiled through EXLA/XLA — the same compiler backend as JAX. Tree logic runs in interpreted Elixir.

The critical design decision: where to place the boundary between JIT and interpreted code.

| Operation | EXLA.Backend | BinaryBackend | Ratio |
|-----------|:-----------:|:------------:|:-----:|
| `Nx.to_number(Nx.sum(...))` | 150 us | 5 us | 30x |
| `Nx.add` | 69 us | 7 us | 9x |

The initial implementation let JIT tensors leak into the tree builder, paying EXLA dispatch overhead for trivial scalar operations. Copying tensors to BinaryBackend at the JIT boundary gave a **3x speedup** (simple model: 70 → 211 ESS/s). No algorithmic change — just placing the compilation boundary correctly.

This is a PPL design insight, not an Elixir trick. Any PPL with a JIT-compiled gradient and an interpreted host faces the same boundary problem. PyMC solves it by writing the tree builder in C++. Exmc solves it by copying tensors at the boundary. The lesson: JIT boundary placement is a first-class PPL design concern.

### 2. Algorithmic Correctness

Two bugs in the multinomial proposal mechanism inflated the duplicate sample rate from 7.8% (PyMC's rate) to 37.7%. Both bugs produced valid posterior samples — they passed every correctness test — but degraded sampling efficiency by 2-3x.

**Bug 1: Capped log-weights.** The trajectory point weight was capped at `exp(0) = 1`, so points with better energy than the starting position were systematically underweighted. The Metropolis-Hastings acceptance cap (`min(1, exp(d))`) for step size adaptation is correct — but the multinomial selection weight must be uncapped.

**Bug 2: Wrong merge formula.** Stan and PyMC use *biased progressive sampling* for the outer tree merge (`P = min(1, w_subtree/w_trajectory)`) but *balanced multinomial* for inner merges. Exmc used balanced for both. The balanced outer merge made the starting point "sticky" — it survived merges more often than it should.

Fixing both: duplicate rate 37.7% → 6.5%. Medium model ESS/s: 116 → 298 (+157%).

Neither bug is described in any published paper. The inner/outer merge distinction appears only in Stan's source code and Betancourt's appendix. This is *implementation folklore* — the kind of knowledge that separates a working sampler from an efficient one.

### 3. Matching Undocumented Stan Practices

Three Stan implementation practices are in source code only, not in published papers:

| Practice | Source | Impact on Exmc |
|----------|--------|:---:|
| Exclude divergent samples from mass matrix estimation | `stan-dev/stan, nuts.hpp` | +109% medium, +180% stress |
| `term_buffer = 50` (not 200) for Phase III | Stan source | +43% Phase II samples |
| 3-point sub-trajectory U-turn checks per merge | Stan source | +46% medium, -41% warmup divergences |

Combined effect on the stress model: 67 → 215 ESS/s. From 0.41x PyMC to 1.16x PyMC. Every improvement came from reading C++ source code, not papers.

## The Gap Decomposition

The remaining simple model gap (0.81x) decomposes cleanly:

**Per-leapfrog overhead:** ~300 us (Exmc) vs ~15 us (PyMC). This is the interpreted-host ceiling — Elixir map creation, Nx scalar operations, EXLA dispatch latency. It's irreducible without rewriting the tree builder in a compiled language.

**ESS per sample:** At parity for well-identified models. The per-step overhead is overcome by algorithmic quality on harder models where adaptation matters more.

**The implication:** For trivial models (d < 3), PyMC will always be faster — the C++ tree builder dominates. For hierarchical models (d > 4), adaptation quality dominates and Exmc's algorithmic improvements overcome the per-step overhead. The crossover is around d = 3-4 parameters.

## GPU Acceleration

Both frameworks support GPU via XLA. For Exmc, GPU gives 2.65x wall-time speedup at d=22 (8-parameter hierarchical model) despite no per-step speedup at that dimensionality — the benefit comes from batched leapfrog amortizing kernel launch overhead across all steps in a subtree.

| Device | Wall Time | ESS/s | Step Size | Divergences |
|--------|--------:|------:|:---------:|:-----------:|
| CPU | 12.4s | 3.9 | 0.330 | 4 |
| GPU | 4.7s | 10.3 | 0.330 | 4 |

Sampling quality is identical — same step sizes, same divergence count, same acceptance probability. Only wall time differs. GPU becomes profitable at d > 100 on a per-step basis, but batched execution makes it useful much earlier.

## What This Means for PPL Users

If you need the PyMC ecosystem — its 30+ distributions, Gaussian processes, ArviZ integration, and community — use PyMC. It's a mature, excellent tool.

If you need any of these, no existing PPL can help:
- Watch your posterior converge in real time from a browser
- Run chains across machines with automatic fault recovery
- Recover from mid-sampling numerical failures without restarting
- Stream results from distributed nodes to a live dashboard

And if your model has hierarchical structure with 5+ parameters, Exmc may be faster out of the box.

The architectural lesson is broader than any single framework: the properties that modern inference workflows need — streaming, fault tolerance, distribution, composition — are not features to be bolted onto batch-oriented runtimes. They're consequences of choosing the right runtime in the first place.

---

*Part 3 will cover the optimization journey: from 34x slower than PyMC to 1.9x faster, and what each step reveals about the JIT boundary problem in mixed-runtime systems.*

*Exmc is open source. The thesis — "Probabilistic Programming on Actor-Model Runtimes" — covers the full technical story.*
