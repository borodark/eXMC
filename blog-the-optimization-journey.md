# 34x Slower to 1.9x Faster: The Optimization Journey

*This is Part 3 of "What If Probabilistic Programming Were Different?" Part 1 introduced the actor-model thesis. Part 2 showed feature parity and benchmark numbers. This part tells the story of how we got there — and what each step reveals about building a PPL on an interpreted runtime.*

---

Exmc's first benchmark against PyMC was sobering. The medium hierarchical model (5 parameters, 2 groups): **3.3 ESS/s vs 113 ESS/s. A 34x gap.**

Today, that same model runs at **298 ESS/s** — 1.90x faster than PyMC. The journey from 34x slower to 1.9x faster took seven distinct optimization phases, each of which taught something about the nature of mixed-runtime PPL design. None required changing the NUTS algorithm. All of them were about understanding where computational work actually happens and why.

## Phase 0: The Starting Point

The initial Exmc sampler was naive. It ran NUTS correctly — the posterior was right, the diagnostics were fine — but every operation went through Nx tensor operations, and every tensor lived on whatever backend EXLA's JIT returned.

```
Medium model: 3.3 ESS/s
PyMC:         113 ESS/s
Gap:          34x
```

The profiler showed the problem immediately: trivial scalar operations (adding two numbers, checking a sign) were taking 70-150 microseconds each because they dispatched through the EXLA runtime. There were hundreds of these per NUTS tree expansion.

## Phase 1: The JIT Boundary Discovery (34x → 23x)

The NUTS tree builder does two kinds of work. **Gradient computation** is the expensive part — matrix multiplies, automatic differentiation, the whole tensor graph. This should be JIT-compiled. **Tree logic** is the cheap part — "did the trajectory turn around?", "which proposal do we accept?", "should we extend left or right?" This is scalar arithmetic and branching.

The problem was that these two kinds of work shared the same tensor backend. EXLA's JIT-compiled `step_fn` (the leapfrog integrator) returned tensors on `EXLA.Backend`. These tensors then flowed into the tree builder, where every `Nx.add` or `Nx.to_number` dispatched through the EXLA runtime — even for operations on single numbers.

| Operation | EXLA.Backend | BinaryBackend | Ratio |
|-----------|:-----------:|:------------:|:-----:|
| `Nx.to_number(Nx.sum(...))` | 150 us | 5 us | 30x |
| `Nx.add` | 69 us | 7 us | 9x |

The fix was four lines of code: `Nx.backend_copy(tensor, Nx.BinaryBackend)` on step_fn outputs at the tree leaf level.

```
Simple: 70 → 211 ESS/s (3.0x)
Medium: 2.1 → 3.3 ESS/s (1.6x)
```

Four more targeted optimizations followed: fusing `joint_logp` into the JIT step function, replacing Nx-based U-turn checks with pure Erlang list arithmetic (13x faster per check), explicit BinaryBackend hints on all tensor creation, and caching the inverse mass matrix as an Erlang list.

```
Simple: 211 → 242 ESS/s (+15%)
Medium: 3.3 → 4.9 ESS/s (+48%)
Gap:    23x
```

**The lesson:** In a mixed-runtime PPL, the placement of the JIT boundary determines performance more than any algorithmic choice. Tensors that cross the boundary in the wrong direction carry their backend with them, turning microsecond operations into hundred-microsecond operations. This problem doesn't exist in single-runtime PPLs (Stan is all C++, NumPyro is all JAX).

## Phase 2: Batched Leapfrog (23x → 19x)

Instead of one JIT call per leapfrog step, batch an entire subtree's trajectory into a single XLA `while` loop. The dispatch savings are dramatic:

| Steps | Individual dispatch | Batched | Speedup |
|:-----:|:------------------:|:------:|:-------:|
| 4 | 1040 us | 400 us | 2.6x |
| 16 | 4160 us | 468 us | 8.9x |
| 64 | 16640 us | 408 us | 40.8x |
| 128 | 33280 us | 498 us | 66.9x |

But integration was treacherous. The first approach — pre-compute all leapfrog states, then build a merge tree bottom-up — looked correct. Per-call verification: 0/50 mismatches. But chained over 1000 sampling iterations, ESS collapsed from 78 to 12. A 6.5x quality regression from a "correct" optimization.

The root cause: a bottom-up merge tree and the top-down recursive `build_subtree` consume random numbers in a different order. Over hundreds of iterations, slightly different multinomial sampling at borderline cases compounds into dramatically different trajectories.

The fix: pre-compute the leapfrog states, but inject them into the *exact same* recursive `build_subtree` via a cached step function and an Erlang `:atomics` counter. Same code path. Same RNG consumption. Bit-identical trees. The optimization becomes invisible to the algorithm.

```
Medium: 4.9 → 4.6 ESS/s (noise; shallow simple trees don't benefit)
Gap:    ~23x → ~19x (medium with adaptation fixes below)
```

**The lesson:** When parallelizing a sequential recursive algorithm, the original recursive structure must be preserved exactly. Approximating it with a "semantically equivalent" reconstruction is insufficient. Batch the computation, but reuse the original control flow.

## Phase 3: The Parameterization Discovery (19x → 2.3x)

This was the biggest single improvement, and it wasn't a speed optimization at all.

Exmc had automatic non-centered parameterization (NCP): for hierarchical models, it rewrites `alpha ~ Normal(mu, sigma)` as `alpha_raw ~ Normal(0, 1)` and reconstructs `alpha = mu + sigma * alpha_raw`. NCP is a standard technique for poorly-identified hierarchical models where the data is weak relative to the prior.

But our benchmark model was *well-identified* — 40 observations per group, informative data. NCP made `alpha_raw` perfectly independent (ESS near 1000) but created complex coupling between `sigma_global` and `sigma_obs` that a diagonal mass matrix couldn't capture. The step size collapsed to 0.034 (vs 0.46 for centered), and `sigma_obs` ESS dropped to 4-158.

| Parameterization | ESS/s | Step size | sigma_obs ESS |
|:----------------:|------:|:---------:|:------------:|
| NCP (automatic) | 5.5 | 0.034 | 4–158 |
| Centered (`ncp: false`) | 49.1 | 0.46 | 141–893 |

A 9x improvement from a flag. The entire "algorithmic quality gap" we'd been chasing was a parameterization problem.

```
Medium: 4.9 → 49.1 ESS/s (9x)
Gap:    2.3x
```

**The lesson:** Before optimizing speed, check that you're solving the right problem. A 9x ESS improvement from parameterization dwarfs months of micro-optimization work (which yielded ~1.5x total). The remaining gap was now entirely architectural — wall-time overhead per step, not sampling quality.

## Phase 4: Adaptation Quality — Reading Stan's Source Code (2.3x → 1.5x)

With centered parameterization achieving ESS-per-sample parity, the remaining gap was wall-time: Exmc took longer per step. But some of that "wall-time gap" was actually adaptation quality in disguise — the mass matrix wasn't as good as Stan's, forcing smaller step sizes and deeper trees.

Three Stan practices, found only in source code, never in papers:

**Practice 1: Per-window Welford reset.** Stan resets the Welford mass matrix estimator at each adaptation window boundary. Exmc accumulated across windows, letting early samples (drawn under a bad mass matrix) contaminate later estimates. Impact: +30% medium ESS/s.

**Practice 2: Divergent sample exclusion.** Stan excludes divergent transitions from the mass matrix estimator. Divergent trees are truncated; their selected positions are biased toward the starting point. Including them inflates variance estimates. Impact: +109% medium, +180% stress (combined with Practice 3).

**Practice 3: `term_buffer = 50`, not 200.** Stan allocates only 50 iterations to Phase III (final step size adaptation), not 200. After fixing the dual averaging initialization (which had been biasing toward epsilon = 1.0), 50 iterations is sufficient. The extra 150 iterations that Exmc wasted on Phase III could have been Phase II mass matrix adaptation — 150 more samples for the final mass matrix window (+43%).

Combined:

```
Simple: 211 → 233 ESS/s (+10%)
Medium: 115 → 116 ESS/s (+1%)
Stress:  67 → 107 ESS/s (+60%)
Gap:     ~1.5x (medium at parity, stress narrowing)
```

**The lesson:** PPL performance depends on *implementation folklore*, not just algorithmic specification. Three practices, invisible in published papers, collectively closed a 60% gap on the stress model. A PPL implementor reading only Hoffman & Gelman (2014) and Betancourt (2017) would miss all three.

## Phase 5: The Rust NIF Experiment (Mixed Results)

With the algorithmic gap closed, the remaining overhead was architectural: ~300 us per leapfrog step in Elixir vs ~15 us in PyMC's C++. Could Rust close this?

Three granularity levels were attempted:

| Strategy | Scope | Result |
|----------|-------|:------:|
| Replace entire outer loop | All tree logic in Rust | **0.86x (slower)** |
| Replace inner subtree only | Merge logic in Rust | **1.5x** |
| Replace full tree build | Everything after JIT in Rust | **1.67x simple, 0.47x medium** |

The outer-loop NIF *failed* because each iteration crossed *both* Elixir↔XLA (for gradient) and Elixir↔Rust (for tree logic). The double boundary crossing added more overhead than the Rust tree logic saved.

The inner-subtree NIF *worked* because it replaced a self-contained block of Elixir computation (leaf slicing + recursive merges) with a single Rust call. One boundary, no back-and-forth.

The full-tree NIF *partially worked*: it pre-computes all leapfrog states, then builds the entire tree in one Rust call. Excellent for shallow trees (simple model: 1.67x), but deep trees waste computation — you have to pre-compute the maximum possible trajectory before you know how deep the tree will go.

**The lesson:** FFI boundary *granularity* matters more than FFI language choice. The optimal scope is the largest block that doesn't require crossing back to another language. For NUTS, that's the inner subtree (pure host computation), not the outer loop (which needs JIT gradient calls).

## Phase 6: The U-Turn Criterion (Stress Model Breakthrough)

The stress model (8 parameters, 3 groups, 200x range in inverse mass matrix) stubbornly lagged at 0.41x PyMC. The U-turn criterion was the culprit.

The original criterion checks `(q⁺ - q⁻) · (M⁻¹ p)` — whether endpoint displacement dotted with momentum has become negative. But `M⁻¹` appears as a per-dimension weight. When one component has 200x larger inverse mass (mu_pop in our stress model), the dot product is dominated by that single component. When mu_pop's trajectory reverses, the U-turn fires even if the other 7 parameters are still productively exploring.

The fix: Betancourt's (2017) generalized criterion, which tracks cumulative momentum `ρ = Σ pᵢ` across the trajectory and checks `ρ · (M⁻¹ p±)`. This removes the endpoint-displacement bias.

```
Stress: 67 → 89 ESS/s (+33%)
Avg tree depth: 2.2 → 2.9 (deeper = better exploration)
Step size: 0.46 → 0.60 (larger = more efficient)
Gap:    0.41x → 0.57x PyMC
```

A sub-trajectory U-turn check (another undocumented Stan practice — 3 checks per merge instead of 1) gave another +46% on medium.

**The lesson:** U-turn criterion choice is invisible on simple models but dominant for hierarchical models with heterogeneous scales. It's the kind of thing you only discover by profiling per-parameter ESS on a sufficiently challenging model.

## Phase 7: The Multinomial Bugs (The Breakthrough)

Two bugs in the multinomial proposal mechanism had been hiding in plain sight since the beginning. Both produced valid posteriors. Both passed every correctness test. Both were invisible in diagnostics. Together, they inflated the duplicate sample rate from 7.8% (PyMC) to 37.7%.

**Bug 1: Capped log-weights.** At each tree leaf, the trajectory point weight was capped at `exp(0) = 1`. Points with better energy than the starting position were systematically underweighted. The acceptance probability cap for step size adaptation (`min(1, exp(d))`) is correct — but the multinomial selection weight must be uncapped.

**Bug 2: Wrong outer merge formula.** Stan and PyMC use biased progressive sampling for outer merges (`P(accept subtree) = min(1, w_subtree/w_trajectory)`) but balanced multinomial for inner merges. Exmc used balanced for both. This made the starting point "sticky" — at tree depth 2, q_0 was selected 25% of the time instead of PyMC's 3.7%.

Neither bug is described in any paper. The inner/outer merge distinction appears only in Betancourt's appendix and Stan's source code.

Fixing both:

```
Duplicate rate: 37.7% → 6.5%

Simple: 233 → 469 ESS/s (+101%)    0.81x PyMC
Medium: 116 → 298 ESS/s (+157%)    1.90x PyMC  ← beats PyMC
Stress: 107 → 215 ESS/s (+101%)    1.16x PyMC  ← beats PyMC
```

**The lesson:** MCMC correctness is robust — wrong multinomial weights still produce valid posteriors because acceptance probability is bounded. But *efficiency* is fragile. A 37.7% duplicate rate means over a third of your compute produces no new information. These bugs are the kind that survive unit tests, integration tests, and posterior accuracy checks. Only a diagnostic that explicitly measures proposal diversity (duplicate rate, index-in-trajectory distribution) can catch them.

## The Full Trajectory

| Phase | Change | Medium ESS/s | Gap vs PyMC |
|:-----:|--------|:-----------:|:-----------:|
| 0 | Baseline | 3.3 | 34x |
| 1 | JIT boundary fix + micro-opts | 4.9 | 23x |
| 2 | Batched leapfrog | 4.6 | 25x |
| 3 | Centered parameterization | 49.1 | 2.3x |
| 4 | Stan adaptation practices | 116 | ~1.0x |
| 5 | Rust NIF (inner subtree) | ~120 | ~0.9x |
| 6 | Rho U-turn + sub-trajectory | ~120 | ~1.0x |
| 7 | Multinomial fixes | **298** | **1.90x faster** |

90x improvement from Phase 0 to Phase 7. No change to the core NUTS algorithm. Every improvement was about understanding *where* the work happens and *why*.

## The Decomposition

The journey reveals that the gap between an interpreted-host PPL and a compiled-host PPL decomposes into three independent factors:

**1. Architectural overhead (per-step):** ~300 us (Exmc) vs ~15 us (PyMC). This is the irreducible cost of BEAM VM map allocation, function dispatch, and EXLA call overhead. It's a constant multiplier on wall-time that favors compiled PPLs for trivial models (d < 3).

**2. Algorithmic quality (ESS per sample):** At parity after Phase 3 (parameterization) and Phase 4 (adaptation). This is framework-independent — the same algorithm, tuned correctly, produces the same posterior quality regardless of host language.

**3. Implementation correctness (multinomial):** The hidden factor. Two bugs that reduced ESS by 2-3x without affecting posterior validity. Fixing them didn't make the sampler faster per step — it made each step produce more useful information.

The crossover is around d = 3-4 parameters. Below that, architectural overhead dominates and PyMC wins. Above that, algorithmic quality dominates and Exmc wins — because the per-step overhead is amortized over deeper trees that produce proportionally more effective samples.

## What This Teaches About PPL Design

**1. JIT boundary placement is a first-class design concern.** This problem doesn't exist if your entire PPL is in one language (Stan, NumPyro). It's the central performance challenge of mixed-runtime PPLs.

**2. Implementation folklore is load-bearing.** Five undocumented practices (3 Stan adaptation details + 2 multinomial properties), none in any paper, collectively account for a 10x performance difference. Publishing algorithms without implementation details is publishing half the story.

**3. Correctness and efficiency are orthogonal.** MCMC is uniquely forgiving — wrong weights, wrong merge formulas, wrong U-turn criteria all produce valid posteriors. This makes efficiency bugs invisible to standard testing. PPL development needs explicit efficiency diagnostics: duplicate rate, per-parameter ESS, index-in-trajectory distributions.

**4. The biggest wins aren't speed optimizations.** Parameterization (9x), multinomial correctness (2-3x), and adaptation quality (3x) all improve how much *information* each sample contains. Per-step speed (JIT boundary, NIF) improves how fast you *generate* samples. The information-per-sample improvements were larger, cheaper, and more generalizable.

**5. Three languages is one too many — unless you manage the boundaries.** Exmc's Elixir/Rust/XLA stack requires managing two FFI boundaries. The inner-subtree NIF works because it touches only one boundary. The outer-loop NIF failed because it touched both. The optimal architecture: JIT the gradient (XLA), interpret the tree (Elixir), and optionally accelerate the inner subtree (Rust) — but never let a single operation cross two boundaries.

---

The gap between 34x slower and 1.9x faster wasn't bridged by making Elixir fast. It was bridged by understanding what "fast" means in a mixed-runtime system: put computation on the right side of the boundary, use the right parameterization, match the undocumented practices of mature implementations, and make sure your multinomial sampling actually works.

The BEAM gave us streaming, fault tolerance, and distribution for free. The price was figuring out where the JIT boundary should be. That price has been paid.

---

*The thesis — "Probabilistic Programming on Actor-Model Runtimes" — contains the full technical analysis with reproducible benchmarks, cost models, and 52 documented architectural decisions. Parts 1 and 2 of this series cover the architectural thesis and feature parity.*
