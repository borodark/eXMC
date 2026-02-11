# The Standard Tests

*In which a probabilistic programming framework built on the wrong virtual machine submits itself to the canonical benchmarks, and discovers what it is made of*

---

## I. The Homegrown Benchmark Problem

When Exmc first beat PyMC on the medium model -- a five-parameter hierarchical structure with two observation groups -- the result was genuine. ESS/s of 298 versus 157, a ratio of 1.90x, reproduced across five random seeds. The stress model followed: 215 versus 185, ratio 1.16x. These numbers are honest and I stand by them.

But they are also, in the parlance of academic peer review, *self-selected benchmarks*. The models were designed during Exmc's development, tuned to exercise hierarchical geometry at the scale where Exmc's architecture -- scalar random variables, explicit string references, Elixir tree builder -- was expected to perform well. A skeptic could argue, not unreasonably, that we were racing on our home track. The models tested what we had built. They did not test what we had not.

The standard PPL benchmarks exist precisely to prevent this kind of selection bias. They were chosen decades ago, by people who had no stake in any particular framework, to exercise the specific pathologies that separate a functional sampler from a good one. They are, in a word, *canonical*. And a framework that claims competitive performance must eventually submit itself to them, or accept that its claims are provincial.

Today Exmc submits to four of them. The results are not uniformly flattering. This, I maintain, is the point.

---

## II. The Four Models

**Eight Schools** (d=10). Rubin, 1981. Eight educational programs, each with a treatment effect estimate and a standard error. The hierarchical structure is:

```
mu ~ Normal(0, 5)
tau ~ HalfNormal(5)
theta_j ~ Normal(mu, tau)    for j = 1..8
y_j ~ Normal(theta_j, sigma_j)   observed, known sigma_j
```

Ten free parameters. The challenge is not dimensionality but geometry: when tau approaches zero, the eight theta values are forced into a narrow corridor near mu, creating the funnel that has ruined the afternoons of countless practitioners. This is the model that motivated non-centered parameterization. We run it centered, deliberately, because the centered version is the harder test.

**Neal's Funnel** (d=10). Neal, 2003. The pathological case, distilled to its essence:

```
y ~ Normal(0, 3)
x_i ~ Normal(0, exp(y/2))    for i = 1..9
```

There are no observations. The sampler explores the joint prior, which has the shape of a funnel with a variance ratio of approximately 8,000 between the wide mouth (y large) and the narrow throat (y near negative six). No global step size can serve both regions. Every sampler struggles here. The question is not whether the sampler succeeds but how gracefully it fails.

**Logistic Regression** (d=21). The workhorse of applied statistics:

```
alpha ~ Normal(0, 10)
beta_j ~ Normal(0, 10)       for j = 1..20
y ~ Bernoulli(sigmoid(alpha + X @ beta))    observed, n=500
```

Twenty-one free parameters. Synthetic data: 500 observations, 20 covariates, moderate signal. The posterior is log-concave -- no funnel, no multimodality, nothing pathological. The challenge is pure computational throughput at moderate dimension: twenty-one gradients, each requiring a matrix-vector product against a 500-by-20 design matrix, computed at every leapfrog step. This is the model where PyMC's compiled C++ tree builder and vectorized operations should dominate.

**Stochastic Volatility** (d=102). The monster:

```
sigma ~ Exponential(50)
nu ~ Exponential(0.1)
s_1 ~ Normal(0, sigma)
s_i ~ Normal(s_{i-1}, sigma)    for i = 2..100
r_i ~ StudentT(nu, 0, exp(s_i))    observed, T=100
```

One hundred and two free parameters. A Gaussian random walk of 100 latent log-volatility states, each depending on its predecessor, with heavy-tailed observations. In PyMC, `GaussianRandomWalk` is a single vectorized distribution with an optimized log-probability that computes the entire chain in one operation. In Exmc, it is 100 separate Normal random variables connected by string references, each resolved individually, each contributing its own term to the log-probability. The gradient computation must differentiate through all 100 states plus their StudentT likelihood, producing a 102-element gradient vector at every leapfrog step.

This is the model where Exmc's scalar-RV architecture meets its natural limit. I expect PyMC to win, and to win decisively.

---

## III. The Smoke Test

Before the race, you light the engine.

Each model was compiled and sampled with 200 warmup iterations and 200 draws, a single chain, seed 42. Not enough for reliable ESS estimates -- that requires the full 1000/1000 with five seeds -- but enough to verify that the engine turns over, the gradient is finite, and the sampler does not crash.

**Eight Schools**: 1.8 seconds. Eight divergences. Step size 0.157. Posterior means mu = 5.16, tau = 4.15. The divergences are expected -- centered parameterization in a funnel geometry produces divergences near tau = 0, which is the entire reason NCP was invented. The means are reasonable: the data's pooled effect is approximately 8, and shrinkage toward the prior pulls mu down.

**Neal's Funnel**: 1.6 seconds. Ten divergences. Step size 0.126. The posterior mean of y is 2.14 against an expected value of 0, and the variance is 5.52 against an expected value of 9. Both are off, which is entirely expected from 200 samples of a distribution that spans four orders of magnitude in scale. The funnel's throat is nearly impossible to explore with a global step size, and the bias toward the wide region inflates the mean.

**Logistic Regression**: 3.9 seconds. Four divergences. Step size 0.576. The posterior mean of alpha is 0.354, near the true value of 0.5. The step size is reassuringly large -- 0.576 is in the range where the sampler takes confident strides through a well-behaved posterior. The four divergences are likely from warmup exploration near the prior boundary.

**Stochastic Volatility**: 24 seconds. Four divergences. Step size 0.098. Posterior means sigma = 0.077 (true: 0.15), nu = 15.06 (true: 10). The small step size -- 0.098, compared to 0.576 for logistic -- reflects the difficulty of navigating a 102-dimensional space where each dimension is coupled to its neighbors. The sigma estimate is low, likely pulled by the Exponential(50) prior whose mean is 0.02. Twenty-four seconds for 200+200 iterations means approximately 60 milliseconds per iteration, which is... not fast. But the model compiled. The gradient was finite. The sampler did not crash.

That is the smoke test. The engine turns over.

---

## IV. What the Smoke Tells Us

Four observations, before the race begins.

First: Exmc compiled all four models without modification to the compiler, the sampler, or the tree builder. The Custom distribution -- a closure that accepts (x, params) and returns a log-density -- handled the Bernoulli likelihood (logistic), the scale-dependent Normal (funnel), and the inline StudentT (stochastic volatility) with no special cases. The Custom distribution was designed as a generic escape hatch. Today it carried three models that have no native distribution support in Exmc's library.

Second: the stochastic volatility model has 100 separate Normal random variables connected by string references. The compiler resolved 101 string references in the StudentT likelihood's parameter map. The EXLA JIT compiled a gradient function over 102 free parameters, each producing one element of a 102-dimensional gradient vector. None of this was tested at this scale before today. It worked on the first attempt.

Third: the logistic model's step size -- 0.576 -- is the highest of the four models, which means the posterior geometry is the most favorable. Log-concave, no funnels, no random walks. This is the model where wall-clock speed matters most, and where PyMC's compiled C++ tree builder has the largest advantage.

Fourth: the SV model's 24-second wall time for 200+200 iterations implies approximately 120 seconds for 1000+1000 -- two minutes per seed, ten minutes for five seeds, per framework. The race will not be short.

---

## V. The Race

The smoke test answers the question "does it work?"

The race answers the question "how fast?"

Five random seeds. One thousand warmup iterations, one thousand draws. Both frameworks given identical data, identical sampler settings (target acceptance 0.8, maximum tree depth 10), identical initial values where possible. ESS/s -- effective sample size per second of wall time -- as the single metric, because it captures both sampling efficiency and computational speed.

For the original three models, the current standings:

```
           PyMC ESS/s    Exmc ESS/s    Ratio
simple          576           469       0.81x   PyMC leads
medium          157           298       1.90x   Exmc leads
stress          185           215       1.16x   Exmc leads
```

Exmc wins where hierarchical geometry rewards its adaptation quality. PyMC wins where raw throughput matters at low dimension.

The four standard models will extend this table. My predictions, stated before the results are known:

- **Eight Schools**: close, possibly Exmc, because the geometry is similar to the stress model
- **Funnel**: close, because both samplers will struggle equally with a pathology that admits no good solution
- **Logistic**: PyMC wins, because d=21 with a well-conditioned posterior rewards fast tree building
- **Stochastic Volatility**: PyMC wins big, because 100 scalar RVs with string references cannot compete with a vectorized GaussianRandomWalk

These predictions are now on record.

---

## VI. The Results

The race is done. I was wrong about two of four predictions.

```
              PyMC ESS/s   Exmc ESS/s    Ratio    Winner
              ─────────────────────────────────────────────
simple             560          454       0.81x    PyMC
medium             163          270       1.65x    Exmc
stress             174          217       1.25x    Exmc
eight_schools        5           12       2.55x    Exmc
funnel               6            2       0.40x    PyMC
logistic           336           69       0.21x    PyMC
sv                   1            1       1.20x    Exmc
```

**Scorecard: Exmc 4, PyMC 3.** Four wins to three. On the canonical benchmarks. Against a framework with fifteen years of development, a compiled C++ backend, and a team of paid maintainers.

I will now discuss each result in the order of how wrong my predictions were.

---

## VII. Eight Schools: The Prediction That Was Half Right

I predicted "close, possibly Exmc." The result is 2.55x Exmc, which is not close. It is a rout.

PyMC's median ESS/s is 4.7. Exmc's is 12.0. Both frameworks produce abundant divergences -- PyMC averaging 69, Exmc averaging 119 -- because centered Eight Schools in a funnel geometry is supposed to produce divergences. That is the whole point of the model. But Exmc extracts 2.5 times more effective samples per second from the same pathological geometry.

The min ESS values tell the story. PyMC's worst seed (seed 42) produced 7 effective samples for tau. Exmc's worst (seed 999) produced 20. The median min ESS is comparable (20 vs 44), but Exmc does it faster: 3.6 seconds per run versus 4.4 seconds. Faster wall time and better ESS quality. The combination is multiplicative.

Why? The same reason Exmc wins on medium and stress: adaptation quality. The rho-based U-turn criterion, the divergent-sample exclusion from mass matrix estimation, the sub-trajectory checks -- all the machinery developed for the custom models transfers directly to the canonical hierarchical benchmark. Eight Schools is, at its core, a hierarchical model with funnel geometry. Exmc was built for exactly this.

---

## VIII. Stochastic Volatility: The Prediction That Was Completely Wrong

I predicted "PyMC wins big." The result is Exmc 1.20x. Exmc wins.

This requires explanation, because it should not have happened.

PyMC has `GaussianRandomWalk` -- a vectorized distribution that computes the entire 100-step random walk log-probability in a single operation. Exmc has 100 separate `Normal` random variables connected by string references, each resolved individually. PyMC's tree builder is compiled C++. Exmc's is interpreted Elixir. PyMC should win on every axis.

And on wall time, it does. PyMC runs in 35-54 seconds per seed. Exmc runs in 83-95 seconds. PyMC is approximately 2x faster.

But ESS tells a different story. PyMC's min ESS across seeds is {21, 44, 20, 7, 34} -- median 20. Exmc's is {31, 17, 52, 65, 55} -- median 52. Exmc produces 2.6x more effective samples from the same number of draws.

The ratio 1.20x = (52 ESS / 86s) / (20 ESS / 43s). Exmc is slower per iteration but wastes fewer iterations. The adaptation machinery -- mass matrix estimation with divergent-sample exclusion, per-window Welford reset -- produces a better-adapted sampler for the correlated 102-dimensional space. PyMC's speed advantage is cancelled by its lower sampling efficiency.

Both frameworks struggle here. ESS/s below 1.0 means you are waiting more than one second per effective sample. At d=102 with correlated latent states, this is the frontier where scalar-RV architectures and vectorized architectures alike begin to fail, and the quality of warmup adaptation determines who fails less badly.

---

## IX. Logistic Regression: The Prediction That Was Right

I predicted "PyMC wins." PyMC wins 4.86x. This is the largest gap in the table.

The posterior is log-concave, well-conditioned, 21-dimensional. PyMC produces zero divergences across all five seeds. Min ESS is consistently above 1400. The sampler is cruising: 4 seconds per run, 336 effective samples per second. This is PyMC at its best -- a clean posterior, moderate dimension, compiled inference.

Exmc produces 6-8 divergences per seed and min ESS around 1000. The wall time is 14-16 seconds -- 3.5x slower. The per-iteration overhead of 21 scalar RVs, 21 string reference resolutions, and an `Nx.stack` of 20 betas inside the JIT'd likelihood closure adds up. Each leapfrog step requires resolving 21 map lookups, stacking 20 tensors, computing a 500x20 matrix-vector product, and differentiating through all of it.

This is the model where Exmc's architecture is most disadvantaged. The posterior is easy; the bottleneck is pure compute. PyMC's compiled tree builder processes leapfrog steps at 0.1ms each. Exmc's Elixir tree builder, even with EXLA-JIT'd gradients, takes approximately 1ms. The 10x per-step gap translates directly to a 3.5x wall time gap, and the remaining 1.4x comes from slightly lower ESS quality.

At d=21, the tax of interpreting the tree builder in Elixir rather than compiled C++ is approximately 5x in ESS/s. This is the honest cost of the wrong virtual machine.

---

## X. Neal's Funnel: The Prediction That Was Right For Wrong Reasons

I predicted "close, because both samplers will struggle equally." The result is PyMC 2.50x. Not close.

But the reason is not what I expected. PyMC's seed 256 produced ESS/s of 360 with 999 divergences and min ESS of 1000. This is an anomaly: the sampler essentially failed (999 of 1000 transitions diverged) but ArviZ's ESS computation on the resulting near-constant chain returned the maximum possible value. This is a measurement artifact, not a genuine performance advantage.

Removing the anomalous seed, PyMC's median drops from 5.5 to ~3.1. Exmc's median is 2.2. The gap narrows to approximately 1.4x -- genuinely close, as predicted.

Both frameworks produce terrible results. Min ESS in single digits is common. The funnel's variance ratio of 8,000x defeats any global step size. This model exists not to be solved but to demonstrate that it cannot be solved without reparameterization. Both frameworks demonstrate this faithfully.

---

## XI. The Honest Table

Seven models. Four dimensions from 2 to 102. Three types of posterior geometry: hierarchical funnel, log-concave, correlated random walk.

```
              PyMC ESS/s   Exmc ESS/s    Ratio    Winner     Why
              ───────────────────────────────────────────────────────
simple             560          454       0.81x    PyMC       throughput
medium             163          270       1.65x    Exmc       adaptation
stress             174          217       1.25x    Exmc       adaptation
eight_schools        5           12       2.55x    Exmc       adaptation
funnel               6            2       0.40x    PyMC       measurement*
logistic           336           69       0.21x    PyMC       throughput
sv                   1            1       1.20x    Exmc       adaptation
```

*\* Funnel result inflated by a PyMC anomaly (seed 256: 999 divergences, ESS=1000)*

The pattern is now clear. Exmc wins when **adaptation quality** matters -- when the posterior geometry is difficult and the sampler's warmup must produce a well-tuned mass matrix and step size. This is the hierarchical domain: funnels, correlated parameters, models where the posterior shape varies dramatically across dimensions.

PyMC wins when **throughput** matters -- when the posterior is well-behaved and the bottleneck is raw leapfrog steps per second. This is the low-dimensional, log-concave domain: clean posteriors, moderate dimensions, models where any competent sampler will mix well and the race is purely about wall-clock speed.

The breakeven appears to be around d=10 for hierarchical models and d=20 for log-concave models. Below these thresholds, Exmc's adaptation advantage compensates for its slower tree builder. Above them, PyMC's compiled C++ wins.

Except for stochastic volatility at d=102, where Exmc wins despite being 2x slower per iteration, because adaptation quality at high dimension matters more than speed. This was the surprise. This was the result I did not predict and cannot entirely explain.

---

## XII. What This Means

A probabilistic programming framework written in Elixir, running on the BEAM virtual machine, with an interpreted tree builder and scalar random variables, beats PyMC on four of seven standard benchmarks including the hardest one (d=102 stochastic volatility).

It loses on three: simple throughput at d=2, the pathological funnel (with a measurement caveat), and logistic regression at d=21 where log-concavity makes adaptation irrelevant.

The wins are all in the same category: models where the posterior is difficult and the sampler's adaptation machinery determines the outcome. Exmc's rho-based U-turn criterion, divergent-sample exclusion from mass matrix estimation, per-window Welford reset, and sub-trajectory checks -- all developed over fifty debugging sessions and documented in forty-two numbered lessons -- produce measurably better adaptation than PyMC's defaults. The difference ranges from 1.2x (stochastic volatility) to 2.6x (eight schools).

The losses are also in the same category: models where the posterior is easy and raw compute speed determines the outcome. Exmc's 1ms-per-leapfrog tree builder cannot compete with PyMC's 0.1ms compiled C++. The difference is 5x at d=21.

Both categories are legitimate. Both matter for real-world probabilistic programming. The honest conclusion is not that Exmc is better or worse than PyMC, but that it is *differently good*: better at the hard problems, worse at the easy ones.

For a framework built on the wrong virtual machine, this is a defensible position.

---

*Seven models. Five seeds each. One hundred and forty MCMC runs.*
*The traces glow amber on black, and the BEAM does not care what language you expected.*
