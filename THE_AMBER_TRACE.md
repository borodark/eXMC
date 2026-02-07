# The Amber Trace

*On the improbable and slightly reckless decision to build a probabilistic programming framework on a virtual machine designed for telephone switches*

---

## I. Against the GIL

It is a peculiar feature of the Python programming language that it cannot do two things at once. I do not mean this metaphorically. I mean it in the most literal and damning sense: the Global Interpreter Lock, that celebrated bottleneck at the heart of CPython, ensures that no matter how many cores your machine possesses, no matter how embarrassingly parallel your workload, the interpreter will execute one thread at a time, politely queuing the rest like supplicants at a Soviet bread counter.

This would be merely an annoyance if Python had not become, through the usual combination of historical accident and network effects, the lingua franca of scientific computing. PyMC -- the finest probabilistic programming framework extant -- must run its Markov chains in this environment. Four chains, the standard for convergence diagnostics, means four operating system processes. Four copies of the model in memory. Four separate JIT compilations. Four interpreters, each paying the full cost of initialization, each sealed behind its own GIL, communicating through serialized pickle streams as though exchanging diplomatic cables between hostile nations.

One does not have to be a partisan of any particular language to find this arrangement intellectually offensive. The chains are independent. Their parallelism is, to use the technical term, embarrassing. And yet the runtime forces them into a pantomime of separation that wastes memory, wastes compute, and -- what is less often remarked upon -- wastes the programmer's time in managing a concurrency model that should not require management at all.

The BEAM virtual machine, upon which Elixir runs, takes the opposite view. Concurrency is not a feature to be bolted on. It is the substrate. Lightweight processes -- millions of them, if you like -- share a heap, communicate through immutable messages, and are scheduled preemptively across all available cores. Four MCMC chains are four processes sharing one compiled model, dispatched with a single call to `Task.async_stream`. The compile happens once. The chains cannot interfere with each other because the language makes interference structurally impossible.

The question, which I confess I found irresistible, was whether the mathematics could be made to follow the runtime. Whether a virtual machine designed for telephone switches could be taught Hamiltonian Monte Carlo.

---

## II. Memoranda

Every non-trivial project is a sequence of bets, and the honest thing to do is write them down.

Exmc's `DECISIONS.md` grew to thirty-five entries. Each records a decision, a rationale, an assumption that must hold for the decision to remain valid, and the implications if it does. This is not, I should emphasize, the common practice of software engineers, who generally prefer to let their architectural choices fossilize in code and become the subject of folklore. Writing decisions down has the inconvenient property of making them falsifiable, which is precisely why one should do it.

Decision one: use Nx, Elixir's tensor library, as the numerical backbone. Nx is to NumPy what a promising undergraduate is to a tenured professor -- younger, more flexible, arguably more principled in design, and not yet proven under fire. Decision two: represent models as a small intermediate representation before compiling them to differentiable closures. Decision four: distributions declare their own constraint transforms. The user is spared the medieval torture of manually applying Jacobian corrections.

The most instructive entry in the document is not a decision that held, but one that didn't. Decision twelve declared, with the breezy confidence of early architecture, that observation log-probability terms could be computed eagerly at compile time. Observed values are constants, the reasoning went, so their contribution to the log-probability is fixed. Compute it once and fold the result into the closure. Elegant. Efficient. Wrong.

Decision twenty-two introduced hierarchical parameter references -- string values like `"mu"` in a distribution's parameter map that point to other random variables. This meant that an observation's target distribution could have parameters that were *not* constants but references to free variables whose values change at every sampler step. The observation term was no longer fixed. It required a deferred closure, evaluated fresh at each iteration.

Decision twelve's note reads: "Partially superseded by D22." This is, I submit, the most valuable sentence in the entire document. Not because it describes a failure -- all assumptions fail eventually -- but because it records the failure honestly, in the same document that recorded the original confidence. Most engineering projects bury their reversals in git history. This one puts them on the same page as the original claim, with a cross-reference.

Thirty-five entries. Not a manifesto, but a *memorandum* -- things that must be remembered, including the things we got wrong.

---

## III. One Thousand Times Too Slow

I shall describe this bug with the specificity it deserves, because it illustrates a principle that software engineers would do well to internalize: the distance between an elegant abstraction and a usable system is sometimes measured in orders of magnitude.

The NUTS sampler requires random numbers. Lots of them. Direction choices, proposal weights, acceptance decisions -- each iteration consumes several draws from a pseudorandom number generator. The natural approach in Nx is to use `Nx.Random`, which provides a pure functional PRNG in the style of JAX: generate a key, split it, draw from it. Deterministic, composable, beautiful.

On BinaryBackend, each call to `Nx.Random.split` took one second.

I will repeat that, because the mind tends to correct what it assumes must be a typographical error. One second. Not one millisecond. The operation triggered Nx's `defn` tracing machinery -- the system that analyzes computation graphs for potential JIT compilation -- which dutifully analyzed the graph, discovered there was no JIT compiler available, and proceeded to interpret the operation at the speed of molasses flowing uphill in January. Every call. Every iteration. A sampler that needed three hundred warmup iterations and three hundred sampling iterations, with five random draws per step, would require approximately three thousand seconds. Nearly an hour. For a model with a single free parameter.

The solution was Decision sixteen, which I consider one of the finest decisions in the document, not because it is clever but because it is ruthless. Abandon `Nx.Random`. Use Erlang's built-in `:rand` module instead. Erlang's PRNG is a plain scalar function: you give it a state, it returns a number and a new state. No tracing. No graph analysis. No ceremony. Microseconds per call, as God intended.

```elixir
rng = :rand.seed_s(:exsss, seed)
{value, rng} = :rand.normal_s(rng)
```

The sampler went from hours to seconds. The lesson is not that Nx.Random is badly designed -- it is well designed, for EXLA. The lesson is that running a JIT-oriented library on an interpreter is not a graceful degradation. It is a thousand-fold performance cliff, and no amount of architectural elegance will save you from falling off it.

---

## IV. Two Feedback Loops

The sampler's first successful run produced samples from a standard Normal distribution. Mean: approximately zero. Variance: approximately 1.5.

The variance was wrong by fifty percent, consistently across seeds. Not stochastic noise -- systematic bias. Something was broken in the dynamics. The culprit was found, after considerable labor, in the NUTS tree builder: when extending a subtree from a trajectory endpoint, the code used the gradient at the *proposal* position rather than the gradient at the *endpoint*. The trajectory is a binary tree of leapfrog states, each carrying its own gradient. Confuse one node's gradient with another and the Hamiltonian dynamics become subtly wrong -- not wrong enough to crash, but wrong enough to inflate the posterior variance by fifty percent.

This was the first feedback-loop bug. The second was more elegant in its stupidity.

During warmup, the dual averaging algorithm adapts the step size. It observes the acceptance rate of each NUTS step, compares it to the target of 0.8, and adjusts epsilon accordingly. The algorithm was working flawlessly. It tracked the acceptance statistics. It computed the smoothed estimates. It produced, at every iteration, an excellent step size.

Nobody was using it.

The `run_phase` function -- the orchestrator for each warmup phase -- was calling the NUTS step function with a *fixed* epsilon captured at the start of the phase. The dual averaging state was updated correctly after each step, but the updated epsilon was never fed back into the next step. The adaptation machinery was running in perfect isolation, producing correct answers that vanished into the void. The step size collapsed to `1e-8`.

The fix for both bugs was trivial once the cause was found: thread gradients correctly through the tree, feed `current_epsilon(da_state)` into each NUTS step. But I dwell on these because they belong to the same species of error -- the error of systems that compute correct intermediate results and then fail to deliver them to the system that needs them. A gradient computed at the wrong position. A step size computed in the right place and used in the wrong one. In both cases, every component was doing its job. The failure was in the handoff.

One is reminded of institutions that produce excellent policy papers which are then filed, unread, in the appropriate cabinet. The algorithm was sound. The wiring was wrong. I submit that most bugs of consequence follow this pattern.

---

## V. The Textbook Lied

There are two formulas in the standard numerical computing curriculum that do not work, and Exmc discovered both of them the hard way.

The first is `log1p(exp(x))`, the textbook definition of the softplus function. It appears in every machine learning textbook, every reference implementation, every tutorial. It is also, for large values of x, a floating-point bomb. When x exceeds approximately 700, `exp(x)` overflows to infinity, `log1p(infinity)` returns infinity, and the softplus -- which should grow linearly for large arguments -- explodes instead. On BinaryBackend, which lacks the overflow guards that GPU runtimes quietly apply, this happens with predictable regularity.

Decision twenty-five replaced the textbook formula with the numerically stable identity: `max(x, 0) + log1p(exp(-|x|))`. The negative absolute value in the exponent ensures that the `exp` argument is always non-positive, which means it is always between 0 and 1, which means it never overflows. The `max(x, 0)` term captures the linear growth. The result is mathematically identical and numerically indestructible.

The log-Jacobian of the logit transform -- `log(sigmoid(z))` in the textbook -- was similarly rewritten as `-softplus(-z)`, because `sigmoid(z)` underflows to zero for large negative z and `log(0)` is negative infinity. The stable form never underflows.

I emphasize these because they illustrate a point that Hitchens the elder would have enjoyed: the people who write the textbooks and the people who implement the formulas on real hardware are, apparently, different populations with insufficient overlap. The formula `log1p(exp(x))` is correct in exact arithmetic. It is wrong on every computer ever built, for sufficiently large x. The textbook does not mention this. The student who implements it faithfully will discover the problem at 2 AM when their sampler produces infinity for no apparent reason.

The second formula that lied was `lgamma`, the logarithm of the gamma function. The Lanczos approximation -- nine coefficients, a rational function, fifteen digits of accuracy -- was implemented faithfully in pure Nx operations. Forward evaluation: flawless. Gradient via automatic differentiation: catastrophic.

Differentiating `c / (x + i)` produces `c / (x + i)^2`. In the tails of a Gamma distribution, these squared terms grow large enough to trigger Elixir's `Complex.divide`, a function that is not equipped for numbers of this magnitude. The result is NaN, which propagates through the gradient, through the leapfrog step, through the tree builder, and into the sampler as a divergent transition that cannot be recovered.

Decision twenty-four was the compromise: when EXLA is available, wrap the gradient computation in `EXLA.jit`. XLA has a native lgamma kernel that never touches the Lanczos coefficients. On EXLA, Gamma and Beta work as sampled priors. On BinaryBackend, they work only as observation likelihoods, where their logpdf evaluates to a constant that vanishes from the gradient.

Two textbook formulas. Both correct in the platonic realm of exact arithmetic. Both wrong on the hardware where the computation actually runs. The stable replacements are not more mathematically sophisticated -- they are, if anything, less elegant. They are simply *honest* about the machine they run on, which is more than can be said for the formulas they replace.

---

## VI. The Productive Contradiction

Decision fourteen states, with admirable clarity: "The NUTS sampler is implemented with plain Elixir functions and Nx tensor ops, not `defn`."

The rationale is sound. On BinaryBackend, `defn` adds tracing complexity without JIT benefit. The sampler loop -- tree building, proposal selection, adaptation -- is control-flow-heavy code that does not benefit from tensor graph compilation. Plain Elixir with Nx operations is simpler, faster to write, easier to debug, and no slower to execute. Decision fourteen is correct.

Decision twenty-four states, with equal clarity: "`Compiler.value_and_grad` auto-detects EXLA via `Code.ensure_loaded?(EXLA)` and wraps the logp closure in `EXLA.jit`."

This is a direct contradiction of decision fourteen. The sampler does, in fact, use JIT compilation. The inner function -- the one that computes the log-probability and its gradient, the most numerically intensive operation in the entire system -- is compiled to native code via XLA when EXLA is available. Decision twenty-four is also correct.

Both decisions are correct because they are correct about different things. Decision fourteen is correct about the *sampler loop*: the tree building, the U-turn detection, the multinomial proposal selection, the dual averaging, the Welford statistics. This is branchy, stateful, scalar-heavy code that would gain nothing from JIT compilation and would be significantly harder to write in `defn`'s restricted subset of Elixir.

Decision twenty-four is correct about the *gradient*: the tensor-heavy, branch-free, mathematically pure function that maps a flat vector to a scalar log-probability and its gradient. This is exactly the kind of computation that XLA was built to optimize, and failing to JIT it would be negligent.

The sampler thus contains two systems of computation living side by side. The outer loop speaks Elixir. The inner gradient speaks XLA. They communicate through `Nx.tensor` and `Nx.to_number`, crossing the JIT boundary with every leapfrog step. Decision seventeen extends this schism further: dual averaging and Welford use *Erlang native floats* -- not even Nx tensors -- because scalar adaptation arithmetic running through BinaryBackend adds approximately 100x overhead for no benefit.

Three number systems in one sampler. Erlang floats for adaptation. Nx tensors for the sampler loop. EXLA-compiled closures for the gradient. A purist would find this intolerable. A pragmatist would note that it works, and that it works precisely because each system handles the computation it was designed for, rather than forcing a single abstraction to serve all purposes.

The document contains its own refutation, and the refutation makes the original decision stronger. I find this more honest than most codebases, which tend to either pretend their contradictions don't exist or resolve them in favor of a consistency that sacrifices performance. Exmc lets the contradiction stand and benefits from both sides of it.

---

## VII. The Geometry of Funnels

Hierarchical Bayesian models have a well-known pathology called funnel geometry, and the fact that it is well-known has not prevented it from ruining the afternoons of countless practitioners.

The setup is simple. A parent controls the variance of a child: `mu ~ N(0, 5)`, `sigma ~ HalfNormal(2)`, `x ~ N(mu, sigma)`. When sigma is large, x roams freely. When sigma is small, x is pinched into a narrow corridor. The NUTS sampler, using a single global step size, cannot serve both regimes. Large steps cause divergences in the narrow region; small steps waste computation in the wide region. The posterior looks like a funnel, and the sampler tumbles down it.

The standard remedy is non-centered parameterization: instead of `x ~ N(mu, sigma)`, sample `z ~ N(0, 1)` and reconstruct `x = mu + sigma * z`. The funnel vanishes because z's geometry is independent of sigma.

In PyMC, the user must apply this transformation by hand, which means the user must first recognize the funnel, then know the remedy, then restructure the model accordingly. This is precisely the kind of expert knowledge that a framework should automate, and so Exmc does.

The `NonCenteredParameterization` rewrite pass scans the IR for Normal random variables whose `mu` and `sigma` are both string references to free random variables -- the exact configuration that produces funnel geometry. When found, it rewrites the node: prior becomes `N(0, 1)`, original parameterization stored in metadata for trace reconstruction. The user's model declaration is unchanged. The user's trace is unchanged. Only the sampler's geometry is transformed, and only where transformation is needed.

This is, to my mind, what a rewrite pipeline is *for*: to apply mathematical knowledge that the user should not be required to possess. The expert knowledge does not disappear -- it is merely relocated from the user's head to the compiler's rewrite pass, where it can be tested, versioned, and applied consistently. Decision thirty-two's assumption is frank about the limitation: auto-NCP is assumed to be always beneficial, but models with highly informative data may need centered parameterization. The opt-out does not yet exist. The document says so.

---

## VIII. The Most Dangerous Bug

I have saved the most instructive bug for its own chapter, because it embodies a principle that I believe applies well beyond software engineering: the answers that are *almost* right are far more dangerous than the answers that are obviously wrong.

The compiler function `resolve_params` resolved string parameter references by looking up values in the current value map. When a distribution declared `sigma: "sigma"`, the compiler fetched the value of sigma from the map. The map held unconstrained values. If sigma had a `:log` transform -- as it does for HalfNormal, Exponential, and others -- the map contained `log(sigma)`, not sigma.

For simple models near sigma = 1.0, where `log(1) = 0`, the error was small. The sampler would converge to a posterior that was *slightly* wrong, in a way that required careful analysis to detect. The numbers looked reasonable. The chains mixed. If you squinted, you could persuade yourself that everything was fine. Integration test ten -- sigma drawn from Exponential(1), child drawn from Normal(0, sigma), one observation at 2.0 -- *passed*. It passed with the wrong log-probability, producing a wrong posterior that happened to fall within the generous tolerances required by stochastic testing.

It "worked" by accident.

For hierarchical models with five free parameters, the error was apocalyptic. The step-size search collapsed to `1e-141`. Every sample diverged. The trace was a flatline.

The fix was `resolve_params_constrained`, which looks up each string reference's transform in the PointMap and applies the forward transform before returning the value. `log(sigma)` becomes `exp(log(sigma)) = sigma`. The constrained value reaches the distribution as intended.

The lesson is one that Orwell would have appreciated: the gap between unconstrained space and constrained space is an abstraction boundary, and abstraction boundaries are precisely where errors hide. The sampler lives in unconstrained space because the mathematics demands it. The distributions live in constrained space because that is what their parameters mean. The boundary between these two spaces must be crossed with the correct passport, and `resolve_params` was forging documents.

What makes this bug dangerous is not that it produced wrong answers -- all bugs do that. It is that it produced wrong answers that *looked right* for the simple cases where you check, and catastrophically wrong answers for the complex cases where you depend on the simple cases having been verified. The test suite, which existed precisely to catch errors like this, was complicit in the deception: the tolerances were generous enough to accommodate both stochastic variance and systematic bias, and the bug hid in the gap between them.

---

## IX. A Function Called to_number

There is a function in Nx called `to_number`. Its name is a promise: give it a tensor, receive a number. The promise is, under certain conditions, a lie.

When the tensor contains a normal floating-point value, `to_number` returns an Elixir float. When the tensor contains NaN, it returns the atom `:nan`. When it contains negative infinity, it returns the atom `:neg_infinity`. When it contains positive infinity, it returns the atom `:infinity`. These are not numbers. They are atoms -- Elixir's equivalent of symbols, opaque identifiers with no arithmetic semantics whatsoever.

The consequences are immediate and total. Erlang arithmetic on atoms does not produce a wrong answer. It does not produce NaN. It crashes the process. `1.0 + :nan` is not a floating-point operation that returns NaN, as it would be in every other numerical language. It is a type error that terminates execution. If the NUTS tree builder calls `to_number` on a log-probability tensor that happens to be negative infinity -- which occurs routinely when the sampler explores the boundary of a constrained distribution's support -- the process dies without ceremony or explanation.

Decision twenty-six addressed this with `is_number/1` guards scattered through the sampler's arithmetic. Before any operation on a value extracted by `to_number`, the code checks whether the value is actually a number. Non-numeric values are treated as divergent steps: the proposal is rejected, the tree stops expanding in that direction, and the sampler moves on.

These guards are now permanent fixtures of the codebase, small sentries posted at every gate where a tensor value crosses into Erlang arithmetic. They are there because a function called `to_number` sometimes returns atoms, and because no amount of righteous indignation about this will prevent the sampler from crashing if the guards are removed.

I pair this with Decision nine, which identifies free random variables by a principle that a civil libertarian would recognize: a random variable is free if and only if no observation node points at it. Not because it has declared itself free. Not because it carries a metadata flag asserting its freedom. It is free by the absence of constraint -- innocent until observed. This is, in its small way, a design philosophy: the system does not require you to declare your status. It infers your status from the structure of the world around you, and if nothing constrains you, you are free.

The two decisions, taken together, paint a portrait of a system that must constantly negotiate between what things *claim* to be and what they *are*. A function called `to_number` that returns atoms. A variable that is free because nothing says otherwise. The code does not trust names. It checks.

---

## X. Four Chains, Zero Locks

I come now to what I consider the vindication of the entire enterprise.

`Sampler.sample_chains(ir, 4, init_values: init)` compiles the model once and dispatches four chains in parallel. Each chain is a BEAM process. Each chain gets its own PRNG state, seeded deterministically from the chain index. Each chain produces its own trace. The four traces return ordered, ready for R-hat convergence diagnostics.

There is no mutex. No semaphore. No atomic compare-and-swap. No lock-free queue. No concurrent hash map. No `threading.Lock()`. No `multiprocessing.Pool()`. No pickle serialization. No shared-nothing message-passing architecture (well, actually, exactly that, but it requires no thought from the programmer).

Erlang's `:rand` uses explicit-state functions that take and return a state value. No process dictionary, no global mutation. Nx tensors are immutable. EXLA closures are thread-safe. The four chains *cannot* interfere with each other, not because the programmer was careful, but because the language makes interference *unrepresentable*.

On a four-core machine, four chains complete in approximately the time of one. The EXLA JIT compilation -- which can take seconds for complex models -- happens once and is shared. This is not a heroic engineering achievement. It is the natural consequence of running concurrent work on a runtime that was designed, from the first day of its existence, for exactly this purpose.

Python's `multiprocessing` module achieves the same parallelism at the cost of four process startups, four model compilations, four memory images, and the permanent anxiety of wondering whether your shared state is actually shared correctly. The BEAM achieves it with one function call and the serene confidence of a system that has been handling concurrent workloads since before most Python programmers were born.

---

## XI. Honor Cui Honor

The visualization deserves its own account, because it represents a second act of defiance against the established order.

ArviZ, the diagnostic toolkit for PyMC, renders through Matplotlib. Matplotlib renders through whatever GUI backend your operating system provides, which may be Tk, Qt, GTK, or a web browser pretending to be a desktop application. The output is a static image -- a PNG embedded in a Jupyter notebook, inert and unresizable. Zoom in and you get pixels, not data.

ExmcViz renders through Scenic, Elixir's scene graph library, which was designed for embedded information displays -- kiosks, industrial panels, the kind of interfaces that must run continuously without crashing. Scenic draws through OpenGL via NanoVG. The scene graph is a live data structure, composited to the framebuffer at sixty frames per second. There is no image encoding. There is no file I/O. The plots exist as geometry in GPU memory.

The architecture enforces a strict boundary: all Nx tensor computation happens in `Data.Prepare`, which converts trace maps into plain Elixir lists and floats. Components receive only primitive data types. Scenic never waits for tensor operations. Tensor operations never block the UI. This is not merely good practice; it is a *structural guarantee* that the visualization cannot stall the sampler and the sampler cannot freeze the display.

The live streaming feature -- `ExmcViz.stream(ir, init, num_samples: 500)` -- connects the sampler directly to the visualization through GenServer message passing, the BEAM's native nervous system. A `StreamCoordinator` GenServer buffers incoming samples and flushes them every ten draws to the `LiveDashboard` scene, which rebuilds its entire graph with updated histograms, ACF, and summary statistics. The title bar ticks upward: "MCMC Live Sampling (150 / 500)." The trace plots grow from left to right, amber lines crawling across a black field. You are watching Bayesian inference happen in real time, rendered by a scene graph library designed for factory floor displays.

The color palette -- amber on true black -- is not a stylistic affectation. On OLED panels, true black means the pixel is *off*: zero power, zero photon emission. The warm spectrum preserves scotopic vision during late-night sampling sessions. Red means divergence. Blue means energy transition. White means the posterior mean on a forest plot. Ten chain colors cycle through the warm spectrum without ever reaching for the blue or green that would destroy your night-adapted vision at two in the morning.

Three processes, three mailboxes, zero shared state. The sampler sends messages. The coordinator buffers. The scene rebuilds. This is not a special architecture designed for live visualization. It is the *default* architecture of any concurrent Elixir application. The only design decision was to use it.

![The pair plot renders pairwise posterior correlations: histograms on the diagonal, scatter plots in the lower triangle, Pearson coefficients in the upper](assets/pair_plot_4k.png)

![The live dashboard during sampling: amber traces growing across a black field, histograms taking shape, statistics updating](assets/live_streaming.png)

---

## XII. One Hundred and Twenty-Three Theses

Martin Luther nailed ninety-five theses to a church door. Exmc has one hundred and twenty-three tests, which I submit is a more rigorous form of argumentation.

Eleven doctests verify that the examples in the documentation produce the results they claim to produce. Eighty-seven unit tests cover the machinery: IR construction, compiler output, transform correctness, rewrite passes, tree building, leapfrog dynamics, mass matrix adaptation, step-size tuning, diagnostics, and predictive sampling. Twenty-five integration tests run the full sampler on actual models and check the results against analytical posteriors.

The integration tests are the difficult ones, because MCMC is inherently stochastic and a correct sampler will, with nonzero probability, produce any result you care to name. The tolerances must be generous enough to accommodate the natural variance of five hundred samples on BinaryBackend -- mean within 0.3, variance within 1.0 of the target -- while remaining tight enough to catch systematic bugs like the variance inflation and the constrained parent error. We have already seen that these tolerances are not generous enough to *prevent* bugs from hiding -- the constrained parent error passed through them like water through a sieve. This is a genuinely hard calibration problem, and I will not pretend that it has been solved definitively. It has been solved *adequately*, which in the Bayesian tradition is the best one can hope for.

Thirty-five architectural decisions, recorded and falsifiable, including the ones that were wrong. Twelve debugging sessions, documented with the clinical detail of a pathology report. One hundred and twenty-three tests in Exmc. Thirty-four more in ExmcViz. Two textbook formulas replaced with stable alternatives. One function that lies about its return type, guarded at every call site. Three number systems coexisting in one sampler. One feedback loop that computed correct step sizes and delivered them to nobody. One bug that produced plausible wrong answers for simple models and catastrophic wrong answers for complex ones. One rewrite pass that eliminates funnel geometry without the user's knowledge or consent.

The project is a prototype, and it works. It works on a runtime that was never intended for this purpose, in a language that has no history in numerical computing, using a tensor library that did not exist five years ago. The posterior distributions it computes are correct. The parallel chains it dispatches are fast. The visualizations it renders are immediate and live. The BEAM, that improbable creation of the Swedish telecommunications industry, has been taught to do Bayesian statistics, and it does them rather well.

Whether anyone asked for this is, I maintain, beside the point. The question is whether the thing is *good*, and the evidence -- one hundred and twenty-three tests of it -- suggests that it is.

---

*Exmc: 123 tests, 35 decisions, 9 distributions, 4 layers, one runtime.*
*ExmcViz: 34 tests, 8 components, 10 chain colors, zero black pixels wasted.*

*The traces glow amber on black, converging on the truth.*
*The BEAM does not care that nobody asked.*
