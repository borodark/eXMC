# What If Probabilistic Programming Were Different?

Every probabilistic programming language works the same way. You define a model in Python, click run, wait, and eventually get back a bag of numbers. PyMC, Stan, NumPyro, Turing.jl — the syntax varies, the runtime doesn't. There's a Python process (or a C++ binary, or a Julia session) that grinds through thousands of gradient evaluations, and when it's done, it hands you a trace object. If a numerical error blows up midway through, you start over. If you want four chains, you spawn four OS processes and hope they all finish. If you want to watch what's happening while it runs — you don't. You wait.

This is fine for notebooks. It is not fine for production systems, for distributed clusters, or for the kind of interactive data science where a practitioner needs to see whether their model is working before committing to a 20-minute sampling run.

What if the runtime were designed differently?

## The Accidental Architecture

The uniformity of existing PPLs isn't a design choice — it's an accident of history. PyMC was born in the Python scientific ecosystem, so it inherited Python's execution model: single-threaded, synchronous, no fault isolation. Stan was born in C++, so it inherited compiled binaries and MPI for distribution. NumPyro was born on JAX, so it inherited functional transformations and `pmap` for same-host parallelism.

None of these runtimes were designed for the things modern inference actually needs:

- **Streaming.** Show me the posterior as it forms, not after it's done.
- **Fault tolerance.** If one chain hits a numerical singularity, don't kill the whole run.
- **Distribution.** Run chains across machines without MPI, Dask, or Ray.
- **Composition.** If I build streaming and distribution separately, they should work together without glue code.

These aren't exotic requirements. They're the baseline for any distributed system built in the last 30 years. Telecom switches, chat servers, and database clusters have solved them since the 1990s — using a runtime called the BEAM.

## The Experiment

Exmc is a probabilistic programming framework built in Elixir on the BEAM virtual machine — the same runtime that powers WhatsApp, Discord's backend, and Ericsson's telecom infrastructure. It implements NUTS (the No-U-Turn Sampler), the same algorithm behind PyMC and Stan, with automatic differentiation via Nx and JIT compilation via EXLA (Google's XLA compiler).

The thesis isn't that Elixir is a better language for math. It's that the *runtime* gives you properties that other PPLs can't have — not because they chose not to, but because their runtimes can't express them.

Here's what falls out naturally.

### Every sample is a message

In Exmc, `sample_stream` sends each posterior draw as a message to any process:

```elixir
Exmc.NUTS.Sampler.sample_stream(model, self(), init, opts)

# In your process:
receive do
  {:exmc_sample, i, point_map, step_stat} ->
    # Update a chart, feed a web dashboard, log to disk — your choice
end
```

The sampler doesn't know or care what the receiver does with the sample. It could be a Scenic window drawing trace plots. It could be a Phoenix LiveView pushing updates to a browser. It could be a GenServer aggregating results from multiple chains. The protocol is one message type. The rest is just processes talking to processes.

PyMC's `pm.sample(callback=fn)` blocks the sampling thread during the callback. Stan has no streaming API at all. The difference isn't a feature — it's an architectural consequence. BEAM processes have mailboxes. Sending a message is non-blocking. Receiving is pattern-matched. This is how the runtime works at every level, from supervisors to database drivers. Streaming inference isn't a feature that was designed — it's a property that emerged.

### If it crashes, try again

NUTS explores a binary tree of trajectory points. Each tree expansion involves a leapfrog integration step that can fail — NaN gradients, infinite energies, out-of-memory on GPU. In PyMC, a numerical failure during tree building kills the sample. In Stan, a divergent transition is logged but the mechanism is baked into C++.

In Exmc, each subtree build is wrapped in `try/rescue`:

```elixir
try do
  build_subtree(state, direction, depth, step_fn)
rescue
  _ -> divergent_placeholder(state)
end
```

The overhead of this on the happy path is zero — literally zero instructions, because BEAM's `try` is implemented as exception registration in the process stack, not as a wrapper function. When a subtree crashes, it's replaced with a divergent placeholder that the tree merge logic already knows how to handle. The sampler continues producing valid posterior samples.

This extends to the distributed case. If a remote node dies mid-chain, `:erpc.call` raises, the chain retries on the coordinator, and the user never notices. The recovery works because each chain is functionally pure — deterministic given a seed and tuning parameters. No shared state to corrupt. No warmup to redo.

### Distribution is just sending messages farther

Here is the entire mental model for distributed MCMC on the BEAM:

1. Start some nodes: `Node.start(:"coordinator@10.0.0.1")`
2. Send the model to each node: `:erpc.call(node, Sampler, :sample_stream, [model, coordinator_pid, init, opts])`
3. Each node compiles independently (supports heterogeneous hardware), warms up, and streams samples back

That's it. No MPI. No Dask. No Ray. No message broker. No serialization library. Erlang's distribution protocol handles term serialization transparently. The model IR is plain Elixir data — maps and strings, under 1KB. PIDs are location-transparent: `send(pid, msg)` works identically whether `pid` is local or on a machine across the network.

The entire distributed sampling system is 203 lines of Elixir.

### The composition test

Here's the result I'm most proud of, because it wasn't designed — it was discovered.

Streaming inference (Chapter 4 of the thesis) and distributed sampling (Chapter 5) were built months apart. Streaming was designed for single-chain ExmcViz visualization. Distribution was designed for batch result collection. They had no knowledge of each other.

To make them work together — 4 remote nodes streaming samples to a live visualization in real time — took 3 lines of code change to an existing module and a 10-line forwarder process. The visualization components needed zero changes. The coordinator needed zero changes. The sampler needed zero changes.

The demo: 4 peer nodes, each independently compiling, warming up, and sampling an 8-parameter hierarchical model. 20,000 samples in 21 seconds. Eight trace plots, histograms, and autocorrelation functions updating live from all four nodes simultaneously.

This composition worked on the first attempt because both features were built on the same primitive: `send(pid, {:exmc_sample, i, point_map, step_stat})`. The streaming protocol doesn't care where the sender is. The distribution protocol doesn't care what the receiver does. They compose because the actor model's message passing is the same mechanism at every scale.

No existing PPL can do this. Not because it's technically impossible in Python or C++ — but because their runtimes don't provide location-transparent processes with mailboxes, so every integration point requires explicit plumbing.

## The Numbers

Does it actually work as a sampler? Yes.

Head-to-head against PyMC (5-seed median, 1 chain, same model, same data):

| Model | Exmc ESS/s | PyMC ESS/s | Ratio |
|-------|-----------|-----------|-------|
| Simple (2 params) | 469 | 576 | 0.81x |
| Medium (5 params, hierarchical) | 298 | 157 | **1.90x** |
| Stress (8 params, 3 groups) | 215 | 185 | **1.16x** |

Exmc beats PyMC on the two harder models. The simple model gap (0.81x) is the irreducible cost of an interpreted host language — Elixir's BEAM VM vs. PyMC's C extensions. For models where adaptation quality matters more than per-step speed, Exmc wins.

With 4-node distribution: 3.4-3.7x near-linear scaling. No tuning, no infrastructure.

## What this means

The point isn't "use Elixir for Bayesian inference." The point is that PPL architecture is downstream of runtime choice, and the runtimes we've been using constrain what PPLs can be.

If your runtime has message passing, you get streaming inference for free. If your runtime has process isolation, you get fault-tolerant sampling for free. If your runtime has transparent distribution, you get multi-node MCMC for free. And if all three properties come from the same mechanism, independently-designed features compose without integration effort.

The question isn't whether you *can* add streaming to PyMC or distribution to Stan. You can — with callbacks, multiprocessing pools, MPI bindings, serialization libraries, and months of engineering. The question is whether these properties should be emergent or engineered. Whether they should fall out of the runtime or be bolted on.

Exmc is an existence proof that they can be emergent. A probabilistic programming framework where liveness, fault tolerance, and distribution aren't features — they're consequences.

---

*This is the first in a series about building a probabilistic programming framework on the BEAM virtual machine. The code is at [repo link]. The thesis — "Probabilistic Programming on Actor-Model Runtimes" — covers the full technical story.*
