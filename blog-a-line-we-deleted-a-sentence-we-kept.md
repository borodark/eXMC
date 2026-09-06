# A Line We Deleted, a Sentence We Kept

There is a particular kind of falsehood that survives because nobody can be
bothered to disagree with it. It is not a lie, exactly. Nobody profits. It sits
in the corner of a file, well-mannered and plausible, and every reader who
passes nods at it and moves on. Ours read like this:

```elixir
Nx.Vulkan ->
  # VulkanoBackend implements compute callbacks (binary/unary
  # SPV ops + host fallbacks). Evaluator dispatches each defn
  # op through the default backend, which is set globally to
  # VulkanoBackend at application boot.
  Nx.Defn.jit(fun, [{:compiler, Nx.Defn.Evaluator} | opts])
```

Four lines. Every clause in them is checkable. Three of them are true.

The false one is the last: *which is set globally to VulkanoBackend at
application boot.* There is no application boot. Our `mix.exs` declares no
`mod:`, so the application has no `start/2`, so there is no moment at which
anything could be set. Nothing in `config/` touches the Nx default backend. The
single call to `Nx.global_default_backend/1` anywhere in the repository lives
inside one test's `setup_all`, and it politely puts things back afterwards.

So for a year, on the path we believed was our GPU path, the sentence described
a thing that never happened.

## Where it came from, which is the interesting part

I assumed the comment had rotted — that it was true once and drifted, the way
comments do. It hadn't. It was never true *here*. It was true somewhere else,
and we brought it home with us.

The library underneath us ships this:

```elixir
def jit(fun, opts \\ []) do
  ensure_default_backend!()
  Nx.Defn.jit(fun, [{:compiler, Nx.Defn.Evaluator} | opts])
end
```

`ensure_default_backend!/0` does exactly what our comment claimed: it checks the
Nx default backend and, finding it wanting, sets it globally to the GPU one. Our
function is that function with the first line removed. Somebody — and the git
history says it was us, in a commit whose message is otherwise a model of care —
copied the body, dropped the call deliberately, and kept the comment that
explained the version *with* it.

That is a better story than rot, and a worse one. Rot is entropy. This was a
sentence that was accurate about a neighbouring universe, carried across the
border, and never asked for its papers.

## Why nothing caught it

Comments are not executed, which everyone knows and nobody feels. A test can
fail. A type can be checked. A comment can only be read, and reading is exactly
the activity that a confident, well-formed, technically-fluent sentence is
designed to survive.

What made ours durable was a second thing, and it is the one worth generalising.
We had a banner. At the start of every test run and every benchmark, the code
printed what it was about to compute with:

```
compiler=Nx.Vulkan (configured: :vulkan) backend=Nx.Vulkan.VulkanoBackend precision=:f64
```

Look at that line. It has the grammar of evidence. It is emitted at runtime, it
has fields, it says `backend=` and then names a backend. It reads like a
measurement.

It was not a measurement. The function behind `backend=` took the *compiler* we
had detected and returned the backend that compiler would *imply*. It never
asked Nx anything. It was the comment again, wearing a lab coat — a derivation
presented in the costume of an observation.

This is worse than an untested comment, because it manufactures the feeling of
having checked. Nine months of runs printed that line. Every one of them
confirmed a belief that no part of the system had verified.

## What it cost

Two models in our benchmark suite ran forty and one hundred and forty times
slower on the "GPU" arm than on the CPU compiler. We spent real time on that. We
measured memory architectures. We compared a discrete card against a
unified-memory board on the theory that PCIe round-trips explained it. We wrote
the phrase *the GPU is slow* into notes, and meant it.

Meanwhile the benchmark suite reported 33 of 33 models passing, a number that
appears in our changelog, and which was — every time, for months — produced by a
different backend than the one it was credited to.

There was no GPU on that path. There was an interpreter, walking an expression
tree, on the CPU, under a banner naming a graphics card.

## The cut

Wave 0 was three changes and only one of them executes.

The banner now prints what it derives *and* what it observes, side by side:

```
compiler=Nx.Vulkan (configured: :vulkan) backend=Nx.Vulkan.VulkanoBackend
  precision=:f64 perop_fallback=true
  nx_default_backend={Nx.BinaryBackend, []} nx_defn_options=[]
```

`backend=` is the old derivation, kept and now labelled as one.
`nx_default_backend=` is `Nx.default_backend()`, which is not an opinion. They
disagree, in public, for the first time.

But the banner is not the proof. The proof is one line further down. We jit a
trivial function and ask the *result* which backend it came from:

```
out_backend=Nx.BinaryBackend
```

The evaluator dispatches each operation on the backend of the tensor it is
given. A `BinaryBackend` result can only mean that every operation went through
`BinaryBackend`'s callbacks. The GPU backend was never asked to compute
anything. Not slowly. At all.

And then the part that makes it evidence rather than an anecdote — the control.
We ran the same probe with the compiler set to `none`, where derived and
observed *ought* to agree:

```
compiler=nil (configured: :none) backend=Nx.BinaryBackend
  nx_default_backend={Nx.BinaryBackend, []}
```

They agree. So the disagreement on the other arm is a signal about the system
and not an artifact of the two fields we had just added. A gate that has never
been shown to pass on a healthy case is not a gate; it is a decoration. The
control cost one command.

## The kludge was not the missing line

It is tempting to file this as a copy-paste error: a line was dropped, a comment
was orphaned, tidy it up and move along.

That reading is too comfortable, and it would leave the mechanism intact. The
missing `ensure_default_backend!()` was a *decision* — a correct one. This
sampler keeps its small tensors on the host on purpose. Sending every scalar
constant and every adaptation counter across a device boundary, one round trip
apiece, for three floats, is not an optimisation. We meant to drop that line and
we should not restore it.

The kludge was the sentence we kept. It was a claim about runtime state, written
in a place where runtime state cannot be checked, and then dressed up in a
runtime banner that made it look as though it had been. The fix is not
punctuation. It is a rule, and it is short enough to remember:

**Derived state and observed state must never wear the same clothes.**

Print both. Label which is which. When they disagree, believe the second one.
And write the comment so that the next person reading it can tell, from the text
alone, whether it is describing this function or the one it was copied from.

Our new comment does something the old one could not: it names the measurement
that supports it, with a date and a host. That does not make it true forever.
Nothing does. But it makes it *falsifiable*, which is the whole of the
difference, and it is the only property that mattered.
