# Why Captured Tensors Must Be BinaryBackend

## The EXLA Tracing Conflict

When `EXLA.jit` compiles a function, it doesn't execute it immediately — it **traces** it.
It calls the function with symbolic placeholder values (`Nx.Defn.Expr` nodes) instead of
real tensors, building a computation graph that XLA can optimize and compile.

The problem arises when the function **closes over** tensors that are already on `EXLA.Backend`:

```elixir
# With Nx.default_backend(EXLA.Backend), this creates an EXLA tensor:
alpha = Nx.tensor(2.0)  # => %Nx.Tensor{data: %EXLA.Backend{...}}

# The logp closure captures alpha
logp_fn = fn flat ->
  x = unpack(flat)
  Beta.logpdf(x, %{alpha: alpha})
end

# EXLA.jit traces logp_fn — flat becomes an Expr placeholder
EXLA.jit(fn flat -> Nx.Defn.value_and_grad(flat, logp_fn) end)
```

Inside `Beta.logpdf`, Nx tries to do `Nx.subtract(alpha, Nx.tensor(1.0))`. At trace time:
- `alpha` is an `EXLA.Backend` tensor (a real, already-computed value on the XLA device)
- The `1.0` constant becomes an `Nx.Defn.Expr` node (a symbolic placeholder)

Nx sees two different "implementations" — `EXLA.Backend` vs `Nx.Defn.Expr` — and raises:

```
cannot invoke Nx function because it relies on two incompatible
tensor implementations: EXLA.Backend and Nx.Defn.Expr
```

## The Fix

Copy captured tensors to `Nx.BinaryBackend` before they get closed over. BinaryBackend
tensors are just raw binary data — when EXLA's tracer encounters them, it inlines their
values as constants in the computation graph. No conflict.

```elixir
alpha = Nx.backend_copy(Nx.tensor(2.0), Nx.BinaryBackend)
# Now alpha is %Nx.Tensor{data: %Nx.BinaryBackend{...}}
# EXLA.jit can inline this as a constant during tracing
```

This is what `Compiler.ensure_binary_backend/1` does — it walks the IR and copies every
tensor param to BinaryBackend before the compiler builds closures that capture them.

## Performance: Is the Copy a Bottleneck?

For this project, no. The tensors captured in closures are **distribution parameters** —
scalars or tiny vectors:

- `mu: Nx.tensor(0.0)` — 8 bytes (f64 scalar)
- `alpha: Nx.tensor(2.0)` — 8 bytes
- Observation values like `Nx.tensor(5.0)` — 8 bytes
- A matmul matrix `a` in a meas_obs — maybe 64 bytes for a 2x2

The `backend_copy` is a memcpy from XLA device memory to a binary. For scalars that's
essentially free.

### Where It Could Matter in a Real PPL

- **Vectorized observations** — observing 10k data points as a single tensor is ~80KB.
  Still trivial to copy once at compile time.
- **Large design matrices** — a regression with 100k rows x 100 features = 80MB. The copy
  happens once when `Compiler.value_and_grad` is called, not per sample step, so it's a
  one-time cost. But 80MB copied off the XLA device could take a few milliseconds.
- **Image/time-series data** — if observations are large tensors (megabytes), the copy adds
  startup latency.

### Why It Doesn't Matter

Even in those cases, the copy is **O(1) per sampling run** — it happens at model compilation,
not inside the NUTS loop. The NUTS loop calls `vag_fn.(flat)` thousands of times and that's
fully JIT-compiled by EXLA. The bottleneck is always the per-step gradient evaluation, never
the one-time parameter copy.

**Summary:** The captured tensors are model metadata (params, observations), not the sampling
state. They're copied once at setup. Even at realistic PPL scale this is not a bottleneck.
