# Handling the Nx backend in projects that use Nx

Nine proposals for choosing, applying and verifying an Nx backend across dev,
test and prod. Each one exists because its absence cost this project real time,
and each cites the evidence rather than asserting the rule.

The governing idea: **a numerical result that cannot say what produced it is not
a result.** Most of what follows is machinery for making that impossible.

---

## P0 — Know which of the TWO axes you are setting

This is the one that catches everyone, so it goes first. Nx has two independent
settings and configuring one does not configure the other:

| key | governs | default |
| --- | --- | --- |
| `:nx, :default_backend` | where EAGER tensors allocate | `{Nx.BinaryBackend, []}` |
| `:nx, :default_defn_options` | what COMPILES `defn` | `[]` |

Verified at `deps/nx/mix.exs:25`. An empty `default_defn_options` means **no
compiler**, so `defn` runs through `Nx.Defn.Evaluator` — a tree-walking
interpreter — over whatever backend you set.

**What it cost here.** exmc's Vulkan arm calls
`Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator)` (`lib/exmc/jit.ex:55`). Backend:
a GPU. Compiler: an interpreter. Two models were 40x and 140x slower than the
EXLA arm and it read as "the GPU is slow" until the split was measured on two
architectures. It is not a device problem; there is no fusing compiler on that
path at all.

Setting `default_backend: EXLA.Backend` and expecting compiled `defn` is the
same mistake in the opposite direction.

## P1 — Set both axes explicitly, per environment, and IMPORT the env file

**Mix auto-loads only `config/config.exs`.** Everything else must be imported
explicitly, and a missing import is silent — indistinguishable from a config
that chose the default.

    # config/config.exs
    import Config
    import_config "#{config_env()}.exs"

**What it cost here.** `config/config.exs:19` reads
`if config_env() == :test, do: import_config "test.exs"`. The benchmark harness
runs under `mix run`, which is `:dev`. So the `EXMC_COMPILER` switch, the EXLA
client setting and the per-op fallback flag were all inert for that entire
suite, which silently took `JIT.auto_detect/0` → EXLA. Its headline "33/33 PASS"
was an EXLA result that never said so, and nothing had run it against the
GPU path at all.

## P2 — Put environment-variable selection in `runtime.exs`

`config/runtime.exs` is evaluated at runtime, identically under Mix and under
releases. It is the only correct home for anything read from the environment.

`Application.compile_env/2` is the opposite tool: it BAKES the value into the
compiled module and Elixir refuses to boot when the runtime value differs. Right
for a hard invariant, wrong for a switch. Know which one you are reaching for.

## P3 — Demand, do not prefer

If a named backend is unavailable, **raise**. Do not fall through to the next
one.

    :exla -> demand(EXLA, :exla)     # raises if unusable
    :auto -> auto_detect()           # falling through is a NAMED choice

**What it cost here** (`lib/exmc/jit.ex`, the comment on `demand/2`): a
non-interactive shell lacked the `LD_LIBRARY_PATH` the pip nvshmem/nvrtc wheels
need, so CUDA EXLA could not resolve `libnvshmem_host.so.3`. The old code
correctly judged EXLA unusable and quietly fell through to Vulkan. The suite went
from 546/1 to 549/10 and read exactly like a code regression. About an hour, and
the whole of it was that nothing said "you asked for EXLA and you are not
getting it".

"Usable" must also be the STRONG check. `Code.ensure_loaded?/1` answers `true`
for a CUDA EXLA whose NIF cannot find its shared library, and then raises on
first use.

## P4 — Print a resolved banner, and store it in every artifact

At startup, resolve and print: requested backend, RESOLVED backend, compiler,
precision. Then put the same fields into every result file you write, alongside
the commit sha of the code and of each numerical dependency, plus a dirty flag.

    compiler=Nx.Vulkan (configured: :vulkan) backend=Nx.Vulkan.VulkanoBackend precision=:f64

Read it back from the library after applying the request — never record what you
asked for as though it were what you got.

**What it cost here.** Every posteriordb figure on record before 2026-09-05 —
including the committed results table — names no backend, no device and no
commit. They cannot be compared to anything, and re-deriving what produced them
required reading the shell script that invoked them.

## P5 — Never set a process-local default and then fan out

| call | scope |
| --- | --- |
| `Nx.default_backend/1` | **process dictionary** (`nx.ex:4894`) |
| `Nx.global_default_backend/1` | Application env (`nx.ex:4923`) |
| `Nx.Defn.default_options/1` | **process dictionary** (`defn.ex:240`) |
| `Nx.Defn.global_default_options/1` | Application env (`defn.ex:260`) |

The process-local pair does NOT inherit into spawned processes. Nx says so
outright: *"if you start a separate process, such as `Task`, the default options
must be set on the new process too."*

So setting a backend process-locally and then running `Task.async_stream` gives
you workers on the GLOBAL default while your logs claim otherwise. Either set it
inside each worker, or use the global setter, or (better) configure it.

Nx marks both global setters as "avoid at runtime… mostly for scripting and
testing", which is a fair warning and not a prohibition: a benchmark harness
choosing a backend at startup is exactly the scripting case.

## P6 — Verify at runtime; do not infer from config

    Nx.default_backend()       #=> {module, opts}
    Nx.Defn.default_options()  #=> keyword; :compiler, or [] meaning Evaluator

Config says what you asked for. These say what you have. When they disagree,
the second one is what ran.

## P7 — Do not use a NIF checksum as a "nothing changed" check

A Rustler artifact embeds its absolute build path, so `MIX_ENV` alone changes
the hash for identical source. Worse, `priv` is commonly symlinked from both
build trees, so `dev` and `test` share ONE `.so` and the last writer keeps it —
a dev run can be executing the test-env build.

Checksums remain valid WITHIN a fixed `MIX_ENV`, which is the only thing they
are usually being asked to prove (that an artifact was not replaced). Across
environments they prove nothing. This was diagnosed by the nx_vulkan side after
we mis-attributed a moving hash to Cargo non-determinism.

## P8 — A layer that transports a value it never reads cannot validate it

If your backend passes a parameter through to a shader or kernel without
interpreting it, no check in that backend can catch a wrong value. The
invariant belongs where the antecedent is known.

**What it cost here.** The synthesised chain shader emitted
`for (uint j = 0u; j < pc.n_obs; j++)`, while `n_obs` was 0 for every model whose
data arrived as closure captures rather than through the observation buffer. The
loop ran zero times, the likelihood evaluated to nothing, and the sampler
returned the PRIOR while reporting it as a posterior — no crash, no error, no
divergence. 32 of 33 benchmark models. Max mean error 55.085 against a reference
of 0.13.

`n_obs` crosses the NIF boundary unread, and `n_obs == 0` is CORRECT for
families that bake their parameters into the SPIR-V, so the backend cannot guard
it without breaking its own callers. The real invariant — *if the generated body
loops on `n_obs`, then `n_obs` must be non-zero* — is knowable only by the
generator, which is where the guard now lives.

## P9 — Test the backend plumbing, not just the numerics

Three tests, all cheap, each closing a failure that is otherwise silent:

1. **Assert the config import is live.** A dropped `import_config` produces no
   error. Assert a key that only the env file sets.
2. **Assert the resolved backend matches the request**, and that a bogus request
   raises rather than degrading.
3. **Assert the generated artifact, not just that it compiles.** A shader that
   passes `glslangValidator` can still compute the wrong thing. Every
   differential test in this repo compared `Nx.Defn.Expr` trees or composed
   `defn` functions; none evaluated a rendered GLSL string numerically, which is
   exactly the gap P8's defect lived in. The catching test was two lines against
   a fixture that already existed.

---

## Checklist

- [ ] Both `:default_backend` and `:default_defn_options` set explicitly, per env
- [ ] `config/config.exs` imports the env file; a test asserts the import is live
- [ ] Environment-driven selection lives in `runtime.exs`
- [ ] A named backend that is unusable RAISES
- [ ] Startup banner prints requested vs resolved; artifacts record both, plus shas
- [ ] No process-local default set before a `Task` fan-out
- [ ] Artifact checksums compared only within one `MIX_ENV`
- [ ] Generated-code invariants enforced in the generator, not the transport
- [ ] At least one test evaluates generated output numerically against a reference
