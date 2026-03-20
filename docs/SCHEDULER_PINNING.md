# The 34% You Leave on the Table

## How BEAM Scheduler Binding Rescued Our NUTS Throughput

The number was 0.73. Not a probability, not an acceptance rate — jobs per second.
Forty-four concurrent MCMC sampling tasks on eighty-eight logical processors,
and the system was delivering 0.73 completions per second. The arithmetic was
not encouraging.

We had recently completed a capacity planning exercise, the kind of responsible
engineering that looks good in documentation and produces tables nobody reads
until something catches fire. The table said we could handle 1,400 instruments
on a 20-minute update cycle. What the table did not say was that we were
measuring the system in its default configuration, which turned out to be
approximately two-thirds of a system.

---

### The Machine

The server is a dual-socket Intel Xeon E5-2699 v4 — forty-four physical cores
across two sockets, eighty-eight threads with hyperthreading, and 256 gigabytes
of RAM divided between two NUMA domains. Each socket has its own memory
controller. When a thread on socket 0 reads memory attached to socket 1, the
latency roughly doubles. This is not a design flaw. It is the cost of scaling
beyond what a single piece of silicon can address.

The BEAM virtual machine, which powers Erlang and Elixir, creates one scheduler
thread per logical processor — eighty-eight schedulers in our case. Each
scheduler has its own run queue and, in theory, its own affinity to a subset of
cores. In practice, the default is `unbound`: the operating system decides which
core runs which scheduler, and it decides this hundreds of times per second based
on criteria that have nothing to do with NUMA locality.

### The Workload

Each NUTS sampling job takes approximately 5 seconds of wall time. The work is
almost entirely inside EXLA's JIT-compiled XLA code: gradient evaluations,
leapfrog integrations, tree building. The working set per job is roughly 3
kilobytes — a parameter vector, a gradient, a momentum vector, an inverse mass
diagonal, and 200 floats of observation data. This fits comfortably in the L1
cache of any modern processor. The L3 cache, at 56 megabytes per socket, could
hold twenty thousand such working sets.

Cache was not the bottleneck. NUMA was.

### The Experiment

Erlang provides the `+sbt` flag to control scheduler binding. There are several
strategies, each expressing a different opinion about where threads should live:

| Flag | Strategy | Philosophy |
|------|----------|-----------|
| `u` | Unbound (default) | Let the OS decide |
| `db` | Default bind | Bind schedulers to logical processors in order |
| `ts` | Thread spread | One scheduler per physical core, spread across topology |
| `tnnps` | Thread no-node processor spread | Spread across NUMA nodes, then processors |

We ran the same benchmark four times. The model is a d=8 Bayesian regime-switching
specification — eight free parameters, two hundred observations, five hundred
warmup iterations, five hundred sampling iterations. This is the exact model
running in production across 102 instruments.

The benchmark script (`benchmark/cpu_pinning_bench.exs`) measures three
scenarios: a single sequential job, ten concurrent jobs, and forty-four
concurrent jobs (matching the production pool size).

### The Results

```
Strategy    Sequential(ms)  10-concurrent(j/s)  44-concurrent(j/s)
--------    --------------  ------------------  ------------------
unbound     4915            0.64                0.73
db          4129            0.83                0.87
ts          3259            0.90                0.88
tnnps       3763            0.85                0.98
```

The unbound configuration, which we had been running in production for three
weeks, was the slowest in every category. Not by a little. At forty-four
concurrent jobs — our production workload — `tnnps` delivered 34% more
throughput than `unbound`. The sequential path showed an even starker gap:
`ts` was 34% faster per job than `unbound`.

The pattern reveals the mechanism. `ts` (thread spread) excels at sequential
work because it places each scheduler on a separate physical core, avoiding
hyperthreading contention. `tnnps` (NUMA-aware spread) excels at high concurrency
because it distributes schedulers evenly across both sockets, maximizing aggregate
memory bandwidth while keeping each scheduler close to its data.

### Why It Matters

On a dual-socket NUMA machine, an unbound scheduler thread can migrate between
sockets between consecutive function calls. When it does, every memory access
that was hitting L1 in 1 nanosecond now crosses the QPI interconnect at 40
nanoseconds. For a single tensor operation this is invisible. For five hundred
leapfrog steps, each touching the same gradient vector, it accumulates.

The EXLA JIT compiler makes this worse in a specific way. XLA compiles model
computations into native x86 code that expects its operands at particular memory
addresses. When the scheduler migrates, the code runs on a different socket but
the tensors are still in the first socket's memory. Every operand fetch is now a
remote NUMA access.

`+sbt tnnps` fixes this by pinning each scheduler to a specific logical
processor and distributing the pins evenly across the NUMA topology. Scheduler 0
stays on socket 0, core 0. Scheduler 22 stays on socket 1, core 0. Once pinned,
memory allocations from that scheduler land in local NUMA memory and stay there.

### The Production Impact

The capacity model with `+sbt tnnps`:

| Instruments | Wall time per round | Capacity at 20-min cycle |
|-------------|--------------------:|------------------------:|
| 100         | 1.4 min            | Yes                     |
| 500         | 7.2 min            | Yes                     |
| 1,000       | 14.4 min           | Yes                     |
| 1,400       | 20.1 min           | Barely                  |
| 2,000       | 28.7 min           | No (need 40-min cycle)  |

Without `+sbt tnnps`, the 1,000-instrument mark would require 19.2 minutes —
leaving almost no headroom. With it, 1,400 instruments fit in a 20-minute window
with room for variance.

### How to Use It

Add to your BEAM launch flags:

```bash
# Elixir
elixir --erl "+sbt tnnps" -S mix run my_app.exs

# iex
iex --erl "+sbt tnnps" -S mix

# In a shell script
ERL_FLAGS="+sbt tnnps"
exec elixir --erl "$ERL_FLAGS" --sname myapp -S mix
```

Verify in the running system:

```elixir
:erlang.system_info(:scheduler_bind_type)
#=> :thread_no_node_processor_spread
```

### Which Strategy to Choose

- **Single-socket machine** (most laptops, small servers): `+sbt ts` — spread
  across physical cores to avoid HT contention
- **Multi-socket NUMA** (Xeon, EPYC): `+sbt tnnps` — NUMA-aware distribution
- **Chiplet design** (Ryzen 3000+, EPYC): `+sbt tnnps` — CCDs are effectively
  NUMA nodes with shared L3
- **Unknown/embedded**: try `+sbt db` as a safe default

Run the benchmark (`benchmark/cpu_pinning_bench.exs`) on your hardware to
confirm. The relative ordering may differ, but `unbound` is almost never the
right answer for compute-heavy workloads.

### The Lesson

The BEAM is designed for I/O-bound workloads — web servers, message brokers,
telephony switches — where scheduler migration between cores is harmless because
the bottleneck is waiting for network packets, not computing gradients. When you
repurpose it for CPU-bound numerical computation, the assumptions embedded in its
defaults stop serving you.

Thirty-four percent is not a rounding error. It is a third of your capacity,
donated to the operating system's scheduling heuristics because nobody asked
the machine to do otherwise.

---

*Benchmark: `CUDA_VISIBLE_DEVICES="" mix run benchmark/cpu_pinning_bench.exs`*
*Compare strategies: `elixir --erl "+sbt tnnps" -S mix run benchmark/cpu_pinning_bench.exs`*

---

### Sources

**Erlang/OTP scheduler binding**:
- Erlang `erl` man page, `+sbt` flag: https://www.erlang.org/doc/apps/erts/erl_cmd.html
- Erlang Efficiency Guide, "System Limits and Tuning": https://www.erlang.org/doc/system/efficiency_guide.html
- Lukas Larsson, "Understanding the Erlang Scheduler" (Erlang User Conference 2013)

**NUMA architecture and effects**:
- Intel Xeon E5-2699 v4 datasheet, ARK: https://ark.intel.com/content/www/us/en/ark/products/91317/intel-xeon-processor-e5-2699-v4-55m-cache-2-20-ghz.html
- Lameter, C. "NUMA (Non-Uniform Memory Access): An Overview." *ACM Queue* 11(7), 2013. https://queue.acm.org/detail.cfm?id=2513149
- Drepper, U. "What Every Programmer Should Know About Memory." 2007. https://people.freebsd.org/~lstewart/articles/cpumemory.pdf (Section 5: NUMA)
- McCurley, J. "NUMA Deep Dive" series (VMware): https://frankdenneman.nl/2016/07/07/numa-deep-dive-part-1-uma-numa/

**BEAM on NUMA — prior art**:
- Lundin, K. "Scalable Erlang on Many-Core Processors." (Ericsson internal, referenced in Erlang/OTP source `erts/emulator/beam/erl_process.c`)
- Erlang source: `erts_sched_bind_atthd_prepare()` implements the binding strategies — see `erts/emulator/beam/erl_process.c` line ~12000
- WhatsApp engineering (2014): scaled BEAM to 2M connections per server using `+sbt` on NUMA hardware

**XLA/EXLA JIT compilation**:
- XLA (Accelerated Linear Algebra) documentation: https://www.tensorflow.org/xla
- EXLA Hex package: https://hex.pm/packages/exla
- XLA caches compiled HLO by graph hash; captured tensors become constants in the graph. See `xla/service/hlo_module.h`

**Benchmarking methodology**:
- All benchmarks run with `CUDA_VISIBLE_DEVICES=""` (CPU-only, no GPU contention)
- JIT cache warmed before measurement (first compilation excluded)
- `Task.async_stream` with explicit `max_concurrency` for parallel runs
- `:timer.tc/1` for wall-clock measurement
- Hardware: 2x Intel Xeon E5-2699 v4 @ 2.20GHz (44 cores / 88 threads), 256GB DDR4, NUMA 2-socket

---

### P.S. What Does "Thread No-Node Processor Spread" Actually Mean?

The name `tnnps` is an Erlang-style compound that reads as three layered
instructions, innermost to outermost:

| Component | Token | Meaning |
|-----------|-------|---------|
| **thread** | `t` | Spread across hardware threads first. Low-numbered schedulers get the first HT thread of each core; higher-numbered get the second. Prevents two schedulers from fighting over one core's execution units. |
| **no_node** | `nn` | Fill one NUMA node completely before crossing to the next. "No [crossing] node [boundaries]" — stay local. This is the NUMA-aware constraint. |
| **processor_spread** | `ps` | Within each NUMA node, spread across physical processors (cores) as widely as possible. Don't cluster schedulers on adjacent cores. |

The confusing part is "no_node." It sounds like "ignore NUMA nodes" but means
the opposite: *respect* node boundaries. The Erlang documentation confirms:
"schedulers are only spread over processors internally in one NUMA node at a
time." The name describes what the algorithm *avoids doing* — crossing node
boundaries — not what it ignores.

One further detail from the docs: **`db` (default_bind) currently maps to
`tnnps`**. The Erlang/OTP team already decided this is the best general-purpose
binding strategy for bound schedulers. They simply don't make it the
default-default, which remains `unbound` — because most BEAM workloads are
I/O-bound web servers where scheduler migration is harmless and binding would
reduce the OS scheduler's ability to balance load across cores responding to
bursty network traffic.

For compute-bound numerical work — gradient evaluations, leapfrog integrations,
tree building — the calculus inverts. Pinning wins because the cost of a NUMA
remote access (40ns) exceeds the benefit of OS load balancing (saving maybe 1ms
of queue imbalance across 88 schedulers that are all equally busy anyway).

Source: [Erlang `erl` command reference, `+sbt` flag](https://www.erlang.org/doc/apps/erts/erl_cmd.html)
