# The Instrument Stopped Measuring

*This is Part 4 of "What If Probabilistic Programming Were Different?" Part 1 introduced the BEAM process runtime thesis. Part 2 showed feature parity and benchmark numbers. Part 3 told the story of seven optimization phases, from 34x slower than PyMC to 1.9x faster. This part is about the day we found out that three of those optimizations had not been running.*

---

For two weeks, one test in the Exmc suite failed on every machine we owned.

It was a timing test. It asserted that sampling four chains in a vectorized batch beat running them through the older parallel path. It failed on Linux with an NVIDIA Ampere card, on FreeBSD with two different Keplers, and on a Jetson Nano. It was written down in our handoff file as "item 4, the wall-clock one," in a list of known failures, with a note that timing assertions are host-dependent and this one was probably a scheduling artifact.

It was not a scheduling artifact. It was the only thing in the entire suite still watching a feature that had silently stopped working, and it had been telling us so, correctly, every single run, for fourteen days.

Here is what it was trying to say.

## Three functions, one discarded variable

Exmc compiles a model down to a fused f64 GPU shader that runs an entire leapfrog chain — the inner loop of NUTS — in a single dispatch. Getting there took most of Part 3. The compiled artifact is handed to the sampler as a tuple, and one element of that tuple is `chain_meta`: the handshake that tells the tree builder *there is a fused shader for this model, use it*.

The sampler has several entry points. Single-chain `sample/3`. A vectorized multi-chain path. A distributed path for running chains across machines. A streaming path that sends each draw as a message.

Three of the four destructured that tuple like this:

```elixir
{vag_fn, step_fn, pm, ncp_info, _chain_meta} = compiled
```

`_chain_meta`. Bound, underscored, thrown away. Never placed in the process dictionary. The tree builder checked for it, found nothing, and fell back to running the model as hundreds of individual tensor operations — which is correct, produces the right posterior, and is catastrophically slower.

We measured it once we knew. Same process, same model, concurrency removed as a confound:

| path | before | after | factor |
|---|---|---|---|
| vectorized (the **default** for `num_chains > 1`) | 7791 ms, 0 dispatches | 1353 ms, 1284 dispatches | **5.8x** |
| distributed | 3209 ms, 0 dispatches | 240 ms, 535 dispatches | **12.9x** |
| streaming | 1162 ms, 0 dispatches | 454 ms, 800 dispatches | **2.6x** |

The middle row is the one that stings. `sample_compiled_tuned` is what the distributed path calls on every worker node. Part 3 of this series is a long account of making the GPU path fast. On the code path you would use to run chains across a cluster, none of it had been running.

Note the second column. Not "slower" — **zero dispatches**. The shader was never invoked. The evidence was sitting in a counter the whole time, and nothing was reading it.

## Why the test could not do its job

The test was not wrong. It was unfalsifiable in practice, which is worse.

It asserted `t_vectorized < t_parallel`: one wall-clock duration against another, where the comparison path used BEAM concurrency and the machine under test might be a 5-watt Jetson or a 24-core workstation. Every engineer who looked at that failure had an available, plausible, non-alarming explanation — the box was busy, the scheduler was unlucky, timing tests are flaky. And every one of those explanations was *reasonable*. That is the trap. The failure carried no information that distinguished "your optimization is disabled" from "the CI box was noisy," so it got filed under noise and stayed there.

**A test that fails for a reason nobody believes is worse than no test.** No test at all leaves an obvious hole. A test like this one fills the hole with something that looks like coverage, fails, and trains everyone who sees it to ignore that part of the system.

The replacement does not measure time:

```elixir
Dispatch.reset_dispatch_count()
Sampler.sample_compiled_tuned(compiled, init, opts)
tuned = Dispatch.dispatch_count()

assert tuned > 0,
       "sample_compiled_tuned issued #{tuned} chain dispatches — " <>
         "chain_meta is being dropped on this path"
```

Exact, integer-valued, host-independent. It cannot be explained away by a busy machine, because a busy machine still dispatches. On a 5W Jetson and a 24-core workstation it asserts the identical thing.

It also has a companion assertion that most people would skip:

```elixir
assert seq_dispatches > 0,
       "the non-vectorized path issued no chain dispatches either -- " <>
         "this box may not dispatch at all, so the comparison above is vacuous"
```

That is the non-vacuity guard. Without it, a host with no working GPU would report zero dispatches everywhere, the comparison would be zero-versus-zero, and the test would pass by being uniformly broken. We have been bitten by exactly this before, which is why it is there.

## Every fix gets a negative control

The habit that came out of this: after writing the test, revert the fix and confirm the test fails *for the stated reason*.

Not "fails." Fails saying `0 chain dispatches`.

Each of the three fixes above was run against its own reverted code before being committed. It costs about ninety seconds per fix. It is the only way to know that the test you just wrote is attached to the thing you think it is attached to — and twice in one day it changed a conclusion that reading the code had gotten wrong.

Which brings us to the rest of the day, because once you start looking for instruments that have quietly stopped measuring, they are everywhere.

## Seven more, in one session

### 1. The count that stayed the same

We fixed the `chain_meta` bug and re-ran the fleet. On one FreeBSD Kepler:

```
before: 640 tests, 4 failures
after:  652 tests, 4 failures
```

Four before, four after. Read as a total, the fix did nothing on that host.

By name, the picture inverts. The wall-clock test went green — the fix worked. And an unrelated flaky test, which had failed on the *other* Kepler in the previous run, happened to land on this one this time. Two changes, canceling exactly.

Had we compared counts, we would have reported "the fix does not work on FreeBSD" and gone looking for a MoltenVK problem that does not exist. **Compare failures by name.** A flat count is not evidence of no change; it is the absence of evidence, which is a different thing that happens to look identical in a summary line.

### 2. The population that changed underneath

Then the Jetson, which auto-detects its compiler:

```
baseline: 640 tests, 8 failures
new run:  652 tests, 6 failures
```

Two failures fixed. Except our test helper flips the *exclusion set* along with the backend:

| arm | excludes |
|---|---|
| Vulkan | `:vulkan_known_failure` — 5 excluded, 2 skipped |
| anything else | `:requires_vulkan` — **31 excluded** |

The Jetson has EXLA installed. A bare `mix test` there resolves to EXLA, not Vulkan — unlike our Kepler boxes, where nothing else is available and the bare command happens to do the right thing. So the run silently switched arms, dropped 26 additional tests, and four of the "fixed" failures had simply not been executed.

This is the previous failure one layer deeper. There, the composition changed under a constant count. Here, the **population** changed — and comparing by name does not save you, because the two runs were not running the same tests. The only thing that catches it is the line the suite prints about itself:

```
exmc: compiler=EXLA (configured: nil) backend=EXLA.Backend precision=:f64
exmc: excluding [:diag, :slow, :requires_vulkan]
```

We print that line because a missing `LD_LIBRARY_PATH` once made a run silently use a different backend for a whole evening. Printing it was the right lesson. Reading it before comparing two numbers is the part we had to learn twice.

### 3. The check that could not run

A collaborating session, working on the GPU backend, cross-compiles an ARM64 shared library and verifies it before deploying. The verification called `file` to confirm the ELF architecture.

`file` is not installed in that build image.

The step printed `command not found`, the `readelf` lines after it succeeded, and the block came out looking green. The architecture bar — the single most important check, the one that catches deploying an x86 object to an ARM board — had never run at all.

**A check that cannot run must fail loudly, not fall through.** The output did not say `UNVERIFIED`. It said `command not found`, in the middle of a wall of passing output, which is functionally the same as saying nothing.

### 4. The mechanism that fit

We noticed our GPU dependency's suite time rise about 10% on both Keplers after a dependency bump, and traced it to a plausible mechanism: a change had turned buffer *uploads* into a device-local allocation plus a staging copy plus a fence, and our chain shader does three uploads per dispatch. The arithmetic worked. Roughly 4500 dispatches, three fences each, about 0.16 ms per fence — same order as the second we had observed. The direction was right. We reported it, upstream believed it, and they built a fix.

It was wrong. The upload path had *not* changed in the interval we measured — that change came one commit later. What changed was **downloads**:

```
2617e5e   download_buffer -> buf.read()        read in place. No submission.
                                               staging_read did not exist.
6b38aee   download_buffer -> staging_read      staging alloc + copy + fence.
```

And the chain shader downloads four buffers per dispatch — `q_chain`, `p_chain`, `grad_chain`, `logp_chain`. Zero fences to four, on the readback side, in exactly the window we measured. Our own design notes say the chain shader is download-dominated by construction: `3*K*d*8` bytes out against `2*d*8` in. The answer was in a document we wrote.

What makes this one instructive is that nothing in our own evidence would ever have dislodged it. The observation was real, the number was the right order of magnitude, the direction was correct, and it fit a story we had already told. It took someone reading the actual diff and saying *that function did not exist at that commit*. Plausibility is not a mechanism. If you cannot point at the lines that changed, you have a correlation and a story.

### 5. The A/B that was not

Then we measured the fix. On the noisy Kepler, n=5:

```
6b38aee:  4.0  4.1  4.1
ab2e779:  4.4  4.7  4.6  4.4  5.0
```

Slower after the fix. A clean-looking regression.

Except the first set ran after a full test suite, with the box warm and clocked up, and the second ran immediately after a CPU-saturating Rust rebuild. Two different machine states. And on FreeBSD, `nvidia-smi` reports `[N/A]` for `clocks.sm`, so there is no way to see how different.

The other session had already hit this from the other side: their first attempt at the same measurement was thrown out because process-to-process variance reached **34% on an identical binary**, and unrecorded GPU clock state alone swung one of their measurements by **2.6x**. Meanwhile the quiet Kepler returned `3.3 3.3 3.3 3.3 3.4` on the same code the noisy one gave `4.4` to `5.0`.

An A/B where the machine state differs between arms is not an A/B. We had changed two things — the code and what ran before it — and attributed the difference to one.

### 6. The machine state nobody checked

Which raises the question of what else about a host we were assuming.

The Jetson had, in every run we had ever done, produced a distinctive failure profile: nine failures, seven of them `ExUnit.TimeoutError` at 60-, 120- and 300-second thresholds. No other host looked like that. It got absorbed into the mental model as "the Jetson is slow, it times out, that is what Jetsons do."

Then somebody looked at `cpu2` and `cpu3` and found them offline.

```
NV Power Mode: 5W
< POWER_MODEL ID=1 NAME=5W >
CPU_ONLINE CORE_2 0
CPU_ONLINE CORE_3 0
CPU_A57 MAX_FREQ 918000        # MAXN: unlimited (1479 MHz)
GPU     MAX_FREQ 640000000     # MAXN: 921.6 MHz
```

Half the cores, 62% of the CPU clock, 67% of the GPU clock. And the config file's own declared default is `DEFAULT=0` — MAXN, all four cores. The board was in 5W because a persisted mode file said so, and had been for as long as anyone had been running tests on it.

Every Jetson number in our records — every baseline we compared against, every timeout we filed as a property of the hardware — was taken on a machine running at roughly a third of its capacity, and nothing in any of those records said so. The failures were real. The attribution was not.

### 7. The amplification that was not

The collaborator's advice, when we started measuring a sub-millisecond effect on hardware with no readable clock, was: amplify. Drive many operations per timed sample, because a 0.16 ms effect inside a 0.5 ms measurement will not survive.

We said our probe already did that. It runs eight tests, takes about four seconds, and drives — we said — roughly 4500 GPU dispatches. We passed that figure upstream. They sized their proposed mechanism against it: 4500 dispatches, four fences each, 18000 fences, about 2.9 seconds of predicted cost against the 1.0 second we had observed. Same order, close enough to be encouraging.

Then we measured the dispatch count instead of remembering it:

```
vectorized (4 chains, 50 warmup + 50 samples)     64
sequential (4 chains, same)                       64
sample_compiled (100 samples)                    126
sample_compiled_tuned (100 samples)              132
stream (50 warmup + 50 samples)                   89
                                        total   ~520
```

Not 4500. About 520. The figure had come from an earlier set of runs that used 500-sample workloads; the probe uses 50 and 100. It was quoted from memory, immediately after saying out loud that the number was load-bearing.

Redo the arithmetic and the discrepancy does not shrink — it inverts. 520 dispatches, four fences each, 0.16 ms per fence: **0.33 seconds predicted against 1.0 observed.** We had gone from "three times under prediction" to "three times over prediction" by correcting a number we had supplied ourselves.

The deeper problem is the one the correction exposes. Four seconds, 520 dispatches: most of that four seconds is BEAM startup, model compilation, GLSL synthesis, SPIR-V cache lookup and backend initialization. It is not dispatch. A sub-millisecond per-dispatch effect was sitting inside a measurement dominated by fixed overhead — which is the exact failure we had been warned about and had explicitly claimed to have avoided. The probe amplifies about nine times less than we said it did.

So the honest output of that measurement is not a number. It is: *this instrument cannot resolve this effect, and every per-dispatch figure derived from it — ours and the one upstream computed from ours — should be withdrawn.*

Being wrong about the measurement is ordinary. Supplying a wrong constant to someone else's calculation, after flagging that constant as the important one, is the part worth writing down. An estimate you have labelled load-bearing is the one place you are not allowed to estimate.

## What all eight have in common

In every case the instrument stopped measuring and **the report still looked like a result.**

That is the whole pattern. Not a crash, not a red X, not an error anyone had to dismiss. A number appeared, in the right format, in the right column, and it was not measuring what its label said. A wall-clock inequality that could not distinguish a disabled optimization from a busy machine. A failure count summing two changes to zero. A test population that shrank by 26. A verification step that printed `command not found` inside a green block. A mechanism that was arithmetically consistent with a code change that had not happened yet. A benchmark whose two arms ran on differently-clocked hardware. A host baseline that silently encoded a power mode. A per-operation cost computed against an operation count that was off by nine times.

The unifying rule, if there is one:

> **An instrument that can stop working must say so out loud, in a way that cannot be confused with a result.**

Green is not evidence. Green plus a negative control is evidence. And the specific practices that fall out of it:

- **Assert on things that cannot be explained away.** Counters, not durations. `dispatch_count() > 0` survives a noisy box; `t_a < t_b` does not.
- **Every test gets a negative control.** Revert the fix, confirm the test fails, confirm it fails *saying the right thing*. Ninety seconds.
- **Every comparison gets a non-vacuity guard.** If both arms can be zero, assert that they are not.
- **Compare by name, not by count** — and before that, confirm both runs ran the same population.
- **Make the run describe itself.** Backend, precision, exclusions, host, power mode, clock state. Then read it.
- **If you cannot point at the diff, you have a story.** A mechanism is lines of code, not arithmetic that comes out right.
- **Two changed things is not an A/B.** Machine state is one of the things.
- **Measure your constants, especially the load-bearing ones.** The moment you say "this number matters" is the moment you are no longer allowed to quote it from memory.

None of this is specific to probabilistic programming. But it lands hard here, because a PPL fails quietly by design: a sampler with a disabled optimization returns the same posterior, just slower, and a sampler with a genuine numerical bug returns a plausible one. There is no exception, no stack trace, no crash. There is a bag of numbers, and it looks exactly like a correct bag of numbers.

Which is why the most useful thing in the whole suite is not a test that passes. It is a test that has been *shown to fail* when the thing it watches is broken.

---

*Exmc is a Bayesian inference library for Elixir — NUTS on the BEAM, with EXLA and Vulkan backends. The dispatch-count guards described here are in `test/exmc/nuts/vulkan/chain_meta_routing_test.exs`.*
