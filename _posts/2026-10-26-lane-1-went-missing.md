---
title: "Lane 1 Went Missing"
date: 2026-10-26
description: "How checking a formal GPU semantics against real hardware, bit for bit, turned up a small ptxas miscompile in shared-memory half2 atomics."
tags: [cuda, ptx, compilers, lean, formal-verification]
---

We weren't looking for a compiler bug. We were trying to write down, very precisely, what a GPU does. The GPU, by way of its compiler, disagreed with us. This time the compiler turned out to be the one that was wrong.

## The setup

We're building **KVA**, a Lean 4 library that gives a formal semantics to CUDA and PTX on NVIDIA Hopper (sm_90a). The idea is that GPU kernels can eventually be *proved* correct, not just tested. Much of the library is built by a team of AI coding agents, each owning one part: numerics, concurrency, data movement, tensor cores. (Yes, that's as fun as it sounds. Mostly.)

A semantics is only useful if it describes the real machine, so we keep ours honest with a **conformance harness**. Every value-level instruction the library defines, such as float adds, conversions and atomics, is run on an H200 and compared bit for bit against its Lean definition. Some are checked exhaustively, others over large random samples. If the GPU and the definition disagree on a single bit, the test fails.

## The symptom

One agent was adding semantics for atomics on **shared memory**. Its tests for packed half-precision adds, `atomicAdd(__half2*)` and `atomicAdd(__nv_bfloat162*)`, started failing, and not just occasionally: roughly 57–80% of results were wrong.

The failures had a pattern. A `__half2` packs two 16-bit lanes into 32 bits, and the wrong answers were *always* in the upper one, lane 1. The same operations on **global** memory passed.

## The investigation

The first suspect was, naturally, us. A brand-new Lean model of a fiddly instruction is a far likelier culprit than NVIDIA's toolchain.

So we read the PTX that nvcc generated. It was correct.

That left the step between PTX and machine code: `ptxas`, the optimising assembler. Recompiling with `-Xptxas -O0` made the failures disappear. With the same PTX and the same Lean model, the only change was turning off the optimiser.

From there it was a matter of shrinking. The failing test was cut down to a tiny reproducer in plain CUDA, with no inline assembly, which left nothing exotic to blame. A second agent re-ran it independently, and it reproduced.

## The bug

sm_90 has no native instruction for an atomic packed-half add on shared memory, so ptxas emulates it with a compare-and-swap loop. When the 32-bit operand happens to be **one half of a 64-bit value**, that loop reads the upper lane from the wrong register.

The trigger looks unusual, but it's perfectly legitimate code: load four halves as one 64-bit word, then add them into shared memory as two `__half2`s.

```cuda
__shared__ __half2 s[...];
unsigned long long v = x[t];                          // four halves in one 64-bit word
atomicAdd(&s[2*t],   as_half2((unsigned)v));          // low half2: upper lane comes out wrong
atomicAdd(&s[2*t+1], as_half2((unsigned)(v >> 32)));  // high half2
```

(`as_half2` just reinterprets 32 bits as a `__half2`.)

Here's a concrete case. Add the 64-bit value `0x123456783c004000` as two `__half2`s. Its low word is `0x3c004000`, which holds the halves 2.0 (lane 0) and 1.0 (lane 1).

| Low word | Lane 1 | Lane 0 |
|---|---|---|
| Expected: `0x3c004000` | `0x3c00` (1.0) | `0x4000` (2.0) |
| Got: `0x56784000` | `0x5678` (103.5) | `0x4000` (2.0) |

Lane 0 is right. Lane 1 holds `0x5678`, which looks suspiciously like the bottom half of the *high* word, `0x12345678`.

## Reproduce it yourself

You'll need a Hopper GPU (sm_90, e.g. H100 or H200) and CUDA 13.2. Download the reproducer, **[lane1_repro.cu](/assets/lane1/lane1_repro.cu)**, 68 lines of plain CUDA with no inline assembly. A single thread loads four halves (1.0, 2.0, 3.0, 4.0) as one 64-bit word and adds them, as two pairs, into zeroed memory. It does this once for shared memory and once for global memory, with both `__half2` and `__nv_bfloat162`. The right answer is just the four values back.

Build it twice, once with the default optimisation and once with ptxas optimisation turned off:

```bash
nvcc -arch=sm_90a -o lane1 lane1_repro.cu
nvcc -arch=sm_90a -Xptxas -O0 -o lane1_O0 lane1_repro.cu
./lane1       # default ptxas optimisation
./lane1_O0    # ptxas -O0
```

With the default build, the shared-memory results are wrong:

```text
__half2        expected 40003c00 44004200 | shared 42003c00 00004200 FAIL | global 40003c00 44004200 PASS
__nv_bfloat162 expected 40003f80 40804040 | shared 40403f80 00004040 FAIL | global 40003f80 40804040 PASS
MISCOMPILED: shared-memory result differs from the expected values
```

The `-O0` build prints `PASS` everywhere. In the low pair, the upper lane comes back as 3.0 (`0x4200`) instead of 2.0, which is a value borrowed from the *other* word. In the high pair, it comes back as 0 instead of 4.0. Global memory is right in both builds.

To see why, look at the machine code: `cuobjdump -sass lane1 | grep -E "HADD2|ATOMS"`. The 64-bit load lands in two registers, `R2` (low word) and `R3` (high word). Each compare-and-swap loop does one `HADD2` per lane. The low pair's upper-lane add reads `R3.H0` where it should read `R2.H1`. The high pair's upper-lane add reads `RZ`, the zero register:

```text
HADD2 R7, R8.H1_H1, R3.H0_H0      // low pair, upper lane: should be R2.H1_H1
HADD2 R9, R8.H0_H0, R2.H0_H0      // low pair, lower lane: correct
...
HADD2 R2, R6.H1_H1, RZ.H0_H0      // high pair, upper lane: should be R3.H1_H1
```

We saw exactly this output with nvcc 13.2 builds V13.2.51 and V13.2.86 (13.2 Update 2), for both `-arch=sm_90a` and `-arch=sm_90`, on an H200. Register numbers may differ on other toolchains.

## Who should care

Probably not many people. It's a narrow bug that needs an unusual-looking pattern to show up.

- **Affected:** nvcc 13.2 targeting sm_90 or sm_90a, running on an H200. We saw it in build V13.2.51 and in 13.2 Update 2 (V13.2.86), the latest 13.2 release.
- **Not affected:** builds with `-Xptxas -O0`, and global memory, which has a native instruction.
- **Workarounds:** don't feed a split 64-bit value straight into a shared-memory packed-half atomic, or compile that file with `-Xptxas -O0`.

Is it already known? As far as we can tell, no. We found no public report, and the [CUDA 13.2 Update 2 release notes](https://docs.nvidia.com/cuda/archive/13.2.2/pdf/CUDA_Toolkit_Release_Notes.pdf) don't list it. The closest thing we found is a [2024 NVIDIA forum thread](https://forums.developer.nvidia.com/t/half2-atomics-generate-unused-code/301822) that noticed this same compare-and-swap code path generating strange, unused code. Nobody there reported wrong results.

## The takeaway

A formal semantics is only as good as its contact with reality. Checking every definition against the hardware, one bit at a time, is what keeps ours honest. This time it also caught someone else's mistake. Ordinary testing would likely have missed this one, because it only appears in a pattern most tests never write.

It also cost us a little philosophical tidiness. KVA's stance has always been that we trust the compiler, and we still do, with a small asterisk now attached. The library keeps a "known miscompiles" list, and lane 1 is on it.
