# Lecture 1: Why Parallelism? Why Efficiency?

Source: [Stanford CS149 2023 Lecture 1](https://www.youtube.com/watch?v=V1tINV2-9p4)

Course materials:

* [CS149 Fall 2023 Lecture 1 course page](https://gfxcourses.stanford.edu/cs149/fall23/lecture/whyparallelism/)
* [Lecture 1 slides PDF](https://gfxcourses.stanford.edu/cs149/fall23content/media/whyparallelism/01_whyparallelism_huXfOJ4.pdf)

## Table of Contents

* [Goal](#goal)
* [Lecture Overview](#lecture-overview)
* [Speedup](#speedup)
* [Efficiency](#efficiency)
* [The Three Course Themes](#the-three-course-themes)
* [Why Parallelism Became Necessary](#why-parallelism-became-necessary)
* [Program as Instructions](#program-as-instructions)
* [Instruction-Level Parallelism](#instruction-level-parallelism)
* [Memory and Locality](#memory-and-locality)
* [GPU Systems Lens](#gpu-systems-lens)
* [Practical Tips and Notes](#practical-tips-and-notes)
* [Lecture Summary](#lecture-summary)
* [Key Terms](#key-terms)
* [Questions](#questions)
* [Answers](#answers)

---

## Goal

The goal of this lecture is to understand why parallelism in modern systems is not an option but a baseline requirement, and why simply "running things in parallel" is not enough.

The core message is as follows.

> The goal of parallel computing is not to use more processors for its own sake, but to solve problems faster and more efficiently. Speedup matters, but if hardware is used inefficiently because of communication, synchronization, work imbalance, or memory movement, parallelism is easily wasted.

This lecture covers the following:

* The basic definition of a parallel computer
* The meaning and limits of speedup
* How communication cost and work imbalance limit parallel performance
* Why work decomposition, assignment, and synchronization matter in parallel programming
* Why single-thread CPU performance improvements no longer come for free as they used to
* The perspective that a processor executes a program as an instruction stream
* The basics of instruction-level parallelism, clock frequency, memory hierarchy, and cache locality
* Why efficiency should be considered first when studying GPUs and accelerators

---

## Lecture Overview

The lecture starts from the definition that "a parallel computer is a computer in which multiple processing elements cooperate to solve a problem quickly." Two words matter here. More important than using many processors is that the problem must be solved quickly and the hardware used efficiently.

The early demo treats students as processors and has them compute a sum of numbers. One person adding all the numbers takes a long time. Putting in two, four, or more people may look faster on the surface, but the cost of communication and coordination soon becomes visible. Partial sums must be shared, someone finishes early and waits, and if too many people participate, the cost of passing results around exceeds the cost of the computation itself. This demo shows a problem that repeats verbatim later in GPU kernels.

The middle section establishes the course themes. A good parallel program divides work safely, assigns it well to processors, and prevents communication and synchronization from eating the speedup. At the same time, you have to know the hardware. Without cache, memory bandwidth, latency, and processor structure, you cannot explain why the same algorithm is slow on a real machine.

The later section is historical context. In the past, single-thread CPU performance improved rapidly, so programs often got faster on the next generation of CPU without parallelizing the code. But as it became hard to keep raising performance with clock frequency scaling and instruction-level parallelism alone, the center of performance improvement shifted to multi-core, SIMD, GPUs, and domain-specific accelerators. Therefore, programmers now have to understand parallelism and machine efficiency directly.

Finally, the lecture introduces the perspective that a program is a list of instructions from the processor's point of view. An instruction performs arithmetic, reads and writes values from registers and memory, and changes control flow via branches. Here, performance is determined not just by the number of operations but greatly by memory access latency, cache hits/misses, and locality.

---

## Speedup

The most direct goal of parallel processing is speedup.

```text
speedup(P processors) = execution time using 1 processor
                      / execution time using P processors
```

For example, if a job that takes 40 seconds on one processor finishes in 10 seconds on four processors, the speedup is 4. Ideally, increasing the number of processors by a factor of `P` reduces execution time to `1/P`. But as the lecture demo shows, real speedup is usually smaller than that.

| Limiting factor | What happens | GPU systems lens |
| --------------- | ------------ | ---------------- |
| Communication | Partial results must be exchanged | Global memory traffic, inter-SM communication, distributed training all-reduce |
| Synchronization | Workers wait for others | Barriers, kernel boundaries, stream dependencies |
| Work imbalance | Some workers idle early | Irregular kernels, sparse workloads, dynamic batching |
| Overhead | Coordination consumes useful time | Kernel launch overhead, scheduling overhead, framework overhead |
| Locality | Data is far from the processor | Cache miss, HBM traffic, PCIe/NVLink movement |

Speedup tells you "how much faster it got," but it does not tell you "how well the machine was used." If you get a 2x speedup with 10 processors, the program got faster, but most of the available hardware may have been idle or spent on overhead.

## Efficiency

An important distinction repeated in the lecture is between fast and efficient.

```text
efficiency(P processors) = speedup(P) / P
```

By this formula, a 2x speedup on 10 processors is 20% efficiency. Conversely, a 3.6x speedup on 4 processors is 90% efficiency. Using more processors is not always the better choice.

Efficiency matters from two perspectives.

| Perspective | Question |
| ----------- | -------- |
| Programmer | How well is the given machine capability being utilized? |
| Hardware designer | Which capabilities should be built in, balancing performance, silicon area, power, and cost? |

On GPUs, this distinction is especially important. Even if a kernel is faster than the CPU, from the whole-GPU perspective it may be using only a little memory bandwidth while most SMs sit idle. Conversely, even if there are many operations with long latency, sufficient thread-level parallelism and good locality can produce high throughput.

## The Three Course Themes

Lecture 1 organizes all of CS149 into three themes.

| Theme | Meaning | Later CS149 topics |
| ----- | ------- | ------------------ |
| Writing scalable parallel programs | Divide and assign work, and manage communication/synchronization | Data parallelism, scheduling, task graphs, CUDA |
| Understanding parallel hardware | Understand how abstractions are implemented in hardware | Multi-core CPU, SIMD, GPU, cache coherence |
| Thinking about efficiency | Measure and judge that faster is not the same as efficient | Locality, bandwidth, memory models, DNN execution |

In the GPU track of this repository, the third theme is the most practical. Knowing CUDA syntax alone is not enough. You must be able to explain why a kernel is memory-bound, why occupancy is low, and why data movement dominates total latency.

## Why Parallelism Became Necessary

The lecture asks the question "why could we avoid parallel processing in the past?" In the past, single-thread CPU performance grew rapidly. Software developers often did not have to parallelize their code, and programs naturally got faster on the next generation of CPU.

Two things drove much of that performance improvement:

1. Superscalar execution that exploits instruction-level parallelism
2. Increasing CPU clock frequency

But that path hit its limits. Raising the clock indefinitely makes power and heat problems bigger, and the amount of independent work that can be automatically extracted from a single instruction stream is limited. So modern performance improvement has moved in the direction of leveraging explicit parallel resources such as multiple cores, SIMD lanes, hardware threads, GPU SMs, tensor cores, and accelerators.

Practically, this means the following:

* You have to find whether there is parallel work inside the algorithm.
* Even if there is parallel work, you have to reduce communication and synchronization.
* You should first suspect that data movement is more expensive than computation.
* You have to change the program structure to fit the form of parallelism the hardware provides.

## Program as Instructions

From the processor's perspective, a program is a sequence of instructions. C/C++ source code is turned into machine instructions through the compiler. The processor fetches, decodes, and executes instructions, and updates register and memory state.

An instruction roughly does the following:

| Instruction kind | Example role |
| ---------------- | ------------ |
| Arithmetic | Add, multiply, compare |
| Memory access | Load from address, store to address |
| Control flow | Branch, jump, call, return |

This perspective matters because the unit of parallelism is not source code lines but actual work and dependencies. A loop can be parallelized if there are no dependencies between iterations. Conversely, even if the source code looks simple, if a memory load stalls for a long time, the processor cannot use its arithmetic units properly.

## Instruction-Level Parallelism

Instruction-level parallelism, or ILP, is a way of executing independent instructions in one instruction stream simultaneously or overlapping. Superscalar CPUs have multiple execution units and try to issue multiple instructions in one cycle if there are no dependencies.

For example, the following two operations can execute at the same time if they are independent of each other.

```c
a = b + c;
x = y * z;
```

But the following has a dependency.

```c
a = b + c;
x = a * z;
```

The second instruction needs the result `a` of the first, so it cannot execute first. The CPU finds possible ILP within such dependencies, but the parallelism automatically obtainable from a single thread is limited. That is why more explicit parallel execution models like multi-core, SIMD, and GPUs are needed.

## Memory and Locality

The latter half of Lecture 1 builds intuition for the memory hierarchy. Programmers see memory as one linear address space, but the real machine moves data through multiple layers such as registers, L1/L2/L3 caches, and DRAM.

The lecture slides divide cache locality into two kinds.

| Locality | Meaning | Example |
| -------- | ------- | ------- |
| Spatial locality | Likelihood that a nearby address will be used soon | Contiguous array scan |
| Temporal locality | Likelihood that the same address will be used again | Reusing a loaded value |

Caches reduce memory access latency when locality exists. But if there is no locality, or the working set is larger than the cache, cache misses increase and the processor stalls waiting for DRAM access.

In the Kaby Lake example on the lecture slides, latency varies greatly depending on where the data is.

| Data location | Approximate latency in cycles |
| ------------- | ----------------------------- |
| L1 cache | 4 |
| L2 cache | 12 |
| L3 cache | 38 |
| DRAM, best case | ~248 |

What matters more than the numbers themselves is the ratio. No matter how many arithmetic units there are, if data does not arrive in time, execution stops. The same problem appears in another form on GPUs. Even if HBM bandwidth is high, uncoalesced access, low arithmetic intensity, and repeated host-device transfers can greatly lower actual throughput.

## GPU Systems Lens

Lecture 1 does not yet cover CUDA syntax, but it already provides the criteria for understanding GPU performance.

| CS149 Lecture 1 concept | GPU/CUDA interpretation |
| ----------------------- | ----------------------- |
| Work decomposition | How work is divided into threads, blocks, and grids |
| Work assignment | Blocks to SMs, warps to schedulers |
| Communication cost | Global memory, shared memory, atomics, collectives |
| Synchronization | `__syncthreads()`, kernel launch boundaries, stream ordering |
| Work imbalance | Divergent branches, irregular loops, variable sequence lengths |
| Locality | Coalescing, shared-memory tiling, cache reuse |
| Efficiency | Occupancy, achieved bandwidth, SM utilization, Tensor Core utilization |

The same questions can be asked in LLM inference and training.

* Does GEMM have sufficient arithmetic intensity?
* In attention, which is the bottleneck, memory movement or synchronization?
* How do batch size and sequence length change GPU utilization?
* How much communication and memory traffic does kernel fusion reduce?
* In distributed training, does all-reduce cost limit scaling efficiency?

## Practical Tips and Notes

### Look at the resources used together with speedup

In performance experiments, do not record only wall-clock time; also write down the scale of hardware used. For example, "it got 2x faster" means something completely different depending on whether 2 CPU cores or 8 GPUs were used.

| Record together | Why it matters |
| --------------- | -------------- |
| Runtime | User-visible performance |
| Processor/GPU count | Resource cost |
| Utilization | Whether hardware was actually busy |
| Memory bandwidth | Whether data movement is the bottleneck |
| Synchronization time | Whether waiting dominates useful work |

### Communication is not a problem to optimize after computation

In the lecture demo, communication limits speedup from the start. It is the same on GPUs. When writing a kernel, you must look at "what each thread computes" and "what data each thread reads from where and writes to where" at the same time.

### Locality is a property of the algorithm

A cache hit looks like an optimization the hardware does automatically, but locality is created by the program. Contiguous access, data reuse, tiling, and fusion are all program-level decisions to raise locality.

### The fast path and the efficient path can differ

For small inputs, the CPU can be faster than the GPU. It is not that the GPU is slow; launch overhead, transfer overhead, and low occupancy can be larger than the useful work. Conversely, for large inputs, the GPU can be overwhelmingly faster even with the same algorithm. Always look at problem size and overhead together.

## Lecture Summary

The conclusion of Lecture 1 is as follows.

* Single-thread performance no longer improves as fast as before.
* To get a large performance gain, you must use multiple processing elements or specialized hardware.
* Parallel programs are hard because of work partitioning, communication, and synchronization.
* Without knowing hardware characteristics, it is easy to misidentify the bottleneck.
* In particular, data movement and locality are the core of modern parallel computing.
* GPU programming should start from reasoning about efficiency, not memorizing the CUDA API.

## Key Terms

| Term | Meaning |
| ---- | ------- |
| Parallel computer | A computer in which multiple processing elements cooperate to solve a problem quickly |
| Speedup | The ratio of execution time with 1 processor to execution time with P processors |
| Efficiency | Speedup divided by the number of processors used |
| Communication | The cost of passing data/results between processors or workers |
| Synchronization | The cost of waiting to align the progress of multiple workers |
| Work imbalance | A state where some workers are idle while only some workers keep working |
| Instruction-level parallelism | Parallelism that overlaps independent instructions within one instruction stream |
| Superscalar execution | A CPU execution scheme that tries to issue multiple instructions in one cycle |
| Memory hierarchy | Multiple latency/capacity layers such as registers, caches, and DRAM |
| Spatial locality | The tendency to access nearby memory addresses consecutively |
| Temporal locality | The tendency to access the same data repeatedly |

## Questions

1. If speedup is 2x, can we always say it is good parallelization?
2. If you get a 2x speedup with 10 processors, what is the efficiency?
3. In the lecture demo, why is speedup limited even when the number of processors is increased?
4. Why could many software developers postpone parallel programming in the past?
5. When single-thread performance scaling slows down, what responsibility does the programmer take on?
6. Why should a program be seen as a list of instructions from the processor's perspective?
7. In what respect does ILP differ from multi-core parallelism?
8. What is the difference between spatial locality and temporal locality?
9. Under what conditions does a cache reduce memory latency?
10. Why can a GPU kernel be inefficient even if it is faster than the CPU?

## Answers

1. No. You must look at the number of processors used and the efficiency together.
2. `2 / 10 = 0.2`, i.e., 20%.
3. Because of passing partial results, synchronization, work imbalance, and coordination overhead.
4. Because CPU clock frequency and single-thread performance improved rapidly, programs often got faster without changing the code.
5. You must parallelize the work, reduce data movement, and structure the program to fit the hardware characteristics.
6. Because actual dependencies, memory accesses, branches, and arithmetic operations determine performance and parallelization potential.
7. ILP is parallelism that hardware finds inside the instruction stream of one thread, and multi-core parallelism is parallelism that explicitly divides work across multiple processing elements.
8. Spatial locality is the property of using nearby addresses together, and temporal locality is the property of using the same data again.
9. When the access pattern has locality and the working set is effectively reused within the cache hierarchy.
10. Because kernel launch, data transfer, memory bandwidth, low occupancy, and synchronization overhead can prevent most of the hardware from being used.
