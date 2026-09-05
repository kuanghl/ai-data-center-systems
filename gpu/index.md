# GPU Systems

A learning track for GPU architecture and CUDA kernel programming. The goal is
to learn how GPUs actually run kernels and why real inference workloads are fast
or slow.

This track is a **hub**: it aggregates a few external courses plus the reference
and lab material that supports them. Each folder plays one role — you either
*learn* it (a course), *look it up* (`appendix/`), or *practice* it (`labs/`).

## Layout

| Folder | Role | Contents |
|---|---|---|
| [`pmpp/`](pmpp/) | Course — **primary spine** | Programming Massively Parallel Processors |
| [`cs149/`](cs149/) | Course — systems foundation | Stanford CS149 Parallel Computing |
| [`heterogeneous-systems/`](heterogeneous-systems/) | Course — complement | Onur Mutlu parallel-pattern course |
| [`gpu-mode/`](gpu-mode/) | Course — advanced feed | GPU MODE modern ML kernels |
| [`articles/`](articles/) | Articles | Kubernetes GPU sharing and platform notes |
| [`appendix/`](appendix/) | Reference | [architecture](appendix/architecture/), [CUDA](appendix/cuda/) docs |
| [`labs/`](labs/) | Practice | [profiling](labs/profiling/), [kernels](labs/kernels/) |

This split also resolves the CUDA overlap: CUDA-the-course lives in
[`pmpp/`](pmpp/), CUDA-the-reference lives in
[`appendix/cuda/`](appendix/cuda/), and CUDA-the-practice lives in
[`labs/kernels/`](labs/kernels/).

## Suggested path

1. **Spine:** work through [PMPP](pmpp/) as the primary course.
2. **Systems foundation:** use [Stanford CS149](cs149/) to connect
   CUDA kernels with CPU parallelism, scheduling, locality, synchronization,
   memory models, and DNN execution.
3. **Complement:** use [Heterogeneous Systems (Onur Mutlu)](heterogeneous-systems/)
   in parallel for a second pass over the classic parallel patterns.
4. **Practice as you go:** for each concept, implement and profile a real kernel
   in [`labs/`](labs/) and confirm the behavior with
   [Nsight Compute](labs/profiling/).
5. **Reference on demand:** consult [`appendix/`](appendix/) for the exact NVIDIA
   architecture and CUDA API details.
6. **Later / advanced:** track [GPU MODE](gpu-mode/) as an ongoing feed
   for modern ML kernels.
