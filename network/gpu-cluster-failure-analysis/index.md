# GPU Cluster Failure Analysis: How ECC, Xid, RDMA, and NCCL Hangs Connect

## Table of Contents

- [Summary](#summary)
- [1. Not All ECC Errors Are the Same Failure](#1-not-all-ecc-errors-are-the-same-failure)
- [2. Xid Is the Most Important Clue in GPU Failures](#2-xid-is-the-most-important-clue-in-gpu-failures)
- [3. Cautions When Reading nvidia-smi ECC Counters](#3-cautions-when-reading-nvidia-smi-ecc-counters)
- [4. Row Remapping and RAS Repair](#4-row-remapping-and-ras-repair)
- [5. NCCL Hangs Are Not Always Network Problems](#5-nccl-hangs-are-not-always-network-problems)
- [6. NVLink and PCIe Must Be Checked Together](#6-nvlink-and-pcie-must-be-checked-together)
- [7. DCGM Is a Good Baseline for Operational Diagnosis](#7-dcgm-is-a-good-baseline-for-operational-diagnosis)
- [8. Operational Decision Criteria: When to Drain and When to Consider RMA](#8-operational-decision-criteria-when-to-drain-and-when-to-consider-rma)
- [9. Failure Response Playbook](#9-failure-response-playbook)
- [10. Conclusion](#10-conclusion)
- [References](#references)

## Summary

When operating a large GPU cluster, there are cases where something looks like a network failure on the surface but the actual cause starts from the GPU RAS, ECC, PCIe, or NVLink side.

Typical symptoms are as follows.

```bash
NCCL watchdog timeout
CUDA error: uncorrectable ECC error encountered
NVRM: Xid
mlx5 CQE error
GPU has fallen off the bus
NVLink error counter increase
```

Seeing logs like these, it is easy to first suspect network problems such as NCCL, RDMA, InfiniBand, RoCE, MTU, PFC, or ECN. But in a GPU cluster, a failure does not stay in one layer.

When a GPU memory error occurs, the CUDA context can break, the NCCL communicator that the affected process was part of fails, and the entire collective operation can appear to have stopped. As a result, the operator feels that "the network stopped", but the actual starting point may be a GPU ECC or Xid event.

This article organizes GPU failures into the following flow.

```text
ECC / RAS Event
  ↓
Xid occurrence
  ↓
CUDA context reset or application abort
  ↓
NCCL communicator failure
  ↓
timeout or error increase at the RDMA / NVLink / PCIe layer
  ↓
distributed training job hang or node drain required
```

The core principle is simple.

```text
Do not look only at NCCL logs.
Do not look only at RDMA counters.
Do not look only at nvidia-smi output.
Tie Xid, ECC, row remap, NVLink, PCIe, DCGM, and scheduler events together into one timeline.
```

## 1. Not All ECC Errors Are the Same Failure

ECC stands for Error Correcting Code. It is a feature for detecting, or partially correcting, bit flips or memory cell errors in GPU memory.

From an operational viewpoint, ECC errors should be viewed in three broad categories.

| Category | Meaning | Operational judgment |
| --- | --- | --- |
| Correctable ECC | the error was detected and corrected | check whether it is a one-off or an increasing trend |
| Uncorrectable Contained ECC | correction is impossible but the blast radius is contained | restart the affected application, consider a GPU reset |
| Uncorrectable Uncontained ECC | the error's impact is not contained | drain the node immediately, consider a GPU reset or RMA |

The important point is that you should not conclude a GPU replacement from the fact that "there is an ECC error". Conversely, you should not dismiss it with "it's correctable, so it's fine" either.

In GPU operations, the following items matter more than absolute values.

```text
- Does it repeat on the same GPU?
- Is the volatile counter increasing quickly?
- Did an uncorrectable error occur?
- Is row remap in pending / failure state?
- Is a hardware failure reproduced in the DCGM diagnostic?
- Does it repeat at the same job, same rank, same GPU index?
```

## 2. Xid Is the Most Important Clue in GPU Failures

In NVIDIA GPU failure analysis, Xid must be checked. Xid is a GPU fault code recorded by the NVIDIA driver, and it can be found in `dmesg` or the system log.

```bash
dmesg -T | egrep -i 'xid|nvrm|ecc|pcie|aer'
journalctl -k | egrep -i 'xid|nvrm|ecc|pcie|aer'
```

The following are representative Xid examples frequently seen in operations. The exact meaning and response can vary depending on the driver, GPU generation, whether MIG is in use, and the Fabric Manager configuration, so the official Xid catalog and GPU debug guidelines should be checked together.

| Xid | Meaning | Operational interpretation |
| --- | --- | --- |
| Xid 31 | GPU memory page fault | possible kernel, driver, or application memory access problem |
| Xid 43 | GPU stopped processing | possible application crash, driver fault, or context failure |
| Xid 48 | Uncorrectable ECC error | treat as a serious ECC event, isolation required |
| Xid 63 | Row remap entry recorded | row remap recorded successfully, check reset/repair state |
| Xid 64 | Row remap entry recording failed | row remap recording failed, consider as an RMA candidate |
| Xid 79 | GPU has fallen off the bus | possible PCIe, power, hardware, or link problem |
| Xid 94 | Contained memory error | containment at the application level possible, restart the affected application |
| Xid 95 | Uncontained memory error | GPU reset required, diagnose after node drain |
| Xid 140 | ECC unrecovered error | unrecovered ECC-family event that may be reported separately in recent drivers |

In particular, Xid 94 and Xid 95 must be distinguished.

Xid 94 is a contained error. That is, an error occurred but the GPU driver has contained the blast radius to a specific application. Generally the affected application or job may fail, but other applications running on the same GPU are not necessarily affected. Operationally, restart the failed job and consider a GPU reset at a convenient time.

By contrast, Xid 95 is an uncontained error. It means the error's impact was not stably contained, so it must be viewed much more seriously. In this case, node drain, GPU reset, DCGM diagnosis, and RMA consideration on recurrence are needed.

So it is safer to write operational documentation like the following.

```text
Xid 94: memory error where application-level containment is possible. Restart the job and check GPU state.
Xid 95: uncontained memory error. After node drain: GPU reset / DCGM diagnosis / RMA consideration.
```

## 3. Cautions When Reading nvidia-smi ECC Counters

GPU ECC state can be checked with `nvidia-smi -q -d ECC`.

```bash
nvidia-smi -q -d ECC
```

The row remapper and repair state should also be checked together.

```bash
nvidia-smi -q | egrep -A40 'ECC Errors|Row Remapper|Remapped Rows|Repair|Pending'
```

The important concepts here are the volatile counter and the aggregate counter.

| Counter | Meaning |
| --- | --- |
| Volatile ECC counter | number of ECC errors since driver load |
| Aggregate ECC counter | number of ECC errors accumulated over the GPU lifetime |

What operators must be careful about is that the aggregate counter should not simply be interpreted as "the current failure state". The aggregate counter is a long-term accumulated value, and whether it can be cleared can be limited by GPU generation and driver policy.

So just because the aggregate counter is not 0 does not mean the GPU should immediately be judged faulty. Instead, check the following together.

```text
- Is the volatile counter increasing?
- Is it an uncorrectable error?
- Did a row remap occur?
- Is it in pending repair state?
- Is there a remapping failure?
- Does it repeat on the same GPU?
- Is it reproduced in the DCGM diagnostic?
```

In short, ECC counters should be viewed not as "a single metric that decides replacement" but as "a signal that starts further diagnosis".

## 4. Row Remapping and RAS Repair

Since the A100-generation GPUs, NVIDIA datacenter GPUs have had strengthened memory error recovery features. The representative features are dynamic page offlining, row remapping, and RAS repair.

Row remapping is a feature that replaces a faulty HBM row with a spare row. From an operational viewpoint, the following state must be checked.

```bash
nvidia-smi -q | egrep -A20 'Row Remapper|Remapped Rows|Pending|Failure'
```

The items to check are the following.

```text
- Remapped Rows
- Pending Row Remapping
- Row Remapping Failure
- Uncorrectable ECC
- Repair Pending
```

More caution is needed than a one-off correctable ECC in the following cases.

```text
- uncorrectable ECC occurred
- row remap pending state persists
- remapping failure occurred
- Xid 48, 63, 64, 94, 95 repeat on the same GPU
- DCGM diagnostic failure
```

In these cases, the GPU should not keep being used; it should be drained in the scheduler and then diagnosed.

In a Kubernetes environment, an operational procedure such as cordon/drain of the node is needed, like the following.

```bash
kubectl cordon <gpu-node>
kubectl drain <gpu-node> --ignore-daemonsets --delete-emptydir-data
```

In a Slurm environment, switch the node to the drain state.

```bash
scontrol update NodeName=<gpu-node> State=DRAIN Reason="GPU ECC/Xid investigation"
```

## 5. NCCL Hangs Are Not Always Network Problems

When an NCCL hang occurs in distributed training, the network is usually checked first.

```bash
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,NET,COLL
```

Of course, RDMA, InfiniBand, and RoCE configuration problems are very common.

The items to check are the following.

```text
- MTU mismatch
- PFC misconfiguration
- ECN misconfiguration
- RoCE GID index mismatch
- mlx5 port error increase
- switch buffer drops
- PCIe AER errors
- NIC firmware / driver mismatch
```

But when a GPU ECC or Xid event occurs, from NCCL's perspective it can look as if a specific rank suddenly disappeared. In that case, the other ranks wait in the collective operation until a timeout occurs.

That is, the following pattern is possible.

```text
GPU ECC error
  ↓
CUDA context failure
  ↓
affected rank aborts
  ↓
NCCL communicator failure
  ↓
watchdog timeout on the other ranks
  ↓
operator perceives it as an NCCL/RDMA failure
```

So when analyzing an NCCL hang, network logs alone are not enough.

At minimum, the following four things must be viewed together.

```bash
# GPU driver / Xid
dmesg -T | egrep -i 'xid|nvrm|ecc'

# GPU state
nvidia-smi -q -d ECC
nvidia-smi -q | egrep -A30 'Row Remapper|Repair|NVLink'

# NIC / RDMA state
dmesg -T | egrep -i 'mlx5|rdma|infiniband|roce|cqe|aer'

# NCCL logs
grep -iE 'nccl|watchdog|timeout|cuda error|unhandled' <job-log>
```

## 6. NVLink and PCIe Must Be Checked Together

Inter-GPU communication involves NVLink, PCIe, NICs, and the switch fabric together. So when looking at a GPU failure, NVLink counters and PCIe errors must also be checked.

```bash
nvidia-smi topo -m
nvidia-smi nvlink --status
nvidia-smi nvlink --errorcounters
```

On the PCIe side, check AER errors in the kernel log.

```bash
dmesg -T | egrep -i 'pcie|aer|fallen off|xid 79'
```

When Xid 79 is seen, it means the GPU disappeared from the PCIe bus, so it is not a simple application problem — hardware, riser, cable, power, PCIe link, motherboard, and thermal issues all need to be examined.

## 7. DCGM Is a Good Baseline for Operational Diagnosis

NVIDIA DCGM is a diagnosis tool close to the de facto standard in datacenter GPU operations. When a failure is suspected, use `dcgmi diag`.

```bash
# basic diagnosis
dcgmi diag -r 1

# a longer diagnosis
dcgmi diag -r 2

# post-mortem diagnosis after a failure
dcgmi diag -r 3

# administrator-led detailed diagnosis
dcgmi diag -r 4
```

The operational policy should be organized like the following.

```text
Level 1: deployment/basic state check
Level 2: quick hardware sanity check after a job failure
Level 3: post-mortem diagnosis after a failure
Level 4: detailed diagnosis after drain
```

In Kubernetes, it is good to have a structure where GPU telemetry is collected into Prometheus via the DCGM Exporter and alerts are raised under specific conditions.

For example, monitor the following metrics.

```text
- ECC correctable / uncorrectable errors
- Xid events
- retired pages
- row remap state
- NVLink error counters
- PCIe replay / AER errors
- GPU temperature
- power violations
- throttling reasons
```

## 8. Operational Decision Criteria: When to Drain and When to Consider RMA

Operational criteria are a problem if too aggressive, and a problem if too loose.

The following is an example of conservative operational criteria.

| Situation | Action |
| --- | --- |
| one-off correctable ECC | observe, check whether the counter is increasing |
| repeatedly increasing correctable ECC | strengthen node observation, DCGM diagnosis |
| Xid 94 | handle the job failure, consider a GPU reset or node drain |
| Xid 95 | immediately drain the node, GPU reset, DCGM diagnosis |
| Xid 48 | treat as a serious ECC event, drain recommended |
| row remap pending | secure a GPU reset or maintenance window |
| row remapping failure | RMA candidate |
| Xid 79 | treat as a PCIe/hardware problem, isolate immediately |
| DCGM diag fail | treat as a hardware issue, consider replacement/RMA |
| repeated failures on the same GPU | exclude from the scheduler, then open a vendor case |

The important principles are the following.

```text
Repetition matters more than a single counter.
Uncorrectable matters more than correctable.
The volatile increasing trend matters more than the aggregate.
The combination of Xid + ECC + DCGM + job failure matters more than a single Xid.
```

## 9. Failure Response Playbook

In actual operations, you can proceed in the following order.

### Step 1. Check the Job Failure Point

```bash
kubectl logs <pod> -n <namespace>
# or
check sacct / scontrol / slurm job logs
```

Keywords to check:

```text
NCCL timeout
CUDA error
uncorrectable ECC
illegal memory access
watchdog
rank failed
```

### Step 2. Check GPU Xid

```bash
dmesg -T | egrep -i 'xid|nvrm'
```

### Step 3. Check ECC / Row Remap

```bash
nvidia-smi -q -d ECC
nvidia-smi -q | egrep -A40 'Row Remapper|Remapped Rows|Repair|Pending'
```

### Step 4. Check NVLink / PCIe

```bash
nvidia-smi topo -m
nvidia-smi nvlink --status
nvidia-smi nvlink --errorcounters
dmesg -T | egrep -i 'pcie|aer|fallen off'
```

### Step 5. Check RDMA / NIC

```bash
dmesg -T | egrep -i 'mlx5|rdma|infiniband|roce|cqe'
```

If possible, check the NIC counters together.

```bash
ethtool -S <interface>
```

In an InfiniBand environment, also check the following.

```bash
ibstat
ibv_devinfo
perfquery
```

### Step 6. DCGM Diagnosis

```bash
dcgmi diag -r 3
```

If the failure repeats, drain the node and run a longer diagnosis.

```bash
dcgmi diag -r 4
```

### Step 7. Isolate in the Scheduler

Kubernetes:

```bash
kubectl cordon <gpu-node>
kubectl drain <gpu-node> --ignore-daemonsets --delete-emptydir-data
```

Slurm:

```bash
scontrol update NodeName=<gpu-node> State=DRAIN Reason="GPU Xid/ECC investigation"
```

## 10. Conclusion

GPU cluster failures are hard to simply classify as "a GPU problem" or "a network problem".

An NCCL hang may be an RDMA problem, but it can also occur because a specific rank died from a GPU ECC or Xid event. Symptoms that look like RDMA errors can also be connected to PCIe, GPU memory, NVLink, or CUDA context failures.

So GPU infrastructure operators must view failures from the following perspective.

```text
Do not look only at NCCL logs.
Do not look only at RDMA counters.
Do not look only at nvidia-smi output.
Look at Xid, ECC, row remap, NVLink, PCIe, DCGM, and scheduler events together.
```

Ultimately, operational decisions can be organized by the following criteria.

```text
Treat Xid 94 as a contained error and check the affected application/job scope.
Treat Xid 95 as an uncontained error and immediately consider node drain.
For correctable ECC, look at the increasing trend and repetition.
Treat uncorrectable ECC as an immediate isolation target.
Do not conclude RMA from the aggregate ECC counter alone.
Treat row remapping failures, DCGM failures, and repeated Xid as RMA candidates.
```

In GPU cluster operations, what matters is not quickly pinning a failure to "one cause". What matters is tying GPU, PCIe, NVLink, RDMA, NCCL, and scheduler events together into a single timeline.

Only by being able to build that timeline do you get close to the true cause.

## References

- [NVIDIA Xid Errors](https://docs.nvidia.com/deploy/xid-errors/)
- [NVIDIA Xid Catalog](https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html)
- [NVIDIA GPU Debug Guidelines](https://docs.nvidia.com/deploy/gpu-debug-guidelines/index.html)
- [NVIDIA GPU Memory Error Management](https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/latest/index.html)
- [NVIDIA nvidia-smi Documentation](https://docs.nvidia.com/deploy/nvidia-smi/)
- [NVIDIA DCGM Documentation](https://docs.nvidia.com/datacenter/dcgm/latest/)
