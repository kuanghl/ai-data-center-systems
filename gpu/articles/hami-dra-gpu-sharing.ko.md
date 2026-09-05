# Kubernetes GPU Partitioning Through DRA and HAMi

Sharing GPUs in Kubernetes requires solving two problems.

First, "which GPU should be assigned to which pod?" Second, "within a single GPU, how do we actually limit memory and compute usage?" Dynamic Resource Allocation (DRA) is the system that tries to solve the first problem with a Kubernetes standard API. HAMi is an implementation that also covers the second problem and operates as a real GPU sharing platform.

```text
DRA  = the standard allocation API where Kubernetes selects devices intelligently
HAMi = an operational platform covering GPU memory/core sharing, scheduling, and runtime limits
```

Missing this difference easily leads to the question "since we have DRA, do we still need HAMi?" The short answer is no — DRA does not automatically create vGPUs. A DRA driver or backend must publish the device inventory, interpret claims, and implement the actual runtime limits. HAMi is a project that has already implemented much of this backend role through a Device Plugin, Scheduler Extender, admission webhook, and `libvgpu.so`.

## The Limits of the Existing Device Plugin

The Kubernetes Device Plugin model succeeded at exposing GPUs as cluster resources, but what it can fundamentally express is close to the "count" of devices.

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

This model alone makes the following requirements hard to express naturally.

| Judgment needed | Limit of the Device Plugin default model |
| --- | --- |
| select GPUs with 24GiB+ of VRAM | the scheduler cannot check device attributes in a standard way |
| select only H100 or A100 | requires vendor annotations or a scheduler extender |
| reflect NVLink/topology conditions | hard to express with just the default resource count |
| select a specific MIG profile | requires a separate plugin policy or custom resources |
| multiple pods sharing the same claim | centered on per-container device requests, lacking expressiveness |

For this reason, many GPU platforms have supplemented device information by combining Node annotations, scheduler extenders, admission webhooks, and custom resources. HAMi belongs to this family.

## What DRA Standardizes

With DRA, Kubernetes can treat devices not as a simple "integer count" but as "objects that can be requested based on attribute conditions". According to the official Kubernetes documentation, the core DRA features use `DeviceClass`, `ResourceClaim`, `ResourceClaimTemplate`, and `ResourceSlice` from the `resource.k8s.io/v1` API group, and are marked as stable features from Kubernetes v1.35 onward.

The core objects can be viewed as follows.

| Object | Meaning |
| --- | --- |
| `DeviceClass` | a kind of device offered by the cluster administrator. Examples: `h100-highmem`, `cost-optimized-gpu` |
| `ResourceClaim` | the device claim a workload asks for |
| `ResourceClaimTemplate` | a template used when each pod needs an independent claim, as with Deployments or Jobs |
| `ResourceSlice` | the actual node/device inventory that the DRA driver publishes to the API server |

Instead of "1 GPU", users can request "a device belonging to a specific class that satisfies the given attribute conditions". For example:

```yaml
apiVersion: resource.k8s.io/v1
kind: ResourceClaimTemplate
metadata:
  name: h100-inference
spec:
  spec:
    devices:
      requests:
      - name: gpu
        exactly:
          deviceClassName: h100-highmem
```

The strength of this model is standardization. It can handle devices other than GPUs — NPUs, FPGAs, DPUs, SmartNICs — under one API family. From a long-term view, if you are designing a platform API from scratch, it is preferable to consider a DRA-style claim model rather than exposing vendor-specific annotations directly to users.

However, DRA itself does not enforce GPU memory quotas, adjust SM usage, or intercept CUDA calls. DRA is an allocation API, and the actual limits are the responsibility of the driver and backend.

## What HAMi Actually Does

HAMi uses several extension points together to share GPUs in smaller units in Kubernetes.

![HAMi control plane to runtime enforcement boundary](assets/hami-control-runtime-boundary.svg)

Users request GPUs with syntax similar to existing Kubernetes resource limits.

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
    nvidia.com/gpumem: 12000
    nvidia.com/gpucores: 40
```

The non-MIG operating path of HAMi, simplified, is as follows.

1. The Device Plugin tells the kubelet about one physical GPU as if it were multiple logical GPU slots.
2. The Scheduler Extender places pods based on the GPU memory, core, model, and status information stored in Node annotations.
3. The selection result is recorded in pod annotations.
4. The Device Plugin's `Allocate` reads the annotations and injects devices, environment variables, and mounts into the container.
5. `libvgpu.so` inside the container hooks the CUDA/NVML call paths and applies the VRAM quota and compute limits.

The key is the last step. HAMi's software-based GPU sharing does not physically partition the hardware the way MIG does. In the Kubernetes control plane, replica IDs and annotations make the GPU "appear split", and at runtime `libvgpu.so` intercepts the CUDA/NVML paths to apply the limits.

## Looking a Bit Closer at HAMi's Implementation Paths

Describing HAMi simply as "a tool that splits GPUs" misses important parts. In reality, it is a structure where the Kubernetes control plane and the runtime hooks inside containers are connected by a defined protocol. The control plane decides which physical GPU and logical slot each pod will use, and the runtime plane enforces the corresponding quota inside the container.

| Layer | Representative module | What it does |
| --- | --- | --- |
| configuration | `cmd/device-plugin/nvidia/vgpucfg.go` | defines options such as `deviceSplitCount`, memory/core scaling ratios, and disabling core limits |
| admission | `pkg/scheduler/webhook.go` | detects pods requesting GPU resources and hands them to the HAMi scheduler path |
| scheduler cache | `pkg/scheduler/scheduler.go`, `pkg/scheduler/nodes.go` | aggregates Node annotations and pod allocation state to maintain the overall GPU picture |
| scheduler policy | `pkg/scheduler/score.go`, `pkg/scheduler/policy/*` | computes node/device fit, binpack/spread scores, and memory/core fragmentation |
| device model | `pkg/device-plugin/nvidiadevice/nvinternal/rm/devices.go` | manages replica IDs, memory, NUMA, and status information for physical GPUs |
| allocation | `pkg/device-plugin/nvidiadevice/nvinternal/plugin/server.go` | responds to the kubelet's `Allocate` request with device, environment variable, and mount information |
| runtime hook | `libvgpu` / `HAMi-core` | hooks the CUDA/NVML APIs to apply memory quotas and compute limits |
| dynamic MIG | NVIDIA MIG-related manager and `mig-parted` integration | adjusts hardware MIG instances instead of software partitioning |

The most important boundary in this table is between the scheduler and the Device Plugin. An ordinary Kubernetes scheduler cannot sufficiently convey the details of the individual device the Device Plugin selected. So HAMi uses pod annotations as an internal protocol. The scheduler records "how much memory and cores this pod will use on this GPU UUID" in the annotations, and the Device Plugin reads these values when the kubelet calls `Allocate` to build the actual container response.

In other words, HAMi's control plane solves the following three problems at the same time.

| Problem | HAMi's handling |
| --- | --- |
| only the GPU count is visible to Kubernetes | the Device Plugin registers multiple logical replicas |
| the scheduler does not know remaining VRAM/cores | device inventory and usage are maintained in Node annotations |
| quota information is missing at the kubelet allocation stage | the scheduler's decision is passed to the Device Plugin through pod annotations |

This structure is practical, but it is not complete with standard APIs alone. This is also why DRA is attractive in the long term. DRA's `ResourceSlice` and `ResourceClaim` point in the direction of standardizing this kind of annotation protocol inside the Kubernetes API.

## The Core of Scheduling Is Fragmentation, Not Count

The core resource the HAMi scheduler looks at is not the simple `nvidia.com/gpu` count. Each GPU has remaining logical slots, memory, and core ratios, and the scheduler policy selects nodes and devices based on these.

For example, even if you publish one 80GB GPU as 10 logical slots, the state of all slots is not the same. One pod may use only 4GB, another may use 40GB, and yet another may request 100% of the cores. At that point, the problem the scheduler must solve is not "how many slots are left" but closer to the following question.

```text
When this pod's gpumem/gpucores request is placed,
on which node and physical GPU does the least fragmentation occur?
```

HAMi's binpack/spread policies change this judgment.

| Policy | Intuition | Suitable situation |
| --- | --- | --- |
| `binpack` | fills GPUs or nodes already in use | when you want to keep empty GPUs/nodes to raise the chance of accepting large jobs |
| `spread` | distributes across multiple GPUs or nodes | when you want to reduce contention between neighboring workloads and heat/power concentration |

So when adopting HAMi, you should look at "how small pods fragment memory and cores" rather than "how many logical GPUs were created". Placing many small inference pods indiscriminately may raise the average utilization, but the contiguous VRAM space needed later to load a large model can run short. This is similar to the fragmentation seen in Kubernetes CPU scheduling, but on GPUs VRAM acts as a much more direct ceiling.

## What the Device Plugin `Allocate` Injects

HAMi's actual limits take effect after the kubelet calls the Device Plugin's `Allocate`. At that point the Device Plugin does not just hand the GPU device files to the container; it also injects the environment variables and mounts the HAMi runtime needs.

The most important values are the following.

| Injected item | Meaning |
| --- | --- |
| `CUDA_DEVICE_MEMORY_LIMIT_<index>` | the memory quota the container sees for that logical GPU |
| `CUDA_DEVICE_SM_LIMIT` | the compute quota set by the `gpucores` request |
| `CUDA_OVERSUBSCRIBE` | passes whether oversubscription is enabled to the runtime |
| `LIBCUDA_LOG_LEVEL` | HAMi-core log level |
| `libvgpu.so` mount | the runtime library that hooks CUDA/NVML calls |
| shared cache/lock directories | used for aggregating usage across multiple processes and synchronizing initialization |

Thanks to this injection approach, HAMi works without modifying application code. Users do not need to change their PyTorch, vLLM, or TensorFlow code. When the container starts, HAMi-core intervenes first in the dynamic linker and CUDA driver API paths.

But this is also HAMi's limitation. The limits are applied through user-space library hooks, not through independent partitions of hardware registers or GPU firmware. So HAMi non-MIG quotas are useful for operations but should not be treated as a security boundary.

## How `libvgpu.so` Enforces Limits

HAMi-core intercepts the CUDA/NVML call paths and produces two kinds of virtualization effects.

The first is virtualization of observed values. When `nvidia-smi` or an NVML memory query runs inside the container, it can be made to show a value matched to the quota instead of the full VRAM of the physical GPU. To the user it looks like a "12GB GPU", but in reality it is using part of an 80GB physical GPU.

The second is allocation limiting. Before memory allocation functions such as `cuMemAlloc_v2`, `cuMemAllocManaged`, and `cuMemoryAllocate` are called, the current usage is compared against the quota, and if the limit is exceeded an error similar to CUDA OOM is returned. Memory allocation requests occur as individual events, so pre-checks and usage aggregation are relatively easy to apply.

Compute limiting is more subtle. It is close to a scheme that adjusts the long-term average usage through kernel execution, utilization sampling, or token/throttling-style policies, without partitioning SMs at the hardware level. So `gpucores: 40` does not mean "exclusively allocate 40% of the SMs to this pod". More precisely, it means "the HAMi runtime tries to keep this workload's long-term average compute usage around 40%".

This difference is especially important for latency-sensitive inference workloads.

| Resource | Nature of the limit | Metrics to observe |
| --- | --- | --- |
| VRAM | allocations exceeding the quota are blocked relatively clearly | allocation failures, model load failures, peak memory usage |
| compute | soft throttling and scheduling have a large effect | throughput, p95/p99 latency, impact on neighboring workloads |
| PCIe/NVLink | not directly partitioned by HAMi quotas | H2D/D2H copy time, NCCL latency, DMA contention |
| L2 cache/memory bandwidth | hard to isolate strongly with software partitioning | kernel execution time variance, effective bandwidth, tail latency |

So the level of validation should differ depending on whether you use HAMi simply for "VRAM partitioning" or also expect "compute isolation".

## Dynamic MIG Is Another Backend Inside the Same Product

Even when HAMi supports MIG, the non-MIG `libvgpu.so` partitioning and MIG are not the same isolation model. The two can be provided within a single operational framework, but the places where resources are partitioned differ.

```text
HAMi non-MIG:
Kubernetes replica + annotation + libvgpu.so hook

HAMi dynamic MIG:
HAMi control plane + MIG profile/instance adjustment + hardware partition
```

On MIG-capable GPUs, hardware GPU instances are created, so fault isolation and resource predictability are higher than with non-MIG software partitioning. On the other hand, profile granularity is fixed, and dynamic reconfiguration requires considering the re-placement and draining of existing workloads and the cost of profile switching.

Operationally, it is clearer to distinguish as follows.

| Requirement | More suitable backend |
| --- | --- |
| want to build diverse fine-grained SKUs like 4GB, 8GB | HAMi non-MIG |
| want to mix development/inference workloads of the same organization to raise utilization | HAMi non-MIG |
| want to greatly reduce fault propagation and performance interference between tenants | MIG |
| want to automate profile-based partitioning on A100/H100 | HAMi dynamic MIG |
| a security boundary for VM products or external customers is important | vGPU/SR-IOV family |

## The Difference Between DRA and HAMi

| Viewpoint | DRA | HAMi |
| --- | --- | --- |
| nature | Kubernetes standard API/framework | GPU sharing implementation/operational platform |
| core purpose | attribute-based device allocation | sharing and scheduling at GPU memory/core granularity |
| actual VRAM limiting | not done by DRA itself | limited via `libvgpu.so` hooks |
| compute limiting | not done by DRA itself | soft throttling based on `gpucores` |
| scheduler integration | integrates with kube-scheduler and the standard APIs | centered on a scheduler extender and annotations |
| user experience | `ResourceClaim`, `DeviceClass` | `nvidia.com/gpumem`, `nvidia.com/gpucores` |
| multi-vendor support | fits the standard API but needs a driver ecosystem | direction of directly supporting multiple accelerator backends |
| oversubscription | not a DRA-native feature | one of HAMi's main features |
| isolation strength | varies by backend | non-MIG is software-hook based, so weak as a security boundary |

So the two technologies occupy different layers rather than competing. DRA is likely to become the standard language for how Kubernetes expresses and requests devices. HAMi plays the role of a backend or platform that provides actual GPU sharing and usage limiting on top of or beside it.

The HAMi project also reflects this flow. The HAMi-DRA subproject can be seen as a path that lets existing HAMi users migrate from Device Plugin-based requests to DRA-based requests. However, in current operational environments the existing HAMi resource model is more intuitive and also easier to connect to SKU, quota, billing, and RBAC systems.

![DRA and HAMi responsibility comparison](assets/dra-vs-hami-responsibility.svg)

## Why Cloud Operators Use HAMi First

The operational model commonly wanted in cloud or in-house AI platforms is the following.

```text
We want to split one 80GB GPU into product or quota units
like 4GB / 8GB / 16GB / 40GB.
```

Or:

```text
We want to place several small inference workloads on one GPU,
but not let each pod exceed its assigned VRAM quota.
```

DRA alone cannot implement this operational model directly. A DRA-compatible driver must express logical devices or consumable capacity and implement the runtime limits. HAMi already provides this model through `gpumem`, `gpucores`, the scheduler, and `libvgpu.so`.

DaoCloud's CNCF case study well shows the practical background of this choice. According to the public material, DaoCloud used HAMi in D.run Compute Cloud and DaoCloud Enterprise to operate more than 10,000 GPUs of capacity across more than 10 data centers. It also reported that after adopting vGPU, average GPU utilization exceeded 80% and GPU-related operational costs dropped 20–30%. It offered vGPU slices as marketplace SKUs and integrated quotas and RBAC at the vGPU level.

From this viewpoint, the following distinction matters.

```text
DRA ResourceClaim = a well-designed Kubernetes API
HAMi gpumem/gpucores = operational units that are easy to turn into SKUs
```

## Where HAMi, MIG, Time-Slicing, MPS, and vGPU Sit

When choosing a GPU sharing technology, look at "at which layer isolation happens" rather than "what is being divided".

| Approach | Partitioning location | Isolation strength | Suitable case |
| --- | --- | --- | --- |
| HAMi non-MIG | Kubernetes + user-space CUDA/NVML hooks | medium-low | inference, notebooks, batch inference within the same trust boundary |
| HAMi dynamic MIG | HAMi dynamically adjusts MIG configuration | high | when you want hardware partitioning and automation together on MIG-capable GPUs |
| NVIDIA MIG | GPU hardware instances | high | when tenant isolation, predictability, and fault isolation matter |
| NVIDIA time-slicing | time-division-style oversubscription | low | burst workloads that use resources briefly, simple sharing |
| MPS | concurrent execution optimization of CUDA processes | low-medium | HPC/multi-process workloads within the same trust boundary |
| NVIDIA vGPU/SR-IOV family | hypervisor/VF/vGPU manager | high | strong VM-based security boundaries and productization |

HAMi non-MIG should not be understood as a security boundary. It injects the physical device and hook library into the container and applies limits at the CUDA/NVML call paths. It is effective at raising resource utilization, but if you need a strong boundary that isolates untrusted tenants from each other, you should consider the MIG, vGPU, or SR-IOV family.

## Operational Cautions

First, oversubscription does not guarantee queueing. Even if HAMi configuration lets you raise the logical capacity, you must separately validate whether workloads wait properly or fail with OOM when physical VRAM is short. In actual HAMi issue #1128, a case was reported where, with `deviceSplitCount`, `deviceMemoryScaling`, and `deviceCoreScaling` set high, insufficient physical VRAM headroom led to `cuMemoryAllocate failed` and OOM. The core of this case is that "10x virtual capacity" does not mean "you can queue 10x the workloads and eventually run them".

Second, measure concurrent startup latency. HAMi-core uses shared files, locks, and a background watcher for in-container initialization and usage aggregation. In high-density inference environments where hundreds of processes call `cuInit` simultaneously, startup latency can become a bottleneck. Issue #1662 reports that when 40–50 pods each ran 4–5 child processes, causing 200–300 simultaneous CUDA initializations per node, about 1 minute of delay was observed due to `libvgpu.so` initialization lock contention. The latest code may have a different lock implementation, but the structural warning that node-level shared usage aggregation and initialization serialization points can become performance bottlenecks remains valid.

Third, `gpucores` does not mean a hardware SM partition. HAMi's compute limiting is close to soft throttling, and the perceived isolation level can vary depending on short bursts and kernel characteristics. Look at p95/p99 latency and the impact on neighboring workloads, not just average throughput.

Fourth, a DRA migration is not a simple API swap. You must consider the Kubernetes control plane version, feature gates on the scheduler and kubelet, the DRA driver, quota/RBAC/billing integration, and user education. Even if DRA is the right long-term direction, the cost of migrating an existing HAMi operational environment directly is not small.

Fifth, define the observation system first. If you adopt HAMi for cost reduction or utilization improvement, it is easy to focus only on average GPU utilization. But real operational decisions also need the following metrics.

| Metric | Why |
| --- | --- |
| per-pod requested/used VRAM | needed to find SKUs whose quota is too large or too small |
| model load failure rate | check whether the memory quota fits the actual workload characteristics |
| p95/p99 latency | check the impact of contention with neighboring workloads on user experience |
| startup latency | check the impact of `libvgpu.so` initialization and lock contention |
| GPU memory bandwidth/copy time | needed to find bottlenecks other than the VRAM quota |
| scheduler Pending reasons | needed to distinguish fragmentation from quota shortage |

Sixth, tell users the failure types clearly. When allocating whole GPUs, the failure patterns are relatively simple. If there is no GPU, the pod goes Pending; if memory is short, OOM occurs in the application. In HAMi, logical slots, `gpumem`, `gpucores`, physical free memory, hook initialization, annotation contention state, and device state all have an effect. Platform documentation should at least distinguish "Pending", "allocation failure", "CUDA OOM after container start", and "startup delay" as separate troubleshooting paths.

## Selection Criteria

| Situation | Consider first |
| --- | --- |
| building a new platform on Kubernetes 1.35+ and long-term standard APIs matter | DRA-first design |
| want to manage NPU, FPGA, and DPU under the same claim model as GPUs | DRA |
| must create GPU memory/core SKUs now and raise utilization | HAMi |
| want to place several inference pods on one GPU and apply quotas | HAMi non-MIG |
| strong tenant isolation and fault isolation matter | MIG, vGPU, SR-IOV |
| must automate hardware partition configuration on H100/A100 | MIG or HAMi dynamic MIG |
| simply want to pack more short burst workloads | NVIDIA time-slicing |
| running multiple CUDA processes concurrently within one job is the core | MPS |

The conclusion is simple. DRA is the system that standardizes Kubernetes' device allocation language, and HAMi is the operating system that actually shares and limits GPUs. Long term, a form where a backend like HAMi combines on top of a DRA-based standard API is natural. But if you must operate a GPU cloud or in-house inference platform right now, "can we actually limit the usage of the GPU slices we sell or allocate to users" matters more than "do we have a standard API". HAMi's practicality shows up exactly at this point.

## Validation Checklist

When adopting or evaluating HAMi, you should at minimum check the following.

| Validation item | What to check |
| --- | --- |
| replica exposure | do `deviceSplitCount` logical GPUs appear as allocatable resources |
| memory quota | are allocations exceeding `gpumem` consistently blocked |
| compute quota | does the `gpucores` value show up in long-term average utilization and latency |
| impact on neighboring workloads | how much do other pods on the same GPU shake p95/p99 latency |
| oversubscription failure | when physical VRAM is short, which outcome occurs — waiting, retry, or failure |
| concurrent startup | is the startup latency acceptable under simultaneous `cuInit` from dozens to hundreds of processes |
| security boundary | is there a policy not to expose non-MIG HAMi to untrusted tenants |
| quota/RBAC | do vGPU-level quotas mesh with the department/tenant permission model |
| observability | can per-pod GPU memory/utilization metrics be used for billing and operational metrics |
| migration plan | is there a long-term compatibility plan with DRA, HAMi-DRA, MIG, and the GPU Operator |

## Topics to Expand On Next

Once you understand the relationship between DRA and HAMi, you can look at the implementation, performance, and operational models in more depth. Especially in production GPU platforms, the scheduler's behavior, runtime limits, SKU design, and migration plans determine real operational quality more than the API choice.

| Topic | What it covers |
| --- | --- |
| HAMi scheduler deep dive | `fitInDevices`, binpack/spread score computation, Node annotation protocol |
| [HAMi-core deep dive](hami-core-deep-dive.ko.md) | `libvgpu.so`, CUDA/NVML hooks, memory usage aggregation, lock contention |
| HAMi vs MIG benchmarks | compare the same workload with non-MIG HAMi, MIG, and time-slicing |
| GPU SKU design notes | 4GB/8GB/16GB SKUs, quotas, billing, admission policy design |
| DRA migration plan | how to move the HAMi resource model to DRA `DeviceClass`/`ResourceClaim` |

## References

- [Dynamic Resource Allocation - Kubernetes](https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/)
- [GPU Virtualization Principles - HAMi](https://project-hami.io/docs/core-concepts/gpu-virtualization)
- [Project-HAMi/HAMi](https://github.com/Project-HAMi/HAMi)
- [Project-HAMi/HAMi-DRA](https://github.com/Project-HAMi/HAMi-DRA)
- [DaoCloud CNCF case study](https://www.cncf.io/case-studies/daocloud/)
- [HAMi issue #1662: libvgpu.so concurrent initialization latency](https://github.com/Project-HAMi/HAMi/issues/1662)
- [HAMi issue #1128: GPU oversubscription and OOM behavior](https://github.com/Project-HAMi/HAMi/issues/1128)
