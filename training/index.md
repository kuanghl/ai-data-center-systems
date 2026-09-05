# MLPerf Training

[MLCommons Training](https://github.com/mlcommons/training) is a collection of reference implementations for the MLPerf Training benchmark. Rather than optimized submission code, it is a solid baseline for reading and experimenting with what compute, network, storage, and orchestration requirements large-scale training workloads create in an AI data center.

As of 2026, [MLPerf Training v6.0](https://github.com/mlcommons/training#mlperf-training-v60-submission-deadline-may-15-2026) is the latest benchmark table, covering LLM pretraining, MoE, LoRA fine-tuning, text-to-image, and recommendation models.

## Reading Materials

| Topic | Link | What to look at from a systems perspective |
| --- | --- | --- |
| Overall overview | [mlcommons/training](https://github.com/mlcommons/training) | benchmark list, common run procedures, dataset/container/target quality criteria |
| MLPerf Training paper | [arXiv:1910.01500](https://arxiv.org/abs/1910.01500) | benchmark design, time-to-train, target quality, closed/open division |
| v6.0 benchmark table | [MLPerf Training v6.0](https://github.com/mlcommons/training#mlperf-training-v60-submission-deadline-may-15-2026) | models, frameworks, datasets, and parameter counts for 2026 submissions |
| Llama 3.1 8B pretraining | [small_llm_pretraining/nemo](https://github.com/mlcommons/training/tree/master/small_llm_pretraining/nemo) | NeMo-based LLM pretraining, C4 dataset, Slurm job, checkpoint resume |
| Llama 3.1 405B pretraining | [large_language_model_pretraining/nemo](https://github.com/mlcommons/training/tree/master/large_language_model_pretraining/nemo) | training a very large dense LLM, distributed checkpoints, multi-node Slurm |
| DeepSeek V3 671B MoE | [llm_moe_pretraining/nemo](https://github.com/mlcommons/training/tree/master/llm_moe_pretraining/nemo) | MoE expert parallelism, GBS constraints, GB300-class multi-node training |
| GPT-OSS 20B MoE | [small_llm_moe_pretraining/primus](https://github.com/mlcommons/training/tree/master/small_llm_moe_pretraining/primus) | Primus-based MoE training, AMD/NVIDIA single-node examples, expert parallelism |
| Llama2 70B LoRA | [llama2_70b_lora](https://github.com/mlcommons/training/tree/master/llama2_70b_lora) | PEFT/LoRA fine-tuning, 8k sequence, Accelerate, FlashAttention |
| FLUX.1 text-to-image | [text_to_image](https://github.com/mlcommons/training/tree/master/text_to_image) | TorchTitan-based diffusion/flow model training, preprocessed embeddings, HSDP/DDP |
| DLRM DCNv2 recommendation | [recommendation_v2/torchrec_dlrm](https://github.com/mlcommons/training/tree/master/recommendation_v2/torchrec_dlrm) | TorchRec embedding sharding, model-parallel embedding tables, Criteo multi-hot data |
| MLCommons R2 Downloader | [mlcommons/r2-downloader](https://github.com/mlcommons/r2-downloader) | automated downloads of C4, tokenizer, Criteo, and FLUX datasets |
| MLCommons training storage | [training.mlcommons-storage.org](https://training.mlcommons-storage.org) | public/member-restricted benchmark datasets and checkpoint metadata |

## 2026 v6.0 Benchmark Summary

| Model | Reference implementation | Framework | Dataset | Systems notes |
| --- | --- | --- | --- | --- |
| FLUX.1 | [text_to_image](https://github.com/mlcommons/training/tree/master/text_to_image) | TorchTitan | CC12M subset | GPU memory, image/text embedding preprocessing, HSDP/DDP |
| Llama 3.1 8B | [small_llm_pretraining/nemo](https://github.com/mlcommons/training/tree/master/small_llm_pretraining/nemo) | NeMo | C4 | tensor/data parallel, C4 dataloader, checkpoint resume |
| Llama2 70B LoRA | [llama2_70b_lora](https://github.com/mlcommons/training/tree/master/llama2_70b_lora) | PyTorch | SCROLLS GovReport | PEFT, 8k context, FlashAttention, 8 GPU fine-tuning |
| Llama 3.1 405B | [large_language_model_pretraining/nemo](https://github.com/mlcommons/training/tree/master/large_language_model_pretraining/nemo) | NeMo | C4 | multi-node Slurm, checkpoint conversion/resume, high-bandwidth fabric |
| DLRM DCNv2 | [recommendation_v2/torchrec_dlrm](https://github.com/mlcommons/training/tree/master/recommendation_v2/torchrec_dlrm) | TorchRec | Criteo 3.5TB multi-hot | embedding all-to-all, host memory, mmap, storage throughput |
| GPT-OSS 20B MoE | [small_llm_moe_pretraining/primus](https://github.com/mlcommons/training/tree/master/small_llm_moe_pretraining/primus) | Primus | C4 | expert parallelism, MoE routing, AMD/NVIDIA portability |
| DeepSeek V3 671B | [llm_moe_pretraining/nemo](https://github.com/mlcommons/training/tree/master/llm_moe_pretraining/nemo) | NeMo | C4 | 256-GPU-class MoE, expert parallelism, large GBS, frequent evaluation |

## Running the Examples

### 1. Llama 3.1 8B: NeMo-based distributed pretraining

This is a small LLM pretraining reference. Fill in the Slurm, container, dataset, and checkpoint paths in `config.sh`, then submit the job.

```bash
git clone https://github.com/mlcommons/training.git mlcommons-training
cd mlcommons-training/small_llm_pretraining/nemo

docker build -t mlperf-llama31-8b -f Dockerfile .

# C4 pre-tokenized dataset
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d /data/llama3_1_8b/c4 \
  https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri

# Llama 3.1 8B tokenizer
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d /data/llama3_1_8b/tokenizer \
  https://training.mlcommons-storage.org/metadata/llama-3-1-8b-tokenizer.uri

# Set the Slurm partition/account, container image, PREPROCESSED_PATH, TOKENIZER_PATH, etc. in config.sh.
source config.sh
bash run_llama31.sh
```

Observation points:

- Check where GPU utilization stalls: dataloader, checkpoint write/read, or NCCL collectives.
- Verify that the tensor/pipeline/data parallel settings in `config.sh` match the actual node/GPU topology.
- With checkpoint resume enabled, you can see how shared storage and network fabric tail latency affect training time.

### 2. DLRM DCNv2: TorchRec embedding sharding and all-to-all

Recommendation models have large embedding tables, making them a good way to observe inter-GPU all-to-all, host memory, and mmap I/O bottlenecks.

```bash
cd mlcommons-training/recommendation_v2/torchrec_dlrm
pip install -r requirements.txt
pip install torchx

bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  https://training.mlcommons-storage.org/metadata/dlrmv2-preprocessed-criteo-click-logs.uri

export TOTAL_TRAINING_SAMPLES=4195197692
export GLOBAL_BATCH_SIZE=65536
export WORLD_SIZE=8
export CRITEO_MULTI_HOT=/data/criteo/multi_hot

torchx run -s local_cwd dist.ddp -j 1x8 --script dlrm_main.py -- \
  --embedding_dim 128 \
  --dense_arch_layer_sizes 512,256,128 \
  --over_arch_layer_sizes 1024,1024,512,256,1 \
  --synthetic_multi_hot_criteo_path "$CRITEO_MULTI_HOT" \
  --num_embeddings_per_feature 40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36 \
  --validation_freq_within_epoch $((TOTAL_TRAINING_SAMPLES / (GLOBAL_BATCH_SIZE * 20))) \
  --epochs 1 \
  --pin_memory \
  --mmap_mode \
  --batch_size $((GLOBAL_BATCH_SIZE / WORLD_SIZE)) \
  --interaction_type=dcn \
  --dcn_num_layers=3 \
  --dcn_low_rank_dim=512 \
  --adagrad \
  --learning_rate 0.005
```

Observation points:

- Toggle `--mmap_mode` on/off to compare startup time, storage read patterns, and steady-state throughput.
- Watch how much all-to-all traffic embedding table sharding generates, together with NCCL/IB/RoCE counters.
- Vary the local batch size and see where the bottleneck moves among GPU compute, network, and host memory bandwidth.

### 3. FLUX.1: TorchTitan HSDP/DDP text-to-image training

The FLUX.1 example is a good way to look at image/text encoder preprocessing and distributed training modes together.

```bash
cd mlcommons-training/text_to_image
export MLCOMMONS_TRAINING_ROOT="$(pwd)/.."

cd torchtitan
docker build -t mlperf-flux -f Dockerfile .
cd ..

mkdir -p /data/flux /models/flux /logs/flux

# With preprocessed embeddings, the frozen encoders do not need to run at every step.
cd /data/flux
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  https://training.mlcommons-storage.org/metadata/flux-1-cc12m-preprocessed.uri
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  https://training.mlcommons-storage.org/metadata/flux-1-coco-preprocessed.uri
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  https://training.mlcommons-storage.org/metadata/flux-1-empty-encodings.uri

cd "$MLCOMMONS_TRAINING_ROOT/text_to_image"

export DATAROOT=/data/flux
export MODELROOT=/models/flux
export LOGDIR=/logs/flux
export CONFIG_FILE=torchtitan/experiments/flux/train_configs/flux_schnell_mlperf_preprocessed.toml
export CONT=mlperf-flux
export SEED=1234

sbatch -N 2 -t 04:00:00 run.sub \
  --parallelism.data_parallel_replicate_degree=2 \
  --parallelism.data_parallel_shard_degree=8
```

Observation points:

- The default is an HSDP structure mixing intra-node sharding with inter-node DDP.
- Using preprocessed embeddings increases storage capacity and read bandwidth needs but reduces encoder compute load.
- To enable checkpointing, set `ENABLE_CHECKPOINTING=True` and `--checkpoint.interval=<steps>` together.

### 4. GPT-OSS 20B MoE: single-node MoE training

The Primus-based MoE example is a good way to compare expert parallelism across different accelerators, such as B200 and MI355X.

```bash
cd mlcommons-training/small_llm_moe_pretraining/primus

# Build from Dockerfile.nvidia for NVIDIA B200, or Dockerfile for AMD MI355X.
docker build -t mlperf-gpt-oss-20b -f Dockerfile.nvidia .

bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d /data/gpt_oss_20b/data \
  https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri

export DATADIR=/data/gpt_oss_20b/data
export MODELDIR=/data/gpt_oss_20b/model
export LOGDIR=/data/gpt_oss_20b/results
export CONT=mlperf-gpt-oss-20b

source config_B200_1x8x1.sh
export NEXP=1
bash run_with_docker.sh
```

Observation points:

- For MoE, all-to-all and expert load balancing matter more than for dense LLMs.
- Even on the same single 8-GPU node, expert parallel performance varies with NVLink, PCIe, and NUMA placement.
- If running on Kubernetes, check CPU pinning, Topology Manager, and GPU-local memory paths together, as in the NUMA-local GPU workload case study in [AI Systems Performance Engineering Chapter 3](../systems-performance/chap03/).
- Separating the AMD/NVIDIA configs lets you compare per-accelerator software stack differences.

### 5. DeepSeek V3 671B: large-scale MoE multi-node Slurm

This example is more of a large-scale cluster design reference than an everyday local experiment. The reference assumes Slurm-based multi-node execution.

```bash
cd mlcommons-training/llm_moe_pretraining/nemo

docker build -t mlperf-deepseek-v3 -f Dockerfile .

python3 -m venv venv
source venv/bin/activate
pip install git+https://github.com/NVIDIA-NeMo/Run.git

bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d /data/deepseekv3/c4 \
  https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri

bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d /data/deepseekv3/tokenizer \
  https://training.mlcommons-storage.org/metadata/llama-3-1-8b-tokenizer.uri

# Adjust the Slurm, IMAGE, DATA_DIR, MODEL_CKPT, and parallelism values in the config file to match your environment.
source config_GB300_64x4x256xtp1pp4cp1.sh
bash run_deepseek_v3_671b.sh
```

Observation points:

- DeepSeek V3 671B is an MoE structure with 671B total parameters and 37B active parameters.
- The benchmark enforces a large GBS, so fabric bandwidth, all-to-all latency, and evaluation overhead matter.
- Checkpoints require a separate download/conversion procedure, and a multi-TB-class shared filesystem design must come first.

## Experiment Ideas

- Feed the same C4 dataset into Llama 3.1 8B dense pretraining and GPT-OSS 20B MoE pretraining, and compare the NCCL collective patterns.
- In DLRM DCNv2, vary `--mmap_mode`, local batch size, and embedding sharding settings to find the storage/network/compute bottleneck transition points.
- Split FLUX.1 into a raw image pipeline and a preprocessed embedding pipeline, and compare CPU preprocessing, storage capacity, and GPU utilization.
- Vary the checkpoint interval and compare write bursts and recovery time across shared filesystem, object storage, and local NVMe.
- Align Slurm job logs, MLPerf logs, GPU profiler output, NCCL debug logs, and fabric counters on the same timeline to reconstruct the time-to-train bottleneck.
