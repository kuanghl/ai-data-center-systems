# MLPerf Storage Benchmark

[MLCommons Storage](https://github.com/mlcommons/storage) is the MLPerf Storage Benchmark Suite for measuring the performance of storage systems that support ML workloads. It is not a plain filesystem benchmark: it reproduces the I/O patterns that actually cause problems in ML systems, such as training data loading, checkpoint save/restore, LLM KV cache offload, and VectorDB search.

## Reading Materials

| Topic | Link | What to look at from an ML perspective |
| --- | --- | --- |
| Overall overview | [mlcommons/storage](https://github.com/mlcommons/storage) | benchmark installation, `--file`/`--object` backend selection, training/checkpointing/vectordb/kvcache categories |
| Documentation index | [docs/README.md](https://github.com/mlcommons/storage/blob/main/docs/README.md) | entry point to the four benchmark workloads and related documentation |
| Quick start | [docs/QUICK_START.md](https://github.com/mlcommons/storage/blob/main/docs/QUICK_START.md) | run examples for local filesystem, S3 object storage, checkpointing, KV cache, and VectorDB |
| Training I/O | [training/README.md](https://github.com/mlcommons/storage/blob/main/training/README.md) | FLUX.1, RetinaNet, DLRMv2 training data loading patterns; dataset sizing/datagen/run procedures |
| Checkpointing | [checkpointing/README.md](https://github.com/mlcommons/storage/blob/main/checkpointing/README.md) | Llama3 8B/70B/405B/1T checkpoint write/read, cache clear, shared/local storage modes |
| Streaming Checkpoint | [docs/Streaming-Chkpt-Guide.md](https://github.com/mlcommons/storage/blob/main/docs/Streaming-Chkpt-Guide.md) | structure for generating and writing large checkpoints in a streaming fashion |
| KV Cache Benchmark | [kv_cache_benchmark/README.md](https://github.com/mlcommons/storage/blob/main/kv_cache_benchmark/README.md) | measuring KV cache offload latency and throughput across GPU VRAM, CPU RAM, and NVMe in LLM inference |
| VectorDB Benchmark | [vdb_benchmark/README.md](https://github.com/mlcommons/storage/blob/main/vdb_benchmark/README.md) | vector search storage performance for Milvus-based DiskANN, HNSW, and AISAQ indexes |
| Object Storage Guide | [docs/OBJECT_STORAGE_GUIDE.md](https://github.com/mlcommons/storage/blob/main/docs/OBJECT_STORAGE_GUIDE.md) | `.env`, bucket, endpoint, and `--object` configuration for S3-compatible object storage |
| Storage Libraries | [docs/STORAGE_LIBRARIES.md](https://github.com/mlcommons/storage/blob/main/docs/STORAGE_LIBRARIES.md) | comparison of `s3dlio`, `minio`, and `s3torchconnector` |
| DataLoader Architecture | [docs/DATALOADER_ARCHITECTURE.md](https://github.com/mlcommons/storage/blob/main/docs/DATALOADER_ARCHITECTURE.md) | map-style vs iterable-style DataLoaders, prefetch, and O_DIRECT differences on object storage and NVMe |
| Multi Endpoint | [docs/MULTI_ENDPOINT_GUIDE.md](https://github.com/mlcommons/storage/blob/main/docs/MULTI_ENDPOINT_GUIDE.md) | how to spread I/O across multiple S3 endpoints |
| Parquet Format | [docs/PARQUET_FORMATS.md](https://github.com/mlcommons/storage/blob/main/docs/PARQUET_FORMATS.md) | training data format experiments with the Parquet reader, row groups, and byte-range GETs |
| Submission rules | [Rules.md](https://github.com/mlcommons/storage/blob/main/Rules.md) | CLOSED/OPEN submissions, power/RU normalized metrics, benchmark compliance |

## Running the Examples

### 1. Training I/O: local NVMe/filesystem

A basic experiment for seeing how well storage feeds the accelerators in a training workload.

```bash
git clone https://github.com/mlcommons/storage.git
cd storage
uv sync

uv run mlpstorage training datagen \
  --model retinanet \
  --num-processes 4 \
  --open --file \
  --data-dir /mnt/nvme_data/retinanet \
  --params dataset.num_files_train=250000

uv run mlpstorage training run \
  --model retinanet \
  --num-accelerators 4 \
  --accelerator-type b200 \
  --client-host-memory-in-gb 64 \
  --open --file \
  --data-dir /mnt/nvme_data/retinanet \
  --params dataset.num_files_train=250000
```

### 2. Training I/O: S3-compatible Object Storage

The dataset is kept as an object key prefix instead of a file path, measuring whether object storage can hold up under training data loading.

```bash
cat > .env <<'EOF'
AWS_ENDPOINT_URL=http://127.0.0.1:9000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
AWS_REGION=us-east-1
STORAGE_LIBRARY=s3dlio
BUCKET=mlp-retinanet
EOF

uv run mlpstorage training datagen \
  --model retinanet \
  --num-processes 4 \
  --open --object \
  --data-dir retinanet \
  --params dataset.num_files_train=250000

uv run mlpstorage training run \
  --model retinanet \
  --num-accelerators 4 \
  --accelerator-type b200 \
  --client-host-memory-in-gb 64 \
  --open --object \
  --data-dir retinanet \
  --params dataset.num_files_train=250000
```

### 3. Checkpointing: LLM training failure-recovery I/O

For large-scale LLM training, look at checkpoint write/read throughput and the tail latency of the slowest rank.

```bash
uv run mlpstorage checkpointing run \
  --model llama3-8b \
  --num-processes 8 \
  --client-host-memory-in-gb 512 \
  --checkpoint-folder /mnt/checkpoint_test \
  --num-checkpoints-read=0

# Drop the OS page cache before the read phase if needed.
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

uv run mlpstorage checkpointing run \
  --model llama3-8b \
  --num-processes 8 \
  --client-host-memory-in-gb 512 \
  --checkpoint-folder /mnt/checkpoint_test \
  --num-checkpoints-write=0
```

### 4. LLM KV Cache Offload

With long contexts and multi-user inference, this measures the latency of the storage tier when the KV cache is pushed out to CPU/NVMe.

```bash
cd kv_cache_benchmark
pip install ".[full]"

python3 kv-cache.py \
  --config config.yaml \
  --model llama3.1-8b \
  --num-users 50 \
  --duration 120 \
  --gpu-mem-gb 0 \
  --cpu-mem-gb 4 \
  --cache-dir /mnt/nvme \
  --output results.json
```

### 5. VectorDB/RAG Storage

Look at how storage affects latency/throughput/recall across Milvus's vector load, index build, and query stages.

```bash
cd storage
uv sync --extra vectordb
uv pip install -e ./vdb_benchmark

docker compose -f vdb_benchmark/docker-compose.yml up -d

./mlpstorage vectordb datasize \
  --dimension 1536 \
  --num-vectors 10000000 \
  --index-type DISKANN \
  --num-shards 10

./mlpstorage vectordb datagen \
  --host 127.0.0.1 \
  --port 19530 \
  --config default \
  --num-vectors 50000 \
  --dimension 1536 \
  --num-shards 1 \
  --force \
  --results-dir /tmp/vdb_results
```

## Experiment Ideas

- Repeat the same RetinaNet datagen/run on local NVMe, NFS, a ZFS dataset, and S3-compatible object storage, and compare accelerator utilization and read throughput.
- Following `docs/DATALOADER_ARCHITECTURE.md`, sweep map-style vs iterable-style DataLoaders, O_DIRECT on/off, worker count, and prefetch depth.
- For checkpointing, separate the write-only and read-only phases, and compare cold reads (page cache dropped) against warm reads (cache kept).
- For the KV cache benchmark, vary `--gpu-mem-gb`, `--cpu-mem-gb`, and `--cache-dir` to find the GPU/CPU/NVMe tiering thresholds.
- For VectorDB, compare a disk-heavy index like DiskANN against a memory-heavy index like HNSW on the same storage backend.
