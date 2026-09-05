# Inference

This section organizes LLM inference model-structure and system-optimization material by topic.

## Layout

| Topic | Nature | Scope |
|---|---|---|
| [Efficient LLM Inference Systems](efficient-llm-inference-systems/) | Course — **primary spine** | performance metrics, hardware, KV cache, quantization, and speculative decoding with weekly hands-on exercises |
| [FlashAttention from First Principles](flashattention-from-first-principles/) | Deep dive | online softmax and tiling, the per-generation bottleneck shift of FA-1 through FA-4 |
| [SGLang in 2026](sglang-production-practices/) | Case study | RadixAttention, Model Gateway, HiCache, prefill-decode disaggregation, production rollout (**English**) |
| [Models](models/) | Reference | per-model architecture and serving profiles — [Kimi K2.5](models/kimi-k2-5-scaling.md), [Kimi K3](models/kimi-k3.md) |

## Suggested path

1. **Foundations:** start with [Efficient LLM Inference Systems](efficient-llm-inference-systems/) to establish metrics, hardware, and the KV cache.
2. **Kernel level:** use [FlashAttention](flashattention-from-first-principles/) to see the principle of not materializing intermediate matrices in HBM and where the center of optimization moved on each GPU generation.
3. **System level:** move up from individual kernel speed to a goodput perspective that satisfies SLOs in [SGLang](sglang-production-practices/).
4. **Per-model lookup:** consult [Models](models/) for the memory and communication characteristics of a target model.

## Resources

### Books

- [Inference Engineering](https://www.baseten.co/inference-engineering/) — a practical guide covering model architecture, GPU hardware, inference engines, optimization techniques, and production serving
- **Hands-On LLM Serving and Optimization** (Chi Wang, Peiheng Hu) — covers KV cache, batching, quantization, speculative decoding, distributed serving, and vLLM optimization with a hands-on focus.
  - [Book](https://orca3.github.io/llm-model-inference/) · [Video](https://www.oreilly.com/library/view/hands-on-llm-serving/9798341621480/) · [Code and Notebooks](https://github.com/orca3/llm-model-inference)
