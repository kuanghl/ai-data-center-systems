# Efficient LLM Inference Systems

This course builds understanding of LLM inference performance metrics and hardware characteristics, and connects KV cache, quantization, and speculative decoding to real measurements. This document guides the overall structure of the course and the appendices; detailed explanations and reference papers are covered in each week's document.

## Curriculum

- [Week 1: Understanding Performance Metrics](week01/README.md) — measures TTFT, TPOT, batch throughput, and memory limits.
- [Week 2: Hardware Foundations for Inference](week02/README.md) — covers the memory hierarchy, Tensor Cores, the roofline model, and inter-GPU communication.
- [Week 3: Transformer Inference and the KV Cache](week03/README.md) — analyzes the memory cost of MHA, MQA, GQA, and MLA for long contexts.
- [Week 4: Quantization](week04/README.md) — compares precision, quantization algorithms, kernel support, and quality/performance trade-offs.
- [Week 5: Speculative Decoding](week05/README.md) — reviews the conditions for latency improvement based on acceptance rate and draft strategies.

## Appendix

- [Hardware Architectures for LLM Inference](appendix/hardware-architectures/README.md)
- [LLM Inference](appendix/llm-inference/README.md)
- [Transformer](appendix/transformer/README.md)

## Source Material

- [Efficient LLM Inference Systems, Algorithms & Production Engineering — Interview Pocket Notes](https://drive.google.com/file/d/1mfTzOnwn8yx4eKObjPvpd-B_toGkQ_tu/view) (2026)

## Related Topics

- [SGLang Production Practices](../sglang-production-practices/README.md) — connects KV cache, speculative decoding, routing, and distributed topology to production SLOs.
- [Model Architecture and Serving Profiles](../models/README.md) — organizes per-model architecture and serving characteristics.
- [FlashAttention from First Principles](../flashattention-from-first-principles/README.md) ([English edition](../flashattention-from-first-principles/README.ko.md)) — explains I/O-aware kernel optimization for exact attention.
