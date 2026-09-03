# Model Architecture and Serving Profiles

This section organizes the core architecture of each LLM and the memory, communication, caching, and parallelization characteristics that must be checked in real inference and serving systems. It focuses on how model design affects the serving profile rather than on the system optimization techniques themselves.

## Model Notes

- [How We Scaled Kimi K2.5: Token Efficiency, Long Context, and Agent Swarms](kimi-k2-5-scaling.md)
- [Anatomy of Kimi K3: 2.8T MoE, KDA, Attention Residuals, and 64-GPU Serving](kimi-k3.md)

## Architecture Foundations

- [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch) — explains Transformer and LLM components from an implementation perspective.
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (2020.01) — organizes the scaling relationship among model size, dataset size, and compute.
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) (2024.05) — discusses how MLA and MoE design affect training and inference cost.

## Architecture Comparisons and References

- [LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)
- [The Big LLM Architecture Comparison](https://www.youtube.com/watch?v=rNlULI-zGcw)
- [The Big LLM Architecture Comparison Blog](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)
