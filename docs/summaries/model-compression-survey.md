# Survey: Model Compression for LLMs — Summary

Source: A Survey on Model Compression for Large Language Models (2024)

TL;DR

- Compression spans quantization, pruning, knowledge distillation (KD), and low-rank factorization. For an inference engine, weight/KV quant and light structured pruning are the lowest-risk wins.

Key points

- Quantization: PTQ (GPTQ/AWQ), QAT for quality-sensitive tasks; activation-aware scaling helps maintain accuracy at low bit-widths.
- Pruning: unstructured (sparse) vs structured (channels/heads/blocks). Semi-structured can preserve throughput; aggressive unstructured sparsity often needs custom kernels to speed up.
- KD: black-box (self-consistency, CoT) and white-box (reverse-KL, layer-wise) reduce model size or improve small models; more relevant for training pipelines than pure inference.
- Low-rank: LoRA/low-rank decompositions reduce parameter count or enable adapters; inference impact depends on kernel support.

Actionable for Lightbulb

- Prioritize quantization and light structured pruning that maps to existing kernels. Avoid bespoke sparse kernels unless Candle adds support.
- Provide hooks for KD-evaluated small models in docs/benchmarks but keep engine changes minimal.
- Document metrics: perplexity/accuracy deltas, tok/s, memory per request, TTFT; include a repeatable small benchmark suite.

Acceptance criteria

- Quantized and pruned tiny models run end-to-end with predictable performance on CPU.
- Benchmarks report consistent wins (≥15% latency or ≥30% memory reduction) under documented conditions, or we publish a clear no-go note.

Citation

- A Survey on Model Compression for Large Language Models (2024).
