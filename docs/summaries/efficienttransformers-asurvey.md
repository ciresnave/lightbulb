# Efficient Transformers: A Survey

**Full PDF:** [View Original](<file:///c:/Users/cires/OneDrive/Desktop/books%20and%20courses/Machine%20Learning/EfficientTransformers-ASurvey.pdf>)

**Markdown:** [View Markdown](../papers/markdown/efficienttransformers-asurvey.md)

**Local PDF (on disk):** c:\Users\cires\OneDrive\Desktop\books and courses\Machine Learning\EfficientTransformers-ASurvey.pdf

## TL;DR

Survey of algorithmic and architectural techniques to make Transformer models computationally and memory efficient: sparse and structured attention, low-rank and kernel approximations, memory mechanisms, recurrence and state-space models, and engineering practices for scaling.

## Why it matters

- Transformers are the backbone of many models we will run on Candle; optimizing them directly impacts latency and cost.

- The paper provides a taxonomy of efficiency techniques we can implement as alternative attention kernels or model transforms.

- Many approaches trade accuracy for computational savings — the survey helps select approaches that preserve accuracy while reducing memory/compute footprint.

- Offers references and baselines useful for benchmarking our kernels and model-level optimizations.

## Key technical takeaways

1. Sparse and structured attention (local windows, strided, block-sparse, BigBird-style) reduce quadratic memory while maintaining context for many tasks.

2. Low-rank and kernelized attention (Linformer, Performer, Nyströmformer, linear attention) approximate the attention matrix with sub-quadratic costs; suitability depends on sequence statistics and task.

3. State-space and recurrence-inspired replacements (S4, Mamba) provide linear-time alternatives for long contexts with different accuracy/efficiency trade-offs.

4. Memory and retrieval layers (external KV stores, memory tokens) decouple long-term context storage from per-step attention compute.

5. System-level optimizations: mixed-precision, fused ops, attention tiling/chunking, and KV-cache engineering are high-impact engineering moves for production workloads.

## Implementation steps for lightbulb

- Provide a pluggable attention backend API in Candle that supports: dense, local-window, block-sparse, and kernelized attention implementations.

- Implement a fused attention kernel optimized for the platform (Rust + SIMD) with optional FP16/BFloat16 support and tiled GEMM for long sequences.

- Add a low-rank attention prototype (Nyström or Linformer-style) as a model transform pass to reduce runtime memory and FLOPs for encoder-heavy workloads.

- Integrate state-space module experiments (e.g., a thin S4 wrapper) to evaluate long-context, streaming workloads.

- Add benchmarking harnesses for throughput/latency/memory across sequence lengths and batch sizes; include regression tests to flag accuracy regressions.

## Acceptance criteria

- Pluggable attention backend added with at least dense and local-window implementations, runnable end-to-end on a small LM.

- Fused dense attention kernel achieves >=1.2x throughput over a naive implementation on 512-token sequences.

- Low-rank attention prototype reduces peak memory by >=30% on encoder workloads with <=3% accuracy degradation on GLUE-like probes.

- Benchmark suite in CI that measures latency, throughput, and memory across representative workloads.
