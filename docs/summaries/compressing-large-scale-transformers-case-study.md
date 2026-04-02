# Compressing Large-Scale Transformer Models — case study summary

TL;DR

- Case study on compressing BERT-like transformer models using pruning, quantization, and distillation; practical trade-offs and empirical results.

Why it matters for lightbulb

- Gives concrete recipe steps and metrics for compression workflows we plan to support (quantized loaders, pruning utilities).

Actionable next steps

- Extract a small workflow (prune -> quantize -> evaluate) and add it as a reproducible script in `docs/benchmarks/`.

Acceptance criteria

- A scripted compression run that reduces model size by >=2x with ≤2% relative downstream task degradation on a small benchmark.
