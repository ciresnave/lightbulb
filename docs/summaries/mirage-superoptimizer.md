# Mirage: A Multi-Level Superoptimizer for Tensor Programs — summary

TL;DR

- Mirage searches over program transformations (tiling, fusion, layout) to generate high-performance kernels across hardware. It auto-discovers optimized implementations that often outperform hand-tuned baselines.

Why it matters for lightbulb

- Guides an optional “kernel tuning” path and explains why we should prefer Candle-provided kernels while leaving room for auto-tuning experiments.

Key points

- Multi-level IR, cost modeling, search; cross-device portability.
- Benefits: performance portability; Risks: tuning time, complexity.

Actionable next steps

- Keep kernel-override hooks thin and optional; document how to plug in tuned kernels when available.
- Acceptance: optional doc-only spike; if enabled, demonstrate a micro-kernel win without changing APIs.
