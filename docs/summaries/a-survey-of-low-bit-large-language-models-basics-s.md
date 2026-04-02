# A Survey Of Low-Bit Large Language Models — Basics (stub)

TL;DR

- Survey of low-bit quantization methods and associated system-level considerations for deploying LLMs at low precision.

Why it matters

- Directly informs quantized loader design, integer GEMM kernels, and calibration tools for correctness vs. speed tradeoffs.

Key takeaways

- Overview of quantization schemes, per-channel calibration, and mixed-precision strategies.

Implementation steps

1. Consolidate recommendations into `docs/summaries/low-bit-llms-survey.md` and road-map quantization tasks.

Acceptance criteria

- A short matrix of quantization schemes vs supported kernels and expected speedups.
