# A Simple and Effective Pruning Approach for LLMs (Wanda) — summary

TL;DR

- One-shot unstructured/structured pruning using score S = |W| · ||X||₂ per-output row. Competitive with SparseGPT at 50–60% sparsity without retraining; supports 2:4 and 4:8 structured sparsity. Small calibration sets suffice.

Why it matters for lightbulb

- Enables fast, training-free pruning to cut compute/memory. Structured 2:4 sparsity can yield ~1.6× GEMM speedups on supported kernels, improving CPU throughput in our offline path.

Key findings

- Score per output row S_ij = |W_ij| · ||X_j||₂ with grouping by output is best.
- One-shot is strong; weight update often unnecessary until extreme sparsity.
- Works across LLaMA families; LoRA/full FT can recover further if desired.
- Structured (2:4, 4:8) performs well and maps to hardware-friendly kernels.

Actionable next steps

- Implement a pruning utility crate feature that computes Wanda scores from a small calibration set and emits a pruned tensor (and optional 2:4 mask).
- Acceptance: 2:4 path achieves ≥1.4–1.6× matmul speedup on a micro-benchmark with ≤1 ppl degradation on a tiny perplexity set.
- Provide a JSON manifest for pruned layers and simple loader hooks in `lightbulb` to apply masks at runtime.
