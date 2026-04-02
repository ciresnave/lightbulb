# Self-Attention with Relative Position Representations — summary

TL;DR

- Relative position encodings improve generalization to longer contexts and allow efficient reuse under sliding windows and cached KV, which benefits memory-bounded decoding and prefix reuse.

Why it matters for lightbulb

- Complements StreamingLLM-style sliding windows and prefix KV caching; supports stable attention under window shifts.

Key points

- Decomposes attention into content-content, content-position, and position-content terms.
- Reduces absolute-position overfitting; better extrapolation.

Actionable next steps

- Ensure loaders and attention paths handle relative/rotary schemes consistently with windowing.
- Acceptance: preserve perplexity and recall when enabling sliding windows vs full context on a micro-benchmark.
