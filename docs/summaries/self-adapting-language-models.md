# Self-Adapting Language Models (SALM) — summary

TL;DR

- SALM adapts inference compute at test-time: dynamically adjusts depth/width, cache reuse, and routing based on input difficulty and confidence. It yields latency/efficiency gains by allocating more compute where needed while keeping easy cases cheap.

Why it matters for lightbulb

- Directly informs scheduler policies: per-token/per-sequence adaptation hooks (early exit, repetition/skip, selective experts), and caching strategies across steps/requests.

Key points

- Signals for adaptation: entropy, margin/max-prob, variance over time, agreement across heads/experts; lightweight probes guide compute budget.
- Mechanisms: early exit, layer skipping/repetition, dynamic batch resizing, expert activation budgets, and cache reuse across similar prefixes.
- Training vs zero-shot: benefits arise even without extra training via calibration/thresholding; joint training improves stability.

Actionable next steps

- Unify adaptation signals (entropy + patience) behind a Policy trait; plug into early-exit and (future) depth/width controls.
- Export per-request traces (signals, chosen depth, time per token) to drive threshold tuning.
- Acceptance: ≥20–30% mean layer/compute reduction with ≤2% accuracy loss on a small mixed eval; stable under batch load.
