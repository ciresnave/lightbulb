# Towards Understanding How Attention Mechanism Works in Deep Learning — summary

TL;DR

- Attention patterns correlate with token saliency and structure but are not direct explanations; stability and calibration matter.

Why it matters for lightbulb

- Suggests observability hooks: record attention entropy, head sparsity, and calibration to inform early-exit/skip/repeat policies.

Key points

- Attention as kernel regression/soft retrieval; head specialization; entropy as a confidence proxy; pathological uniform/peaky cases.

Actionable next steps

- Add optional per-layer attention entropy metrics during decode steps (behind a flag) and log via `tracing`.
- Feed these signals into dynamic compute policies (exit/skip/repeat) once implemented.

Acceptance criteria

- Metrics appear in logs and are shaped `(layer, head)`; simple thresholds can be computed without overhead >5% on CPU.
