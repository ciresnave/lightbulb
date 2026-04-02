# A Sober Look at Progress in Language Model Reasoning — summary

TL;DR

- Highlights pitfalls in evaluation, data leakage, and reproducibility for reasoning; recommends stricter protocols.

Why it matters for lightbulb

- Guides our eval harness: fixed seeds, held-out tasks, and robust scripts for CI/certification of features like early-exit or dynamic depth.

Actionable next steps

- Provide deterministic eval scripts with seed control, task splits, and report templates for latency/quality curves.
- Acceptance: reproducible runs across machines with small variance in metrics.
