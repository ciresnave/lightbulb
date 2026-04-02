# Trained Transformers Learn Linear Models in Context — summary

TL;DR

- Shows that transformers often perform in-context learning by implicitly fitting linear models to recent context. Suggests benefits from well-structured context and caching.

Why it matters for lightbulb

- Motivates prefix KV caching, prompt program structure, and careful chunking that preserves local linear relations.

Key points

- Linear regression behavior emerges; sensitive to context ordering and features.
- Predicts gains from curated prefix reuse and structured prompts.

Actionable next steps

- Prioritize prefix caching for repeated structures and add metrics to validate reuse effectiveness.
- Acceptance: TTFT drop ≥15% on repeated-prefix workloads with parity on outputs.
