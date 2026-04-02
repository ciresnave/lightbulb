# Knowledge Distillation for LLMs — survey summary

TL;DR

- KD transfers knowledge from a larger teacher to a smaller student via soft targets, intermediate hints, or behavior cloning. For inference frameworks, KD defines how to produce strong small models that run cheaper while retaining accuracy.

Why it matters for lightbulb

- Guides model zoo curation and evaluation: students distilled from high-quality teachers are prime candidates for CPU-only/offline paths.

Key points

- Techniques: vanilla KD, intermediate feature/attention KD, response-based KD, self-distillation, data augmentation.
- For LLMs: instruction KD, rationales/CoT KD, preference/distillation from alignment models.

Actionable next steps

- Establish eval templates for distilled students (ppl, QA, reasoning); document recommended formats and Candle loaders.
- Acceptance: show a distilled student meeting target latency/memory on CPU with ≤X% drop vs baseline teacher on a small eval.
