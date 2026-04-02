# Multi-Head Temporal Latent Attention — summary

TL;DR

- Introduces temporal latent variables to capture long-range temporal dependencies with reduced attention cost; useful for tasks with strong temporal structure.

Why it matters for lightbulb

- Another design point for linear/structured mixers in hybrid schedules; potential benefits on temporal datasets.

Key points

- Latent compression of history; multi-head structure for diversity.
- Reduced compute with preserved temporal modeling quality.

Actionable next steps

- Track feasibility within Candle; treat as linear/structured mixer in policy experiments.
- Acceptance: maintain performance on temporal micro-benchmarks with reduced KV/compute.
