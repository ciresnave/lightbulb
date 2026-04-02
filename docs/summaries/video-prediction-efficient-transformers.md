# Video Prediction by Efficient Transformers — summary

TL;DR

- Applies efficient transformer variants to video prediction and spatio-temporal modeling, with memory/time optimizations.

Why it matters for lightbulb

- Provides evidence for hybrid attention scheduling and intermittent full-attention layers in temporal domains.

Actionable next steps

- Add a synthetic temporal benchmark to `docs/benchmarks/temporal/` and record hybrid vs full-attention memory trade-offs.

Acceptance criteria

- Demonstrate lower memory usage on a synthetic task using a hybrid attention schedule with similar performance.
