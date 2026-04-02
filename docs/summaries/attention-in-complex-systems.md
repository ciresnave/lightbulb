# Attention Mechanisms and Their Application to Complex Systems — summary (contextual note)

TL;DR

- Attention formalizes selective information routing; in complex systems it enables modularity and competition/cooperation between components.

Why it matters for lightbulb

- Motivates modular runtime knobs: per-expert/per-head routing, contention metrics, and budgets that can be scheduled dynamically.

Key points

- Selection as soft routing; budgets/constraints ensure stability; emergent specialization.

Actionable next steps

- Record per-layer contention (tokens per head/expert) and expose a scheduler budget to cap hotspots.

Acceptance criteria

- Observability shows contention metrics; a budget cap reduces p95 latency under bursty loads without quality regression on held-out prompts.
