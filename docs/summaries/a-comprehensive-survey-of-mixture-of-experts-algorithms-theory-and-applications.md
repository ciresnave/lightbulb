# A Comprehensive Survey of Mixture-of-Experts — summary

TL;DR

- Survey of MoE architectures, routing algorithms, training stability, and inference-time capacity management.

Why it matters for lightbulb

- MoE techniques directly inform our planned capacity-aware inference scheduler and routing strategies.

Actionable next steps

- Incorporate per-expert batching and capacity-aware routing into the Scheduler design doc.

Acceptance criteria

- Prototype routing policy that limits per-expert load and improves p95 latency on multi-token workloads.
