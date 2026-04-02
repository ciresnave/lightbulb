# Growing Transformers: Modular Composition and Layer-wise Expansion — summary

TL;DR

- Grows capacity by adding modules/layers on a frozen substrate; supports specialization without retraining the foundation.

Why it matters for lightbulb

- Suggests a path to plug-in specialized modules (experts/adapters) while keeping a stable core; aligns with MoE and adapter registries.

Actionable next steps

- Document module plug points and hot-swap policies; plan evals for specialized modules.
- Acceptance: functional module hot-swap demo with stable latency and measurable domain gains.
