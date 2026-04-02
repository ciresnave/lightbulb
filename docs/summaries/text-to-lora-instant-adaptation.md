# TextToLoRA: Instant Transformer Adaptation — summary

TL;DR

- Maps text descriptions to LoRA adapters for instant domain/task adaptation without fine-tuning per task. Enables quick specialization at inference.

Why it matters for lightbulb

- Suggests a loader/runtime path to hot-swap domain adapters based on request metadata, improving quality without retraining.

Key points

- Text-to-adapter mapping; minimal overhead for loading/switching adapters.
- Quality gains in domain-specific tasks with minimal setup.

Actionable next steps

- Define an adapter registry and selection policy (by prompt metadata) in the scheduler.
- Acceptance: show quality gains on domain micro-benchmarks with negligible TTFT increase when switching adapters.
