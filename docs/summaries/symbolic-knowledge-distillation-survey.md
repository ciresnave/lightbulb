# Symbolic Knowledge Distillation — survey summary

TL;DR

- Distills models into symbolic forms (rules/graphs/programs) that are compact and interpretable, enabling verification and targeted inference. Hybrids combine neural features with symbolic controllers.

Why it matters for lightbulb

- Informs prompt program executor and lightweight verifiers; symbolic controllers can drive early-exit or depth policies.

Key points

- Methods: rule extraction, program induction, graph distillation; hybrid neuro-symbolic systems.
- Benefits: interpretability, verification, sample efficiency; risks: coverage gaps, brittleness.

Actionable next steps

- Add hooks in prompt program executor for rule-based gating (exit/verify); log triggers.
- Acceptance: demonstrate a rule-based exit that reduces depth ≥15% with neutral accuracy on a small structured task.
