# Reasoning with Language Model Prompting — survey summary

TL;DR

- Reviews prompting families (CoT, SC, ToT, GoT/HoT, CRIT, programmatic prompts) and control signals to guide reasoning, sampling, and tool use.

Why it matters for lightbulb

- Grounds the prompt program executor: structured sequences/branches with per-step budgets and verifiers; integrates with early-exit and scheduling.

Actionable next steps

- Extend the prompt program schema with step-level budgets, retry policies, and logging hooks.
- Acceptance: run a 5–8 step CRIT-like template end-to-end with ≤5% overhead vs single-turn on equal tokens.
