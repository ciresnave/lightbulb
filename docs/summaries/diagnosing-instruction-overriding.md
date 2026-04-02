# Diagnosing Instruction Overriding in Reasoning Models — summary

TL;DR

- Identifies when models override system/user instructions during multi-step reasoning; proposes detection and mitigation strategies.

Why it matters for lightbulb

- Suggests monitors for instruction drift within prompt programs and decode loops; can trigger interventions or exits.

Actionable next steps

- Add a simple instruction-alignment score per step; trigger restatement or stop on severe drift.
- Acceptance: reduce instruction overrides on a synthetic eval with minimal extra cost.
