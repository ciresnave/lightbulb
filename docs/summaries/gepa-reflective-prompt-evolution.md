# GEPA: Reflective Prompt Evolution — summary

TL;DR

- Evolves prompts via reflective feedback loops to improve task performance over time with limited supervision.

Why it matters for lightbulb

- Suggests a background tuner for prompt templates in the prompt program executor; captures wins offline and reuses online.

Actionable next steps

- Add a prompt-evolution mode (offline) that logs variants and outcomes; load best variants at runtime.
- Acceptance: improved accuracy on targeted tasks with no regression on others; audit trail of prompts and scores.
