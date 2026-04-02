# AGENTKB: Cross-Domain Experience for Agentic Problem Solving — summary

TL;DR

- Builds a knowledge base of past agent experiences for retrieval and reuse across domains; improves efficiency and success.

Why it matters for lightbulb

- Suggests a memory store keyed by task signatures and outcomes for tool-use prompts; enables reuse in offline CPU settings.

Actionable next steps

- Add a simple experience store (local JSON/SQLite) with retrieval-by-similarity; log successes/failures.
- Acceptance: faster convergence on repeated task families with reduced steps.
