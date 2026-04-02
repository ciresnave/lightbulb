# A Systematic Analysis Of Hybrid Linear Attention (stub)

TL;DR

- Short summary: evaluation and analysis of hybrid linear attention methods and their runtime tradeoffs.

Why it matters

- Hybrid attention methods can reduce attention complexity and be useful for long-context CPU inference.

Key takeaways

- When hybrid linear attention works and when it breaks down; useful heuristics for runtime selection.

Implementation steps

1. Add hybrid-attention heuristics to `docs/summaries/hybrid-linear-attention-analysis.md` and runtime selection rules.

Acceptance criteria

- Heuristics added and a simple decision rule implemented in the roadmap’s kernel selection notes.
