# From Complex to Simple: Unraveling the Cognitive Tree for Reasoning with Small LMs — summary

TL;DR

- A “cognitive tree” strategy decomposes problems into simpler sub-questions and revisits them adaptively; small models benefit from structured decomposition with minimal overhead.

Why it matters for lightbulb

- Aligns with our scheduler’s planned orchestration: subtask routing, limited breadth search, and reusable context segments.

Key points

- Hierarchical planning with feedback outperforms flat CoT for small models.
- Reuse sub-answers to avoid recomputation; cache verified snippets.

Actionable next steps

- Provide a simple decomposition DSL (bullet steps) and an executor that queries the model per step with caching.
- Add a cache key design (prompt + step hash) to store verified sub-answers.

Acceptance criteria

- On a small math/logic set, show accuracy gains for a ≤1B model using the decomposition executor at ≤1.2x token cost.
