# Transformers are Graph Neural Networks — summary

TL;DR

- Shows equivalence/relationship between transformers and GNN message passing; informs mixing and structural biases.

Why it matters for lightbulb

- Supports graph-structured prompts and suggests where sparse/structured attention could align with graph operations.

Actionable next steps

- Optional: sparse attention masks for graph neighborhoods in experiments; document feasibility within Candle.
- Acceptance: doc-only spike; if implemented, micro-benchmark parity on a small graph task.
