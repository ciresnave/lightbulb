# HNSW: Efficient Approximate Nearest Neighbor Search — summary

TL;DR

- Hierarchical Navigable Small World (HNSW) graphs enable fast ANN search with strong recall and sublinear query time.

Why it matters for lightbulb

- Provides a retrieval backbone for prompt-program tools (RAG, KG augment) with predictable latency.

Actionable next steps

- Define a retrieval tool interface; log recall/latency; allow local index use in offline mode.
- Acceptance: demonstrate a small HNSW-backed RAG toy with stable latency and measurable recall.
