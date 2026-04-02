# R-KV: Redundancy-Aware KV Cache Compression — Summary

Source: R-KV — Redundancy-Aware KV Cache Compression for Training-Free Reasoning Models Acceleration (2024)

TL;DR

- At decode time, score past tokens by importance (attention-based) and redundancy (cosine similarity) and keep only the most useful/least redundant entries under a KV budget.
- Joint score: Z = λ·I − (1−λ)·R. Authors report λ≈0.1, redundancy pooling with α=8, and a small safety buffer Bbuffer≈128 tokens.
- Achieves near/full accuracy with 10–34% KV budget and up to 6.6× throughput on long reasoning sequences (paper’s reported results), with caveats for paged attention integration.

Key ideas

- Importance Ih via attention weights; Redundancy Rh via cosine similarity between candidate token’s key and representative vectors (e.g., max-pooled per block/head with GQA-friendly pooling).
- Maintain a small buffer of the most recent tokens to avoid dropping very fresh context (Bbuffer).
- Selection under a target budget picks tokens with highest Z while enforcing head/layer constraints; integrates with multi-head/GQA via pooling.

Actionable for Lightbulb

- Add an experimental KvPager policy: r-kv eviction.
  - Inputs: budget fraction b in (0.1–0.5), λ in (0–1), α (redundancy pooling factor), Bbuffer, selection cadence (every N decode steps), and per-layer/page limits.
  - Implementation: compute importance from attention weights already produced during decode; compute redundancy using cosine similarity in key space with light pooled reps; select survivors per budget.
- Acceptance criteria (guarded by a feature flag):
  - On curated long prompts (math/code/QA), with budget b∈{0.2,0.34}, final outputs match full-KV baseline within a small tolerance (token- or likelihood-based).
  - With b≈0.34, memory reduction ≥ 60% and end-to-end throughput improves ≥ 1.5× on CPU for long decodes (document measurements; GPU TBD).
  - No cache corruption across 10k-token soaks; deterministic with fixed seeds.

Risks/notes

- Paged-attention compatibility: selection must map cleanly to pages; otherwise fragmentation overhead can offset gains. Consider page-aligned scoring to minimize reshuffles.
- Cost of similarity computations: amortize by scoring at intervals and using pooled reps per block/page rather than per-token full sims.
- Numerical drift: keep a conservative baseline path and tight tests.

Citation

- R-KV: Redundancy-Aware KV Cache Compression for Training-Free Reasoning Models Acceleration (2024).
