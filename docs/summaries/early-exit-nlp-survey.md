# A Survey of Early Exit Deep Neural Networks in NLP — summary

TL;DR

- Early exit enables input-adaptive depth for latency/efficiency. Effective criteria include entropy/max-prob with patience, distribution/similarity checks, ensembles, or learned uncertainty; thresholds can be static or dynamic (MAB/UCB) and adapted to domain.

Why it matters for lightbulb

- Integrates naturally with the scheduler: decide to halt decoding early per-step or per-chunk, or to stop forward depth per-token. Supports anytime prediction, OOD robustness, and distributed/edge-cloud splits.

Key findings

- Training: separate vs joint; knowledge distillation variants.
- Criteria: entropy, max-prob, patience; distribution/similarity; ensemble; learned confidence.
- Thresholds: static vs dynamic via MAB/UCB; domain adaptation via threshold tuning and feature alignment (e.g., GANs).
- Applications: reduced latency, robustness, split inference, self-speculative decoding.

Actionable next steps

- Implement entropy-based per-token exit with patience and optional MAB-based dynamic thresholds.
- Acceptance: ≥25–40% average layer reduction with ≤2% accuracy loss on a small eval mix; export per-request exit traces and histograms.
