# Reward Modeling as Reasoning — summary

TL;DR

- Reward models can be cast as latent reasoning processes that compare candidate outputs; careful aggregation and calibration reduce preference noise and improve selection.

Why it matters for lightbulb

- Lightbulb’s n-best decoding can use pluggable scorers (reward-like) to select or re-rank candidates without full RL pipelines.

Key points

- Pairwise and listwise comparisons reduce bias; uncertainty estimates help avoid overconfident bad picks.
- Compositional scoring (style + correctness + safety) improves robustness.

Actionable next steps

- Add a scoring API with: pairwise comparator, listwise aggregator, optional uncertainty calibration, and weighted composition.
- Expose CLI flags to switch between logprob-only, verifier, and reward-like scoring modes.

Acceptance criteria

- On a reasoning subset, re-ranking improves exact-match ≥2–3pp at equal or lower token cost vs. baseline sampling.
