# Shorter is Better: Guiding Reasoning Models to Optimal Inference Length — summary

TL;DR

- Many reasoning tasks don’t benefit from very deep chains; dynamic stopping based on confidence/consistency can reduce overthinking while preserving or improving accuracy.

Why it matters for lightbulb

- Reinforces early-exit and depth adaptation policies: allocate extra depth only when confidence is low or inconsistency is detected.

Key points

- Use signals like entropy, self-consistency variance, and agreement with lightweight verifiers to stop early.
- Overthinking often harms final accuracy on specific benchmarks.

Actionable next steps

- Add a “shorter-better” heuristic to early-exit policy with a patience floor and maximum depth cap per input class.
- Acceptance: ≥15–25% compute reduction with neutral/improved accuracy on a small reasoning subset.
