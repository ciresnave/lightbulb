# Reasoning LLMs Are Wandering Solution Explorers — summary

TL;DR

- Many reasoning trajectories meander; structured search, pruning, and guidance (sketch first, solve later) can reduce wasted tokens and improve success rates.

Why it matters for lightbulb

- Reinforces our dynamic compute stack: breadth-k then prune via verifier; add sketch planning before detailed decoding.

Key points

- Warm-up with outline/sketch improves downstream accuracy; pruning on inconsistencies saves budget.
- Diverse sampling helps until diminishing returns; enforce diversity-aware penalties.

Actionable next steps

- Add a two-stage “sketch-then-solve” mode that first generates an outline with a short budget, then expands top-1–2 candidates.
- Implement diversity-aware sampling (penalties on n-gram or semantic overlap across candidates).

Acceptance criteria

- For math or logic mini-task, achieve equal/greater accuracy with ≥20% fewer tokens vs. naive self-consistency.
