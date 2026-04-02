# Efficient Reasoning Models — survey summary

TL;DR

- Surveys methods to improve reasoning efficiency: pruning thought chains, selective verification, self-consistency sampling budgets, curriculum and data selection, and anytime inference.

Why it matters for lightbulb

- Provides a menu of inference-time controls (budget caps, verification gates) that we can surface in the scheduler for predictable latency/quality trade-offs.

Actionable next steps

- Add budget-aware decoding knobs (max chains, max samples, verifier frequency) and export cost-quality curves.
- Acceptance: controllable latency with monotonic quality-cost trade-offs on small reasoning evals.
