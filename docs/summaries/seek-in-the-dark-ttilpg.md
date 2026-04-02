# Seek in the Dark: Test-Time Instance-Level Policy Gradient — summary

TL;DR

- Optimizes a per-instance policy at test time via lightweight policy gradient, improving reasoning by exploring a small local policy space.

Why it matters for lightbulb

- Provides a pathway for per-request adaptation without training the base model; can tune exit thresholds or sampling parameters on-the-fly.

Key points

- Uses small exploration budgets; rewards incorporate accuracy/verifier signals.
- Sensitive to budget and stability; needs guardrails.

Actionable next steps

- Prototype a tiny per-request tuner for a couple of policy knobs (e.g., temperature, patience), capped by a budget.
- Acceptance: measurable accuracy gains on a small reasoning subset with bounded latency overhead.
