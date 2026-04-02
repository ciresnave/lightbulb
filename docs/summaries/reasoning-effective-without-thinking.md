# Reasoning Models Can Be Effective Without Thinking — summary

TL;DR

- Shallow reasoning or even direct answers can perform well on many instances; heavy chains are not universally required. Allocate deep compute selectively.

Why it matters for lightbulb

- Reinforces dynamic compute allocation: prefer short responses unless confidence is low or verification fails.

Key points

- Many tasks have easy instances; overthinking wastes compute.
- Confidence and verification signals guide when to invest more.

Actionable next steps

- Provide a “shallow-first” decode mode that attempts brief chains, escalating only on low-confidence.
- Acceptance: ≥20% compute savings with neutral accuracy on mixed reasoning workloads.
