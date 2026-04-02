# Rereading Improves Reasoning — summary

TL;DR

- Multi-pass reasoning—re-encoding context or prior drafts—improves final accuracy by allowing correction and refinement. It can be scheduled selectively based on uncertainty.

Why it matters for lightbulb

- Suggests a scheduler option for selective rereads or partial re-prefill to correct drift without always paying full cost.

Key points

- Reread triggers: low confidence, disagreement across samples, detected contradictions.
- Balance cost via chunked rereads or partial KV refresh.

Actionable next steps

- Add a “reread-on-low-confidence” policy that redoes a short prefill segment before continuing.
- Acceptance: improved accuracy on a small reasoning subset with <20% extra tokens on the cases where reread triggers.
