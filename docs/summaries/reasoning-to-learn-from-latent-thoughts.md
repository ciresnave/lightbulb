# Reasoning to Learn from Latent Thoughts — summary

TL;DR

- Training on latent thought traces (partial chains not exposed at inference) can improve reasoning without forcing long outputs; selective supervision reduces verbosity.

Why it matters for lightbulb

- Suggests a middle path between full CoT and short answers: during fine-tuning or selection, use hidden intermediate states to guide learning/choice, but keep outputs concise.

Key points

- Latent thoughts act as auxiliary signals; exposing them only when helpful mitigates overthinking and prompt length bloat.
- Works well combined with early-exit and path compression.

Actionable next steps

- Add an evaluation mode that generates k partial chains, applies a latent-thought verifier, and selects a concise final answer.
- Expose a training stub that records hidden chains to a JSONL for future fine-tuning.

Acceptance criteria

- Match or exceed baseline CoT accuracy with ≤30% fewer output tokens on a small reasoning set.
