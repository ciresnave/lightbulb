# Reassessing Layer Pruning in LLMs — summary

TL;DR

- Tail-prune the last ~25% of layers, then fine-tune only the lm head + last 1–3 layers. This simple strategy rivals or beats sophisticated metrics; iterative pruning shows limited benefit.

Why it matters for lightbulb

- Offers a pragmatic recipe to shrink models for CPU-only/offline use without heavy retraining. Partial-layer FT is efficient and effective.

Key findings

- Reverse-order (tail) pruning is highly competitive across models.
- Partial-layer FT > LoRA in this context; minimal SFT tokens can recover quality.
- Produced strong pruned Llama-3.1-6.3B variants with lightweight fine-tuning.

Actionable next steps

- Add a pruning profile to remove the final K layers (configurable, default ~25%), with optional partial-layer FT hooks (document-only for now).
- Acceptance: ≤2% average accuracy drop on a small mixed eval; provide a script to run brief FT on lm head + last 1–3 layers to recover where needed.
