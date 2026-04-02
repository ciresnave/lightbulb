# A Systematic Analysis of Hybrid Linear Attention — summary

TL;DR

- Interleaving linear-time mixers with full attention achieves Transformer-level recall at much lower KV memory. Best ratios are typically 3:1–6:1 (linear:full). Gated, hierarchical recurrent backbones (e.g., HGRN-2, GatedDeltaNet) matter more than the exact linear mixer choice.

Why it matters for lightbulb

- Hybrid attention can reduce KV cache by 4–7x while keeping language modeling loss nearly flat and preserving retrieval recall when a periodic full-attention layer is used. This directly reduces memory pressure in the scheduler/KV pager and boosts long-context throughput on CPU.

Key findings

- Standalone ranking of linear mixers does not predict hybrid performance; gating + hierarchical recurrence + controlled forgetting are the key ingredients.
- Recall improves as the schedule shifts toward more frequent full attention; LM loss remains mostly flat across 3:1–6:1 ratios.
- Pareto analysis shows strong trade-offs at 3:1–6:1; detailed FLOP comparisons provided for 340M/1.3B.
- Achieves 4–7x KV memory reduction vs pure Transformer at comparable recall.

Actionable next steps

- Add a “hybrid-attn policy” slot to the scheduler/config to interleave linear-like layers with full attention (simulated via keep/skip for now). Acceptance: offer ratios {3:1, 4:1, 6:1} and log applied pattern per decode.
- Target: ≥4× reduction in effective KV memory at ≤0.5 ppl delta on a small LM eval and ≤3% drop on a recall-sensitive micro-benchmark.
- Document required architectural ingredients (selective gating, hierarchical recurrence, controlled forgetting) and track Candle support for such backbones.
