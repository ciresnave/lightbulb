# Hyena Hierarchy — summary

TL;DR

- Hyena is a subquadratic convolutional alternative to attention with implicit long-range interactions via learned filters and multiplicative gating, scaling well to very long contexts.

Why it matters for lightbulb

- Another candidate for the “linear” mixer role in hybrid schedules; pairs with periodic full attention to maintain retrieval and reduce KV.

Key points

- Long convolution kernels via structured transforms; hierarchical composition.
- Competitive perplexity with far lower memory than full attention at long context.

Actionable next steps

- Treat Hyena-like layers as linear mixers in policy; measure recall/perplexity under 3:1–6:1 schedules.
- Acceptance: similar targets to hybrid-attn with SSMs; document kernel support in Candle.
