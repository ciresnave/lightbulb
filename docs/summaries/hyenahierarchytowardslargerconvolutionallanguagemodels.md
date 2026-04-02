# HyenaHierarchyTowardsLargerConvolutionalLanguageModels

Full PDF: [Local PDF](file:///c%3A/Users/cires/OneDrive/Desktop/books%20and%20courses/Machine%20Learning/HyenaHierarchyTowardsLargerConvolutionalLanguageModels.pdf)

Markdown: ../papers/markdown/hyenahierarchytowardslargerconvolutionallanguagemodels.md

## TL;DR

Introduces Hyena Hierarchy: a convolutional-style, long-range sequence operator that interleaves implicitly parameterized long convolutions with multiplicative gating to approximate attention-quality modeling at subquadratic cost.

## Why it matters

- Provides a memory- and compute-efficient architecture for long-context sequence modeling which may reduce inference costs for Lightbulb when transformer attention becomes a bottleneck.
- Provides a memory- and compute-efficient architecture for long-context sequence modeling which may reduce inference costs for Lightbulb when transformer attention becomes a bottleneck. The paper reports matching Transformer perplexity on The Pile at ~335M params with ~20% fewer FLOPs at sequence length 2k, and empirical speedups of ~2x vs optimized attention at 8k and ~100x at 64k.

## Key technical takeaways

1. Hyena Hierarchy uses hierarchical convolutions and gating to capture long-range dependencies with lower memory than full attention.
1. Hyena Hierarchy uses hierarchical implicit convolutions (implicit filter parameterizations such as feed-forward nets or SSM-like parametrizations) and gating, enabling long filters without linear parameter growth.
2. Empirical scaling: matches transformer quality on WikiText103 and The Pile at sub-billion scales; shows a 20% reduction in FLOPs at 2k and orders-of-magnitude speed advantages at very long sequences.
3. Training notes: implicit parametrizations and gating require careful initialization; the authors use FFT-based fast convolution evaluation and tune gating and stability hyperparameters to avoid numerical issues.

4. Implementation caveats: the implicit filter parametrization choice (SSM-like or FFN mapping t->h) impacts expressivity and training stability; fast evaluation needs convolution algorithms (FFT/overlap-add) and attention-free matrix forms for memory savings.

## Implementation steps for Lightbulb

- Prototype a small Hyena layer in `research/models/` and compare throughput/memory vs an equivalent transformer block on a fixed long-context dataset.
- Evaluate hybrid architectures (Hyena + local attention) for streaming or retrieval-augmented generation tasks Lightbulb targets.

- Implement a minimal Hyena operator in `research/models/hyena.py` using an implicit filter parametrization (small FFN mapping positions to filter taps) and an overlap-add FFT convolution backend; add gating and test on synthetic recall/induction tasks.
- Reproduce a small-scale Language Modeling benchmark on WikiText103 (or a short subset of The Pile) at 335M param-equivalent configuration and measure perplexity and FLOPs to validate the ~20% FLOPs reduction claim.
- Add a micro-benchmark comparing throughput/memory vs a FlashAttention-based transformer at sequence lengths 2k, 8k, and 64k to validate speed/memory tradeoffs.

## Acceptance criteria

- A benchmark notebook exists showing latency, memory, and perplexity for Hyena vs transformer at sequence lengths {2k,8k,64k}.
- At least one hybrid architecture demonstrates lower memory use with comparable perplexity (within 5%) on a Lightbulb test set.
