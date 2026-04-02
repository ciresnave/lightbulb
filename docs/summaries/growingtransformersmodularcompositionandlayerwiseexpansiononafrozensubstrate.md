# Growing Transformers: Modular Composition and Layer-wise Expansion on a Frozen Substrate

Full PDF: [Local PDF](file:///c%3A/Users/cires/OneDrive/Desktop/books%20and%20courses/Machine%20Learning/GrowingTransformersModularCompositionAndLayerWiseExpansionOnAFrozenSubstrate.pdf)

Markdown: ../papers/markdown/growingtransformersmodularcompositionandlayerwiseexpansiononafrozensubstrate.md

## TL;DR

Shows constructive scaling via frozen embeddings: a frozen substrate lets you compose specialist modules and grow a model layer-by-layer, enabling modular composition and progressive depth expansion without full retraining.

## Why it matters

- Offers alternative, efficient scaling workflows (module merging, layer-wise growth) that could lower compute/resource costs for Lightbulb experiments and enable safer iterative development.

## Key technical takeaways

1. Frozen deterministic embeddings can serve as a universal substrate enabling module composition and merging with minimal interference.
2. Layer-wise constructive training yields stable convergence and allows progressive depth expansion with manageable compute.
3. Post-training merging of specialist models (logit averaging) can improve capability without catastrophic forgetting.

## Implementation steps for Lightbulb

- Add an `experiments/growing` playground: (a) run a small frozen-embedding Transformer, (b) train small specialist modules, (c) test merging and layer-wise stacking on toy tasks.
- Measure inference/merge stability and the cost vs end-to-end retraining baseline.

## Acceptance criteria

- A short experiment that demonstrates successful module merging or layer-wise growth on a toy dataset with code in `experiments/growing/`.
- Documented recipe for repeating the experiment locally (CPU-friendly where possible).
