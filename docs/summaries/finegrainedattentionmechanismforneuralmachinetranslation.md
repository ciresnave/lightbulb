# FineGrainedAttentionMechanismForNeuralMachineTranslation

Full PDF: [Local PDF](file:///c%3A/Users/cires/OneDrive/Desktop/books%20and%20courses/Machine%20Learning/FineGrainedAttentionMechanismForNeuralMachineTranslation.pdf)

Markdown: ../papers/markdown/finegrainedattentionmechanismforneuralmachinetranslation.md

## TL;DR

 (one-line summary here)

## Why it matters

- Introduces per-dimension (2D) attention where each dimension of the context vector receives its own attention weight, improving alignment and translation quality — useful idea for Lightbulb when exploring richer attention parameterizations or low-rank per-dimension modulation.

## Key technical takeaways

1. Fine-grained (2D) attention assigns attention scores per-dimension of context vectors rather than a single scalar per token, allowing the model to exploit internal structure of context vectors.
2. Empirical gains on En-De and En-Fi indicate BLEU improvements and better alignment interpretations.
3. Analysis shows the mechanism can reveal how different vector subspaces contribute to translation decisions.

## Implementation steps for Lightbulb

- Prototype a lightweight 2D-attention module (dimension-wise gating or low-rank per-dim scores) in an experimental transformer block and benchmark on a small translation or synthetic alignment task (CPU-only for initial runs).
- Provide a feature-flagged path so we can toggle between scalar attention and fine-grained attention to measure perf/quality tradeoffs.

## Acceptance criteria

- Demonstrate measurable BLEU (or proxy alignment) improvement on a small dataset vs baseline, or document no gain plus resource-cost metrics.
- The feature-flagged implementation can be toggled without breaking the baseline path and passes unit tests.
