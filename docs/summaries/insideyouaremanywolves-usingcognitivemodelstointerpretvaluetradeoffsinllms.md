# InsideYouAreManyWolves-UsingCognitiveModelsToInterpretValueTradeOffsInLLMs

Full PDF: [Local PDF](file:///c%3A/Users/cires/OneDrive/Desktop/books%20and%20courses/Machine%20Learning/InsideYouAreManyWolves-UsingCognitiveModelsToInterpretValueTradeOffsInLLMs.pdf)

Markdown: ../papers/markdown/insideyouaremanywolves-usingcognitivemodelstointerpretvaluetradeoffsinllms.md

## TL;DR

Applies cognitive multi-agent metaphors to interpret value trade-offs within LLMs, offering targeted probes and attributions that reveal competing internal objectives.

## Why it matters

- Provides practical interpretability techniques that help detect internal preference conflicts — important for Lightbulb's evaluation and alignment checks.

## Key technical takeaways

1. Decomposition approach: treats model internals as overlapping 'subagents' and proposes attribution/ablation experiments to quantify their influence.
2. Diagnostic tasks: constructs probe datasets and interventions to reveal where trade-offs (accuracy vs safety) manifest inside the network.
3. Metrics: recommends per-layer contribution metrics instead of relying solely on end-task outcomes.

## Implementation steps for Lightbulb

- Build a small probe suite in `tools/probes/` to run ablation and attribution experiments on target models.
- Run the probe suite on one representative model and capture per-layer attribution scores; store outputs in the evaluation artifacts directory.

## Acceptance criteria

- Probe suite runs end-to-end and produces per-layer attribution tables for two diagnostic tasks.
- A short notebook documents one discovered internal trade-off and a tested mitigation (e.g., fine-tuning or routing adjustment).
