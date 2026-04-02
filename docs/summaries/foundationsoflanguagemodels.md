# FoundationsOfLanguageModels

Full PDF: [Local PDF](file:///c%3A/Users/cires/OneDrive/Desktop/books%20and%20courses/Machine%20Learning/FoundationsOfLanguageModels.pdf)

Markdown: ../papers/markdown/foundationsoflanguagemodels.md

## TL;DR

An accessible, chaptered primer on LLM foundations (pretraining, generative models, prompting, alignment, inference) that organizes core techniques and practical considerations — a good reference for team onboarding and engineering checklists.

## Why it matters

- Collects dependable, implementation-oriented knowledge about scaling, prompting, alignment, and inference; useful as a living reference for Lightbulb engineers implementing model-loading, inference, and alignment pipelines.

## Key technical takeaways

1. Covers pretraining approaches, scaling considerations, and practical inference techniques including efficient decoding and batching.
2. Summarizes prompting/chain-of-thought, alignment/FT/HF methods, and inference-time trade-offs (latency vs quality).
3. Acts as a catalog of practical recipes suitable for engineering (data curation, optimization tips, inference patterns).

## Implementation steps for Lightbulb

- Add a short `docs/engineering/llm-foundations.md` distilled from this resource with the team's preferred recipes (tokenization choices, decode hyperparameters, scoring/regression tests).
- Use the book's inference chapter to validate and refine Candle loader/inference options and scheduler defaults (batch sizes, timeout/TTFT knobs).

## Acceptance criteria

- A one-page distilled engineering checklist is present in `docs/engineering/` referencing key sections and recommended settings.
- At least one cookbook item (e.g., decode params, batching recipe) implemented and regression-tested.
