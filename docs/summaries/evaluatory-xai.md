# Evaluatory XAI

Full PDF: [Local PDF](file:///c%3A/Users/cires/OneDrive/Desktop/books%20and%20courses/Machine%20Learning/Evaluatory%20XAI.pdf)

Markdown: ../papers/markdown/evaluatory-xai.md

## TL;DR

Proposes a human–AI interaction (HAII) evaluation paradigm for XAI that links five measurable factors (knowledge, trust, performance, probability estimates, and mental models) and specifies a training/testing/evaluation workflow to compare explanation methods across users and tasks.

## Why it matters

- Provides a practical, psychology-grounded evaluation framework so we can test which explanation styles actually help users make better decisions — crucial for Lightbulb features that expose model reasoning or explainability to users.

## Key technical takeaways

1. Evaluation should be task- and user-specific; no single explanation format fits all users or contexts.
2. Five evaluation axes (knowledge, trust, performance, probability estimates, mental model) form a causal framework for measuring explanation utility.
3. The paper recommends an experimental paradigm with training, testing, and evaluation phases and highlights risks like information overload and persuasive/misleading explanations.

## Implementation steps for Lightbulb

- Add a small XAI evaluation harness in our QA tools that: (a) runs controlled A/B tests of explanation variants, (b) collects measures along the five axes, and (c) produces a short report.
- Instrument the UI/CLI debug outputs to expose explanation formats (saliency, concept, counterfactual) behind feature flags so we can quickly A/B different formats with the harness.
- For adaptive explanation delivery, prototype a prefix–explanation cache keyed by common system prompts (see Roadmap M2 prefix KV caching) and track interaction outcomes for later optimization.

## Acceptance criteria

- End-to-end XAI harness can run a small user/emulation study and output the five-axis metrics in JSON.
- For at least two explanation variants on a small task, the harness produces measurable differences (e.g., change in performance or calibration) or documents that they are indistinguishable.
- No linter or markdown errors in the file; `file://` link matches the path in `docs/papers/pdf-index.txt`.
