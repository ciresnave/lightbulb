# Graph-enhanced Large Language Models in Asynchronous Plan Reasoning

Full PDF: [Local PDF](file:///c%3A/Users/cires/OneDrive/Desktop/books%20and%20courses/Machine%20Learning/GraphEnhancedLargeLanguageModelsInAsynchronousPlanReasoning.pdf)

Markdown: ../papers/markdown/graphenhancedlargelanguagemodelsinasynchronousplanreasoning.md

## TL;DR

PLaG (Plan Like a Graph) augments LLM reasoning with explicit graph representations for asynchronous plan reasoning and shows substantial performance gains on the AsynchHow benchmark, while highlighting limits as task complexity increases.

## Why it matters

- Demonstrates that combining symbolic/graph structures with LLM prompts can improve planning, which is relevant for Lightbulb projects that aim to orchestrate multi-step plans or tool-use where structural constraints matter.

## Key technical takeaways

1. Constructing a graph representation of plan states and feeding structured prompts (PLaG) leads to better correctness/efficiency in asynchronous planning tasks.
2. LLMs benefit from graph-augmented context but still degrade with rising complexity — hybrid symbolic-LLM systems can be a pragmatic compromise.

## Implementation steps for Lightbulb

- Prototype a small PLaG-style wrapper: parse task descriptions into a simple graph, serialize into a compact prompt template, and feed to a local LLM to evaluate scheduling strategies.
- Add a microbenchmark (AsynchHow-like) to measure gains vs pure prompt-only strategies and log plan traces for analysis.

## Acceptance criteria

- A small prototype shows improved plan validity/efficiency vs baseline prompts on a handful of synthetic asynchronous planning tasks.
- Prototype includes unit tests validating graph serialization and end-to-end traces.
