# FromComplexToSimple — Unraveling the Cognitive Tree for Reasoning with Small Language Models

Full PDF: [Local PDF](file:///c%3A/Users/cires/OneDrive/Desktop/books%20and%20courses/Machine%20Learning/FromComplexToSimple-UnravelingTheCognitiveTreeForReasoningWithSmallLanguageModels.pdf)

Markdown: ../papers/markdown/fromcomplextosimple-unravelingthecognitivetreeforreasoningwithsmalllanguagemodels.md

## TL;DR

Proposes CogTree, a dual-process (intuitive + reflective) iterative framework that decomposes complex problems into simple sub-questions and uses a reflect-and-score loop to improve answers — enabling small (<=7B) models to approach larger-model reasoning performance.

## Why it matters

- Offers an engineering-friendly recipe to boost reasoning in small models by orchestrating generation (fast, multi-sample intuition) and selective reflective scoring/verification — aligns with Lightbulb goals to extract practical, CPU-friendly inference improvements.

## Key technical takeaways

1. CogTree constructs a tree of sub-questions: root is the original query; leaves are solvable sub-questions answered directly.
2. Two modules: Intuitive system (fast multi-sample generation) and Reflective system (comparative scoring and iterative refinement) — iterative interplay drives improvements.
3. Demonstrates that careful orchestration and scoring can let small models match much larger models on reasoning benchmarks when the pipeline is well-designed.

## Implementation steps for Lightbulb

- Implement a lightweight CogTree harness: (a) expand queries into candidate sub-questions, (b) run Intuitive (multi-sample) generation with our scheduler, (c) run Reflective scoring function (reuse RL/eval harness) and prune/guide tree growth.
- Expose the pipeline via a CLI `lightbulb-cogtree` tool that accepts a prompt, runs the iterative loop, and returns the final answer plus trace/logs for diagnostics and caching into prefix KV.
- Add unit tests simulating small reasoning problems and an acceptance test comparing outputs to a reference generation from a larger model.

## Acceptance criteria

- A working CogTree demo that runs on a 7B model locally and achieves comparable scores to the paper's reported small-model baseline on a held-out reasoning subset.
- The pipeline produces reproducible traces (intuitions + reflections) and integrates with the Scheduler and KvPager so traces can be inspected and cached.
