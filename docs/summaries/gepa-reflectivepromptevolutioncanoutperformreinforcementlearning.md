# GEPA — Reflective Prompt Evolution Can Outperform Reinforcement Learning

Full PDF: [Local PDF](file:///c%3A/Users/cires/OneDrive/Desktop/books%20and%20courses/Machine%20Learning/GEPA-ReflectivePromptEvolutionCanOutperformReinforcementLearning.pdf)

Markdown: ../papers/markdown/gepa-reflectivepromptevolutioncanoutperformreinforcementlearning.md

## TL;DR

GEPA is a prompt optimizer that uses natural-language reflection and genetic/Pareto combination to evolve prompts and system-level trajectories; it often outperforms RL methods while using far fewer rollouts by leveraging language for interpretable updates.

## Why it matters

- Shows that natural-language-based reflection + evolutionary search can be a more sample-efficient way to improve LLM-driven systems than black-box policy gradients — a practical path for Lightbulb when optimizing prompt-based controllers or tool-using agents without heavy RL infrastructure.

## Key technical takeaways

1. GEPA samples system-level trajectories, reflects on failures in natural language, proposes prompt updates, and tests them — building a feedback loop that converges rapidly with few rollouts.
2. Combines Pareto-frontier selection with genetic-style combination of complementary prompt updates to retain diverse, high-performing prompts.
3. Empirically outperforms GRPO and MIPROv2 on several tasks, with up to 35x fewer rollouts.

## Implementation steps for Lightbulb

- Prototype a GEPA-style prompt optimizer integrated into our test harness: (a) sample trajectories under current prompt, (b) generate reflective diagnostics using the same or a smaller LLM, (c) propose prompt edits, (d) evaluate and combine winners via Pareto selection.
- Use GEPA as a lightweight alternative to heavy RL for tuning system prompts and tool orchestration in local demos.

## Acceptance criteria

- Demonstrate a >10% improvement over baseline on a small task dataset using GEPA-style optimization with <=100 rollouts.
- The optimizer is packaged as a script `scripts/gepa_optimize.py` and includes a simple logging/trace format for evaluation.
