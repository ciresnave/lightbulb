# Missing Premise Exacerbates Overthinking: Are Reasoning Models Losing Critical Thinking Skill?

**Original PDF:** [MissingPremiseExacerbatesOverthinking-AreReasoningModelsLosingCriticalThinkingSkill.pdf](../../../../Desktop/books%20and%20courses/Machine%20Learning/MissingPremiseExacerbatesOverthinking-AreReasoningModelsLosingCriticalThinkingSkill.pdf)
**Original Markdown:** [missingpremiseexacerbatesoverthinking-arereasoningmodelslosingcriticalthinkingskill.md](../papers/markdown/missingpremiseexacerbatesoverthinking-arereasoningmodelslosingcriticalthinkingskill.md)

---

## TL;DR

This paper reveals that reasoning-focused LLMs tend to produce excessively long, redundant responses to ill-posed questions with missing premises (MiP), failing to efficiently identify unsolvable queries. Non-reasoning models, in contrast, are more robust, quickly recognizing and abstaining from such queries. The phenomenon, termed MiP-Overthinking, highlights a lack of critical thinking in current reasoning model training.

## Why it matters (for Candle and reproducible ML)

- Candle and similar Rust ML libraries benefit from models that can efficiently handle ill-posed queries, improving benchmarking and telemetry.
- Understanding MiP-Overthinking is crucial for designing reproducible experiments and robust agentic workflows.
- The findings inform better training recipes and evaluation metrics for reasoning models in open-source ML infrastructure.

## Key technical takeaways

- **MiP-Overthinking:** Reasoning models generate 2–4× longer responses for MiP questions, often failing to abstain even when recognizing unsolvability.
- **Critical thinking gap:** Non-reasoning models outperform reasoning models in identifying and abstaining from ill-posed queries.
- **Dataset construction:** MiP datasets created via rule-based generation, body-question swapping, and essential-premise removal across math benchmarks.
- **Metrics:** Response length, abstain rate, and accuracy on well-defined questions; step-level similarity analysis reveals redundancy in reasoning chains.
- **Training flaw:** Overthinking is propagated through RL and distillation, indicating a need for improved length constraints and critical thinking incentives.

## Implementation steps (for Candle or similar)

1. **Benchmark models** on MiP datasets to evaluate response length, abstain rate, and critical thinking.
2. **Incorporate length constraints** and abstention incentives in RL and SFT training recipes.
3. **Analyze reasoning chains** for redundancy and self-doubt patterns using step-level similarity metrics.
4. **Design evaluation protocols** that reward efficient, critical responses to ill-posed queries.
5. **Integrate findings** into agentic workflows and telemetry for reproducible ML experiments.

## Acceptance criteria

- Models demonstrate efficient handling of ill-posed queries, with high abstain rates and concise responses.
- Training recipes include length constraints and critical thinking incentives.
- Benchmarks and telemetry reflect improvements in reasoning efficiency and robustness.
- Reproducible experiments validate mitigation of MiP-Overthinking in agentic ML systems.

---

**For Candle:** This work guides the development of reasoning models and agentic workflows that avoid overthinking, support reproducible benchmarking, and improve telemetry in Rust ML libraries.
