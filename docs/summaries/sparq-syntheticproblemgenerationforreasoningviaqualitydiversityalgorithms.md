# SPARQ: Synthetic Problem Generation for Reasoning via Quality-Diversity Algorithms

**Links:**  
[Original PDF](c:/Users/cires/OneDrive/Desktop/books%20and%20courses/Machine%20Learning/SPARQ_SyntheticProblemGenerationForReasoningViaQualityDiversityAlgorithms.pdf)  
[Source Markdown](../papers/markdown/sparq-syntheticproblemgenerationforreasoningviaqualitydiversityalgorithms.md)

---

## TL;DR

SPARQ is a synthetic data generation algorithm that produces high-quality and diverse math problem-solution pairs using a single LLM and quality-diversity algorithms. By measuring solve-rate as a proxy for problem difficulty, SPARQ filters and generates data that improves model reasoning and generalization, achieving up to 24% performance gains.

## Why it matters

High-quality, diverse synthetic data is crucial for training robust reasoning models. SPARQ enables scalable generation of challenging and varied problems without relying on ground-truth or large oracles, advancing the capabilities of student models and supporting better in-distribution and out-of-distribution generalization.

## Key technical takeaways

- **Solve-Rate Filtering:**
  - Problem difficulty is estimated via solve-rate, allowing for effective filtering of generated data.
- **Quality-Diversity Algorithms:**
  - Synthetic problems are generated and mutated to maximize both quality and diversity.
- **Scaling Laws:**
  - Model and data scaling laws are confirmed for synthetically generated problems, benefiting downstream generalization.
- **Performance Gains:**
  - Fine-tuning on filtered synthetic data leads to significant improvements in pass@1 accuracy and generalization.
- **Ablation Studies:**
  - Quality improves in-distribution performance; diversity benefits out-of-distribution generalization.

## Implementation steps (Candle/Rust context)

1. **Synthetic Data Generation:**
   - Implement algorithms to generate and mutate problem-solution pairs using a single LLM.
2. **Solve-Rate Estimation:**
   - Measure solve-rate for each problem to filter by difficulty and quality.
3. **Quality-Diversity Optimization:**
   - Apply mutation and selection strategies to maximize data quality and diversity.
4. **Fine-Tuning:**
   - Train models on filtered synthetic data and benchmark performance.
5. **Evaluation:**
   - Analyze scaling laws, in-distribution, and out-of-distribution generalization.

## Acceptance criteria

- Implementation generates and filters synthetic problems using solve-rate and quality-diversity algorithms.
- Fine-tuning on generated data improves model reasoning and generalization.
- Evaluation demonstrates scaling laws and performance gains.
- Code is modular, reproducible, and links to both the original PDF and markdown source.
