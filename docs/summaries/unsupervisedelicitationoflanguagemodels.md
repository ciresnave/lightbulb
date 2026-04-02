# Unsupervised Elicitation of Language Models

**Original PDF:** [UnsupervisedElicitationOfLanguageModels.pdf](../../../../Desktop/books%20and%20courses/Machine%20Learning/UnsupervisedElicitationOfLanguageModels.pdf)
**Source Markdown:** [unsupervisedelicitationoflanguagemodels.md](../papers/markdown/unsupervisedelicitationoflanguagemodels.md)

---

## TL;DR

This paper introduces Internal Coherence Maximization (ICM), an unsupervised algorithm for fine-tuning language models using their own generated labels, without external supervision. ICM matches or exceeds the performance of human-supervised training on several tasks, especially where models have superhuman capabilities.

## Why it matters

ICM enables post-training of frontier language models for tasks too complex for reliable human supervision, unlocking new capabilities and improving performance in areas where human feedback is limited or unreliable.

## Key technical takeaways

- ICM fine-tunes models by maximizing mutual predictability and logical consistency of generated labels.
- Outperforms human supervision on tasks where LMs are superhuman, and matches golden label training on standard benchmarks.
- Successfully trains unsupervised reward models and RL-based assistants, outperforming human-supervised counterparts.
- Demonstrates practical utility for post-training frontier models into general assistants.

## Implementation steps (for Candle)

1. Implement ICM algorithm for unsupervised label generation and fine-tuning.
2. Apply ICM to train reward models and RL-based assistants.
3. Benchmark performance against human-supervised and golden label baselines.
4. Extend to tasks with superhuman LM capabilities and production-scale settings.

## Acceptance criteria

- Candle implementation of ICM matches or exceeds human-supervised training on selected tasks.
- Demonstrated improvements in unsupervised reward modeling and RL-based assistant training.
- Summary links to both the original PDF and markdown source.
