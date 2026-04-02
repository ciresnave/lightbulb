# Video Prediction by Efficient Transformers

**Original PDF:** [VideoPredictionByEfficientTransformers.pdf](../../../../Desktop/books%20and%20courses/Machine%20Learning/VideoPredictionByEfficientTransformers.pdf)
**Source Markdown:** [videopredictionbyefficienttransformers.md](../papers/markdown/videopredictionbyefficienttransformers.md)

---

## TL;DR

This paper introduces a family of efficient Transformer-based models for video prediction, featuring a novel local spatial-temporal separation attention mechanism. Three variants—full autoregressive, partial autoregressive, and non-autoregressive—are compared, with the non-autoregressive model offering faster inference and improved quality.

## Why it matters

Efficient video prediction is crucial for applications like autonomous vehicles, anomaly detection, and reinforcement learning. Transformer-based models overcome limitations of ConvLSTMs, enabling faster, more accurate, and scalable video representation learning.

## Key technical takeaways

- Local spatial-temporal separation attention reduces Transformer complexity for video tasks.
- Non-autoregressive models mitigate quality degradation and speed up inference, though they require extra parameters and loss functions.
- All proposed models are competitive with state-of-the-art ConvLSTM-based approaches.
- Source code is available for reproducibility and further research.

## Implementation steps (for Candle)

1. Implement local spatial-temporal separation attention in Candle's Transformer module.
2. Develop full, partial, and non-autoregressive video prediction variants.
3. Benchmark models on standard video prediction datasets.
4. Compare inference speed and prediction quality across variants.

## Acceptance criteria

- Candle implementation matches or exceeds ConvLSTM baselines in speed and quality.
- All three model variants are available and benchmarked.
- Summary links to both the original PDF and markdown source.
