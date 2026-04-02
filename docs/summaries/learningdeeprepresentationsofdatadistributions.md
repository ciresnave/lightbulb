# LearningDeepRepresentationsOfDataDistributions

Full PDF: [Local PDF](file:///c%3A/Users/cires/OneDrive/Desktop/books%20and%20courses/Machine%20Learning/LearningDeepRepresentationsOfDataDistributions.pdf)

Markdown: ../papers/markdown/learningdeeprepresentationsofdatadistributions.md

## TL;DR

Survey of representation-learning techniques (PCA/ICA, contrastive learning, VAEs, normalizing flows, diffusion/denoising) with principles connecting compression, coding rate, and downstream utility.

## Why it matters

- Provides principled guidance for selecting embedding and generative techniques for retrieval, compression, and out-of-distribution detection inside Lightbulb.

## Key technical takeaways

1. Contrastive/self-supervised methods (InfoNCE/SimCLR-style) produce robust embeddings for retrieval and downstream tasks.
2. Generative/density models (VAEs, normalizing flows) provide explicit likelihood estimates useful for OOD detection and calibration.
3. Compression-centric views (rate-distortion, coding rate reduction) unify many representation choices and suggest proxy losses for representation quality.
4. Design/optimization notes: architecture, objective weighting, and training strategy significantly affect representation quality; careful evaluation with downstream tasks is necessary.

## Implementation steps for Lightbulb

- Benchmark two embedding methods (contrastive vs existing) on our retrieval datasets; record recall@k, latency, and index size.
- Prototype an OOD detector using a small VAE or normalizing flow trained on in-distribution data; evaluate AUROC on held-out OOD examples.

## Acceptance criteria

- A benchmark compares at least two embedding methods on retrieval metrics and documents results in `evaluation/embeddings/` artifacts.
- A prototype OOD detector demonstrates improved AUROC over the current baseline on one held-out OOD dataset.
