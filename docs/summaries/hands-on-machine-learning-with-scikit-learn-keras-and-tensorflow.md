# Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (O'Reilly)

Full PDF: [Local PDF](file:///c%3A/Users/cires/OneDrive/Desktop/books%20and%20courses/Machine%20Learning/Hands-On-Machine-Learning-With-Scikit-Learn-Keras-And-TensorFlow.pdf)

Markdown: ../papers/markdown/hands-on-machine-learning-with-scikit-learn-keras-and-tensorflow.md

## TL;DR

Comprehensive practical guide covering ML fundamentals, model building, deep learning with TensorFlow/Keras, and production considerations — a useful engineering reference for standard best practices.

## Why it matters

- Good source for onboarding and for codifying engineering checklists (data prep, validation, experiment hygiene) that Lightbulb should follow for reproducible experiments.

## Key technical takeaways

1. Practical recipes for data pipelines, model validation, hyperparameter tuning, and deployment best practices.
2. Clear examples bridging ML theory to code (Scikit-Learn + Keras) that are useful for quick prototyping and testing ideas before scaling to LLM experiments.

## Implementation steps for Lightbulb

- Extract a mini-checklist for experiments (data split, seed handling, metric definitions) and add it to `docs/engineering/experiment-checklist.md`.
- Use the book's recommended validation recipes to standardize our small-model benchmarks and CI regression tests.

## Acceptance criteria

- An `experiment-checklist.md` file exists and is referenced by at least one current experiment; CI uses the checklist for at least one regression test.
