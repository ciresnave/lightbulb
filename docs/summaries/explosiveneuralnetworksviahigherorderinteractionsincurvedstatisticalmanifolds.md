# ExplosiveNeuralNetworksViaHigherOrderInteractionsInCurvedStatisticalManifolds

Full PDF: [Local PDF](file:///c%3A/Users/cires/OneDrive/Desktop/books%20and%20courses/Machine%20Learning/ExplosiveNeuralNetworksViaHigherOrderInteractionsInCurvedStatisticalManifolds.pdf)

Markdown: ../papers/markdown/explosiveneuralnetworksviahigherorderinteractionsincurvedstatisticalmanifolds.md

## TL;DR

Introduces "curved neural networks", compact parameterizations that capture higher-order interactions (HOIs) and produce analytically tractable phenomena (explosive phase transitions, multistability); shows HOIs can boost associative-memory capacity and robustness.

## Why it matters

- Provides analytic models and intuition for higher-order effects that may explain behaviors of modern networks; useful for Lightbulb research on efficient memory/attention mechanisms and activation design.

## Key technical takeaways

1. Curved neural networks implement HOIs with few parameters and are amenable to mean-field and replica analysis.
2. HOIs produce self-regulating annealing-like dynamics that can accelerate memory retrieval and cause explosive order-disorder transitions.
3. Analytical results suggest HOIs can improve memory capacity compared to classical associative-memory models.

## Implementation steps for Lightbulb

- Create a small NumPy-based experiment reproducing the paper's toy memory retrieval task to validate capacity/robustness gains.
- Explore simple transformer-block modifications (activation shapes or attention mixers) that approximate low-order HOIs and evaluate on synthetic retrieval tasks.
- Document the experiments and, if promising, add an experimental branch demonstrating a small HOI-inspired block.

## Acceptance criteria

- Repro script for a toy retrieval experiment that shows improved capacity or robustness with HOI-inspired dynamics.
- An `experiments/hoi` folder with code, README, and a short report committed to the repo.
