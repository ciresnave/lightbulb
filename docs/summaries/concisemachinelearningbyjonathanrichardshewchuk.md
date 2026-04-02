# Concise Machine Learning

**Full PDF:** [View Original](<file:///c:/Users/cires/OneDrive/Desktop/books%20and%20courses/Machine%20Learning/ConciseMachineLearningByJonathanRichardShewchuk.pdf>)

**Markdown:** [View Markdown](../papers/markdown/concisemachinelearningbyjonathanrichardshewchuk.md)

## TL;DR

Lecture‑style concise notes that cover core supervised and unsupervised ML algorithms (linear models, SVMs, decision trees, ensembles), practical neural network training (optimizers, normalization, architectures), and essential numerical/linear‑algebra tools (SVD/PCA). Very implementation-oriented: emphasizes algorithmic tradeoffs, numerical stability, and simple recipes for robust training.

## Why it matters

- Comprehensive, practical reference of classical and modern ML algorithms we will expose in Lightbulb's API surface.

- Contains concrete engineering guidance (optimizers, numerical stability, regularization, initialization) that map directly to low-level kernels and trainer modules in Candle.

- Covers linear algebra building blocks (SVD, PCA, least squares) we should optimize for CPU inference/training workloads.

- Good source of test cases and validation probes (GLM, small CNNs, SVMs) for benchmarking kernels and end-to-end training loops.

## Key technical takeaways

1. Classic linear methods (least squares, logistic regression) remain useful baselines; numerically stable solvers and regularization (ridge, Lasso) are essential for robust behavior.

2. Support Vector Machines and margin methods demonstrate model/feature trade-offs that are useful for small-data regimes; kernel methods rely on efficient nearest-neighbor/approximate-NN implementations.

3. Decision trees and ensembles are strong, interpretable baselines; they benefit from careful feature engineering and efficient data structures for splits and histograms.

4. Neural network training: practical recipes—ReLU activations, proper initialization, batch normalization, Adam/AdamW, careful learning-rate schedules, and gradient clipping—have large effect on convergence and stability.

5. Convolutional networks and residual connections are covered with concrete architectural patterns; batching, data pipelines, and augmentation are noted as important system-level considerations.

6. Linear algebra tools: SVD/PCA, randomized projections, eigen-decompositions, and fast GEMM are repeatedly used; attention to numerical precision and stable algorithms is required.

## Implementation steps for lightbulb

- Implement/verify numerically-stable linear solvers (least-squares with QR/SVD fallback) in the linear algebra module.

- Add optimizer implementations (SGD, Adam, AdamW, RMSProp) with configurable hyperparams and support for mixed-precision paths.

- Provide kernel primitives: fused conv + bias + activation, batch-normalization, residual block builder, and efficient GEMM with tiling for long sequences and small batch sizes.

- Implement data utilities: streaming data loader, common augmentations, and histogram-based split helpers for tree algorithms.

- Add tests and reference training recipes that reproduce small-scale results (e.g., CIFAR-like tiny CNN, logistic regression convergence, PCA reconstruction errors).

## Acceptance criteria

- Reference optimizers (SGD, AdamW) implemented and validated against simple convergence tests (logistic regression, shallow NN) with matching loss curves to a known baseline.

- Numerically-stable least-squares that handle ill-conditioned matrices (compare QR vs normal equations) with test coverage.

- Fused convolution + activation primitive demonstrates measurable latency improvement vs unfused ops on representative inputs.

- Add a small training example (tiny CNN) in the examples/test suite that trains end-to-end within expected loss/accuracy ranges.
