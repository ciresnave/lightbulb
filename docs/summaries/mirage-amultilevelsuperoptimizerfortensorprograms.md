# Mirage: A Multi-Level Superoptimizer for Tensor Programs

**Full PDF:** [View Original](C:\Users\cires\OneDrive\Desktop\books and courses\Machine Learning\Mirage-AMultiLevelSuperoptimizerForTensorPrograms.pdf)  
**Markdown:** [View Markdown](../papers/markdown/mirage-amultilevelsuperoptimizerfortensorprograms.md)

## TL;DR

Mirage is the first multi-level superoptimizer for tensor programs, introducing a unified graph representation across GPU compute hierarchies and novel optimization techniques that outperform existing frameworks on deep neural network workloads.

## Why it matters

- Enables automated, high-performance optimization of tensor programs for ML workloads, reducing manual engineering
- Discovers novel optimizations by combining algebraic, schedule, and custom kernel transformations
- Provides strong theoretical guarantees for optimality and equivalence verification
- Outperforms state-of-the-art frameworks (TVM, Halide, etc.) on widely used and heavily optimized DNNs
- Aligns with lightbulb's goals of efficient, scalable, and reproducible ML model deployment

## Key technical takeaways

1. Introduces "Graphs" as a uniform representation for tensor programs at kernel, thread block, and thread levels
2. Uses abstraction-based pruning to reduce search space and provide optimality guarantees
3. Employs probabilistic equivalence verification for correctness with strong theoretical backing
4. Automates discovery of optimizations that span multiple GPU hierarchy levels, including custom kernel generation
5. Publicly available codebase for reproducibility and benchmarking

## Implementation steps for lightbulb

- Integrate Mirage-inspired multi-level optimization modules for tensor programs in Candle
- Prototype unified graph representations for kernel/thread scheduling and algebraic transformations
- Benchmark performance against existing frameworks on representative DNN workloads
- Add telemetry for optimization efficiency, kernel generation, and equivalence verification metrics
- Link to Mirage codebase for reproducibility and community contributions

## Acceptance criteria

- Implement multi-level tensor program optimizer with test coverage for kernel, thread block, and thread transformations
- Demonstrate performance improvements over baseline frameworks on at least one DNN benchmark
- Telemetry dashboard reports optimization, kernel generation, and verification metrics
- Integration tests confirm correctness and reproducibility of Mirage-inspired optimizations in lightbulb
