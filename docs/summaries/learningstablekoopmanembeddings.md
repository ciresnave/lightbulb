# Learning Stable Koopman Embeddings

**Full PDF:** [View Original](C:\Users\cires\OneDrive\Desktop\books and courses\Machine Learning\LearningStableKoopmanEmbeddings.pdf)  
**Markdown:** [View Markdown](../papers/markdown/learningstablekoopmanembeddings.md)

## TL;DR

This paper presents a data-driven method for learning stable models of nonlinear systems by lifting the state space to a higher-dimensional linear manifold using Koopman embeddings. The approach enables unconstrained optimization of both the embedding and Koopman operator while enforcing model stability, and is validated on simulated systems.

## Why it matters

- Stability is critical for predictive models in control, robotics, and ML systems—unstable models can yield unbounded, unusable predictions
- Koopman-based embeddings allow nonlinear dynamics to be analyzed and controlled using linear system theory, relevant for efficient model-based ML
- The method bridges neural network expressiveness with strong stability guarantees, aligning with lightbulb's focus on robust, efficient architectures
- Enables unconstrained optimization, simplifying implementation and integration into ML pipelines
- Supports learning stable models from data, which is essential for safe deployment in real-world applications

## Key technical takeaways

1. Proposes a framework that jointly learns the Koopman embedding and operator from data, with explicit stability (contraction) constraints
2. Proves that any discrete-time contracting nonlinear model can be learned in this framework (extension of prior continuous-time results)
3. Uses direct parameterization of stable linear systems, allowing unconstrained optimization and simplifying computation
4. Demonstrates improved robustness and stability over alternative parameterizations in experiments
5. Connects Koopman operator theory with practical neural network-based system identification

## Implementation steps for lightbulb

- Prototype Koopman embedding modules for system identification tasks in Candle-based ML pipelines
- Integrate stability-constrained optimization routines for learning dynamical models from data
- Benchmark learned models for stability, prediction accuracy, and computational efficiency
- Add telemetry for model stability metrics and prediction error over time
- Explore applications to control, simulation, or robust inference in ML systems

## Acceptance criteria

- Demonstrate learning of stable nonlinear models from data with <5% prediction error on test systems
- Show that learned models remain stable (bounded predictions) under long-term simulation
- Achieve measurable improvements in robustness and accuracy over unconstrained baselines
- Integration tests confirm stability and performance in at least one real-world or simulated ML/control task
