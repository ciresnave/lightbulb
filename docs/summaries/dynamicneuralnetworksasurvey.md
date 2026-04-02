Title: Dynamic Neural Networks: A Survey
Authors: Yizeng Han, Gao Huang, Shiji Song, Le Yang, Honghui Wang, Yulin Wang

TL;DR
Dynamic Neural Networks adapt their structure or computation per input (sample-wise, spatial-wise, temporal-wise). They trade static uniform computation for adaptive, efficient, and often more accurate processing by selectively executing or configuring subnetworks.

Why it matters
Dynamic networks enable models to be both more compute-efficient and more flexible. For production systems where latency and cost matter, dynamic behavior (early exiting, gating, conditional computation) can reduce average compute while retaining high accuracy. They also offer adaptivity to input complexity and can improve robustness.

Key takeaways

- Taxonomy: sample-wise (example-dependent depth/width), spatial-wise (conditional computation over spatial locations), temporal-wise (varying across time steps).
- Mechanisms: early-exit branches, layer skipping, gating networks, mixture-of-experts and routing, dynamic parameters (input-dependent weights).
- Training and optimization: specialized objectives, latency-aware losses, knowledge distillation for early-exit branches, and training schemes to stabilize routing/gating.
- Challenges: training stability, hardware-friendly implementations, deployment complexity, measuring and controlling latency vs accuracy tradeoffs, and benchmarking across tasks.

Implementation steps

1. Identify the dynamic dimension suitable for the task (sample, spatial, temporal).  
2. Select a mechanism: early-exit for sample-wise fast-exit, gating or layer skipping for conditional depth, MoE or routing for selective expert invocation.  
3. Design an efficient controller (lightweight gating head or confidence estimator) and incorporate latency/compute penalties into the loss.  
4. Use distillation and auxiliary classifiers for early-exit branches; regularize gating to avoid degenerate collapse.  
5. Benchmark across latency/accuracy Pareto curves, profile on target hardware, and iterate on controller complexity.

Acceptance criteria

- The dynamic variant should match or exceed baseline accuracy while reducing average FLOPs or latency by a measurable margin (e.g., 20% lower average latency with <1% accuracy drop).  
- Controller decisions must be stable (no frequent oscillation) and reproducible.  
- The implementation is deployable on target hardware with documented runtime behavior and a latency/accuracy tradeoff curve.

# Dynamic Neural Networks — A Survey

TL;DR:

Why it matters:

Key takeaways:

Implementation steps:

Acceptance criteria:
