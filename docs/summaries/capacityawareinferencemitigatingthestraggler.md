# Capacity-Aware Inference: Mitigating The Straggler (stub)

TL;DR

- Survey/notes on strategies to reduce straggling requests in model serving via capacity-aware scheduling and load balancing.

Why it matters

- Informs scheduler heuristics and capacity-aware request routing in the `lightbulb` runtime.

Key takeaways

- Practical heuristics for batching, preemption, and prioritization to reduce tail latency.

Implementation steps

1. Add scheduling heuristics to the scheduler design notes and candidate experiments.

Acceptance criteria

- A short set of scheduler test cases that reproduce straggler reductions in microbenchmarks.
