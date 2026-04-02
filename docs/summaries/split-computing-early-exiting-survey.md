# Split Computing and Early Exiting — survey summary

TL;DR

- Split computing partitions models across edge/cloud and combines with early exit to minimize latency and bandwidth. Decisions may be dynamic per-input based on confidence and network conditions, offering robust QoS under varying constraints.

Why it matters for lightbulb

- Our scheduler can support split points (layer cut) and early exits concurrently: run initial layers locally, exit early if confident, or offload remainder. Even if we stay CPU-only, this informs architecture and metrics for future distributed modes.

Key points

- Partition strategies: single split vs multi-split; profiling-based placement; adaptive switching.
- Early exit criteria: entropy/max-prob, patience; domain adaptation for edge data.
- Communication: compress activations, selective feature transmission, quantization.
- QoS: budgeted latency, bandwidth caps, and anytime prediction.

Actionable next steps

- Introduce a conceptual “split point” in configs and record per-layer activations size to estimate network cost (doc + metrics now, implement later).
- Acceptance: export per-request traces including hypothetical offload bytes and exit layer; demonstrate ≥20% estimated latency/bandwidth savings in a what-if analysis.
