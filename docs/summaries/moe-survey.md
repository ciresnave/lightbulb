# A Comprehensive Survey of Mixture-of-Experts (MoE) — summary

TL;DR

- MoE sparsifies compute by routing tokens to a small subset of experts (subnetworks), enabling parameter scaling with near-constant FLOPs per token. Practical wins come from robust routing (Top-1/Top-2/Switch), capacity management, and communication-efficient execution (expert parallelism + all-to-all optimizations) that tame stragglers at inference.

Why it matters for lightbulb

- Lightbulb’s scheduler needs to batch tokens per expert, respect capacity, and avoid tail latencies. With capacity-aware routing and expert-parallel batching, we can unlock high throughput and stable p95 latency on Mixtral-like models while keeping memory balanced.

Key concepts and findings

- Routing functions
  - Top-K (K∈{1,2}) with softmax gate; Switch/Top-1 minimizes duplication; Top-2 improves quality at added comms.
  - Load balancing: auxiliary losses (e.g., importance/probability balancing), gating noise, router z-loss for stability.
  - Alternatives: hashed/learned routing, shared/base experts, and hybrid experts for generalization.
- Capacity and token drop
  - Capacity factor defines max tokens per expert per step; overflow policies: drop (Switch), reroute, or expand capacity locally.
  - Dropless or near-dropless methods reduce degradation at high loads but require careful scheduling.
- Parallelism and communication
  - Expert Parallelism (EP) distributes experts across devices; AllToAll shuffles token embeddings to assigned experts.
  - Combine with DP/TP/SP (sequence parallel) for scale; micro-batching and token grouping reduce bubbles.
  - Inference: stragglers from skewed routing dominate p95; capacity-aware batching and local expansion mitigate tails.
- Inference scheduling patterns
  - Group tokens by target expert per layer; pre-allocate per-expert queues with capacity; process experts in waves.
  - Capacity-aware inference (token drop + expanded capacity m) reduces stragglers and improves p95 without hurting QoS.
  - Caching expert states and overlapping comm/compute further reduce stalls.
- Quality vs efficiency trade-offs
  - Top-1 is fastest but can degrade; Top-2 improves quality but doubles comm; Switch variants simplify routing.
  - Wider experts vs more experts: memory/comms vs specialization; shared experts stabilize rare tokens.

Actionable next steps for lightbulb

- Scheduler: Add MoE-aware path that builds per-expert micro-batches each layer, honoring capacity and emitting AllToAll hooks (abstracted for Candle backends).
- Capacity-aware routing option: integrate token drop and local expansion (m) knobs; expose p95/p99 latency metrics per layer.
- Observability: track per-expert queue depths, utilization, and token drop rate; export histograms for routing skew.
- Acceptance targets
  - Functional Mixtral-like demo with stable throughput under mixed loads.
  - p95 step latency reduced ≥30% with ≤2% token drop on synthetic/gated traces (aligns with capacity-aware inference).
- References
  - Also see: docs/summaries/capacity-aware-inference-moe.md
