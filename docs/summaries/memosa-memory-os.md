# MemOSA: A Memory OS for AI Systems — summary

TL;DR

- Proposes system-level memory orchestration for AI workloads: dynamic placement, tiering (HBM/DRAM/NVMe), and scheduling to maximize throughput and meet SLAs.

Why it matters for lightbulb

- Informs KV pager/offload design and long-context scheduling; motivates metrics and policies for tiered KV (RAM/disk) in CPU-first setups.

Key points

- Memory tiering, eviction/placement policies, streaming and prefetch, admission control.
- SLA-driven scheduling and observability.

Actionable next steps

- Add metrics for KV residency (hot/handoff/offloaded) and page fault counters; simulate RAM↔disk paging.
- Acceptance: show stable throughput with bounded tail latency under long-context workloads using tiered KV in a CPU-only environment.
