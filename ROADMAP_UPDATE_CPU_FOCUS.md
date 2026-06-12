# Roadmap Update: CPU-Focused Additions

## Overview

After reviewing the numerically-named paper summaries from the literature index and the synergy pairs document, I've integrated **20+ additional CPU-focused, systems-level optimizations** into the roadmap. These complement the previous research-driven additions and align perfectly with Lightbulb's CPU-first philosophy.

## Key Synergies Identified

Based on `synergy_pairs.md`, several features should be developed together or in sequence:

### High-Priority Synergy Clusters

1. **MoE Routing Ecosystem** (M4)
   - Survey (general strategies) → 2506-10943v2 (overhead benchmarking) → 2506-16500v1 (sparse routing)
   - Should be implemented as a cohesive system

2. **Quantization + Sparsity** (M3)
   - Low-bit survey → 2508-15884v1 (empirical quant+sparsity) → 2508-13678v1 (profiling)
   - Mutually reinforcing; implement together

3. **Scheduler + Early Exit** (M4)
   - 2506-04761v2 (adaptive depth) ↔ 2508-15126v1 (preemption + early exit)
   - Synergistic policies for elastic batching

4. **Verifier Pipelines** (M4)
   - 2506-15882v1 (primitives) ↔ 2508-15260v1 (hybrid pipelines)
   - Share code and benchmarks

## New Additions by Milestone

### M3 — Acceleration Features (CPU Kernel Focus)

**CPU Kernel Optimizations** (4 techniques):
- **Kernel Fusion** (2507-00951v1)
  - Fuse bias+gelu, matmul+add operations
  - Target: ≥10% throughput improvement
  - Low-risk, high-reward for CPU

- **Cache-Friendly Blocking** (2506-21103v1)
  - Blocking/tiling for attention and GEMM
  - Target: ≥20% L1/L2 cache miss reduction
  - Critical for small-batch inference

- **Micro-Prefetch** (2508-19828v1)
  - Software prefetch hints at tile boundaries
  - Adaptive prefetch distance based on L1/L2 miss rates
  - Target: Improved tail latency (95/99) on small batches

- **int8 GEMM** (2509-07017v1)
  - Quantization-aware accumulation
  - Cache-aligned layouts
  - Target: Positive throughput gains on quantized models

**Blocked Sparsity + Quantization Integration** (2 papers):
- **Interaction Analysis** (2508-13678v1, 2508-15884v1)
  - Per-block calibration
  - Mixed-precision accumulation
  - Empirical tuning loop
  - Target: 30% throughput improvement, ≤1% accuracy loss
  - *Synergy: Mutually reinforcing implementation*

**Per-Layer Sparsity Masks** (2506-22443v1):
- Compact formats (bit-packed, RLE)
- Tile-aligned, branch-free kernels
- Runtime selection (dense/masked)

### M4 — Advanced Scheduling (Systems Focus)

**Enhanced Memory-Aware Scheduler** (5 new features):
- **Hybrid LRU-LFU Eviction** (2509-03646v2)
  - Weights recency and frequency
  - Robust across varied token reuse patterns
  - Reduces recompute while bounding memory

- **Per-Core Partitioning** (2510-05949v1)
  - Work distribution across cores
  - Coordinated prefetch strategies
  - Reduces contention and tail latency

- **Cooperative Yield API** (2508-15126v1)
  - For long-running tasks
  - Remaining-work estimator
  - Lightweight checkpoints vs heavy context switches
  - Target: 15-30% reduction in 95th-percentile latency
  - *Synergy: Pairs with adaptive depth (2506-04761v2)*

- **Fairness Heuristics** (2509-14234v1)
  - Multi-tenant priority classes
  - Adaptive preemption
  - Prevents low-priority starvation

- **Profiling Instrumentation** (2508-13678v1 Part 2)
  - Per-layer memory peak estimation
  - Lightweight hooks for live decisions
  - Supports KvPager and Scheduler

**Production MoE Routing** (3 papers):
- **Routing Overhead Benchmarks** (2506-10943v2)
  - Per-token latency telemetry
  - Memory-touch cost measurement
  - Load-balancer fallback
  - Target: 30% routing latency reduction
  - *Synergy: Informs MoE survey implementation*

- **Sparse Routing Analysis** (2506-16500v1)
  - Capacity caps prevent hotspots
  - Lightweight token bucketing by score
  - Conservative budgeted reassignments
  - Target: 20% tail latency improvement
  - *Synergy: Complements MoE survey strategies*

**Verification-First Sampling** (2 papers):
- **Verifier Primitives** (2506-15882v1)
  - Symbolic-numeric checks
  - Batched verification
  - Bounded latency (<5ms)
  - *Synergy: Foundation for hybrid pipeline*

- **Hybrid Verifier Pipelines** (2508-15260v1)
  - Two-stage: symbolic → numeric
  - Early rejection hooks
  - Target: >50% error reduction, ≥10% compute reduction
  - *Synergy: Builds on primitives from 2506-15882v1*

**Adaptive Layer Selection** (2506-04761v2):
- Shallow-then-deep heuristic
- Lightweight confidence estimators
- Target: ≥20% FLOPs/token reduction, <1% error increase
- *Synergy: Integrates with scheduler preemption (2508-15126v1)*

### M5 — Frontier Options (Advanced CPU Features)

**Adaptive Mixed-Precision Profiling** (2 papers):
- **Per-Layer Profiling** (2510-06557v1)
  - Microprofiling at startup or dynamically
  - Conservative defaults minimize overhead
  - Better throughput/accuracy curves vs uniform precision

- **Per-Core Profiling** (2510-04871v1)
  - int4/int8 kernel selection per physical core
  - Heterogeneous CPU support
  - Improved per-core throughput

**Low-Rank Attention** (2508-19828v1 Part 2):
- Tunable rank parameter
- Reduces attention complexity for long contexts
- Target: Throughput gains with <1.5% perplexity degradation

**Lightweight Model Sharding** (2509-13341v1):
- For CPU clusters
- Async partitioning and pipelining
- Hides communication latencies
- Target: Demonstrable multi-node scaling

## Quantified Impact Summary

### M3 Additions
- **10%+** throughput from kernel fusion
- **20%+** cache miss reduction from blocking
- **30%** throughput from quant+sparsity (≤1% accuracy loss)
- Improved tail latency from micro-prefetch

### M4 Additions
- **15-30%** reduction in 95th-percentile latency (scheduler)
- **30%** routing latency reduction (MoE)
- **20%** tail latency improvement (sparse routing)
- **>50%** error reduction from verifiers
- **≥10%** compute reduction (verification-first)
- **≥20%** FLOPs/token reduction (adaptive layers)

### M5 Additions
- Better throughput/accuracy curves (adaptive precision)
- **<1.5%** perplexity degradation (low-rank attention)
- Multi-node scaling (CPU sharding)

## Implementation Priorities Based on Synergies

### Immediate (M3 - Parallel Development)
1. **Kernel Optimizations** - Independent, high ROI
   - Fusion, blocking, prefetch, int8
   - Can all be developed in parallel

2. **Quant + Sparsity Cluster** - Develop together
   - 2508-13678v1 + 2508-15884v1 as a unit
   - Mutually reinforcing

### Next Phase (M4 - Coordinated)
1. **Scheduler + Early Exit Cluster**
   - 2508-15126v1 (preemption) + 2506-04761v2 (adaptive depth)
   - Implement scheduler features first, then integrate adaptive depth

2. **MoE Routing Ecosystem**
   - Survey foundations → overhead benchmarks → sparse routing
   - Sequential implementation recommended

3. **Verifier Pipeline**
   - 2506-15882v1 (primitives) → 2508-15260v1 (hybrid)
   - Clear dependency order

### Later (M5 - Optimization)
1. Adaptive precision profiling
2. Low-rank attention experiments
3. CPU cluster sharding

## Updated Paper Count

**Previous integration**: ~30 papers  
**New CPU-focused additions**: ~20 papers  
**Total integrated**: **50+ papers** from 100+ summary collection

## Key References Added

### M3
- docs/summaries/2507-00951v1.md (kernel fusion)
- docs/summaries/2506-21103v1.md (cache blocking)
- docs/summaries/2508-19828v1.md (micro-prefetch)
- docs/summaries/2509-07017v1.md (int8 GEMM)
- docs/summaries/2508-13678v1.md (quant+sparsity)
- docs/summaries/2508-15884v1.md (quant+sparsity empirical)
- docs/summaries/2506-22443v1.md (sparsity masks)

### M4
- docs/summaries/2509-03646v2.md (LRU-LFU eviction)
- docs/summaries/2510-05949v1.md (per-core partitioning)
- docs/summaries/2508-15126v1.md (scheduler preemption)
- docs/summaries/2509-14234v1.md (fairness heuristics)
- docs/summaries/2506-10943v2.md (MoE routing overhead)
- docs/summaries/2506-16500v1.md (sparse routing)
- docs/summaries/2506-15882v1.md (verifier primitives)
- docs/summaries/2508-15260v1.md (hybrid verifiers)
- docs/summaries/2506-04761v2.md (adaptive layers)

### M5
- docs/summaries/2510-06557v1.md (per-layer precision)
- docs/summaries/2510-04871v1.md (per-core profiling)
- docs/summaries/2509-13341v1.md (CPU sharding)

## Next Steps

1. **Review synergy clusters** - Ensure implementation order respects dependencies
2. **Prototype kernel optimizations** (M3) - High ROI, can start immediately
3. **Design Policy trait** (M4) - Foundation for dynamic features
4. **Set up benchmarking harness** - Critical for validating all optimizations
5. **Create GitHub issues** - Break down features by synergy cluster

## Questions for Consideration

1. **M3 CPU kernels**: Should these be Candle PRs upstream or Lightbulb-specific?
2. **Quant+Sparsity**: Develop as single integrated feature or separate with shared test suite?
3. **Scheduler complexity**: Implement incrementally (yield → preemption → fairness) or as cohesive system?
4. **Verifier library**: Standalone crate or embedded in Lightbulb?
5. **MoE priority**: Early or wait for production workloads to justify complexity?

---

**Status**: Roadmap now comprehensively covers CPU-first optimizations with clear synergies and implementation order guidance.
