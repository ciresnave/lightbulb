# Path to M5 Completion: What Needs to Be Done

> ## ⚠️ SUPERSEDED — this is not a status source
> 
> **`ROADMAP.md` is the single roadmap.** Its VERIFIED STATUS block carries
> figures measured at a named ref. This document's status claims were made at
> a state the project has left and have **not** been re-validated — treat them
> as provenance, not as findings.
> 
> Dated January 2026. Its status predates the GGUF tokenizer work, the Fuel
> pin bump, and the CI gates.

**Date**: January 2026  
**Current Status**: M0-M2 COMPLETE, M3 PARTIAL, M3.4-M5 PLANNED  
**Goal**: Complete M0-M5 production core for v1.0 release

---

## Executive Summary

Reviewing the ROADMAP.md reveals we've completed significant infrastructure but have **added many integration points under already-completed milestones**. To reach M5 and v1.0, we need to:

1. **Complete remaining M3 items** (speculative decoding, AWQ/SmoothQuant)
2. **Execute M3.7 integrations** across M4-M6 (name mapping in model loading, cache, LoRA, etc.)
3. **Implement M4 scheduling features**
4. **Build M5 frontier optimizations**
5. **Polish M5.5 for v1.0 release** (CLI, deployment, benchmarks)

**The Challenge**: We've added significant work to "completed" milestones (M3.5, M3.6, M3.7) that actually needs to be executed in M4-M6.

---

## Current State Analysis

### ✅ FULLY COMPLETE (No Further Work)

**M0 — Baseline (0.1)**
- ✅ CPU-only local LLaMA loader
- ✅ Conditional integration tests
- ✅ Docs and helper scripts

**M1 — Core Engine (0.2)**
- ✅ Continuous batching MVP
- ✅ Paged KV façade hardened
- ✅ Observability (tracing, metrics)

**M1.4 — Parallel Batching**
- ✅ BatchManager<M> architecture
- ✅ BatchedTransformer implementation
- ✅ ScatteredKvCache
- ✅ Integration tests passing

**M1.5 — Hardware Adaptivity**
- ✅ system-analysis crate integrated
- ✅ Device-adaptive chunk sizing
- ✅ Hardware detection working

**M2 — Performance Enablers**
- ✅ StreamingLLM policy complete
- ✅ Prefix KV caching complete
- ✅ Intelligent cache management (Phases 1-2)
- ✅ H2O eviction policy
- ✅ Cache span tagging system
- ✅ FlashAttention integration (M3.4)
- ✅ Quantized model loaders
- ✅ Lightning GGUF (Phases 1-2)

**M3.3 — Kernel Fusion**
- ✅ Infrastructure built and tested
- ⚠️ Disabled due to Candle API limitations (future: requires upstream changes)

**M3.5 — Testing & Hardening**
- ✅ External crate evaluation complete
- ✅ Correctness validation framework designed
- ✅ Load & stress testing infrastructure implemented
- ✅ Regression detection system designed
- ✅ Documentation complete (2100+ lines)

**M3.6 — Multi-GPU Inference**
- ✅ Tensor parallelism foundations
- ✅ Pipeline parallelism foundations
- ✅ Distributed KV cache manager
- ✅ BatchedTransformer integration
- ✅ 17 tests passing (hardware validation pending)
- ✅ Documentation complete

**M3.7 — Dynamic Name Mapping (CORE)**
- ✅ Core name_mapping.rs module (428 lines)
- ✅ Integration with pruning system
- ✅ Architecture detection (LLaMA, GPT, Mistral)
- ✅ Tests passing

---

### 🚧 PARTIALLY COMPLETE (Needs Finishing)

**M3 — Acceleration Features (0.4)**

**INCOMPLETE ITEMS**:
1. **Speculative Decoding MVP** 📋 TODO
   - Draft+target dual-model orchestration
   - Verify-accept loop
   - Target: >1.3× speedup on CPU
   - Estimate: 2-3 weeks

2. **AWQ/SmoothQuant Enablement** 📋 TODO
   - Integrate if Candle exposes quant ops
   - Otherwise: contrib-friendly shim
   - Quality-per-bit improvement validation
   - Estimate: 2 weeks

3. **CPU Kernel Optimizations** ⏸️ BLOCKED
   - Cache-friendly blocking strategies
   - Micro-prefetch hints
   - int8 GEMM micro-kernels
   - **Blocker**: Requires Candle upstream changes OR custom linear layer
   - **Decision**: Defer to post-v1.0 (not critical path)

4. **Blocked Sparsity & Quantization** 📋 TODO
   - Blocked-sparsity kernels
   - Per-block calibration
   - Mixed-precision accumulation
   - Target: ≥30% throughput improvement, ≤1% accuracy loss
   - Estimate: 2-3 weeks

5. **Per-Layer Sparsity Masks** 📋 TODO
   - Compact mask file format (bit-packed, RLE)
   - Tile-aligned masked kernels
   - Runtime selection between dense/masked paths
   - Estimate: 1-2 weeks

6. **Decode-Loop Overhead Reductions** 📋 TODO
   - Batch reuse optimization
   - Minimal host<->device sync
   - CUDA Graphs interface (feature-gated)
   - Estimate: 1-2 weeks

**M3 Completion Estimate**: 8-12 weeks if all items pursued, OR 4-5 weeks for critical path only

---

### 📋 PLANNED BUT NOT STARTED

**M3.7 Integrations** (Added under completed milestone, actually M4-M6 work)

1. **M4.1: Model Loading Integration** 📋 TODO (3 weeks)
   - Add TensorNameMapper to ModelLoader
   - Eliminate hardcoded tensor names in `src/model/loader.rs`
   - Support 5+ architectures (LLaMA, GPT, Mistral, Qwen, Phi)
   - Three phases: Core → Enhanced → Advanced
   - Reference: `docs/NAME_MAPPING_MODEL_LOADING.md`

2. **M5.1: Cache Management Integration** 📋 TODO (3 weeks)
   - Architecture-aware layer detection for KV cache
   - Variable layer count support (automatic vs hardcoded 32)
   - Relationship-aware eviction with N-dimensional token graphs
   - Dynamic cache sizing based on architecture

3. **M5.4: LoRA Integration** 📋 TODO (2 weeks)
   - Automatic format detection (HuggingFace, custom, PEFT)
   - Map LoRA adapter names to base model
   - Validation and shape compatibility checking

4. **M3.6.1: Multi-GPU Enhancement** 📋 TODO (1 week)
   - Architecture-aware layer distribution across GPUs
   - Handle heterogeneous architectures (MoE, variable sizes)
   - Load-balanced GPU assignments based on tensor sizes

5. **M4.1: Quantization Integration** 📋 TODO (in parallel with model loading)
   - Layer-specific quantization levels
   - Component-aware mixed precision
   - Sensitivity-based quantization

6. **M5.6: Tool Registry Integration** 📋 TODO (2 weeks)
   - Auto-detect model capabilities from architecture
   - Register appropriate tools based on model type

7. **M6.5: LLM-Assisted Name Mapping** 📋 TODO (4 weeks, POST-v1.0)
   - Small LLM for probabilistic name matching
   - Three-tier fallback (regex → LLM → manual)
   - Confidence-based matching
   - Reference: `docs/LLM_ASSISTED_NAME_MAPPING.md`

**M3.7 Integrations Estimate**: 11 weeks (excluding M6.5 which is post-v1.0)

---

**M4 — Advanced Scheduling (0.5)** 📋 PLANNED

**ALL ITEMS TODO**:

1. **Memory-Aware Priority Scheduler** 📋 TODO (2 weeks)
   - Request classes, preemption, dynamic batch sizing
   - Tiered KV orchestration (RAM/disk paging)
   - Hybrid LRU-LFU eviction
   - Per-core partitioning
   - Runtime configuration updates

2. **Multi-Stage Inference Pipeline** 📋 TODO (3 weeks)
   - Chained stages (decompose → identify → generate → verify → synthesize)
   - Per-stage model selection with automatic routing
   - KV cache sharing across compatible stages
   - Dependency graph enforcement
   - Structured decomposition schemas

3. **Metadata-Driven Scheduling** 📋 TODO (1 week)
   - Request metadata schema (priority, tags, context hints)
   - Tag-based routing to specialized models
   - Constraint satisfaction hooks
   - Episode/trace metadata for RL

4. **State Persistence and Recovery** 📋 TODO (2 weeks)
   - Checkpoint/restore full inference state
   - Privacy-preserving state encryption
   - Graceful degradation on OOM
   - Session resumption across restarts

5. **Convergence Detection** 📋 TODO (1 week)
   - Configurable convergence policies
   - Per-request convergence state tracking
   - Early termination signals
   - Safety limits to prevent infinite loops

6. **Knowledge-Aware Decomposition** 📋 TODO (2 weeks)
   - Problem decomposition adapts based on KB state
   - Dynamic problem simplification
   - Re-decomposition triggers

7. **Explicit Unknown Identification** 📋 TODO (1 week)
   - Dedicated component extracting information gaps
   - Structured unknowns
   - Gap analysis enabling targeted retrieval

8. **Consistency Checking for KB** 📋 TODO (2 weeks)
   - NLI-based validator preventing contradictions
   - Validates new facts against existing KB
   - Detects contradictions, logical inconsistencies, temporal conflicts

9. **Explicit KB Construction** ✅ IMPLEMENTED (M4.5)
   - KV-eviction-to-KB architecture with inline summaries
   - [KB:key] placeholder system
   - <RETRIEVE:key> syntax
   - System instructions marked with high importance

10. **Basic MoE Support** 📋 TODO (1 week)
    - Efficient routing-friendly batching for Mixtral-like models

11. **Prompt Program Executor** 📋 TODO (1 week)
    - Minimal interpreter for multi-turn prompt graphs
    - Per-request state
    - Stop/resume support

12. **Multi-Agent Coordination** 📋 TODO (1 week)
    - Shared and private memory namespaces
    - Concurrency guards

13. **Dynamic Compute Allocation** 📋 TODO (2 weeks)
    - Policy trait system for adaptation signals
    - Actions: exit/skip/repeat/route

14. **Verification-First Sampling** 📋 TODO (1 week)
    - Pluggable verifier library
    - Symbolic checks (type, range, schema)
    - Numeric validation

**M4 Completion Estimate**: 20-24 weeks (6 months) - VERY LARGE MILESTONE

**RECOMMENDATION**: Split M4 into M4.1, M4.2, M4.3 sub-milestones for manageable delivery

---

**M5 — Frontier Options (0.6)** 📋 PLANNED

**ALL ITEMS TODO**:

1. **KV Cache Optimization** 📋 TODO (3 weeks)
   - H2O eviction heuristics (✅ already implemented, needs integration)
   - KIVI/KVQuant-style KV quant (2-4 bit)
   - R-KV training-free compression
   - Low-rank attention approximations
   - Relationship-aware eviction (needs name mapping integration)

2. **Pruning Utilities** 📋 TODO (2 weeks)
   - Wanda scoring for unstructured/structured pruning
   - Reverse-order tail prune (~25%)
   - Optional partial-layer fine-tuning

3. **Test-Time Depth Adaptation** 📋 TODO (1 week)
   - CoLa prototype
   - Offline harness for skip/repeat layer decisions

4. **Adaptive Mixed-Precision Profiling** 📋 TODO (1 week)
   - Per-layer microprofiling
   - Per-core int4/int8 kernel profiling
   - Fast profiling with conservative defaults

5. **Reasoning Efficiency Controls** 📋 TODO (2 weeks)
   - Budget-aware decoding knobs
   - Shorter-is-better heuristic
   - Confidence-based chain termination
   - Re-ranking API
   - Multi-level abstraction management

6. **Reasoning Path Compression** 📋 TODO (1 week)
   - Cache compressed reasoning templates
   - Selective reread/re-prefill

7. **Adaptive Chunking** 📋 TODO (1 week)
   - Configurable chunking policy
   - Chunk-level cache reuse

8. **Test-Time Adaptation** 📋 TODO (1 week)
   - Per-instance policy gradient tuner
   - Text-to-LoRA adapter registry

9. **Modular Specialization** 📋 TODO (1 week)
   - Module hot-swap framework

10. **Three-Tier Memory Management** ✅ INFRASTRUCTURE READY
    - infra-storage crate integrated
    - infra-fingerprinting crate integrated
    - Needs: implementation wiring

11. **Communication Protocol Evolution** 📋 TODO (2 weeks)
    - Dedicated layer learning compressed communication

12. **Tool Registry & Invocation** 📋 TODO (3 weeks)
    - Extensible plugin system
    - Schema definitions
    - Capability detection
    - **Depends on**: M3.7 name mapping integration (M5.6)

13. **MCP Service Discovery** 📋 TODO (1 week)
    - gRPC reflection for dynamic endpoint discovery

14. **Multi-Modal Tensor Routing** 📋 TODO (1 week)
    - Modality-aware metadata
    - Modality-specific processing paths

15. **Multi-Method Generation Routing** 📋 TODO (2 weeks)
    - Plugin architecture for generation backends

16. **Tiered Answer Generation** 📋 TODO (2 weeks)
    - Five-tier routing framework

17. **Modular NN Architecture Support** 📋 TODO (2 weeks)
    - Support for segmented models

18. **Offloading/Systems Experiments** 📋 TODO (2 weeks)
    - FlexGen-style GPU/CPU/disk scheduling
    - PowerInfer-style hot/cold neuron experiments

19. **Device-Side Kernel Sequences** 📋 TODO (1 week)
    - Execute DAGs on GPU without host round-trips

20. **Repeatable Job System** 📋 TODO (1 week)
    - Job execution modes

21. **Activation Caching** 📋 TODO (1 week)
    - Layer-level caches mapping inputs to outputs

22. **Usage-Driven Early Exit** 📋 TODO (1 week)
    - Early exit strategies from runtime patterns

23. **Evaluation Infrastructure** 📋 TODO (2 weeks)
    - SPARQ-like synthetic problem generators

24. **RL/Training Support** 📋 TODO (1 week)
    - Episode traces and rewards

25. **Context Condensation Pipeline** 📋 TODO (2 weeks)
    - Multi-phase context refinement

26. **Semantic Anchor Navigation** 📋 TODO (1 week)
    - Named anchors with semantic tags

27. **Declarative Data Window Management** 📋 TODO (1 week)
    - Token-aware scrollable views

28. **Tool Workflow Pattern Library** 📋 TODO (1 week)
    - Formalized chaining modes

29. **Dynamic Module Loading** 📋 TODO (1 week)
    - Sequential loading/unloading for edge deployment

**M5 Completion Estimate**: 40-44 weeks (~10 months) - EXTREMELY LARGE MILESTONE

**RECOMMENDATION**: Split M5 into M5.1, M5.2, M5.3, M5.4 for manageable delivery

---

**M5.5 — CLI, Deployment & Operations** 📋 PLANNED (v1.0 Release)

**ALL ITEMS TODO**:

1. **Command-Line Interface** 📋 TODO (2 weeks)
   - Server mode: `lightbulb serve`
   - Chat mode: `lightbulb chat`
   - Conversion utilities
   - Config management

2. **Container & Orchestration** 📋 TODO (1 week)
   - Docker images with multi-stage builds
   - Kubernetes manifests and Helm charts
   - Horizontal pod autoscaling

3. **Observability & Monitoring** 📋 TODO (1 week)
   - Prometheus metrics exporter
   - Grafana dashboard templates
   - OpenTelemetry integration
   - Structured logging

4. **Operational Reliability** 📋 TODO (1 week)
   - Graceful shutdown
   - Health check endpoints
   - Rate limiting
   - Admin API

5. **Configuration & Dev Experience** 📋 TODO (1 week)
   - HuggingFace model ID direct loading
   - YAML/JSON config with validation
   - Feature flags
   - Debug mode

6. **Model Conversion Utilities** 📋 TODO (1 week)
   - HF safetensors → Candle converter
   - GGUF ↔ safetensors conversion
   - Quantization tools

7. **Comprehensive Test Coverage** 📋 TODO (2 weeks)
   - 95%+ code coverage target
   - All feature combinations tested
   - Edge case validation
   - 48-hour soak test

8. **Benchmark Suite** 📋 TODO (2 weeks)
   - Performance benchmarks
   - Quality benchmarks
   - Comparison with competitors
   - Reproducible scripts

9. **Release Deliverables** 📋 TODO (1 week)
   - Pre-built binaries (Linux, macOS, Windows)
   - Container images
   - Documentation site
   - Quickstart guide

**M5.5 Completion Estimate**: 12-14 weeks (3 months)

---

## Critical Path Analysis

### Minimum Viable v1.0 (Fast Track)

To get to v1.0 **as quickly as possible**, focus on critical path only:

**Phase 1: Complete M3 Core (4-5 weeks)**
1. ✅ Skip CPU kernel optimizations (blocked on Candle)
2. ✅ Skip blocked sparsity (not critical)
3. ✅ Skip per-layer sparsity masks (not critical)
4. ✅ Implement speculative decoding (2-3 weeks)
5. ✅ Implement AWQ/SmoothQuant (2 weeks)
6. ✅ Basic decode-loop optimizations (1 week)

**Phase 2: M3.7 Model Loading Integration (3 weeks)**
- This unlocks architecture-agnostic loading (critical for v1.0)

**Phase 3: Minimal M4 (8-10 weeks)**
- Focus on core scheduling only:
  1. Memory-aware priority scheduler (2 weeks)
  2. Basic state persistence (2 weeks)
  3. Basic MoE support (1 week)
  4. Dynamic compute allocation (2 weeks)
  5. Verification-first sampling (1 week)
- Defer: Multi-stage pipelines, KB reasoning, convergence detection

**Phase 4: Minimal M5 (8-10 weeks)**
- Focus on performance critical items:
  1. KV cache optimization (H2O already done, add R-KV) (2 weeks)
  2. Pruning utilities (2 weeks)
  3. Reasoning efficiency controls (2 weeks)
  4. Tool registry (basic) (2 weeks)
- Defer: All experimental features

**Phase 5: M5.5 Production Polish (12 weeks)**
- Full CLI, deployment, testing, benchmarks

**CRITICAL PATH TOTAL**: 35-40 weeks (~9 months)

---

### Recommended Approach: Phased Delivery

**Instead of monolithic M4 and M5**, break into sub-milestones:

**M3.FINAL: Complete Acceleration (5 weeks)**
- Speculative decoding
- AWQ/SmoothQuant
- Decode-loop optimizations

**M4.1: Core Scheduling + Name Mapping (6 weeks)**
- Memory-aware scheduler
- Model loading integration (M3.7)
- Basic state persistence

**M4.2: Advanced Scheduling (8 weeks)**
- Multi-stage pipelines
- Convergence detection
- KB reasoning

**M5.1: Performance Optimizations (6 weeks)**
- KV cache optimization
- Pruning utilities
- Reasoning controls

**M5.2: Tool Infrastructure (6 weeks)**
- Tool registry
- MCP service discovery
- Cache management integration (M3.7)

**M5.3: Experimental Features (8 weeks)**
- Test-time adaptation
- Modular specialization
- Protocol evolution

**M5.5: Production Release (12 weeks)**
- CLI, deployment, testing, benchmarks

**PHASED DELIVERY TOTAL**: 51 weeks (~12 months)

---

## Immediate Next Steps (This Week)

**Decision Point**: Choose strategy:

### Option A: Fast Track to v1.0 (9 months)
**This week**: Start M3 completion
1. Implement speculative decoding MVP
2. Evaluate AWQ/SmoothQuant integration points
3. Document deferred items clearly

### Option B: Comprehensive Implementation (12 months)
**This week**: Start M3.FINAL
1. Begin speculative decoding implementation
2. Plan M4.1 sub-milestone in detail
3. Create M4/M5 split proposals

### Option C: Focus on M3.7 Integrations First
**This week**: Start model loading integration
1. Implement M4.1 name mapping in ModelLoader
2. This unlocks architecture-agnostic capabilities
3. Then return to M3 completion

---

## Recommendations

### For v1.0 Success:

1. **Choose Fast Track (Option A)**
   - Focus ruthlessly on critical path
   - Defer experimental features to v2.0
   - Target: v1.0 in Q3 2026 (9 months)

2. **Split M4 and M5 into Sub-Milestones**
   - Makes progress measurable
   - Enables early wins
   - Reduces risk of scope creep

3. **Prioritize M3.7 Model Loading Integration Early**
   - This is **foundational** for v1.0
   - Unlocks architecture-agnostic promise
   - Relatively small effort (3 weeks) with huge value

4. **Defer to v2.0**:
   - LLM-assisted name mapping (M6.5)
   - Multi-stage reasoning pipelines
   - Most M5 experimental features
   - Advanced KB reasoning
   - All M6+ research features

5. **Establish Clear v1.0 Scope**:
   - ✅ Architecture-agnostic model loading (M3.7 integrations)
   - ✅ Efficient inference (FlashAttention, quantization, batching)
   - ✅ Basic scheduling (priority, state persistence)
   - ✅ Production deployment (CLI, containers, observability)
   - ✅ Comprehensive testing and benchmarks
   - ❌ Advanced reasoning (defer to v2.0)
   - ❌ Federated features (defer to v2.0)
   - ❌ Sentient AI (defer to v3.0+)

---

## Success Metrics for v1.0

**Must Have**:
- ✅ Load 5+ architectures (LLaMA, GPT, Mistral, Qwen, Phi) without code changes
- ✅ FlashAttention 2-5× GPU speedup
- ✅ Continuous batching with 8+ concurrent requests
- ✅ Priority-based scheduling
- ✅ 95%+ test coverage
- ✅ CLI with server/chat modes
- ✅ Docker + Kubernetes deployment
- ✅ Benchmarks published vs TGI/vLLM

**Nice to Have** (v2.0):
- Speculative decoding 1.3× speedup
- AWQ/SmoothQuant quality-per-bit gains
- Multi-stage reasoning pipelines
- Advanced KV cache compression

---

## Conclusion

**Current Reality**: We've made excellent progress on M0-M3 core infrastructure, but added significant integration work under "completed" milestones.

**Path Forward**: Choose between 9-month fast track (Option A) or 12-month comprehensive approach (Option B).

**Recommendation**: **Fast Track (Option A)** - Get v1.0 out with solid core features, defer advanced research to v2.0.

**Next Action**: Decide on strategy and commit to specific milestone plan for next 3 months.
