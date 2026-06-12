# Roadmap Integration Summary

## Overview

I've integrated insights from your extensive research documentation (100+ paper summaries) into the Lightbulb roadmap. The integration maintains the existing structure and format while adding valuable, actionable features backed by recent research.

## Key Additions by Milestone

### M3 — Acceleration Features (0.4)

**Enhanced with CPU-First Optimizations:**

1. **CPU Kernel Optimizations**
   - Kernel fusion (bias+gelu, matmul+add) - 10%+ throughput
   - Cache-friendly blocking for attention/GEMM - 20%+ cache miss reduction
   - Micro-prefetch for small-batch GEMM - improved tail latency
   - int8 GEMM with quantization-aware accumulation

2. **Blocked Sparsity and Quantization**
   - Configurable block-size kernels
   - Per-block calibration for quant+sparsity interaction
   - Mixed-precision accumulation
   - Target: 30% throughput improvement, ≤1% accuracy loss

3. **Per-Layer Sparsity Masks**
   - Compact formats (bit-packed, RLE)
   - Tile-aligned, branch-free masked kernels
   - Runtime path selection (dense/masked)

### M4 — Advanced Scheduling (0.5)

**Massively enhanced with systems-level features:**

1. **Tiered Memory Orchestration (MemOSA)**
   - RAM/disk paging for KV cache
   - SLA-driven admission control
   - Bounded tail latency guarantees

2. **Learned Routing Policies (Router-R1)**
   - Multi-round routing decisions for MoE
   - Quality/latency tradeoff balancing
   - Logging infrastructure for offline RL tuning

3. **Instruction Alignment Monitoring**
   - Detect instruction override during multi-turn reasoning
   - Automatic restatement triggers
   - Drift detection and mitigation

4. **Multi-Agent Coordination (MIRIX)**
   - Shared/private memory namespaces
   - Concurrency guards for stable multi-agent runs
   - No data races, bounded overhead

5. **Dynamic Compute Allocation (Unified Policy System)**
   - Policy trait for pluggable adaptation signals
   - Self-adapting inference (SALM)
   - Early exit with entropy + patience + dynamic thresholds
   - Per-input difficulty-based resource allocation
   - 25-40% compute reduction target with ≤2% accuracy loss

### M5 — Frontier Options (0.6)

**Massively expanded with:**

1. **Reasoning Efficiency Controls**
   - Budget-aware decoding (max chains, samples, verifier frequency)
   - Overthinking detection and mitigation
   - Shorter-is-better heuristic with confidence-based stopping
   - Re-ranking API with uncertainty calibration
   - Compositional scoring (style + correctness + safety)
   - Target: 15-25% compute reduction with neutral/improved accuracy

2. **Reasoning Path Compression**
   - Cache compressed templates for recurring tasks
   - Selective reread/re-prefill on low confidence
   - Chunked KV refresh
   - 20% token reduction target

3. **Adaptive Chunking**
   - Configurable fixed vs learned segmentation
   - Chunk-level cache reuse
   - Boundary logging and reuse rate tracking

4. **Test-Time Adaptation**
   - Per-instance policy gradient tuner (budget-capped)
   - Text-to-LoRA adapter registry
   - Prompt-metadata-based adapter selection
   - Domain specialization without retraining

5. **Modular Specialization (Growing Transformers)**
   - Hot-swap framework for specialized experts
   - Frozen substrate with pluggable modules
   - Stable latency with measurable domain gains

6. **Evaluation Infrastructure**
   - SPARQ-like synthetic problem generators
   - ReasoningGym adapter for verifiable rewards
   - Broader coverage across difficulty levels

7. **RL/Training Support**
   - Episode traces in replay-friendly format
   - Async rollout logging (AReaL patterns)
   - Principal weight diagnostics for fine-tuning validation

### M6 — Research Explorations (0.7+) **NEW MILESTONE**

**Added entirely new research track:**

1. **Embodied Agent Foundation (RIG)**
   - Unified reasoning + imagination + action control
   - Joint learning of logical inference and predictive modeling
   - Self-correction and planning capabilities

2. **Neurosymbolic Integration**
   - Hybrid neural/symbolic reasoning
   - Graph-enhanced planning
   - Asynchronous plan reasoning

3. **Alternative Architectures**
   - Hyena hierarchies
   - Mamba SSM (selective state spaces)
   - Autoregressive UNet experiments
   - Candle compatibility feasibility studies

## Research Papers Integrated

### Core Efficiency & Reasoning

- `efficient-reasoning-models-survey.md` - Budget controls, verification gates
- `dynamic-neural-networks-survey.md` - Unified dynamic compute framework
- `thought-terminator.md` - Overthinking detection
- `optimal-inference-length.md` - Shorter-is-better strategies
- `dont-overthink-it.md` - Confidence-based stopping
- `self-adapting-language-models.md` - SALM adaptive inference
- `early-exit-nlp-survey.md` - Early exit strategies

### Memory & System Design

- `memosa-memory-os.md` - Tiered KV orchestration
- `mirix-multi-agent-memory.md` - Multi-agent coordination
- `dynamic-chunking.md` - Adaptive sequence segmentation
- `r-kv-kv-cache-compression.md` - KV compression strategies

### Reasoning Path Optimization

- `reasoning-path-compression.md` - Template compression and reuse
- `rereading-improves-reasoning.md` - Selective re-prefill
- `reward-modeling-as-reasoning.md` - Re-ranking and scoring APIs
- `diagnosing-instruction-overriding.md` - Instruction alignment

### Adaptive & Learned Systems

- `router-r1-rl-routing.md` - Learned routing policies
- `text-to-lora-instant-adaptation.md` - Runtime adapter selection
- `seek-in-the-dark-ttilpg.md` - Test-time policy tuning
- `growing-transformers.md` - Modular composition

### Training & Evaluation

- `sparq-synthetic-problems.md` - Quality diversity problem generation
- `reasoning-gym-rl-envs.md` - Verifiable reward environments
- `areal-async-rl-reasoning.md` - Async RL patterns
- `lift-the-veil-principal-weights.md` - Weight diagnostics

### Advanced Research

- `rig-synergizingreasoningandimaginationinendtoendgeneralistpolicy.md` - Embodied agents
- `graphenhancedlargelanguagemodelsinasynchronousplanreasoning.md` - Graph planning
- `hyenahierarchytowardslargerconvolutionallanguagemodels.md` - Hyena mixers
- `mambalineartimesequencemodelingwithselectivestatespaces.md` - Mamba SSM
- `frombytestoideas-languagemodelingwithautoregressiveunets.md` - AR UNets

## Integration Philosophy

All additions follow these principles:

1. **Candle-First**: Only features that can be implemented with Candle or have clear Candle integration paths
2. **Measurable**: Every feature has concrete acceptance criteria with quantified targets
3. **Pragmatic**: Focused on deployable wins (throughput, latency, memory, quality)
4. **Portable**: CPU/WGPU/CUDA support where applicable
5. **Well-Referenced**: Every addition links back to specific research summaries

## Release Track Updates

Updated from 6 milestones (0.1-0.6) to 7 (0.1-0.7+):

- **0.5** now emphasizes "dynamic compute allocation" and "advanced scheduling"
- **0.6** expanded to include "reasoning efficiency controls," "test-time adaptation," and "modular specialization"
- **0.7+** new research track for embodied agents, neurosymbolic integration, and alternative architectures

## Quantified Impact Targets

The integrated roadmap now has specific, measurable goals:

- **Compute Reduction**: 15-40% across different techniques
- **Accuracy Preservation**: ≤2% degradation tolerance
- **Memory Reduction**: 30-50% for KV cache optimizations
- **Token Savings**: 20% for path compression
- **Latency Improvements**: 30% p95 reduction for MoE routing
- **Throughput Gains**: 1.5-1.6× for various optimizations

## Next Steps

1. **Review the updated ROADMAP.md** - Ensure alignment with your vision
2. **Prioritize milestones** - Decide which M4-M6 features to tackle first
3. **Create GitHub issues** - Break down milestones into trackable tasks
4. **Set up eval harness** - Implement the benchmarking infrastructure for acceptance criteria
5. **Document API contracts** - Define the Policy trait, scoring API, and adapter registry interfaces

## Questions to Consider

1. Should any features be moved between milestones based on dependencies?
2. Are the acceptance criteria realistic for your target hardware?
3. Which research explorations (M6) are most interesting for early prototyping?
4. Do you want to add any features from papers I didn't highlight?

---

Total papers integrated: ~30+ most relevant from your 100+ summary collection
Lines added to roadmap: ~150
New milestone created: M6 (Research Explorations)
