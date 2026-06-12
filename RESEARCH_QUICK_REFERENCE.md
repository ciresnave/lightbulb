# Lightbulb Research Integration Quick Reference

## At-a-Glance: What's New

### 🧠 Reasoning Intelligence

| Feature                     | Benefit                               | Target Gain                        | Milestone |
| --------------------------- | ------------------------------------- | ---------------------------------- | --------- |
| Overthinking Detection      | Stop chains before accuracy degrades  | 15-25% compute reduction           | M5        |
| Shorter-is-Better Heuristic | Adaptive max depth per input class    | Neutral/improved accuracy          | M5        |
| Reasoning Path Compression  | Reuse compressed templates            | 20% token reduction                | M5        |
| Selective Rereading         | Re-prefill on low confidence          | <20% extra tokens, higher accuracy | M5        |
| Budget-Aware Controls       | Max chains/samples/verifier frequency | Monotonic quality-cost curves      | M5        |

### ⚡ Dynamic Compute

| Feature               | Benefit                              | Target Gain              | Milestone |
| --------------------- | ------------------------------------ | ------------------------ | --------- |
| Policy Trait System   | Unified adaptation interface         | Pluggable policies       | M4        |
| Early Exit            | Per-token entropy-based termination  | 25-40% layer reduction   | M4        |
| SALM Adaptation       | Difficulty-based resource allocation | 20-30% compute reduction | M4        |
| Dynamic Thresholds    | Multi-armed bandit tuning            | Domain adaptation        | M4        |
| CoLa Depth Adaptation | Skip/repeat layer decisions          | 10-20% depth reduction   | M5        |

### 💾 Memory & System

| Feature            | Benefit                          | Target Gain                        | Milestone |
| ------------------ | -------------------------------- | ---------------------------------- | --------- |
| Tiered KV (MemOSA) | RAM/disk paging with SLA control | Bounded tail latency               | M4        |
| Dynamic Chunking   | Learned sequence segmentation    | Better throughput vs fixed windows | M5        |
| Multi-Agent Memory | Shared/private namespaces        | Stable multi-agent coordination    | M4        |
| R-KV Compression   | Training-free KV reduction       | 30-50% KV memory savings           | M5        |

### 🎯 Adaptive Inference

| Feature                 | Benefit                           | Target Gain                    | Milestone |
| ----------------------- | --------------------------------- | ------------------------------ | --------- |
| Test-Time Policy Tuning | Per-instance parameter adaptation | Measurable accuracy gains      | M5        |
| Text-to-LoRA            | Instant domain adaptation         | Quality gains, negligible TTFT | M5        |
| Learned Routing         | Multi-round MoE/tool routing      | Better quality/latency balance | M4        |
| Modular Hot-Swap        | Plugin specialized experts        | Domain gains, stable latency   | M5        |

### 📊 Evaluation & Training

| Feature                          | Benefit                        | Target Gain                | Milestone |
| -------------------------------- | ------------------------------ | -------------------------- | --------- |
| SPARQ Generators                 | Diverse synthetic problems     | Broader coverage           | M5        |
| ReasoningGym Adapter             | Verifiable reward environments | Reproducible policy tuning | M5        |
| Episode Logging                  | RL-friendly replay format      | Offline training support   | M5        |
| Principal Weight Diagnostics     | Fine-tuning validation         | Quality assurance          | M5        |
| Instruction Alignment Monitoring | Detect/mitigate override       | Reduced drift              | M4        |

### 🔬 Research (M6)

| Feature                                         | Benefit                   | Status              |
| ----------------------------------------------- | ------------------------- | ------------------- |
| RIG (Reasoning+Imagination+Action)              | Unified embodied agent    | Proof-of-concept    |
| Neurosymbolic Integration                       | Neural+symbolic hybrid    | Design spikes       |
| Alternative Architectures (Hyena, Mamba, UNets) | Beyond standard attention | Feasibility studies |

## Key Research Papers by Theme

### Must-Read for Implementation

1. **Self-Adapting Language Models (SALM)** - `self-adapting-language-models.md`
   - Core framework for dynamic compute allocation
   - Directly applicable to scheduler design

2. **Dynamic Neural Networks Survey** - `dynamic-neural-networks-survey.md`
   - Unifies early exit, CoLa, and routing under one framework
   - Policy trait design guide

3. **MemOSA** - `memosa-memory-os.md`
   - System-level memory orchestration
   - Tiered KV paging patterns

4. **Efficient Reasoning Models Survey** - `efficient-reasoning-models-survey.md`
   - Budget controls and verification gates
   - Cost-quality tradeoff mechanisms

### Immediate Value Adds

5. **Thought Terminator** - `thought-terminator.md`
   - Overthinking metrics and calibration
   - Ready-to-use stopping rules

6. **Don't Overthink It** - `dont-overthink-it.md`
   - Shorter chains → better accuracy
   - Confidence-based caps

7. **Reward Modeling as Reasoning** - `reward-modeling-as-reasoning.md`
   - Re-ranking API design
   - Compositional scoring patterns

8. **Router-R1** - `router-r1-rl-routing.md`
   - Learned routing for MoE
   - Multi-round decision making

### Advanced Optimizations

9. **R-KV** - `r-kv-kv-cache-compression.md`
   - Training-free KV compression
   - 1.5× throughput at 34% budget

10. **Text-to-LoRA** - `text-to-lora-instant-adaptation.md`
    - Runtime adapter switching
    - Zero fine-tuning overhead

11. **Dynamic Chunking** - `dynamic-chunking.md`
    - Hierarchical sequence modeling
    - Better than fixed windows

12. **Reasoning Path Compression** - `reasoning-path-compression.md`
    - Template caching strategies
    - 20% token savings

### Research Inspiration

13. **RIG** - `rig-synergizingreasoningandimaginationinendtoendgeneralistpolicy.md`
    - Embodied agent foundation
    - Reasoning + imagination + action in one model

14. **Growing Transformers** - `growing-transformers.md`
    - Modular composition patterns
    - Hot-swap architecture

15. **MIRIX** - `mirix-multi-agent-memory.md`
    - Multi-agent coordination
    - Memory namespace design

## Implementation Priority Suggestions

### High Priority (M4 - Start Now)

1. **Policy Trait System** - Foundation for all dynamic compute features
2. **Basic Early Exit** - Entropy + patience, immediate wins
3. **Tiered KV Orchestration** - Critical for long-context workloads
4. **Instruction Alignment Monitoring** - Prompt program quality guard

### Medium Priority (M5 - Next Quarter)

1. **Reasoning Efficiency Controls** - Budget knobs and overthinking detection
2. **Re-ranking API** - Better output selection without full RL
3. **Reasoning Path Compression** - Template caching for common patterns
4. **SPARQ/ReasoningGym Integration** - Eval infrastructure

### Research Track (M6 - Experimental)

1. **Test-Time Adaptation** - Per-instance tuning experiments
2. **Text-to-LoRA** - Adapter registry prototype
3. **RIG Embodied Agents** - If pursuing agentic use cases
4. **Alternative Architectures** - Mamba/Hyena feasibility

## Acceptance Criteria Cheat Sheet

| Metric               | Good   | Great  | Excellent      |
| -------------------- | ------ | ------ | -------------- |
| Compute Reduction    | 15-20% | 25-30% | 35-40%         |
| Accuracy Degradation | ≤2%    | ≤1%    | 0% or improved |
| Memory Reduction     | 20-30% | 35-45% | 45-50%         |
| Token Savings        | 15-20% | 20-30% | 30%+           |
| Latency Improvement  | 15-20% | 25-35% | 35%+           |

## Quick Start Checklist

- [ ] Review updated `ROADMAP.md`
- [ ] Read integration summary in `ROADMAP_INTEGRATION_SUMMARY.md`
- [ ] Pick 2-3 high-priority M4 features to implement
- [ ] Read corresponding research papers (linked in roadmap)
- [ ] Define Policy trait interface in `src/engine.rs`
- [ ] Set up basic eval harness for acceptance criteria
- [ ] Create GitHub issues for chosen features
- [ ] Implement, measure, iterate

## Questions & Customization

**Want to adjust priorities?** 
- Consider your hardware constraints (CPU-only → prioritize memory optimizations)
- Consider your workload (long-context → tiered KV, reasoning-heavy → efficiency controls)

**Missing something?**
- 70+ other papers in the summaries folder not yet integrated
- Happy to add specific features on request

**Want deeper dives?**
- Each summary has "Why it matters for lightbulb" and "Actionable next steps"
- Can create detailed implementation guides for specific features

---

**Total research integrated**: 30+ papers from 100+ summaries  
**New features added**: 40+ across milestones M4-M6  
**Quantified targets**: 20+ specific acceptance criteria  
**Ready to build!** 🚀
