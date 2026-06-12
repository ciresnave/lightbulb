# Lightbulb Roadmap

## Active Development

### AWQ Phase 3 - CUDA Integration
- **Status**: Blocked on toolchain (requires CUDA 13.0 + VS 2022)
- **Goal**: Complete AWQ quantization with CUDA kernel support
- **Dependencies**: User to upgrade CUDA 12.4 → 13.0

### Candlelight Ecosystem Integration
- **Status**: ✅ Complete for Lightbulb
- **Next**: Apply to MLMF and Cognition projects
- **Benefit**: Unified Candle dependency management across all projects

## Proposed: K-D Introspection Toolkit (KDIT)

### Vision
Give practitioners live, comparable proxies for **Kolmogorov Complexity (K)** and **Logical Depth (D)** across models, layers, data, and training runs. Expose actionable signals that steer pruning, quantization, distillation, architecture tweaks, curriculum, and data deduplication.

### Core Philosophy
**Emergent sweet-spot ≈ low K, sufficient D**

- **Noise Region** (high K, low D): Memorization, overparameterization
- **Regularity Region** (low K, low D): Underfitting, too-easy data
- **Emergence Region** (low K, high D): Sweet spot - efficient yet capable
- **Strange Region** (high K, high D): Hidden structure, entangled design

### K (Compressibility) Proxies

1. **MDL/Description Length**: Bits to store weights + optimizer state + sparse masks + low-rank factors
2. **Neural Compression Ratio**: `size(zip(weights)) / size(weights)` using LZ4/zstd + learned codebooks
3. **Weight Entropy & Rank**: Per-layer entropy, spectral decay, low-rank reconstruction error
4. **Activation Compressibility**: Compression ratio of mini-batch activations (opt-in sampling)
5. **Dataset Redundancy Score**: Near-duplicate detection via embedding dedup + cluster entropy

### D (Irreducible Work) Proxies

1. **Sequential/Effective Depth**: Critical-path ops, unrollable loops, recurrence steps
2. **Compute-Work**: Real FLOPs, memory traffic, kernel launch count; per-token for seq models
3. **Convergence Work**: Gradient steps to reach target loss for frozen subtask (few-shot probes)
4. **Iterative Reasoning Steps**: Number of refinement passes/decoder layers used (adaptive depth)
5. **Causal Step Dependence**: Attention-path length / dependency radius for tokens/patches

### Proposed Components

#### 1. Instrumentation API
```rust
// Rust-native implementation for Lightbulb
use lightbulb::kdit::{KDProbe, KDSession};

let probe = KDProbe::builder()
    .compress_weights(true)
    .sample_activations(0.1)  // 10% of batches
    .compute_backend(ComputeBackend::Cuda)
    .build();

let mut session = KDSession::new(&model, probe);

for batch in train_loop() {
    session.tick_start(batch.id);
    let loss = train_step(&model, &batch)?;
    session.tick_end(loss);
}

let summary = session.finalize()?;
```

#### 2. KDBoard (Visualization)
- **Quadrant Map**: Plot (K̂, D̂) for layers/modules/runs with drift vectors
- **Pareto Frontier**: K vs accuracy/robustness; D vs latency/throughput
- **Strange Detector**: Heatmap of modules with high K̂ & high D̂
- **Trajectory Tracer**: Visualize distillation runs as arrows Teacher→Student

#### 3. KD-Aware Schedulers
Automated reactions to KD signals:
- **Auto-Prune**: Trigger pruning on modules with high K̂ & low contribution
- **Rank-Collapse**: Apply low-rank factorization where spectral decay indicates compressibility
- **Quant-Suggest**: Recommend per-layer bit-widths using error-aware K reductions
- **Fuse-or-Cache**: Propose op fusion or KV-cache where D̂ spikes but reuse is high
- **Curriculum Re-weighting**: Shift sampling toward harder clusters when D̂ is starved

```rust
use lightbulb::kdit::controllers::{AutoPrune, RankCollapse, QuantSuggest};

let controller = KDController::builder()
    .add_action(AutoPrune::new(threshold_k_high & contrib_low))
    .add_action(RankCollapse::new(TargetRank::Auto))
    .add_action(QuantSuggest::new(mdl_reduction = 0.35))
    .build();

controller.bind(&mut session, &optimizer);
```

#### 4. KD-Aware Distillation
Minimize K while preserving the "shape" of D:
- **Intermediate Trajectory Loss**: Match multi-step hidden states (CKA/CKA-temporal)
- **Path-Length Regularizer**: Penalize students that collapse iterative depth below minimum
- **Budgeted Emergence Objective**:
  ```
  min_θ  L_task + α·K̂(θ) + β·KL(D_student || D_min_teacher)
  ```

```rust
use lightbulb::kdit::distill::{KDTeacher, KDStudent, kd_loss};

let teacher = KDTeacher::load("teacher.safetensors")?.with_introspection(true);
let student = KDStudent::new(model);

let loss = kd_loss(
    &outputs_student,
    &outputs_teacher,
    KDLossConfig {
        preserve_depth: true,
        depth_profile: teacher.depth_profile(),
        compress_penalty_alpha: 1e-5,
    }
)?;
```

#### 5. Dataset K-D Tools
- **Embedding Dedup & MDL**: Report bits to encode dataset after dedup + learned codebook
- **KD-Curriculum**: Schedule examples to raise D (hard) or drop K (rule-like) based on objective
- **Cluster-to-Rule Mining**: Symbolic pattern miner that proposes simple rules for auxiliary tasks

#### 6. Export & Reproducibility
- **KDFiles** (`.kdf`): Self-contained run artifacts with KD summaries, layer stats, action decisions
- **KDBench**: Tiny benchmark tasks (linear probes, few-step solvers) for consistent D probes

### Detection & Remedies Reference

| Region         | K̂    | D̂    | Likely Cause                       | Suggested Actions                                                                         |
| -------------- | ---- | ---- | ---------------------------------- | ----------------------------------------------------------------------------------------- |
| **Noise**      | High | Low  | Memorization, overparameterization | Prune; low-rank; stronger augmentation; rule-mining aux heads; KD from reasoning teacher  |
| **Regularity** | Low  | Low  | Underfitting, too-easy data        | Increase depth/steps; harder curriculum; longer contexts                                  |
| **Emergence**  | Low  | High | Sweet spot                         | Consider op fusion, caching; maintain depth floor during KD                               |
| **Strange**    | High | High | Hidden structure; entangled design | Factorize modules; re-architect; split tasks; symbolic aux; analyze attention bottlenecks |

### Example: KD-Guided Optimization Loop

```rust
let mut session = KDSession::new(&model, probe);
let mut controller = KDController::new(vec![
    Box::new(AutoPrune::default()),
    Box::new(RankCollapse::default()),
    Box::new(QuantSuggest::new(0.4)),
]);

for epoch in 0..num_epochs {
    for batch in train_loader {
        session.tick_start();
        let loss = train_step(&model, &batch)?;
        session.tick_end(loss);
    }
    
    let report = session.epoch_report();
    
    // React to signals
    let actions = controller.plan(&report)?;
    controller.apply(&mut model, actions)?;
    
    session.mark_epoch(actions);
}

let best = session.select_pareto(
    metric = Metric::ValAccuracy,
    k_budget = KBudget::Reduce(0.40),
    d_floor = DFloor::NoLatencyRegression,
)?;

session.save("run.kdf")?;
```

### Minimal Data Model (Per Module, Per Epoch)

```json
{
  "module": "blocks.7.attn",
  "K": {
    "weights_bits": 5.3e6,
    "entropy": 1.21,
    "compression_ratio": 0.42,
    "rank_approx_error": 0.07
  },
  "D": {
    "critical_path_ops": 3.1e9,
    "sequential_steps": 12,
    "cache_hit_rate": 0.18,
    "attn_path_len_p50": 64
  },
  "actions": ["rank_collapse:k=128"]
}
```

### Implementation Milestones

#### Phase 1: MVP (2-4 weeks)
- [ ] Weight/activation compression metrics
- [ ] Profiler integration (CUDA/CPU)
- [ ] Per-layer K̂ & D̂ reports
- [ ] Basic KDBoard visualization
- [ ] Export to `.kdf` format

#### Phase 2: Automation (4-8 weeks)
- [ ] KD-aware schedulers (prune/quant/rank)
- [ ] Strange detector heatmaps
- [ ] Distillation loss pack
- [ ] Dataset dedup tools
- [ ] Integration with existing Lightbulb quantization pipeline

#### Phase 3: Advanced Features (8-12 weeks)
- [ ] Symbolic pattern miner (optional)
- [ ] Curriculum optimizer
- [ ] AutoML plugin for KD Pareto search
- [ ] Cross-framework export (ONNX with KD metadata)

### Why This Matters for Lightbulb

1. **Surfaces Real Costs**: Shows when you're paying for parameters (K) instead of computation (D)
2. **Goal-Seeking Distillation**: Toward Emergence region instead of blind size-cutting
3. **Unified Optimization Space**: Compression, latency, and generalization become navigable with concrete levers
4. **Production-Ready Intelligence**: Moves beyond "prune 50%" to "prune where K is high and contribution is low"

### Integration Points with Existing Lightbulb Features

- **AWQ Quantization**: Use K̂ proxies to guide per-layer bit-width selection
- **KV Cache Management**: Use D̂ metrics to identify optimal cache sizes/eviction policies
- **Speculative Decoding**: Optimize draft model selection based on K-D Pareto frontier
- **Multi-GPU**: Balance K̂ across devices, minimize D̂ communication overhead
- **LoRA Fine-tuning**: Target low-rank adaptations where K̂ signals compressibility

### Research Questions

1. Can we establish **universal K̂/D̂ calibration** across model families?
2. What are the **stability properties** of KD trajectories during training?
3. Can **symbolic rule mining** provide interpretable shortcuts that reduce both K and D?
4. How do KD profiles correlate with **emergent capabilities** (reasoning, planning)?

### References & Inspiration

- Kolmogorov Complexity (Li & Vitányi)
- Logical Depth (Bennett, 1988)
- Minimum Description Length (Rissanen)
- Neural Compression (Havasi et al., 2019)
- Lottery Ticket Hypothesis (Frankle & Carbin, 2019)
- Spectral Analysis of Neural Networks (Martin & Mahoney, 2021)

---

## Other Future Work

### Short-term
- Complete MLMF integration with Candlelight
- Complete Cognition integration with Candlelight
- Fix axum-server compatibility issue

### Medium-term
- Marlin kernel optimization for 4-bit quantization
- FlashAttention-3 integration when available
- Distributed training coordination improvements

### Long-term
- Full KDIT implementation
- AutoML for inference optimization
- Model surgery toolkit based on KD insights
