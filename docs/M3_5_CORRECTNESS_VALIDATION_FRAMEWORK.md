# M3.5 Correctness Validation Framework

**Status**: Design Phase  
**Milestone**: M3.5 Testing & Hardening  
**Owner**: Core Team  
**Created**: October 2025

## Executive Summary

This document defines a comprehensive correctness validation framework to ensure Lightbulb maintains production-grade numerical accuracy across all optimizations and feature combinations. Building on existing test infrastructure (correctness_tests.rs, batched_transformer_correctness.rs), we extend coverage to validate cross-feature interactions, detect regressions early, and provide confidence for production deployment.

**Target**: <2% accuracy delta for all optimizations compared to reference implementations

---

## 1. Current Testing Baseline

### Existing Test Coverage

**Unit Tests** (in-tree):
- `tests/correctness_tests.rs` - BatchedTransformer vs Candle Llama (1e-4 tolerance)
- `tests/batched_transformer_correctness.rs` - Multi-step generation validation
- `tests/flash_attention_tests.rs` - FlashAttention numerical parity (1e-3 tolerance)
- `tests/model_correctness.rs` - Model-level correctness
- `src/model/custom_transformer.rs` - RoPE property tests
- `src/model/custom_transformer_block.rs` - Block-level shape/dimension tests

**Integration Tests**:
- `tests/batch_manager_integration.rs` - BatchManager vs direct Llama
- `tests/parallel_model_manager_integration.rs` - Multi-sequence processing
- `tests/integration_local_model.rs` - Full server integration

**Performance Validation**:
- `examples/benchmark_flashattention.rs` - FlashAttention speedup verification

### Coverage Gaps (M3.5 Addresses These)

1. **Cross-Feature Validation**: No tests for feature combination correctness
   - FlashAttention + quantization + speculative decoding?
   - Sliding window + prefix caching + batching?
   - CPU vs CUDA numerical consistency?

2. **Determinism Verification**: Limited fixed-seed validation
   - Same input/seed → same output guarantee?
   - Reproducibility across runs?

3. **Quality Benchmarks**: No automated quality metrics
   - HELM, MMLU, HumanEval integration?
   - Perplexity benchmarks on standard datasets?

4. **Regression Detection**: Manual performance tracking
   - No automated alerting for degradation
   - No historical trend analysis
   - No automated bisection for regressions

5. **Stress Testing**: Limited load validation
   - What happens at 128k token context?
   - Concurrent request correctness under load?
   - Memory leak detection?

---

## 2. Correctness Validation Framework Design

### 2.1 Cross-Feature Validation Matrix

**Goal**: Test all meaningful feature combinations to ensure optimizations compose correctly

**Test Matrix** (28 configurations):

| Device | Precision | Speculation | FlashAttn | Context | Quantization |
| ------ | --------- | ----------- | --------- | ------- | ------------ |
| CPU    | FP32      | ❌           | ❌         | Short   | None         |
| CPU    | FP32      | ✅ (N=2)     | ❌         | Short   | None         |
| CPU    | FP32      | ❌           | ❌         | Long    | None         |
| CPU    | FP16      | ❌           | ❌         | Short   | Q4_0         |
| CPU    | FP16      | ❌           | ❌         | Long    | Q8_0         |
| CUDA   | FP32      | ❌           | ❌         | Short   | None         |
| CUDA   | FP32      | ✅ (N=2)     | ❌         | Short   | None         |
| CUDA   | FP32      | ❌           | ✅         | Short   | None         |
| CUDA   | FP32      | ✅ (N=2)     | ✅         | Short   | None         |
| CUDA   | FP32      | ❌           | ✅         | Long    | None         |
| CUDA   | FP32      | ✅ (N=3)     | ✅         | Long    | None         |
| CUDA   | FP16      | ❌           | ✅         | Short   | None         |
| CUDA   | FP16      | ❌           | ✅         | Long    | None         |
| CUDA   | FP16      | ❌           | ❌         | Short   | Q4_0         |
| CUDA   | FP16      | ❌           | ✅         | Short   | Q4_0         |
| ...    | ...       | ...         | ...       | ...     | ...          |

**Context Sizes**:
- Short: 512 tokens
- Medium: 4096 tokens
- Long: 32768 tokens
- Max: 128000 tokens (edge case)

**Validation Approach**:
1. **Reference**: CPU FP32 no-optimization baseline
2. **Candidate**: Each feature combination
3. **Comparison**: Logits match within tolerance (relaxed for quantization)
4. **Tolerance Targets**:
   - Full precision (FP32/FP16): <1e-4 relative error
   - Quantized (Q8_0): <1e-2 relative error
   - Quantized (Q4_0): <5e-2 relative error
   - Cross-device (CPU ↔ CUDA): <1e-3 relative error

**Test Harness**: `tests/cross_feature_validation.rs`

```rust
/// Cross-feature validation test matrix
/// 
/// Tests all meaningful combinations of:
/// - Device (CPU, CUDA)
/// - Precision (FP32, FP16)
/// - Quantization (None, Q4_0, Q8_0)
/// - FlashAttention (enabled/disabled)
/// - Speculative decoding (N=1,2,3,4)
/// - Context length (512, 4096, 32768, 128000)

use lightbulb::model::BatchedTransformerConfig;

#[derive(Debug, Clone)]
struct TestConfiguration {
    device: Device,
    dtype: DType,
    quantization: Option<QuantizationType>,
    flash_attention: bool,
    speculation_lookahead: usize,
    context_length: usize,
    batch_size: usize,
}

/// Reference implementation: CPU FP32 no optimizations
fn create_reference_config() -> TestConfiguration {
    TestConfiguration {
        device: Device::Cpu,
        dtype: DType::F32,
        quantization: None,
        flash_attention: false,
        speculation_lookahead: 1,
        context_length: 512,
        batch_size: 1,
    }
}

/// Generate all valid test configurations
fn generate_test_matrix() -> Vec<TestConfiguration> {
    // 28 configurations covering major combinations
    // ...
}

/// Run inference with given configuration
fn run_inference(config: &TestConfiguration, input_tokens: &[u32]) -> Result<Tensor> {
    // Load model with configuration
    // Run forward pass
    // Return logits
}

/// Compare two configurations for numerical consistency
fn compare_configurations(
    reference: &TestConfiguration,
    candidate: &TestConfiguration,
    input_tokens: &[u32],
) -> Result<ValidationResult> {
    let ref_logits = run_inference(reference, input_tokens)?;
    let cand_logits = run_inference(candidate, input_tokens)?;
    
    // Determine tolerance based on quantization
    let (rtol, atol) = determine_tolerance(candidate);
    
    // Compare logits
    let max_diff = compute_max_relative_error(&ref_logits, &cand_logits)?;
    
    Ok(ValidationResult {
        passed: max_diff < rtol,
        max_relative_error: max_diff,
        config_name: format!("{:?}", candidate),
    })
}

#[test]
fn test_cross_feature_validation_matrix() -> Result<()> {
    let test_configs = generate_test_matrix();
    let reference = create_reference_config();
    let input_tokens = vec![1, 2, 3, 4, 5]; // "Hello world"
    
    let mut results = Vec::new();
    for config in test_configs {
        let result = compare_configurations(&reference, &config, &input_tokens)?;
        results.push(result);
        
        if !result.passed {
            eprintln!("FAILED: {} (error: {:.6e})", result.config_name, result.max_relative_error);
        }
    }
    
    // Summary report
    let passed = results.iter().filter(|r| r.passed).count();
    let failed = results.iter().filter(|r| !r.passed).count();
    
    println!("Cross-Feature Validation: {}/{} passed", passed, results.len());
    
    assert_eq!(failed, 0, "Some configurations failed validation");
    Ok(())
}
```

**Integration**: Add to CI pipeline, run on every PR and nightly

---

### 2.2 Determinism Verification

**Goal**: Guarantee reproducible outputs for identical inputs (debugging, auditing, deterministic testing)

**Test Scenarios**:

1. **Single-run determinism**: Same seed → same output
2. **Cross-run determinism**: Restart server → same output
3. **Cross-device determinism**: CPU vs CUDA (within tolerance)

**Test Harness**: `tests/determinism_tests.rs`

```rust
/// Determinism validation tests
/// 
/// Ensures:
/// 1. Fixed seed produces identical outputs across runs
/// 2. Server restart doesn't affect determinism
/// 3. CPU and CUDA produce consistent results (within tolerance)

use lightbulb::model::BatchedTransformer;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[test]
fn test_single_run_determinism() -> Result<()> {
    let seed = 42;
    let input_tokens = vec![1, 2, 3, 4, 5];
    
    // Run 1
    let mut rng1 = ChaCha8Rng::seed_from_u64(seed);
    let logits1 = run_inference_with_rng(&input_tokens, &mut rng1)?;
    
    // Run 2 (same seed)
    let mut rng2 = ChaCha8Rng::seed_from_u64(seed);
    let logits2 = run_inference_with_rng(&input_tokens, &mut rng2)?;
    
    // Must be EXACTLY identical (bit-for-bit)
    assert_tensors_exact(&logits1, &logits2, "determinism")?;
    
    Ok(())
}

#[test]
fn test_cross_device_determinism() -> Result<()> {
    // Same input, CPU vs CUDA should produce consistent results
    let input_tokens = vec![1, 2, 3, 4, 5];
    let seed = 42;
    
    let cpu_logits = run_on_device(&input_tokens, Device::Cpu, seed)?;
    let cuda_logits = run_on_device(&input_tokens, Device::Cuda(0), seed)?;
    
    // Tolerance: 1e-3 (device-specific numerical differences)
    assert_tensors_close(&cpu_logits, &cuda_logits, "cross-device", 1e-3, 1e-5)?;
    
    Ok(())
}

#[test]
fn test_multi_step_determinism() -> Result<()> {
    // Multi-step generation should be deterministic
    let prompt_tokens = vec![1, 2, 3];
    let seed = 42;
    let num_steps = 10;
    
    // Run 1
    let generated1 = generate_sequence(&prompt_tokens, num_steps, seed)?;
    
    // Run 2 (same seed)
    let generated2 = generate_sequence(&prompt_tokens, num_steps, seed)?;
    
    assert_eq!(generated1, generated2, "Multi-step generation not deterministic");
    
    Ok(())
}
```

**Key Implementation Notes**:
- Use `ChaCha8Rng` with fixed seed for all randomness (sampling, dropout if enabled)
- Document any sources of non-determinism (e.g., CUDA kernel launch order for parallel ops)
- Provide API to set global seed: `lightbulb::set_seed(u64)`

---

### 2.3 Quality Benchmarks Integration

**Goal**: Track end-to-end generation quality across optimizations

**Benchmark Suites**:

1. **HELM (Holistic Evaluation of Language Models)**
   - Comprehensive benchmark covering 42 scenarios
   - Tracks: accuracy, calibration, robustness, fairness, efficiency
   - Metrics: Exact match, F1, BLEU, ROUGE, perplexity

2. **MMLU (Massive Multitask Language Understanding)**
   - 57 tasks covering STEM, humanities, social sciences
   - Multiple choice format
   - Metric: Accuracy (0-100%)

3. **HumanEval**
   - 164 Python programming problems
   - Code generation evaluation
   - Metric: pass@k (k=1,10,100)

4. **Perplexity on Standard Datasets**
   - WikiText-2, WikiText-103
   - Metric: Perplexity (lower is better)

**Implementation Strategy**:

**Phase 1 (M3.5)**: Perplexity benchmarks (fastest to implement)
```rust
// tests/quality_benchmarks/perplexity_tests.rs

use lightbulb::model::BatchedTransformer;

#[test]
fn test_perplexity_wikitext2_baseline() -> Result<()> {
    let test_set = load_wikitext2_test()?;
    let model = load_baseline_model()?; // CPU FP32 no optimizations
    
    let perplexity = compute_perplexity(&model, &test_set)?;
    
    // Baseline target (from Candle reference)
    let expected_perplexity = 15.0; // Example
    let tolerance = 0.5; // 3% tolerance
    
    assert!(
        (perplexity - expected_perplexity).abs() < tolerance,
        "Perplexity regression: {:.2} (expected {:.2})",
        perplexity,
        expected_perplexity
    );
    
    Ok(())
}

#[test]
fn test_perplexity_with_flash_attention() -> Result<()> {
    // Verify FlashAttention doesn't degrade quality
    let test_set = load_wikitext2_test()?;
    
    let baseline_model = load_model(FlashAttention::Disabled)?;
    let flash_model = load_model(FlashAttention::Enabled)?;
    
    let baseline_ppl = compute_perplexity(&baseline_model, &test_set)?;
    let flash_ppl = compute_perplexity(&flash_model, &test_set)?;
    
    let delta_percent = (flash_ppl - baseline_ppl).abs() / baseline_ppl * 100.0;
    
    assert!(
        delta_percent < 2.0,
        "FlashAttention caused {:.2}% perplexity change (threshold: 2%)",
        delta_percent
    );
    
    Ok(())
}
```

**Phase 2 (M4+)**: MMLU/HumanEval integration
- Larger scope, requires dataset downloads and scaffolding
- MMLU: Multiple choice evaluation harness
- HumanEval: Code execution sandbox + pass@k computation

**Phase 3 (M5+)**: Full HELM integration
- Most comprehensive, heaviest dependencies
- Consider using HELM as external validation (not in CI)
- Run periodically (weekly) rather than per-commit

**Acceptance Criteria**:
- Perplexity delta <2% for all optimizations (vs baseline)
- MMLU accuracy delta <1% (absolute) for all optimizations
- HumanEval pass@1 delta <5% (absolute) for all optimizations

---

### 2.4 Regression Detection Infrastructure

**Goal**: Automatically detect and alert on performance/quality regressions

**Components**:

1. **CI-Integrated Benchmark Runner**
   - Runs on every PR + nightly builds
   - Fast subset on PR (5min), full suite nightly (30min)
   - Stores results in database (SQLite for now, ClickHouse later)

2. **Automated Alerting**
   - Slack/Discord webhook on >10% performance degradation
   - GitHub issue auto-creation on >5% sustained regression (3+ commits)
   - Email to maintainers on critical failures

3. **Historical Trend Analysis**
   - Track metrics over time (throughput, latency, perplexity)
   - Visualize trends (Grafana + Prometheus for production)
   - Identify gradual degradation (boiling frog problem)

4. **Automated Bisection**
   - When regression detected, auto-bisect to find culprit commit
   - Use `git bisect` + benchmark runner
   - Report in GitHub issue with suspect commit

**Implementation**:

**Database Schema** (`benchmarks.db`):
```sql
CREATE TABLE benchmark_runs (
    id INTEGER PRIMARY KEY,
    commit_hash TEXT NOT NULL,
    commit_message TEXT,
    branch TEXT,
    timestamp INTEGER NOT NULL,
    runner TEXT, -- "GitHub Actions", "local", etc.
    
    -- Metrics
    throughput_tokens_per_sec REAL,
    latency_p50_ms REAL,
    latency_p95_ms REAL,
    latency_p99_ms REAL,
    
    perplexity_wikitext2 REAL,
    memory_peak_mb REAL,
    
    -- Test pass rates
    unit_tests_passed INTEGER,
    unit_tests_total INTEGER,
    integration_tests_passed INTEGER,
    integration_tests_total INTEGER
);

CREATE INDEX idx_commit_hash ON benchmark_runs(commit_hash);
CREATE INDEX idx_timestamp ON benchmark_runs(timestamp);
```

**CI Script** (`.github/workflows/benchmark.yml`):
```yaml
name: Benchmark and Regression Detection

on:
  pull_request:
  schedule:
    - cron: '0 2 * * *' # Nightly at 2 AM UTC

jobs:
  benchmark:
    runs-on: ubuntu-latest-gpu # GPU runner
    steps:
      - uses: actions/checkout@v3
      - name: Run benchmarks
        run: cargo bench --bench regression_suite
      
      - name: Store results
        run: |
          python scripts/store_benchmark_results.py \
            --commit ${{ github.sha }} \
            --branch ${{ github.ref }} \
            --results target/criterion/results.json
      
      - name: Check for regressions
        run: |
          python scripts/detect_regressions.py \
            --commit ${{ github.sha }} \
            --threshold 0.10 \ # 10% threshold
            --notify-slack ${{ secrets.SLACK_WEBHOOK }}
```

**Regression Detection Script** (`scripts/detect_regressions.py`):
```python
import sqlite3
import sys

def detect_regressions(commit_hash, threshold=0.10):
    db = sqlite3.connect('benchmarks.db')
    cursor = db.cursor()
    
    # Get current run
    current = cursor.execute(
        "SELECT throughput_tokens_per_sec, latency_p95_ms, perplexity_wikitext2 "
        "FROM benchmark_runs WHERE commit_hash = ?",
        (commit_hash,)
    ).fetchone()
    
    # Get baseline (7-day rolling average)
    baseline = cursor.execute(
        "SELECT AVG(throughput_tokens_per_sec), AVG(latency_p95_ms), AVG(perplexity_wikitext2) "
        "FROM benchmark_runs "
        "WHERE timestamp > strftime('%s', 'now', '-7 days')"
    ).fetchone()
    
    # Compute deltas
    throughput_delta = (current[0] - baseline[0]) / baseline[0]
    latency_delta = (current[1] - baseline[1]) / baseline[1]
    perplexity_delta = (current[2] - baseline[2]) / baseline[2]
    
    # Check for regressions
    regressions = []
    if throughput_delta < -threshold:
        regressions.append(f"Throughput: {throughput_delta:.1%} decrease")
    if latency_delta > threshold:
        regressions.append(f"Latency (p95): {latency_delta:.1%} increase")
    if perplexity_delta > threshold / 5: # Tighter threshold for quality
        regressions.append(f"Perplexity: {perplexity_delta:.1%} increase")
    
    if regressions:
        print("REGRESSIONS DETECTED:")
        for r in regressions:
            print(f"  - {r}")
        sys.exit(1)
    else:
        print("No regressions detected")
        sys.exit(0)
```

**Alerting**:
- Slack/Discord webhook on regression
- GitHub issue auto-creation
- Grafana alerts for production deployments

---

## 3. Implementation Plan

### Phase 1: Core Infrastructure (Week 1-2)

**Tasks**:
1. Create `tests/cross_feature_validation.rs` with test matrix generator
2. Implement `tests/determinism_tests.rs` with seed control
3. Set up benchmark database (`benchmarks.db` + schema)
4. Write CI scripts for benchmark storage and regression detection

**Deliverables**:
- Cross-feature validation running in CI (on PR)
- Determinism tests passing
- Benchmark database collecting results (nightly)

### Phase 2: Quality Benchmarks (Week 3-4)

**Tasks**:
1. Implement perplexity computation on WikiText-2
2. Create `tests/quality_benchmarks/perplexity_tests.rs`
3. Set perplexity baseline for each feature configuration
4. Integrate into regression detection (perplexity delta <2%)

**Deliverables**:
- Perplexity benchmarks running in CI
- Automated alerting on quality regressions

### Phase 3: Advanced Monitoring (Week 5+)

**Tasks**:
1. Implement automated bisection script
2. Set up Grafana dashboard for historical trends
3. Integrate Prometheus metrics for production monitoring
4. Explore MMLU/HumanEval integration (M4+)

**Deliverables**:
- Automated bisection on regressions
- Historical trend visualization
- Production monitoring dashboards

---

## 4. Success Metrics

**Quantitative**:
- ✅ 28 feature combinations validated in CI (<10 min runtime)
- ✅ 100% determinism for fixed-seed runs (bit-for-bit identical)
- ✅ Perplexity delta <2% for all optimizations vs baseline
- ✅ <5% false positive rate on regression detection
- ✅ Mean time to detection (MTTD) <24 hours for regressions

**Qualitative**:
- Increased confidence in production deployments
- Faster PR review cycle (automated correctness checks)
- Early detection of subtle bugs (cross-feature interactions)
- Historical visibility into performance trends

---

## 5. Maintenance & Evolution

**Ongoing**:
- Update test matrix as new features added
- Tune regression detection thresholds based on false positive rate
- Expand quality benchmarks (MMLU, HumanEval in M4+)
- Monitor CI runtime (optimize if tests become too slow)

**Quarterly Reviews**:
- Analyze regression detection effectiveness
- Identify gaps in test coverage
- Update baselines as models/datasets evolve

---

## Appendix A: Tolerance Guidelines

| Comparison                 | Relative Tolerance | Absolute Tolerance | Notes                               |
| -------------------------- | ------------------ | ------------------ | ----------------------------------- |
| Baseline vs Batched        | 1e-4               | 1e-6               | Same precision, no optimizations    |
| Baseline vs FlashAttention | 1e-3               | 1e-5               | FlashAttention uses F16 temporarily |
| FP32 vs FP16               | 1e-3               | 1e-5               | Precision change                    |
| Full vs Q8_0               | 1e-2               | 1e-4               | 8-bit quantization                  |
| Full vs Q4_0               | 5e-2               | 1e-3               | 4-bit quantization (lossy)          |
| CPU vs CUDA                | 1e-3               | 1e-5               | Device-specific numerics            |
| Perplexity delta           | 2%                 | -                  | Quality benchmark threshold         |

---

## Appendix B: Reference Implementations

**Baseline**: Candle's native `Llama` implementation
- No batching, no optimizations
- CPU FP32
- Direct comparison target for correctness

**Alternative baselines**:
- HuggingFace Transformers (PyTorch): For cross-framework validation
- llama.cpp: For quantization correctness comparison

---

## Conclusion

This framework provides comprehensive correctness validation across:
1. Feature combinations (cross-feature matrix)
2. Reproducibility (determinism tests)
3. Quality (perplexity benchmarks)
4. Regression prevention (automated detection + alerting)

**Immediate focus** (M3.5): Implement Phase 1 (core infrastructure) and Phase 2 (perplexity benchmarks)

**Future work** (M4+): Expand quality benchmarks (MMLU, HumanEval, HELM), production monitoring

This ensures Lightbulb maintains production-grade correctness as we scale to Multi-GPU (M3.6) and beyond.
