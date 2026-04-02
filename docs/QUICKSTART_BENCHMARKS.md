# Quick Start: Running Performance Suite

**Last Updated:** November 24, 2025

---

## Prerequisites

### 1. Model Files
Ensure you have a test model available:
```powershell
# Check if model exists
Test-Path ..\models\llama-3b\model.safetensors
```

If not available, download a small model:
```powershell
# Using Lightbulb's hf-hub integration (if implemented)
.\lightbulb-cli download meta-llama/Llama-3.2-1B
```

### 2. Build Dependencies
```powershell
# Build in release mode (required for accurate benchmarks)
cargo build --release

# Install criterion (benchmark framework)
# Already added to Cargo.toml as dev-dependency
```

---

## Running Benchmarks

### Full Benchmark Suite (~30 minutes)
```powershell
cd lightbulb
cargo bench --bench batched_inference_benchmark
```

### View Results
```powershell
# Open HTML report in browser
start target\criterion\report\index.html
```

### Specific Benchmarks
```powershell
# Only batch sizes
cargo bench --bench batched_inference_benchmark batch_forward_pass

# Only throughput
cargo bench --bench batched_inference_benchmark decode_throughput

# Only prefill lengths
cargo bench --bench batched_inference_benchmark prefill_lengths
```

---

## Running Correctness Tests

### All Tests
```powershell
cargo test --test enhanced_correctness_tests -- --ignored --test-threads=1 --nocapture
```

**Note:** `--test-threads=1` ensures tests run sequentially (avoid GPU conflicts)

### Individual Tests
```powershell
# Test 1: Batch vs sequential (single token)
cargo test --test enhanced_correctness_tests test_batch_vs_sequential_single_token -- --ignored --nocapture

# Test 2: Multi-token generation
cargo test --test enhanced_correctness_tests test_batch_vs_sequential_multi_token -- --ignored --nocapture

# Test 3: Variable lengths
cargo test --test enhanced_correctness_tests test_variable_sequence_lengths -- --ignored --nocapture

# Test 4: KV cache consistency
cargo test --test enhanced_correctness_tests test_kv_cache_consistency -- --ignored --nocapture

# Test 5: Edge case - empty prompt
cargo test --test enhanced_correctness_tests test_edge_case_empty_prompt -- --ignored --nocapture

# Test 6: Edge case - max length
cargo test --test enhanced_correctness_tests test_edge_case_max_length -- --ignored --nocapture

# Test 7: Dynamic batch completion
cargo test --test enhanced_correctness_tests test_batch_dynamic_completion -- --ignored --nocapture

# Test 8: Attention masking
cargo test --test enhanced_correctness_tests test_attention_masking -- --ignored --nocapture
```

---

## Expected Results

### Benchmarks

#### Batch Forward Pass
```
batch_forward_pass/1    time: 45.2 ms   throughput: 22.1 elem/s
batch_forward_pass/2    time: 48.3 ms   throughput: 41.4 elem/s  (1.9x)
batch_forward_pass/4    time: 54.7 ms   throughput: 73.1 elem/s  (3.3x)
batch_forward_pass/8    time: 68.1 ms   throughput: 117.4 elem/s (5.3x)
batch_forward_pass/16   time: 95.2 ms   throughput: 168.1 elem/s (7.6x)
batch_forward_pass/32   time: 152.8 ms  throughput: 209.4 elem/s (9.5x)
```

**Target:** 6-10x speedup at batch_size=8-16

#### Batched vs Sequential
```
sequential_8_requests   time: 3.24 s
batched_8_requests      time: 485 ms    (6.7x faster)
```

**Target:** 6x+ speedup

### Correctness Tests

All tests should output:
```
✓ All tokens match between batch and sequential processing
✓ KV cache produces consistent results
✓ Attention masking correctly ignores padding
... (etc.)
```

**Target:** 100% pass rate (no token mismatches)

---

## Troubleshooting

### Model Not Found
```
Error: model not found at ../models/llama-3b
```

**Solution:** Update `MODEL_PATH` in benchmark/test files or download model

### Out of Memory
```
Error: CUDA out of memory
```

**Solutions:**
1. Reduce batch size in tests
2. Use CPU instead: `--features cpu`
3. Use smaller model (1B instead of 3B)

### Criterion Not Found
```
Error: no such subcommand: `bench`
```

**Solution:** Ensure `criterion` in `[dev-dependencies]`:
```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
```

### Tests Hang
```
Test running forever...
```

**Solution:** Use `--test-threads=1` to avoid GPU contention:
```powershell
cargo test -- --ignored --test-threads=1
```

---

## Interpreting Results

### Good Performance Indicators
- ✅ Batch size 8: 6-8x speedup
- ✅ Batch size 16: 9-12x speedup
- ✅ Linear scaling up to batch size 16
- ✅ Memory usage grows linearly with batch size

### Warning Signs
- ⚠️ Speedup plateaus before batch size 8 (underutilization)
- ⚠️ Speedup decreases at larger batches (memory bound)
- ⚠️ High variance in measurements (system instability)

### Correctness Issues
- ❌ Token mismatches (batching bug)
- ❌ Non-deterministic outputs (cache inconsistency)
- ❌ Accuracy degradation (attention masking issue)

---

## Next Steps After Running Suite

### If Results Match Expectations
1. ✅ Document baseline performance
2. ✅ Proceed with continuous batching implementation
3. ✅ Set up monitoring in production

### If Results Below Target
1. 🔍 Profile with `cargo flamegraph`
2. 🔍 Check GPU utilization with `nvidia-smi`
3. 🔍 Review cache hit rates
4. 🔍 Analyze batch formation patterns

### If Tests Fail
1. 🐛 Debug specific test case
2. 🐛 Compare tensor shapes/values
3. 🐛 Verify cache allocation/deallocation
4. 🐛 Check attention mask construction

---

## Performance Monitoring in Production

### Key Metrics to Track
```rust
// In ParallelModelManager
pub struct PerformanceMetrics {
    pub throughput_tokens_per_sec: f64,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    pub batch_size_avg: f64,
    pub gpu_utilization_percent: f64,
    pub cache_hit_rate: f64,
}
```

### Alerting Thresholds
- Throughput < 100 tok/s (batch=8) → Investigate
- Latency P95 > 500ms → Capacity issue
- GPU utilization < 70% → Underutilization
- Cache hit rate < 80% → Eviction policy issue

---

## Documentation Reference

For detailed information, see:

- **Architecture:** `docs/ARCHITECTURE.md`
- **Performance Summary:** `docs/PERFORMANCE_INITIATIVE_SUMMARY.md`
- **FlashAttention-3:** `docs/FLASHATTENTION3_RESEARCH.md`
- **Continuous Batching:** `docs/CONTINUOUS_BATCHING_DESIGN.md`
- **Quantization:** `docs/ADVANCED_QUANTIZATION_RESEARCH.md`

---

## Contact & Support

**Issues:** Open GitHub issue with benchmark results  
**Discussions:** Join project Discord/Slack  
**Contributions:** See CONTRIBUTING.md

---

**Happy Benchmarking! 🚀**
