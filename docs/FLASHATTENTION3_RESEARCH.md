# FlashAttention-3 Integration Research

**Date:** November 24, 2025  
**Status:** Research Phase  
**Target:** Upgrade from FlashAttention-2 to FlashAttention-3

---

## Executive Summary

**Recommendation:** Monitor Candle for FlashAttention-3 support. Integrate once available.

**Expected Benefit:** 1.5-2x additional speedup over FlashAttention-2  
**Implementation Effort:** Low (API change only)  
**Risk:** Low (drop-in replacement)

---

## FlashAttention Evolution

### FlashAttention-1 (2022)
- **Innovation:** Tiling and recomputation to reduce HBM access
- **Speedup:** 2-4x vs standard attention
- **Limitation:** Forward pass only, no backward for training

### FlashAttention-2 (2023) - CURRENT
- **Improvements:**
  - 2x faster than FA-1 (better parallelism)
  - Supports training (backward pass)
  - Optimized for A100/H100
- **Status:** ✅ Integrated via Candle (`candle-flash-attn`)
- **Performance:** 2-4x memory reduction, 2-4x speedup vs standard

### FlashAttention-3 (2024)
- **Paper:** "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"
- **Target:** H100 GPUs (Hopper architecture)
- **Key Innovations:**
  1. **Asynchronous compute & memory ops** (WGMMA + TMA)
  2. **FP8 low-precision** support (2x throughput)
  3. **Incoherent processing** (overlap softmax with GEMM)
  4. **Ping-pong scheduling** (hide memory latency)

**Performance vs FA-2:**
- FP16: 1.5-2.0x faster on H100
- FP8:  2.0-2.5x faster on H100 (when model supports it)
- A100: Minimal benefit (lacks Hopper features)

---

## Current State in Candle

### FlashAttention-2 Integration

Lightbulb currently uses FlashAttention-2 via Candle:

```rust
// In custom_transformer_block.rs
use candle_flash_attn::flash_attn;

let attn_output = flash_attn(
    &q,           // [batch, seqlen, num_heads, head_dim]
    &k_cache,     // [batch, cache_len, num_kv_heads, head_dim]
    &v_cache,
    softmax_scale,
    causal,       // true for autoregressive
)?;
```

**Enabled via:** `features = ["flash-attn"]` in Cargo.toml

### FlashAttention-3 Status

**As of November 2025:**
- ❌ Not yet integrated in Candle
- ⏳ Official FA-3 released by Tri Dao (original author)
- 🔍 Community tracking: https://github.com/huggingface/candle/issues/...

**Blockers:**
1. FA-3 CUDA code needs adaptation for Candle's architecture
2. Requires CUDA 12.0+ and Hopper GPU (H100, H200, B100)
3. FP8 support requires additional infrastructure

---

## Integration Plan

### Phase 1: Monitor Candle (CURRENT)
- Watch Candle repository for FA-3 PR/release
- Test FA-3 on H100 if access available
- Benchmark FA-2 vs FA-3 on representative workloads

### Phase 2: API Update (When Available)
**Estimated Effort:** 1-2 days

1. **Update Candle dependency:**
```toml
[dependencies]
candlelight = { version = "0.x", features = ["flash-attn-3"] }
```

2. **Update attention call (likely minimal):**
```rust
// Expected API (may differ):
use candle_flash_attn::flash_attn_v3;

let attn_output = flash_attn_v3(
    &q, &k_cache, &v_cache,
    softmax_scale,
    causal,
    use_fp8,  // New parameter?
)?;
```

3. **Feature flag:**
```toml
# In Cargo.toml
[features]
flash-attn-2 = ["candlelight/flash-attn"]
flash-attn-3 = ["candlelight/flash-attn-3"]
cuda-full = ["flash-attn-3", ...]
```

### Phase 3: Validation
1. **Correctness tests:** Verify FA-3 matches FA-2 outputs
2. **Performance benchmarks:** Measure actual speedup
3. **Regression testing:** Ensure no degradation on A100/consumer GPUs

### Phase 4: Rollout
- Default to FA-3 on H100+ GPUs
- Fall back to FA-2 on older hardware
- Document GPU-specific performance characteristics

---

## Expected Performance Gains

### Scenario 1: H100 GPU, FP16
- **Current (FA-2):** 10-50x speedup vs sequential
- **With FA-3:** 15-75x speedup (1.5x improvement)
- **Use case:** General production deployment

### Scenario 2: H100 GPU, FP8 (if model supports)
- **Current (FA-2):** 10-50x speedup
- **With FA-3:** 20-125x speedup (2.0-2.5x improvement)
- **Use case:** Maximum throughput, accuracy trade-off acceptable

### Scenario 3: A100 GPU
- **Current (FA-2):** 10-50x speedup
- **With FA-3:** 10-50x speedup (no improvement, lacks Hopper features)
- **Action:** Stay on FA-2

### Scenario 4: Consumer GPUs (4090, 4080)
- **Current (FA-2):** 5-20x speedup
- **With FA-3:** 5-20x speedup (no improvement)
- **Action:** Stay on FA-2

---

## Hardware Requirements

### Minimum for FA-3 Benefits
- **GPU:** NVIDIA H100, H200, or B100 (Hopper/Blackwell)
- **CUDA:** 12.0+
- **Compute Capability:** 9.0+ (Hopper)
- **Driver:** R525+

### Testing Hardware Checklist
```bash
# Check GPU architecture
nvidia-smi --query-gpu=name,compute_cap --format=csv

# Verify CUDA version
nvcc --version

# Check driver
nvidia-smi
```

---

## FP8 Considerations

### What is FP8?
- **Format:** 8-bit floating point (4-bit exponent, 3-bit mantissa)
- **Precision:** Lower than FP16, higher than INT8
- **Benefit:** 2x throughput vs FP16 on H100

### When to Use FP8
✅ **Use if:**
- Model trained/fine-tuned with FP8
- Accuracy loss acceptable (<1% degradation)
- Maximum throughput required

❌ **Avoid if:**
- Model not FP8-aware (significant accuracy loss)
- Precision-critical tasks (math, coding)
- No access to Hopper GPUs

### FP8 Integration Path
1. **Phase 1:** FA-3 with FP16 (default, safe)
2. **Phase 2:** Benchmark FP8 accuracy on test tasks
3. **Phase 3:** Offer FP8 as optional feature

---

## Alternative: Custom FA-3 Integration

If Candle integration is delayed, consider direct integration:

**Pros:**
- Access FA-3 features immediately
- Full control over implementation

**Cons:**
- High complexity (CUDA kernel expertise required)
- Maintenance burden (track FA-3 updates)
- Build system complexity (nvcc, PTX, etc.)

**Estimated Effort:** 2-4 weeks full-time

**Recommendation:** ❌ Wait for Candle integration  
Candle team has more CUDA expertise and will handle edge cases better.

---

## Monitoring Checklist

### Weekly Tasks
- [ ] Check Candle GitHub for FA-3 related issues/PRs
- [ ] Monitor Tri Dao's FA-3 repository for updates
- [ ] Test Candle nightlies if FA-3 appears

### Preparation Tasks
- [ ] Set up H100 access (cloud instance or local)
- [ ] Create FA-2 baseline benchmarks
- [ ] Design FA-3 validation test suite
- [ ] Draft migration guide for users

---

## Benchmarking Plan (When FA-3 Available)

### Metrics to Measure
1. **Latency:** First token time (prefill), per-token time (decode)
2. **Throughput:** Tokens/second at various batch sizes
3. **Memory:** Peak VRAM usage
4. **Accuracy:** KL divergence vs FA-2 outputs

### Test Configurations
| Batch Size | Seq Length | Model     | GPU  |
| ---------- | ---------- | --------- | ---- |
| 1          | 512        | Llama-7B  | H100 |
| 8          | 512        | Llama-7B  | H100 |
| 16         | 1024       | Llama-13B | H100 |
| 32         | 2048       | Llama-70B | H100 |
| 1          | 512        | Llama-7B  | A100 |

### Success Criteria
- ✅ 1.5x+ speedup on H100 vs FA-2
- ✅ <0.1% accuracy degradation
- ✅ No memory regression
- ✅ Stable across batch sizes

---

## Risk Mitigation

### Risk 1: API Breaking Changes
**Mitigation:** Support both FA-2 and FA-3 via feature flags

### Risk 2: Performance Regression on Older GPUs
**Mitigation:** Auto-detect GPU architecture, use FA-2 for non-Hopper

### Risk 3: Accuracy Issues with FP8
**Mitigation:** Default to FP16, require explicit opt-in for FP8

### Risk 4: Build Complexity
**Mitigation:** Rely on Candle's prebuilt binaries when possible

---

## Timeline Estimate

| Phase         | Duration | Dependencies         |
| ------------- | -------- | -------------------- |
| Monitoring    | Ongoing  | Candle team progress |
| Integration   | 1-2 days | Candle FA-3 release  |
| Testing       | 3-5 days | H100 access          |
| Documentation | 1 day    | Test results         |
| Rollout       | 1 day    | Approval             |

**Total Time:** ~1 week after Candle integration

---

## References

### Papers
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)

### Repositories
- [Official FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [Candle](https://github.com/huggingface/candle)
- [Candle Flash Attention](https://github.com/huggingface/candle/tree/main/candle-flash-attn)

### Tracking
- Candle Issue: TBD (monitor for FA-3 discussion)
- Community Discord: Hugging Face Discord #candle channel

---

**Next Steps:**
1. Set up GitHub watch on Candle repository
2. Join Hugging Face Discord for updates
3. Prepare H100 benchmarking environment
4. Create migration checklist when FA-3 lands

**Status:** ⏳ Waiting on upstream (Candle integration)
