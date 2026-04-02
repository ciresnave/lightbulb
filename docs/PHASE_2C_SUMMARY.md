# Phase 2C: Performance Monitoring & Batch Analysis

## Overview
Phase 2C adds comprehensive performance monitoring and batch statistics to understand the current sequential processing approach and establish baselines for future optimization.

## Implemented Features

### 1. **BatchStats Structure**
A comprehensive performance tracking system that captures:

```rust
pub struct BatchStats {
    pub total_batches: usize,
    pub total_requests_processed: usize,
    pub total_tokens_generated: usize,
    pub total_forward_time_ms: f64,
    pub prefill_requests: usize,
    pub decode_requests: usize,
}
```

### 2. **Performance Metrics**

#### Computed Metrics:
- **Average Batch Size**: `total_requests / total_batches`
- **Tokens Per Second**: `total_tokens / (total_forward_time_ms / 1000)`
- **Average Forward Time**: `total_forward_time_ms / total_batches`
- **Prefill/Decode Ratio**: Understanding workload composition

#### Tracked Operations:
- Forward pass timing (per request)
- Token generation counts
- Batch composition (Prefill vs Decode)

### 3. **API Methods**

```rust
// Get current statistics
pub fn stats(&self) -> &BatchStats

// Reset statistics (useful for benchmarking specific workloads)
pub fn reset_stats(&mut self)
```

### 4. **New Test: `test_batch_performance`**

A comprehensive performance test with 6 concurrent requests generating 10 tokens each.

**Results** (llama-3b, f32, CPU):
```
Total batches processed: 10
Total requests: 60
Total tokens generated: 60
Average batch size: 6.00
Total forward time: 151808.26 ms
Average forward time per batch: 15180.83 ms
Throughput: 0.40 tokens/sec
Prefill operations: 6
Decode operations: 54
Prefill/Decode ratio: 0.11
```

## Key Insights

### Current Architecture Limitations

1. **Sequential Processing**:
   - Each request processed individually within batch
   - N requests = N forward() calls
   - Cannot leverage parallelism across requests

2. **Per-Request Caches**:
   - Each request maintains independent `Cache`
   - Necessary due to Llama model API: `forward(&mut self, x: &Tensor, index_pos: usize, cache: &mut Cache)`
   - Model expects single cache, not batch of caches

3. **Prefill vs Decode Ratio**:
   - Prefill: 1 operation per request (process full prompt)
   - Decode: `max_new_tokens` operations per request
   - Ratio of 0.11 means 91% of operations are decode steps

### Performance Baseline

**CPU (f32) Performance**:
- ~0.40-0.55 tokens/second with llama-3b
- ~15 seconds per batch step with 6 requests
- Dominated by model forward time

## Why Not True Batching Yet?

The Candle Llama model's API is designed for single-request inference:

```rust
pub fn forward(&mut self, x: &Tensor, index_pos: usize, cache: &mut Cache) 
    -> Result<Tensor>
```

**Challenges**:
1. Takes single `Cache`, not `Vec<Cache>` or batched cache
2. `index_pos` is scalar (single position), not per-request positions
3. KV cache management assumes single sequence

**Solution Path** (Phase 2D):
- Integrate with `BatchExecutor`'s `ScatteredKvCache`
- Use `get_indices_and_mask()` for proper cache indexing
- Either:
  - Modify Llama model to accept batched cache (fork/custom)
  - Use ScatteredKvCache to present unified view

## Test Coverage

✅ **39 Total Tests** (34 unit + 5 model tests):
- All existing tests pass
- New performance test validates statistics
- Multi-request test now shows batch metrics

## Documentation Improvements

Added comprehensive inline documentation:
- `BatchStats` structure and methods
- Performance metric calculations
- Current limitations clearly marked with `// TODO:`
- Rationale for sequential processing

## Next Steps (Phase 2D)

### True Batched Forward Pass with ScatteredKvCache

**Goal**: Process all requests in batch with single forward() call

**Approach**:
1. Modify or wrap Llama model to accept batched cache
2. Integrate ScatteredKvCache for KV management
3. Handle mixed Prefill/Decode batches:
   - Separate into groups by state
   - Prefill group: variable sequence lengths
   - Decode group: single token per request
4. Proper cache index management via BatchExecutor

**Expected Improvements**:
- Reduce forward() calls from N to 1-2 per batch
- Enable GPU parallelism across requests
- Better memory locality
- 5-10x throughput improvement (CPU)
- 10-50x throughput improvement (GPU)

### Additional Optimizations (Phase 2E)

1. **Sampling Strategies**:
   - Top-k, top-p (nucleus sampling)
   - Temperature scaling
   - Beam search

2. **GGUF Support**:
   - Load quantized models
   - 4-bit, 8-bit quantization
   - Candle's `quantized::gguf_file` module

3. **Flash Attention**:
   - Enable `use_flash_attn` in loader
   - 2-3x memory efficiency
   - Faster attention computation

4. **Streaming Responses**:
   - Callback interface for token-by-token delivery
   - Don't wait for full completion

## Conclusion

Phase 2C establishes comprehensive performance monitoring that will be critical for validating improvements in Phase 2D. The statistics clearly show:

- Current sequential approach works correctly
- Throughput baseline established (~0.4-0.5 tok/s CPU)
- Prefill/Decode ratio understood (1:9)
- Clear path forward for optimization

The infrastructure is now instrumented and ready for the major performance leap that true batched inference will provide.

**Status**: ✅ Phase 2C Complete
**Next**: Phase 2D - ScatteredKvCache Integration & True Batching
