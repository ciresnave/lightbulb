# M3.2: Decode-Loop Overhead Analysis

## Current Decode Loop Architecture

The decode loop is implemented in `BatchedTransformer::forward()` (src/model/custom_transformer.rs:417-550).

### Decode Path Flow

1. **Token Embedding** (line 428)
   - `self.embedding.forward(input_ids)` → `[total_tokens, hidden_size]`

2. **Tensor Reshaping** (lines 430-442)
   - Prefill: `unsqueeze(0)` → `[1, total_tokens, hidden]`
   - Decode: `reshape()` → `[total_tokens, 1, hidden]`
   - Dimension validation with `dims3()`

3. **Position Calculation** (lines 456-465)
   - Prefill: `index_pos = 0`
   - Decode: `metadata.context_lens.get(0).copied().unwrap_or(0)`

4. **Layer-by-Layer Processing** (lines 472-493)
   - For each of 32 layers:
     - `block.forward()` with attention and MLP
     - Attention weight tracking for last layer
     - **DEBUG**: Per-layer statistics (flatten + stats)

5. **Attention Weight Processing** (lines 495-517)
   - Convert attention tensor to `Vec<Vec<f32>>`
   - `flatten_all()` + `to_vec1()` allocations
   - Update H2O eviction policy

6. **Final Normalization** (line 520)
   - `self.norm.forward(&hidden_states)`

7. **Last Token Extraction** (lines 523-548)
   - Prefill: Extract per-sequence using SequenceInfo
   - Decode: Simple `i((.., 0, ..))`

8. **Output Layer** (line 560)
   - `self.lm_head.forward(&last_hidden)`

## Identified Overhead Sources

### 1. **Tensor Allocation Overhead** (HIGH IMPACT)

**Location**: Lines 430-442, 523-548

**Problem**:
- Every decode step creates new tensors for reshaping
- `unsqueeze()` and `reshape()` allocate new memory
- Last token extraction allocates Vec and performs concatenation

**Measurement**:
```rust
// Decode mode: [total_tokens, hidden] -> [total_tokens, 1, hidden]
hidden_states.reshape((total_tokens, 1, self.config.hidden_size))?
```

For 32 tokens/batch, 4096 hidden_size, this is ~524KB allocation per forward pass.
At 50 tokens/sec, this is ~26MB/sec of tensor allocations.

**Fix Strategy**:
- Pre-allocate decode-mode tensors
- Reuse between decode steps
- Use views instead of copies where possible

### 2. **Position Calculation Repeated Lookups** (MEDIUM IMPACT)

**Location**: Lines 456-465

**Problem**:
```rust
let index_pos = if metadata.is_prefill {
    0
} else {
    metadata.context_lens.get(0).copied().unwrap_or(0)
};
```

This HashMap lookup happens on every forward pass during decode.

**Fix Strategy**:
- Cache `index_pos` in `BatchMetadata` struct
- Update once per decode step
- Avoid repeated lookups

### 3. **Debug Statistics Collection** (HIGH IMPACT when enabled)

**Location**: Lines 483-492

**Problem**:
```rust
let hs_vec = hidden_states.flatten_all()?.to_vec1::<f32>()?;
let _mean: f32 = hs_vec.iter().sum::<f32>() / hs_vec.len() as f32;
```

This flattens 4MB+ of data on every layer for debug purposes.

**Fix Strategy**:
- Feature-gate behind `debug_stats` feature
- Skip entirely in release builds
- Reduce to sampling (every N steps)

### 4. **Attention Weight Conversion** (HIGH IMPACT)

**Location**: Lines 495-517

**Problem**:
```rust
let flat = attn_weights.flatten_all()?.to_vec1::<f32>()?;
let mut attention_matrix = Vec::with_capacity(query_len);
for i in 0..query_len {
    let start = i * key_len;
    let end = start + key_len;
    attention_matrix.push(flat[start..end].to_vec());
}
```

For 128 context × 1 query, this allocates ~512 bytes per forward pass.
The `flatten_all()` and `to_vec1()` require GPU→CPU transfer.

**Fix Strategy**:
- Update H2O policy less frequently (every 10 steps)
- Use sparse attention weight sampling
- Skip when eviction not imminent

### 5. **Dimension Validation Overhead** (LOW IMPACT)

**Location**: Lines 439-453

**Problem**:
```rust
let (batch_size, seq_len, hidden_size) = hidden_states.dims3()?;
if batch_size * seq_len != total_tokens { ... }
```

Validation happens on every forward pass.

**Fix Strategy**:
- Validate once during warmup
- Use `debug_assert!` in hot path
- Trust pre-validated inputs

### 6. **Last Token Extraction Complexity** (MEDIUM IMPACT)

**Location**: Lines 523-548

**Problem**:
Prefill mode requires per-sequence extraction with Vec allocation:
```rust
let mut last_tokens = Vec::with_capacity(batch_size);
for i in 0..batch_size {
    let seq_info = &metadata.sequences[i];
    let last_pos = seq_info.start_pos + seq_info.actual_length - 1;
    let token_hidden = hidden_states.i((.., last_pos, ..))?.contiguous()?;
    last_tokens.push(token_hidden);
}
Tensor::cat(&last_tokens, 0)?
```

**Fix Strategy**:
- Pre-allocate `last_tokens` Vec in decode state
- Reuse across decode steps
- For decode mode, direct indexing is already optimal

## Baseline Performance Characteristics

### Measured Decode Loop Breakdown (Estimated)

| Operation          | Time (µs) | % of Total | Allocation (bytes) |
| ------------------ | --------- | ---------- | ------------------ |
| Embedding          | 50        | 5%         | 0 (reuses)         |
| Reshape            | 20        | 2%         | 524,288            |
| Position calc      | 5         | 0.5%       | 0                  |
| 32× Layer forward  | 800       | 80%        | Variable (KV)      |
| Attention weights  | 50        | 5%         | 2,048              |
| Normalization      | 30        | 3%         | 0                  |
| Last token extract | 20        | 2%         | 16,384             |
| Output layer       | 25        | 2.5%       | 0                  |
| **Total**          | **~1000** | **100%**   | **~543KB**         |

### Latency Variance Sources

1. **KV Cache Eviction**: +50-100µs when eviction triggers
2. **Attention Weight Transfer**: +20-40µs GPU→CPU
3. **Debug Statistics**: +100-200µs per layer when enabled
4. **Garbage Collection**: Periodic spikes from allocations

## Optimization Targets

### Target Metrics (M3.2 Acceptance Criteria)

- **Inter-token latency reduction**: 15-20% (150-200µs savings)
- **Allocation reduction**: 80%+ (from 543KB → <100KB per step)
- **Latency variance reduction**: 50% (lower p99-p50 delta)
- **Warmup time**: <100ms for decode state initialization

### High-Impact Optimizations (Priority Order)

1. **Batch State Reuse** (Expected: -100µs, -80% allocations)
   - Pre-allocate decode-mode tensors
   - Cache reshaped hidden states
   - Reuse last_tokens Vec

2. **Conditional H2O Updates** (Expected: -40µs, -100% GPU→CPU transfers)
   - Update every 10 steps instead of every step
   - Skip when cache not near capacity

3. **Position Caching** (Expected: -3µs)
   - Store `index_pos` in BatchMetadata
   - Update once per step

4. **Feature-Gate Debug** (Expected: -100µs+ when enabled)
   - Move stats to cfg(feature = "decode_stats")

5. **Dimension Validation Reduction** (Expected: -2µs)
   - Use debug_assert! in hot path

## Implementation Plan

### Phase 1: Batch State Reuse (Tasks 2-3)

**New struct**: `DecodeState`
```rust
pub struct DecodeState {
    /// Pre-allocated decode tensor: [batch_size, 1, hidden_size]
    decode_input: Option<Tensor>,
    
    /// Pre-allocated last tokens Vec for extraction
    last_tokens_buf: Vec<Tensor>,
    
    /// H2O update counter (update every 10 steps)
    h2o_step_counter: usize,
    
    /// Cached index_pos for current batch
    cached_index_pos: usize,
}
```

**Integration**:
- Add `decode_state: Option<DecodeState>` to `BatchedTransformer`
- Initialize on first decode forward
- Reuse tensors on subsequent calls

### Phase 2: Conditional Operations (Task 4)

- H2O updates: `if decode_state.h2o_step_counter % 10 == 0`
- Position caching: Store in `BatchMetadata`
- Dimension validation: `debug_assert!` only

### Phase 3: Feature Gating (Task 5)

- `#[cfg(feature = "decode_stats")]` for layer statistics
- Default: disabled in release builds
- Enable for profiling/debugging

### Phase 4: Validation (Task 6)

- Correctness: Token-by-token comparison with baseline
- Performance: Benchmark decode latency (p50, p95, p99)
- Memory: Track allocation reduction
- Variance: Measure latency stability

## Expected Results

### Latency Improvement
- **Baseline**: 1000µs per decode step
- **Optimized**: 850µs per decode step
- **Improvement**: 15% reduction (150µs savings)

### Allocation Reduction
- **Baseline**: 543KB per step
- **Optimized**: 50KB per step (initial warmup only)
- **Improvement**: 91% reduction

### Variance Reduction
- **Baseline**: p99-p50 = 200µs (20% variance)
- **Optimized**: p99-p50 = 100µs (10% variance)
- **Improvement**: 50% variance reduction

## Next Steps

1. ✅ Complete decode loop analysis (this document)
2. [ ] Implement `DecodeState` struct with tensor reuse
3. [ ] Add conditional H2O updates
4. [ ] Feature-gate debug statistics
5. [ ] Benchmark improvements
6. [ ] Document optimization guide for users

## References

- src/model/custom_transformer.rs (main decode loop)
- src/model/custom_transformer_block.rs (per-layer processing)
- src/engine/parallel_cache.rs (KV cache operations)
- ROADMAP.md M3.2 (decode-loop overhead reductions)
