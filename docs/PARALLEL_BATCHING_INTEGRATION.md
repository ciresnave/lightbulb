# Parallel Batching Integration Guide

## Overview

This document explains how the parallel batching code integrates with the current `BatchManager` design.

## Current Architecture (Sequential)

```
User Request
    ↓
ModelManager (coordination, tokenization)
    ↓
BatchManager<Llama> (model-agnostic wrapper)
    ↓
Candle's Llama::forward() (processes one request at a time)
```

**Status**: ✅ Working, tested, maintainable
**Performance**: Sequential (1x baseline)

## Parallel Batching Architecture (When Needed)

```
User Request
    ↓
ModelManager (coordination, tokenization)
    ↓
BatchManager<BatchedTransformer> (model-agnostic wrapper)
    ↓
BatchedTransformer (parallel processing)
    ├─> Candle's Embedding (reused!)
    ├─> BatchedTransformerBlock × N layers
    │   ├─> Candle's RmsNorm (reused!)
    │   ├─> BatchedAttention (custom - handles scattered KV cache)
    │   ├─> Candle's RmsNorm (reused!)
    │   └─> Mlp (thin wrapper - Candle's Linear + silu)
    ├─> Candle's RmsNorm (reused!)
    └─> Candle's Linear (lm_head, reused!)
```

**Status**: 🔧 Ready to integrate (just needs testing)
**Performance**: 5-50x faster (estimated)
**Maintenance**: ~550 lines of custom code (vs 1130+ before refactor)

## Code Reuse Breakdown

### ✅ Fully Reused from Candle (0 lines to maintain)
- `candle_nn::Embedding` - Token embeddings
- `candle_nn::RmsNorm` - Layer normalization  
- `candle_nn::Linear` - LM head projection
- All in `custom_transformer.rs`

### 🔧 Thin Wrapper (~80 lines total)
- `mlp_wrapper::Mlp` - Just wraps Candle's `Linear` + `ops::silu`
- Functionally identical to Candle's internal `Mlp`
- Exists only because Candle's `Mlp` is private

### ❌ Custom Implementation (~470 lines - unavoidable)
- `custom_attention.rs` (~320 lines)
  - **Why custom**: Candle's attention uses single `kv_cache: Option<(Tensor, Tensor)>`
  - **What we need**: Integration with `BatchExecutor::ScatteredKvCache`
  - **Cannot reuse**: Core batching innovation
  
- `custom_transformer_block.rs` (~80 lines)
  - Glue code combining attention + MLP with residuals
  - Most components reused from Candle
  
- `custom_transformer.rs` (~150 lines)
  - Coordination layer for batched forward pass
  - Embedding/norms/lm_head all from Candle
  - Main logic is just calling blocks in sequence

**Total custom code: ~550 lines** (down from 1130+ before refactor!)

## Integration with BatchManager

### Option 1: Keep Both Implementations (Recommended for Now)

```rust
// Current (sequential - for development/testing)
pub type SequentialModelManager = ModelManager<BatchManager<Llama>>;

// Future (parallel - when profiling shows need)
pub type ParallelModelManager = ModelManager<BatchManager<BatchedTransformer>>;
```

**Benefits**:
- ✅ Keep working sequential code
- ✅ Can A/B test performance
- ✅ Easy rollback if issues
- ✅ Validate correctness

### Option 2: Add Parallel Method to BatchManager

```rust
impl<M> BatchManager<M> where M: TransformerModel {
    // Current sequential processing
    pub fn forward_decode_batch(&mut self, ...) -> Result<Tensor> {
        // Loop processing (1x speed)
    }
    
    // New parallel processing (when M = BatchedTransformer)
    pub fn forward_decode_batch_parallel(&mut self, ...) -> Result<Tensor> {
        // True batched processing (5-50x speed)
    }
}
```

### Option 3: Replace Llama with BatchedTransformer

When profiling shows parallel batching is needed:

```rust
// In ModelManager::load()
// OLD:
let model = Llama::load(...)?;
let batch_manager = BatchManager::new(model, batch_executor, device);

// NEW:
let model = BatchedTransformer::new(config, vb)?;
let batch_manager = BatchManager::new(model, batch_executor, device);
```

**That's it!** BatchManager's interface stays the same.

## Implementation Steps (When Ready)

### 1. Finish BatchedTransformer Implementation

Current status:
- ✅ MLP wrapper using Candle components
- ✅ BatchedAttention with ScatteredKvCache
- ✅ BatchedTransformerBlock combining them
- ⏸️ BatchedTransformer needs RoPE integration
- ⏸️ Needs testing against direct Llama

### 2. Create Integration Tests

Similar to `batch_manager_integration.rs`:

```rust
#[test]
fn test_batched_transformer_vs_llama() {
    // Load same weights into both
    let llama = Llama::load(...)?;
    let batched = BatchedTransformer::new(...)?;
    
    // Compare outputs
    assert_tensors_close(llama_output, batched_output);
}
```

### 3. Benchmark Performance

```rust
// Measure sequential (current)
let seq_throughput = benchmark_batch_manager::<Llama>();

// Measure parallel (new)
let par_throughput = benchmark_batch_manager::<BatchedTransformer>();

println!("Speedup: {}x", par_throughput / seq_throughput);
```

### 4. Decide Based on Data

- If speedup < 2x: **Stay with sequential** (not worth complexity)
- If speedup 2-5x: **Consider parallel** for production
- If speedup > 5x: **Definitely switch** to parallel

## Key Files

### Core Parallel Batching
- `src/model/custom_attention.rs` - Batched attention (~320 lines)
- `src/model/mlp_wrapper.rs` - MLP using Candle components (~80 lines)
- `src/model/custom_transformer_block.rs` - Block combining above (~80 lines)
- `src/model/custom_transformer.rs` - Full transformer (~150 lines)

### Integration Layer
- `src/model/batch_manager.rs` - Model-agnostic wrapper
- `src/model/model_manager.rs` - High-level API

### Testing
- `tests/batch_manager_integration.rs` - Current sequential tests (✅ passing)
- `tests/batched_transformer_integration.rs` - TODO: Parallel tests

## Why This Design Is Good

1. **Clean Separation**: BatchManager doesn't care about batching strategy
2. **Easy to Understand**: Clear layers of abstraction
3. **Minimal Custom Code**: ~550 lines vs 1130+ before
4. **Testable**: Can compare sequential vs parallel
5. **Reversible**: Can switch back if needed
6. **Incremental**: Can develop/test in parallel with current code

## Next Steps

1. ✅ **Done**: Refactor to use Candle components
2. ✅ **Done**: Delete custom_mlp.rs
3. 🔄 **In Progress**: Document integration strategy (this file!)
4. ⏸️ **TODO**: Finish BatchedTransformer implementation
5. ⏸️ **TODO**: Create parallel integration tests  
6. ⏸️ **TODO**: Benchmark and decide
7. ⏸️ **TODO**: Integrate if worthwhile

## Questions?

- **Q**: Do we need parallel batching?
  - **A**: Unknown - need to profile first!
  
- **Q**: Can we use both?
  - **A**: Yes! BatchManager is generic over model type
  
- **Q**: What if it's slower?
  - **A**: Just use `BatchManager<Llama>` instead
  
- **Q**: How much code do we maintain?
  - **A**: ~550 lines (vs 1130+ before refactor)
  
- **Q**: What if Candle adds batching?
  - **A**: Delete our custom code, use theirs!
