# Batching Implementation Quick Reference

**For**: Phase 2D True Batched Inference  
**Goal**: Achieve 6.00x speedup (0.54 → 3.24 tokens/sec)

---

## The Problem

**Current (Sequential)**:
```rust
// Processes ONE request at a time
for req in batch {
    let logits = model.forward(&tokens, pos, &mut cache)?; // ❌ Sequential
}
```

**Why it's slow**:
- 10 requests × 185ms each = 1850ms total
- Throughput: 0.54 tokens/sec

---

## The Solution

**Batched Execution**:
```rust
// Process ENTIRE batch at once
let logits = model.forward_batched(&all_tokens, &all_positions, &cache, &metadata)?;
// ✅ Batched: ~308ms for 10 requests
```

**Why it's fast**:
- 1 batch × 308ms = 308ms total
- Throughput: 3.24 tokens/sec
- **6.00x speedup** ✅

---

## Key Code Pattern (from candle-vllm)

### **Decode Batch** (most common - 90% of operations):

```rust
// 1. Prepare batched tensors
let tokens = Tensor::new(
    [42, 7, 13],  // Next token for each sequence
    &device
)?.reshape((3, 1))?;  // [batch_size=3, seq_len=1]

let positions = Tensor::new(
    [5, 2, 3],  // Position in each sequence
    &device
)?.reshape((3, 1))?;  // [batch_size=3, seq_len=1]

// 2. Create metadata
let metadata = BatchMetadata {
    is_prefill: false,
    batch_size: 3,
    request_ids: [id1, id2, id3],
    slot_offsets: [5, 2, 3],  // Where each seq is in cache
    sequence_lengths: [1, 1, 1],  // 1 token each
};

// 3. Forward pass
let logits = model.forward_batched(
    &tokens,      // [3, 1]
    &positions,   // [3, 1]
    &kv_cache,    // Shared across all requests
    &metadata,    // Batch structure
)?;  // Returns: [3, 1, vocab_size]

// 4. Extract results
let logits = logits.squeeze(1)?;  // [3, vocab_size]
// logits[0] = next token probs for request 1
// logits[1] = next token probs for request 2
// logits[2] = next token probs for request 3
```

### **Prefill Batch** (variable-length prompts):

```rust
// Sequences:
// Seq1: [1, 2, 3, 4, 5]  (len=5)
// Seq2: [6, 7]           (len=2)
// Seq3: [8, 9, 10]       (len=3)

// 1. Concatenate
let tokens = Tensor::new(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  // Concatenated
    &device
)?;  // [total_tokens=10]

let positions = Tensor::new(
    [0, 1, 2, 3, 4, 0, 1, 0, 1, 2],   // Position in each seq
    &device
)?;  // [total_tokens=10]

// 2. Metadata describes structure
let metadata = BatchMetadata {
    is_prefill: true,
    batch_size: 3,
    request_ids: [id1, id2, id3],
    cu_seqlens: Some(vec![0, 5, 7, 10]),  // Cumulative lengths
    sequence_lengths: vec![5, 2, 3],
    max_seqlen: Some(5),
};

// 3. Forward pass
let logits = model.forward_batched(
    &tokens,      // [10]
    &positions,   // [10]
    &kv_cache,
    &metadata,
)?;  // Returns: [10, vocab_size]

// 4. Extract last token logits for each sequence
let results = vec![
    logits.get(4)?,   // Seq1: index 4 (last of first 5)
    logits.get(6)?,   // Seq2: index 6 (last of next 2)
    logits.get(9)?,   // Seq3: index 9 (last of final 3)
];
```

---

## Critical Signature Change

### **Old (Standard Candle)**:
```rust
fn forward(
    &mut self,           // ❌ Mutable
    x: &Tensor,          // [seq_len] - single sequence
    index_pos: usize,    // ❌ Single position
    cache: &mut Cache    // ❌ Exclusive ownership
) -> Result<Tensor>
```

### **New (Batched)**:
```rust
fn forward_batched(
    &self,               // ✅ Immutable
    tokens: &Tensor,     // [batch_size, seq_len] - batched
    positions: &Tensor,  // [batch_size, seq_len] - batched
    cache: &KvCache,     // ✅ Shared, immutable
    metadata: &BatchMetadata,  // ✅ Describes batch
) -> Result<Tensor>      // [batch_size, seq_len, vocab_size]
```

---

## Implementation Checklist

### Phase 1: Data Structures (1 day)

**File**: `src/model/batch_metadata.rs` (NEW)
```rust
pub struct BatchMetadata {
    pub is_prefill: bool,
    pub batch_size: usize,
    pub request_ids: Vec<RequestId>,
    pub slot_offsets: Vec<usize>,
    pub sequence_lengths: Vec<usize>,
    pub cu_seqlens: Option<Vec<usize>>,  // For prefill
    pub max_seqlen: Option<usize>,
}
```

### Phase 2: Batched Layers (1-2 days)

**File**: `src/model/batched_llama.rs` (NEW)
```rust
pub struct BatchedLlama {
    embedding: Embedding,
    layers: Vec<TransformerBlock>,  // Direct access
    norm: RmsNorm,
    lm_head: Linear,
}

impl BatchedLlama {
    pub fn forward_batched(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        kv_cache: &ScatteredKvCache,
        metadata: &BatchMetadata,
    ) -> Result<Tensor> {
        // Implement batched forward
    }
}
```

**File**: `src/model/batched_attention.rs` (NEW)
```rust
pub struct BatchedAttention {
    // Attention layer with batched operations
    
    fn forward_batched(
        &self,
        hidden: &Tensor,
        positions: &Tensor,
        kv_cache: &ScatteredKvCache,
        layer_idx: usize,
        metadata: &BatchMetadata,
    ) -> Result<Tensor> {
        // Implement batched attention
    }
}
```

### Phase 3: Integration (0.5 day)

**File**: `src/model.rs` (MODIFY)
```rust
impl Model {
    pub fn forward_batch(&mut self, batch: &[Arc<GenerationRequest>]) 
        -> Result<Vec<ForwardResult>> {
        
        let (prefill, decode) = Self::partition_batch(batch);
        
        let mut results = Vec::new();
        
        if !prefill.is_empty() {
            results.extend(self.forward_prefill_batch(&prefill)?);
        }
        
        if !decode.is_empty() {
            results.extend(self.forward_decode_batch(&decode)?);
        }
        
        Ok(results)
    }
}
```

### Phase 4: Testing (0.5 day)

1. **Correctness**: Compare batched vs sequential outputs
2. **Performance**: Measure actual speedup
3. **Integration**: Update BatchExecutor

---

## Key Differences from Standard Candle

| Aspect           | Standard Candle     | Batched (candle-vllm)   |
| ---------------- | ------------------- | ----------------------- |
| **Ownership**    | `&mut cache`        | `&cache` (shared)       |
| **Position**     | `usize`             | `Tensor` (batched)      |
| **Input shape**  | `[seq_len]`         | `[batch_size, seq_len]` |
| **KV updates**   | Direct writes       | Slot-mapped writes      |
| **Layer access** | Via model.forward() | Direct layer calls      |
| **Metadata**     | None                | BatchMetadata           |

---

## Batch Size Calculation

**Decode batch** (easiest case):
```
Batch size = number of requests in decode phase
Each request contributes 1 token

tokens:    [batch_size, 1]
positions: [batch_size, 1]
output:    [batch_size, vocab_size]
```

**Prefill batch** (concatenated):
```
Batch size = number of requests in prefill phase
Total tokens = sum of prompt lengths

tokens:    [total_tokens]
positions: [total_tokens]
output:    [total_tokens, vocab_size]

Then extract last token for each sequence using cu_seqlens
```

---

## Expected Metrics

**Before** (Sequential):
- Throughput: 0.54 tokens/sec
- Batch time: 1850ms (10 requests)
- Per-request: 185ms

**After** (Batched):
- Throughput: 3.24 tokens/sec
- Batch time: 308ms (10 requests)
- Per-request: 30.8ms (amortized)

**Speedup**: **6.00x** ✅

---

## Common Pitfalls

❌ **Don't**:
- Use `&mut cache` (prevents sharing)
- Call `model.forward()` in a loop
- Process prefill and decode in same batch
- Forget to update slot_offsets after each step

✅ **Do**:
- Use shared immutable cache
- Call `model.forward_batched()` once
- Separate prefill and decode batches
- Track sequence positions accurately
- Use metadata to describe batch structure

---

## Reference Files

**Study these**:
1. `idea_sources/candle-vllm/src/openai/pipelines/llm_engine.rs:490-508`
   - How to prepare batched inputs and call forward
   
2. `idea_sources/candle-vllm/src/openai/models/llama.rs:108-149`
   - Batched model forward implementation
   
3. `idea_sources/candle-vllm/src/paged_attention/input_metadata.rs`
   - BatchMetadata structure (their InputMetadata)

**Our existing code**:
1. `src/cache/scattered_kv.rs`
   - Already supports batched access (no changes needed)
   
2. `src/model.rs:540-609`
   - Current sequential forward_batch (to be replaced)

---

## Quick Start

1. **Read** `VLLM_BATCHING_ANALYSIS.md` for full details
2. **Study** candle-vllm's llm_engine.rs:490-508
3. **Create** BatchMetadata struct
4. **Implement** BatchedLlama wrapper
5. **Test** with small batches first
6. **Measure** performance improvement

**Estimated time**: 2-3 days for full implementation

---

**Goal**: Turn 10 sequential 185ms calls into 1 batched 308ms call = **6x faster** 🚀
