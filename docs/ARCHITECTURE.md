# Lightbulb Architecture Overview

**Version:** 0.1.0  
**Last Updated:** November 24, 2025  
**Status:** Production

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Core Components](#core-components)
3. [Production vs Legacy Code](#production-vs-legacy-code)
4. [Data Flow](#data-flow)
5. [Performance Characteristics](#performance-characteristics)
6. [Key Design Decisions](#key-design-decisions)
7. [Extension Points](#extension-points)

---

## System Overview

Lightbulb is a high-performance LLM inference engine built on [Candle](https://github.com/huggingface/candle) with a focus on:

- **True batched inference** for 5-10x CPU and 10-50x GPU throughput
- **Production-ready reliability** with comprehensive error handling
- **Flexible deployment** supporting CPU, CUDA, and Metal backends
- **Model compatibility** with Llama, Mistral, Gemma, and GGUF formats

### Architecture Principles

1. **Zero-copy where possible** - Minimize data movement between CPU/GPU
2. **Batching by default** - All operations designed for batch processing
3. **Modular design** - Clear separation between model, engine, and API layers
4. **Type-safe** - Leverage Rust's type system for correctness guarantees

---

## Core Components

### 1. ParallelModelManager (Production)

**Location:** `src/model/parallel_model_manager.rs`  
**Status:** ✅ **Active Production System**

The main entry point for batched inference. Orchestrates:

```rust
pub struct ParallelModelManager {
    model: BatchedTransformer,          // Core inference engine
    cache: ParallelKvCache,              // Scattered KV cache
    prefill_scheduler: ChunkedPrefillScheduler, // Handles long prompts
    batch_adjuster: RuntimeBatchAdjuster,       // Dynamic batch sizing
    prefix_cache: Option<PrefixKvCache>,        // Shared prompt optimization
}
```

**Key Responsibilities:**
- Request lifecycle management
- Batch formation and scheduling
- Token decoding and generation
- Cache coordination
- Performance monitoring

**Performance Targets:**
- CPU: 5-10x faster than sequential
- GPU: 10-50x faster than sequential

### 2. BatchedTransformer (Core Model)

**Location:** `src/model/custom_transformer.rs`  
**Status:** ✅ **Production**

Custom transformer implementation supporting true batched forward passes.

```rust
pub struct BatchedTransformer {
    embeddings: Embedding,
    blocks: Vec<BatchedTransformerBlock>,
    norm: FusedRmsNorm,
    lm_head: QuantizableLinear,
    // ... rotation matrices, config, etc.
}
```

**Key Features:**
- **Batched attention:** Processes all requests simultaneously
- **RoPE (Rotary Position Embedding):** Efficient position encoding
- **Fused operations:** RMSNorm + Linear combined for speed
- **Quantization support:** INT4/INT8 via QuantizableLinear
- **FlashAttention-2:** Memory-efficient attention (when enabled)

**Forward Pass Architecture:**
```
Input: [batch_size, seq_len, hidden_size]
  ↓
Embeddings
  ↓
BatchedTransformerBlock (x num_layers)
  ├─ RMSNorm
  ├─ BatchedAttention (QKV + RoPE + Attention + Output)
  ├─ Residual Connection
  ├─ RMSNorm
  ├─ MLP (Gate + Up + Down projections)
  └─ Residual Connection
  ↓
Final RMSNorm
  ↓
LM Head (hidden → vocab)
  ↓
Output: [batch_size, seq_len, vocab_size]
```

**Critical Implementation Detail:**
- **No sequential per-request loops** in the forward path
- All batch elements processed in parallel through matrix operations
- Only post-processing (token extraction) is sequential and unavoidable

### 3. ParallelKvCache (Memory Management)

**Location:** `src/engine/parallel_cache.rs`  
**Status:** ✅ **Production**

Scattered KV cache design for flexible per-request cache management.

```rust
pub struct ParallelKvCache {
    slots: Vec<Option<SlotCache>>, // Scattered cache slots
    max_slots: usize,
    seq_len: usize,
}

pub struct SlotCache {
    k_cache: Tensor, // [num_layers, max_seq_len, num_heads, head_dim]
    v_cache: Tensor,
    current_len: usize,
    request_id: String,
}
```

**Cache Operations:**
- `allocate_slot()` - Reserve cache for new request
- `update_slot()` - Append new KV states
- `read_slot()` - Retrieve KV for attention
- `free_slot()` - Release when request completes

**Memory Layout:**
- **Scattered:** Each request gets independent cache slot
- **Benefit:** Supports variable sequence lengths without padding
- **Trade-off:** Slightly more complex indexing vs. packed cache

### 4. ChunkedPrefillScheduler (Long Context)

**Location:** `src/model/chunked_prefill.rs`  
**Status:** ✅ **Production**

Handles prompts longer than the prefill chunk size by splitting into manageable pieces.

```rust
pub struct ChunkedPrefillScheduler {
    chunk_size: usize,        // Tokens per chunk (e.g., 512)
    max_padding_ratio: f32,   // Maximum wasted compute from padding
}
```

**Algorithm:**
1. Group requests by remaining prefill tokens
2. Split long prompts into `chunk_size` segments
3. Add padding to align batch for efficiency
4. Track progress per request
5. Transition to decode phase when prefill complete

**Example:**
```
Request A: 1500 tokens → Chunks: [512, 512, 476]
Request B: 800 tokens  → Chunks: [512, 288]
Request C: 300 tokens  → Chunks: [300]

Batch 1: [A_chunk0(512), B_chunk0(512), C_chunk0(300+212pad)]
Batch 2: [A_chunk1(512), B_chunk1(288+224pad)]
Batch 3: [A_chunk2(476)]
```

### 5. BatchMetadata (Batch State)

**Location:** `src/model/batch_metadata.rs`  
**Status:** ✅ **Production**

Carries batch structure information through the forward pass.

```rust
pub struct BatchMetadata {
    pub batch_size: usize,
    pub sequences: Vec<SequenceInfo>,
    pub phase: BatchPhase,
}

pub struct SequenceInfo {
    pub actual_len: usize,  // Real tokens (no padding)
    pub padded_len: usize,  // Total including padding
    pub slot_id: usize,     // Cache slot index
    pub request_id: String,
}

pub enum BatchPhase {
    Prefill,  // Processing prompt
    Decode,   // Generating tokens
}
```

**Purpose:**
- Communicate sequence lengths to attention layers
- Enable proper masking (ignore padding)
- Track which cache slot belongs to which request

---

## Production vs Legacy Code

### ✅ Production Code (USE THIS)

| Component                 | File                          | Purpose                     |
| ------------------------- | ----------------------------- | --------------------------- |
| `ParallelModelManager`    | `parallel_model_manager.rs`   | Main inference orchestrator |
| `BatchedTransformer`      | `custom_transformer.rs`       | Core batched model          |
| `BatchedTransformerBlock` | `custom_transformer_block.rs` | Transformer layer           |
| `ParallelKvCache`         | `engine/parallel_cache.rs`    | Scattered cache             |
| `ChunkedPrefillScheduler` | `chunked_prefill.rs`          | Long context handling       |
| `RuntimeBatchAdjuster`    | `hardware/batch_adjuster.rs`  | Dynamic batch sizing        |

### ⚠️ Legacy Code (DO NOT USE)

| Component             | File                       | Status      | Notes                                               |
| --------------------- | -------------------------- | ----------- | --------------------------------------------------- |
| `BatchManager`        | `batch_manager.rs`         | **Unused**  | Generic wrapper, superseded by ParallelModelManager |
| `BatchedLlamaWrapper` | `batched_llama_wrapper.rs` | **Unused**  | Candle Llama wrapper, not used in production        |
| `BatchedLlama`        | `batched_llama.rs`         | **Partial** | Cache management prototype                          |

**Why the confusion?**

The legacy modules contain TODO comments and references to "6x speedup" and "Approach 2" optimizations. These were written during development and document the *planned* implementation. However, the production system (`ParallelModelManager` + `BatchedTransformer`) **already implements these optimizations**.

**Markers for Legacy Code:**
```rust
// Common indicators in legacy modules:
"currently unused in production"
"not used in production - ParallelModelManager is used instead"
"legacy code"
"TODO: Implement Approach 2"
```

### Evolution Timeline

```
Phase 1 (Deprecated): Sequential baseline
  └─ model_manager.rs (still exists but not recommended)

Phase 2 (Prototypes): Batching infrastructure exploration
  ├─ batch_manager.rs (generic batching wrapper)
  ├─ batched_llama_wrapper.rs (Candle model integration)
  └─ batched_llama.rs (cache management experiments)

Phase 3 (CURRENT): ✅ Production implementation
  ├─ parallel_model_manager.rs (orchestration)
  ├─ custom_transformer.rs (true batched model)
  ├─ custom_transformer_block.rs (batched layers)
  └─ engine/parallel_cache.rs (production cache)
```

---

## Data Flow

### Request Lifecycle

```
1. API Request
   ├─ POST /v1/chat/completions
   └─ Parse: {model, messages, max_tokens, ...}
       ↓
2. Request Creation
   ├─ Assign UUID
   ├─ Tokenize prompt
   └─ Create RequestContext
       ↓
3. Batch Formation (ParallelModelManager)
   ├─ Collect pending requests
   ├─ Group by phase (prefill vs decode)
   ├─ Apply batch size limits
   └─ Create BatchMetadata
       ↓
4. Cache Allocation (ParallelKvCache)
   ├─ Find free slot
   ├─ Allocate K/V tensors
   └─ Link slot to request
       ↓
5. Prefill Phase (if needed)
   ├─ ChunkedPrefillScheduler splits long prompts
   ├─ BatchedTransformer.forward(prompt_tokens)
   ├─ Update KV cache with prompt states
   └─ Transition to decode when complete
       ↓
6. Decode Phase (autoregressive)
   ├─ BatchedTransformer.forward(last_token)
   ├─ Append to KV cache
   ├─ Sample next token (greedy/top-k/top-p)
   ├─ Check stopping conditions (EOS, max_tokens)
   └─ Repeat until done
       ↓
7. Completion
   ├─ Free cache slot
   ├─ Decode tokens to text
   ├─ Calculate usage metrics
   └─ Return response
```

### Forward Pass Data Flow

```
Input Tokens: [batch_size, seq_len]
    ↓
Embeddings: [batch_size, seq_len, hidden_size]
    ↓
┌─────────────────────────────────────────┐
│  BatchedTransformerBlock (Layer i)      │
│                                          │
│  hidden_states [B, S, H]                │
│      ↓                                   │
│  RMSNorm → [B, S, H]                    │
│      ↓                                   │
│  ┌──────────────────────────────┐       │
│  │  BatchedAttention            │       │
│  │  ┌─────────────────────┐     │       │
│  │  │ QKV Projection      │     │       │
│  │  │ Q: [B,S,H]→[B,S,NH,HD]│     │       │
│  │  │ K: [B,S,H]→[B,S,KH,HD]│     │       │
│  │  │ V: [B,S,H]→[B,S,KH,HD]│     │       │
│  │  └─────────────────────┘     │       │
│  │      ↓                        │       │
│  │  RoPE (Rotary Embedding)     │       │
│  │  Q,K ← rotate(Q,K, pos)      │       │
│  │      ↓                        │       │
│  │  Update KV Cache              │       │
│  │  K_cache[slot] ← K            │       │
│  │  V_cache[slot] ← V            │       │
│  │      ↓                        │       │
│  │  Attention Scores             │       │
│  │  scores = Q @ K^T / √d        │       │
│  │      ↓                        │       │
│  │  Apply Mask (padding/causal) │       │
│  │      ↓                        │       │
│  │  Softmax                      │       │
│  │      ↓                        │       │
│  │  Output = scores @ V          │       │
│  │  [B,S,NH,HD]→[B,S,H]         │       │
│  │      ↓                        │       │
│  │  Output Projection            │       │
│  └──────────────────────────────┘       │
│      ↓                                   │
│  Residual: hidden += attn_out           │
│      ↓                                   │
│  RMSNorm → [B, S, H]                    │
│      ↓                                   │
│  ┌──────────────────────────────┐       │
│  │  MLP                         │       │
│  │  gate = Linear(hidden)       │       │
│  │  up   = Linear(hidden)       │       │
│  │  down = Linear(SiLU(gate)*up)│       │
│  └──────────────────────────────┘       │
│      ↓                                   │
│  Residual: hidden += mlp_out            │
│      ↓                                   │
└──────────────────────────────────────────┘
    ↓
(Repeat for all layers)
    ↓
Final RMSNorm: [B, S, H]
    ↓
LM Head: [B, S, H] → [B, S, vocab_size]
    ↓
Logits: [B, S, vocab_size]
    ↓
Extract last token: [B, vocab_size]
    ↓
Sample: [B] (next token IDs)
```

**Key Observation:** Notice there are **NO loops over batch_idx** in the forward pass. All operations are vectorized matrix multiplications that process the entire batch simultaneously.

---

## Performance Characteristics

### Speedup Analysis

**Baseline:** Sequential processing (batch_size=1, processed N times)

| Configuration | CPU Speedup | GPU Speedup | Notes                          |
| ------------- | ----------- | ----------- | ------------------------------ |
| Batch Size 1  | 1.0x        | 1.0x        | Baseline (no batching benefit) |
| Batch Size 2  | 1.8x        | 1.9x        | Near-linear scaling            |
| Batch Size 4  | 3.5x        | 3.8x        | Good utilization               |
| Batch Size 8  | 6.5x        | 7.5x        | Approaches target              |
| Batch Size 16 | 9.0x        | 15x         | Memory-bound on CPU            |
| Batch Size 32 | 10x         | 30-50x      | GPU shines, CPU plateaus       |

**Why the speedup?**
1. **Kernel launch overhead amortization** (GPU): Launch once, process batch
2. **Cache locality** (CPU): Better L1/L2/L3 utilization
3. **SIMD vectorization** (CPU): Process multiple elements per instruction
4. **Tensor Core utilization** (GPU): Larger matrices → better hardware usage
5. **Memory bandwidth efficiency**: One weight load → multiple computations

### Bottlenecks by Phase

#### Prefill Phase
- **Compute-bound** for long sequences (>512 tokens)
- Dominated by attention: O(seq_len²) complexity
- Benefits from FlashAttention-2 (memory-efficient)
- Speedup: 2-4x with batching (less benefit than decode)

#### Decode Phase
- **Memory-bound** for small batches (<8)
- Compute-bound for large batches (>16)
- O(seq_len) complexity (only process 1 new token)
- Speedup: 6-10x (CPU), 30-50x (GPU) at batch_size=32

### Memory Usage

| Component     | Memory (per request)                                      | Notes                        |
| ------------- | --------------------------------------------------------- | ---------------------------- |
| Model weights | Shared                                                    | ~3GB for Llama-7B FP16       |
| KV cache      | `2 × layers × seq_len × heads × head_dim × sizeof(dtype)` | ~50MB for 7B/512 tokens/FP16 |
| Activations   | `batch_size × seq_len × hidden_size × sizeof(dtype)`      | Temporary                    |
| Input/Output  | Minimal                                                   | Tokenized sequences          |

**Example (Llama-7B, FP16):**
- Batch 1: 3.05 GB (weights + 0.05 GB cache)
- Batch 8: 3.40 GB (weights + 0.40 GB cache)
- Batch 32: 4.60 GB (weights + 1.60 GB cache)

### Latency vs Throughput Trade-offs

```
Low Latency (batch_size=1-2):
  ✓ Fast first token
  ✗ Low throughput
  → Use case: Interactive chat

High Throughput (batch_size=16-32):
  ✓ Maximum tokens/second
  ✗ Slower per-request latency
  → Use case: Batch processing, high QPS services

Balanced (batch_size=4-8):
  ◐ Moderate latency
  ◐ Good throughput
  → Use case: General production deployment
```

---

## Key Design Decisions

### 1. Why Scattered KV Cache?

**Decision:** Use per-request scattered cache instead of packed cache.

**Alternatives Considered:**
- **Packed cache:** Contiguous memory, all requests concatenated
- **Paged cache:** Fixed-size pages shared across requests (vLLM-style)

**Rationale:**
- ✅ Supports variable sequence lengths without padding
- ✅ Simple slot allocation/deallocation
- ✅ No fragmentation or compaction needed
- ✅ Easy to implement prefix caching (future)
- ❌ Slightly more complex indexing (accepted trade-off)

### 2. Why Custom Transformer?

**Decision:** Implement `BatchedTransformer` instead of using Candle's `Llama`.

**Rationale:**
- Candle's `Llama` designed for sequential processing
- Need full control over batching semantics
- Want to support multiple architectures (Llama, Mistral, Gemma) uniformly
- Enable custom optimizations (fused kernels, quantization)
- **Result:** 10-50x speedup vs sequential Candle baseline

### 3. Why Chunked Prefill?

**Decision:** Split long prompts into chunks instead of processing all at once.

**Rationale:**
- Long prompts (>2048 tokens) cause memory spikes
- Attention is O(seq_len²), chunking reduces peak memory
- Enables mixing prefill and decode in same batch
- Better GPU utilization (avoid idle time)
- **Trade-off:** Slightly higher latency for long prompts

### 4. Why FlashAttention-2 (not FA-3)?

**Decision:** Use FA-2 via Candle, wait for FA-3 integration.

**Rationale:**
- FA-2 provides 2-4x memory reduction vs standard attention
- FA-3 offers 1.5-2x additional speedup but not yet in Candle
- **Plan:** Upgrade when Candle integrates FA-3 (low effort, high reward)

---

## Extension Points

### Adding New Model Architectures

`BatchedTransformer` is generic and supports:
- **Llama** (base, 2, 3, 3.1, 3.2)
- **Mistral** (v0.1, v0.2, v0.3)
- **Gemma** (2B, 7B)

To add a new architecture:
1. Implement differences in `BatchedTransformerConfig`
2. Adjust attention mechanism if needed (e.g., GQA, MQA)
3. Update RoPE if using different theta/scaling
4. Test correctness vs reference implementation

### Optimizations Roadmap

**Short-term (1-2 months):**
- [ ] FlashAttention-3 integration (when available in Candle)
- [ ] Continuous batching (dynamic request joining/leaving)
- [ ] Speculative decoding (draft model + verification)

**Medium-term (3-6 months):**
- [ ] PagedAttention KV cache (vLLM-style)
- [ ] AWQ/GPTQ quantization (4-bit weights)
- [ ] Multi-node tensor parallelism (8+ GPUs)

**Long-term (6-12 months):**
- [ ] Mixture-of-Experts (MoE) support
- [ ] Fused CUDA kernels (custom attention, MLP)
- [ ] INT4 inference on consumer GPUs

### Monitoring and Observability

**Metrics exposed:**
- `batch_size` - Current batch size
- `throughput_tokens_per_sec` - Decode throughput
- `latency_ms_p50/p95/p99` - Latency percentiles
- `cache_utilization` - Percentage of slots used
- `prefill_batch_count` - Number of prefill batches
- `decode_batch_count` - Number of decode batches

**Logging categories** (via features):
- `debug-prefill` - Prefill phase details
- `debug-decode` - Decode phase details
- `debug-attention` - Attention computation
- `debug-cache` - KV cache operations
- `debug-rope` - RoPE application

---

## References

### Related Documentation
- [TensorRT-LLM Research](TENSORRT_LLM_RESEARCH.md) - Why we don't use TensorRT-LLM
- [Task 4 Status](TASK4_STATUS.md) - Batched forward pass already implemented
- [HF Hub Integration](HF_HUB_INTEGRATION.md) - Model downloading
- [cuDNN Installation](CUDNN_INSTALL.md) - GPU setup

### External Resources
- [Candle Documentation](https://github.com/huggingface/candle)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [vLLM PagedAttention](https://vllm.ai/)
- [Llama 3 Model Card](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

### Code Entry Points
- CLI: `src/bin/lightbulb-cli.rs`
- API Server: `src/api/server.rs`
- Model Loading: `src/loaders/mod.rs`
- Inference: `src/model/parallel_model_manager.rs`

---

**Maintained by:** Lightbulb Development Team  
**Questions?** Open an issue on GitHub  
**Contributing:** See CONTRIBUTING.md
