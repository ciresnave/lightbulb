# Chunk Size Optimization Analysis

## Problem Statement

Multi-chunk prefill requires choosing a `chunk_size` that balances:
1. **Padding waste** (memory + compute cost of unused tokens)
2. **Transfer overhead** (kernel launch cost for multiple small batches)
3. **Latency** (time to first token)
4. **Memory capacity** (staying within VRAM/RAM limits)

## Current Implementation

Located in `src/model/chunked_prefill.rs`:

```rust
fn calculate_dynamic_chunk_size(&self, requests: &[PrefillRequest]) -> usize {
    let max_remaining = requests.iter()
        .filter(|r| r.has_more())
        .map(|r| r.remaining())
        .max()
        .unwrap_or(0);
    
    let aligned = ((max_remaining + 31) / 32) * 32;
    aligned.min(self.config.chunk_size)  // Cap at 512
}
```

**Strategy**: 
- Find longest remaining prompt
- Round to nearest 32
- Cap at 512

**Issues**:
- No empirical basis for 32-alignment or 512 cap
- Doesn't consider hardware characteristics
- No cost model for padding vs transfer

## Performance Model

### Total Cost Function

```
Cost = Transfer_Cost + Compute_Cost + Memory_Cost

Transfer_Cost = num_batches × (launch_latency + tokens/bandwidth)
Compute_Cost = attention_cost + ffn_cost
Memory_Cost = working_set × memory_latency

Where:
  attention_cost ∝ num_layers × (padded_length)²
  ffn_cost ∝ num_layers × padded_length × hidden_dim
  working_set = batch_size × padded_length × hidden_dim
```

### Padding Efficiency

```
efficiency = actual_tokens / (actual_tokens + padding_tokens)

Example: 100 tokens padded to 128
  efficiency = 100 / 128 = 78%
  waste = 22%
```

**Critical**: Attention is O(n²), so padding waste is **quadratic**!

```
Actual:  O(100²) = 10,000 operations
Padded:  O(128²) = 16,384 operations
Waste:   64% more compute
```

### Transfer vs Padding Tradeoff

Given N tokens to process:

**Small chunks** (e.g., 64):
- ✅ Less padding per chunk
- ✅ Better efficiency per batch
- ❌ More batches = more launch overhead
- ❌ Worse cache behavior (more passes)

**Large chunks** (e.g., 1024):
- ✅ Fewer batches = less launch overhead
- ✅ Better cache reuse
- ❌ More padding per chunk
- ❌ Lower efficiency per batch
- ❌ May exceed memory limits

**Sweet spot**: Depends on hardware!

## Hardware-Specific Considerations

### GPU (NVIDIA)

**Optimal alignment**:
- Warp size: 32 threads
- Tensor cores: Multiple of 8 or 16 (for mixed precision)
- **Recommendation**: Align to 16 or 32

**Chunk size**:
- Large kernel launch overhead (~10-50μs)
- High memory bandwidth (1-2 TB/s)
- **Recommendation**: Larger chunks (256-1024) to amortize launch cost

### GPU (AMD)

**Optimal alignment**:
- Wavefront size: 64 threads
- **Recommendation**: Align to 64

**Chunk size**:
- Similar launch overhead to NVIDIA
- **Recommendation**: Larger chunks (256-1024)

### CPU

**Optimal alignment**:
- AVX-512: 16×float32 (64 bytes)
- AVX2: 8×float32 (32 bytes)
- Cache line: 64 bytes
- **Recommendation**: Align to 8, 16, or 32 depending on SIMD

**Chunk size**:
- No kernel launch overhead
- Limited memory bandwidth (~50-200 GB/s)
- Attention O(n²) very expensive on CPU
- **Recommendation**: Smaller chunks (64-256) to minimize attention cost

## Recommended Strategy

### Phase 1: Empirical Benchmarking

Run `benchmark_chunk_sizes.rs` to measure:
1. Throughput (tokens/second) for different chunk sizes
2. Padding efficiency
3. Time to first token (latency)
4. Memory usage

Test configurations:
```
Chunk sizes: [64, 128, 256, 512, 1024, 2048]
Alignments: [8, 16, 32, 64]
```

### Phase 2: Hardware-Adaptive Defaults

```rust
pub fn default_chunk_config(device: &Device) -> ChunkedPrefillConfig {
    match device {
        Device::Cuda(_) => ChunkedPrefillConfig {
            chunk_size: 512,      // Balance launch overhead vs memory
            alignment: 32,        // Warp size
            max_batch_size: 8,    // Larger batches for GPU
            pad_token_id: 0,
        },
        Device::Cpu => ChunkedPrefillConfig {
            chunk_size: 128,      // Minimize attention O(n²) on CPU
            alignment: 16,        // AVX-512 / 16 floats
            max_batch_size: 4,    // Memory constrained
            pad_token_id: 0,
        },
        Device::Metal(_) => ChunkedPrefillConfig {
            chunk_size: 256,      // Apple Silicon sweet spot
            alignment: 32,        // Metal thread group size
            max_batch_size: 6,
            pad_token_id: 0,
        },
    }
}
```

### Phase 3: Adaptive Chunking

Instead of fixed chunk size, adapt based on:

```rust
fn calculate_adaptive_chunk_size(
    &self,
    requests: &[PrefillRequest],
    device: &Device,
) -> usize {
    let max_remaining = requests.iter()
        .map(|r| r.remaining())
        .max()
        .unwrap_or(0);
    
    // Factor 1: Batch efficiency
    let avg_remaining = requests.iter()
        .map(|r| r.remaining())
        .sum::<usize>() / requests.len().max(1);
    
    let efficiency = avg_remaining as f64 / max_remaining as f64;
    
    // Factor 2: Device characteristics
    let (base_chunk, alignment) = match device {
        Device::Cuda(_) => (512, 32),
        Device::Cpu => (128, 16),
        Device::Metal(_) => (256, 32),
    };
    
    // Factor 3: Adjust based on efficiency
    let chunk_size = if efficiency < 0.5 {
        // Very unbalanced batch, use smaller chunks
        base_chunk / 2
    } else if efficiency > 0.8 {
        // Well-balanced batch, can use larger chunks
        base_chunk * 2
    } else {
        base_chunk
    };
    
    // Round and cap
    let aligned = ((chunk_size + alignment - 1) / alignment) * alignment;
    aligned.min(self.config.max_chunk_size)
}
```

## Cost Model Example

Consider processing 1000 tokens:

### Scenario A: chunk_size=128, alignment=32

```
Chunks needed: ⌈1000/128⌉ = 8 batches
Padding per chunk: ~0-28 tokens (if perfectly aligned)
Total tokens processed: ~1000-1224

Transfer cost: 8 × launch_latency
Attention cost: 8 × O(128²) = 8 × 16,384 = 131,072 ops
Efficiency: ~82-100%
```

### Scenario B: chunk_size=512, alignment=32

```
Chunks needed: ⌈1000/512⌉ = 2 batches
Padding: 
  Batch 1: 512 tokens (0 padding)
  Batch 2: 488 → 512 (24 padding)
Total tokens processed: 1024

Transfer cost: 2 × launch_latency
Attention cost: 2 × O(512²) = 2 × 262,144 = 524,288 ops
Efficiency: ~98%
```

**Analysis**:
- Scenario A: 8× fewer attention ops per chunk, but 4× more batches
- Scenario B: 4× fewer batches, but 4× more attention ops per chunk
- **Winner depends on launch_latency vs attention_cost ratio**

### GPU (high launch cost, fast compute):
- Scenario B wins (fewer launches)
- 2 × launch_latency << 4× speedup from fewer batches

### CPU (no launch cost, slow attention):
- Scenario A wins (less attention)
- O(n²) attention dominates

## Implementation Checklist

- [x] Document current chunk sizing logic
- [x] Analyze performance tradeoffs
- [ ] **Run benchmark_chunk_sizes.rs on target hardware**
- [ ] Measure optimal chunk_size and alignment empirically
- [ ] Implement hardware-adaptive defaults
- [ ] Consider adaptive chunking based on batch efficiency
- [ ] Add telemetry for padding efficiency in production
- [ ] Tune based on real workload characteristics

## References

- vLLM continuous batching: https://arxiv.org/abs/2309.06180
- FlashAttention memory efficiency: https://arxiv.org/abs/2205.14135
- PagedAttention for LLM serving: https://arxiv.org/abs/2309.06180

## Key Insight

**There is no universal "best" chunk size!**

The optimal value depends on:
1. Hardware (GPU vs CPU, memory bandwidth, compute speed)
2. Model architecture (attention vs FFN ratio, number of layers)
3. Workload (prompt length distribution, batch size)
4. Optimization goal (throughput vs latency vs efficiency)

**Run benchmarks for your specific configuration.**
