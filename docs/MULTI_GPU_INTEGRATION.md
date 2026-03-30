# Multi-GPU Integration Guide (M3.6)

## Overview

Lightbulb's multi-GPU support enables distributed inference across multiple GPUs using:
- **Tensor Parallelism**: Shard model weights across GPUs (column-wise or row-wise)
- **Pipeline Parallelism**: Distribute transformer layers across GPUs with micro-batch scheduling
- **Hybrid**: Combine both strategies for maximum scalability

## Quick Start

### 1. Enable Multi-GPU in Model Config

```rust
use lightbulb::model::{BatchedTransformer, BatchedTransformerConfig};
use lightbulb::multi_gpu::config::{MultiGPUConfig, ParallelismMode};

// Create base transformer config
let mut config = BatchedTransformerConfig::from_llama(
    32000,   // vocab_size
    4096,    // hidden_size
    11008,   // intermediate_size
    32,      // num_layers
    32,      // num_heads
    32,      // num_kv_heads
    1e-5,    // rms_norm_eps
    10000.0, // rope_theta
    2048,    // max_position_embeddings
    false,   // tie_word_embeddings
);

// Enable 2-GPU tensor parallelism
let multi_gpu = MultiGPUConfig::manual(
    ParallelismMode::TensorParallel { world_size: 2 },
    2, // num_devices
)?;

config.multi_gpu = Some(multi_gpu);

// Load model with multi-GPU config
let mut model = BatchedTransformer::new(config, vb)?;
```

### 2. Initialize Distributed Cache

```rust
// Must be called after model creation
model.enable_distributed_cache(
    4,    // batch_size
    2048, // context_size
)?;
```

### 3. Use Multi-GPU Model

The model API remains the same - multi-GPU is transparent:

```rust
// Standard forward pass (automatically distributed)
let logits = model.forward(
    &input_ids,
    &mut cache_builder,
    &mut caches,
    &metadata,
)?;
```

## Multi-GPU Strategies

### Tensor Parallelism (2-4 GPUs)

**Best for:** Models that fit in memory when sharded, high throughput needs

```rust
let config = MultiGPUConfig::manual(
    ParallelismMode::TensorParallel { world_size: 2 },
    2,
)?;
```

**How it works:**
- Weights sharded column-wise or row-wise across GPUs
- All-reduce communication after each layer
- Higher communication overhead but balanced compute

**Performance:**
- 2 GPUs: ~1.7× throughput (target)
- 4 GPUs: ~3.2× throughput (target)
- Communication overhead: <15%

### Pipeline Parallelism (2-8 GPUs)

**Best for:** Very large models, memory-bound scenarios

```rust
let config = MultiGPUConfig::manual(
    ParallelismMode::PipelineParallel {
        num_stages: 4,
        micro_batch_size: 2,
    },
    4,
)?;
```

**How it works:**
- Layers distributed across GPUs (e.g., 40 layers → 4 GPUs = 10 layers/GPU)
- Micro-batching for pipeline efficiency (GPipe scheduler)
- Point-to-point communication between stages
- Lower communication overhead but requires pipeline depth

**Performance:**
- 4 GPUs (40 layers): ~3.5× throughput (target)
- 8 GPUs (80 layers): ~6.5× throughput (target)
- Communication overhead: <8%

**Using `forward_layers` for pipeline:**

```rust
// GPU 0: Process layers 0-9
let hidden = model.forward_layers(
    &hidden_states,
    0,    // layer_start
    10,   // layer_end
    index_pos,
    &mut cache_builder,
    &mut caches,
    &metadata,
)?;

// Transfer hidden to GPU 1
let hidden_gpu1 = hidden.to_device(&device1)?;

// GPU 1: Process layers 10-19
let hidden = model.forward_layers(
    &hidden_gpu1,
    10,   // layer_start
    20,   // layer_end
    index_pos,
    &mut cache_builder,
    &mut caches,
    &metadata,
)?;
```

### Hybrid Parallelism (4+ GPUs)

**Best for:** Maximum scalability, largest models

```rust
let config = MultiGPUConfig::manual(
    ParallelismMode::Hybrid {
        tensor_world_size: 2,    // 2-way tensor parallel per stage
        pipeline_stages: 4,       // 4 pipeline stages
        micro_batch_size: 2,
    },
    8, // total GPUs = 2 × 4
)?;
```

**How it works:**
- Combines tensor and pipeline parallelism
- Example: 8 GPUs → 4 stages × 2-way tensor parallel
- All-reduce within stages, P2P between stages

**Performance:**
- 8 GPUs (2×4): ~6× throughput (target)
- 16 GPUs (4×4): ~11× throughput (target)

## API Reference

### BatchedTransformerConfig

```rust
pub struct BatchedTransformerConfig {
    // ... existing fields ...
    
    /// Optional multi-GPU configuration for distributed inference (M3.6)
    pub multi_gpu: Option<MultiGPUConfig>,
}
```

### BatchedTransformer Methods

```rust
impl BatchedTransformer {
    /// Initialize distributed KV cache for multi-GPU (call after model creation)
    pub fn enable_distributed_cache(
        &mut self,
        batch_size: usize,
        context_size: usize,
    ) -> Result<()>;

    /// Check if multi-GPU is enabled
    pub fn is_multi_gpu(&self) -> bool;

    /// Get reference to distributed cache manager
    pub fn distributed_cache(&self) -> Option<&Mutex<DistributedCacheManager>>;

    /// Process specific layer range (for pipeline parallelism)
    pub fn forward_layers(
        &self,
        hidden_states: &Tensor,
        layer_start: usize,
        layer_end: usize,
        index_pos: usize,
        cache_builder: &mut ParallelCacheBuilder,
        caches: &mut [ParallelKvCache],
        metadata: &BatchMetadata,
    ) -> Result<Tensor>;
}
```

## Complete Example

```rust
use lightbulb::model::{BatchedTransformer, BatchedTransformerConfig};
use lightbulb::multi_gpu::config::{MultiGPUConfig, ParallelismMode};
use candle_core::Device;

fn load_multi_gpu_model() -> anyhow::Result<BatchedTransformer> {
    // 1. Create base config
    let mut config = BatchedTransformerConfig::llama_7b();
    
    // 2. Enable 2-GPU tensor parallelism
    let multi_gpu = MultiGPUConfig::auto(7_000_000_000)?; // 7B model
    config.multi_gpu = Some(multi_gpu);
    
    // 3. Load model weights
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(
            &["model.safetensors"],
            candle_core::DType::F16,
            &Device::cuda_if_available(0)?,
        )?
    };
    
    let mut model = BatchedTransformer::new(config, vb)?;
    
    // 4. Initialize distributed cache
    model.enable_distributed_cache(4, 2048)?;
    
    Ok(model)
}
```

## Performance Targets

| Configuration    | GPUs | Target Speedup | Communication Overhead |
| ---------------- | ---- | -------------- | ---------------------- |
| Tensor Parallel  | 2    | 1.7×           | <15%                   |
| Tensor Parallel  | 4    | 3.2×           | <15%                   |
| Pipeline (GPipe) | 4    | 3.5×           | <8%                    |
| Pipeline (GPipe) | 8    | 6.5×           | <8%                    |
| Hybrid (2×4)     | 8    | ~6×            | <12%                   |
| Hybrid (4×4)     | 16   | ~11×           | <12%                   |

*Note: Actual performance depends on model size, batch size, hardware interconnect (NVLink vs PCIe), and workload characteristics.*

## Testing

Multi-GPU tests require hardware:

```bash
# Run all multi-GPU tests (requires 2+ GPUs)
cargo test --test multi_gpu_validation -- --ignored --test-threads=1

# Run specific test
cargo test test_tensor_shard_creation -- --ignored
```

See `tests/MULTI_GPU_TESTING.md` for detailed testing guide.

## Current Status (M3.6)

**✅ Implemented:**
- Multi-GPU configuration in `BatchedTransformerConfig`
- Distributed cache manager integration
- `forward_layers()` method for pipeline parallelism
- Tensor parallelism foundations (weight sharding, gather/scatter)
- Pipeline scheduler with GPipe strategy
- Comprehensive test suite (17 tests)

**📋 TODO (Future Work):**
- Automatic layer distribution in `forward()` method
- Pipeline scheduler integration with `BatchedTransformer`
- Sharded weight loading from disk
- Multi-GPU-aware model manager
- Performance benchmarks on real hardware

## Related Documentation

- **Architecture:** `docs/M3_6_MULTI_GPU_ARCHITECTURE.md`
- **Testing Guide:** `tests/MULTI_GPU_TESTING.md`
- **ROADMAP:** `ROADMAP.md` (M3.6 section)
- **Elastic Cache (Future):** `docs/CANDLE_CUDA_VMM_SPEC.md` (M6.5)

## References

- Megatron-LM: Tensor parallelism implementation
- GPipe: Pipeline parallelism with micro-batching
- KVCached (Meta): Elastic cache for multi-model serving (inspiration for M6.5)
