# M3.6 Multi-GPU Inference Architecture

**Milestone**: M3.6 Multi-GPU Inference  
**Status**: IN PROGRESS  
**Date**: October 2025  
**Prerequisites**: M3.4 FlashAttention, M3.5 Testing & Hardening

## Executive Summary

M3.6 enables Lightbulb to run large models (70B+ parameters) across multiple GPUs using **tensor parallelism** and **pipeline parallelism**. This document defines the core architecture, abstractions, and implementation strategy.

**Goals**:
- Run 70B models on 2×40GB GPUs
- Communication overhead <15%
- Throughput 1.5-2× higher than sequential offloading
- Scale to 4 GPUs
- Maintain correctness (numerical parity with single-GPU)

---

## 1. Core Abstractions

### 1.1 DeviceTopology

Discovers and manages multi-GPU configuration:

```rust
/// Multi-GPU device topology and capabilities
#[derive(Debug, Clone)]
pub struct DeviceTopology {
    /// All available CUDA devices
    pub devices: Vec<Device>,
    
    /// Memory capacity per device (bytes)
    pub memory_capacity: Vec<usize>,
    
    /// Available memory per device (bytes)
    pub memory_available: Vec<usize>,
    
    /// Interconnect topology (NVLink, PCIe bandwidth)
    pub interconnect: InterconnectTopology,
    
    /// Peer-to-peer access matrix (can GPU i access GPU j directly?)
    pub p2p_access: Vec<Vec<bool>>,
}

impl DeviceTopology {
    /// Discover all available CUDA devices
    pub fn discover() -> Result<Self> {
        let mut devices = Vec::new();
        let mut device_id = 0;
        
        // Probe CUDA devices until we hit an error
        loop {
            match Device::cuda_if_available(device_id) {
                Ok(device) => {
                    devices.push(device);
                    device_id += 1;
                }
                Err(_) => break,
            }
        }
        
        if devices.is_empty() {
            anyhow::bail!("No CUDA devices available for multi-GPU inference");
        }
        
        // Query memory capacity (TODO: Candle API for memory info)
        let memory_capacity = vec![80 * 1024 * 1024 * 1024; devices.len()]; // Placeholder: 80GB
        let memory_available = memory_capacity.clone();
        
        // Detect interconnect topology
        let interconnect = InterconnectTopology::detect(&devices)?;
        
        // Query peer-to-peer access
        let p2p_access = Self::query_p2p_access(&devices)?;
        
        Ok(Self {
            devices,
            memory_capacity,
            memory_available,
            interconnect,
            p2p_access,
        })
    }
    
    /// Check if peer-to-peer access is available between two devices
    fn query_p2p_access(devices: &[Device]) -> Result<Vec<Vec<bool>>> {
        // TODO: Query actual P2P capabilities via CUDA
        // For now, assume all GPUs can access each other
        let n = devices.len();
        Ok(vec![vec![true; n]; n])
    }
    
    /// Get recommended parallelism strategy based on topology
    pub fn recommend_strategy(&self, model_size_bytes: usize) -> ParallelismMode {
        let num_gpus = self.devices.len();
        let total_memory = self.memory_available.iter().sum::<usize>();
        
        if model_size_bytes > total_memory {
            panic!("Model too large for available GPU memory");
        }
        
        // If model fits on single GPU, no parallelism needed
        if model_size_bytes < self.memory_available[0] {
            return ParallelismMode::Single;
        }
        
        // If model fits on 2 GPUs with tensor parallelism, prefer that
        if num_gpus >= 2 && model_size_bytes < (self.memory_available[0] + self.memory_available[1]) {
            return ParallelismMode::TensorParallel { world_size: 2 };
        }
        
        // Otherwise, use pipeline parallelism with more stages
        ParallelismMode::PipelineParallel {
            num_stages: num_gpus.min(4),
            micro_batch_size: 1,
        }
    }
}

/// Interconnect topology between GPUs
#[derive(Debug, Clone)]
pub enum InterconnectTopology {
    /// NVLink (high bandwidth, low latency)
    NVLink { bandwidth_gbps: f32 },
    
    /// PCIe (lower bandwidth, higher latency)
    PCIe { bandwidth_gbps: f32 },
    
    /// Mixed (some NVLink, some PCIe)
    Mixed { links: Vec<InterconnectLink> },
}

#[derive(Debug, Clone)]
pub struct InterconnectLink {
    pub from_device: usize,
    pub to_device: usize,
    pub link_type: LinkType,
    pub bandwidth_gbps: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum LinkType {
    NVLink,
    PCIe,
}

impl InterconnectTopology {
    fn detect(devices: &[Device]) -> Result<Self> {
        // TODO: Query actual interconnect via CUDA
        // For now, assume NVLink for 2-GPU, PCIe for 4+
        if devices.len() == 2 {
            Ok(Self::NVLink { bandwidth_gbps: 600.0 }) // NVLink 4.0
        } else {
            Ok(Self::PCIe { bandwidth_gbps: 32.0 }) // PCIe 4.0 x16
        }
    }
}
```

### 1.2 Parallelism Modes

Three modes of multi-GPU execution:

```rust
/// Parallelism strategy for multi-GPU inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelismMode {
    /// Single GPU (no parallelism)
    Single,
    
    /// Tensor parallelism: split weights column/row-wise across GPUs
    TensorParallel {
        /// Number of GPUs (world size)
        world_size: usize,
    },
    
    /// Pipeline parallelism: split layers across GPUs
    PipelineParallel {
        /// Number of pipeline stages (GPUs)
        num_stages: usize,
        
        /// Micro-batch size for pipeline scheduling
        micro_batch_size: usize,
    },
    
    /// Hybrid: tensor parallelism within stages, pipeline between stages
    Hybrid {
        /// Tensor parallel world size per stage
        tensor_world_size: usize,
        
        /// Number of pipeline stages
        pipeline_stages: usize,
        
        /// Micro-batch size
        micro_batch_size: usize,
    },
}

impl ParallelismMode {
    /// Total number of GPUs required
    pub fn num_gpus(&self) -> usize {
        match self {
            Self::Single => 1,
            Self::TensorParallel { world_size } => *world_size,
            Self::PipelineParallel { num_stages, .. } => *num_stages,
            Self::Hybrid { tensor_world_size, pipeline_stages, .. } => {
                tensor_world_size * pipeline_stages
            }
        }
    }
    
    /// Communication pattern description
    pub fn communication_pattern(&self) -> &str {
        match self {
            Self::Single => "None",
            Self::TensorParallel { .. } => "All-reduce per layer (high frequency)",
            Self::PipelineParallel { .. } => "Point-to-point between stages (low frequency)",
            Self::Hybrid { .. } => "All-reduce within stages + P2P between stages",
        }
    }
}
```

### 1.3 Multi-GPU Configuration

User-facing configuration for multi-GPU inference:

```rust
/// Multi-GPU inference configuration
#[derive(Debug, Clone)]
pub struct MultiGPUConfig {
    /// Parallelism mode
    pub mode: ParallelismMode,
    
    /// Device topology (discovered automatically)
    pub topology: DeviceTopology,
    
    /// Sharding strategy for tensor parallelism
    pub sharding_strategy: ShardingStrategy,
    
    /// Communication backend (NCCL, custom)
    pub communication_backend: CommunicationBackend,
    
    /// Enable KV cache distribution across GPUs
    pub distributed_kv_cache: bool,
    
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
}

impl MultiGPUConfig {
    /// Create automatic configuration based on model size
    pub fn auto(model_size_bytes: usize) -> Result<Self> {
        let topology = DeviceTopology::discover()?;
        let mode = topology.recommend_strategy(model_size_bytes);
        
        Ok(Self {
            mode,
            topology,
            sharding_strategy: ShardingStrategy::ColumnWise,
            communication_backend: CommunicationBackend::Candle,
            distributed_kv_cache: true,
            load_balancing: LoadBalancingStrategy::MemoryAware,
        })
    }
    
    /// Create manual configuration
    pub fn manual(mode: ParallelismMode, num_devices: usize) -> Result<Self> {
        let topology = DeviceTopology::discover()?;
        
        if topology.devices.len() < num_devices {
            anyhow::bail!(
                "Requested {} GPUs but only {} available",
                num_devices,
                topology.devices.len()
            );
        }
        
        Ok(Self {
            mode,
            topology,
            sharding_strategy: ShardingStrategy::ColumnWise,
            communication_backend: CommunicationBackend::Candle,
            distributed_kv_cache: true,
            load_balancing: LoadBalancingStrategy::MemoryAware,
        })
    }
}

/// Weight sharding strategy for tensor parallelism
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardingStrategy {
    /// Column-wise sharding (split along output dimension)
    ColumnWise,
    
    /// Row-wise sharding (split along input dimension)
    RowWise,
    
    /// Hybrid (column for some layers, row for others)
    Hybrid,
}

/// Communication backend for cross-GPU operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommunicationBackend {
    /// Use Candle's built-in multi-device support
    Candle,
    
    /// NCCL (NVIDIA Collective Communications Library) - if available
    #[allow(dead_code)]
    NCCL,
}

/// Load balancing strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalancingStrategy {
    /// Balance by memory usage
    MemoryAware,
    
    /// Balance by compute time
    ComputeAware,
    
    /// Static round-robin
    RoundRobin,
}
```

---

## 2. Tensor Parallelism Design

### 2.1 Weight Sharding

Split weight matrices across GPUs:

```rust
/// Sharded weight tensor distributed across GPUs
pub struct TensorShard {
    /// Local shard on this device
    pub local_shard: Tensor,
    
    /// Device this shard resides on
    pub device: Device,
    
    /// Rank of this GPU (0 to world_size-1)
    pub rank: usize,
    
    /// Total number of GPUs
    pub world_size: usize,
    
    /// Sharding dimension (0 = row-wise, 1 = column-wise)
    pub shard_dim: usize,
    
    /// Original full shape (before sharding)
    pub full_shape: Vec<usize>,
}

impl TensorShard {
    /// Create sharded weights from full tensor
    pub fn from_full_tensor(
        full_tensor: &Tensor,
        devices: &[Device],
        shard_dim: usize,
    ) -> Result<Vec<Self>> {
        let world_size = devices.len();
        let full_shape = full_tensor.dims().to_vec();
        let shard_size = full_shape[shard_dim] / world_size;
        
        let mut shards = Vec::new();
        for (rank, device) in devices.iter().enumerate() {
            let start = rank * shard_size;
            let end = (rank + 1) * shard_size;
            
            // Narrow the tensor along shard dimension
            let local_shard = full_tensor.narrow(shard_dim, start, end - start)?;
            
            // Copy to target device
            let local_shard = local_shard.to_device(device)?;
            
            shards.push(Self {
                local_shard,
                device: device.clone(),
                rank,
                world_size,
                shard_dim,
                full_shape: full_shape.clone(),
            });
        }
        
        Ok(shards)
    }
    
    /// All-reduce across GPUs (sum shards)
    pub fn all_reduce(shards: &[Tensor]) -> Result<Tensor> {
        if shards.is_empty() {
            anyhow::bail!("Cannot all-reduce empty shard list");
        }
        
        // Sum all shards
        let mut result = shards[0].clone();
        for shard in &shards[1..] {
            result = (result + shard)?;
        }
        
        Ok(result)
    }
    
    /// Gather shards along dimension (concatenate)
    pub fn gather(shards: &[Tensor], dim: usize) -> Result<Tensor> {
        if shards.is_empty() {
            anyhow::bail!("Cannot gather empty shard list");
        }
        
        Tensor::cat(shards, dim)
    }
}
```

### 2.2 Sharded Linear Layer

Distributed matrix multiplication:

```rust
/// Linear layer with tensor parallelism
pub struct ShardedLinear {
    /// Weight shards across GPUs
    pub weight_shards: Vec<TensorShard>,
    
    /// Bias (replicated on all GPUs)
    pub bias: Option<Tensor>,
    
    /// Sharding strategy
    pub strategy: ShardingStrategy,
}

impl ShardedLinear {
    /// Forward pass with column-wise sharding
    /// Input: [batch, in_features]
    /// Weight: [out_features, in_features] → split along out_features
    /// Output: [batch, out_features] (after all-reduce)
    pub fn forward_column_wise(&self, input: &Tensor) -> Result<Tensor> {
        let mut local_outputs = Vec::new();
        
        // Each GPU computes its local output
        for shard in &self.weight_shards {
            // Move input to GPU
            let local_input = input.to_device(&shard.device)?;
            
            // Local matmul: [batch, in_features] @ [out_features_shard, in_features]^T
            let local_output = local_input.matmul(&shard.local_shard.t()?)?;
            
            local_outputs.push(local_output);
        }
        
        // Concatenate along output dimension
        TensorShard::gather(&local_outputs, 1)
    }
    
    /// Forward pass with row-wise sharding
    /// Input: [batch, in_features] → split along in_features
    /// Weight: [out_features, in_features] → split along in_features
    /// Output: [batch, out_features] (after all-reduce sum)
    pub fn forward_row_wise(&self, input: &Tensor) -> Result<Tensor> {
        let world_size = self.weight_shards.len();
        let in_features = input.dim(1)?;
        let shard_size = in_features / world_size;
        
        let mut local_outputs = Vec::new();
        
        // Each GPU computes partial output with its input/weight shard
        for (rank, shard) in self.weight_shards.iter().enumerate() {
            let start = rank * shard_size;
            let end = (rank + 1) * shard_size;
            
            // Narrow input along feature dimension
            let input_shard = input.narrow(1, start, end - start)?;
            let input_shard = input_shard.to_device(&shard.device)?;
            
            // Local matmul: [batch, in_features_shard] @ [out_features, in_features_shard]^T
            let local_output = input_shard.matmul(&shard.local_shard.t()?)?;
            
            local_outputs.push(local_output);
        }
        
        // All-reduce (sum partial outputs)
        TensorShard::all_reduce(&local_outputs)
    }
}
```

---

## 3. Pipeline Parallelism Design

### 3.1 Pipeline Stages

Distribute layers across GPUs:

```rust
/// Single stage in pipeline parallelism
pub struct PipelineStage {
    /// GPU device for this stage
    pub device: Device,
    
    /// Stage ID (0 to num_stages-1)
    pub stage_id: usize,
    
    /// Transformer layers in this stage
    pub layers: Vec<usize>, // Layer indices
    
    /// Input buffer for micro-batches
    pub input_buffer: VecDeque<Tensor>,
    
    /// Output buffer for next stage
    pub output_buffer: VecDeque<Tensor>,
}

impl PipelineStage {
    /// Process one micro-batch through this stage
    pub fn process_micro_batch(
        &mut self,
        input: Tensor,
        model: &BatchedTransformer,
    ) -> Result<Tensor> {
        // Move input to this stage's device
        let input = input.to_device(&self.device)?;
        
        // Forward through assigned layers
        let mut output = input;
        for &layer_idx in &self.layers {
            output = model.forward_single_layer(layer_idx, output)?;
        }
        
        Ok(output)
    }
}

/// Pipeline parallel execution scheduler
pub struct PipelineScheduler {
    /// All pipeline stages
    pub stages: Vec<PipelineStage>,
    
    /// Micro-batch size
    pub micro_batch_size: usize,
    
    /// Scheduling strategy (GPipe, PipeDream, interleaved)
    pub strategy: PipelineStrategy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStrategy {
    /// GPipe: strict forward/backward separation (simpler, more bubbles)
    GPipe,
    
    /// PipeDream: interleaved forward/backward (less bubbles, more complex)
    PipeDream,
    
    /// Interleaved 1F1B: one forward, one backward per stage (balance)
    Interleaved1F1B,
}

impl PipelineScheduler {
    /// Execute pipeline with micro-batching
    pub fn execute(
        &mut self,
        input: Tensor,
        model: &BatchedTransformer,
    ) -> Result<Tensor> {
        let batch_size = input.dim(0)?;
        let num_micro_batches = (batch_size + self.micro_batch_size - 1) / self.micro_batch_size;
        
        match self.strategy {
            PipelineStrategy::GPipe => self.execute_gpipe(input, model, num_micro_batches),
            PipelineStrategy::PipeDream => self.execute_pipedream(input, model, num_micro_batches),
            PipelineStrategy::Interleaved1F1B => self.execute_1f1b(input, model, num_micro_batches),
        }
    }
    
    /// GPipe: strict separation of forward passes, then backward
    fn execute_gpipe(
        &mut self,
        input: Tensor,
        model: &BatchedTransformer,
        num_micro_batches: usize,
    ) -> Result<Tensor> {
        // Split input into micro-batches
        let micro_batches = self.split_into_micro_batches(input)?;
        let mut outputs = Vec::new();
        
        // Forward pass: process all micro-batches through pipeline
        for micro_batch in micro_batches {
            let mut intermediate = micro_batch;
            
            // Pass through each stage sequentially
            for stage in &mut self.stages {
                intermediate = stage.process_micro_batch(intermediate, model)?;
            }
            
            outputs.push(intermediate);
        }
        
        // Concatenate outputs
        Tensor::cat(&outputs, 0)
    }
    
    /// 1F1B: Interleaved one-forward-one-backward (more efficient)
    fn execute_1f1b(
        &mut self,
        input: Tensor,
        model: &BatchedTransformer,
        num_micro_batches: usize,
    ) -> Result<Tensor> {
        // TODO: Implement interleaved 1F1B schedule
        // This requires more complex scheduling to minimize bubbles
        // For now, fall back to GPipe
        self.execute_gpipe(input, model, num_micro_batches)
    }
    
    fn execute_pipedream(
        &mut self,
        input: Tensor,
        model: &BatchedTransformer,
        num_micro_batches: usize,
    ) -> Result<Tensor> {
        // TODO: Implement PipeDream scheduling
        self.execute_gpipe(input, model, num_micro_batches)
    }
    
    fn split_into_micro_batches(&self, input: Tensor) -> Result<Vec<Tensor>> {
        let batch_size = input.dim(0)?;
        let mut micro_batches = Vec::new();
        
        for start in (0..batch_size).step_by(self.micro_batch_size) {
            let end = (start + self.micro_batch_size).min(batch_size);
            let micro_batch = input.narrow(0, start, end - start)?;
            micro_batches.push(micro_batch);
        }
        
        Ok(micro_batches)
    }
}
```

---

## 4. Distributed KV Cache

### 4.1 Cache Coordination

```rust
/// Distributed KV cache manager
pub struct DistributedCacheManager {
    /// Local cache shards per GPU
    pub local_caches: Vec<CacheBuilder>,
    
    /// Device topology
    pub topology: DeviceTopology,
    
    /// Synchronization strategy
    pub sync_strategy: CacheSyncStrategy,
}

#[derive(Debug, Clone, Copy)]
pub enum CacheSyncStrategy {
    /// Replicate full cache on each GPU (simple, high memory)
    Replicated,
    
    /// Shard cache across GPUs (complex, memory-efficient)
    Sharded,
    
    /// Hybrid: replicate frequently accessed, shard rest
    Hybrid,
}

impl DistributedCacheManager {
    /// Update cache after generation step
    pub fn update_cache(
        &mut self,
        layer_idx: usize,
        k_new: &Tensor,
        v_new: &Tensor,
        positions: &[usize],
    ) -> Result<()> {
        match self.sync_strategy {
            CacheSyncStrategy::Replicated => {
                // Update all local caches (simple but memory-intensive)
                for local_cache in &mut self.local_caches {
                    local_cache.update_cache(layer_idx, k_new, v_new, positions)?;
                }
            }
            CacheSyncStrategy::Sharded => {
                // Shard K/V across GPUs
                // TODO: Implement sharded cache strategy
                unimplemented!("Sharded cache strategy")
            }
            CacheSyncStrategy::Hybrid => {
                // TODO: Implement hybrid strategy
                unimplemented!("Hybrid cache strategy")
            }
        }
        
        Ok(())
    }
}
```

---

## 5. Implementation Phases

### Phase 1: Tensor Parallelism (2 GPUs)
**Target**: Run 13B model on 2 GPUs

1. Implement `DeviceTopology::discover()`
2. Implement `TensorShard` with column-wise sharding
3. Create `ShardedLinear` layer
4. Test with small model (1B params)
5. Validate numerical correctness (<1e-3 tolerance)

### Phase 2: Pipeline Parallelism (2-4 GPUs)
**Target**: Run 70B model on 4 GPUs with pipeline

1. Implement `PipelineStage` abstraction
2. Implement GPipe scheduler
3. Test layer-wise distribution
4. Benchmark throughput (target 1.5-2× vs sequential)

### Phase 3: Distributed KV Cache
**Target**: Multi-step generation with cache coordination

1. Implement `DistributedCacheManager`
2. Add replicated cache strategy (simple first)
3. Test correctness with multi-step generation
4. Optimize cache synchronization

### Phase 4: Integration & Optimization
**Target**: Production-ready multi-GPU

1. Integrate with BatchedTransformer
2. Add FlashAttention compatibility
3. Implement load balancing
4. Performance tuning (minimize communication)

---

## 6. Performance Targets

| Configuration             | Target            | Acceptance Criteria                |
| ------------------------- | ----------------- | ---------------------------------- |
| Tensor parallel (2 GPU)   | <15% overhead     | Communication time <15% of compute |
| Pipeline parallel (4 GPU) | 1.5-2× throughput | vs sequential GPU offloading       |
| KV cache distributed      | <10% overhead     | vs single-GPU cache                |
| Numerical correctness     | <1e-3 tolerance   | Multi-GPU vs single-GPU            |
| Scaling efficiency        | >75% @ 4 GPUs     | (4-GPU throughput) / (1-GPU × 4)   |

---

## 7. Limitations & Future Work

### Current Limitations
- Maximum 4 GPUs (implementation complexity)
- CUDA only (no ROCm/Metal support)
- Replicated KV cache (high memory usage)
- No hybrid tensor+pipeline parallelism

### Future Work (M4+)
- Scale to 8+ GPUs with hybrid parallelism
- ROCm support for AMD GPUs
- Sharded KV cache strategy
- NCCL integration for faster communication
- Automatic parallelism selection
- Pipeline bubble minimization (PipeDream, interleaved 1F1B)

---

## 8. References

**Research**:
- GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism
- PipeDream: Generalized Pipeline Parallelism for DNN Training
- Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism
- ZeRO: Memory Optimizations Toward Training Trillion Parameter Models

**Implementation**:
- Candle multi-device support: `Device::cuda_if_available(device_id)`
- FlashAttention compatibility (M3.4)
- M3.5 testing framework for validation

---

## Conclusion

This architecture provides a solid foundation for multi-GPU inference in Lightbulb. Starting with tensor parallelism for 2 GPUs (simpler, immediate value), then expanding to pipeline parallelism for 4+ GPUs (more complex, better scaling).

**Next steps**: Begin implementation with Phase 1 (tensor parallelism foundations).
