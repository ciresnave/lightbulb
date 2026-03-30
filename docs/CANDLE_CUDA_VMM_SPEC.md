# Candle CUDA VMM Implementation Specification

**Purpose**: Detailed specification for building `candle-cuda-vmm`, a Rust crate providing CUDA Virtual Memory Management bindings for elastic KV cache allocation in Lightbulb.

**Target Audience**: LLM assistant or developer tasked with implementing CUDA VMM bindings

**Context**: Lightbulb is a Candle-based LLM inference engine. We need elastic KV cache management inspired by Meta's KVCached library to support:
- Dynamic memory allocation/deallocation for KV caches
- Multi-model serving with shared GPU memory pools
- Reduced time-to-first-token (TTFT) via on-demand page mapping
- Memory efficiency for bursty multi-tenant workloads

---

## Project Overview

**Crate Name**: `candle-cuda-vmm`  
**Location**: New crate in Candle ecosystem
**Dependencies**:
- `candle-core` (for Device, Tensor integration)
- `cuda-sys` or similar CUDA FFI bindings
- `libc` for memory operations

**Architecture**:
```
candle-cuda-vmm/
├── src/
│   ├── lib.rs              # Public API and re-exports
│   ├── virtual_memory.rs   # VirtualMemoryPool abstraction
│   ├── physical_memory.rs  # PhysicalMemoryHandle and allocation
│   ├── mapping.rs          # Page mapping/unmapping operations
│   ├── cuda_ffi.rs         # Raw CUDA VMM FFI bindings
│   └── error.rs            # Error types
├── tests/
│   ├── basic_allocation.rs
│   ├── mapping_tests.rs
│   └── multi_pool.rs
└── examples/
    ├── simple_usage.rs
    └── kv_cache_simulation.rs
```

---

## Required CUDA VMM APIs

The following CUDA Virtual Memory Management APIs must be wrapped in safe Rust:

### 1. **Physical Memory Allocation**
```c
// CUDA API
CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, 
                     size_t size, 
                     const CUmemAllocationProp *prop,
                     unsigned long long flags);

CUresult cuMemRelease(CUmemGenericAllocationHandle handle);
```

**Rust Wrapper Requirements**:
```rust
/// Handle to physical GPU memory allocation
pub struct PhysicalMemoryHandle {
    handle: CUmemGenericAllocationHandle,
    size: usize,
    device: Device, // Candle Device
}

impl PhysicalMemoryHandle {
    /// Allocate physical GPU memory
    pub fn new(size: usize, device: &Device) -> Result<Self>;
    
    /// Get size in bytes
    pub fn size(&self) -> usize;
    
    /// Get associated device
    pub fn device(&self) -> &Device;
}

// Auto-release on drop
impl Drop for PhysicalMemoryHandle {
    fn drop(&mut self) {
        // Call cuMemRelease
    }
}
```

### 2. **Virtual Address Reservation**
```c
// CUDA API
CUresult cuMemAddressReserve(CUdeviceptr *ptr,
                              size_t size,
                              size_t alignment,
                              CUdeviceptr addr,
                              unsigned long long flags);

CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size);
```

**Rust Wrapper Requirements**:
```rust
/// Virtual address space reservation
pub struct VirtualAddressRange {
    ptr: CUdeviceptr,
    size: usize,
    alignment: usize,
}

impl VirtualAddressRange {
    /// Reserve contiguous virtual address space
    pub fn new(size: usize, alignment: usize) -> Result<Self>;
    
    /// Get base virtual address
    pub fn base_address(&self) -> usize;
    
    /// Get total size
    pub fn size(&self) -> usize;
}

// Auto-free on drop
impl Drop for VirtualAddressRange {
    fn drop(&mut self) {
        // Call cuMemAddressFree
    }
}
```

### 3. **Memory Mapping**
```c
// CUDA API
CUresult cuMemMap(CUdeviceptr ptr,
                  size_t size,
                  size_t offset,
                  CUmemGenericAllocationHandle handle,
                  unsigned long long flags);

CUresult cuMemUnmap(CUdeviceptr ptr, size_t size);
```

**Rust Wrapper Requirements**:
```rust
/// Map physical memory to virtual address range
pub fn map_memory(
    virtual_range: &VirtualAddressRange,
    offset: usize,
    physical_handle: &PhysicalMemoryHandle,
    physical_offset: usize,
    size: usize,
) -> Result<()>;

/// Unmap memory from virtual address range
pub fn unmap_memory(
    virtual_range: &VirtualAddressRange,
    offset: usize,
    size: usize,
) -> Result<()>;
```

### 4. **Access Control**
```c
// CUDA API
CUresult cuMemSetAccess(CUdeviceptr ptr,
                        size_t size,
                        const CUmemAccessDesc *desc,
                        size_t count);
```

**Rust Wrapper Requirements**:
```rust
/// Memory access permissions
pub enum AccessFlags {
    None,
    Read,
    ReadWrite,
}

/// Set memory access permissions
pub fn set_memory_access(
    virtual_range: &VirtualAddressRange,
    offset: usize,
    size: usize,
    device: &Device,
    flags: AccessFlags,
) -> Result<()>;
```

---

## High-Level Abstractions for Lightbulb

### VirtualMemoryPool (Main API)

```rust
/// Elastic memory pool with virtual memory backing
pub struct VirtualMemoryPool {
    virtual_range: VirtualAddressRange,
    physical_pages: Vec<Option<PhysicalMemoryHandle>>,
    page_size: usize,
    total_capacity: usize,
    mapped_size: usize,
    device: Device,
}

impl VirtualMemoryPool {
    /// Create a new virtual memory pool
    ///
    /// # Arguments
    /// * `capacity` - Maximum virtual address space (e.g., 128GB)
    /// * `page_size` - Page granularity (e.g., 2MB for large pages)
    /// * `device` - CUDA device
    ///
    /// # Returns
    /// Pool with reserved virtual address space, no physical memory allocated
    pub fn new(capacity: usize, page_size: usize, device: Device) -> Result<Self>;
    
    /// Allocate and map physical pages on-demand
    ///
    /// # Arguments
    /// * `offset` - Offset in virtual address space
    /// * `size` - Number of bytes to allocate
    ///
    /// # Returns
    /// Base virtual address of allocated region
    pub fn allocate(&mut self, offset: usize, size: usize) -> Result<usize>;
    
    /// Unmap and free physical pages
    ///
    /// # Arguments
    /// * `offset` - Offset in virtual address space
    /// * `size` - Number of bytes to free
    pub fn deallocate(&mut self, offset: usize, size: usize) -> Result<()>;
    
    /// Get current physical memory usage
    pub fn physical_memory_usage(&self) -> usize;
    
    /// Get virtual address space capacity
    pub fn capacity(&self) -> usize;
    
    /// Get base virtual address
    pub fn base_address(&self) -> usize;
    
    /// Check if a range is currently mapped
    pub fn is_mapped(&self, offset: usize, size: usize) -> bool;
    
    /// Compact pool by coalescing free pages
    pub fn compact(&mut self) -> Result<()>;
}
```

### SharedMemoryPool (Multi-Model Support)

```rust
/// Shared memory pool for multiple models
pub struct SharedMemoryPool {
    pools: HashMap<String, VirtualMemoryPool>, // model_id -> pool
    global_physical_limit: usize,
    current_physical_usage: usize,
    device: Device,
}

impl SharedMemoryPool {
    /// Create shared pool with global physical memory limit
    pub fn new(physical_limit: usize, device: Device) -> Result<Self>;
    
    /// Register a model with virtual address space reservation
    pub fn register_model(&mut self, model_id: &str, virtual_capacity: usize) -> Result<()>;
    
    /// Allocate from specific model's pool
    pub fn allocate_for_model(&mut self, model_id: &str, size: usize) -> Result<usize>;
    
    /// Free from specific model's pool
    pub fn deallocate_for_model(&mut self, model_id: &str, offset: usize, size: usize) -> Result<()>;
    
    /// Get per-model memory statistics
    pub fn get_model_stats(&self, model_id: &str) -> Option<MemoryStats>;
    
    /// Global memory statistics
    pub fn global_stats(&self) -> GlobalMemoryStats;
}

pub struct MemoryStats {
    pub virtual_capacity: usize,
    pub physical_usage: usize,
    pub mapped_pages: usize,
    pub fragmentation_ratio: f32,
}
```

---

## Integration Points with Lightbulb

### 1. **KV Cache Allocation**

Current Lightbulb code uses static `ParallelCacheBuilder`. With `candle-cuda-vmm`:

```rust
// Before (static allocation)
pub struct ParallelCacheBuilder {
    kv_cache: Tensor, // Fixed size allocation
    // ...
}

// After (elastic allocation)
pub struct ElasticCacheBuilder {
    virtual_pool: VirtualMemoryPool,
    allocated_blocks: Vec<(usize, usize)>, // (offset, size)
    max_tokens: usize,
    current_tokens: usize,
    // ...
}

impl ElasticCacheBuilder {
    pub fn new(max_capacity: usize, device: Device) -> Result<Self> {
        let virtual_pool = VirtualMemoryPool::new(
            max_capacity * 1024 * 1024, // e.g., 128GB virtual
            2 * 1024 * 1024,            // 2MB pages
            device,
        )?;
        Ok(Self {
            virtual_pool,
            allocated_blocks: Vec::new(),
            max_tokens: max_capacity / token_size,
            current_tokens: 0,
        })
    }
    
    pub fn allocate_for_tokens(&mut self, num_tokens: usize) -> Result<()> {
        let size = num_tokens * self.token_size();
        let offset = self.current_tokens * self.token_size();
        self.virtual_pool.allocate(offset, size)?;
        self.allocated_blocks.push((offset, size));
        self.current_tokens += num_tokens;
        Ok(())
    }
    
    pub fn free_tokens(&mut self, num_tokens: usize) -> Result<()> {
        if let Some((offset, size)) = self.allocated_blocks.pop() {
            self.virtual_pool.deallocate(offset, size)?;
            self.current_tokens -= num_tokens;
        }
        Ok(())
    }
}
```

### 2. **Multi-Model Serving**

```rust
// In Lightbulb's engine
pub struct MultiModelEngine {
    shared_pool: SharedMemoryPool,
    models: HashMap<String, Model>,
}

impl MultiModelEngine {
    pub fn new(total_gpu_memory: usize, device: Device) -> Result<Self> {
        let shared_pool = SharedMemoryPool::new(total_gpu_memory, device)?;
        Ok(Self {
            shared_pool,
            models: HashMap::new(),
        })
    }
    
    pub fn load_model(&mut self, model_id: &str, virtual_capacity: usize) -> Result<()> {
        self.shared_pool.register_model(model_id, virtual_capacity)?;
        // Load model weights...
        Ok(())
    }
    
    pub fn process_request(&mut self, model_id: &str, tokens: &[u32]) -> Result<Tensor> {
        let kv_size = tokens.len() * KV_BYTES_PER_TOKEN;
        let addr = self.shared_pool.allocate_for_model(model_id, kv_size)?;
        // Process inference...
        Ok(output)
    }
}
```

---

## Safety Requirements

1. **Memory Safety**:
   - All CUDA calls must be checked for errors
   - Use Rust's type system to prevent use-after-free
   - Implement RAII patterns (Drop trait) for automatic cleanup

2. **Thread Safety**:
   - `VirtualMemoryPool` should be `Send` but not `Sync` (single-threaded per pool)
   - Use `Arc<Mutex<SharedMemoryPool>>` for multi-threaded access

3. **Error Handling**:
   ```rust
   #[derive(Debug, thiserror::Error)]
   pub enum VmmError {
       #[error("CUDA error: {0}")]
       CudaError(String),
       
       #[error("Out of virtual address space")]
       OutOfVirtualMemory,
       
       #[error("Out of physical memory")]
       OutOfPhysicalMemory,
       
       #[error("Invalid offset: {0}")]
       InvalidOffset(usize),
       
       #[error("Mapping failed: {0}")]
       MappingFailed(String),
   }
   ```

---

## Testing Requirements

1. **Basic Allocation Tests**:
   - Allocate/deallocate single page
   - Allocate multiple pages
   - Deallocate in different orders (FIFO, LIFO, random)

2. **Stress Tests**:
   - Allocate until physical memory exhausted
   - Rapid allocation/deallocation cycles
   - Fragmentation scenarios

3. **Multi-Pool Tests**:
   - Multiple pools sharing physical memory
   - Pool eviction and rebalancing
   - Fairness testing

4. **Integration Tests**:
   - Simulate KV cache workload
   - Multi-model serving scenario
   - Long-running stability test

---

## Performance Targets

Based on KVCached benchmarks and production requirements:

1. **Allocation Latency**: <100μs per page (2MB)
2. **TTFT Improvement**: 1.2-28× faster vs static allocation (multi-model scenarios)
3. **Memory Overhead**: <5% metadata/bookkeeping overhead
4. **Throughput**: No degradation vs static allocation for single-model workloads

---

## Documentation Requirements

1. **API Documentation**:
   - Rustdoc for all public APIs
   - Usage examples for common patterns
   - Safety notes for unsafe operations

2. **User Guide**:
   - Quick start tutorial
   - Integration guide for Candle users
   - Performance tuning recommendations

3. **Internal Design Doc**:
   - CUDA VMM architecture overview
   - Page management algorithm
   - Multi-pool coordination strategy

---

## Delivery Checklist

- [ ] All CUDA VMM APIs wrapped with safe Rust interfaces
- [ ] `VirtualMemoryPool` implemented and tested
- [ ] `SharedMemoryPool` implemented and tested
- [ ] Integration example with Candle Tensor
- [ ] Comprehensive test suite (>80% coverage)
- [ ] Benchmarks comparing static vs elastic allocation
- [ ] Documentation (API docs + user guide)
- [ ] CI setup with CUDA tests
- [ ] Published to crates.io (or Candle repo)

---

## References

- **KVCached Paper**: [Prism: Multi-LLM Serving with VMM](https://www.arxiv.org/pdf/2505.04021)
- **CUDA VMM Docs**: [NVIDIA CUDA Virtual Memory Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html)
- **vAttention**: [Virtual Memory for PagedAttention Alternative](https://arxiv.org/abs/2508.08448)
- **Candle Source**: [huggingface/candle](https://github.com/huggingface/candle)

---

## Implementation Prompt for LLM

> **Prompt**:
> 
> You are tasked with implementing `candle-cuda-vmm`, a Rust crate providing safe bindings to CUDA Virtual Memory Management APIs for the Candle deep learning framework.
> 
> **Requirements**:
> 1. Wrap CUDA VMM APIs (`cuMemCreate`, `cuMemAddressReserve`, `cuMemMap`, etc.) in safe Rust
> 2. Implement `VirtualMemoryPool` for elastic memory allocation with on-demand page mapping
> 3. Implement `SharedMemoryPool` for multi-model memory sharing
> 4. Ensure all allocations use RAII patterns (Drop trait) for automatic cleanup
> 5. Add comprehensive error handling with `thiserror`
> 6. Write tests covering allocation, deallocation, multi-pool scenarios
> 7. Provide integration examples with Candle's `Device` and `Tensor` types
> 
> **Key Design Principles**:
> - Safety first: no manual memory management exposed to users
> - Performance: minimize allocation latency (<100μs per 2MB page)
> - Ergonomics: simple API for common use cases (KV cache allocation)
> - Compatibility: works with existing Candle code without major changes
> 
> **Use Case**: Enable Lightbulb inference engine to support elastic KV cache allocation for multi-model serving, reducing TTFT by 1.2-28× vs static allocation in bursty workloads.
> 
> Refer to the full specification in `docs/CANDLE_CUDA_VMM_SPEC.md` for detailed API requirements, safety constraints, and performance targets.
> 
> Begin by creating the project structure, then implement core CUDA FFI bindings, followed by high-level abstractions. Prioritize correctness over performance initially, then optimize critical paths.
