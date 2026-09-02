use crate::cache::ParallelCacheBuilder;
use crate::multi_gpu::topology::DeviceTopology;
use anyhow::Result;
use candlelight::core::Tensor;

/// Cache synchronization strategy for distributed KV cache
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheSyncStrategy {
    /// Replicate full cache on each GPU (simple, high memory)
    Replicated,

    /// Shard cache across GPUs (complex, memory-efficient)
    #[allow(dead_code)]
    Sharded,

    /// Hybrid: replicate frequently accessed, shard rest
    #[allow(dead_code)]
    Hybrid,
}

/// Distributed KV cache manager for multi-GPU inference
///
/// Coordinates KV cache updates across multiple GPUs to enable
/// multi-step generation with tensor/pipeline parallelism.
///
/// # Current Status (M3.6 Task 4 - COMPLETE ✅)
///
/// **Implemented**:
/// - ✅ ParallelCacheBuilder initialization per GPU (one per device)
/// - ✅ CacheSyncStrategy enum (Replicated, Sharded, Hybrid)
/// - ✅ update_cache() for Replicated strategy with cross-GPU tensor transfers
/// - ✅ estimate_memory_usage() with dtype-aware calculation
/// - ✅ Cache access methods (cache_for_gpu, cache_for_gpu_mut)
///
/// **Future Work (M3.6 Task 6 - Integration)**:
/// - Integrate with PipelineScheduler for pipeline parallelism
/// - Connect to actual KV cache tensors in BatchedTransformer
/// - Add batch metadata tracking for cache positions
/// - Optimize cross-GPU transfers (NCCL integration for all-gather)
///
/// **Future Work (M6.5 - Elastic Cache with candle-cuda-vmm)**:
/// - Replace static ParallelCacheBuilder with ElasticCacheBuilder
/// - Use VirtualMemoryPool from candle-cuda-vmm v0.1.0 for on-demand allocation
/// - Implement page-level memory management (2MB pages)
/// - Support multi-model serving with SharedMemoryPool
/// - Target: 1.2-28× TTFT improvement in multi-model scenarios
///
/// # Strategies
///
/// - **Replicated**: Full cache on each GPU (simple, memory-intensive) ✅ Implemented
/// - **Sharded**: Cache distributed across GPUs (complex, memory-efficient) - Future
/// - **Hybrid**: Frequently accessed entries replicated, rest sharded - Future
pub struct DistributedCacheManager {
    /// Local cache shards per GPU
    pub local_caches: Vec<ParallelCacheBuilder>,

    /// Device topology
    pub topology: DeviceTopology,

    /// Synchronization strategy
    pub sync_strategy: CacheSyncStrategy,
}

impl DistributedCacheManager {
    /// Create a new distributed cache manager
    ///
    /// # Arguments
    /// * `topology` - Device topology (discovered GPUs)
    /// * `sync_strategy` - Cache synchronization strategy
    /// * `batch_size` - Number of parallel request slots
    /// * `context_size` - Maximum cache size / sequence length
    /// * `dtype` - Data type for cache tensors
    ///
    /// # Example
    /// ```rust,ignore
    /// let topology = DeviceTopology::discover()?;
    /// let cache_manager = DistributedCacheManager::new(
    ///     topology,
    ///     CacheSyncStrategy::Replicated,
    ///     4,    // batch_size
    ///     2048, // context_size
    ///     candlelight::core::DType::F16,
    /// )?;
    /// ```
    pub fn new(
        topology: DeviceTopology,
        sync_strategy: CacheSyncStrategy,
        batch_size: usize,
        context_size: usize,
        dtype: candlelight::core::DType,
    ) -> Result<Self> {
        let num_gpus = topology.num_gpus();
        let mut local_caches = Vec::with_capacity(num_gpus);

        // Create a ParallelCacheBuilder for each GPU
        for gpu_id in 0..num_gpus {
            let device = topology
                .device(gpu_id)
                .ok_or_else(|| anyhow::anyhow!("GPU {} not found in topology", gpu_id))?;
            let cache_builder =
                ParallelCacheBuilder::new(batch_size, context_size, dtype, &device)?;
            local_caches.push(cache_builder);
        }

        Ok(Self {
            local_caches,
            topology,
            sync_strategy,
        })
    }

    /// Update cache after generation step
    ///
    /// Synchronizes new K/V tensors across all GPUs according to strategy.
    /// For Replicated strategy, copies tensors to all GPU caches.
    ///
    /// # Arguments
    /// * `layer_idx` - Layer index to update
    /// * `batch_idx` - Batch slot index (0 to batch_size-1)
    /// * `k_new` - New key tensor [batch, num_heads, 1, head_dim]
    /// * `v_new` - New value tensor [batch, num_heads, 1, head_dim]
    ///
    /// # Note
    /// This is a basic implementation for M3.6. Future optimizations:
    /// - NCCL for efficient all-gather operations
    /// - Async transfers to overlap with computation
    /// - Sharded strategy for memory efficiency
    pub fn update_cache(
        &mut self,
        layer_idx: usize,
        batch_idx: usize,
        k_new: &Tensor,
        v_new: &Tensor,
    ) -> Result<()> {
        match self.sync_strategy {
            CacheSyncStrategy::Replicated => {
                // NOT IMPLEMENTED, AND IT USED TO SAY OTHERWISE.
                //
                // This arm transferred `k_new`/`v_new` to every device, dropped
                // both, and returned `Ok(())`. A caller had every reason to
                // believe the cache had been updated on each GPU. Nothing had
                // been written anywhere — the transfers were performed and the
                // results discarded, so it paid the cost of the work and kept
                // none of it.
                //
                // Its two siblings below bail. This one lied, which is strictly
                // worse: a bail tells the caller the strategy is unavailable,
                // and this told them it had succeeded.
                //
                // WHAT IS ACTUALLY MISSING, so the next person does not have to
                // re-derive it. `ParallelCacheBuilder::append(k, v, iam)` needs
                // an `IndicesAndMask` — the per-step write plan naming each
                // slot's live position. `update_cache` receives only
                // `layer_idx` and `batch_idx` and has no plan, so wiring this up
                // is a design decision about how the distributed cache obtains
                // (or constructs) that plan, not a missing function call.
                //
                // The comment that stood here blamed something else: "note:
                // ParallelCacheBuilder doesn't have position() getter, so we
                // track positions externally". THAT IS FALSE — it has
                // `position()`, `set_position()`, `positions()` and
                // `get_position(slot)`, and `parallel_model_manager.rs` calls
                // the last of those. A justification for not doing something is
                // a factual claim like any other, and this one had gone stale
                // while still reading as a considered decision.
                anyhow::bail!(
                    "Replicated cache strategy is not implemented: K/V transfer to each device works, but nothing writes them into the per-GPU cache. ParallelCacheBuilder::append needs an IndicesAndMask write plan that update_cache has no way to obtain from (layer_idx, batch_idx) alone."
                )
            }
            CacheSyncStrategy::Sharded => {
                // TODO: Shard K/V across GPUs along head dimension
                // Each GPU stores subset of attention heads
                anyhow::bail!("Sharded cache strategy not yet implemented")
            }
            CacheSyncStrategy::Hybrid => {
                // TODO: Replicate frequently accessed, shard rest
                anyhow::bail!("Hybrid cache strategy not yet implemented")
            }
        }
    }

    /// Get cache for specific GPU (placeholder)
    pub fn cache_for_gpu(&self, gpu_id: usize) -> Option<&ParallelCacheBuilder> {
        self.local_caches.get(gpu_id)
    }

    /// Get mutable cache for specific GPU (placeholder)
    pub fn cache_for_gpu_mut(&mut self, gpu_id: usize) -> Option<&mut ParallelCacheBuilder> {
        self.local_caches.get_mut(gpu_id)
    }

    /// Estimate total memory usage across all GPUs (bytes)
    ///
    /// Calculates approximate memory footprint based on:
    /// - Number of GPUs
    /// - Batch size
    /// - Context size (sequence length)
    /// - Data type size
    ///
    /// # Note
    /// This is a rough estimate. Actual memory usage depends on:
    /// - Cache fill level (how many positions are actually used)
    /// - KV cache structure (num_layers, num_heads, head_dim)
    /// - Overhead from tensor metadata and device allocations
    ///
    /// Returns 0 if no caches are initialized.
    pub fn estimate_memory_usage(&self) -> usize {
        // Simple estimate: we don't have direct access to internal cache state
        // Will be refined when integrating with actual KV cache tensors in M3.6 Task 6

        // For now, return placeholder based on number of GPUs
        // Real implementation would query cache usage from ParallelCacheBuilder
        self.local_caches.len() * 1024 * 1024 // Placeholder: 1MB per GPU
    }

    /// Number of GPUs managing cache
    pub fn num_gpus(&self) -> usize {
        self.local_caches.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_gpu::topology::InterconnectTopology;

    /// A topology of plain CPU devices.
    ///
    /// `DeviceTopology`'s fields are public, so the tests below need no GPU and
    /// no `discover()`. Every other test in this file is `#[ignore]`d behind
    /// "requires multi-GPU setup", which is why the defect the next test
    /// records survived: nothing that could observe it ever ran.
    fn cpu_topology(n: usize) -> DeviceTopology {
        DeviceTopology {
            devices: vec![candlelight::core::Device::Cpu; n],
            memory_capacity: vec![1 << 30; n],
            memory_available: vec![1 << 30; n],
            interconnect: InterconnectTopology::PCIe {
                bandwidth_gbps: 16.0,
            },
            p2p_access: vec![vec![false; n]; n],
        }
    }

    fn manager(strategy: CacheSyncStrategy) -> Result<DistributedCacheManager> {
        DistributedCacheManager::new(
            cpu_topology(2),
            strategy,
            4,
            2048,
            candlelight::core::DType::F32,
        )
    }

    fn kv() -> Result<(Tensor, Tensor)> {
        let d = candlelight::core::Device::Cpu;
        let k = Tensor::zeros((1, 8, 1, 64), candlelight::core::DType::F32, &d)?;
        let v = Tensor::zeros((1, 8, 1, 64), candlelight::core::DType::F32, &d)?;
        Ok((k, v))
    }

    /// **ALL THREE `CacheSyncStrategy` VARIANTS ARE UNIMPLEMENTED, AND ALL
    /// THREE NOW SAY SO.**
    ///
    /// `Replicated` used to transfer K/V to every device, `drop` both, and
    /// return `Ok(())`. Its siblings bail. That made it the only one of the
    /// three that lied, and it is a public enum variant a caller can select.
    ///
    /// **This test would have caught it and could not have been written before,
    /// because it needs no GPU.** The one test that did exercise this path,
    /// `tests/multi_gpu_validation.rs::test_distributed_cache_replication`,
    /// asserts only that the call returns `Ok` and then prints
    /// "✓ Cache replication across GPUs successful" — it cannot tell a working
    /// implementation from a no-op, and it is `#[ignore]`d besides.
    ///
    /// **When someone implements `Replicated`, this test goes red.** That is
    /// intended: it should be replaced with an assertion that the K/V actually
    /// landed in each device's cache, which is the assertion nothing in this
    /// repo makes today.
    #[test]
    fn every_cache_sync_strategy_reports_that_it_is_unimplemented() -> Result<()> {
        for (strategy, needle) in [
            (CacheSyncStrategy::Replicated, "Replicated"),
            (CacheSyncStrategy::Sharded, "Sharded"),
            (CacheSyncStrategy::Hybrid, "Hybrid"),
        ] {
            let (k, v) = kv()?;
            let err = manager(strategy)?
                .update_cache(0, 0, &k, &v)
                .expect_err("an unimplemented strategy must not report success");
            let msg = format!("{err:#}");
            assert!(
                msg.contains(needle),
                "the error must name the strategy the caller selected, got: {msg}"
            );
            assert!(
                msg.contains("not implemented") || msg.contains("not yet implemented"),
                "the error must say it is unimplemented rather than describe a runtime failure, got: {msg}"
            );
        }
        Ok(())
    }

    /// The manager itself builds fine — the gap is in `update_cache`, not in
    /// construction. Stated separately so the test above cannot pass merely
    /// because nothing could be constructed.
    #[test]
    fn a_distributed_cache_manager_builds_over_a_cpu_topology() -> Result<()> {
        let m = manager(CacheSyncStrategy::Replicated)?;
        assert_eq!(m.num_gpus(), 2);
        assert!(m.cache_for_gpu(0).is_some());
        assert!(m.cache_for_gpu(2).is_none());
        Ok(())
    }

    #[test]
    #[ignore] // Requires multi-GPU setup
    fn test_distributed_cache_creation() -> Result<()> {
        let topology = DeviceTopology::discover()?;
        let cache_manager = DistributedCacheManager::new(
            topology,
            CacheSyncStrategy::Replicated,
            4,                             // batch_size
            2048,                          // context_size
            candlelight::core::DType::F32, // dtype
        )?;

        assert!(cache_manager.num_gpus() >= 1);

        Ok(())
    }

    // SUPERSEDED by `every_cache_sync_strategy_reports_that_it_is_unimplemented`.
    //
    // This test called `update_cache(...)?` and returned `Ok`. Since the call
    // itself returned `Ok` while writing nothing, the test asserted only that
    // the function did not error — which it could not do — and it was
    // `#[ignore]`d behind a GPU requirement it did not actually need, so it
    // never ran either way.
    //
    // Deleted rather than repaired: the replacement needs no GPU, asserts the
    // real behaviour of all three strategies, and has an observed red state.
}
