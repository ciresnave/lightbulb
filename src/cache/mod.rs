//! KV Cache implementations for parallel batching
//!
//! Forked from Candle's ScatteredCacheBuilder with modifications for:
//! - Independent per-slot position tracking (no global increment)
//! - Slot clearing capability
//! - Efficient position setting (O(1) instead of O(n))

pub mod parallel_cache_builder;
pub mod prefix_cache;

pub use parallel_cache_builder::{IndicesAndMask, ParallelCacheBuilder, ParallelKvCache};
pub use prefix_cache::{PrefixCacheConfig, PrefixCacheStats, PrefixKvCache, PrefixKvEntry};
