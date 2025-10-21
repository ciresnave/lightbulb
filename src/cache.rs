//! Paged KV cache abstractions

pub struct KvCacheConfig {
    pub block_size: usize,
}

pub struct KvCache;

impl KvCache {
    pub fn new(_cfg: KvCacheConfig) -> Self {
        Self
    }
}
