//! Sled database backend implementation
//!
//! This module provides a storage backend implementation using the sled embedded database.

use super::backend::{
    StorageBackend, StorageConfig, StorageError, StorageResult, StorageStats,
    StorageTransactionCore, StorageTransactionExt,
};
use async_trait::async_trait;
use sled::{Batch, Db};
use std::sync::Arc;

/// Sled-based storage backend implementation
#[derive(Clone)]
pub struct SledBackend {
    db: Arc<Db>,
}

/// Transaction implementation for the Sled database
pub struct SledTransaction {
    db: Arc<Db>,
    batch: Batch,
}

// Safety: Sled's Db is already Send + Sync
unsafe impl Send for SledTransaction {}
unsafe impl Sync for SledTransaction {}

#[async_trait]
impl StorageTransactionCore for SledTransaction {
    async fn get_raw(&self, key: &str) -> StorageResult<Option<Vec<u8>>> {
        self.db
            .get(key.as_bytes())
            .map(|opt| opt.map(|iv| iv.to_vec()))
            .map_err(|e| StorageError::BackendError(e.to_string()))
    }

    async fn set_raw(&mut self, key: &str, value: &[u8]) -> StorageResult<()> {
        self.batch.insert(key.as_bytes(), value);
        Ok(())
    }

    async fn delete(&mut self, key: &str) -> StorageResult<()> {
        self.batch.remove(key.as_bytes());
        Ok(())
    }

    async fn commit(self: Box<Self>) -> StorageResult<()> {
        self.db
            .apply_batch(self.batch)
            .map_err(|e| StorageError::BackendError(e.to_string()))?;
        self.db
            .flush()
            .map_err(|e| StorageError::BackendError(e.to_string()))?;
        Ok(())
    }

    async fn rollback(self: Box<Self>) -> StorageResult<()> {
        // Sled's batch is discarded automatically when dropped
        Ok(())
    }
}

// Automatically implement the extension trait
impl StorageTransactionExt for SledTransaction {}

#[async_trait]
impl StorageBackend for SledBackend {
    async fn init(config: StorageConfig) -> StorageResult<Self> {
        match config {
            StorageConfig::Sled {
                path,
                cache_capacity,
            } => {
                let db = sled::Config::new()
                    .path(path)
                    .cache_capacity(cache_capacity)
                    .flush_every_ms(Some(1000))
                    .open()
                    .map_err(|e| StorageError::BackendError(e.to_string()))?;

                Ok(Self { db: Arc::new(db) })
            }
            _ => Err(StorageError::ConfigError("Expected Sled config".into())),
        }
    }

    async fn get_raw(&self, key: &str) -> StorageResult<Option<Vec<u8>>> {
        self.db
            .get(key.as_bytes())
            .map(|opt| opt.map(|iv| iv.to_vec()))
            .map_err(|e| StorageError::BackendError(e.to_string()))
    }

    async fn set_raw(&self, key: &str, value: &[u8]) -> StorageResult<()> {
        self.db
            .insert(key.as_bytes(), value)
            .map(|_| ())
            .map_err(|e| StorageError::BackendError(e.to_string()))
    }

    async fn delete(&self, key: &str) -> StorageResult<()> {
        self.db
            .remove(key.as_bytes())
            .map(|_| ())
            .map_err(|e| StorageError::BackendError(e.to_string()))
    }

    async fn exists(&self, key: &str) -> StorageResult<bool> {
        self.db
            .contains_key(key.as_bytes())
            .map_err(|e| StorageError::BackendError(e.to_string()))
    }

    async fn list_keys(&self, prefix: &str) -> StorageResult<Vec<String>> {
        let mut keys = Vec::new();
        for key_result in self.db.scan_prefix(prefix.as_bytes()) {
            let (key, _) = key_result.map_err(|e| StorageError::BackendError(e.to_string()))?;
            if let Ok(key_str) = String::from_utf8(key.to_vec()) {
                keys.push(key_str);
            }
        }
        Ok(keys)
    }

    async fn begin_transaction(&self) -> StorageResult<Box<dyn StorageTransactionCore>> {
        Ok(Box::new(SledTransaction {
            db: self.db.clone(),
            batch: Batch::default(),
        }))
    }

    async fn stats(&self) -> StorageResult<StorageStats> {
        // Use Sled's iterator to count keys and sum up value sizes
        let mut total_keys = 0u64;
        let mut total_size = 0u64;

        for item in self.db.iter() {
            let (key, value) = item.map_err(|e| StorageError::BackendError(e.to_string()))?;
            total_keys += 1;
            total_size += value.len() as u64;
            total_size += key.len() as u64; // Include key sizes in total
        }

        // Get compaction status - Sled does this automatically, but we can check if there are any pending operations
        let is_compacting = false; // Sled doesn't expose compaction status directly

        Ok(StorageStats {
            total_keys,
            total_size,
            is_compacting,
        })
    }

    async fn flush(&self) -> StorageResult<()> {
        self.db
            .flush()
            .map(|_| ()) // Convert usize to unit
            .map_err(|e| StorageError::BackendError(e.to_string()))
    }
}
