//! In-memory storage backend implementation
//!
//! This module provides an in-memory storage backend for testing and development purposes.
//! It provides fast, non-persistent storage using Hash Maps.

use super::backend::{
    StorageBackend, StorageConfig, StorageError, StorageResult, StorageStats,
    StorageTransactionCore, StorageTransactionExt,
};
use async_trait::async_trait;
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

/// In-memory storage backend implementation
#[derive(Clone)]
pub struct MemoryBackend {
    /// The in-memory data store
    data: Arc<RwLock<HashMap<String, Vec<u8>>>>,
}

/// Transaction implementation for the memory backend
pub struct MemoryTransaction {
    /// Reference to the main data store
    data: Arc<RwLock<HashMap<String, Vec<u8>>>>,
    /// Local transaction cache
    cache: HashMap<String, Option<Vec<u8>>>, // None means deleted
    /// Whether the transaction is committed
    committed: bool,
}

// Safety: Both RwLock and HashMap are Send + Sync when properly wrapped
unsafe impl Send for MemoryTransaction {}
unsafe impl Sync for MemoryTransaction {}

#[async_trait]
impl StorageTransactionCore for MemoryTransaction {
    async fn get_raw(&self, key: &str) -> StorageResult<Option<Vec<u8>>> {
        // Check transaction cache first
        if let Some(cached_value) = self.cache.get(key) {
            return Ok(cached_value.clone());
        }

        // Fall back to main data store
        let data = self
            .data
            .read()
            .map_err(|_| StorageError::BackendError("Lock poisoned".to_string()))?;
        Ok(data.get(key).cloned())
    }

    async fn set_raw(&mut self, key: &str, value: &[u8]) -> StorageResult<()> {
        if self.committed {
            return Err(StorageError::BackendError(
                "Transaction already committed".to_string(),
            ));
        }
        self.cache.insert(key.to_string(), Some(value.to_vec()));
        Ok(())
    }

    async fn delete(&mut self, key: &str) -> StorageResult<()> {
        if self.committed {
            return Err(StorageError::BackendError(
                "Transaction already committed".to_string(),
            ));
        }
        self.cache.insert(key.to_string(), None);
        Ok(())
    }

    async fn commit(mut self: Box<Self>) -> StorageResult<()> {
        if self.committed {
            return Err(StorageError::BackendError(
                "Transaction already committed".to_string(),
            ));
        }

        let mut data = self
            .data
            .write()
            .map_err(|_| StorageError::BackendError("Lock poisoned".to_string()))?;

        // Apply all changes from the transaction cache
        for (key, value) in self.cache.drain() {
            match value {
                Some(v) => {
                    data.insert(key, v);
                }
                None => {
                    data.remove(&key);
                }
            }
        }

        self.committed = true;
        Ok(())
    }

    async fn rollback(mut self: Box<Self>) -> StorageResult<()> {
        self.cache.clear();
        self.committed = true;
        Ok(())
    }
}

// Automatically implement the extension trait
impl StorageTransactionExt for MemoryTransaction {}

#[async_trait]
impl StorageBackend for MemoryBackend {
    async fn init(_config: StorageConfig) -> StorageResult<Self> {
        Ok(Self {
            data: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    async fn get_raw(&self, key: &str) -> StorageResult<Option<Vec<u8>>> {
        let data = self
            .data
            .read()
            .map_err(|_| StorageError::BackendError("Lock poisoned".to_string()))?;
        Ok(data.get(key).cloned())
    }

    async fn set_raw(&self, key: &str, value: &[u8]) -> StorageResult<()> {
        let mut data = self
            .data
            .write()
            .map_err(|_| StorageError::BackendError("Lock poisoned".to_string()))?;
        data.insert(key.to_string(), value.to_vec());
        Ok(())
    }

    async fn delete(&self, key: &str) -> StorageResult<()> {
        let mut data = self
            .data
            .write()
            .map_err(|_| StorageError::BackendError("Lock poisoned".to_string()))?;
        data.remove(key);
        Ok(())
    }

    async fn exists(&self, key: &str) -> StorageResult<bool> {
        let data = self
            .data
            .read()
            .map_err(|_| StorageError::BackendError("Lock poisoned".to_string()))?;
        Ok(data.contains_key(key))
    }

    async fn list_keys(&self, prefix: &str) -> StorageResult<Vec<String>> {
        let data = self
            .data
            .read()
            .map_err(|_| StorageError::BackendError("Lock poisoned".to_string()))?;

        let mut keys: Vec<String> = data
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();

        keys.sort(); // Provide consistent ordering
        Ok(keys)
    }

    async fn begin_transaction(&self) -> StorageResult<Box<dyn StorageTransactionCore>> {
        Ok(Box::new(MemoryTransaction {
            data: self.data.clone(),
            cache: HashMap::new(),
            committed: false,
        }))
    }

    async fn stats(&self) -> StorageResult<StorageStats> {
        let data = self
            .data
            .read()
            .map_err(|_| StorageError::BackendError("Lock poisoned".to_string()))?;

        let total_keys = data.len() as u64;
        let total_size = data.iter().map(|(k, v)| k.len() + v.len()).sum::<usize>() as u64;

        Ok(StorageStats {
            total_keys,
            total_size,
            is_compacting: false, // Memory backend doesn't compact
        })
    }

    async fn flush(&self) -> StorageResult<()> {
        // Nothing to flush for memory backend
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct TestData {
        id: u32,
        name: String,
    }

    #[tokio::test]
    async fn test_memory_backend_basic_operations() {
        let backend = MemoryBackend::init(StorageConfig::SQLite {
            path: ":memory:".to_string(),
        })
        .await
        .unwrap();

        // Test set and get
        backend.set_raw("key1", b"value1").await.unwrap();
        let value = backend.get_raw("key1").await.unwrap();
        assert_eq!(value, Some(b"value1".to_vec()));

        // Test exists
        assert!(backend.exists("key1").await.unwrap());
        assert!(!backend.exists("nonexistent").await.unwrap());

        // Test delete
        backend.delete("key1").await.unwrap();
        let value = backend.get_raw("key1").await.unwrap();
        assert_eq!(value, None);
    }

    #[tokio::test]
    async fn test_memory_backend_serialization() {
        let backend = MemoryBackend::init(StorageConfig::SQLite {
            path: ":memory:".to_string(),
        })
        .await
        .unwrap();

        let test_data = TestData {
            id: 42,
            name: "Test".to_string(),
        };

        // Test serialized set and get
        backend.set("test_key", &test_data).await.unwrap();
        let retrieved: Option<TestData> = backend.get("test_key").await.unwrap();
        assert_eq!(retrieved, Some(test_data));
    }

    #[tokio::test]
    async fn test_memory_backend_list_keys() {
        let backend = MemoryBackend::init(StorageConfig::SQLite {
            path: ":memory:".to_string(),
        })
        .await
        .unwrap();

        // Set multiple keys
        backend.set_raw("prefix:key1", b"value1").await.unwrap();
        backend.set_raw("prefix:key2", b"value2").await.unwrap();
        backend.set_raw("other:key3", b"value3").await.unwrap();

        // List keys with prefix
        let keys = backend.list_keys("prefix:").await.unwrap();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"prefix:key1".to_string()));
        assert!(keys.contains(&"prefix:key2".to_string()));
    }

    #[tokio::test]
    async fn test_memory_backend_transaction() {
        let backend = MemoryBackend::init(StorageConfig::SQLite {
            path: ":memory:".to_string(),
        })
        .await
        .unwrap();

        // Set initial value
        backend.set_raw("key1", b"initial").await.unwrap();

        // Start transaction and modify
        let mut txn = backend.begin_transaction().await.unwrap();
        txn.set_raw("key1", b"modified").await.unwrap();
        txn.set_raw("key2", b"new").await.unwrap();

        // Values should not be visible outside transaction yet
        assert_eq!(
            backend.get_raw("key1").await.unwrap(),
            Some(b"initial".to_vec())
        );
        assert_eq!(backend.get_raw("key2").await.unwrap(), None);

        // But should be visible within transaction
        assert_eq!(
            txn.get_raw("key1").await.unwrap(),
            Some(b"modified".to_vec())
        );
        assert_eq!(txn.get_raw("key2").await.unwrap(), Some(b"new".to_vec()));

        // Commit transaction
        txn.commit().await.unwrap();

        // Now values should be visible
        assert_eq!(
            backend.get_raw("key1").await.unwrap(),
            Some(b"modified".to_vec())
        );
        assert_eq!(
            backend.get_raw("key2").await.unwrap(),
            Some(b"new".to_vec())
        );
    }

    #[tokio::test]
    async fn test_memory_backend_transaction_rollback() {
        let backend = MemoryBackend::init(StorageConfig::SQLite {
            path: ":memory:".to_string(),
        })
        .await
        .unwrap();

        // Set initial value
        backend.set_raw("key1", b"initial").await.unwrap();

        // Start transaction and modify
        let mut txn = backend.begin_transaction().await.unwrap();
        txn.set_raw("key1", b"modified").await.unwrap();

        // Rollback transaction
        txn.rollback().await.unwrap();

        // Original value should be preserved
        assert_eq!(
            backend.get_raw("key1").await.unwrap(),
            Some(b"initial".to_vec())
        );
    }

    #[tokio::test]
    async fn test_memory_backend_stats() {
        let backend = MemoryBackend::init(StorageConfig::SQLite {
            path: ":memory:".to_string(),
        })
        .await
        .unwrap();

        // Initially empty
        let stats = backend.stats().await.unwrap();
        assert_eq!(stats.total_keys, 0);
        assert_eq!(stats.total_size, 0);

        // Add some data
        backend.set_raw("key1", b"value1").await.unwrap();
        backend.set_raw("key2", b"value2").await.unwrap();

        let stats = backend.stats().await.unwrap();
        assert_eq!(stats.total_keys, 2);
        assert!(stats.total_size > 0);
    }
}
