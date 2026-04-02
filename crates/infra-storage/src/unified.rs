//! Unified backend enum for testing and runtime backend selection
//!
//! This module provides a unified interface for all storage backends through an enum
//! that implements StorageBackend by delegating to the appropriate backend implementation.

use crate::backend::*;
use crate::{MemoryBackend, RocksDBBackend, SQLiteBackend, SledBackend};
use async_trait::async_trait;

/// Unified backend enum that wraps all supported backend types
pub enum UnifiedBackend {
    /// Memory backend variant
    Memory(MemoryBackend),
    /// RocksDB backend variant
    RocksDB(RocksDBBackend),
    /// SQLite backend variant
    SQLite(SQLiteBackend),
    /// Sled backend variant
    #[cfg(feature = "sled-backend")]
    Sled(SledBackend),
}

/// Unified transaction enum that wraps all transaction types
pub enum UnifiedTransaction {
    /// Memory transaction variant
    Memory(Box<dyn StorageTransactionCore>),
    /// RocksDB transaction variant
    RocksDB(Box<dyn StorageTransactionCore>),
    /// SQLite transaction variant
    SQLite(Box<dyn StorageTransactionCore>),
    /// Sled transaction variant
    #[cfg(feature = "sled-backend")]
    Sled(Box<dyn StorageTransactionCore>),
}

#[async_trait]
impl StorageTransactionCore for UnifiedTransaction {
    async fn get_raw(&self, key: &str) -> StorageResult<Option<Vec<u8>>> {
        match self {
            UnifiedTransaction::Memory(txn) => txn.get_raw(key).await,
            UnifiedTransaction::RocksDB(txn) => txn.get_raw(key).await,
            UnifiedTransaction::SQLite(txn) => txn.get_raw(key).await,
            #[cfg(feature = "sled-backend")]
            UnifiedTransaction::Sled(txn) => txn.get_raw(key).await,
        }
    }

    async fn set_raw(&mut self, key: &str, value: &[u8]) -> StorageResult<()> {
        match self {
            UnifiedTransaction::Memory(txn) => txn.set_raw(key, value).await,
            UnifiedTransaction::RocksDB(txn) => txn.set_raw(key, value).await,
            UnifiedTransaction::SQLite(txn) => txn.set_raw(key, value).await,
            #[cfg(feature = "sled-backend")]
            UnifiedTransaction::Sled(txn) => txn.set_raw(key, value).await,
        }
    }

    async fn delete(&mut self, key: &str) -> StorageResult<()> {
        match self {
            UnifiedTransaction::Memory(txn) => txn.delete(key).await,
            UnifiedTransaction::RocksDB(txn) => txn.delete(key).await,
            UnifiedTransaction::SQLite(txn) => txn.delete(key).await,
            #[cfg(feature = "sled-backend")]
            UnifiedTransaction::Sled(txn) => txn.delete(key).await,
        }
    }

    async fn commit(self: Box<Self>) -> StorageResult<()> {
        match *self {
            UnifiedTransaction::Memory(txn) => txn.commit().await,
            UnifiedTransaction::RocksDB(txn) => txn.commit().await,
            UnifiedTransaction::SQLite(txn) => txn.commit().await,
            #[cfg(feature = "sled-backend")]
            UnifiedTransaction::Sled(txn) => txn.commit().await,
        }
    }

    async fn rollback(self: Box<Self>) -> StorageResult<()> {
        match *self {
            UnifiedTransaction::Memory(txn) => txn.rollback().await,
            UnifiedTransaction::RocksDB(txn) => txn.rollback().await,
            UnifiedTransaction::SQLite(txn) => txn.rollback().await,
            #[cfg(feature = "sled-backend")]
            UnifiedTransaction::Sled(txn) => txn.rollback().await,
        }
    }
}

impl StorageTransactionExt for UnifiedTransaction {}

#[async_trait]
impl StorageBackend for UnifiedBackend {
    async fn init(config: StorageConfig) -> StorageResult<Self> {
        match config {
            StorageConfig::Memory => {
                let backend = MemoryBackend::init(config).await?;
                Ok(UnifiedBackend::Memory(backend))
            }
            StorageConfig::RocksDB { .. } => {
                let backend = RocksDBBackend::init(config).await?;
                Ok(UnifiedBackend::RocksDB(backend))
            }
            StorageConfig::SQLite { .. } => {
                let backend = SQLiteBackend::init(config).await?;
                Ok(UnifiedBackend::SQLite(backend))
            }
            #[cfg(feature = "sled-backend")]
            StorageConfig::Sled { .. } => {
                let backend = SledBackend::init(config).await?;
                Ok(UnifiedBackend::Sled(backend))
            }
            #[cfg(not(feature = "sled-backend"))]
            StorageConfig::Sled { .. } => Err(StorageError::ConfigError(
                "Sled backend not enabled".to_string(),
            )),
        }
    }

    async fn get_raw(&self, key: &str) -> StorageResult<Option<Vec<u8>>> {
        match self {
            UnifiedBackend::Memory(backend) => backend.get_raw(key).await,
            UnifiedBackend::RocksDB(backend) => backend.get_raw(key).await,
            UnifiedBackend::SQLite(backend) => backend.get_raw(key).await,
            #[cfg(feature = "sled-backend")]
            UnifiedBackend::Sled(backend) => backend.get_raw(key).await,
        }
    }

    async fn set_raw(&self, key: &str, value: &[u8]) -> StorageResult<()> {
        match self {
            UnifiedBackend::Memory(backend) => backend.set_raw(key, value).await,
            UnifiedBackend::RocksDB(backend) => backend.set_raw(key, value).await,
            UnifiedBackend::SQLite(backend) => backend.set_raw(key, value).await,
            #[cfg(feature = "sled-backend")]
            UnifiedBackend::Sled(backend) => backend.set_raw(key, value).await,
        }
    }

    async fn delete(&self, key: &str) -> StorageResult<()> {
        match self {
            UnifiedBackend::Memory(backend) => backend.delete(key).await,
            UnifiedBackend::RocksDB(backend) => backend.delete(key).await,
            UnifiedBackend::SQLite(backend) => backend.delete(key).await,
            #[cfg(feature = "sled-backend")]
            UnifiedBackend::Sled(backend) => backend.delete(key).await,
        }
    }

    async fn exists(&self, key: &str) -> StorageResult<bool> {
        match self {
            UnifiedBackend::Memory(backend) => backend.exists(key).await,
            UnifiedBackend::RocksDB(backend) => backend.exists(key).await,
            UnifiedBackend::SQLite(backend) => backend.exists(key).await,
            #[cfg(feature = "sled-backend")]
            UnifiedBackend::Sled(backend) => backend.exists(key).await,
        }
    }

    async fn list_keys(&self, prefix: &str) -> StorageResult<Vec<String>> {
        match self {
            UnifiedBackend::Memory(backend) => backend.list_keys(prefix).await,
            UnifiedBackend::RocksDB(backend) => backend.list_keys(prefix).await,
            UnifiedBackend::SQLite(backend) => backend.list_keys(prefix).await,
            #[cfg(feature = "sled-backend")]
            UnifiedBackend::Sled(backend) => backend.list_keys(prefix).await,
        }
    }

    async fn begin_transaction(&self) -> StorageResult<Box<dyn StorageTransactionCore>> {
        match self {
            UnifiedBackend::Memory(backend) => {
                let txn = backend.begin_transaction().await?;
                Ok(Box::new(UnifiedTransaction::Memory(txn)))
            }
            UnifiedBackend::RocksDB(backend) => {
                let txn = backend.begin_transaction().await?;
                Ok(Box::new(UnifiedTransaction::RocksDB(txn)))
            }
            UnifiedBackend::SQLite(backend) => {
                let txn = backend.begin_transaction().await?;
                Ok(Box::new(UnifiedTransaction::SQLite(txn)))
            }
            #[cfg(feature = "sled-backend")]
            UnifiedBackend::Sled(backend) => {
                let txn = backend.begin_transaction().await?;
                Ok(Box::new(UnifiedTransaction::Sled(txn)))
            }
        }
    }

    async fn flush(&self) -> StorageResult<()> {
        match self {
            UnifiedBackend::Memory(backend) => backend.flush().await,
            UnifiedBackend::RocksDB(backend) => backend.flush().await,
            UnifiedBackend::SQLite(backend) => backend.flush().await,
            #[cfg(feature = "sled-backend")]
            UnifiedBackend::Sled(backend) => backend.flush().await,
        }
    }

    async fn stats(&self) -> StorageResult<StorageStats> {
        match self {
            UnifiedBackend::Memory(backend) => backend.stats().await,
            UnifiedBackend::RocksDB(backend) => backend.stats().await,
            UnifiedBackend::SQLite(backend) => backend.stats().await,
            #[cfg(feature = "sled-backend")]
            UnifiedBackend::Sled(backend) => backend.stats().await,
        }
    }
}
