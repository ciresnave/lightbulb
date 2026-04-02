use async_trait::async_trait;
use serde::{de::DeserializeOwned, Serialize};
use thiserror::Error;

/// Errors that can occur during storage operations
#[derive(Error, Debug)]
pub enum StorageError {
    /// The requested key was not found in the storage
    #[error("Key not found: {0}")]
    KeyNotFound(String),

    /// An error occurred during serialization or deserialization
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// An error occurred in the storage backend implementation
    #[error("Storage backend error: {0}")]
    BackendError(String),

    /// Invalid configuration was provided
    #[error("Invalid configuration: {0}")]
    ConfigError(String),
}

/// Result type for storage operations
pub type StorageResult<T> = Result<T, StorageError>;

/// Configuration options for storage backends
#[derive(Debug, Clone)]
pub enum StorageConfig {
    /// RocksDB storage configuration
    RocksDB {
        /// Path to the RocksDB database directory
        path: String,
        /// Maximum number of open files that can be used by the database
        max_open_files: i32,
        /// Whether to create a new database if one does not exist
        create_if_missing: bool,
    },
    /// SQLite storage configuration
    SQLite {
        /// Path to the SQLite database file
        path: String,
    },
    /// Sled storage configuration
    Sled {
        /// Path to the Sled database directory
        path: String,
        /// Maximum cache capacity in bytes
        cache_capacity: u64,
    },
    /// In-memory storage configuration (for testing)
    Memory,
}

/// Statistics about the storage backend
#[derive(Debug, Clone)]
pub struct StorageStats {
    /// Total number of keys in the storage
    pub total_keys: u64,
    /// Total size of stored data in bytes
    pub total_size: u64,
    /// Whether the storage is currently compacting
    pub is_compacting: bool,
}

/// Core trait that must be implemented by all storage backends.
///
/// This trait provides the foundational storage operations that all backends must support,
/// including raw byte operations, serialized value operations, transactions, and statistics.
/// Implementations must be thread-safe (Send + Sync) and support async operations.
#[async_trait]
pub trait StorageBackend: Send + Sync {
    /// Initialize the storage backend with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration specific to the storage backend type
    ///
    /// # Returns
    ///
    /// A new instance of the storage backend or an error if initialization fails
    #[must_use]
    async fn init(config: StorageConfig) -> StorageResult<Self>
    where
        Self: Sized;

    /// Get raw bytes by key
    #[must_use]
    async fn get_raw(&self, key: &str) -> StorageResult<Option<Vec<u8>>>;

    /// Set raw bytes with key
    async fn set_raw(&self, key: &str, value: &[u8]) -> StorageResult<()>;

    /// Get a value by key with serialization
    #[must_use]
    async fn get<V>(&self, key: &str) -> StorageResult<Option<V>>
    where
        V: DeserializeOwned + Send,
    {
        match self.get_raw(key).await? {
            Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            None => Ok(None),
        }
    }

    /// Set a value with key with serialization
    async fn set<V>(&self, key: &str, value: &V) -> StorageResult<()>
    where
        V: Serialize + Send + Sync,
    {
        let serialized = serde_json::to_vec(value)?;
        self.set_raw(key, &serialized).await
    }

    /// Delete a key
    async fn delete(&self, key: &str) -> StorageResult<()>;

    /// Check if a key exists
    #[must_use]
    async fn exists(&self, key: &str) -> StorageResult<bool>;

    /// List all keys with the given prefix
    #[must_use]
    async fn list_keys(&self, prefix: &str) -> StorageResult<Vec<String>>;

    /// Atomic compare and swap operation
    #[must_use]
    async fn compare_and_swap<V>(
        &self,
        key: &str,
        expected: Option<&V>,
        new: &V,
    ) -> StorageResult<bool>
    where
        V: Serialize + DeserializeOwned + PartialEq + Send + Sync,
    {
        let mut txn = self.begin_transaction().await?;

        let current_bytes = txn.get_raw(key).await?;
        let current: Option<V> = match current_bytes {
            Some(bytes) => Some(serde_json::from_slice(&bytes)?),
            None => None,
        };

        if current.as_ref() == expected {
            let new_bytes = serde_json::to_vec(new)?;
            txn.set_raw(key, &new_bytes).await?;
            txn.commit().await?;
            Ok(true)
        } else {
            txn.rollback().await?;
            Ok(false)
        }
    }

    /// Begin a new storage transaction.
    ///
    /// Transactions provide atomic operations across multiple keys. All operations
    /// within a transaction either succeed together or fail together. Transactions
    /// must be explicitly committed or rolled back.
    ///
    /// # Returns
    ///
    /// A new transaction object that implements StorageTransactionCore
    #[must_use]
    async fn begin_transaction(&self) -> StorageResult<Box<dyn StorageTransactionCore>>;

    /// Flush any pending writes to disk.
    ///
    /// This ensures that all previously committed data is persisted to stable storage.
    async fn flush(&self) -> StorageResult<()>;

    /// Get current storage statistics.
    ///
    /// Retrieves metrics about the storage backend including total keys, size, and
    /// compaction status.
    ///
    /// # Returns
    ///
    /// Statistics about the current state of the storage
    #[must_use]
    async fn stats(&self) -> StorageResult<StorageStats>;
}

/// Base transaction trait for non-generic operations.
///
/// This trait defines the core operations that all storage transactions must support.
/// It provides raw byte-level operations and transaction control (commit/rollback).
/// The trait is kept separate from StorageTransactionExt to allow for backend-specific
/// optimizations of the raw operations.
#[async_trait]
pub trait StorageTransactionCore: Send + Sync + 'static {
    /// Delete a key within the transaction
    async fn delete(&mut self, key: &str) -> StorageResult<()>;

    /// Get raw bytes within the transaction
    #[must_use]
    async fn get_raw(&self, key: &str) -> StorageResult<Option<Vec<u8>>>;

    /// Set raw bytes within the transaction
    async fn set_raw(&mut self, key: &str, value: &[u8]) -> StorageResult<()>;

    /// Commit the transaction
    async fn commit(self: Box<Self>) -> StorageResult<()>;

    /// Rollback the transaction
    async fn rollback(self: Box<Self>) -> StorageResult<()>;
}

/// Extension trait to add serialization support to transactions.
///
/// This trait provides convenience methods for working with serializable types
/// in transactions. It builds on top of StorageTransactionCore to add
/// automatic serialization/deserialization of values.
#[async_trait]
pub trait StorageTransactionExt: StorageTransactionCore {
    /// Set a value within the transaction
    async fn set<V>(&mut self, key: &str, value: &V) -> StorageResult<()>
    where
        V: Serialize + Send + Sync,
    {
        let serialized = serde_json::to_vec(value)?;
        self.set_raw(key, &serialized).await
    }

    /// Get a value within the transaction
    #[must_use]
    async fn get<V>(&self, key: &str) -> StorageResult<Option<V>>
    where
        V: DeserializeOwned + Send,
    {
        match self.get_raw(key).await? {
            Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            None => Ok(None),
        }
    }
}
