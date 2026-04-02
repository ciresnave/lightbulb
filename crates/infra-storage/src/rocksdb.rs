use super::backend::{
    StorageBackend, StorageConfig, StorageError, StorageResult, StorageStats,
    StorageTransactionCore, StorageTransactionExt,
};
use async_trait::async_trait;
use rocksdb::{WriteBatch, WriteOptions, DB};
use std::sync::Arc;

/// RocksDB-based storage backend implementation
pub struct RocksDBBackend {
    /// The underlying RocksDB instance
    db: Arc<DB>,
}

/// Transaction implementation for RocksDB using write batches
pub struct RocksDBTransaction {
    /// Reference to the database for transaction scope
    db: Arc<DB>,
    /// The write batch for transaction operations
    batch: WriteBatch,
}

// Safety: RocksDB's components are thread safe
unsafe impl Send for RocksDBTransaction {}
unsafe impl Sync for RocksDBTransaction {}

#[async_trait]
impl StorageTransactionCore for RocksDBTransaction {
    async fn get_raw(&self, key: &str) -> StorageResult<Option<Vec<u8>>> {
        self.db
            .get(key.as_bytes())
            .map(|opt| opt.map(|v| v.to_vec()))
            .map_err(|e| StorageError::BackendError(e.to_string()))
    }

    async fn set_raw(&mut self, key: &str, value: &[u8]) -> StorageResult<()> {
        self.batch.put(key.as_bytes(), value);
        Ok(())
    }

    async fn delete(&mut self, key: &str) -> StorageResult<()> {
        self.batch.delete(key.as_bytes());
        Ok(())
    }

    async fn commit(self: Box<Self>) -> StorageResult<()> {
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(true);
        self.db
            .write_opt(self.batch, &write_opts)
            .map_err(|e| StorageError::BackendError(e.to_string()))?;
        Ok(())
    }

    async fn rollback(self: Box<Self>) -> StorageResult<()> {
        // For RocksDB, rollback just means dropping the batch without committing
        Ok(())
    }
}

// Automatically implement the extension trait
impl StorageTransactionExt for RocksDBTransaction {}

#[async_trait]
impl StorageBackend for RocksDBBackend {
    async fn init(config: StorageConfig) -> StorageResult<Self> {
        match config {
            StorageConfig::RocksDB {
                path,
                max_open_files,
                create_if_missing,
            } => {
                let mut opts = rocksdb::Options::default();
                opts.set_max_open_files(max_open_files);
                opts.create_if_missing(create_if_missing);
                opts.set_use_fsync(true); // Ensure durability

                let db = rocksdb::DB::open(&opts, &path)
                    .map_err(|e| StorageError::BackendError(e.to_string()))?;

                Ok(Self { db: Arc::new(db) })
            }
            _ => Err(StorageError::ConfigError("Expected RocksDB config".into())),
        }
    }

    async fn get_raw(&self, key: &str) -> StorageResult<Option<Vec<u8>>> {
        self.db
            .get(key.as_bytes())
            .map(|opt| opt.map(|v| v.to_vec()))
            .map_err(|e| StorageError::BackendError(e.to_string()))
    }

    async fn set_raw(&self, key: &str, value: &[u8]) -> StorageResult<()> {
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(true);
        self.db
            .put_opt(key.as_bytes(), value, &write_opts)
            .map_err(|e| StorageError::BackendError(e.to_string()))
    }

    async fn delete(&self, key: &str) -> StorageResult<()> {
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(true);
        self.db
            .delete_opt(key.as_bytes(), &write_opts)
            .map_err(|e| StorageError::BackendError(e.to_string()))
    }
    async fn begin_transaction(&self) -> StorageResult<Box<dyn StorageTransactionCore>> {
        Ok(Box::new(RocksDBTransaction {
            db: self.db.clone(),
            batch: WriteBatch::default(),
        }))
    }

    async fn exists(&self, key: &str) -> StorageResult<bool> {
        self.get_raw(key).await.map(|opt| opt.is_some())
    }

    async fn list_keys(&self, prefix: &str) -> StorageResult<Vec<String>> {
        let mut iterator = self.db.iterator(rocksdb::IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));

        let mut keys = Vec::new();
        while let Some(Ok((key, _))) = iterator.next() {
            if let Ok(key_str) = String::from_utf8(key.to_vec()) {
                if key_str.starts_with(prefix) {
                    keys.push(key_str);
                } else {
                    break; // We've moved past the prefix
                }
            }
        }

        Ok(keys)
    }

    async fn flush(&self) -> StorageResult<()> {
        self.db
            .flush()
            .map_err(|e| StorageError::BackendError(e.to_string()))
    }

    async fn stats(&self) -> StorageResult<StorageStats> {
        let mut total_keys = 0u64;
        let mut total_size = 0u64;

        // Get all key-value pairs to count and calculate size
        let iterator = self.db.iterator(rocksdb::IteratorMode::Start);
        for item in iterator {
            let (key, value) = item.map_err(|e| StorageError::BackendError(e.to_string()))?;
            total_keys += 1;
            total_size += key.len() as u64; // Include key size
            total_size += value.len() as u64; // Include value size
        }

        // Get compaction status indirectly through file info
        let is_compacting = match self.db.live_files() {
            Ok(files) => {
                // If any files are at level 0, compaction might be needed/ongoing
                let l0_files = files.iter().filter(|f| f.level == 0).count();
                // Assume compaction is happening if we have multiple L0 files
                l0_files > 4 // RocksDB typically triggers compaction at 4 L0 files
            }
            Err(_) => false,
        };

        Ok(StorageStats {
            total_keys,
            total_size,
            is_compacting,
        })
    }
}
