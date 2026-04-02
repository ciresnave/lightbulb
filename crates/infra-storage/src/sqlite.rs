use crate::backend::{
    StorageBackend, StorageConfig, StorageError, StorageResult, StorageStats,
    StorageTransactionCore,
};
use async_trait::async_trait;
use libsql::{params, Builder, Connection, Database, Transaction};

use std::sync::Arc;

/// SQLite-based storage backend implementation using libsql for database operations.
///
/// This backend provides:
/// - ACID compliant transactions
/// - Automatic write-ahead logging (WAL)
/// - SQL-based prefix scanning
/// - Efficient blob storage
///
/// The implementation uses a single table 'kv_store' with a key-value schema:
/// - key: TEXT PRIMARY KEY
/// - value: BLOB
pub struct SQLiteBackend {
    db: Arc<Database>,
    conn: Connection,
}

// Manual Clone implementation since we need to handle Arc wrapping
impl Clone for SQLiteBackend {
    fn clone(&self) -> Self {
        Self {
            db: self.db.clone(),
            conn: self.conn.clone(),
        }
    }
}

/// Active transaction for the SQLite database.
///
/// Provides atomic operations and implements the StorageTransactionCore trait.
/// All operations within a transaction are rolled back if the transaction is
/// dropped without being committed.
pub struct SQLiteTransaction {
    txn: Transaction,
}

#[async_trait]
impl StorageBackend for SQLiteBackend {
    async fn init(config: StorageConfig) -> StorageResult<Self> {
        let (db, conn) = match config {
            StorageConfig::SQLite { path } => {
                // Ensure parent directory exists
                if let Some(parent) = std::path::Path::new(&path).parent() {
                    std::fs::create_dir_all(parent).map_err(|e| {
                        StorageError::BackendError(format!("Failed to create directory: {e}"))
                    })?;
                }

                let db: libsql::Database =
                    Builder::new_local(&path).build().await.map_err(|e| {
                        StorageError::BackendError(format!("Failed to open database: {e}"))
                    })?;

                let db = Arc::new(db);

                let conn = db.connect().map_err(|e| {
                    StorageError::BackendError(format!("Failed to create connection: {e}"))
                })?;

                (db, conn)
            }
            _ => {
                return Err(StorageError::BackendError(
                    "Expected SQLite configuration".to_string(),
                ))
            }
        };

        // Create the key-value table if it doesn't exist
        conn.execute(
            "CREATE TABLE IF NOT EXISTS kv_store (
                key TEXT PRIMARY KEY,
                value BLOB NOT NULL
            )",
            params![],
        )
        .await
        .map_err(|e| StorageError::BackendError(format!("Failed to create table: {e}")))?;

        Ok(Self { db, conn })
    }

    async fn begin_transaction(&self) -> StorageResult<Box<dyn StorageTransactionCore>> {
        let txn =
            self.conn.transaction().await.map_err(|e| {
                StorageError::BackendError(format!("Failed to begin transaction: {e}"))
            })?;

        Ok(Box::new(SQLiteTransaction { txn }))
    }

    async fn get_raw(&self, key: &str) -> StorageResult<Option<Vec<u8>>> {
        let mut stmt = self
            .conn
            .prepare("SELECT value FROM kv_store WHERE key = ?")
            .await
            .map_err(|e| StorageError::BackendError(format!("Failed to prepare query: {e}")))?;

        let mut rows = stmt
            .query([key])
            .await
            .map_err(|e| StorageError::BackendError(format!("Failed to execute query: {e}")))?;

        match rows.next().await {
            Ok(Some(row)) => {
                let value: Vec<u8> = row
                    .get(0)
                    .map_err(|e| StorageError::BackendError(e.to_string()))?;
                Ok(Some(value))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(StorageError::BackendError(e.to_string())),
        }
    }
    async fn stats(&self) -> StorageResult<StorageStats> {
        // Get total key count and total size in one query
        let mut stmt = self
            .conn
            .prepare("SELECT COUNT(*), COALESCE(SUM(LENGTH(value)), 0) FROM kv_store")
            .await
            .map_err(|e| {
                StorageError::BackendError(format!("Failed to prepare stats query: {e}"))
            })?;

        let mut rows = stmt.query(params![]).await.map_err(|e| {
            StorageError::BackendError(format!("Failed to execute stats query: {e}"))
        })?;

        let (count, total_size) = match rows.next().await {
            Ok(Some(row)) => {
                let count: i64 = row
                    .get(0)
                    .map_err(|e| StorageError::BackendError(e.to_string()))?;
                let size: i64 = row
                    .get(1)
                    .map_err(|e| StorageError::BackendError(e.to_string()))?;
                (count as u64, size as u64)
            }
            Ok(None) => (0, 0),
            Err(e) => return Err(StorageError::BackendError(e.to_string())),
        };

        Ok(StorageStats {
            total_keys: count,
            total_size,
            is_compacting: false, // SQLite handles this automatically
        })
    }

    async fn set_raw(&self, key: &str, value: &[u8]) -> StorageResult<()> {
        let mut stmt = self
            .conn
            .prepare("INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)")
            .await
            .map_err(|e| StorageError::BackendError(format!("Failed to prepare statement: {e}")))?;

        stmt.execute(params![key, value])
            .await
            .map_err(|e| StorageError::BackendError(format!("Failed to execute statement: {e}")))?;

        Ok(())
    }

    async fn delete(&self, key: &str) -> StorageResult<()> {
        let mut stmt = self
            .conn
            .prepare("DELETE FROM kv_store WHERE key = ?")
            .await
            .map_err(|e| StorageError::BackendError(format!("Failed to prepare statement: {e}")))?;

        stmt.execute(params![key])
            .await
            .map_err(|e| StorageError::BackendError(format!("Failed to execute statement: {e}")))?;

        Ok(())
    }

    async fn exists(&self, key: &str) -> StorageResult<bool> {
        let mut stmt = self
            .conn
            .prepare("SELECT 1 FROM kv_store WHERE key = ? LIMIT 1")
            .await
            .map_err(|e| StorageError::BackendError(format!("Failed to prepare statement: {e}")))?;

        let mut rows = stmt
            .query(params![key])
            .await
            .map_err(|e| StorageError::BackendError(format!("Failed to execute statement: {e}")))?;

        Ok(rows
            .next()
            .await
            .map_err(|e| StorageError::BackendError(e.to_string()))?
            .is_some())
    }

    async fn list_keys(&self, prefix: &str) -> StorageResult<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT key FROM kv_store WHERE key LIKE ? || '%'")
            .await
            .map_err(|e| StorageError::BackendError(e.to_string()))?;

        let mut keys = Vec::new();
        let mut rows = stmt
            .query(params![prefix])
            .await
            .map_err(|e| StorageError::BackendError(e.to_string()))?;

        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| StorageError::BackendError(e.to_string()))?
        {
            let key: String = row
                .get(0)
                .map_err(|e| StorageError::BackendError(e.to_string()))?;
            keys.push(key);
        }

        Ok(keys)
    }
    async fn flush(&self) -> StorageResult<()> {
        // Get the WAL checkpoint status
        let mut stmt = self
            .conn
            .prepare("PRAGMA wal_checkpoint(TRUNCATE)")
            .await
            .map_err(|e| {
                StorageError::BackendError(format!("Failed to prepare checkpoint query: {e}"))
            })?;

        let mut rows = stmt.query(params![]).await.map_err(|e| {
            StorageError::BackendError(format!("Failed to execute checkpoint query: {e}"))
        })?;

        // Consume the status row but don't error if no rows (which can happen)
        let _ = rows.next().await;

        Ok(())
    }
}

#[async_trait]
impl StorageTransactionCore for SQLiteTransaction {
    async fn get_raw(&self, key: &str) -> StorageResult<Option<Vec<u8>>> {
        let mut stmt = self
            .txn
            .prepare("SELECT value FROM kv_store WHERE key = ?")
            .await
            .map_err(|e| StorageError::BackendError(e.to_string()))?;

        let mut rows = stmt
            .query([key])
            .await
            .map_err(|e| StorageError::BackendError(e.to_string()))?;

        match rows.next().await {
            Ok(Some(row)) => {
                let value: Vec<u8> = row
                    .get(0)
                    .map_err(|e| StorageError::BackendError(e.to_string()))?;
                Ok(Some(value))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(StorageError::BackendError(e.to_string())),
        }
    }

    async fn set_raw(&mut self, key: &str, value: &[u8]) -> StorageResult<()> {
        self.txn
            .execute(
                "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
                params![key, value],
            )
            .await
            .map_err(|e| StorageError::BackendError(e.to_string()))?;

        Ok(())
    }

    async fn delete(&mut self, key: &str) -> StorageResult<()> {
        self.txn
            .execute("DELETE FROM kv_store WHERE key = ?", params![key])
            .await
            .map_err(|e| StorageError::BackendError(e.to_string()))?;

        Ok(())
    }

    async fn commit(self: Box<Self>) -> StorageResult<()> {
        self.txn
            .commit()
            .await
            .map_err(|e| StorageError::BackendError(e.to_string()))?;
        Ok(())
    }

    async fn rollback(self: Box<Self>) -> StorageResult<()> {
        self.txn
            .rollback()
            .await
            .map_err(|e| StorageError::BackendError(e.to_string()))?;
        Ok(())
    }
}
