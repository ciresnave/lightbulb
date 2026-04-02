//! Integration tests for all storage backends
//!
//! This module provides comprehensive tests that run against all supported storage backends
//! to ensure consistent behavior across different implementations.

use infra_storage::*;
use pretty_assertions::assert_eq;
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct TestData {
    id: u32,
    name: String,
    tags: Vec<String>,
}

/// Test configuration for different backends
async fn get_test_configs() -> Vec<(&'static str, StorageConfig, Option<TempDir>)> {
    let mut configs = vec![("Memory", StorageConfig::Memory, None)];

    // Add RocksDB config
    let temp_dir = TempDir::new().unwrap();
    let rocksdb_path = temp_dir
        .path()
        .join("rocksdb")
        .to_string_lossy()
        .to_string();
    configs.push((
        "RocksDB",
        StorageConfig::RocksDB {
            path: rocksdb_path,
            max_open_files: 1000,
            create_if_missing: true,
        },
        Some(temp_dir),
    ));

    // Add SQLite config
    let temp_dir = TempDir::new().unwrap();
    let sqlite_path = temp_dir
        .path()
        .join("sqlite.db")
        .to_string_lossy()
        .to_string();
    configs.push((
        "SQLite",
        StorageConfig::SQLite { path: sqlite_path },
        Some(temp_dir),
    ));

    #[cfg(feature = "sled-backend")]
    {
        let temp_dir = TempDir::new().unwrap();
        let sled_path = temp_dir.path().join("sled").to_string_lossy().to_string();
        configs.push((
            "Sled",
            StorageConfig::Sled {
                path: sled_path,
                cache_capacity: 10 * 1024 * 1024, // 10MB
            },
            Some(temp_dir),
        ));
    }

    configs
}

/// Macro to run a test against all backends
macro_rules! test_all_backends {
    ($test_name:ident, $test_fn:expr) => {
        #[tokio::test]
        async fn $test_name() {
            let configs = get_test_configs().await;

            for (name, config, _temp_dir) in configs {
                println!("Testing backend: {}", name);
                let backend = UnifiedBackend::init(config).await.unwrap();
                $test_fn(backend).await.unwrap();
            }
        }
    };
}

test_all_backends!(
    test_basic_operations,
    |backend: UnifiedBackend| async move {
        // Test set and get raw
        backend.set_raw("test_key", b"test_value").await?;
        let value = backend.get_raw("test_key").await?;
        assert_eq!(value, Some(b"test_value".to_vec()));

        // Test non-existent key
        let value = backend.get_raw("non_existent").await?;
        assert_eq!(value, None);

        // Test exists
        assert!(backend.exists("test_key").await?);
        assert!(!backend.exists("non_existent").await?);

        // Test delete
        backend.delete("test_key").await?;
        let value = backend.get_raw("test_key").await?;
        assert_eq!(value, None);

        Ok::<(), StorageError>(())
    }
);

test_all_backends!(test_serialization, |backend: UnifiedBackend| async move {
    let test_data = TestData {
        id: 42,
        name: "Test Object".to_string(),
        tags: vec!["tag1".to_string(), "tag2".to_string()],
    };

    // Test serialized set and get
    backend.set("test_data", &test_data).await?;
    let retrieved: Option<TestData> = backend.get("test_data").await?;
    assert_eq!(retrieved, Some(test_data));

    Ok::<(), StorageError>(())
});

test_all_backends!(test_list_keys, |backend: UnifiedBackend| async move {
    // Set multiple keys with different prefixes
    backend.set_raw("user:1", b"Alice").await?;
    backend.set_raw("user:2", b"Bob").await?;
    backend.set_raw("user:3", b"Charlie").await?;
    backend.set_raw("post:1", b"Hello World").await?;
    backend.set_raw("post:2", b"Goodbye").await?;

    // List keys with user prefix
    let user_keys = backend.list_keys("user:").await?;
    assert_eq!(user_keys.len(), 3);
    assert!(user_keys.contains(&"user:1".to_string()));
    assert!(user_keys.contains(&"user:2".to_string()));
    assert!(user_keys.contains(&"user:3".to_string()));

    // List keys with post prefix
    let post_keys = backend.list_keys("post:").await?;
    assert_eq!(post_keys.len(), 2);
    assert!(post_keys.contains(&"post:1".to_string()));
    assert!(post_keys.contains(&"post:2".to_string()));

    Ok::<(), StorageError>(())
});

test_all_backends!(
    test_transaction_commit,
    |backend: UnifiedBackend| async move {
        // Set initial values
        backend.set_raw("key1", b"initial1").await?;
        backend.set_raw("key2", b"initial2").await?;

        // Start transaction
        let mut txn = backend.begin_transaction().await?;

        // Modify values in transaction
        txn.set_raw("key1", b"modified1").await?;
        txn.set_raw("key3", b"new_value").await?;
        txn.delete("key2").await?;

        // Commit transaction
        txn.commit().await?;

        // Verify changes
        assert_eq!(backend.get_raw("key1").await?, Some(b"modified1".to_vec()));
        assert_eq!(backend.get_raw("key2").await?, None);
        assert_eq!(backend.get_raw("key3").await?, Some(b"new_value".to_vec()));

        Ok::<(), StorageError>(())
    }
);

test_all_backends!(
    test_transaction_rollback,
    |backend: UnifiedBackend| async move {
        // Set initial value
        backend.set_raw("key1", b"initial").await?;

        // Start transaction
        let mut txn = backend.begin_transaction().await?;

        // Modify value in transaction
        txn.set_raw("key1", b"modified").await?;
        txn.set_raw("key2", b"new").await?;

        // Rollback transaction
        txn.rollback().await?;

        // Verify no changes
        assert_eq!(backend.get_raw("key1").await?, Some(b"initial".to_vec()));
        assert_eq!(backend.get_raw("key2").await?, None);

        Ok::<(), StorageError>(())
    }
);

test_all_backends!(
    test_compare_and_swap,
    |backend: UnifiedBackend| async move {
        let initial_value = TestData {
            id: 1,
            name: "Initial".to_string(),
            tags: vec![],
        };

        let new_value = TestData {
            id: 1,
            name: "Updated".to_string(),
            tags: vec!["updated".to_string()],
        };

        // Set initial value
        backend.set("cas_key", &initial_value).await?;

        // Successful CAS with correct expected value
        let success = backend
            .compare_and_swap("cas_key", Some(&initial_value), &new_value)
            .await?;
        assert!(success);

        // Verify value was updated
        let current: Option<TestData> = backend.get("cas_key").await?;
        assert_eq!(current, Some(new_value.clone()));

        // Failed CAS with wrong expected value
        let wrong_expected = TestData {
            id: 2,
            name: "Wrong".to_string(),
            tags: vec![],
        };

        let another_value = TestData {
            id: 1,
            name: "Another".to_string(),
            tags: vec!["another".to_string()],
        };

        let success = backend
            .compare_and_swap("cas_key", Some(&wrong_expected), &another_value)
            .await?;
        assert!(!success);

        // Verify value was not changed
        let current: Option<TestData> = backend.get("cas_key").await?;
        assert_eq!(current, Some(new_value));

        Ok::<(), StorageError>(())
    }
);

test_all_backends!(test_stats, |backend: UnifiedBackend| async move {
    // Initially should be empty or nearly empty
    let initial_stats = backend.stats().await?;
    let initial_keys = initial_stats.total_keys;

    // Add some data
    backend.set_raw("key1", b"value1").await?;
    backend.set_raw("key2", b"longer_value_here").await?;

    let stats = backend.stats().await?;
    assert_eq!(stats.total_keys, initial_keys + 2);
    assert!(stats.total_size >= initial_stats.total_size);

    Ok::<(), StorageError>(())
});

test_all_backends!(test_flush, |backend: UnifiedBackend| async move {
    // Set some data
    backend.set_raw("key1", b"value1").await?;

    // Flush should not error
    backend.flush().await?;

    // Data should still be accessible
    let value = backend.get_raw("key1").await?;
    assert_eq!(value, Some(b"value1".to_vec()));

    Ok::<(), StorageError>(())
});

// Test specific backends with their specialized configurations
#[tokio::test]
async fn test_rocksdb_backend() -> StorageResult<()> {
    let temp_dir = TempDir::new().unwrap();
    let config = StorageConfig::RocksDB {
        path: temp_dir.path().to_string_lossy().to_string(),
        max_open_files: 1000,
        create_if_missing: true,
    };

    let backend = RocksDBBackend::init(config).await?;

    // Test basic operations
    backend.set_raw("test", b"value").await?;
    assert_eq!(backend.get_raw("test").await?, Some(b"value".to_vec()));

    // Test transaction
    let mut txn = backend.begin_transaction().await?;
    txn.set_raw("txn_test", b"txn_value").await?;
    txn.commit().await?;

    assert_eq!(
        backend.get_raw("txn_test").await?,
        Some(b"txn_value".to_vec())
    );

    Ok(())
}

#[tokio::test]
async fn test_sqlite_backend() -> StorageResult<()> {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test.db");
    let config = StorageConfig::SQLite {
        path: db_path.to_string_lossy().to_string(),
    };

    let backend = SQLiteBackend::init(config).await?;

    // Test basic operations
    backend.set_raw("test", b"value").await?;
    assert_eq!(backend.get_raw("test").await?, Some(b"value".to_vec()));

    // Test transaction
    let mut txn = backend.begin_transaction().await?;
    txn.set_raw("txn_test", b"txn_value").await?;
    txn.commit().await?;

    assert_eq!(
        backend.get_raw("txn_test").await?,
        Some(b"txn_value".to_vec())
    );

    Ok(())
}

#[tokio::test]
async fn test_memory_backend() -> StorageResult<()> {
    let backend = MemoryBackend::init(StorageConfig::Memory).await?;

    // Test basic operations
    backend.set_raw("test", b"value").await?;
    assert_eq!(backend.get_raw("test").await?, Some(b"value".to_vec()));

    // Test transaction
    let mut txn = backend.begin_transaction().await?;
    txn.set_raw("txn_test", b"txn_value").await?;
    txn.commit().await?;

    assert_eq!(
        backend.get_raw("txn_test").await?,
        Some(b"txn_value".to_vec())
    );

    Ok(())
}

#[cfg(feature = "sled-backend")]
#[tokio::test]
async fn test_sled_backend() -> StorageResult<()> {
    let temp_dir = TempDir::new().unwrap();
    let config = StorageConfig::Sled {
        path: temp_dir.path().to_string_lossy().to_string(),
        cache_capacity: 10 * 1024 * 1024,
    };

    let backend = SledBackend::init(config).await?;

    // Test basic operations
    backend.set_raw("test", b"value").await?;
    assert_eq!(backend.get_raw("test").await?, Some(b"value".to_vec()));

    // Test transaction
    let mut txn = backend.begin_transaction().await?;
    txn.set_raw("txn_test", b"txn_value").await?;
    txn.commit().await?;

    assert_eq!(
        backend.get_raw("txn_test").await?,
        Some(b"txn_value".to_vec())
    );

    Ok(())
}

/// Test concurrent access to backends (where supported)
#[tokio::test]
async fn test_concurrent_access() -> StorageResult<()> {
    let backend = MemoryBackend::init(StorageConfig::Memory).await?;

    // Clone backend for concurrent access
    let backend1 = backend.clone();
    let backend2 = backend.clone();

    let handle1 = tokio::spawn(async move {
        for i in 0..100 {
            backend1
                .set_raw(&format!("key1_{i}"), format!("value1_{i}").as_bytes())
                .await?;
        }
        Ok::<(), StorageError>(())
    });

    let handle2 = tokio::spawn(async move {
        for i in 0..100 {
            backend2
                .set_raw(&format!("key2_{i}"), format!("value2_{i}").as_bytes())
                .await?;
        }
        Ok::<(), StorageError>(())
    });

    handle1.await.unwrap()?;
    handle2.await.unwrap()?;

    // Verify all data was written
    for i in 0..100 {
        let val1 = backend.get_raw(&format!("key1_{i}")).await?;
        let val2 = backend.get_raw(&format!("key2_{i}")).await?;

        assert_eq!(val1, Some(format!("value1_{i}").into_bytes()));
        assert_eq!(val2, Some(format!("value2_{i}").into_bytes()));
    }

    Ok(())
}
