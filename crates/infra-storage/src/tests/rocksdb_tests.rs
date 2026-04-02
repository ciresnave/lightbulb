use crate::backend::{StorageBackend, StorageConfig, StorageResult};
use crate::rocksdb::RocksDBBackend;
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

#[tokio::test]
async fn test_rocksdb_basic_operations() -> StorageResult<()> {
    let temp_dir = TempDir::new().unwrap();
    let config = StorageConfig::RocksDB {
        path: temp_dir.path().to_str().unwrap().to_string(),
        max_open_files: 1000,
        create_if_missing: true,
    };

    let backend = RocksDBBackend::init(config).await?;

    // Test set and get
    backend.set_raw("key1", b"value1").await?;
    assert_eq!(backend.get_raw("key1").await?, Some(b"value1".to_vec()));

    // Test non-existent key
    assert_eq!(backend.get_raw("nonexistent").await?, None);

    // Test delete
    backend.delete("key1").await?;
    assert_eq!(backend.get_raw("key1").await?, None);

    // Test exists
    backend.set_raw("key2", b"value2").await?;
    assert!(backend.exists("key2").await?);
    assert!(!backend.exists("nonexistent").await?);

    // Test list keys
    backend.set_raw("prefix1:key1", b"value1").await?;
    backend.set_raw("prefix1:key2", b"value2").await?;
    backend.set_raw("prefix2:key3", b"value3").await?;

    let prefix1_keys = backend.list_keys("prefix1:").await?;
    assert_eq!(prefix1_keys.len(), 2);
    assert!(prefix1_keys.contains(&"prefix1:key1".to_string()));
    assert!(prefix1_keys.contains(&"prefix1:key2".to_string()));

    Ok(())
}

#[tokio::test]
async fn test_rocksdb_transactions() -> StorageResult<()> {
    let temp_dir = TempDir::new().unwrap();
    let config = StorageConfig::RocksDB {
        path: temp_dir.path().to_str().unwrap().to_string(),
        max_open_files: 1000,
        create_if_missing: true,
    };

    let backend = RocksDBBackend::init(config).await?;

    // Test successful transaction
    {
        let mut txn = backend.begin_transaction().await?;
        txn.set_raw("tx_key1", b"tx_value1").await?;
        txn.set_raw("tx_key2", b"tx_value2").await?;
        txn.commit().await?;
    }

    assert_eq!(
        backend.get_raw("tx_key1").await?,
        Some(b"tx_value1".to_vec())
    );
    assert_eq!(
        backend.get_raw("tx_key2").await?,
        Some(b"tx_value2".to_vec())
    );

    // Test transaction rollback
    {
        let mut txn = backend.begin_transaction().await?;
        txn.set_raw("tx_key3", b"tx_value3").await?;
        txn.rollback().await?;
    }

    assert_eq!(backend.get_raw("tx_key3").await?, None);

    // Test compare_and_swap
    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct TestStruct {
        value: i32,
    }

    let original = TestStruct { value: 42 };
    let new_value = TestStruct { value: 43 };

    // Initial set
    backend.set("cas_key", &original).await?;

    // Successful CAS
    assert!(
        backend
            .compare_and_swap("cas_key", Some(&original), &new_value)
            .await?
    );

    // Failed CAS (value changed)
    assert!(
        !backend
            .compare_and_swap("cas_key", Some(&original), &new_value)
            .await?
    );

    // Check final value
    let final_value: TestStruct = backend.get("cas_key").await?.unwrap();
    assert_eq!(final_value.value, 43);

    Ok(())
}

#[tokio::test]
async fn test_rocksdb_stats() -> StorageResult<()> {
    let temp_dir = TempDir::new().unwrap();
    let config = StorageConfig::RocksDB {
        path: temp_dir.path().to_str().unwrap().to_string(),
        max_open_files: 1000,
        create_if_missing: true,
    };

    let backend = RocksDBBackend::init(config).await?;

    // Add some test data
    for i in 0..100 {
        backend.set_raw(&format!("key{i}"), b"test_value").await?;
    }

    // Get stats
    let stats = backend.stats().await?;

    // Basic validation
    assert!(stats.total_keys > 0);
    assert!(stats.total_size > 0);

    // Test flush
    backend.flush().await?;

    Ok(())
}
