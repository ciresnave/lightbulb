# Infra-Storage

High-performance, multi-backend storage abstraction layer for the DynAniml ecosystem. Provides unified access to various storage backends including in-memory, RocksDB, SQLite, and Sled with comprehensive benchmarking and configuration management.

## Features

- **Multi-backend support**: Memory, RocksDB, SQLite, and Sled backends
- **Unified API**: Single interface for all storage operations across different backends
- **Async/await support**: Full async operations with Tokio integration
- **Comprehensive benchmarking**: Built-in performance testing for all backends
- **Type-safe configuration**: Builder pattern for configuration management
- **Production-ready**: Error handling, logging, and monitoring integration
- **Memory-efficient**: Zero-copy operations where possible
- **Thread-safe**: Concurrent access support with appropriate locking

## Quick Start

```rust
use infra_storage::{StorageBackend, Config, backends::MemoryBackend};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create memory backend
    let mut backend = MemoryBackend::new();
    
    // Store and retrieve data
    backend.put(b"key1", b"value1").await?;
    let value = backend.get(b"key1").await?;
    assert_eq!(value, Some(b"value1".to_vec()));
    
    Ok(())
}
```

## Backend Comparison

| Backend | Read Latency | Write Latency | Persistence | Memory Usage | Use Case |
|---------|-------------|---------------|-------------|--------------|-----------|
| Memory | ~100ns | ~150ns | No | High | Caching, testing |
| Sled | ~220ns | ~220ns | Yes | Medium | Fast persistent storage |
| RocksDB | ~360ns | ~280μs | Yes | Low | High-throughput databases |
| SQLite | ~20μs | ~1.5ms | Yes | Low | ACID transactions |

## Architecture

```text
┌─────────────────────────────────────────┐
│            Client Application           │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│          StorageBackend Trait           │
│     (get, put, delete, exists, etc.)    │
└─────────────┬───┬───┬────────────┬──────┘
              │   │   │            │
    ┌─────────▼─┐ │   │  ┌─────────▼─┐
    │  Memory   │ │   │  │   Sled    │
    │  Backend  │ │   │  │  Backend  │
    └───────────┘ │   │  └───────────┘
                  │   │
        ┌─────────▼─┐ │
        │  RocksDB  │ │
        │  Backend  │ │
        └───────────┘ │
                      │
            ┌─────────▼─┐
            │  SQLite   │
            │  Backend  │
            └───────────┘
```

## Backends

### Memory Backend

Fast in-memory storage using HashMap. Ideal for caching and testing.

```rust
use infra_storage::backends::MemoryBackend;

let mut backend = MemoryBackend::new();
backend.put(b"key", b"value").await?;
```

**Pros:**
- Fastest performance (~100-150ns latency)
- No setup required
- Zero persistence overhead

**Cons:**
- No persistence
- Memory usage grows with data
- Lost on restart

### RocksDB Backend

High-performance LSM-tree storage engine.

```rust
use infra_storage::backends::RocksDBBackend;

let backend = RocksDBBackend::new("./data").await?;
backend.put(b"key", b"value").await?;
```

**Pros:**
- Excellent read performance
- Handles large datasets efficiently
- Configurable compaction
- Industry proven

**Cons:**
- Higher write latency
- Complex configuration options
- Larger binary size

### SQLite Backend

ACID-compliant relational database.

```rust
use infra_storage::backends::SQLiteBackend;

let backend = SQLiteBackend::new("data.db").await?;
backend.put(b"key", b"value").await?;
```

**Pros:**
- ACID transactions
- SQL query capabilities
- Portable file format
- Zero configuration

**Cons:**
- Higher latency for simple operations
- Single writer limitation
- SQL overhead for key-value operations

### Sled Backend

Modern embedded database written in Rust.

```rust
use infra_storage::backends::SledBackend;

let backend = SledBackend::new("./data")?;
backend.put(b"key", b"value").await?;
```

**Pros:**
- Fast persistent storage
- Written in Rust
- Good balance of performance and features
- ACID transactions

**Cons:**
- Newer, less battle-tested
- API still evolving
- Limited ecosystem

## Unified Backend

For applications that need to switch between backends at runtime:

```rust
use infra_storage::{UnifiedBackend, Config, BackendType};

let config = Config::builder()
    .backend_type(BackendType::RocksDB)
    .data_dir("./storage")
    .build();

let backend = UnifiedBackend::new(config).await?;
backend.put(b"key", b"value").await?;
```

## Configuration

Comprehensive configuration management with builder pattern:

```rust
use infra_storage::{Config, BackendType, CompressionType};

let config = Config::builder()
    .backend_type(BackendType::RocksDB)
    .data_dir("./data")
    .cache_size(128 * 1024 * 1024) // 128MB cache
    .compression(CompressionType::Zstd)
    .max_open_files(1000)
    .write_buffer_size(64 * 1024 * 1024) // 64MB
    .enable_statistics(true)
    .build();
```

### Configuration Options

- `backend_type`: Storage backend to use
- `data_dir`: Directory for persistent storage
- `cache_size`: Block cache size for RocksDB
- `compression`: Compression algorithm (None, Snappy, Zstd)
- `max_open_files`: Maximum open file descriptors
- `write_buffer_size`: Write buffer size
- `enable_statistics`: Enable performance statistics

## Bulk Operations

Efficient bulk operations for batch processing:

```rust
// Bulk write
let entries = vec![
    (b"key1".to_vec(), b"value1".to_vec()),
    (b"key2".to_vec(), b"value2".to_vec()),
];
backend.bulk_put(entries).await?;

// Bulk read
let keys = vec![b"key1".to_vec(), b"key2".to_vec()];
let values = backend.bulk_get(keys).await?;
```

## Transactions

ACID transactions for consistent operations:

```rust
// Start transaction
let mut tx = backend.begin_transaction().await?;

// Perform operations
tx.put(b"key1", b"value1").await?;
tx.put(b"key2", b"value2").await?;

// Commit transaction
tx.commit().await?;
```

## Statistics and Monitoring

Built-in statistics for monitoring and optimization:

```rust
let stats = backend.get_statistics().await?;

println!("Total operations: {}", stats.total_operations);
println!("Cache hit ratio: {:.2}%", stats.cache_hit_ratio * 100.0);
println!("Average latency: {:?}", stats.average_latency);
```

## Benchmarking

Comprehensive benchmarking suite included:

```bash
cd crates/infra-storage
cargo bench
```

Benchmark categories:
- **Single operations**: Individual read/write operations
- **Bulk operations**: Batch read/write operations  
- **Transactions**: Multi-operation atomic transactions
- **Statistics**: Metadata and monitoring operations

## Performance Tips

1. **Choose the right backend**:
   - Memory: Ultra-low latency, no persistence
   - Sled: Good balance, fast persistent storage
   - RocksDB: High throughput, configurable
   - SQLite: ACID compliance, SQL capabilities

2. **Optimize configuration**:
   - Increase cache size for read-heavy workloads
   - Tune write buffer size for write-heavy workloads
   - Enable compression for storage efficiency

3. **Use bulk operations**:
   - Batch multiple operations together
   - Reduces overhead and improves throughput

4. **Enable statistics judiciously**:
   - Useful for monitoring but adds overhead
   - Consider enabling only in production monitoring

## Error Handling

All operations return `Result<T, StorageError>` with comprehensive error information:

```rust
use infra_storage::StorageError;

match backend.get(b"key").await {
    Ok(Some(value)) => println!("Found: {:?}", value),
    Ok(None) => println!("Not found"),
    Err(StorageError::Io(e)) => eprintln!("IO error: {}", e),
    Err(StorageError::Backend(e)) => eprintln!("Backend error: {}", e),
    Err(e) => eprintln!("Other error: {}", e),
}
```

## Thread Safety

All backends are thread-safe and can be shared across async tasks:

```rust
use std::sync::Arc;

let backend = Arc::new(backend);

// Spawn multiple tasks
let handles: Vec<_> = (0..10).map(|i| {
    let backend = backend.clone();
    tokio::spawn(async move {
        backend.put(format!("key{}", i).as_bytes(), b"value").await
    })
}).collect();

// Wait for all tasks
for handle in handles {
    handle.await??;
}
```

## Testing

Run the comprehensive test suite:

```bash
# Unit tests
cargo test

# Integration tests  
cargo test --test '*'

# Benchmarks
cargo bench

# All tests
cargo test --all-targets
```

## Contributing

1. Add new backends by implementing the `StorageBackend` trait
2. Add comprehensive tests for any new functionality
3. Update benchmarks for performance regression testing
4. Follow Rust conventions and run `cargo clippy`

## License

This project is part of the DynAniml ecosystem. See the workspace LICENSE file for details.
