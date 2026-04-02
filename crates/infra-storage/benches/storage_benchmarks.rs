use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use infra_storage::*;
use std::time::Instant;
use tempfile::TempDir;
use tokio::runtime::Runtime;

/// Get benchmark configurations for all backends
fn get_benchmark_configs() -> Vec<(&'static str, StorageConfig, Option<TempDir>)> {
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

/// Benchmark read/write throughput for different data sizes
fn benchmark_read_write_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let data_sizes = vec![64, 256, 1024, 4096, 16384]; // bytes

    for (backend_name, config, _temp_dir) in get_benchmark_configs() {
        let backend = rt.block_on(async { UnifiedBackend::init(config).await.unwrap() });

        for &size in &data_sizes {
            let data = vec![42u8; size];
            let key = format!("benchmark_key_{size}");

            // Benchmark write operations
            c.bench_with_input(
                BenchmarkId::new(format!("{backend_name}_write"), size),
                &size,
                |b, &_size| {
                    b.iter_custom(|iters| {
                        let start = Instant::now();
                        rt.block_on(async {
                            for _ in 0..iters {
                                backend.set_raw(&key, &data).await.unwrap();
                            }
                        });
                        start.elapsed()
                    });
                },
            );

            // Pre-populate for read benchmark
            rt.block_on(async {
                backend.set_raw(&key, &data).await.unwrap();
            });

            // Benchmark read operations
            c.bench_with_input(
                BenchmarkId::new(format!("{backend_name}_read"), size),
                &size,
                |b, &_size| {
                    b.iter_custom(|iters| {
                        let start = Instant::now();
                        rt.block_on(async {
                            for _ in 0..iters {
                                let _ = backend.get_raw(&key).await.unwrap();
                            }
                        });
                        start.elapsed()
                    });
                },
            );
        }
    }
}

/// Benchmark bulk operations
fn benchmark_bulk_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let item_counts = vec![100, 1000];

    for (backend_name, config, _temp_dir) in get_benchmark_configs() {
        let backend = rt.block_on(async { UnifiedBackend::init(config).await.unwrap() });

        for &count in &item_counts {
            let data = vec![42u8; 256]; // 256 byte values

            // Benchmark bulk write
            c.bench_with_input(
                BenchmarkId::new(format!("{backend_name}_bulk_write"), count),
                &count,
                |b, &count| {
                    b.iter_custom(|iters| {
                        let start = Instant::now();
                        rt.block_on(async {
                            for _ in 0..iters {
                                for i in 0..count {
                                    let key = format!("bulk_key_{iters}_{i}");
                                    backend.set_raw(&key, &data).await.unwrap();
                                }
                            }
                        });
                        start.elapsed()
                    });
                },
            );

            // Pre-populate for read benchmark
            rt.block_on(async {
                for i in 0..count {
                    let key = format!("bulk_read_key_{i}");
                    backend.set_raw(&key, &data).await.unwrap();
                }
            });

            // Benchmark bulk read
            c.bench_with_input(
                BenchmarkId::new(format!("{backend_name}_bulk_read"), count),
                &count,
                |b, &count| {
                    b.iter_custom(|iters| {
                        let start = Instant::now();
                        rt.block_on(async {
                            for _ in 0..iters {
                                for i in 0..count {
                                    let key = format!("bulk_read_key_{i}");
                                    let _ = backend.get_raw(&key).await.unwrap();
                                }
                            }
                        });
                        start.elapsed()
                    });
                },
            );
        }
    }
}

/// Benchmark transaction performance
fn benchmark_transaction_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let operation_counts = vec![10, 50];

    for (backend_name, config, _temp_dir) in get_benchmark_configs() {
        let backend = rt.block_on(async { UnifiedBackend::init(config).await.unwrap() });

        for &ops in &operation_counts {
            let data = vec![42u8; 256];

            c.bench_with_input(
                BenchmarkId::new(format!("{backend_name}_transaction"), ops),
                &ops,
                |b, &ops| {
                    b.iter_custom(|iters| {
                        let start = Instant::now();
                        rt.block_on(async {
                            for iter in 0..iters {
                                let mut txn = backend.begin_transaction().await.unwrap();

                                for i in 0..ops {
                                    let key = format!("txn_key_{iter}_{i}");
                                    txn.set_raw(&key, &data).await.unwrap();
                                }

                                txn.commit().await.unwrap();
                            }
                        });
                        start.elapsed()
                    });
                },
            );
        }
    }
}

/// Benchmark storage stats collection
fn benchmark_stats_collection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    for (backend_name, config, _temp_dir) in get_benchmark_configs() {
        let backend = rt.block_on(async { UnifiedBackend::init(config).await.unwrap() });

        // Pre-populate with some data
        rt.block_on(async {
            let data = vec![42u8; 1024];
            for i in 0..100 {
                let key = format!("stats_key_{i}");
                backend.set_raw(&key, &data).await.unwrap();
            }
        });

        c.bench_function(&format!("{backend_name}_stats"), |b| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                rt.block_on(async {
                    for _ in 0..iters {
                        let _ = backend.stats().await.unwrap();
                    }
                });
                start.elapsed()
            });
        });
    }
}

criterion_group!(
    storage_benchmarks,
    benchmark_read_write_throughput,
    benchmark_bulk_operations,
    benchmark_transaction_performance,
    benchmark_stats_collection
);

criterion_main!(storage_benchmarks);
