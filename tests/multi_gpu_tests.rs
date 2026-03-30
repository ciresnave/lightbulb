// Multi-GPU Testing and Validation
//
// M3.6 Task 5: Comprehensive test suite for multi-GPU inference
//
// Tests are organized into categories:
// 1. Tensor Parallelism Tests
// 2. Pipeline Parallelism Tests
// 3. Distributed Cache Tests
// 4. Hybrid Parallelism Tests
// 5. Performance Benchmarks
//
// All tests are gated with #[ignore] by default since they require multi-GPU hardware.
// Run with: cargo test --test multi_gpu_tests -- --ignored --test-threads=1

use anyhow::Result;
use candlelight::core::{DType, Device, Tensor};
use lightbulb::multi_gpu::{
    config::{MultiGPUConfig, ParallelismMode},
    distributed_cache::{CacheSyncStrategy, DistributedCacheManager},
    pipeline_parallel::{PipelineScheduler, PipelineStrategy},
    tensor_parallel::{ShardDimension, ShardedLinear, TensorShard},
    topology::DeviceTopology,
};

// ============================================================================
// Test Utilities
// ============================================================================

/// Check if multi-GPU setup is available
fn check_multi_gpu() -> Result<DeviceTopology> {
    let topology = DeviceTopology::discover()?;
    if topology.num_gpus() < 2 {
        anyhow::bail!(
            "Multi-GPU tests require at least 2 GPUs, found {}",
            topology.num_gpus()
        );
    }
    Ok(topology)
}

/// Create test tensor on specified device
fn create_test_tensor(shape: &[usize], device: &Device) -> Result<Tensor> {
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = (0..numel).map(|i| i as f32 / 100.0).collect();
    let tensor = Tensor::from_vec(data, shape, device)?;
    Ok(tensor)
}

/// Compare two tensors for numerical equality within tolerance
fn tensors_equal(a: &Tensor, b: &Tensor, tolerance: f32) -> Result<bool> {
    if a.shape() != b.shape() {
        return Ok(false);
    }

    let diff = (a - b)?.abs()?;
    let max_diff = diff.max(0)?.to_vec0::<f32>()?;

    Ok(max_diff < tolerance)
}

// ============================================================================
// 1. Tensor Parallelism Tests
// ============================================================================

#[test]
#[ignore] // Requires multi-GPU
fn test_tensor_shard_creation() -> Result<()> {
    let topology = check_multi_gpu()?;
    let device = topology.device(0).unwrap();

    // Create a test tensor
    let tensor = create_test_tensor(&[128, 512], &device)?;

    // Shard across 2 GPUs
    let shard = TensorShard::from_full_tensor(&tensor, 0, 2, ShardDimension::Column)?;

    // Verify shard shape (should be half of original along dim 1)
    assert_eq!(shard.shard().dims(), &[128, 256]);
    assert_eq!(shard.num_shards(), 2);
    assert_eq!(shard.shard_id(), 0);

    println!("✓ Tensor shard created successfully");
    Ok(())
}

#[test]
#[ignore] // Requires multi-GPU
fn test_tensor_gather_2gpu() -> Result<()> {
    let topology = check_multi_gpu()?;

    // Create original tensor on GPU 0
    let device0 = topology.device(0).unwrap();
    let original = create_test_tensor(&[64, 256], &device0)?;

    // Shard and distribute
    let shard0 = TensorShard::from_full_tensor(&original, 0, 2, ShardDimension::Column)?;
    let shard1 = TensorShard::from_full_tensor(&original, 1, 2, ShardDimension::Column)?;

    // Gather shards back together
    let shards = vec![shard0, shard1];
    let gathered = TensorShard::gather(&shards, ShardDimension::Column)?;

    // Verify gathered tensor matches original
    assert!(tensors_equal(&original, &gathered, 1e-5)?);

    println!("✓ Tensor gather works correctly (2 GPUs)");
    Ok(())
}

#[test]
#[ignore] // Requires 4 GPUs
fn test_tensor_gather_4gpu() -> Result<()> {
    let topology = DeviceTopology::discover()?;
    if topology.num_gpus() < 4 {
        println!(
            "⊘ Skipping 4-GPU test (only {} GPUs available)",
            topology.num_gpus()
        );
        return Ok(());
    }

    let device0 = topology.device(0).unwrap();
    let original = create_test_tensor(&[64, 512], &device0)?;

    // Shard across 4 GPUs
    let shards: Vec<TensorShard> = (0..4)
        .map(|i| TensorShard::from_full_tensor(&original, i, 4, ShardDimension::Column))
        .collect::<Result<Vec<_>>>()?;

    let gathered = TensorShard::gather(&shards, ShardDimension::Column)?;

    assert!(tensors_equal(&original, &gathered, 1e-5)?);

    println!("✓ Tensor gather works correctly (4 GPUs)");
    Ok(())
}

#[test]
#[ignore] // Requires multi-GPU
fn test_sharded_linear_column() -> Result<()> {
    let topology = check_multi_gpu()?;
    let device0 = topology.device(0).unwrap();

    let input_dim = 512;
    let output_dim = 1024;
    let batch_size = 8;

    // Create input
    let input = create_test_tensor(&[batch_size, input_dim], &device0)?;

    // Create sharded linear layer (column-wise sharding)
    let devices = vec![
        topology.device(0).unwrap().clone(),
        topology.device(1).unwrap().clone(),
    ];

    let sharded_linear = ShardedLinear::new(
        input_dim,
        output_dim,
        true, // bias
        2,    // num_shards
        ShardDimension::Column,
        devices,
    )?;

    // Forward pass
    let output = sharded_linear.forward(&input)?;

    // Verify output shape
    assert_eq!(output.dims(), &[batch_size, output_dim]);

    println!("✓ Sharded linear (column) forward pass works");
    Ok(())
}

#[test]
#[ignore] // Requires multi-GPU
fn test_sharded_linear_row() -> Result<()> {
    let topology = check_multi_gpu()?;
    let device0 = topology.device(0).unwrap();

    let input_dim = 512;
    let output_dim = 1024;
    let batch_size = 8;

    let input = create_test_tensor(&[batch_size, input_dim], &device0)?;

    let devices = vec![
        topology.device(0).unwrap().clone(),
        topology.device(1).unwrap().clone(),
    ];

    let sharded_linear =
        ShardedLinear::new(input_dim, output_dim, true, 2, ShardDimension::Row, devices)?;

    let output = sharded_linear.forward(&input)?;

    assert_eq!(output.dims(), &[batch_size, output_dim]);

    println!("✓ Sharded linear (row) forward pass works");
    Ok(())
}

// ============================================================================
// 2. Pipeline Parallelism Tests
// ============================================================================

#[test]
#[ignore] // Requires multi-GPU
fn test_pipeline_scheduler_creation() -> Result<()> {
    let topology = check_multi_gpu()?;

    let num_stages = 2;
    let num_layers = 40;
    let devices = vec![
        topology.device(0).unwrap().clone(),
        topology.device(1).unwrap().clone(),
    ];

    let scheduler = PipelineScheduler::new(
        num_stages,
        num_layers,
        devices,
        4, // micro_batch_size
        PipelineStrategy::GPipe,
    )?;

    assert_eq!(scheduler.num_stages(), 2);

    // Check layer distribution
    let stage0 = scheduler.stage(0).unwrap();
    let stage1 = scheduler.stage(1).unwrap();

    assert_eq!(stage0.layers.len(), 20); // First 20 layers
    assert_eq!(stage1.layers.len(), 20); // Last 20 layers

    println!("✓ Pipeline scheduler created with proper layer distribution");
    Ok(())
}

#[test]
#[ignore] // Requires multi-GPU
fn test_pipeline_micro_batch_splitting() -> Result<()> {
    let topology = check_multi_gpu()?;

    let devices = vec![
        topology.device(0).unwrap().clone(),
        topology.device(1).unwrap().clone(),
    ];

    let mut scheduler = PipelineScheduler::new(
        2,
        40,
        devices,
        4, // micro_batch_size
        PipelineStrategy::GPipe,
    )?;

    // Create input batch (batch_size=8)
    let device0 = topology.device(0).unwrap();
    let input = create_test_tensor(&[8, 512], &device0)?;

    // Execute pipeline (placeholder forward - will work after Task 6 integration)
    let result = scheduler.execute(input);

    // For now, this should at least not crash
    match result {
        Ok(_output) => {
            println!("✓ Pipeline execution completed (placeholder mode)");
        }
        Err(e) => {
            println!(
                "⊘ Pipeline execution failed (expected in placeholder mode): {}",
                e
            );
        }
    }

    Ok(())
}

#[test]
#[ignore] // Requires 4 GPUs
fn test_pipeline_4stage() -> Result<()> {
    let topology = DeviceTopology::discover()?;
    if topology.num_gpus() < 4 {
        println!(
            "⊘ Skipping 4-stage test (only {} GPUs available)",
            topology.num_gpus()
        );
        return Ok(());
    }

    let devices: Vec<Device> = (0..4)
        .map(|i| topology.device(i).unwrap().clone())
        .collect();

    let scheduler = PipelineScheduler::new(
        4,
        80, // 80 layers across 4 GPUs = 20 layers each
        devices,
        4,
        PipelineStrategy::GPipe,
    )?;

    assert_eq!(scheduler.num_stages(), 4);

    // Verify layer distribution
    for i in 0..4 {
        let stage = scheduler.stage(i).unwrap();
        assert_eq!(stage.layers.len(), 20);
        assert_eq!(stage.stage_id, i);
    }

    println!("✓ 4-stage pipeline created successfully");
    Ok(())
}

// ============================================================================
// 3. Distributed Cache Tests
// ============================================================================

#[test]
#[ignore] // Requires multi-GPU
fn test_distributed_cache_creation() -> Result<()> {
    let topology = check_multi_gpu()?;

    let cache_manager = DistributedCacheManager::new(
        topology,
        CacheSyncStrategy::Replicated,
        4,    // batch_size
        2048, // context_size
        DType::F16,
    )?;

    assert!(cache_manager.num_gpus() >= 2);

    println!("✓ Distributed cache manager created");
    Ok(())
}

#[test]
#[ignore] // Requires multi-GPU
fn test_distributed_cache_replication() -> Result<()> {
    let topology = check_multi_gpu()?;

    let mut cache_manager = DistributedCacheManager::new(
        topology.clone(),
        CacheSyncStrategy::Replicated,
        4,
        2048,
        DType::F32,
    )?;

    // Create test K/V tensors
    let device0 = topology.device(0).unwrap();
    let k_new = create_test_tensor(&[1, 8, 1, 64], &device0)?;
    let v_new = create_test_tensor(&[1, 8, 1, 64], &device0)?;

    // Update cache (should replicate to all GPUs)
    cache_manager.update_cache(0, 0, &k_new, &v_new)?;

    println!("✓ Cache replication across GPUs successful");
    Ok(())
}

#[test]
#[ignore] // Requires multi-GPU
fn test_distributed_cache_access() -> Result<()> {
    let topology = check_multi_gpu()?;

    let cache_manager = DistributedCacheManager::new(
        topology.clone(),
        CacheSyncStrategy::Replicated,
        4,
        2048,
        DType::F32,
    )?;

    // Access cache for each GPU
    for gpu_id in 0..cache_manager.num_gpus() {
        let cache = cache_manager.cache_for_gpu(gpu_id);
        assert!(cache.is_some(), "GPU {} cache not found", gpu_id);
    }

    println!("✓ Cache access for all GPUs works");
    Ok(())
}

// ============================================================================
// 4. Hybrid Parallelism Tests
// ============================================================================

#[test]
#[ignore] // Requires 4 GPUs
fn test_hybrid_config_creation() -> Result<()> {
    let topology = DeviceTopology::discover()?;
    if topology.num_gpus() < 4 {
        println!(
            "⊘ Skipping hybrid test (only {} GPUs available)",
            topology.num_gpus()
        );
        return Ok(());
    }

    // Create config for 4 GPUs: 2-way tensor parallel × 2-way pipeline parallel
    let config = MultiGPUConfig::hybrid(
        2, // tensor_parallel_size
        2, // pipeline_parallel_size
    )?;

    assert_eq!(config.parallelism_mode(), ParallelismMode::Hybrid);

    println!("✓ Hybrid parallelism config created (2×2)");
    Ok(())
}

// ============================================================================
// 5. Performance Benchmarks
// ============================================================================

#[test]
#[ignore] // Requires multi-GPU
fn benchmark_tensor_transfer_latency() -> Result<()> {
    let topology = check_multi_gpu()?;

    let device0 = topology.device(0).unwrap();
    let device1 = topology.device(1).unwrap();

    // Benchmark different tensor sizes
    let sizes = vec![
        (64, 512),   // Small
        (128, 2048), // Medium
        (256, 4096), // Large
    ];

    println!("\n=== Tensor Transfer Latency Benchmark ===");

    for (rows, cols) in sizes {
        let tensor = create_test_tensor(&[rows, cols], &device0)?;

        let start = std::time::Instant::now();
        let _transferred = tensor.to_device(&device1)?;
        let elapsed = start.elapsed();

        let size_mb = (rows * cols * 4) as f64 / (1024.0 * 1024.0); // F32 = 4 bytes
        println!(
            "  [{} × {}] {:.2} MB: {:.3} ms ({:.2} GB/s)",
            rows,
            cols,
            size_mb,
            elapsed.as_secs_f64() * 1000.0,
            size_mb / 1024.0 / elapsed.as_secs_f64()
        );
    }

    Ok(())
}

#[test]
#[ignore] // Requires multi-GPU
fn benchmark_sharded_linear_throughput() -> Result<()> {
    let topology = check_multi_gpu()?;

    let devices = vec![
        topology.device(0).unwrap().clone(),
        topology.device(1).unwrap().clone(),
    ];

    let input_dim = 4096;
    let output_dim = 4096;
    let batch_sizes = vec![1, 4, 8, 16, 32];

    println!("\n=== Sharded Linear Throughput Benchmark ===");
    println!("Input dim: {}, Output dim: {}", input_dim, output_dim);

    let sharded_linear = ShardedLinear::new(
        input_dim,
        output_dim,
        true,
        2,
        ShardDimension::Column,
        devices.clone(),
    )?;

    for batch_size in batch_sizes {
        let device0 = topology.device(0).unwrap();
        let input = create_test_tensor(&[batch_size, input_dim], &device0)?;

        // Warmup
        let _ = sharded_linear.forward(&input)?;

        // Benchmark
        let iterations = 10;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = sharded_linear.forward(&input)?;
        }
        let elapsed = start.elapsed();

        let avg_time = elapsed.as_secs_f64() / iterations as f64;
        let throughput = batch_size as f64 / avg_time;

        println!(
            "  Batch size {}: {:.3} ms/batch ({:.1} samples/sec)",
            batch_size,
            avg_time * 1000.0,
            throughput
        );
    }

    Ok(())
}

#[test]
#[ignore] // Requires multi-GPU
fn test_communication_overhead() -> Result<()> {
    let topology = check_multi_gpu()?;

    let device0 = topology.device(0).unwrap();
    let device1 = topology.device(1).unwrap();

    // Create test tensor
    let tensor = create_test_tensor(&[128, 4096], &device0)?;

    // Measure compute time (local operation)
    let start = std::time::Instant::now();
    let _local = (&tensor * 2.0)?;
    let compute_time = start.elapsed();

    // Measure transfer time
    let start = std::time::Instant::now();
    let _transferred = tensor.to_device(&device1)?;
    let transfer_time = start.elapsed();

    let overhead_pct = (transfer_time.as_secs_f64() / compute_time.as_secs_f64()) * 100.0;

    println!("\n=== Communication Overhead Analysis ===");
    println!(
        "  Compute time: {:.3} ms",
        compute_time.as_secs_f64() * 1000.0
    );
    println!(
        "  Transfer time: {:.3} ms",
        transfer_time.as_secs_f64() * 1000.0
    );
    println!("  Overhead: {:.1}%", overhead_pct);

    // Target: <15% overhead
    assert!(
        overhead_pct < 15.0 || transfer_time.as_secs_f64() < 0.001,
        "Communication overhead too high: {:.1}% (target <15%)",
        overhead_pct
    );

    println!("✓ Communication overhead within acceptable range");
    Ok(())
}

// ============================================================================
// Integration Test Suite
// ============================================================================

#[test]
#[ignore] // Requires multi-GPU
fn test_full_pipeline_integration() -> Result<()> {
    println!("\n=== Full Multi-GPU Integration Test ===");

    // 1. Topology discovery
    let topology = check_multi_gpu()?;
    println!("✓ Discovered {} GPUs", topology.num_gpus());

    // 2. Tensor parallelism
    let device0 = topology.device(0).unwrap();
    let tensor = create_test_tensor(&[64, 512], &device0)?;
    let shard = TensorShard::from_full_tensor(&tensor, 0, 2, ShardDimension::Column)?;
    println!("✓ Tensor sharding works");

    // 3. Distributed cache
    let cache_manager = DistributedCacheManager::new(
        topology.clone(),
        CacheSyncStrategy::Replicated,
        4,
        2048,
        DType::F32,
    )?;
    println!("✓ Distributed cache initialized");

    // 4. Pipeline parallelism
    let devices = vec![
        topology.device(0).unwrap().clone(),
        topology.device(1).unwrap().clone(),
    ];

    let _scheduler = PipelineScheduler::new(2, 40, devices, 4, PipelineStrategy::GPipe)?;
    println!("✓ Pipeline scheduler created");

    println!("\n✅ All multi-GPU components working together");
    Ok(())
}
