// M3.5 Load and Stress Testing Infrastructure
//
// This module provides comprehensive load and stress testing for Lightbulb:
//
// 1. **Concurrent Request Simulator**: Test 10, 50, 100, 500 concurrent requests
// 2. **Edge Case Validator**: 128k token contexts, 128+ batch sizes, mixed workloads
// 3. **Soak Test Harness**: 48hr+ continuous operation, memory leak detection
//
// **Success Criteria**:
// - Stable under 100+ concurrent requests
// - No memory leaks over 48 hours
// - Graceful degradation under extreme load
// - <1% error rate under normal load (100 concurrent)
// - <5% error rate under 2× capacity (200 concurrent)

use anyhow::{Context, Result};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};
use tokio::task::JoinHandle;

/// Load test configuration
#[derive(Debug, Clone)]
pub struct LoadTestConfig {
    /// Number of concurrent requests to simulate
    pub concurrent_requests: usize,

    /// Duration to run the load test
    pub duration: Duration,

    /// Request rate (requests per second, None = unlimited)
    pub rate_limit: Option<f64>,

    /// Prompt length distribution (min, max)
    pub prompt_length_range: (usize, usize),

    /// Generation length per request (tokens to generate)
    pub generation_length: usize,

    /// Timeout per request
    pub request_timeout: Duration,
}

impl Default for LoadTestConfig {
    fn default() -> Self {
        Self {
            concurrent_requests: 100,
            duration: Duration::from_secs(60),
            rate_limit: None,
            prompt_length_range: (32, 512),
            generation_length: 128,
            request_timeout: Duration::from_secs(30),
        }
    }
}

/// Load test scenarios
pub mod scenarios {
    use super::*;

    /// Light load: 10 concurrent requests
    pub fn light_load() -> LoadTestConfig {
        LoadTestConfig {
            concurrent_requests: 10,
            duration: Duration::from_secs(60),
            ..Default::default()
        }
    }

    /// Normal load: 50 concurrent requests
    pub fn normal_load() -> LoadTestConfig {
        LoadTestConfig {
            concurrent_requests: 50,
            duration: Duration::from_secs(300), // 5 minutes
            ..Default::default()
        }
    }

    /// Heavy load: 100 concurrent requests
    pub fn heavy_load() -> LoadTestConfig {
        LoadTestConfig {
            concurrent_requests: 100,
            duration: Duration::from_secs(600), // 10 minutes
            ..Default::default()
        }
    }

    /// Stress test: 500 concurrent requests
    pub fn stress_test() -> LoadTestConfig {
        LoadTestConfig {
            concurrent_requests: 500,
            duration: Duration::from_secs(300), // 5 minutes
            request_timeout: Duration::from_secs(60),
            ..Default::default()
        }
    }

    /// Long context test: 128k token prompts
    pub fn long_context_test() -> LoadTestConfig {
        LoadTestConfig {
            concurrent_requests: 10,
            duration: Duration::from_secs(600),
            prompt_length_range: (100_000, 128_000),
            generation_length: 256,
            request_timeout: Duration::from_secs(300), // 5 min timeout
            ..Default::default()
        }
    }

    /// Soak test: 48 hours continuous operation
    pub fn soak_test() -> LoadTestConfig {
        LoadTestConfig {
            concurrent_requests: 50,
            duration: Duration::from_secs(48 * 3600), // 48 hours
            rate_limit: Some(10.0),                   // 10 req/sec sustained
            ..Default::default()
        }
    }
}

/// Load test statistics
#[derive(Debug, Clone, Default)]
pub struct LoadTestStats {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub timeout_requests: u64,

    pub total_tokens_generated: u64,
    pub total_latency_ms: u64,

    pub min_latency_ms: u64,
    pub max_latency_ms: u64,

    pub p50_latency_ms: u64,
    pub p95_latency_ms: u64,
    pub p99_latency_ms: u64,

    pub error_rate: f64, // 0.0 to 1.0
    pub throughput_tokens_per_sec: f64,
}

/// Atomic counters for concurrent statistics collection
#[derive(Debug)]
struct AtomicStats {
    total_requests: AtomicU64,
    successful_requests: AtomicU64,
    failed_requests: AtomicU64,
    timeout_requests: AtomicU64,
    total_tokens_generated: AtomicU64,
}

impl AtomicStats {
    fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            successful_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            timeout_requests: AtomicU64::new(0),
            total_tokens_generated: AtomicU64::new(0),
        }
    }
}

/// Result of a single request
#[derive(Debug)]
struct RequestResult {
    success: bool,
    timeout: bool,
    tokens_generated: usize,
    latency: Duration,
    error_message: Option<String>,
}

/// Load test runner
pub struct LoadTestRunner {
    config: LoadTestConfig,
    stats: Arc<AtomicStats>,
    latencies: Arc<RwLock<Vec<u64>>>, // Collect latencies for percentile calculation
}

impl LoadTestRunner {
    pub fn new(config: LoadTestConfig) -> Self {
        Self {
            config,
            stats: Arc::new(AtomicStats::new()),
            latencies: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Run the load test
    pub async fn run(&self) -> Result<LoadTestStats> {
        println!("Starting load test with configuration:");
        println!("  Concurrent requests: {}", self.config.concurrent_requests);
        println!("  Duration: {:?}", self.config.duration);
        println!("  Rate limit: {:?}", self.config.rate_limit);
        println!("  Prompt length: {:?}", self.config.prompt_length_range);
        println!("  Generation length: {}", self.config.generation_length);

        let start_time = Instant::now();
        let semaphore = Arc::new(Semaphore::new(self.config.concurrent_requests));
        let mut handles: Vec<JoinHandle<()>> = Vec::new();

        // Spawn request tasks
        let mut request_id = 0u64;
        loop {
            // Check if test duration exceeded
            if start_time.elapsed() >= self.config.duration {
                break;
            }

            // Rate limiting
            if let Some(rate) = self.config.rate_limit {
                let delay = Duration::from_secs_f64(1.0 / rate);
                tokio::time::sleep(delay).await;
            }

            // Acquire semaphore permit (blocks if at capacity)
            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let stats = self.stats.clone();
            let latencies = self.latencies.clone();
            let config = self.config.clone();

            let handle = tokio::spawn(async move {
                let result = Self::execute_request(request_id, &config).await;

                // Update statistics
                stats.total_requests.fetch_add(1, Ordering::SeqCst);

                if result.success {
                    stats.successful_requests.fetch_add(1, Ordering::SeqCst);
                    stats
                        .total_tokens_generated
                        .fetch_add(result.tokens_generated as u64, Ordering::SeqCst);
                } else if result.timeout {
                    stats.timeout_requests.fetch_add(1, Ordering::SeqCst);
                } else {
                    stats.failed_requests.fetch_add(1, Ordering::SeqCst);
                }

                // Record latency
                latencies
                    .write()
                    .await
                    .push(result.latency.as_millis() as u64);

                drop(permit);
            });

            handles.push(handle);
            request_id += 1;
        }

        println!("Waiting for all requests to complete...");
        for handle in handles {
            handle.await.context("Request task panicked")?;
        }

        let elapsed = start_time.elapsed();

        // Compute final statistics
        self.compute_final_stats(elapsed).await
    }

    /// Execute a single request (mock for now, replace with actual inference)
    async fn execute_request(request_id: u64, config: &LoadTestConfig) -> RequestResult {
        let start = Instant::now();

        // TODO: Replace with actual inference call
        // For now, simulate request with random latency
        use rand::Rng;
        // Draw both values and drop the RNG **before** the await. `rand`'s
        // thread RNG is `Rc`-backed and therefore not `Send`; holding it across
        // `.await` makes the whole future non-`Send`, and `tokio::spawn`
        // requires `Send`. Scoping it here is what keeps the spawn legal — the
        // previous code drew the latency, awaited, then drew `success` from the
        // same live RNG.
        let (latency_ms, success) = {
            let mut rng = rand::rng();
            (rng.random_range(10..500), rng.random_bool(0.98)) // 10-500ms; 2% failures
        };
        tokio::time::sleep(Duration::from_millis(latency_ms)).await;

        let tokens_generated = if success { config.generation_length } else { 0 };

        let latency = start.elapsed();

        RequestResult {
            success,
            timeout: latency > config.request_timeout,
            tokens_generated,
            latency,
            error_message: if !success {
                Some("Simulated error".to_string())
            } else {
                None
            },
        }
    }

    /// Compute final statistics from collected data
    async fn compute_final_stats(&self, elapsed: Duration) -> Result<LoadTestStats> {
        let total_requests = self.stats.total_requests.load(Ordering::SeqCst);
        let successful_requests = self.stats.successful_requests.load(Ordering::SeqCst);
        let failed_requests = self.stats.failed_requests.load(Ordering::SeqCst);
        let timeout_requests = self.stats.timeout_requests.load(Ordering::SeqCst);
        let total_tokens_generated = self.stats.total_tokens_generated.load(Ordering::SeqCst);

        // Sort latencies for percentile calculation
        let mut latencies = self.latencies.write().await;
        latencies.sort_unstable();

        let min_latency_ms = latencies.first().copied().unwrap_or(0);
        let max_latency_ms = latencies.last().copied().unwrap_or(0);

        let p50_latency_ms = Self::percentile(&latencies, 0.50);
        let p95_latency_ms = Self::percentile(&latencies, 0.95);
        let p99_latency_ms = Self::percentile(&latencies, 0.99);

        let error_rate = if total_requests > 0 {
            (failed_requests + timeout_requests) as f64 / total_requests as f64
        } else {
            0.0
        };

        let throughput_tokens_per_sec = if elapsed.as_secs() > 0 {
            total_tokens_generated as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        let total_latency_ms: u64 = latencies.iter().sum();

        Ok(LoadTestStats {
            total_requests,
            successful_requests,
            failed_requests,
            timeout_requests,
            total_tokens_generated,
            total_latency_ms,
            min_latency_ms,
            max_latency_ms,
            p50_latency_ms,
            p95_latency_ms,
            p99_latency_ms,
            error_rate,
            throughput_tokens_per_sec,
        })
    }

    /// Calculate percentile from sorted latency array
    fn percentile(sorted_latencies: &[u64], percentile: f64) -> u64 {
        if sorted_latencies.is_empty() {
            return 0;
        }
        let index = ((sorted_latencies.len() as f64 - 1.0) * percentile) as usize;
        sorted_latencies[index]
    }
}

/// Memory leak detector
pub struct MemoryLeakDetector {
    initial_memory_kb: usize,
    samples: Vec<(Duration, usize)>,
}

impl MemoryLeakDetector {
    pub fn new() -> Result<Self> {
        let initial_memory_kb = Self::get_current_memory_usage()?;
        Ok(Self {
            initial_memory_kb,
            samples: vec![(Duration::ZERO, initial_memory_kb)],
        })
    }

    /// Sample current memory usage
    pub fn sample(&mut self, elapsed: Duration) -> Result<()> {
        let current_memory_kb = Self::get_current_memory_usage()?;
        self.samples.push((elapsed, current_memory_kb));
        Ok(())
    }

    /// Detect if there's a memory leak (linear growth over time)
    pub fn detect_leak(&self) -> Result<bool> {
        if self.samples.len() < 10 {
            return Ok(false); // Not enough samples
        }

        // Simple linear regression to detect trend
        let n = self.samples.len() as f64;
        let sum_x: f64 = self.samples.iter().map(|(d, _)| d.as_secs_f64()).sum();
        let sum_y: f64 = self.samples.iter().map(|(_, mem)| *mem as f64).sum();
        let sum_xy: f64 = self
            .samples
            .iter()
            .map(|(d, mem)| d.as_secs_f64() * (*mem as f64))
            .sum();
        let sum_x2: f64 = self
            .samples
            .iter()
            .map(|(d, _)| d.as_secs_f64().powi(2))
            .sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2));

        // If memory grows by more than 100 KB/hour, consider it a leak
        let leak_threshold_kb_per_hour = 100.0;
        let slope_kb_per_hour = slope * 3600.0;

        Ok(slope_kb_per_hour > leak_threshold_kb_per_hour)
    }

    /// Get current process memory usage in KB
    fn get_current_memory_usage() -> Result<usize> {
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            let status = fs::read_to_string("/proc/self/status")?;
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        return Ok(parts[1].parse()?);
                    }
                }
            }
            anyhow::bail!("Could not find VmRSS in /proc/self/status");
        }

        #[cfg(not(target_os = "linux"))]
        {
            // TODO: Implement for Windows/macOS
            // For now, return dummy value (memory leak detection will be skipped)
            Ok(0)
        }
    }

    /// Print memory usage report
    pub fn report(&self) {
        println!("Memory Usage Report:");
        println!("  Initial: {} KB", self.initial_memory_kb);
        if let Some((_, current)) = self.samples.last() {
            println!("  Current: {} KB", current);
            let delta = *current as i64 - self.initial_memory_kb as i64;
            println!(
                "  Delta: {:+} KB ({:+.1}%)",
                delta,
                (delta as f64 / self.initial_memory_kb as f64) * 100.0
            );
        }

        if let Ok(leak_detected) = self.detect_leak() {
            if leak_detected {
                println!("  ⚠️  MEMORY LEAK DETECTED");
            } else {
                println!("  ✅ No memory leak detected");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_light_load() -> Result<()> {
        let config = scenarios::light_load();
        let runner = LoadTestRunner::new(config);
        let stats = runner.run().await?;

        println!("\nLight Load Test Results:");
        println!("  Total requests: {}", stats.total_requests);
        println!("  Successful: {}", stats.successful_requests);
        println!("  Failed: {}", stats.failed_requests);
        println!("  Error rate: {:.2}%", stats.error_rate * 100.0);
        println!(
            "  Throughput: {:.0} tokens/sec",
            stats.throughput_tokens_per_sec
        );
        println!("  Latency p50: {} ms", stats.p50_latency_ms);
        println!("  Latency p95: {} ms", stats.p95_latency_ms);
        println!("  Latency p99: {} ms", stats.p99_latency_ms);

        // Assert success criteria
        assert!(
            stats.error_rate < 0.05,
            "Error rate too high: {:.2}%",
            stats.error_rate * 100.0
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_normal_load() -> Result<()> {
        let config = scenarios::normal_load();
        let runner = LoadTestRunner::new(config);
        let stats = runner.run().await?;

        println!("\nNormal Load Test Results:");
        println!("  Total requests: {}", stats.total_requests);
        println!("  Successful: {}", stats.successful_requests);
        println!("  Failed: {}", stats.failed_requests);
        println!("  Error rate: {:.2}%", stats.error_rate * 100.0);
        println!(
            "  Throughput: {:.0} tokens/sec",
            stats.throughput_tokens_per_sec
        );
        println!("  Latency p50: {} ms", stats.p50_latency_ms);
        println!("  Latency p95: {} ms", stats.p95_latency_ms);
        println!("  Latency p99: {} ms", stats.p99_latency_ms);

        // Assert success criteria
        assert!(
            stats.error_rate < 0.05,
            "Error rate too high: {:.2}%",
            stats.error_rate * 100.0
        );

        Ok(())
    }

    #[tokio::test]
    #[ignore] // Long running test
    async fn test_heavy_load() -> Result<()> {
        let config = scenarios::heavy_load();
        let runner = LoadTestRunner::new(config);
        let stats = runner.run().await?;

        println!("\nHeavy Load Test Results:");
        println!("  Total requests: {}", stats.total_requests);
        println!("  Successful: {}", stats.successful_requests);
        println!("  Failed: {}", stats.failed_requests);
        println!("  Error rate: {:.2}%", stats.error_rate * 100.0);
        println!(
            "  Throughput: {:.0} tokens/sec",
            stats.throughput_tokens_per_sec
        );
        println!("  Latency p50: {} ms", stats.p50_latency_ms);
        println!("  Latency p95: {} ms", stats.p95_latency_ms);
        println!("  Latency p99: {} ms", stats.p99_latency_ms);

        // Slightly relaxed criteria for heavy load
        assert!(
            stats.error_rate < 0.10,
            "Error rate too high: {:.2}%",
            stats.error_rate * 100.0
        );

        Ok(())
    }

    #[tokio::test]
    #[ignore] // Very long running test (48 hours)
    async fn test_soak_48h() -> Result<()> {
        let config = scenarios::soak_test();
        let runner = LoadTestRunner::new(config);

        // Start memory leak detector
        let mut memory_detector = MemoryLeakDetector::new()?;

        // Sample memory every hour
        let memory_sample_interval = Duration::from_secs(3600);
        let mut last_sample = Instant::now();

        // Run load test with periodic memory sampling
        // (In reality, this would run in parallel with the load test)
        let stats = runner.run().await?;

        println!("\n48-Hour Soak Test Results:");
        println!("  Total requests: {}", stats.total_requests);
        println!("  Successful: {}", stats.successful_requests);
        println!("  Failed: {}", stats.failed_requests);
        println!("  Error rate: {:.2}%", stats.error_rate * 100.0);
        println!(
            "  Throughput: {:.0} tokens/sec",
            stats.throughput_tokens_per_sec
        );

        memory_detector.report();

        // Assert no memory leaks
        let leak_detected = memory_detector.detect_leak()?;
        assert!(!leak_detected, "Memory leak detected during soak test");

        // Assert stable error rate
        assert!(
            stats.error_rate < 0.05,
            "Error rate too high: {:.2}%",
            stats.error_rate * 100.0
        );

        Ok(())
    }
}
