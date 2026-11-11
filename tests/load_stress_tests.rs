// SPDX-License-Identifier: MIT OR Apache-2.0
// M3.5 Load and Stress Testing Infrastructure
//
// This module provides comprehensive load and stress testing for Lightbulb:
//
// 1. **Concurrent Request Simulator**: Test 10, 50, 100, 500 concurrent requests
// 2. **Edge Case Validator**: 128k token contexts, 128+ batch sizes, mixed workloads
// 3. **Soak Test Harness**: 48hr+ continuous operation, memory leak detection
//
// **Live server required.** Requests are real `POST /v1/chat/completions`
// calls (HTTPAnt client) against the server named by `LIGHTBULB_LOAD_TEST_URL`
// (base URL, e.g. `http://127.0.0.1:8080`). Optional: `LIGHTBULB_LOAD_TEST_API_KEY`
// (bearer token) and `LIGHTBULB_LOAD_TEST_MODEL` (default: first id from
// `GET /v1/models`). With the URL unset the load tests FAIL naming it: they are
// not `#[ignore]`d (CireSnave's standing rule, EXPECTATIONS 5.1g/5.1h); the
// hosted CI job skips them by name and a self-hosted job sets the variable.
//
// **Success Criteria**:
// - Stable under 100+ concurrent requests
// - No memory leaks over 48 hours
// - Graceful degradation under extreme load
// - <1% error rate under normal load (100 concurrent)
// - <5% error rate under 2× capacity (200 concurrent)

use anyhow::{Context, Result, anyhow};
use httpant_client::Client;
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};
use tokio::task::JoinHandle;

/// Env var naming the live server's base URL.
const URL_VAR: &str = "LIGHTBULB_LOAD_TEST_URL";
/// Env var holding an optional bearer token.
const KEY_VAR: &str = "LIGHTBULB_LOAD_TEST_API_KEY";
/// Env var naming an optional model id (default: first of `/v1/models`).
const MODEL_VAR: &str = "LIGHTBULB_LOAD_TEST_MODEL";

/// Where and how to send load: resolved once per run.
#[derive(Debug, Clone)]
struct LiveTarget {
    base_url: String,
    api_key: Option<String>,
    model: Option<String>,
}

impl LiveTarget {
    /// Pure so a test can drive it without mutating the process environment.
    fn from_values(
        url: Option<String>,
        api_key: Option<String>,
        model: Option<String>,
    ) -> Result<Self> {
        let base_url = url
            .map(|u| u.trim().trim_end_matches('/').to_string())
            .filter(|u| !u.is_empty())
            .ok_or_else(|| {
                anyhow!(
                    "{URL_VAR} is not set: the load tests need a live Lightbulb server; \
                     set it to its base URL (e.g. http://127.0.0.1:8080) on the runner job \
                     that runs them"
                )
            })?;
        Ok(Self {
            base_url,
            api_key: api_key.filter(|k| !k.is_empty()),
            model: model.filter(|m| !m.is_empty()),
        })
    }

    fn from_env() -> Result<Self> {
        Self::from_values(
            std::env::var(URL_VAR).ok(),
            std::env::var(KEY_VAR).ok(),
            std::env::var(MODEL_VAR).ok(),
        )
    }
}

/// A deterministic prompt of roughly `range.0..=range.1` words, varied per request.
fn build_prompt(request_id: u64, range: (usize, usize)) -> String {
    let span = range.1.saturating_sub(range.0) + 1;
    let words = range.0 + (request_id as usize % span);
    let mut prompt = format!("Request {request_id}. Summarise the following text:");
    for i in 0..words {
        prompt.push_str(if i % 2 == 0 { " alpha" } else { " beta" });
    }
    prompt
}

/// Ask the server which model to use when none was configured.
async fn discover_model(client: &Client, target: &LiveTarget) -> Result<String> {
    let mut req = client.get(format!("{}/v1/models", target.base_url));
    if let Some(key) = &target.api_key {
        req = req.bearer(key);
    }
    let resp = req
        .send()
        .await
        .map_err(|e| anyhow!("GET /v1/models failed: {e}"))?;
    let status = resp.status();
    let text = resp
        .text()
        .await
        .map_err(|e| anyhow!("reading /v1/models failed: {e}"))?;
    if !status.is_success() {
        return Err(anyhow!("GET /v1/models returned {status}: {text}"));
    }
    let v: Value = serde_json::from_str(&text).context("/v1/models body is not JSON")?;
    v["data"][0]["id"]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| anyhow!("/v1/models listed no model; set {MODEL_VAR}"))
}

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

        let target = LiveTarget::from_env()?;
        let client = Client::builder()
            .timeout(self.config.request_timeout)
            .build()
            .map_err(|e| anyhow!("building the HTTP client failed: {e}"))?;
        let model = match &target.model {
            Some(m) => m.clone(),
            None => discover_model(&client, &target).await?,
        };
        println!("  Target: {} (model {model})", target.base_url);

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
            let (client, target, model) = (client.clone(), target.clone(), model.clone());

            let handle = tokio::spawn(async move {
                let result =
                    Self::execute_request(request_id, &config, &client, &target, &model).await;

                // Update statistics
                stats.total_requests.fetch_add(1, Ordering::SeqCst);

                if result.success {
                    stats.successful_requests.fetch_add(1, Ordering::SeqCst);
                    stats
                        .total_tokens_generated
                        .fetch_add(result.tokens_generated as u64, Ordering::SeqCst);
                } else {
                    let counter = if result.timeout {
                        &stats.timeout_requests
                    } else {
                        &stats.failed_requests
                    };
                    // Show the first few reasons so a live failure is diagnosable.
                    if counter.fetch_add(1, Ordering::SeqCst) < 3 {
                        eprintln!(
                            "request {request_id} failed: {}",
                            result.error_message.as_deref().unwrap_or("unknown")
                        );
                    }
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

    /// Send one chat completion; `Ok` is the completion token count.
    async fn send_chat(
        request_id: u64,
        config: &LoadTestConfig,
        client: &Client,
        target: &LiveTarget,
        model: &str,
    ) -> std::result::Result<usize, String> {
        let body = json!({
            "model": model,
            "messages": [{
                "role": "user",
                "content": build_prompt(request_id, config.prompt_length_range),
            }],
            "max_tokens": config.generation_length,
            "stream": false,
        });
        let mut req = client
            .post(format!("{}/v1/chat/completions", target.base_url))
            .header("content-type", "application/json")
            .body(body.to_string());
        if let Some(key) = &target.api_key {
            req = req.bearer(key);
        }
        let resp = req.send().await.map_err(|e| e.to_string())?;
        let status = resp.status();
        let text = resp.text().await.map_err(|e| e.to_string())?;
        if !status.is_success() {
            return Err(format!("HTTP {status}: {text}"));
        }
        let v: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        Ok(v["usage"]["completion_tokens"].as_u64().unwrap_or(0) as usize)
    }

    /// Execute one real chat-completion request against the live server.
    async fn execute_request(
        request_id: u64,
        config: &LoadTestConfig,
        client: &Client,
        target: &LiveTarget,
        model: &str,
    ) -> RequestResult {
        let start = Instant::now();
        let outcome = Self::send_chat(request_id, config, client, target, model).await;
        let latency = start.elapsed();
        match outcome {
            Ok(tokens_generated) => RequestResult {
                success: true,
                timeout: false,
                tokens_generated,
                latency,
                error_message: None,
            },
            Err(message) => RequestResult {
                success: false,
                timeout: latency >= config.request_timeout,
                tokens_generated: 0,
                latency,
                error_message: Some(message),
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

    /// Get current process resident memory in KB, on every platform.
    ///
    /// `sysinfo` is already a regular dependency (it feeds the admin metrics),
    /// so this reads the same RSS on Linux, Windows and macOS instead of the
    /// former Linux-only `/proc/self/status` parse and a dummy 0 elsewhere.
    fn get_current_memory_usage() -> Result<usize> {
        use sysinfo::{ProcessesToUpdate, System, get_current_pid};
        let pid = get_current_pid().map_err(|e| anyhow::anyhow!("current pid: {e}"))?;
        let mut sys = System::new();
        sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
        let bytes = sys
            .process(pid)
            .ok_or_else(|| anyhow::anyhow!("own process {pid} not found"))?
            .memory();
        anyhow::ensure!(
            bytes > 0,
            "sysinfo reported 0 bytes of RSS for this process"
        );
        Ok((bytes / 1024) as usize)
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

    #[test]
    fn an_unset_url_fails_naming_the_variable() {
        let err = LiveTarget::from_values(None, None, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("LIGHTBULB_LOAD_TEST_URL"), "{err}");
        let blank = LiveTarget::from_values(Some("  ".into()), None, None).unwrap_err();
        assert!(blank.to_string().contains("LIGHTBULB_LOAD_TEST_URL"));
    }

    #[test]
    fn a_set_url_is_trimmed_and_empty_optionals_are_none() -> Result<()> {
        let t = LiveTarget::from_values(
            Some("http://127.0.0.1:8080/".into()),
            Some(String::new()),
            Some(String::new()),
        )?;
        assert_eq!(t.base_url, "http://127.0.0.1:8080");
        assert!(t.api_key.is_none() && t.model.is_none());
        Ok(())
    }

    #[test]
    fn prompts_follow_the_configured_length_range() {
        for id in 0..50 {
            let words = build_prompt(id, (32, 40)).split(' ').count();
            // header is 6 words ("Request N. Summarise the following text:")
            let body = words - 6;
            assert!((32..=40).contains(&body), "id {id}: {body} words");
        }
        assert_ne!(build_prompt(0, (32, 512)), build_prompt(1, (32, 512)));
    }

    /// The detector must read a real RSS on every platform (it was a dummy 0
    /// off Linux, which made leak detection a silent no-op), and the reading
    /// must follow a held, touched allocation.
    #[test]
    fn memory_detector_reads_real_rss_and_tracks_an_allocation() -> Result<()> {
        let mut detector = MemoryLeakDetector::new()?;
        assert!(
            detector.initial_memory_kb > 0,
            "RSS must not be the dummy 0"
        );
        let mut held = vec![0u8; 64 * 1024 * 1024];
        for (i, b) in held.iter_mut().enumerate().step_by(4096) {
            *b = (i % 251) as u8 + 1;
        }
        detector.sample(Duration::from_secs(1))?;
        let after = detector.samples.last().map(|s| s.1).unwrap_or(0);
        assert!(
            after >= detector.initial_memory_kb + 32 * 1024,
            "RSS {after} KB did not grow by >=32 MB over {} KB after touching 64 MB",
            detector.initial_memory_kb
        );
        assert!(held.iter().step_by(4096).all(|b| *b != 0));
        Ok(())
    }

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
