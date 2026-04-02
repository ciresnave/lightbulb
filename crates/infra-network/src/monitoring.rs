//! Network monitoring and health tracking.

use crate::types::{NodeId, NetworkResult};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tokio::time::{interval, Instant};

/// Network monitor for tracking health and performance metrics
#[derive(Debug, Clone)]
pub struct NetworkMonitor {
    /// Node being monitored
    node_id: NodeId,
    /// Monitoring configuration
    config: MonitoringConfig,
    /// Current network metrics
    metrics: Arc<RwLock<NetworkMetrics>>,
    /// Peer health information
    peer_health: Arc<RwLock<HashMap<NodeId, PeerHealth>>>,
    /// Monitor start time
    start_time: Instant,
}

/// Configuration for network monitoring
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Monitoring update interval
    pub update_interval: Duration,
    /// Health check timeout
    pub health_check_timeout: Duration,
    /// Number of samples to keep for averaging
    pub sample_window: usize,
    /// Threshold for considering a peer unhealthy (ms)
    pub unhealthy_latency_threshold: u64,
    /// Maximum allowed packet loss percentage
    pub max_packet_loss: f64,
}

/// Network performance metrics
#[derive(Debug)]
pub struct NetworkMetrics {
    /// Total bytes sent
    pub bytes_sent: AtomicU64,
    /// Total bytes received
    pub bytes_received: AtomicU64,
    /// Total messages sent
    pub messages_sent: AtomicU64,
    /// Total messages received
    pub messages_received: AtomicU64,
    /// Current active connections
    pub active_connections: u64,
    /// Average latency in milliseconds
    pub average_latency: f64,
    /// Packet loss percentage
    pub packet_loss: f64,
    /// Network utilization percentage
    pub network_utilization: f64,
    /// Last update time
    pub last_updated: SystemTime,
}

/// Health information for a peer
#[derive(Debug, Clone)]
pub struct PeerHealth {
    /// Peer node ID
    pub peer_id: NodeId,
    /// Current health status
    pub status: HealthStatus,
    /// Average latency to peer (ms)
    pub average_latency: f64,
    /// Packet loss to peer (percentage)
    pub packet_loss: f64,
    /// Connection uptime
    pub uptime: Duration,
    /// Last health check time
    pub last_check: SystemTime,
    /// Historical latency samples
    pub latency_samples: Vec<u64>,
    /// Connection failures count
    pub failure_count: u64,
}

/// Health status of a peer
#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    /// Peer is healthy and responsive
    Healthy,
    /// Peer is experiencing issues but still reachable
    Degraded,
    /// Peer is unreachable or unresponsive
    Unhealthy,
    /// Health status unknown
    Unknown,
}

/// Network health report
#[derive(Debug, Clone)]
pub struct HealthReport {
    /// Overall network health score (0.0 to 1.0)
    pub overall_health: f64,
    /// Number of healthy peers
    pub healthy_peers: usize,
    /// Number of degraded peers
    pub degraded_peers: usize,
    /// Number of unhealthy peers
    pub unhealthy_peers: usize,
    /// Network performance summary
    pub performance_summary: PerformanceSummary,
    /// Report generation time
    pub generated_at: SystemTime,
}

/// Performance summary
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    /// Average latency across all peers
    pub average_latency: f64,
    /// Maximum latency observed
    pub max_latency: f64,
    /// Minimum latency observed
    pub min_latency: f64,
    /// Overall packet loss percentage
    pub packet_loss: f64,
    /// Network throughput (bytes/sec)
    pub throughput: f64,
}

/// Network alert
#[derive(Debug, Clone)]
pub struct NetworkAlert {
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert message
    pub message: String,
    /// Affected peer (if applicable)
    pub peer_id: Option<NodeId>,
    /// Alert timestamp
    pub timestamp: SystemTime,
    /// Additional context
    pub context: HashMap<String, String>,
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning condition
    Warning,
    /// Error condition
    Error,
    /// Critical condition
    Critical,
}

impl NetworkMonitor {
    /// Create a new network monitor
    pub fn new(node_id: NodeId, config: MonitoringConfig) -> Self {
        Self {
            node_id,
            config,
            metrics: Arc::new(RwLock::new(NetworkMetrics::new())),
            peer_health: Arc::new(RwLock::new(HashMap::new())),
            start_time: Instant::now(),
        }
    }

    /// Start monitoring background tasks
    pub async fn start_monitoring(&self) {
        let metrics = Arc::clone(&self.metrics);
        let peer_health = Arc::clone(&self.peer_health);
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut update_interval = interval(config.update_interval);
            
            loop {
                update_interval.tick().await;
                Self::update_metrics(
                    Arc::clone(&metrics),
                    Arc::clone(&peer_health),
                    &config,
                ).await;
            }
        });
    }

    /// Record a message sent
    pub async fn record_message_sent(&self, bytes: u64) {
        let metrics = self.metrics.read().await;
        metrics.messages_sent.fetch_add(1, Ordering::Relaxed);
        metrics.bytes_sent.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record a message received
    pub async fn record_message_received(&self, bytes: u64) {
        let metrics = self.metrics.read().await;
        metrics.messages_received.fetch_add(1, Ordering::Relaxed);
        metrics.bytes_received.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record latency measurement for a peer
    pub async fn record_latency(&self, peer_id: &NodeId, latency_ms: u64) -> NetworkResult<()> {
        let mut peer_health = self.peer_health.write().await;
        
        let health = peer_health.entry(peer_id.clone()).or_insert_with(|| {
            PeerHealth::new(peer_id.clone())
        });

        health.add_latency_sample(latency_ms, self.config.sample_window);
        health.update_status(&self.config);

        Ok(())
    }

    /// Record connection failure for a peer
    pub async fn record_connection_failure(&self, peer_id: &NodeId) -> NetworkResult<()> {
        let mut peer_health = self.peer_health.write().await;
        
        let health = peer_health.entry(peer_id.clone()).or_insert_with(|| {
            PeerHealth::new(peer_id.clone())
        });

        health.failure_count += 1;
        health.status = HealthStatus::Unhealthy;
        health.last_check = SystemTime::now();

        Ok(())
    }

    /// Update peer connection status
    pub async fn update_peer_status(&self, peer_id: &NodeId, connected: bool) -> NetworkResult<()> {
        let mut peer_health = self.peer_health.write().await;
        
        let health = peer_health.entry(peer_id.clone()).or_insert_with(|| {
            PeerHealth::new(peer_id.clone())
        });

        if connected {
            health.status = HealthStatus::Healthy;
        } else {
            health.status = HealthStatus::Unhealthy;
        }
        health.last_check = SystemTime::now();

        Ok(())
    }

    /// Get current network metrics
    pub async fn get_metrics(&self) -> NetworkResult<NetworkMetrics> {
        let metrics = self.metrics.read().await;
        Ok(metrics.clone())
    }

    /// Get health information for a specific peer
    pub async fn get_peer_health(&self, peer_id: &NodeId) -> NetworkResult<Option<PeerHealth>> {
        let peer_health = self.peer_health.read().await;
        Ok(peer_health.get(peer_id).cloned())
    }

    /// Get health information for all peers
    pub async fn get_all_peer_health(&self) -> NetworkResult<Vec<PeerHealth>> {
        let peer_health = self.peer_health.read().await;
        Ok(peer_health.values().cloned().collect())
    }

    /// Generate a comprehensive health report
    pub async fn generate_health_report(&self) -> NetworkResult<HealthReport> {
        let peer_health = self.peer_health.read().await;
        let metrics = self.metrics.read().await;

        let mut healthy_peers = 0;
        let mut degraded_peers = 0;
        let mut unhealthy_peers = 0;

        let mut latencies = Vec::new();
        let mut total_packet_loss = 0.0;

        for health in peer_health.values() {
            match health.status {
                HealthStatus::Healthy => healthy_peers += 1,
                HealthStatus::Degraded => degraded_peers += 1,
                HealthStatus::Unhealthy => unhealthy_peers += 1,
                HealthStatus::Unknown => {}
            }

            latencies.push(health.average_latency);
            total_packet_loss += health.packet_loss;
        }

        let total_peers = peer_health.len();
        let overall_health = if total_peers > 0 {
            (healthy_peers as f64 + 0.5 * degraded_peers as f64) / total_peers as f64
        } else {
            1.0
        };

        let average_latency = if !latencies.is_empty() {
            latencies.iter().sum::<f64>() / latencies.len() as f64
        } else {
            0.0
        };

        let performance_summary = PerformanceSummary {
            average_latency,
            max_latency: latencies.iter().fold(0.0, |a, &b| a.max(b)),
            min_latency: latencies.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            packet_loss: if total_peers > 0 { total_packet_loss / total_peers as f64 } else { 0.0 },
            throughput: self.calculate_throughput(&metrics).await,
        };

        Ok(HealthReport {
            overall_health,
            healthy_peers,
            degraded_peers,
            unhealthy_peers,
            performance_summary,
            generated_at: SystemTime::now(),
        })
    }

    /// Check for network alerts
    pub async fn check_alerts(&self) -> NetworkResult<Vec<NetworkAlert>> {
        let mut alerts = Vec::new();
        let peer_health = self.peer_health.read().await;

        for health in peer_health.values() {
            // Check for high latency
            if health.average_latency > self.config.unhealthy_latency_threshold as f64 {
                alerts.push(NetworkAlert {
                    severity: AlertSeverity::Warning,
                    message: format!("High latency detected: {:.2}ms", health.average_latency),
                    peer_id: Some(health.peer_id.clone()),
                    timestamp: SystemTime::now(),
                    context: HashMap::new(),
                });
            }

            // Check for unhealthy peers
            if health.status == HealthStatus::Unhealthy {
                alerts.push(NetworkAlert {
                    severity: AlertSeverity::Error,
                    message: "Peer is unhealthy".to_string(),
                    peer_id: Some(health.peer_id.clone()),
                    timestamp: SystemTime::now(),
                    context: HashMap::new(),
                });
            }

            // Check for packet loss
            if health.packet_loss > self.config.max_packet_loss {
                alerts.push(NetworkAlert {
                    severity: AlertSeverity::Warning,
                    message: format!("High packet loss: {:.2}%", health.packet_loss),
                    peer_id: Some(health.peer_id.clone()),
                    timestamp: SystemTime::now(),
                    context: HashMap::new(),
                });
            }
        }

        Ok(alerts)
    }

    /// Get monitoring uptime
    pub fn get_uptime(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Calculate network throughput
    async fn calculate_throughput(&self, metrics: &NetworkMetrics) -> f64 {
        let uptime = self.get_uptime();
        if uptime.as_secs() == 0 {
            return 0.0;
        }

        let total_bytes = metrics.bytes_sent.load(Ordering::Relaxed) + 
                         metrics.bytes_received.load(Ordering::Relaxed);
        
        total_bytes as f64 / uptime.as_secs() as f64
    }    /// Update network metrics periodically
    async fn update_metrics(
        metrics: Arc<RwLock<NetworkMetrics>>,
        peer_health: Arc<RwLock<HashMap<NodeId, PeerHealth>>>,
        _config: &MonitoringConfig,
    ) {
        let mut metrics = metrics.write().await;
        let peer_health = peer_health.read().await;

        // Update active connections
        metrics.active_connections = peer_health
            .values()
            .filter(|h| h.status == HealthStatus::Healthy)
            .count() as u64;

        // Calculate average latency
        let latencies: Vec<f64> = peer_health
            .values()
            .map(|h| h.average_latency)
            .collect();

        metrics.average_latency = if !latencies.is_empty() {
            latencies.iter().sum::<f64>() / latencies.len() as f64
        } else {
            0.0
        };

        // Calculate packet loss
        let packet_losses: Vec<f64> = peer_health
            .values()
            .map(|h| h.packet_loss)
            .collect();

        metrics.packet_loss = if !packet_losses.is_empty() {
            packet_losses.iter().sum::<f64>() / packet_losses.len() as f64
        } else {
            0.0
        };

        metrics.last_updated = SystemTime::now();
    }
}

impl NetworkMetrics {
    /// Create new network metrics
    pub fn new() -> Self {
        Self {
            bytes_sent: AtomicU64::new(0),
            bytes_received: AtomicU64::new(0),
            messages_sent: AtomicU64::new(0),
            messages_received: AtomicU64::new(0),
            active_connections: 0,
            average_latency: 0.0,
            packet_loss: 0.0,
            network_utilization: 0.0,
            last_updated: SystemTime::now(),
        }
    }
}

impl Clone for NetworkMetrics {
    fn clone(&self) -> Self {
        Self {
            bytes_sent: AtomicU64::new(self.bytes_sent.load(Ordering::Relaxed)),
            bytes_received: AtomicU64::new(self.bytes_received.load(Ordering::Relaxed)),
            messages_sent: AtomicU64::new(self.messages_sent.load(Ordering::Relaxed)),
            messages_received: AtomicU64::new(self.messages_received.load(Ordering::Relaxed)),
            active_connections: self.active_connections,
            average_latency: self.average_latency,
            packet_loss: self.packet_loss,
            network_utilization: self.network_utilization,
            last_updated: self.last_updated,
        }
    }
}

impl PeerHealth {
    /// Create new peer health tracker
    pub fn new(peer_id: NodeId) -> Self {
        Self {
            peer_id,
            status: HealthStatus::Unknown,
            average_latency: 0.0,
            packet_loss: 0.0,
            uptime: Duration::new(0, 0),
            last_check: SystemTime::now(),
            latency_samples: Vec::new(),
            failure_count: 0,
        }
    }

    /// Add a latency sample and update average
    pub fn add_latency_sample(&mut self, latency_ms: u64, window_size: usize) {
        self.latency_samples.push(latency_ms);
        
        // Keep only the last N samples
        if self.latency_samples.len() > window_size {
            self.latency_samples.remove(0);
        }

        // Update average
        self.average_latency = self.latency_samples.iter().sum::<u64>() as f64 / 
                              self.latency_samples.len() as f64;
        
        self.last_check = SystemTime::now();
    }

    /// Update health status based on current metrics
    pub fn update_status(&mut self, config: &MonitoringConfig) {
        if self.average_latency > config.unhealthy_latency_threshold as f64 ||
           self.packet_loss > config.max_packet_loss ||
           self.failure_count > 5 {
            self.status = HealthStatus::Unhealthy;
        } else if self.average_latency > (config.unhealthy_latency_threshold as f64 * 0.7) ||
                  self.packet_loss > (config.max_packet_loss * 0.7) {
            self.status = HealthStatus::Degraded;
        } else {
            self.status = HealthStatus::Healthy;
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            update_interval: Duration::from_secs(30),
            health_check_timeout: Duration::from_secs(5),
            sample_window: 10,
            unhealthy_latency_threshold: 1000, // 1 second
            max_packet_loss: 5.0, // 5%
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;    #[tokio::test]
    async fn test_network_monitor_creation() {
        let node_id = NodeId::from_string("test_node");
        let config = MonitoringConfig::default();
        let monitor = NetworkMonitor::new(node_id, config);

        let metrics = monitor.get_metrics().await.unwrap();
        assert_eq!(metrics.bytes_sent.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.active_connections, 0);
    }    #[tokio::test]
    async fn test_record_message_metrics() {
        let node_id = NodeId::from_string("test_node");
        let config = MonitoringConfig::default();
        let monitor = NetworkMonitor::new(node_id, config);

        monitor.record_message_sent(100).await;
        monitor.record_message_received(200).await;

        let metrics = monitor.get_metrics().await.unwrap();
        assert_eq!(metrics.bytes_sent.load(Ordering::Relaxed), 100);
        assert_eq!(metrics.bytes_received.load(Ordering::Relaxed), 200);
        assert_eq!(metrics.messages_sent.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.messages_received.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]    async fn test_record_latency() {
        let node_id = NodeId::from_string("test_node");
        let peer_id = NodeId::from_string("test_peer");
        let config = MonitoringConfig::default();
        let monitor = NetworkMonitor::new(node_id, config);

        monitor.record_latency(&peer_id, 50).await.unwrap();
        monitor.record_latency(&peer_id, 60).await.unwrap();

        let health = monitor.get_peer_health(&peer_id).await.unwrap();
        assert!(health.is_some());
        
        let health = health.unwrap();
        assert_eq!(health.average_latency, 55.0);
        assert_eq!(health.status, HealthStatus::Healthy);
    }

    #[tokio::test]    async fn test_generate_health_report() {
        let node_id = NodeId::from_string("test_node");
        let peer_id = NodeId::from_string("test_peer");
        let config = MonitoringConfig::default();
        let monitor = NetworkMonitor::new(node_id, config);

        monitor.record_latency(&peer_id, 50).await.unwrap();
        monitor.update_peer_status(&peer_id, true).await.unwrap();

        let report = monitor.generate_health_report().await.unwrap();
        assert_eq!(report.healthy_peers, 1);
        assert_eq!(report.degraded_peers, 0);
        assert_eq!(report.unhealthy_peers, 0);
        assert!(report.overall_health > 0.9);
    }

    #[tokio::test]    async fn test_check_alerts() {
        let node_id = NodeId::from_string("test_node");
        let peer_id = NodeId::from_string("test_peer");
        let config = MonitoringConfig::default();
        let monitor = NetworkMonitor::new(node_id, config);

        // Record high latency
        monitor.record_latency(&peer_id, 2000).await.unwrap();

        let alerts = monitor.check_alerts().await.unwrap();
        assert!(!alerts.is_empty());
        assert_eq!(alerts[0].severity, AlertSeverity::Warning);
    }

    #[test]    fn test_peer_health() {
        let peer_id = NodeId::from_string("test_peer");
        let mut health = PeerHealth::new(peer_id);
        let config = MonitoringConfig::default();

        assert_eq!(health.status, HealthStatus::Unknown);

        health.add_latency_sample(50, 10);
        health.update_status(&config);
        assert_eq!(health.status, HealthStatus::Healthy);

        health.add_latency_sample(2000, 10);
        health.update_status(&config);
        assert_eq!(health.status, HealthStatus::Unhealthy);
    }

    #[test]
    fn test_monitoring_config_default() {
        let config = MonitoringConfig::default();
        
        assert_eq!(config.update_interval, Duration::from_secs(30));
        assert_eq!(config.health_check_timeout, Duration::from_secs(5));
        assert_eq!(config.sample_window, 10);
        assert_eq!(config.unhealthy_latency_threshold, 1000);
        assert_eq!(config.max_packet_loss, 5.0);
    }

    #[test]
    fn test_network_metrics() {
        let metrics = NetworkMetrics::new();
        
        assert_eq!(metrics.bytes_sent.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.bytes_received.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.active_connections, 0);
        assert_eq!(metrics.average_latency, 0.0);
    }
}
