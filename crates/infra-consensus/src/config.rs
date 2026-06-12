//! Configuration for the consensus layer
//!
//! This module provides configuration structures for Raft nodes and clusters.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use crate::{NodeId, ConsensusResult, ConsensusError};

/// Configuration for a Raft node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftConfig {
    /// Unique identifier for this node
    pub node_id: NodeId,
    
    /// Heartbeat interval for the leader
    pub heartbeat_interval: Duration,
    
    /// Election timeout range (min, max)
    pub election_timeout: (Duration, Duration),
    
    /// Maximum number of log entries per AppendEntries message
    pub max_append_entries: usize,
    
    /// Maximum size of a single log entry in bytes
    pub max_entry_size: usize,
    
    /// Whether to enable pre-vote to prevent disruptions
    pub enable_pre_vote: bool,
    
    /// Maximum number of in-flight replication requests
    pub max_replication_lag: u64,
    
    /// Snapshot threshold - take snapshot after this many log entries
    pub snapshot_threshold: u64,
    
    /// Whether to enable leader lease for read optimization
    pub enable_leader_lease: bool,
    
    /// Duration for leader lease
    pub leader_lease_duration: Duration,
}

impl Default for RaftConfig {
    fn default() -> Self {
        Self {
            node_id: 1,
            heartbeat_interval: Duration::from_millis(50),
            election_timeout: (Duration::from_millis(150), Duration::from_millis(300)),
            max_append_entries: 100,
            max_entry_size: 1024 * 1024, // 1MB
            enable_pre_vote: true,
            max_replication_lag: 1000,
            snapshot_threshold: 10000,
            enable_leader_lease: false,
            leader_lease_duration: Duration::from_millis(500),
        }
    }
}

impl RaftConfig {
    /// Create a new RaftConfig with the specified node ID
    pub fn new(node_id: NodeId) -> Self {
        Self {
            node_id,
            ..Default::default()
        }
    }
    
    /// Set the heartbeat interval
    pub fn with_heartbeat_interval(mut self, interval: Duration) -> Self {
        self.heartbeat_interval = interval;
        self
    }
    
    /// Set the election timeout range
    pub fn with_election_timeout(mut self, min: Duration, max: Duration) -> Self {
        if min >= max {
            // Use a reasonable default if invalid range is provided
            self.election_timeout = (Duration::from_millis(150), Duration::from_millis(300));
        } else {
            self.election_timeout = (min, max);
        }
        self
    }
    
    /// Set the maximum append entries count
    pub fn with_max_append_entries(mut self, count: usize) -> Self {
        self.max_append_entries = count.max(1); // At least 1
        self
    }
    
    /// Enable or disable pre-vote
    pub fn with_pre_vote(mut self, enable: bool) -> Self {
        self.enable_pre_vote = enable;
        self
    }
    
    /// Set snapshot threshold
    pub fn with_snapshot_threshold(mut self, threshold: u64) -> Self {
        self.snapshot_threshold = threshold.max(100); // At least 100
        self
    }
    
    /// Validate the configuration
    pub fn validate(&self) -> ConsensusResult<()> {
        if self.heartbeat_interval.is_zero() {
            return Err(ConsensusError::Config(
                "Heartbeat interval must be greater than zero".to_string(),
            ));
        }
        
        if self.election_timeout.0 >= self.election_timeout.1 {
            return Err(ConsensusError::Config(
                "Election timeout min must be less than max".to_string(),
            ));
        }
        
        if self.heartbeat_interval >= self.election_timeout.0 {
            return Err(ConsensusError::Config(
                "Heartbeat interval must be less than election timeout".to_string(),
            ));
        }
        
        if self.max_append_entries == 0 {
            return Err(ConsensusError::Config(
                "Max append entries must be greater than zero".to_string(),
            ));
        }
        
        if self.max_entry_size == 0 {
            return Err(ConsensusError::Config(
                "Max entry size must be greater than zero".to_string(),
            ));
        }
        
        Ok(())
    }
}

/// Configuration for a Raft cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// Initial cluster members (node_id -> address)
    pub members: HashMap<NodeId, String>,
    
    /// Bootstrap the cluster if no existing state
    pub bootstrap: bool,
    
    /// Cluster name for identification
    pub cluster_name: String,
    
    /// Data directory for persistent state
    pub data_dir: String,
    
    /// Network configuration
    pub network: NetworkConfig,
    
    /// Storage configuration
    pub storage: StorageConfig,
}

/// Network configuration for the cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Bind address for this node
    pub bind_address: String,
    
    /// Public address advertised to other nodes
    pub advertise_address: Option<String>,
    
    /// Connection timeout
    pub connect_timeout: Duration,
    
    /// Request timeout
    pub request_timeout: Duration,
    
    /// Maximum number of retry attempts
    pub max_retries: usize,
    
    /// Base delay for exponential backoff
    pub retry_base_delay: Duration,
    
    /// Maximum delay for exponential backoff
    pub retry_max_delay: Duration,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            bind_address: "127.0.0.1:8080".to_string(),
            advertise_address: None,
            connect_timeout: Duration::from_secs(3),
            request_timeout: Duration::from_secs(5),
            max_retries: 3,
            retry_base_delay: Duration::from_millis(100),
            retry_max_delay: Duration::from_secs(10),
        }
    }
}

/// Storage configuration for the cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Type of storage backend
    pub backend: StorageBackendType,
    
    /// Sync writes to disk
    pub sync_writes: bool,
    
    /// Compression for log entries
    pub compression: CompressionType,
    
    /// Maximum size of log file before rotation
    pub max_log_size: u64,
    
    /// Number of log files to keep
    pub log_retention: usize,
}

/// Types of storage backends
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum StorageBackendType {
    /// In-memory storage (for testing)
    Memory,
    /// File-based storage
    File,
    /// RocksDB storage
    RocksDB,
}

/// Compression types for log entries
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CompressionType {
    /// No compression
    None,
    /// LZ4 compression
    Lz4,
    /// Zstd compression
    Zstd,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            backend: StorageBackendType::File,
            sync_writes: true,
            compression: CompressionType::Lz4,
            max_log_size: 64 * 1024 * 1024, // 64MB
            log_retention: 10,
        }
    }
}

impl ClusterConfig {
    /// Create a new cluster configuration
    pub fn new(cluster_name: impl Into<String>, data_dir: impl Into<String>) -> Self {
        Self {
            members: HashMap::new(),
            bootstrap: false,
            cluster_name: cluster_name.into(),
            data_dir: data_dir.into(),
            network: NetworkConfig::default(),
            storage: StorageConfig::default(),
        }
    }
    
    /// Add a member to the cluster
    pub fn add_member(mut self, node_id: NodeId, address: impl Into<String>) -> Self {
        self.members.insert(node_id, address.into());
        self
    }
    
    /// Enable bootstrap mode
    pub fn with_bootstrap(mut self, bootstrap: bool) -> Self {
        self.bootstrap = bootstrap;
        self
    }
    
    /// Set network configuration
    pub fn with_network(mut self, network: NetworkConfig) -> Self {
        self.network = network;
        self
    }
    
    /// Set storage configuration
    pub fn with_storage(mut self, storage: StorageConfig) -> Self {
        self.storage = storage;
        self
    }
    
    /// Validate the cluster configuration
    pub fn validate(&self) -> ConsensusResult<()> {
        if self.cluster_name.is_empty() {
            return Err(ConsensusError::Config(
                "Cluster name cannot be empty".to_string(),
            ));
        }
        
        if self.data_dir.is_empty() {
            return Err(ConsensusError::Config(
                "Data directory cannot be empty".to_string(),
            ));
        }
        
        if self.bootstrap && self.members.is_empty() {
            return Err(ConsensusError::Config(
                "Bootstrap mode requires at least one member".to_string(),
            ));
        }
        
        // Validate network configuration
        if self.network.bind_address.is_empty() {
            return Err(ConsensusError::Config(
                "Bind address cannot be empty".to_string(),
            ));
        }
        
        if self.network.connect_timeout.is_zero() {
            return Err(ConsensusError::Config(
                "Connect timeout must be greater than zero".to_string(),
            ));
        }
        
        if self.network.request_timeout.is_zero() {
            return Err(ConsensusError::Config(
                "Request timeout must be greater than zero".to_string(),
            ));
        }
        
        // Validate storage configuration
        if self.storage.max_log_size == 0 {
            return Err(ConsensusError::Config(
                "Max log size must be greater than zero".to_string(),
            ));
        }
        
        if self.storage.log_retention == 0 {
            return Err(ConsensusError::Config(
                "Log retention must be greater than zero".to_string(),
            ));
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raft_config_default() {
        let config = RaftConfig::default();
        assert_eq!(config.node_id, 1);
        assert!(config.enable_pre_vote);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_raft_config_builder() {
        let config = RaftConfig::new(42)
            .with_heartbeat_interval(Duration::from_millis(100))
            .with_election_timeout(Duration::from_millis(200), Duration::from_millis(400))
            .with_pre_vote(false)
            .with_snapshot_threshold(5000);
        
        assert_eq!(config.node_id, 42);
        assert_eq!(config.heartbeat_interval, Duration::from_millis(100));
        assert_eq!(config.election_timeout.0, Duration::from_millis(200));
        assert_eq!(config.election_timeout.1, Duration::from_millis(400));
        assert!(!config.enable_pre_vote);
        assert_eq!(config.snapshot_threshold, 5000);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_election_timeout() {
        let config = RaftConfig::new(1)
            .with_election_timeout(Duration::from_millis(300), Duration::from_millis(200));
        
        // Should use default values for invalid range
        assert_eq!(config.election_timeout.0, Duration::from_millis(150));
        assert_eq!(config.election_timeout.1, Duration::from_millis(300));
    }

    #[test]
    fn test_cluster_config() {
        let config = ClusterConfig::new("test-cluster", "/tmp/test")
            .add_member(1, "127.0.0.1:8001")
            .add_member(2, "127.0.0.1:8002")
            .with_bootstrap(true);
        
        assert_eq!(config.cluster_name, "test-cluster");
        assert_eq!(config.data_dir, "/tmp/test");
        assert_eq!(config.members.len(), 2);
        assert!(config.bootstrap);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_cluster_config_validation() {
        let mut config = ClusterConfig::new("", "/tmp/test");
        assert!(config.validate().is_err());
        
        config.cluster_name = "test".to_string();
        config.data_dir = "".to_string();
        assert!(config.validate().is_err());
        
        config.data_dir = "/tmp/test".to_string();
        config.bootstrap = true;
        assert!(config.validate().is_err()); // No members for bootstrap
          config = config.add_member(1, "127.0.0.1:8001");
        assert!(config.validate().is_ok());
    }
}
