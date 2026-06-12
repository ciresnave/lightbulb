//! Configuration for Raft consensus nodes

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuration for a Raft node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftConfig {
    /// Unique identifier for this node
    pub node_id: u64,

    /// Network configuration
    pub network_config: NetworkConfig,

    /// Storage configuration
    pub storage_config: StorageConfig,

    /// State machine configuration
    pub state_config: StateConfig,

    /// Path for persistent storage
    pub storage_path: PathBuf,

    /// Cluster configuration
    pub cluster: ClusterConfig,
}

/// Network-related configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Address to listen on
    pub listen_addr: String,

    /// Port for Raft communication
    pub raft_port: u16,

    /// Port for cluster management
    pub mgmt_port: u16,

    /// Timeout for network operations (ms)
    pub timeout_ms: u64,

    /// Maximum message size
    pub max_message_size: usize,
}

/// Storage-related configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Maximum size of the log before compaction
    pub max_log_size: usize,

    /// How often to create snapshots
    pub snapshot_interval: u64,

    /// Whether to sync writes to disk
    pub sync_writes: bool,

    /// Cache size for storage operations
    pub cache_size: usize,
}

/// State machine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateConfig {
    /// Maximum size of state machine data
    pub max_state_size: usize,

    /// Whether to validate state transitions
    pub validate_transitions: bool,

    /// Custom state machine settings
    pub custom_settings: std::collections::HashMap<String, String>,
}

/// Cluster-wide configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// Initial cluster members
    pub initial_members: std::collections::BTreeMap<u64, String>,

    /// Election timeout range (min, max) in ms
    pub election_timeout_range: (u64, u64),

    /// Heartbeat interval in ms
    pub heartbeat_interval: u64,

    /// Maximum concurrent replication streams
    pub max_replication_streams: usize,
}

impl Default for RaftConfig {
    fn default() -> Self {
        Self {
            node_id: 1,
            network_config: NetworkConfig::default(),
            storage_config: StorageConfig::default(),
            state_config: StateConfig::default(),
            storage_path: PathBuf::from("./raft-data"),
            cluster: ClusterConfig::default(),
        }
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            listen_addr: "127.0.0.1".to_string(),
            raft_port: 8000,
            mgmt_port: 8001,
            timeout_ms: 5000,
            max_message_size: 64 * 1024 * 1024, // 64MB
        }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            max_log_size: 1024 * 1024 * 1024, // 1GB
            snapshot_interval: 10000,
            sync_writes: true,
            cache_size: 128 * 1024 * 1024, // 128MB
        }
    }
}

impl Default for StateConfig {
    fn default() -> Self {
        Self {
            max_state_size: 1024 * 1024 * 1024, // 1GB
            validate_transitions: true,
            custom_settings: std::collections::HashMap::new(),
        }
    }
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            initial_members: std::collections::BTreeMap::new(),
            election_timeout_range: (1500, 3000),
            heartbeat_interval: 500,
            max_replication_streams: 3,
        }
    }
}
