//! Raft consensus implementation using OpenRaft
//!
//! This module provides a high-level wrapper around OpenRaft for distributed consensus
//! in DynAniML clusters. It includes:
//! - Node management and configuration
//! - Log replication and state machine
//! - Leader election and membership changes
//! - Metrics and monitoring

use std::sync::Arc;

use anyhow::Result;
use openraft::{self};

pub mod config;
pub mod error;
pub mod network;
pub mod node;
pub mod state;
pub mod storage;
pub mod types;

pub use config::RaftConfig;
pub use error::RaftError;
pub use network::RaftNetworkConnection;
pub use node::RaftNode;
pub use state::StateManager;
pub use storage::RaftStore;
pub use types::*;

/// Type alias for OpenRaft's configuration
pub type Config = openraft::Config;

/// Default Raft configuration optimized for DynAniML clusters
pub fn default_config() -> Config {
    Config {
        heartbeat_interval: 500,    // 500ms between heartbeats
        election_timeout_min: 1500, // Min 1.5s election timeout
        election_timeout_max: 3000, // Max 3s election timeout
        snapshot_policy: openraft::SnapshotPolicy::LogsSinceLast(5000),
        max_payload_entries: 1000,       // Max 1000 entries per append
        replication_lag_threshold: 1000, // Alert if replica lags by 1000 entries
        ..Default::default()
    }
}

/// Create a new RaftNode with the given configuration
pub async fn create_node(node_id: u64, config: RaftConfig) -> Result<Arc<RaftNode>, RaftError> {
    let store = RaftStore::new(config.storage_path.clone())?;
    let network = RaftNetworkConnection::new(config.network_config.clone())?;
    let state = StateManager::new(config.state_config.clone())?;

    let node = RaftNode::new(node_id, store, network, state, config).await?;
    Ok(Arc::new(node))
}
