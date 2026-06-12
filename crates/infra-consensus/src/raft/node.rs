//! Core RaftNode implementation with OpenRaft

use openraft::{BasicNode, Snapshot as RaftSnapshot};
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::Response;

use super::config::RaftConfig;
use super::error::RaftError;
use super::network::RaftNetworkConnection;
use super::state::StateManager;
use super::storage::RaftStore;
use super::types::*;

/// Main Raft node implementation
#[derive(Clone)]
pub struct RaftNode {
    /// Node configuration
    config: RaftConfig,
    /// Node identifier
    node_id: u64,
    /// Storage layer
    store: Arc<RaftStore>,
    /// Network layer
    network: Arc<RaftNetworkConnection>,
    /// State machine
    state: Arc<StateManager>,
    /// Raft core (from OpenRaft)
    raft: Arc<openraft::Raft<TypeConfig>>,
    /// Node metrics
    metrics: Arc<RwLock<NodeMetrics>>,
}

/// Metrics for monitoring node health
#[derive(Debug, Default, Clone)]
pub struct NodeMetrics {
    /// Number of append entries received
    append_entries_received: u64,
    /// Number of votes cast
    votes_cast: u64,
    /// Number of leadership changes
    leadership_changes: u64,
    /// Last leadership change timestamp
    last_leadership_change: Option<chrono::DateTime<chrono::Utc>>,
}

impl RaftNode {
    /// Create a new RaftNode instance
    pub async fn new(
        node_id: u64,
        store: RaftStore,
        network: RaftNetworkConnection,
        state: StateManager,
        config: RaftConfig,
    ) -> Result<Self, RaftError> {
        // Create OpenRaft config
        let raft_config = super::default_config();

        // Create network factory
        let network_factory =
            super::network::RaftNetworkFactoryImpl::new(config.network_config.clone());

        // Create OpenRaft instance
        let raft = openraft::Raft::new(
            node_id,
            Arc::new(raft_config),
            network_factory,
            store.clone(),
            state.clone(),
        )
        .await
        .map_err(|e| RaftError::OpenRaft(format!("{:?}", e)))?;

        Ok(Self {
            config,
            node_id,
            store: Arc::new(store),
            network: Arc::new(network),
            state: Arc::new(state),
            raft: Arc::new(raft),
            metrics: Arc::new(RwLock::new(NodeMetrics::default())),
        })
    }
    /// Get node ID
    pub fn node_id(&self) -> u64 {
        self.node_id
    }

    /// Get node configuration
    pub fn config(&self) -> &RaftConfig {
        &self.config
    }

    /// Get storage reference
    pub fn store(&self) -> &Arc<RaftStore> {
        &self.store
    }
    /// Submit an entry to the Raft log
    pub async fn submit_entry(
        &self,
        command: Command,
    ) -> Result<Response<RaftResponse>, RaftError> {
        let result = self
            .raft
            .client_write(command)
            .await
            .map_err(|e| RaftError::OpenRaft(format!("{:?}", e)))?;
        // Convert ClientWriteResponse to our RaftResponse
        let response = RaftResponse {
            success: true,
            data: Some(serde_json::to_vec(&result.response()).unwrap_or_default()),
            error: None,
        };

        Ok(Response::new(response))
    }
    /// Add a node to the cluster
    pub async fn add_node(
        &self,
        node_id: u64,
        address: String,
    ) -> Result<Response<RaftResponse>, RaftError> {
        // Connect to the new node
        self.network.connect(node_id, address.clone()).await?;

        // Add node through Raft
        let node = BasicNode { addr: address };
        let result = self
            .raft
            .add_learner(node_id, node, true)
            .await
            .map_err(|e| RaftError::OpenRaft(format!("{:?}", e)))?;

        // Convert the result to our RaftResponse
        let response = RaftResponse {
            success: true,
            data: Some(serde_json::to_vec(&result.response()).unwrap_or_default()),
            error: None,
        };

        Ok(Response::new(response))
    }

    /// Remove a node from the cluster
    pub async fn remove_node(&self, node_id: u64) -> Result<Response<RaftResponse>, RaftError> {
        // Get current membership and remove this node
        // This is a simplified implementation - in practice you'd need to get current membership
        // and create a new membership without this node
        let members = std::collections::BTreeSet::new(); // Would contain remaining members

        let result = self
            .raft
            .change_membership(members, false)
            .await
            .map_err(|e| RaftError::OpenRaft(format!("{:?}", e)))?;

        // Clean up network connection
        self.network.remove_peer(node_id).await;

        // Convert the result to our RaftResponse
        let response = RaftResponse {
            success: true,
            data: Some(serde_json::to_vec(&result.response()).unwrap_or_default()),
            error: None,
        };

        Ok(Response::new(response))
    }

    /// Create a snapshot of the current state
    pub async fn create_snapshot(&self) -> Result<Response<RaftSnapshot<TypeConfig>>, RaftError> {
        let snapshot = self.state.create_snapshot().await?;
        Ok(Response::new(snapshot))
    }

    /// Get current metrics
    pub async fn metrics(&self) -> NodeMetrics {
        self.metrics.read().await.clone()
    }
    /// Check if this node is the leader
    pub async fn is_leader(&self) -> bool {
        // Use ensure_linearizable instead of deprecated is_leader
        self.raft.ensure_linearizable().await.is_ok()
    }

    /// Get current leader ID
    pub async fn current_leader(&self) -> Option<u64> {
        self.raft.current_leader().await
    }
}

impl NodeMetrics {
    /// Get the number of append entries received
    pub fn append_entries_received(&self) -> u64 {
        self.append_entries_received
    }

    /// Get the number of votes cast
    pub fn votes_cast(&self) -> u64 {
        self.votes_cast
    }

    /// Get the number of leadership changes
    pub fn leadership_changes(&self) -> u64 {
        self.leadership_changes
    }

    /// Get the last leadership change timestamp
    pub fn last_leadership_change(&self) -> Option<chrono::DateTime<chrono::Utc>> {
        self.last_leadership_change
    }
}
