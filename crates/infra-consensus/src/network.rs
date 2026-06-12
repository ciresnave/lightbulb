//! Network transport for Raft consensus
//!
//! This module provides networking abstractions for Raft node communication.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, warn};

use crate::{NodeId, ConsensusResult, ConsensusError};

/// Types of Raft messages exchanged between nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RaftMessage {
    /// Vote request during leader election
    VoteRequest {
        /// Term of the candidate
        term: u64,
        /// Candidate requesting the vote
        candidate_id: NodeId,
        /// Index of candidate's last log entry
        last_log_index: u64,
        /// Term of candidate's last log entry
        last_log_term: u64,
    },
    
    /// Vote response
    VoteResponse {
        /// Current term for candidate to update itself
        term: u64,
        /// True if candidate received vote
        vote_granted: bool,
    },
    
    /// Append entries (heartbeat and log replication)
    AppendEntries {
        /// Leader's term
        term: u64,
        /// Leader's ID for followers to redirect clients
        leader_id: NodeId,
        /// Index of log entry immediately preceding new ones
        prev_log_index: u64,
        /// Term of prev_log_index entry
        prev_log_term: u64,
        /// Log entries to store (empty for heartbeat)
        entries: Vec<LogEntry>,
        /// Leader's commit index
        leader_commit: u64,
    },
    
    /// Append entries response
    AppendEntriesResponse {
        /// Current term for leader to update itself
        term: u64,
        /// True if follower contained entry matching prev_log_index and prev_log_term
        success: bool,
        /// Follower's last log index for fast backup
        last_log_index: u64,
    },
    
    /// Install snapshot message
    InstallSnapshot {
        /// Leader's term
        term: u64,
        /// Leader's ID for followers to redirect clients
        leader_id: NodeId,
        /// Snapshot metadata
        snapshot: SnapshotData,
    },
    
    /// Install snapshot response
    InstallSnapshotResponse {
        /// Current term for leader to update itself
        term: u64,
        /// True if snapshot was successfully installed
        success: bool,
    },
}

/// Log entry data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Entry index
    pub index: u64,
    /// Entry term
    pub term: u64,
    /// Entry data
    pub data: Vec<u8>,
}

/// Snapshot data for state transfer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotData {
    /// Snapshot metadata
    pub metadata: SnapshotMetadata,
    /// Snapshot data chunks
    pub data: Vec<u8>,
}

/// Snapshot metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    /// Last included index
    pub last_included_index: u64,
    /// Last included term
    pub last_included_term: u64,
    /// Configuration at the time of snapshot
    pub configuration: Vec<NodeId>,
    /// Snapshot size in bytes
    pub size: u64,
    /// Snapshot checksum
    pub checksum: u64,
}

/// Network transport trait for Raft communication
#[async_trait::async_trait]
pub trait NetworkTransport: Send + Sync {
    /// Send a message to the specified node
    async fn send_message(
        &self,
        target: NodeId,
        message: RaftMessage,
    ) -> ConsensusResult<RaftMessage>;
    
    /// Send a message with timeout
    async fn send_message_with_timeout(
        &self,
        target: NodeId,
        message: RaftMessage,
        timeout: Duration,
    ) -> ConsensusResult<RaftMessage>;
    
    /// Broadcast a message to all nodes except self
    async fn broadcast_message(
        &self,
        message: RaftMessage,
        exclude: Option<NodeId>,
    ) -> ConsensusResult<Vec<(NodeId, ConsensusResult<RaftMessage>)>>;
    
    /// Get current cluster membership
    async fn get_cluster_members(&self) -> ConsensusResult<HashMap<NodeId, SocketAddr>>;
    
    /// Add a new node to the cluster
    async fn add_node(&self, node_id: NodeId, address: SocketAddr) -> ConsensusResult<()>;
    
    /// Remove a node from the cluster
    async fn remove_node(&self, node_id: NodeId) -> ConsensusResult<()>;
    
    /// Start the network transport
    async fn start(&self) -> ConsensusResult<()>;
    
    /// Stop the network transport
    async fn stop(&self) -> ConsensusResult<()>;
}

/// HTTP-based network transport implementation
pub struct HttpNetworkTransport {
    /// This node's ID
    node_id: NodeId,
    /// Cluster members (node_id -> address)
    members: Arc<RwLock<HashMap<NodeId, SocketAddr>>>,
    /// HTTP client for outgoing requests
    client: reqwest::Client,
    /// Default request timeout
    timeout: Duration,
    /// Maximum retry attempts
    max_retries: usize,
    /// Base delay for exponential backoff
    retry_base_delay: Duration,
}

impl HttpNetworkTransport {
    /// Create a new HTTP network transport
    pub fn new(
        node_id: NodeId,
        members: HashMap<NodeId, SocketAddr>,
        timeout: Duration,
        max_retries: usize,
        retry_base_delay: Duration,
    ) -> Self {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .expect("Failed to create HTTP client");
        
        Self {
            node_id,
            members: Arc::new(RwLock::new(members)),
            client,
            timeout,
            max_retries,
            retry_base_delay,
        }
    }
    
    /// Get the URL for a node
    async fn get_node_url(&self, node_id: NodeId) -> ConsensusResult<String> {
        let members = self.members.read().await;
        let addr = members.get(&node_id).ok_or_else(|| {
            ConsensusError::NodeNotFound(node_id)
        })?;
        Ok(format!("http://{}/raft", addr))
    }
    
    /// Send HTTP request with retries
    async fn send_with_retries(
        &self,
        url: String,
        message: RaftMessage,
        timeout: Duration,
    ) -> ConsensusResult<RaftMessage> {
        let mut attempts = 0;
        let mut delay = self.retry_base_delay;
        
        loop {
            match self.send_http_request(&url, &message, timeout).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    attempts += 1;
                    if attempts >= self.max_retries {
                        return Err(e);
                    }
                    
                    warn!(
                        "HTTP request failed (attempt {}/{}): {:?}. Retrying in {:?}",
                        attempts, self.max_retries, e, delay
                    );
                    
                    tokio::time::sleep(delay).await;
                    delay = std::cmp::min(delay * 2, Duration::from_secs(30));
                }
            }
        }
    }
    
    /// Send a single HTTP request
    async fn send_http_request(
        &self,
        url: &str,
        message: &RaftMessage,
        timeout: Duration,
    ) -> ConsensusResult<RaftMessage> {
        debug!("Sending HTTP request to {}: {:?}", url, message);
        
        let response = self
            .client
            .post(url)
            .timeout(timeout)
            .json(message)
            .send()
            .await
            .map_err(|e| {
                ConsensusError::Network(format!("HTTP request failed: {}", e))
            })?;
        
        if !response.status().is_success() {
            return Err(ConsensusError::Network(format!(
                "HTTP request failed with status: {}",
                response.status()
            )));
        }
        
        let raft_response: RaftMessage = response.json().await.map_err(|e| {
            ConsensusError::Network(format!("Failed to parse response: {}", e))
        })?;
        
        debug!("Received HTTP response: {:?}", raft_response);
        Ok(raft_response)
    }
}

#[async_trait::async_trait]
impl NetworkTransport for HttpNetworkTransport {
    async fn send_message(
        &self,
        target: NodeId,
        message: RaftMessage,
    ) -> ConsensusResult<RaftMessage> {
        self.send_message_with_timeout(target, message, self.timeout).await
    }
    
    async fn send_message_with_timeout(
        &self,
        target: NodeId,
        message: RaftMessage,
        timeout: Duration,
    ) -> ConsensusResult<RaftMessage> {
        if target == self.node_id {
            return Err(ConsensusError::Network(
                "Cannot send message to self".to_string(),
            ));
        }
        
        let url = self.get_node_url(target).await?;
        self.send_with_retries(url, message, timeout).await
    }
    
    async fn broadcast_message(
        &self,
        message: RaftMessage,
        exclude: Option<NodeId>,
    ) -> ConsensusResult<Vec<(NodeId, ConsensusResult<RaftMessage>)>> {
        let members = self.members.read().await;
        let mut futures = Vec::new();
        
        for &node_id in members.keys() {
            if node_id == self.node_id || Some(node_id) == exclude {
                continue;
            }
            
            let message_clone = message.clone();
            let transport = self.clone();
            
            let future = async move {
                let result = transport
                    .send_message(node_id, message_clone)
                    .await;
                (node_id, result)
            };
            
            futures.push(future);
        }
        
        let results = futures::future::join_all(futures).await;
        Ok(results)
    }
    
    async fn get_cluster_members(&self) -> ConsensusResult<HashMap<NodeId, SocketAddr>> {
        let members = self.members.read().await;
        Ok(members.clone())
    }
    
    async fn add_node(&self, node_id: NodeId, address: SocketAddr) -> ConsensusResult<()> {
        let mut members = self.members.write().await;
        members.insert(node_id, address);
        debug!("Added node {} at address {}", node_id, address);
        Ok(())
    }
    
    async fn remove_node(&self, node_id: NodeId) -> ConsensusResult<()> {
        let mut members = self.members.write().await;
        if members.remove(&node_id).is_some() {
            debug!("Removed node {}", node_id);
            Ok(())
        } else {
            Err(ConsensusError::NodeNotFound(node_id))
        }
    }
    
    async fn start(&self) -> ConsensusResult<()> {
        debug!("Starting HTTP network transport for node {}", self.node_id);
        // HTTP transport doesn't need explicit startup
        Ok(())
    }
    
    async fn stop(&self) -> ConsensusResult<()> {
        debug!("Stopping HTTP network transport for node {}", self.node_id);
        // HTTP transport doesn't need explicit shutdown
        Ok(())
    }
}

impl Clone for HttpNetworkTransport {
    fn clone(&self) -> Self {
        Self {
            node_id: self.node_id,
            members: Arc::clone(&self.members),
            client: self.client.clone(),
            timeout: self.timeout,
            max_retries: self.max_retries,
            retry_base_delay: self.retry_base_delay,
        }
    }
}

/// Mock network transport for testing
pub struct MockNetworkTransport {
    node_id: NodeId,
    members: HashMap<NodeId, SocketAddr>,
    // We can add message simulation here
}

impl MockNetworkTransport {
    /// Create a new mock network transport
    pub fn new(node_id: NodeId, members: HashMap<NodeId, SocketAddr>) -> Self {
        Self { node_id, members }
    }
}

#[async_trait::async_trait]
impl NetworkTransport for MockNetworkTransport {
    async fn send_message(
        &self,
        _target: NodeId,
        message: RaftMessage,
    ) -> ConsensusResult<RaftMessage> {
        // Simulate network delay
        tokio::time::sleep(Duration::from_millis(1)).await;
        
        // For testing, we can return different responses based on message type
        match message {
            RaftMessage::VoteRequest { term, .. } => {
                Ok(RaftMessage::VoteResponse {
                    term,
                    vote_granted: true,
                })
            }
            RaftMessage::AppendEntries { term, .. } => {
                Ok(RaftMessage::AppendEntriesResponse {
                    term,
                    success: true,
                    last_log_index: 0,
                })
            }
            RaftMessage::InstallSnapshot { term, .. } => {
                Ok(RaftMessage::InstallSnapshotResponse {
                    term,
                    success: true,
                })
            }
            _ => Err(ConsensusError::Network(
                "Unexpected message type in mock".to_string(),
            )),
        }
    }
    
    async fn send_message_with_timeout(
        &self,
        target: NodeId,
        message: RaftMessage,
        _timeout: Duration,
    ) -> ConsensusResult<RaftMessage> {
        self.send_message(target, message).await
    }
    
    async fn broadcast_message(
        &self,
        message: RaftMessage,
        exclude: Option<NodeId>,
    ) -> ConsensusResult<Vec<(NodeId, ConsensusResult<RaftMessage>)>> {
        let mut results = Vec::new();
        
        for &node_id in self.members.keys() {
            if node_id == self.node_id || Some(node_id) == exclude {
                continue;
            }
            
            let result = self.send_message(node_id, message.clone()).await;
            results.push((node_id, result));
        }
        
        Ok(results)
    }
    
    async fn get_cluster_members(&self) -> ConsensusResult<HashMap<NodeId, SocketAddr>> {
        Ok(self.members.clone())
    }
    
    async fn add_node(&self, _node_id: NodeId, _address: SocketAddr) -> ConsensusResult<()> {
        Ok(())
    }
    
    async fn remove_node(&self, _node_id: NodeId) -> ConsensusResult<()> {
        Ok(())
    }
    
    async fn start(&self) -> ConsensusResult<()> {
        Ok(())
    }
    
    async fn stop(&self) -> ConsensusResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_raft_message_serialization() {
        let msg = RaftMessage::VoteRequest {
            term: 42,
            candidate_id: 1,
            last_log_index: 100,
            last_log_term: 41,
        };
        
        let serialized = serde_json::to_string(&msg).unwrap();
        let deserialized: RaftMessage = serde_json::from_str(&serialized).unwrap();
        
        match deserialized {
            RaftMessage::VoteRequest { term, candidate_id, .. } => {
                assert_eq!(term, 42);
                assert_eq!(candidate_id, 1);
            }
            _ => panic!("Unexpected message type"),
        }
    }
    
    #[tokio::test]
    async fn test_mock_network_transport() {
        let mut members = HashMap::new();
        members.insert(1, SocketAddr::from_str("127.0.0.1:8001").unwrap());
        members.insert(2, SocketAddr::from_str("127.0.0.1:8002").unwrap());
        
        let transport = MockNetworkTransport::new(1, members);
        
        let vote_request = RaftMessage::VoteRequest {
            term: 1,
            candidate_id: 1,
            last_log_index: 0,
            last_log_term: 0,
        };
        
        let response = transport.send_message(2, vote_request).await.unwrap();
        
        match response {
            RaftMessage::VoteResponse { term, vote_granted } => {
                assert_eq!(term, 1);
                assert!(vote_granted);
            }
            _ => panic!("Unexpected response type"),
        }
    }
    
    #[tokio::test]
    async fn test_broadcast_message() {
        let mut members = HashMap::new();
        members.insert(1, SocketAddr::from_str("127.0.0.1:8001").unwrap());
        members.insert(2, SocketAddr::from_str("127.0.0.1:8002").unwrap());
        members.insert(3, SocketAddr::from_str("127.0.0.1:8003").unwrap());
        
        let transport = MockNetworkTransport::new(1, members);
        
        let heartbeat = RaftMessage::AppendEntries {
            term: 1,
            leader_id: 1,
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
        };
        
        let results = transport.broadcast_message(heartbeat, None).await.unwrap();
        
        assert_eq!(results.len(), 2); // Excludes self (node 1)
        for (node_id, result) in results {
            assert!(result.is_ok());
            assert!(node_id == 2 || node_id == 3);
        }
    }
}
