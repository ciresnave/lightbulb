//! Raft node implementation
//!
//! This module provides the main Raft node implementation for consensus.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::{mpsc, oneshot, RwLock};
use tokio::time::interval;
use tracing::{debug, error, info, warn};

use crate::config::{RaftConfig, ClusterConfig};
use crate::network::{NetworkTransport, RaftMessage, LogEntry};
use crate::storage::{DynStorage, StorageBackend};
use crate::{NodeId, ConsensusResult, ConsensusError, LeadershipState, ClusterMembership};

/// Current state of a Raft node
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeState {
    /// Node is a follower
    Follower,
    /// Node is a candidate (during election)
    Candidate,
    /// Node is the leader
    Leader,
    /// Node is shut down
    Shutdown,
}

/// Raft node implementation
pub struct RaftNode {
    /// Node configuration
    config: RaftConfig,
    /// Cluster configuration
    cluster_config: ClusterConfig,
    /// Current node state
    state: Arc<RwLock<NodeState>>,
    /// Persistent storage
    storage: Arc<DynStorage>,
    /// Network transport
    network: Arc<dyn NetworkTransport>,
    /// Current term (volatile)
    current_term: Arc<RwLock<u64>>,
    /// Current leader ID (volatile)
    current_leader: Arc<RwLock<Option<NodeId>>>,
    /// Last time we heard from leader (volatile)
    last_heartbeat: Arc<RwLock<Instant>>,
    /// Command channel for external operations
    command_tx: mpsc::UnboundedSender<RaftCommand>,
    /// Shutdown signal
    shutdown_tx: Option<mpsc::UnboundedSender<()>>,
}

/// Commands sent to the Raft node
#[derive(Debug)]
pub enum RaftCommand {
    /// Propose a new log entry
    Propose {
        /// The data to be replicated across the cluster
        data: Vec<u8>,
        /// Channel to send back the result of the proposal
        response_tx: oneshot::Sender<ConsensusResult<u64>>,
    },
    /// Get current leader
    GetLeader {
        /// Channel to send back the current leader ID (None if no leader)
        response_tx: oneshot::Sender<Option<NodeId>>,
    },
    /// Get cluster membership
    GetMembership {
        /// Channel to send back the current cluster membership
        response_tx: oneshot::Sender<ClusterMembership>,
    },
    /// Add a new node to the cluster
    AddNode {
        /// The unique identifier for the new node
        node_id: NodeId,
        /// The network address of the new node
        address: SocketAddr,
        /// Channel to send back the result of the add operation
        response_tx: oneshot::Sender<ConsensusResult<()>>,
    },
    /// Remove a node from the cluster
    RemoveNode {
        /// The unique identifier of the node to remove
        node_id: NodeId,
        /// Channel to send back the result of the remove operation
        response_tx: oneshot::Sender<ConsensusResult<()>>,
    },
    /// Shutdown the node
    Shutdown,
}

/// Client handle for interacting with the Raft node
#[derive(Clone)]
pub struct RaftNodeHandle {
    command_tx: mpsc::UnboundedSender<RaftCommand>,
}

impl RaftNodeHandle {
    /// Propose a new log entry
    pub async fn propose(&self, data: Vec<u8>) -> ConsensusResult<u64> {
        let (response_tx, response_rx) = oneshot::channel();
        
        self.command_tx
            .send(RaftCommand::Propose { data, response_tx })
            .map_err(|_| ConsensusError::InvalidState("Node is shut down".to_string()))?;
        
        response_rx
            .await
            .map_err(|_| ConsensusError::InvalidState("Command channel closed".to_string()))?
    }
    
    /// Get the current leader
    pub async fn get_leader(&self) -> Option<NodeId> {
        let (response_tx, response_rx) = oneshot::channel();
        
        if self.command_tx
            .send(RaftCommand::GetLeader { response_tx })
            .is_err()
        {
            return None;
        }
        
        response_rx.await.unwrap_or(None)
    }
    
    /// Get cluster membership
    pub async fn get_membership(&self) -> Option<ClusterMembership> {
        let (response_tx, response_rx) = oneshot::channel();
        
        if self.command_tx
            .send(RaftCommand::GetMembership { response_tx })
            .is_err()
        {
            return None;
        }
        
        response_rx.await.ok()
    }
    
    /// Add a new node to the cluster
    pub async fn add_node(&self, node_id: NodeId, address: SocketAddr) -> ConsensusResult<()> {
        let (response_tx, response_rx) = oneshot::channel();
        
        self.command_tx
            .send(RaftCommand::AddNode {
                node_id,
                address,
                response_tx,
            })
            .map_err(|_| ConsensusError::InvalidState("Node is shut down".to_string()))?;
        
        response_rx
            .await
            .map_err(|_| ConsensusError::InvalidState("Command channel closed".to_string()))?
    }
    
    /// Remove a node from the cluster
    pub async fn remove_node(&self, node_id: NodeId) -> ConsensusResult<()> {
        let (response_tx, response_rx) = oneshot::channel();
        
        self.command_tx
            .send(RaftCommand::RemoveNode { node_id, response_tx })
            .map_err(|_| ConsensusError::InvalidState("Node is shut down".to_string()))?;
        
        response_rx
            .await
            .map_err(|_| ConsensusError::InvalidState("Command channel closed".to_string()))?
    }
    
    /// Shutdown the node
    pub async fn shutdown(&self) {
        let _ = self.command_tx.send(RaftCommand::Shutdown);
    }
}

impl RaftNode {
    /// Create a new Raft node
    pub async fn new(
        config: RaftConfig,
        cluster_config: ClusterConfig,
        storage: DynStorage,
        network: Arc<dyn NetworkTransport>,
    ) -> ConsensusResult<(Self, RaftNodeHandle)> {        // Validate configurations
        config.validate()?;
        cluster_config.validate()?;
        
        let (command_tx, _command_rx) = mpsc::unbounded_channel();
        
        let node = Self {
            config,
            cluster_config,
            state: Arc::new(RwLock::new(NodeState::Follower)),
            storage: Arc::new(storage),
            network,
            current_term: Arc::new(RwLock::new(0)),
            current_leader: Arc::new(RwLock::new(None)),
            last_heartbeat: Arc::new(RwLock::new(Instant::now())),
            command_tx: command_tx.clone(),
            shutdown_tx: None,
        };
        
        let handle = RaftNodeHandle { command_tx };
        
        Ok((node, handle))
    }
    
    /// Start the Raft node
    pub async fn start(mut self) -> ConsensusResult<()> {
        info!("Starting Raft node {}", self.config.node_id);
        
        // Load persistent state
        let term = self.storage.get_current_term().await?;
        *self.current_term.write().await = term;
        
        // Start network transport
        self.network.start().await?;
        
        // Set initial state
        *self.state.write().await = NodeState::Follower;
        *self.last_heartbeat.write().await = Instant::now();
        
        // Create shutdown channel
        let (shutdown_tx, mut shutdown_rx) = mpsc::unbounded_channel();
        self.shutdown_tx = Some(shutdown_tx);
        
        // Create command receiver for the main loop
        let (new_command_tx, command_rx) = mpsc::unbounded_channel();
        
        // Replace the command sender in node and give new sender to tasks
        let node_clone = RaftNodeForTasks {
            config: self.config.clone(),
            cluster_config: self.cluster_config.clone(),
            state: Arc::clone(&self.state),
            storage: Arc::clone(&self.storage),
            network: Arc::clone(&self.network),
            current_term: Arc::clone(&self.current_term),
            current_leader: Arc::clone(&self.current_leader),
            last_heartbeat: Arc::clone(&self.last_heartbeat),
            command_tx: new_command_tx.clone(),
        };
        
        // Update our command sender
        self.command_tx = new_command_tx;
        
        // Spawn main event loop
        let main_task = tokio::spawn(async move {
            node_clone.run_main_loop(command_rx).await
        });
        
        // Wait for shutdown signal
        let _ = shutdown_rx.recv().await;
        
        // Shutdown
        *self.state.write().await = NodeState::Shutdown;
        self.network.stop().await?;
        main_task.abort();
        
        info!("Raft node {} shut down", self.config.node_id);
        Ok(())
    }
      /// Clone node data for async tasks
    fn clone_for_tasks(&self) -> RaftNodeForTasks {
        RaftNodeForTasks {
            config: self.config.clone(),
            cluster_config: self.cluster_config.clone(),
            state: Arc::clone(&self.state),
            storage: Arc::clone(&self.storage),
            network: Arc::clone(&self.network),
            current_term: Arc::clone(&self.current_term),
            current_leader: Arc::clone(&self.current_leader),
            last_heartbeat: Arc::clone(&self.last_heartbeat),
            command_tx: self.command_tx.clone(),
        }
    }
    
    /// Get current leadership state
    pub async fn get_leadership_state(&self) -> LeadershipState {
        match *self.state.read().await {
            NodeState::Leader => LeadershipState::Leader,
            NodeState::Candidate => LeadershipState::Candidate,
            NodeState::Follower => LeadershipState::Follower,
            NodeState::Shutdown => LeadershipState::Follower,
        }
    }
    
    /// Get current cluster membership
    pub async fn get_cluster_membership(&self) -> ClusterMembership {
        let members = self.network.get_cluster_members().await
            .unwrap_or_default()
            .into_iter()
            .map(|(id, addr)| (id, addr.to_string()))
            .collect();
        
        ClusterMembership {
            nodes: members,
            config_version: 1, // TODO: Implement proper config versioning
        }
    }
}

/// Raft node data for async tasks
#[derive(Clone)]
struct RaftNodeForTasks {
    config: RaftConfig,
    cluster_config: ClusterConfig,
    state: Arc<RwLock<NodeState>>,
    storage: Arc<DynStorage>,
    network: Arc<dyn NetworkTransport>,
    current_term: Arc<RwLock<u64>>,
    current_leader: Arc<RwLock<Option<NodeId>>>,
    last_heartbeat: Arc<RwLock<Instant>>,
    command_tx: mpsc::UnboundedSender<RaftCommand>,
}

impl RaftNodeForTasks {
    /// Get current cluster membership
    async fn get_cluster_membership(&self) -> ClusterMembership {
        let members = self.network.get_cluster_members().await
            .unwrap_or_default()
            .into_iter()
            .map(|(id, addr)| (id, addr.to_string()))
            .collect();
            
        ClusterMembership {
            nodes: members,
            config_version: 1,
        }
    }
    
    /// Handle a base command that works in any state
    async fn handle_base_command(&self, command: RaftCommand) -> bool {
        match command {
            RaftCommand::GetMembership { response_tx } => {
                let membership = self.get_cluster_membership().await;
                let _ = response_tx.send(membership);
                true
            }
            RaftCommand::Shutdown => {
                *self.state.write().await = NodeState::Shutdown;
                true
            }
            _ => false
        }
    }
    
    /// Main event loop
    async fn run_main_loop(self, mut command_rx: mpsc::UnboundedReceiver<RaftCommand>) {
        let mut election_timer = interval(self.config.election_timeout.0);
        let mut heartbeat_timer = interval(self.config.heartbeat_interval);
        
        loop {
            let current_state = *self.state.read().await;
            
            match current_state {
                NodeState::Shutdown => break,
                NodeState::Follower => {
                    tokio::select! {
                        _ = election_timer.tick() => {
                            if self.should_start_election().await {
                                self.start_election().await;
                            }
                        }
                        Some(command) = command_rx.recv() => {
                            if !self.handle_base_command(command.clone()).await {
                                match command {
                                    RaftCommand::GetLeader { response_tx } => {
                                        let _ = response_tx.send(*self.current_leader.read().await);
                                    }
                                    _ => {
                                        // Forward other commands to current leader if known
                                        if let Some(_leader_id) = *self.current_leader.read().await {
                                            // TODO: Implement command forwarding
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                NodeState::Candidate => {
                    tokio::select! {
                        _ = election_timer.tick() => {
                            self.start_election().await;
                        }
                        Some(command) = command_rx.recv() => {
                            if !self.handle_base_command(command.clone()).await {
                                match command {
                                    RaftCommand::GetLeader { response_tx } => {
                                        let _ = response_tx.send(None);
                                    }
                                    _ => {} // Ignore other commands during election
                                }
                            }
                        }
                    }
                }
                NodeState::Leader => {
                    tokio::select! {
                        _ = heartbeat_timer.tick() => {
                            self.send_heartbeats().await;
                        }
                        Some(command) = command_rx.recv() => {
                            if !self.handle_base_command(command.clone()).await {
                                match command {
                                    RaftCommand::GetLeader { response_tx } => {
                                        let _ = response_tx.send(Some(self.config.node_id));
                                    }
                                    RaftCommand::Propose { data, response_tx } => {
                                        let result = self.handle_propose(data).await;
                                        let _ = response_tx.send(result);
                                    }
                                    RaftCommand::AddNode { node_id, address, response_tx } => {
                                        let result = self.network.add_node(node_id, address).await;
                                        let _ = response_tx.send(result);
                                    }
                                    RaftCommand::RemoveNode { node_id, response_tx } => {
                                        let result = self.network.remove_node(node_id).await;
                                        let _ = response_tx.send(result);
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    /// Check if we should start an election
    async fn should_start_election(&self) -> bool {
        let last_heartbeat = *self.last_heartbeat.read().await;
        let election_timeout = self.config.election_timeout.1;
        
        last_heartbeat.elapsed() > election_timeout
    }
    
    /// Start a new election
    async fn start_election(&self) {
        info!("Node {} starting election", self.config.node_id);
        
        // Transition to candidate state
        *self.state.write().await = NodeState::Candidate;
        
        // Increment current term
        let new_term = {
            let mut term = self.current_term.write().await;
            *term += 1;
            *term
        };
        
        // Vote for ourselves
        if let Err(e) = self.storage.set_current_term(new_term).await {
            error!("Failed to set current term: {:?}", e);
            return;
        }
        
        if let Err(e) = self.storage.set_voted_for(Some(self.config.node_id)).await {
            error!("Failed to set voted for: {:?}", e);
            return;
        }
        
        // Get last log info
        let last_log_index = self.storage.get_last_log_index().await.unwrap_or(0);
        let last_log_term = self.storage.get_last_log_term().await.unwrap_or(0);
        
        // Send vote requests
        let vote_request = RaftMessage::VoteRequest {
            term: new_term,
            candidate_id: self.config.node_id,
            last_log_index,
            last_log_term,
        };
        
        let results = self.network
            .broadcast_message(vote_request, Some(self.config.node_id))
            .await;
        
        if let Ok(responses) = results {
            let mut votes = 1; // Vote for ourselves
            let total_nodes = responses.len() + 1; // +1 for ourselves
            
            for (node_id, result) in responses {
                if let Ok(RaftMessage::VoteResponse { vote_granted, .. }) = result {
                    if vote_granted {
                        votes += 1;
                        debug!("Received vote from node {}", node_id);
                    }
                }
            }
            
            let majority = (total_nodes / 2) + 1;
            if votes >= majority {
                info!("Node {} won election with {} votes", self.config.node_id, votes);
                self.become_leader().await;
            } else {
                info!("Node {} lost election with {} votes", self.config.node_id, votes);
                *self.state.write().await = NodeState::Follower;
            }
        }
    }
    
    /// Become the leader
    async fn become_leader(&self) {
        info!("Node {} becoming leader", self.config.node_id);
        
        *self.state.write().await = NodeState::Leader;
        *self.current_leader.write().await = Some(self.config.node_id);
        
        // Send initial heartbeats
        self.send_heartbeats().await;
    }
    
    /// Send heartbeats to all followers
    async fn send_heartbeats(&self) {
        let current_term = *self.current_term.read().await;
        let last_log_index = self.storage.get_last_log_index().await.unwrap_or(0);
        let last_log_term = self.storage.get_last_log_term().await.unwrap_or(0);
        let commit_index = self.storage.get_commit_index().await.unwrap_or(0);
        
        let heartbeat = RaftMessage::AppendEntries {
            term: current_term,
            leader_id: self.config.node_id,
            prev_log_index: last_log_index,
            prev_log_term: last_log_term,
            entries: vec![], // Empty for heartbeat
            leader_commit: commit_index,
        };
        
        let results = self.network
            .broadcast_message(heartbeat, Some(self.config.node_id))
            .await;
        
        if let Ok(responses) = results {
            for (node_id, result) in responses {
                match result {
                    Ok(RaftMessage::AppendEntriesResponse { success, .. }) => {
                        if success {
                            debug!("Heartbeat acknowledged by node {}", node_id);
                        } else {
                            debug!("Heartbeat rejected by node {}", node_id);
                        }
                    }
                    Err(e) => {
                        warn!("Failed to send heartbeat to node {}: {:?}", node_id, e);
                    }
                    _ => {
                        warn!("Unexpected response from node {}", node_id);
                    }
                }
            }
        }
    }
    
    /// Handle incoming commands
    async fn handle_command(&self, command: RaftCommand) {
        match command {
            RaftCommand::Propose { data, response_tx } => {
                let result = self.handle_propose(data).await;
                let _ = response_tx.send(result);
            }
            RaftCommand::GetLeader { response_tx } => {
                let leader = *self.current_leader.read().await;
                let _ = response_tx.send(leader);
            }
            RaftCommand::GetMembership { response_tx } => {
                let membership = self.get_cluster_membership().await;
                let _ = response_tx.send(membership);
            }
            RaftCommand::AddNode { node_id, address, response_tx } => {
                let result = self.network.add_node(node_id, address).await;
                let _ = response_tx.send(result);
            }
            RaftCommand::RemoveNode { node_id, response_tx } => {
                let result = self.network.remove_node(node_id).await;
                let _ = response_tx.send(result);
            }
            RaftCommand::Shutdown => {
                *self.state.write().await = NodeState::Shutdown;
            }
        }
    }
    
    /// Handle propose command
    async fn handle_propose(&self, data: Vec<u8>) -> ConsensusResult<u64> {
        // Only leaders can handle proposals
        if *self.state.read().await != NodeState::Leader {
            return Err(ConsensusError::InvalidState(
                "Only leaders can handle proposals".to_string(),
            ));
        }
        
        let current_term = *self.current_term.read().await;
        let next_index = self.storage.get_last_log_index().await? + 1;
        
        let entry = LogEntry {
            index: next_index,
            term: current_term,
            data,
        };
        
        // Append to local log
        self.storage.append_log_entries(vec![entry.clone()]).await?;
        
        // Replicate to followers (simplified - in a real implementation, this would be more complex)
        let append_entries = RaftMessage::AppendEntries {
            term: current_term,
            leader_id: self.config.node_id,
            prev_log_index: next_index - 1,
            prev_log_term: if next_index > 1 {
                self.storage.get_log_entry(next_index - 1).await?
                    .map(|e| e.term)
                    .unwrap_or(0)
            } else {
                0
            },
            entries: vec![entry],
            leader_commit: self.storage.get_commit_index().await?,
        };
        
        let results = self.network
            .broadcast_message(append_entries, Some(self.config.node_id))
            .await?;
        
        let mut success_count = 1; // Count ourselves
        let total_nodes = results.len() + 1;
        
        for (_, result) in results {
            if let Ok(RaftMessage::AppendEntriesResponse { success: true, .. }) = result {
                success_count += 1;
            }
        }
        
        let majority = (total_nodes / 2) + 1;
        if success_count >= majority {
            // Commit the entry
            self.storage.set_commit_index(next_index).await?;
            Ok(next_index)
        } else {
            Err(ConsensusError::InvalidState(
                "Failed to replicate to majority".to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::MockNetworkTransport;
    use std::{collections::HashMap, str::FromStr};

    #[tokio::test]
    async fn test_raft_node_creation() {
        let config = RaftConfig::new(1);
        let cluster_config = ClusterConfig::new("test", "/tmp/test");
        let storage = DynStorage::new_memory_storage();
        
        let mut members = HashMap::new();
        members.insert(1, SocketAddr::from_str("127.0.0.1:8001").unwrap());
        let network = Arc::new(MockNetworkTransport::new(1, members));
        
        let result = RaftNode::new(config, cluster_config, storage, network).await;
        assert!(result.is_ok());
        
        let (node, _handle) = result.unwrap();
        assert_eq!(node.config.node_id, 1);
    }
    
    #[tokio::test]
    async fn test_raft_node_handle() {
        let config = RaftConfig::new(1);
        let cluster_config = ClusterConfig::new("test", "/tmp/test");
        let storage = DynStorage::new_memory_storage();
        
        let mut members = HashMap::new();
        members.insert(1, SocketAddr::from_str("127.0.0.1:8001").unwrap());
        let network = Arc::new(MockNetworkTransport::new(1, members));
        
        let (node, handle) = RaftNode::new(config, cluster_config, storage, network)
            .await
            .unwrap();
        
        // Start node in background
        let node_handle = tokio::spawn(async move {
            node.start().await.unwrap();
        });
        
        // Give the node time to start
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        
        // Test getting leader (should be None initially)
        let leader = handle.get_leader().await;
        assert_eq!(leader, None);
        
        // Test getting membership
        let membership = handle.get_membership().await;
        assert!(membership.is_some());
        
        // Shutdown node
        handle.shutdown().await;
        
        // Wait for node to stop
        node_handle.await.unwrap();
    }
}
