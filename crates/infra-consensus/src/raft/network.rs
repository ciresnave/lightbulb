//! Network communication layer for Raft consensus

use openraft::error::RPCError;
use openraft::{BasicNode, RaftNetwork, RaftNetworkFactory};
use std::future::Future;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::transport::Channel;

use super::types::*;
use openraft::network::RPCOption;

/// Network implementation for Raft communication
#[derive(Debug)]
pub struct RaftNetworkConnection {
    /// Network configuration
    config: super::config::NetworkConfig,
    /// Connected peers
    peers: Arc<RwLock<std::collections::HashMap<u64, Peer>>>,
}

/// Represents a connected peer in the cluster
#[derive(Debug)]
struct Peer {
    /// Remote node address
    #[allow(dead_code)] // Reserved for future network management
    addr: String,
    /// gRPC channel
    #[allow(dead_code)] // Reserved for future network management
    channel: Channel,
    /// Last seen timestamp
    #[allow(dead_code)] // Reserved for future network management
    last_seen: chrono::DateTime<chrono::Utc>,
}

impl RaftNetworkConnection {
    /// Create a new RaftNetworkConnection instance
    pub fn new(config: super::config::NetworkConfig) -> Result<Self, std::io::Error> {
        Ok(Self {
            config,
            peers: Arc::new(RwLock::new(std::collections::HashMap::new())),
        })
    }

    /// Get network configuration
    pub fn config(&self) -> &super::config::NetworkConfig {
        &self.config
    }

    /// Connect to a new peer
    pub async fn connect(&self, node_id: u64, addr: String) -> Result<(), std::io::Error> {
        let mut peers = self.peers.write().await;
        let channel = Channel::from_shared(addr.clone())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?
            .connect()
            .await
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        peers.insert(
            node_id,
            Peer {
                addr,
                channel,
                last_seen: chrono::Utc::now(),
            },
        );
        Ok(())
    }

    /// Remove a peer from the network
    pub async fn remove_peer(&self, node_id: u64) {
        let mut peers = self.peers.write().await;
        peers.remove(&node_id);
    }
}

impl RaftNetwork<TypeConfig> for RaftNetworkConnection {
    fn append_entries(
        &mut self,
        _rpc: openraft::raft::AppendEntriesRequest<TypeConfig>,
        _option: RPCOption,
    ) -> impl Future<
        Output = Result<
            openraft::raft::AppendEntriesResponse<u64>,
            RPCError<u64, BasicNode, openraft::error::RaftError<u64>>,
        >,
    > + Send {
        async move {
            // TODO: Implement actual gRPC call
            // For now, return a dummy response
            Ok(openraft::raft::AppendEntriesResponse::Success)
        }
    }

    fn vote(
        &mut self,
        rpc: openraft::raft::VoteRequest<u64>,
        _option: RPCOption,
    ) -> impl Future<
        Output = Result<
            openraft::raft::VoteResponse<u64>,
            RPCError<u64, BasicNode, openraft::error::RaftError<u64>>,
        >,
    > + Send {
        async move {
            // TODO: Implement actual gRPC call
            // For now, return a dummy response
            Ok(openraft::raft::VoteResponse {
                vote: rpc.vote,
                vote_granted: true,
                last_log_id: None,
            })
        }
    }

    fn full_snapshot(
        &mut self,
        vote: openraft::Vote<u64>,
        _snapshot: openraft::storage::Snapshot<TypeConfig>,
        _cancel: impl Future<Output = openraft::error::ReplicationClosed>
            + openraft::OptionalSend
            + 'static,
        _option: RPCOption,
    ) -> impl Future<
        Output = Result<
            openraft::raft::SnapshotResponse<u64>,
            openraft::error::StreamingError<TypeConfig, openraft::error::Fatal<u64>>,
        >,
    > + Send {
        async move {
            // TODO: Implement actual snapshot transfer
            Ok(openraft::raft::SnapshotResponse { vote })
        }
    }

    fn install_snapshot(
        &mut self,
        _rpc: openraft::raft::InstallSnapshotRequest<TypeConfig>,
        _option: RPCOption,
    ) -> impl Future<
        Output = Result<
            openraft::raft::InstallSnapshotResponse<u64>,
            RPCError<
                u64,
                BasicNode,
                openraft::error::RaftError<u64, openraft::error::InstallSnapshotError>,
            >,
        >,
    > + Send {
        async move {
            // TODO: Implement actual install snapshot
            Ok(openraft::raft::InstallSnapshotResponse {
                vote: openraft::Vote::default(),
            })
        }
    }
}

/// Network factory for creating connections
pub struct RaftNetworkFactoryImpl {
    config: super::config::NetworkConfig,
}

impl RaftNetworkFactoryImpl {
    /// Creates a new RaftNetworkFactory with the given configuration
    pub fn new(config: super::config::NetworkConfig) -> Self {
        Self { config }
    }
}

impl RaftNetworkFactory<TypeConfig> for RaftNetworkFactoryImpl {
    type Network = RaftNetworkConnection;

    fn new_client(
        &mut self,
        _target: u64,
        _node: &openraft::BasicNode,
    ) -> impl Future<Output = Self::Network> + Send {
        let config = self.config.clone();
        async move { RaftNetworkConnection::new(config).unwrap() }
    }
}
