//! Error types for Raft consensus operations

use openraft::BasicNode;
use thiserror::Error;

/// Errors that can occur during Raft operations
#[derive(Error, Debug)]
pub enum RaftError {
    #[error("Storage error: {0}")]
    /// Storage layer error
    Storage(#[from] StorageError),

    #[error("Network error: {0}")]
    /// Network communication error
    Network(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    /// Serialization/deserialization error
    Serialization(#[from] serde_json::Error),

    #[error("Node {node_id} not found")]
    /// Node not found in cluster
    NodeNotFound {
        /// The ID of the node that was not found
        node_id: u64,
    },

    #[error("Not the leader. Current leader: {leader_id:?}")]
    /// Current node is not the leader
    NotLeader {
        /// The ID of the current leader, if known
        leader_id: Option<u64>,
    },

    #[error("Timeout waiting for consensus")]
    /// Consensus operation timed out
    ConsensusTimeout,

    #[error("Configuration error: {0}")]
    /// Configuration error
    Config(String),

    #[error("State error: {0}")]
    /// State error
    State(String),

    #[error("Internal error: {0}")]
    /// Internal error
    Internal(String),

    #[error("OpenRaft error: {0}")]
    /// OpenRaft error
    OpenRaft(String),
}
/// Error that can occur during storage operations
#[derive(Error, Debug)]
pub enum StorageError {
    #[error("IO error: {0}")]
    /// I/O operation error
    Io(#[from] std::io::Error),

    #[error("Log entry {id} not found")]
    /// Log entry not found
    LogNotFound {
        /// The ID of the log entry that was not found
        id: u64,
    },

    #[error("Invalid log entry format")]
    /// Invalid log entry format
    InvalidLogFormat,

    #[error("Snapshot error: {0}")]
    /// Snapshot operation error
    Snapshot(String),

    #[error("Storage corruption detected: {0}")]
    /// Storage corruption detected
    Corruption(String),
}

impl From<openraft::error::RaftError<u64, openraft::error::ClientWriteError<u64, BasicNode>>>
    for RaftError
{
    fn from(
        err: openraft::error::RaftError<u64, openraft::error::ClientWriteError<u64, BasicNode>>,
    ) -> Self {
        RaftError::Network(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("{:?}", err),
        ))
    }
}
