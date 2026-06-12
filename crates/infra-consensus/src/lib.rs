//! # Infrastructure Consensus
//!
//! Distributed consensus algorithms for DynAniML cluster coordination.
//!
//! This crate provides:
//! - Raft consensus algorithm implementation
//! - PBFT (Practical Byzantine Fault Tolerance) support
//! - Leader election and failover mechanisms
//! - Log replication and consistency guarantees
//! - Cluster membership management

#![deny(missing_docs)]
#![warn(clippy::all)]

pub mod raft;

pub use raft::{
    config::RaftConfig,
    create_node,
    error::RaftError,
    node::RaftNode,
    types::{Command, RaftResponse, TypeConfig},
};

/// Re-export OpenRaft types
pub use openraft::{
    self,
    storage::{LogState, RaftLogStorage, RaftStateMachine},
    BasicNode, LogId, Raft, RaftLogReader, RaftTypeConfig, Vote,
};

/// Current version of the Infrastructure Consensus library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Placeholder function to ensure the crate compiles
pub fn placeholder() -> &'static str {
    "Infrastructure Consensus - Placeholder Implementation"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
