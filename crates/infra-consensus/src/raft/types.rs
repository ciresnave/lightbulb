//! Core type definitions for the Raft consensus implementation

#![allow(missing_docs)]

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use openraft::{
    storage::{RaftSnapshotBuilder, RaftStateMachine},
    BasicNode, Entry as RaftEntry, EntryPayload, LogId, Snapshot, SnapshotMeta, StorageError,
    StoredMembership, TokioRuntime,
};
use serde::{Deserialize, Serialize};
use std::fmt::Display;
use std::io::Cursor;

// Use the macro to declare our types properly
openraft::declare_raft_types!(
    pub TypeConfig:
        D            = Command,
        R            = RaftResponse,
        NodeId       = u64,
        Node         = BasicNode,
        Entry        = RaftEntry<TypeConfig>,
        SnapshotData = Cursor<Vec<u8>>,
        AsyncRuntime = TokioRuntime,
);

/// Command data to be replicated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Command {
    /// The unique identifier of the command
    pub id: String,
    /// The actual data/command to be replicated
    pub data: Vec<u8>,
    /// Command type
    pub command_type: CommandType,
    /// Optional metadata
    pub metadata: CommandMetadata,
}

/// Metadata for commands
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandMetadata {
    /// Timestamp when the command was created
    pub timestamp: DateTime<Utc>,
    /// Source node that created the command
    pub source_node: u64,
}

/// Types of commands that can be replicated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommandType {
    /// State machine mutation
    Mutation,
    /// Configuration change
    Config,
    /// Custom command type
    Custom(String),
}

/// Response from Raft operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftResponse {
    /// Success/failure status
    pub success: bool,
    /// Optional data returned
    pub data: Option<Vec<u8>>,
    /// Error message if operation failed
    pub error: Option<String>,
}

impl Display for Command {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Command({}, {} bytes)", self.id, self.data.len())
    }
}

impl Display for RaftResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (&self.success, &self.error) {
            (true, _) => write!(f, "Success"),
            (false, Some(err)) => write!(f, "Error: {}", err),
            (false, None) => write!(f, "Error: Unknown"),
        }
    }
}

/// State machine managed by Raft
#[derive(Debug, Default)]
pub struct StateMachine {
    /// The state machine data as key-value pairs
    pub data: std::collections::BTreeMap<String, Vec<u8>>,
    /// Last applied log entry ID
    pub last_applied_log: Option<LogId<u64>>,
    /// Last known cluster membership
    pub last_membership: StoredMembership<u64, BasicNode>,
}

impl RaftStateMachine<TypeConfig> for StateMachine {
    type SnapshotBuilder = StateMachineSnapshotBuilder;

    fn applied_state(
        &mut self,
    ) -> impl std::future::Future<
        Output = Result<(Option<LogId<u64>>, StoredMembership<u64, BasicNode>), StorageError<u64>>,
    > + Send {
        async move { Ok((self.last_applied_log.clone(), self.last_membership.clone())) }
    }

    fn apply<I>(
        &mut self,
        entries: I,
    ) -> impl std::future::Future<Output = Result<Vec<RaftResponse>, StorageError<u64>>> + Send
    where
        I: IntoIterator<Item = RaftEntry<TypeConfig>> + Send,
        I::IntoIter: Send,
    {
        async move {
            let mut responses = Vec::new();

            for entry in entries {
                // Update last applied log
                self.last_applied_log = Some(entry.log_id.clone());

                // Handle membership changes - check if it's a membership entry
                match &entry.payload {
                    EntryPayload::Membership(membership) => {
                        self.last_membership =
                            StoredMembership::new(Some(entry.log_id.clone()), membership.clone());
                        responses.push(RaftResponse {
                            success: true,
                            data: None,
                            error: None,
                        });
                    }
                    EntryPayload::Normal(cmd) => match &cmd.command_type {
                        CommandType::Mutation => {
                            self.data.insert(cmd.id.clone(), cmd.data.clone());
                            responses.push(RaftResponse {
                                success: true,
                                data: None,
                                error: None,
                            });
                        }
                        _ => {
                            responses.push(RaftResponse {
                                success: false,
                                data: None,
                                error: Some("Unsupported command type".into()),
                            });
                        }
                    },
                    EntryPayload::Blank => {
                        responses.push(RaftResponse {
                            success: true,
                            data: None,
                            error: None,
                        });
                    }
                }
            }
            Ok(responses)
        }
    }
    fn begin_receiving_snapshot(
        &mut self,
    ) -> impl std::future::Future<Output = Result<Box<Cursor<Vec<u8>>>, StorageError<u64>>> + Send
    {
        async move { Ok(Box::new(Cursor::new(Vec::new()))) }
    }

    fn install_snapshot(
        &mut self,
        meta: &SnapshotMeta<u64, BasicNode>,
        snapshot: Box<Cursor<Vec<u8>>>,
    ) -> impl std::future::Future<Output = Result<(), StorageError<u64>>> + Send {
        let meta = meta.clone();
        let snapshot_data = snapshot.get_ref().clone();
        async move {
            self.data = bincode::decode_from_slice(&snapshot_data, bincode::config::standard())
                .expect("Failed to deserialize snapshot data")
                .0;

            self.last_applied_log = meta.last_log_id.clone();
            self.last_membership = meta.last_membership.clone();

            Ok(())
        }
    }

    fn get_current_snapshot(
        &mut self,
    ) -> impl std::future::Future<Output = Result<Option<Snapshot<TypeConfig>>, StorageError<u64>>> + Send
    {
        async move {
            let data = bincode::encode_to_vec(&self.data, bincode::config::standard())
                .expect("Failed to serialize data");

            let snapshot = Snapshot {
                meta: SnapshotMeta {
                    last_log_id: self.last_applied_log.clone(),
                    last_membership: self.last_membership.clone(),
                    snapshot_id: format!("snapshot-{}", chrono::Utc::now().timestamp()),
                },
                snapshot: Box::new(Cursor::new(data)),
            };

            Ok(Some(snapshot))
        }
    }

    fn get_snapshot_builder(
        &mut self,
    ) -> impl std::future::Future<Output = Self::SnapshotBuilder> + Send {
        async move { StateMachineSnapshotBuilder::new() }
    }
}

/// Snapshot builder for state machine
pub struct StateMachineSnapshotBuilder {
    data: Vec<u8>,
}

impl StateMachineSnapshotBuilder {
    /// Creates a new StateMachineSnapshotBuilder
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
}

#[async_trait]
impl RaftSnapshotBuilder<TypeConfig> for StateMachineSnapshotBuilder {
    fn build_snapshot(
        &mut self,
    ) -> impl std::future::Future<Output = Result<Snapshot<TypeConfig>, StorageError<u64>>> + Send
    {
        async move {
            let snapshot = Snapshot {
                meta: SnapshotMeta {
                    last_log_id: None,
                    last_membership: StoredMembership::new(None, Default::default()),
                    snapshot_id: format!("snapshot-{}", chrono::Utc::now().timestamp()),
                },
                snapshot: Box::new(Cursor::new(self.data.clone())),
            };
            Ok(snapshot)
        }
    }
}
