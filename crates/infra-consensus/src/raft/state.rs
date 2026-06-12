//! State machine implementation for Raft consensus

use openraft::storage::{RaftSnapshotBuilder, RaftStateMachine};
use std::collections::BTreeMap;
use std::io::Cursor;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::error::RaftError;
use super::types::{RaftResponse, TypeConfig};

/// State manager
#[derive(Debug, Clone)]
pub struct StateManager {
    /// Current state
    state: Arc<RwLock<BTreeMap<Vec<u8>, Vec<u8>>>>,
    /// Configuration
    config: super::config::StateConfig,
}

impl StateManager {
    /// Create a new state manager
    pub fn new(config: super::config::StateConfig) -> Result<Self, RaftError> {
        Ok(Self {
            state: Arc::new(RwLock::new(BTreeMap::new())),
            config,
        })
    }

    /// Get state configuration
    pub fn config(&self) -> &super::config::StateConfig {
        &self.config
    }

    /// Apply an entry to the state machine
    pub async fn apply_entry(&self, entry: &openraft::Entry<TypeConfig>) -> Result<(), RaftError> {
        let mut state = self.state.write().await;
        // Apply commands based on the entry payload
        match &entry.payload {
            openraft::EntryPayload::Blank => {
                // No-op for blank entries
            }
            openraft::EntryPayload::Normal(command) => {
                // Apply the command to our state
                state.insert(command.id.as_bytes().to_vec(), command.data.clone());
            }
            openraft::EntryPayload::Membership(_) => {
                // Handle membership changes if needed
            }
        }
        Ok(())
    }

    /// Create a snapshot of the current state
    pub async fn create_snapshot(&self) -> Result<openraft::Snapshot<TypeConfig>, RaftError> {
        let state = self.state.read().await;
        let serialized = serde_json::to_vec(&*state).map_err(|e| RaftError::Serialization(e))?;

        Ok(openraft::Snapshot {
            meta: openraft::SnapshotMeta {
                last_log_id: None, // TODO: Get from storage
                last_membership: Default::default(),
                snapshot_id: format!("snapshot-{}", chrono::Utc::now().timestamp()),
            },
            snapshot: Box::new(Cursor::new(serialized)),
        })
    }

    /// Install a snapshot
    pub async fn install_snapshot(
        &self,
        snapshot: &openraft::Snapshot<TypeConfig>,
    ) -> Result<(), RaftError> {
        let snapshot_data = snapshot.snapshot.get_ref();
        let new_state: BTreeMap<Vec<u8>, Vec<u8>> =
            serde_json::from_slice(snapshot_data).map_err(|e| RaftError::Serialization(e))?;

        let mut state = self.state.write().await;
        *state = new_state;
        Ok(())
    }
}

// Implement the RaftStateMachine trait
impl RaftStateMachine<TypeConfig> for StateManager {
    type SnapshotBuilder = StateSnapshotBuilder;

    fn applied_state(
        &mut self,
    ) -> impl std::future::Future<
        Output = Result<
            (
                Option<openraft::LogId<u64>>,
                openraft::StoredMembership<u64, openraft::BasicNode>,
            ),
            openraft::StorageError<u64>,
        >,
    > + Send {
        async move { Ok((None, Default::default())) }
    }

    fn apply<I>(
        &mut self,
        entries: I,
    ) -> impl std::future::Future<Output = Result<Vec<RaftResponse>, openraft::StorageError<u64>>> + Send
    where
        I: IntoIterator<Item = openraft::Entry<TypeConfig>> + Send,
        I::IntoIter: Send,
    {
        let state = self.state.clone();
        async move {
            let mut results = Vec::new();
            for entry in entries {
                match entry.payload {
                    openraft::EntryPayload::Blank => {
                        results.push(RaftResponse {
                            success: true,
                            data: None,
                            error: None,
                        });
                    }
                    openraft::EntryPayload::Normal(command) => {
                        // Apply the command to our state
                        let mut state_guard = state.write().await;
                        state_guard.insert(command.id.as_bytes().to_vec(), command.data.clone());
                        results.push(RaftResponse {
                            success: true,
                            data: None,
                            error: None,
                        });
                    }
                    openraft::EntryPayload::Membership(_) => {
                        results.push(RaftResponse {
                            success: true,
                            data: None,
                            error: None,
                        });
                    }
                }
            }

            Ok(results)
        }
    }

    fn begin_receiving_snapshot(
        &mut self,
    ) -> impl std::future::Future<Output = Result<Box<Cursor<Vec<u8>>>, openraft::StorageError<u64>>>
           + Send {
        async move { Ok(Box::new(Cursor::new(Vec::new()))) }
    }

    fn install_snapshot(
        &mut self,
        _meta: &openraft::SnapshotMeta<u64, openraft::BasicNode>,
        snapshot: Box<Cursor<Vec<u8>>>,
    ) -> impl std::future::Future<Output = Result<(), openraft::StorageError<u64>>> + Send {
        let state = self.state.clone();
        let snapshot_data = snapshot.get_ref().clone();
        async move {
            if let Ok(new_state) =
                serde_json::from_slice::<BTreeMap<Vec<u8>, Vec<u8>>>(&snapshot_data)
            {
                let mut current_state = state.write().await;
                *current_state = new_state;
            }

            Ok(())
        }
    }

    fn get_current_snapshot(
        &mut self,
    ) -> impl std::future::Future<
        Output = Result<Option<openraft::Snapshot<TypeConfig>>, openraft::StorageError<u64>>,
    > + Send {
        async move {
            Ok(None) // TODO: Implement proper snapshot storage
        }
    }

    fn get_snapshot_builder(
        &mut self,
    ) -> impl std::future::Future<Output = Self::SnapshotBuilder> + Send {
        async move { StateSnapshotBuilder::new() }
    }
}

/// Snapshot builder for the state machine
pub struct StateSnapshotBuilder {
    data: Vec<u8>,
}

impl StateSnapshotBuilder {
    /// Creates a new StateSnapshotBuilder
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
}

impl RaftSnapshotBuilder<TypeConfig> for StateSnapshotBuilder {
    fn build_snapshot(
        &mut self,
    ) -> impl std::future::Future<
        Output = Result<openraft::Snapshot<TypeConfig>, openraft::StorageError<u64>>,
    > + Send {
        async move {
            Ok(openraft::Snapshot {
                meta: openraft::SnapshotMeta {
                    last_log_id: None,
                    last_membership: Default::default(),
                    snapshot_id: format!("snapshot-{}", chrono::Utc::now().timestamp()),
                },
                snapshot: Box::new(Cursor::new(self.data.clone())),
            })
        }
    }
}
