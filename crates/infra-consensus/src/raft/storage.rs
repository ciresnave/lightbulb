//! Storage implementation for Raft log and state

use openraft::storage::{LogFlushed, LogState, RaftLogReader, RaftLogStorage, RaftStateMachine};
use openraft::{
    BasicNode, Entry as RaftEntry, LeaderId, LogId, Snapshot, SnapshotMeta, StorageError,
    StoredMembership, Vote,
};
use std::collections::BTreeMap;
use std::future::Future;
use std::ops::RangeBounds;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::types::*;

/// Storage implementation for Raft
#[derive(Debug, Clone)]
pub struct RaftStore {
    /// Path to the storage directory
    pub path: PathBuf,
    /// In-memory log storage
    pub log: Arc<RwLock<BTreeMap<u64, RaftEntry<TypeConfig>>>>,
    /// Current vote state
    pub vote: Arc<RwLock<Option<Vote<u64>>>>,
    /// State machine instance
    pub state_machine: Arc<RwLock<StateMachine>>,
    /// Last purged log ID
    pub last_purged_log_id: Arc<RwLock<Option<LogId<u64>>>>,
    /// Last applied log ID
    pub last_applied_log_id: Arc<RwLock<Option<LogId<u64>>>>,
    /// Current cluster membership
    pub membership: Arc<RwLock<StoredMembership<u64, BasicNode>>>,
    /// Current snapshot
    pub snapshot: Arc<RwLock<Option<Snapshot<TypeConfig>>>>,
}

impl RaftStore {
    /// Creates a new RaftStore with the given storage path
    pub fn new(path: PathBuf) -> Result<Self, super::error::StorageError> {
        Ok(Self {
            path,
            log: Arc::new(RwLock::new(BTreeMap::new())),
            vote: Arc::new(RwLock::new(None)),
            state_machine: Arc::new(RwLock::new(StateMachine::default())),
            last_purged_log_id: Arc::new(RwLock::new(None)),
            last_applied_log_id: Arc::new(RwLock::new(None)),
            membership: Arc::new(RwLock::new(StoredMembership::default())),
            snapshot: Arc::new(RwLock::new(None)),
        })
    }
    /// Store a Raft entry directly (reserved for future optimization)
    #[allow(dead_code)]
    async fn store_entry(&mut self, entry: RaftEntry<TypeConfig>) -> Result<(), StorageError<u64>> {
        let mut log = self.log.write().await;
        log.insert(entry.log_id.index, entry);
        Ok(())
    }

    /// Delete entries from index onwards (reserved for future log compaction)
    #[allow(dead_code)]
    async fn delete_from(&mut self, index: u64) -> Result<(), StorageError<u64>> {
        let mut log = self.log.write().await;
        log.retain(|&k, _| k < index);
        Ok(())
    }
}

impl RaftLogReader<TypeConfig> for RaftStore {
    fn try_get_log_entries<RB: RangeBounds<u64> + Clone + std::fmt::Debug + Send>(
        &mut self,
        range: RB,
    ) -> impl Future<Output = Result<Vec<RaftEntry<TypeConfig>>, StorageError<u64>>> + Send {
        let log = Arc::clone(&self.log);
        async move {
            let log = log.read().await;
            let mut entries = Vec::new();

            let start = match range.start_bound() {
                std::ops::Bound::Included(&start) => start,
                std::ops::Bound::Excluded(&start) => start + 1,
                std::ops::Bound::Unbounded => 0,
            };

            let end = match range.end_bound() {
                std::ops::Bound::Included(&end) => end + 1,
                std::ops::Bound::Excluded(&end) => end,
                std::ops::Bound::Unbounded => u64::MAX,
            };

            for idx in start..end {
                if let Some(entry) = log.get(&idx) {
                    entries.push(entry.clone());
                }
            }

            Ok(entries)
        }
    }
}

impl RaftLogStorage<TypeConfig> for RaftStore {
    type LogReader = Self;

    fn get_log_state(
        &mut self,
    ) -> impl Future<Output = Result<LogState<TypeConfig>, StorageError<u64>>> + Send {
        let log = Arc::clone(&self.log);
        let last_purged = Arc::clone(&self.last_purged_log_id);

        async move {
            let log = log.read().await;
            let last_purged_log_id = last_purged.read().await.clone();
            let last_log_id = log
                .keys()
                .last()
                .map(|&index| LogId::new(LeaderId::new(1, 0), index));

            Ok(LogState {
                last_purged_log_id,
                last_log_id,
            })
        }
    }

    fn save_vote(
        &mut self,
        vote: &Vote<u64>,
    ) -> impl Future<Output = Result<(), StorageError<u64>>> + Send {
        let vote_storage = Arc::clone(&self.vote);
        let vote_clone = vote.clone();

        async move {
            let mut stored_vote = vote_storage.write().await;
            *stored_vote = Some(vote_clone);
            Ok(())
        }
    }

    fn read_vote(
        &mut self,
    ) -> impl Future<Output = Result<Option<Vote<u64>>, StorageError<u64>>> + Send {
        let vote_storage = Arc::clone(&self.vote);

        async move {
            let vote = vote_storage.read().await;
            Ok(vote.clone())
        }
    }

    fn get_log_reader(&mut self) -> impl Future<Output = Self::LogReader> + Send {
        let result = Self {
            path: self.path.clone(),
            log: Arc::clone(&self.log),
            vote: Arc::clone(&self.vote),
            state_machine: Arc::clone(&self.state_machine),
            last_purged_log_id: Arc::clone(&self.last_purged_log_id),
            last_applied_log_id: Arc::clone(&self.last_applied_log_id),
            membership: Arc::clone(&self.membership),
            snapshot: Arc::clone(&self.snapshot),
        };

        async move { result }
    }

    fn append<I>(
        &mut self,
        entries: I,
        callback: LogFlushed<TypeConfig>,
    ) -> impl Future<Output = Result<(), StorageError<u64>>> + Send
    where
        I: IntoIterator<Item = RaftEntry<TypeConfig>> + Send,
        I::IntoIter: Send,
    {
        let log_storage = Arc::clone(&self.log);

        async move {
            let mut log = log_storage.write().await;
            for entry in entries {
                log.insert(entry.log_id.index, entry);
            }

            // Call the callback to indicate persistence completion
            let _ = callback.log_io_completed(Ok(()));
            Ok(())
        }
    }

    fn truncate(
        &mut self,
        log_id: LogId<u64>,
    ) -> impl Future<Output = Result<(), StorageError<u64>>> + Send {
        let log_storage = Arc::clone(&self.log);

        async move {
            let mut log = log_storage.write().await;
            log.retain(|&index, _| index < log_id.index);
            Ok(())
        }
    }

    fn purge(
        &mut self,
        log_id: LogId<u64>,
    ) -> impl Future<Output = Result<(), StorageError<u64>>> + Send {
        let log_storage = Arc::clone(&self.log);
        let purged_storage = Arc::clone(&self.last_purged_log_id);

        async move {
            let mut log = log_storage.write().await;
            log.retain(|&index, _| index > log_id.index);

            let mut last_purged = purged_storage.write().await;
            *last_purged = Some(log_id);
            Ok(())
        }
    }
}

// State machine implementation
/// Snapshot builder for creating Raft snapshots
pub struct SnapshotBuilder;

impl RaftStateMachine<TypeConfig> for RaftStore {
    type SnapshotBuilder = crate::raft::types::StateMachineSnapshotBuilder;

    fn applied_state(
        &mut self,
    ) -> impl Future<
        Output = Result<(Option<LogId<u64>>, StoredMembership<u64, BasicNode>), StorageError<u64>>,
    > + Send {
        let applied = Arc::clone(&self.last_applied_log_id);
        let membership = Arc::clone(&self.membership);

        async move {
            let applied_log_id = applied.read().await.clone();
            let membership = membership.read().await.clone();
            Ok((applied_log_id, membership))
        }
    }
    fn apply<I>(
        &mut self,
        entries: I,
    ) -> impl Future<Output = Result<Vec<RaftResponse>, StorageError<u64>>> + Send
    where
        I: IntoIterator<Item = RaftEntry<TypeConfig>> + Send,
        I::IntoIter: Send,
    {
        let state_machine = Arc::clone(&self.state_machine);
        let applied = Arc::clone(&self.last_applied_log_id);

        async move {
            let mut sm = state_machine.write().await;
            let mut applied_log_id = applied.write().await;

            // Collect entries to pass to state machine
            let entries_vec: Vec<_> = entries.into_iter().collect();

            // Call the state machine's apply method
            let responses = sm.apply(entries_vec.iter().cloned()).await?;

            // Update the last applied log ID
            if let Some(last_entry) = entries_vec.last() {
                *applied_log_id = Some(last_entry.log_id);
            }

            Ok(responses)
        }
    }
    fn begin_receiving_snapshot(
        &mut self,
    ) -> impl Future<
        Output = Result<
            Box<<TypeConfig as openraft::RaftTypeConfig>::SnapshotData>,
            StorageError<u64>,
        >,
    > + Send {
        async move { Ok(Box::new(std::io::Cursor::new(Vec::new()))) }
    }

    fn install_snapshot(
        &mut self,
        meta: &SnapshotMeta<u64, BasicNode>,
        snapshot: Box<<TypeConfig as openraft::RaftTypeConfig>::SnapshotData>,
    ) -> impl Future<Output = Result<(), StorageError<u64>>> + Send {
        let snapshot_storage = Arc::clone(&self.snapshot);
        let meta_clone = meta.clone();

        async move {
            let mut snap = snapshot_storage.write().await;
            *snap = Some(Snapshot {
                meta: meta_clone,
                snapshot: Box::new(*snapshot),
            });
            Ok(())
        }
    }

    fn get_current_snapshot(
        &mut self,
    ) -> impl Future<Output = Result<Option<Snapshot<TypeConfig>>, StorageError<u64>>> + Send {
        let snapshot_storage = Arc::clone(&self.snapshot);

        async move {
            let snapshot = snapshot_storage.read().await;
            Ok(snapshot.clone())
        }
    }
    fn get_snapshot_builder(&mut self) -> impl Future<Output = Self::SnapshotBuilder> + Send {
        async move { crate::raft::types::StateMachineSnapshotBuilder::new() }
    }
}
