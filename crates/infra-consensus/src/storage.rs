//! Storage abstraction for Raft consensus
//!
//! This module provides storage traits and implementations for persistent Raft state.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::network::{LogEntry, SnapshotData};
use crate::{NodeId, ConsensusResult, ConsensusError};

/// Storage backend trait for Raft persistence
#[async_trait::async_trait]
pub trait StorageBackend: Send + Sync {
    /// Get the current term
    async fn get_current_term(&self) -> ConsensusResult<u64>;
    
    /// Set the current term
    async fn set_current_term(&self, term: u64) -> ConsensusResult<()>;
    
    /// Get the node we voted for in the current term
    async fn get_voted_for(&self) -> ConsensusResult<Option<NodeId>>;
    
    /// Set the node we voted for in the current term
    async fn set_voted_for(&self, node_id: Option<NodeId>) -> ConsensusResult<()>;
    
    /// Get log entries in the specified range [start, end)
    async fn get_log_entries(
        &self,
        start: u64,
        end: u64,
    ) -> ConsensusResult<Vec<LogEntry>>;
    
    /// Append log entries
    async fn append_log_entries(&self, entries: Vec<LogEntry>) -> ConsensusResult<()>;
    
    /// Delete log entries from index onwards
    async fn delete_log_entries_from(&self, from_index: u64) -> ConsensusResult<()>;
    
    /// Get the index of the last log entry
    async fn get_last_log_index(&self) -> ConsensusResult<u64>;
    
    /// Get the term of the last log entry
    async fn get_last_log_term(&self) -> ConsensusResult<u64>;
    
    /// Get log entry at the specified index
    async fn get_log_entry(&self, index: u64) -> ConsensusResult<Option<LogEntry>>;
    
    /// Get the current commit index
    async fn get_commit_index(&self) -> ConsensusResult<u64>;
    
    /// Set the current commit index
    async fn set_commit_index(&self, index: u64) -> ConsensusResult<()>;
    
    /// Save a snapshot
    async fn save_snapshot(&self, snapshot: SnapshotData) -> ConsensusResult<()>;
    
    /// Load the latest snapshot
    async fn load_snapshot(&self) -> ConsensusResult<Option<SnapshotData>>;
    
    /// Delete old snapshots (keep only the latest)
    async fn cleanup_snapshots(&self) -> ConsensusResult<()>;
    
    /// Sync all pending writes to durable storage
    async fn sync(&self) -> ConsensusResult<()>;
}

/// DynAniML storage implementation
pub struct DynStorage {
    backend: Box<dyn StorageBackend>,
}

impl DynStorage {
    /// Create a new DynStorage with the specified backend
    pub fn new(backend: Box<dyn StorageBackend>) -> Self {
        Self { backend }
    }
    
    /// Create a file-based storage backend
    pub async fn new_file_storage(data_dir: impl AsRef<Path>) -> ConsensusResult<Self> {
        let backend = FileStorageBackend::new(data_dir).await?;
        Ok(Self::new(Box::new(backend)))
    }
    
    /// Create an in-memory storage backend (for testing)
    pub fn new_memory_storage() -> Self {
        let backend = MemoryStorageBackend::new();
        Self::new(Box::new(backend))
    }
    
    /// Get a reference to the underlying storage backend
    pub fn backend(&self) -> &dyn StorageBackend {
        self.backend.as_ref()
    }
}

#[async_trait::async_trait]
impl StorageBackend for DynStorage {
    async fn get_current_term(&self) -> ConsensusResult<u64> {
        self.backend.get_current_term().await
    }
    
    async fn set_current_term(&self, term: u64) -> ConsensusResult<()> {
        self.backend.set_current_term(term).await
    }
    
    async fn get_voted_for(&self) -> ConsensusResult<Option<NodeId>> {
        self.backend.get_voted_for().await
    }
    
    async fn set_voted_for(&self, node_id: Option<NodeId>) -> ConsensusResult<()> {
        self.backend.set_voted_for(node_id).await
    }
    
    async fn get_log_entries(&self, start: u64, end: u64) -> ConsensusResult<Vec<LogEntry>> {
        self.backend.get_log_entries(start, end).await
    }
    
    async fn append_log_entries(&self, entries: Vec<LogEntry>) -> ConsensusResult<()> {
        self.backend.append_log_entries(entries).await
    }
    
    async fn delete_log_entries_from(&self, from_index: u64) -> ConsensusResult<()> {
        self.backend.delete_log_entries_from(from_index).await
    }
    
    async fn get_last_log_index(&self) -> ConsensusResult<u64> {
        self.backend.get_last_log_index().await
    }
    
    async fn get_last_log_term(&self) -> ConsensusResult<u64> {
        self.backend.get_last_log_term().await
    }
    
    async fn get_log_entry(&self, index: u64) -> ConsensusResult<Option<LogEntry>> {
        self.backend.get_log_entry(index).await
    }
    
    async fn get_commit_index(&self) -> ConsensusResult<u64> {
        self.backend.get_commit_index().await
    }
    
    async fn set_commit_index(&self, index: u64) -> ConsensusResult<()> {
        self.backend.set_commit_index(index).await
    }
    
    async fn save_snapshot(&self, snapshot: SnapshotData) -> ConsensusResult<()> {
        self.backend.save_snapshot(snapshot).await
    }
    
    async fn load_snapshot(&self) -> ConsensusResult<Option<SnapshotData>> {
        self.backend.load_snapshot().await
    }
    
    async fn cleanup_snapshots(&self) -> ConsensusResult<()> {
        self.backend.cleanup_snapshots().await
    }
    
    async fn sync(&self) -> ConsensusResult<()> {
        self.backend.sync().await
    }
}

/// In-memory storage backend for testing
pub struct MemoryStorageBackend {
    state: Arc<RwLock<MemoryState>>,
}

#[derive(Debug, Default)]
struct MemoryState {
    current_term: u64,
    voted_for: Option<NodeId>,
    log_entries: BTreeMap<u64, LogEntry>,
    commit_index: u64,
    snapshot: Option<SnapshotData>,
}

impl MemoryStorageBackend {
    /// Create a new memory storage backend
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(MemoryState::default())),
        }
    }
}

#[async_trait::async_trait]
impl StorageBackend for MemoryStorageBackend {
    async fn get_current_term(&self) -> ConsensusResult<u64> {
        let state = self.state.read().await;
        Ok(state.current_term)
    }
    
    async fn set_current_term(&self, term: u64) -> ConsensusResult<()> {
        let mut state = self.state.write().await;
        state.current_term = term;
        Ok(())
    }
    
    async fn get_voted_for(&self) -> ConsensusResult<Option<NodeId>> {
        let state = self.state.read().await;
        Ok(state.voted_for)
    }
    
    async fn set_voted_for(&self, node_id: Option<NodeId>) -> ConsensusResult<()> {
        let mut state = self.state.write().await;
        state.voted_for = node_id;
        Ok(())
    }
    
    async fn get_log_entries(&self, start: u64, end: u64) -> ConsensusResult<Vec<LogEntry>> {
        let state = self.state.read().await;
        let mut entries = Vec::new();
        
        for index in start..end {
            if let Some(entry) = state.log_entries.get(&index) {
                entries.push(entry.clone());
            }
        }
        
        Ok(entries)
    }
    
    async fn append_log_entries(&self, entries: Vec<LogEntry>) -> ConsensusResult<()> {
        let mut state = self.state.write().await;
        
        for entry in entries {
            state.log_entries.insert(entry.index, entry);
        }
        
        Ok(())
    }
    
    async fn delete_log_entries_from(&self, from_index: u64) -> ConsensusResult<()> {
        let mut state = self.state.write().await;
        
        let keys_to_remove: Vec<u64> = state
            .log_entries
            .range(from_index..)
            .map(|(&k, _)| k)
            .collect();
        
        for key in keys_to_remove {
            state.log_entries.remove(&key);
        }
        
        Ok(())
    }
    
    async fn get_last_log_index(&self) -> ConsensusResult<u64> {
        let state = self.state.read().await;
        Ok(state.log_entries.keys().last().copied().unwrap_or(0))
    }
    
    async fn get_last_log_term(&self) -> ConsensusResult<u64> {
        let state = self.state.read().await;
        if let Some(entry) = state.log_entries.values().last() {
            Ok(entry.term)
        } else {
            Ok(0)
        }
    }
    
    async fn get_log_entry(&self, index: u64) -> ConsensusResult<Option<LogEntry>> {
        let state = self.state.read().await;
        Ok(state.log_entries.get(&index).cloned())
    }
    
    async fn get_commit_index(&self) -> ConsensusResult<u64> {
        let state = self.state.read().await;
        Ok(state.commit_index)
    }
    
    async fn set_commit_index(&self, index: u64) -> ConsensusResult<()> {
        let mut state = self.state.write().await;
        state.commit_index = index;
        Ok(())
    }
    
    async fn save_snapshot(&self, snapshot: SnapshotData) -> ConsensusResult<()> {
        let mut state = self.state.write().await;
        state.snapshot = Some(snapshot);
        Ok(())
    }
    
    async fn load_snapshot(&self) -> ConsensusResult<Option<SnapshotData>> {
        let state = self.state.read().await;
        Ok(state.snapshot.clone())
    }
    
    async fn cleanup_snapshots(&self) -> ConsensusResult<()> {
        // Nothing to cleanup in memory
        Ok(())
    }
    
    async fn sync(&self) -> ConsensusResult<()> {
        // Nothing to sync in memory
        Ok(())
    }
}

/// File-based storage backend
pub struct FileStorageBackend {
    data_dir: PathBuf,
    state: Arc<RwLock<FileState>>,
}

#[derive(Debug, Default)]
struct FileState {
    current_term: u64,
    voted_for: Option<NodeId>,
    commit_index: u64,
    last_log_index: u64,
    last_log_term: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct PersistentState {
    current_term: u64,
    voted_for: Option<NodeId>,
    commit_index: u64,
}

impl FileStorageBackend {
    /// Create a new file storage backend
    pub async fn new(data_dir: impl AsRef<Path>) -> ConsensusResult<Self> {
        let data_dir = data_dir.as_ref().to_path_buf();
        
        // Create data directory if it doesn't exist
        if !data_dir.exists() {
            fs::create_dir_all(&data_dir).map_err(|e| {
                ConsensusError::Storage(format!("Failed to create data directory: {}", e))
            })?;
        }
        
        let mut backend = Self {
            data_dir,
            state: Arc::new(RwLock::new(FileState::default())),
        };
        
        // Load existing state
        backend.load_state().await?;
        
        Ok(backend)
    }
    
    /// Get path to the state file
    fn state_file_path(&self) -> PathBuf {
        self.data_dir.join("raft_state.json")
    }
    
    /// Get path to the log file
    fn log_file_path(&self) -> PathBuf {
        self.data_dir.join("raft_log.jsonl")
    }
    
    /// Get path to snapshots directory
    fn snapshots_dir(&self) -> PathBuf {
        self.data_dir.join("snapshots")
    }
    
    /// Load persistent state from disk
    async fn load_state(&mut self) -> ConsensusResult<()> {
        let state_file = self.state_file_path();
        
        if state_file.exists() {
            let content = fs::read_to_string(&state_file).map_err(|e| {
                ConsensusError::Storage(format!("Failed to read state file: {}", e))
            })?;
            
            let persistent_state: PersistentState = serde_json::from_str(&content)
                .map_err(|e| {
                    ConsensusError::Storage(format!("Failed to parse state file: {}", e))
                })?;
            
            let mut state = self.state.write().await;
            state.current_term = persistent_state.current_term;
            state.voted_for = persistent_state.voted_for;
            state.commit_index = persistent_state.commit_index;
            
            info!("Loaded persistent state: term={}, voted_for={:?}, commit_index={}", 
                  state.current_term, state.voted_for, state.commit_index);
        }
        
        // Load log metadata
        self.load_log_metadata().await?;
        
        Ok(())
    }
    
    /// Save persistent state to disk
    async fn save_state(&self) -> ConsensusResult<()> {
        let state = self.state.read().await;
        let persistent_state = PersistentState {
            current_term: state.current_term,
            voted_for: state.voted_for,
            commit_index: state.commit_index,
        };
        
        let content = serde_json::to_string_pretty(&persistent_state).map_err(|e| {
            ConsensusError::Storage(format!("Failed to serialize state: {}", e))
        })?;
        
        let state_file = self.state_file_path();
        fs::write(&state_file, content).map_err(|e| {
            ConsensusError::Storage(format!("Failed to write state file: {}", e))
        })?;
        
        debug!("Saved persistent state to {:?}", state_file);
        Ok(())
    }
    
    /// Load log metadata (last index and term)
    async fn load_log_metadata(&self) -> ConsensusResult<()> {
        let log_file = self.log_file_path();
        
        if !log_file.exists() {
            return Ok(());
        }
        
        let content = fs::read_to_string(&log_file).map_err(|e| {
            ConsensusError::Storage(format!("Failed to read log file: {}", e))
        })?;
        
        let mut last_index = 0;
        let mut last_term = 0;
        
        for line in content.lines() {
            if let Ok(entry) = serde_json::from_str::<LogEntry>(line) {
                if entry.index > last_index {
                    last_index = entry.index;
                    last_term = entry.term;
                }
            }
        }
        
        let mut state = self.state.write().await;
        state.last_log_index = last_index;
        state.last_log_term = last_term;
        
        debug!("Loaded log metadata: last_index={}, last_term={}", last_index, last_term);
        Ok(())
    }
}

#[async_trait::async_trait]
impl StorageBackend for FileStorageBackend {
    async fn get_current_term(&self) -> ConsensusResult<u64> {
        let state = self.state.read().await;
        Ok(state.current_term)
    }
    
    async fn set_current_term(&self, term: u64) -> ConsensusResult<()> {
        {
            let mut state = self.state.write().await;
            state.current_term = term;
        }
        self.save_state().await
    }
    
    async fn get_voted_for(&self) -> ConsensusResult<Option<NodeId>> {
        let state = self.state.read().await;
        Ok(state.voted_for)
    }
    
    async fn set_voted_for(&self, node_id: Option<NodeId>) -> ConsensusResult<()> {
        {
            let mut state = self.state.write().await;
            state.voted_for = node_id;
        }
        self.save_state().await
    }
    
    async fn get_log_entries(&self, start: u64, end: u64) -> ConsensusResult<Vec<LogEntry>> {
        let log_file = self.log_file_path();
        
        if !log_file.exists() {
            return Ok(vec![]);
        }
        
        let content = fs::read_to_string(&log_file).map_err(|e| {
            ConsensusError::Storage(format!("Failed to read log file: {}", e))
        })?;
        
        let mut entries = Vec::new();
        
        for line in content.lines() {
            if let Ok(entry) = serde_json::from_str::<LogEntry>(line) {
                if entry.index >= start && entry.index < end {
                    entries.push(entry);
                }
            }
        }
        
        // Sort by index
        entries.sort_by_key(|e| e.index);
        Ok(entries)
    }
    
    async fn append_log_entries(&self, entries: Vec<LogEntry>) -> ConsensusResult<()> {
        if entries.is_empty() {
            return Ok(());
        }
        
        let log_file = self.log_file_path();
        let mut content = String::new();
        
        for entry in &entries {
            let line = serde_json::to_string(entry).map_err(|e| {
                ConsensusError::Storage(format!("Failed to serialize log entry: {}", e))
            })?;
            content.push_str(&line);
            content.push('\n');
        }
        
        // Append to log file
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_file)
            .and_then(|mut file| {
                use std::io::Write;
                file.write_all(content.as_bytes())
            })
            .map_err(|e| {
                ConsensusError::Storage(format!("Failed to append to log file: {}", e))
            })?;
        
        // Update metadata
        if let Some(last_entry) = entries.last() {
            let mut state = self.state.write().await;
            state.last_log_index = last_entry.index;
            state.last_log_term = last_entry.term;
        }
        
        debug!("Appended {} log entries", entries.len());
        Ok(())
    }
    
    async fn delete_log_entries_from(&self, from_index: u64) -> ConsensusResult<()> {
        let log_file = self.log_file_path();
        
        if !log_file.exists() {
            return Ok(());
        }
        
        let content = fs::read_to_string(&log_file).map_err(|e| {
            ConsensusError::Storage(format!("Failed to read log file: {}", e))
        })?;
        
        let mut new_content = String::new();
        let mut last_index = 0;
        let mut last_term = 0;
        
        for line in content.lines() {
            if let Ok(entry) = serde_json::from_str::<LogEntry>(line) {
                if entry.index < from_index {
                    new_content.push_str(line);
                    new_content.push('\n');
                    last_index = entry.index;
                    last_term = entry.term;
                }
            }
        }
        
        fs::write(&log_file, new_content).map_err(|e| {
            ConsensusError::Storage(format!("Failed to write log file: {}", e))
        })?;
        
        // Update metadata
        let mut state = self.state.write().await;
        state.last_log_index = last_index;
        state.last_log_term = last_term;
        
        debug!("Deleted log entries from index {}", from_index);
        Ok(())
    }
    
    async fn get_last_log_index(&self) -> ConsensusResult<u64> {
        let state = self.state.read().await;
        Ok(state.last_log_index)
    }
    
    async fn get_last_log_term(&self) -> ConsensusResult<u64> {
        let state = self.state.read().await;
        Ok(state.last_log_term)
    }
    
    async fn get_log_entry(&self, index: u64) -> ConsensusResult<Option<LogEntry>> {
        let entries = self.get_log_entries(index, index + 1).await?;
        Ok(entries.into_iter().next())
    }
    
    async fn get_commit_index(&self) -> ConsensusResult<u64> {
        let state = self.state.read().await;
        Ok(state.commit_index)
    }
    
    async fn set_commit_index(&self, index: u64) -> ConsensusResult<()> {
        {
            let mut state = self.state.write().await;
            state.commit_index = index;
        }
        self.save_state().await
    }
    
    async fn save_snapshot(&self, snapshot: SnapshotData) -> ConsensusResult<()> {
        let snapshots_dir = self.snapshots_dir();
        
        if !snapshots_dir.exists() {
            fs::create_dir_all(&snapshots_dir).map_err(|e| {
                ConsensusError::Storage(format!("Failed to create snapshots directory: {}", e))
            })?;
        }
        
        let snapshot_file = snapshots_dir.join(format!(
            "snapshot_{}_{}.json",
            snapshot.metadata.last_included_term,
            snapshot.metadata.last_included_index
        ));
        
        let content = serde_json::to_string_pretty(&snapshot).map_err(|e| {
            ConsensusError::Storage(format!("Failed to serialize snapshot: {}", e))
        })?;
        
        fs::write(&snapshot_file, content).map_err(|e| {
            ConsensusError::Storage(format!("Failed to write snapshot file: {}", e))
        })?;
        
        info!("Saved snapshot to {:?}", snapshot_file);
        Ok(())
    }
    
    async fn load_snapshot(&self) -> ConsensusResult<Option<SnapshotData>> {
        let snapshots_dir = self.snapshots_dir();
        
        if !snapshots_dir.exists() {
            return Ok(None);
        }
        
        let mut latest_snapshot = None;
        let mut latest_index = 0;
        
        let entries = fs::read_dir(&snapshots_dir).map_err(|e| {
            ConsensusError::Storage(format!("Failed to read snapshots directory: {}", e))
        })?;
        
        for entry in entries {
            let entry = entry.map_err(|e| {
                ConsensusError::Storage(format!("Failed to read directory entry: {}", e))
            })?;
            
            let path = entry.path();
            if path.is_file() && path.extension().map_or(false, |ext| ext == "json") {
                let content = fs::read_to_string(&path).map_err(|e| {
                    ConsensusError::Storage(format!("Failed to read snapshot file: {}", e))
                })?;
                
                if let Ok(snapshot) = serde_json::from_str::<SnapshotData>(&content) {
                    if snapshot.metadata.last_included_index > latest_index {
                        latest_index = snapshot.metadata.last_included_index;
                        latest_snapshot = Some(snapshot);
                    }
                }
            }
        }
        
        if latest_snapshot.is_some() {
            info!("Loaded latest snapshot with index {}", latest_index);
        }
        
        Ok(latest_snapshot)
    }
    
    async fn cleanup_snapshots(&self) -> ConsensusResult<()> {
        let snapshots_dir = self.snapshots_dir();
        
        if !snapshots_dir.exists() {
            return Ok(());
        }
        
        let mut snapshots = Vec::new();
        
        let entries = fs::read_dir(&snapshots_dir).map_err(|e| {
            ConsensusError::Storage(format!("Failed to read snapshots directory: {}", e))
        })?;
        
        for entry in entries {
            let entry = entry.map_err(|e| {
                ConsensusError::Storage(format!("Failed to read directory entry: {}", e))
            })?;
            
            let path = entry.path();
            if path.is_file() && path.extension().map_or(false, |ext| ext == "json") {
                if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                    snapshots.push((path.clone(), file_name.to_string()));
                }
            }
        }
        
        // Sort by filename (which includes term and index) and keep only the latest
        snapshots.sort_by(|a, b| a.1.cmp(&b.1));
        
        if snapshots.len() > 1 {
            for (path, _) in &snapshots[..snapshots.len() - 1] {
                if let Err(e) = fs::remove_file(path) {
                    warn!("Failed to remove old snapshot {:?}: {}", path, e);
                }
            }
            info!("Cleaned up {} old snapshots", snapshots.len() - 1);
        }
        
        Ok(())
    }
    
    async fn sync(&self) -> ConsensusResult<()> {
        // For file storage, we write synchronously, so no additional sync needed
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_memory_storage_backend() {
        let storage = MemoryStorageBackend::new();
        
        // Test term operations
        assert_eq!(storage.get_current_term().await.unwrap(), 0);
        storage.set_current_term(42).await.unwrap();
        assert_eq!(storage.get_current_term().await.unwrap(), 42);
        
        // Test vote operations
        assert_eq!(storage.get_voted_for().await.unwrap(), None);
        storage.set_voted_for(Some(1)).await.unwrap();
        assert_eq!(storage.get_voted_for().await.unwrap(), Some(1));
        
        // Test log operations
        let entries = vec![
            LogEntry {
                index: 1,
                term: 1,
                data: b"entry1".to_vec(),
            },
            LogEntry {
                index: 2,
                term: 1,
                data: b"entry2".to_vec(),
            },
        ];
        
        storage.append_log_entries(entries.clone()).await.unwrap();
        assert_eq!(storage.get_last_log_index().await.unwrap(), 2);
        assert_eq!(storage.get_last_log_term().await.unwrap(), 1);
        
        let retrieved = storage.get_log_entries(1, 3).await.unwrap();
        assert_eq!(retrieved.len(), 2);
        assert_eq!(retrieved[0].index, 1);
        assert_eq!(retrieved[1].index, 2);
    }
    
    #[tokio::test]
    async fn test_file_storage_backend() {
        let temp_dir = TempDir::new().unwrap();
        let storage = FileStorageBackend::new(temp_dir.path()).await.unwrap();
        
        // Test term persistence
        storage.set_current_term(42).await.unwrap();
        storage.set_voted_for(Some(1)).await.unwrap();
        
        // Create a new storage instance to test persistence
        let storage2 = FileStorageBackend::new(temp_dir.path()).await.unwrap();
        assert_eq!(storage2.get_current_term().await.unwrap(), 42);
        assert_eq!(storage2.get_voted_for().await.unwrap(), Some(1));
    }
    
    #[tokio::test]
    async fn test_dyn_storage() {
        let storage = DynStorage::new_memory_storage();
        
        // Test basic operations
        storage.set_current_term(10).await.unwrap();
        assert_eq!(storage.get_current_term().await.unwrap(), 10);
        
        let entries = vec![LogEntry {
            index: 1,
            term: 10,
            data: b"test data".to_vec(),
        }];
        
        storage.append_log_entries(entries).await.unwrap();
        assert_eq!(storage.get_last_log_index().await.unwrap(), 1);
        
        let entry = storage.get_log_entry(1).await.unwrap().unwrap();
        assert_eq!(entry.data, b"test data");
    }
}
