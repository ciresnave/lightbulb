//! State Persistence and Recovery
//!
//! Checkpoint/restore infrastructure for full inference state:
//! - KV cache snapshots
//! - Knowledge Base facts and history
//! - Pipeline execution state
//! - Decomposition history and active problems
//! - Request metadata and partial generations
//!
//! Enables:
//! - Graceful shutdown/restart without losing context
//! - Long-running iterative reasoning workflows
//! - Distributed inference with state migration
//! - Debugging and replay of inference sessions

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use super::decomposition::{Decomposition, DecompositionHistory, Problem};
use super::knowledge_base::{EvictionRecord, Fact, KnowledgeBase};

/// Checkpoint identifier (timestamp-based)
pub type CheckpointId = String;

/// Full inference state snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceCheckpoint {
    /// Unique checkpoint ID
    pub id: CheckpointId,

    /// Timestamp of checkpoint creation
    pub timestamp: u64,

    /// KV cache state (serialized)
    pub kv_cache_state: Option<KvCacheSnapshot>,

    /// Knowledge Base snapshot
    pub kb_snapshot: KnowledgeBaseSnapshot,

    /// Pipeline execution state
    pub pipeline_state: Option<PipelineSnapshot>,

    /// Decomposition history
    pub decomposition_history: Vec<DecompositionHistory>,

    /// Active decompositions (problem ID → decomposition)
    pub active_decompositions: HashMap<String, Decomposition>,

    /// Active problems (problem ID → problem)
    pub active_problems: HashMap<String, Problem>,

    /// Request metadata
    pub metadata: HashMap<String, String>,

    /// Partial generations (request ID → partial text)
    pub partial_generations: HashMap<String, String>,
}

/// KV cache snapshot (layer-wise)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheSnapshot {
    /// Number of layers
    pub num_layers: usize,

    /// Sequence length at checkpoint time
    pub seq_len: usize,

    /// Per-layer cache metadata (sizes, shapes)
    pub layer_metadata: Vec<LayerCacheMetadata>,

    /// Actual cache data (serialized tensors)
    /// In production: would use memory-mapped files or external storage
    /// For now: serialize to bytes
    pub cache_data: Vec<u8>,
}

/// Metadata for a single layer's KV cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerCacheMetadata {
    pub layer_idx: usize,
    pub k_shape: Vec<usize>,
    pub v_shape: Vec<usize>,
    pub dtype: String, // "f32", "f16", etc.
}

/// Knowledge Base snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeBaseSnapshot {
    /// All facts in KB
    pub facts: Vec<Fact>,

    /// Eviction history (for debugging)
    pub eviction_history: Vec<EvictionRecord>,

    /// Statistics snapshot
    pub stats: KnowledgeBaseStatsSnapshot,
}

/// KB statistics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeBaseStatsSnapshot {
    pub fact_count: usize,
    pub eviction_count: usize,
    pub retrieval_count: usize,
}

/// Pipeline execution state snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineSnapshot {
    /// Current stage ID
    pub current_stage: String,

    /// Completed stages
    pub completed_stages: Vec<String>,

    /// Stage execution order
    pub execution_order: Vec<String>,

    /// Per-stage data (stage ID → serialized data)
    pub stage_data: HashMap<String, Vec<u8>>,

    /// Pipeline metadata
    pub metadata: HashMap<String, String>,
}

/// Checkpoint manager
pub struct CheckpointManager {
    /// Checkpoint directory
    checkpoint_dir: PathBuf,

    /// Maximum checkpoints to retain (LRU eviction)
    max_checkpoints: usize,

    /// Checkpoint metadata cache (id → metadata)
    checkpoint_metadata: HashMap<CheckpointId, CheckpointMetadata>,
}

/// Checkpoint metadata (lightweight)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub id: CheckpointId,
    pub timestamp: u64,
    pub size_bytes: u64,
    pub num_facts: usize,
    pub num_active_problems: usize,
    pub has_kv_cache: bool,
    pub has_pipeline_state: bool,
    /// Parent checkpoint ID (for branching)
    pub parent_id: Option<CheckpointId>,
    /// Branch name (for identification)
    pub branch_name: Option<String>,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    pub fn new(checkpoint_dir: impl AsRef<Path>, max_checkpoints: usize) -> Result<Self> {
        let checkpoint_dir = checkpoint_dir.as_ref().to_path_buf();
        fs::create_dir_all(&checkpoint_dir).context("Failed to create checkpoint directory")?;

        let mut manager = Self {
            checkpoint_dir,
            max_checkpoints,
            checkpoint_metadata: HashMap::new(),
        };

        // Load existing checkpoint metadata
        manager.load_metadata()?;

        Ok(manager)
    }

    /// Save a checkpoint
    pub fn save(&mut self, checkpoint: &InferenceCheckpoint) -> Result<PathBuf> {
        let checkpoint_path = self.checkpoint_dir.join(format!("{}.json", checkpoint.id));

        // Serialize to JSON (could use bincode for efficiency)
        let file = File::create(&checkpoint_path).context("Failed to create checkpoint file")?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, checkpoint)
            .context("Failed to serialize checkpoint")?;

        // Update metadata
        let metadata = CheckpointMetadata {
            id: checkpoint.id.clone(),
            timestamp: checkpoint.timestamp,
            size_bytes: fs::metadata(&checkpoint_path)?.len(),
            num_facts: checkpoint.kb_snapshot.facts.len(),
            num_active_problems: checkpoint.active_problems.len(),
            has_kv_cache: checkpoint.kv_cache_state.is_some(),
            has_pipeline_state: checkpoint.pipeline_state.is_some(),
            parent_id: checkpoint.metadata.get("parent_id").map(|s| s.to_string()),
            branch_name: checkpoint
                .metadata
                .get("branch_name")
                .map(|s| s.to_string()),
        };

        self.checkpoint_metadata
            .insert(checkpoint.id.clone(), metadata);

        // Save metadata index
        self.save_metadata()?;

        // Evict old checkpoints if over limit
        self.evict_old_checkpoints()?;

        Ok(checkpoint_path)
    }

    /// Load a checkpoint
    pub fn load(&self, checkpoint_id: &str) -> Result<InferenceCheckpoint> {
        let checkpoint_path = self.checkpoint_dir.join(format!("{}.json", checkpoint_id));

        let file = File::open(&checkpoint_path).context("Failed to open checkpoint file")?;
        let reader = BufReader::new(file);
        let checkpoint: InferenceCheckpoint =
            serde_json::from_reader(reader).context("Failed to deserialize checkpoint")?;

        Ok(checkpoint)
    }

    /// List all checkpoints (sorted by timestamp, newest first)
    pub fn list_checkpoints(&self) -> Vec<CheckpointMetadata> {
        let mut checkpoints: Vec<_> = self.checkpoint_metadata.values().cloned().collect();
        checkpoints.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        checkpoints
    }

    /// Get the most recent checkpoint
    pub fn get_latest(&self) -> Option<CheckpointMetadata> {
        self.checkpoint_metadata
            .values()
            .max_by_key(|m| m.timestamp)
            .cloned()
    }

    /// Delete a checkpoint
    pub fn delete(&mut self, checkpoint_id: &str) -> Result<()> {
        let checkpoint_path = self.checkpoint_dir.join(format!("{}.json", checkpoint_id));

        if checkpoint_path.exists() {
            fs::remove_file(&checkpoint_path).context("Failed to delete checkpoint file")?;
        }

        self.checkpoint_metadata.remove(checkpoint_id);
        self.save_metadata()?;

        Ok(())
    }

    /// Delete all checkpoints
    pub fn clear_all(&mut self) -> Result<()> {
        for checkpoint_id in self.checkpoint_metadata.keys().cloned().collect::<Vec<_>>() {
            self.delete(&checkpoint_id)?;
        }
        Ok(())
    }

    /// Create a branch from an existing checkpoint
    ///
    /// This creates a new checkpoint that is a copy of the parent, allowing
    /// exploration of different reasoning paths from a common starting point.
    ///
    /// # Arguments
    /// * `parent_id` - The checkpoint to branch from
    /// * `branch_name` - Human-readable name for this branch (e.g., "computational", "structural")
    ///
    /// # Returns
    /// The checkpoint ID of the new branch
    pub fn branch(&mut self, parent_id: &str, branch_name: &str) -> Result<CheckpointId> {
        let parent = self.load(parent_id)?;

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let branch_id = format!("checkpoint_{}_branch_{}", timestamp, branch_name);

        let mut branch_checkpoint = parent.clone();
        branch_checkpoint.id = branch_id.clone();
        branch_checkpoint.timestamp = timestamp;
        branch_checkpoint
            .metadata
            .insert("parent_id".to_string(), parent_id.to_string());
        branch_checkpoint
            .metadata
            .insert("branch_name".to_string(), branch_name.to_string());

        self.save(&branch_checkpoint)?;
        Ok(branch_id)
    }

    /// List all branches of a checkpoint
    ///
    /// Returns all checkpoints that were created as branches from the given parent.
    pub fn list_branches(&self, parent_id: &str) -> Vec<CheckpointMetadata> {
        self.checkpoint_metadata
            .values()
            .filter(|meta| meta.parent_id.as_deref() == Some(parent_id))
            .cloned()
            .collect()
    }

    /// Merge multiple branches into a single checkpoint
    ///
    /// Combines knowledge bases from multiple branches using confidence-based
    /// conflict resolution (highest confidence wins).
    ///
    /// # Arguments
    /// * `branch_ids` - The checkpoint IDs to merge
    /// * `merge_name` - Name for the merged checkpoint
    ///
    /// # Returns
    /// The checkpoint ID of the merged result
    pub fn merge_branches(
        &mut self,
        branch_ids: &[&str],
        merge_name: &str,
    ) -> Result<CheckpointId> {
        let branches: Vec<_> = branch_ids
            .iter()
            .map(|id| self.load(id))
            .collect::<Result<_>>()?;

        // Merge KBs: union of facts with highest confidence wins
        let merged_kb =
            merge_knowledge_bases(&branches.iter().map(|b| &b.kb_snapshot).collect::<Vec<_>>())?;

        // Combine decomposition histories (preserve all)
        let merged_history: Vec<_> = branches
            .iter()
            .flat_map(|b| b.decomposition_history.clone())
            .collect();

        // Merge active decompositions (keep all unique)
        let mut merged_decomps = HashMap::new();
        for branch in &branches {
            for (key, decomp) in &branch.active_decompositions {
                merged_decomps.insert(key.clone(), decomp.clone());
            }
        }

        // Merge active problems (keep all unique)
        let mut merged_problems = HashMap::new();
        for branch in &branches {
            for (key, problem) in &branch.active_problems {
                merged_problems.insert(key.clone(), problem.clone());
            }
        }

        // Create merged checkpoint
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let merge_id = format!("checkpoint_{}_merged_{}", timestamp, merge_name);

        let mut metadata = HashMap::new();
        metadata.insert("merge_name".to_string(), merge_name.to_string());
        metadata.insert("merged_from".to_string(), branch_ids.join(","));

        let merged_checkpoint = InferenceCheckpoint {
            id: merge_id.clone(),
            timestamp,
            kv_cache_state: None, // KV cache not merged (would need more complex logic)
            kb_snapshot: merged_kb,
            pipeline_state: None, // Pipeline state not merged
            decomposition_history: merged_history,
            active_decompositions: merged_decomps,
            active_problems: merged_problems,
            metadata,
            partial_generations: HashMap::new(),
        };

        self.save(&merged_checkpoint)?;
        Ok(merge_id)
    }

    /// Load metadata index from disk
    fn load_metadata(&mut self) -> Result<()> {
        let metadata_path = self.checkpoint_dir.join("metadata.json");

        if metadata_path.exists() {
            let file = File::open(&metadata_path).context("Failed to open metadata file")?;
            let reader = BufReader::new(file);
            self.checkpoint_metadata =
                serde_json::from_reader(reader).context("Failed to deserialize metadata")?;
        }

        Ok(())
    }

    /// Save metadata index to disk
    fn save_metadata(&self) -> Result<()> {
        let metadata_path = self.checkpoint_dir.join("metadata.json");
        let file = File::create(&metadata_path).context("Failed to create metadata file")?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &self.checkpoint_metadata)
            .context("Failed to serialize metadata")?;
        Ok(())
    }

    /// Evict old checkpoints (LRU)
    fn evict_old_checkpoints(&mut self) -> Result<()> {
        while self.checkpoint_metadata.len() > self.max_checkpoints {
            // Find oldest checkpoint
            let oldest_id = self
                .checkpoint_metadata
                .values()
                .min_by_key(|m| m.timestamp)
                .map(|m| m.id.clone());

            if let Some(id) = oldest_id {
                self.delete(&id)?;
            } else {
                break;
            }
        }
        Ok(())
    }
}

/// Helper to create a checkpoint from current state
pub fn create_checkpoint(
    kb: &KnowledgeBase,
    decomposition_history: Vec<DecompositionHistory>,
    active_decompositions: HashMap<String, Decomposition>,
    active_problems: HashMap<String, Problem>,
) -> InferenceCheckpoint {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    let id = format!("checkpoint_{}", timestamp);

    // Snapshot KB
    let kb_snapshot = KnowledgeBaseSnapshot {
        facts: kb.get_all_facts(),
        eviction_history: kb.eviction_history().to_vec(),
        stats: KnowledgeBaseStatsSnapshot {
            fact_count: kb.stats().fact_count,
            eviction_count: kb.stats().eviction_count,
            retrieval_count: kb.stats().retrieval_count,
        },
    };

    InferenceCheckpoint {
        id,
        timestamp,
        kv_cache_state: None, // TODO: Implement KV cache serialization
        kb_snapshot,
        pipeline_state: None, // TODO: Implement pipeline state serialization
        decomposition_history,
        active_decompositions,
        active_problems,
        metadata: HashMap::new(),
        partial_generations: HashMap::new(),
    }
}

/// Helper to restore KB from checkpoint
pub fn restore_kb_from_checkpoint(checkpoint: &InferenceCheckpoint) -> Result<KnowledgeBase> {
    let mut kb = KnowledgeBase::new();

    // Restore facts
    for fact in &checkpoint.kb_snapshot.facts {
        kb.add_fact(fact.clone())
            .context("Failed to restore fact to KB")?;
    }

    Ok(kb)
}

/// Helper to merge multiple knowledge base snapshots
///
/// Combines facts from multiple KB snapshots using confidence-based conflict resolution:
/// - If a fact key exists in multiple branches, keep the one with highest confidence
/// - All unique facts are preserved
/// - Eviction histories are combined
///
/// # Arguments
/// * `snapshots` - References to KB snapshots to merge
///
/// # Returns
/// A new KB snapshot containing merged facts
fn merge_knowledge_bases(snapshots: &[&KnowledgeBaseSnapshot]) -> Result<KnowledgeBaseSnapshot> {
    let mut fact_map: HashMap<String, Fact> = HashMap::new();

    // Merge facts: highest confidence wins for conflicts
    for snapshot in snapshots {
        for fact in &snapshot.facts {
            match fact_map.get(&fact.key) {
                None => {
                    // New fact, add it
                    fact_map.insert(fact.key.clone(), fact.clone());
                }
                Some(existing) => {
                    // Conflict: keep fact with higher confidence
                    if fact.confidence > existing.confidence {
                        fact_map.insert(fact.key.clone(), fact.clone());
                    }
                }
            }
        }
    }

    // Combine eviction histories (for debugging merged branches)
    let mut merged_evictions = Vec::new();
    for snapshot in snapshots {
        merged_evictions.extend(snapshot.eviction_history.clone());
    }

    let fact_count = fact_map.len();
    let facts: Vec<Fact> = fact_map.into_values().collect();

    Ok(KnowledgeBaseSnapshot {
        facts,
        eviction_history: merged_evictions,
        stats: KnowledgeBaseStatsSnapshot {
            fact_count,
            eviction_count: 0, // Reset for merged KB
            retrieval_count: 0,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_checkpoint_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let manager = CheckpointManager::new(temp_dir.path(), 5).unwrap();

        assert_eq!(manager.max_checkpoints, 5);
        assert_eq!(manager.checkpoint_metadata.len(), 0);
    }

    #[test]
    fn test_save_and_load_checkpoint() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = CheckpointManager::new(temp_dir.path(), 5).unwrap();

        let kb = KnowledgeBase::new();
        let checkpoint = create_checkpoint(&kb, vec![], HashMap::new(), HashMap::new());

        // Save
        manager.save(&checkpoint).unwrap();

        // Load
        let loaded = manager.load(&checkpoint.id).unwrap();
        assert_eq!(loaded.id, checkpoint.id);
        assert_eq!(loaded.timestamp, checkpoint.timestamp);
    }

    #[test]
    fn test_list_checkpoints() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = CheckpointManager::new(temp_dir.path(), 5).unwrap();

        let kb = KnowledgeBase::new();

        // Create multiple checkpoints
        for _ in 0..3 {
            let checkpoint = create_checkpoint(&kb, vec![], HashMap::new(), HashMap::new());
            manager.save(&checkpoint).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        let checkpoints = manager.list_checkpoints();
        assert_eq!(checkpoints.len(), 3);

        // Should be sorted by timestamp (newest first)
        assert!(checkpoints[0].timestamp >= checkpoints[1].timestamp);
        assert!(checkpoints[1].timestamp >= checkpoints[2].timestamp);
    }

    #[test]
    fn test_get_latest_checkpoint() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = CheckpointManager::new(temp_dir.path(), 5).unwrap();

        let kb = KnowledgeBase::new();

        // Create multiple checkpoints
        for _ in 0..3 {
            let checkpoint = create_checkpoint(&kb, vec![], HashMap::new(), HashMap::new());
            manager.save(&checkpoint).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        let latest = manager.get_latest().unwrap();
        let all_checkpoints = manager.list_checkpoints();

        // Latest should be first in sorted list
        assert_eq!(latest.id, all_checkpoints[0].id);
    }

    #[test]
    fn test_delete_checkpoint() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = CheckpointManager::new(temp_dir.path(), 5).unwrap();

        let kb = KnowledgeBase::new();
        let checkpoint = create_checkpoint(&kb, vec![], HashMap::new(), HashMap::new());

        manager.save(&checkpoint).unwrap();
        assert_eq!(manager.list_checkpoints().len(), 1);

        manager.delete(&checkpoint.id).unwrap();
        assert_eq!(manager.list_checkpoints().len(), 0);
    }

    #[test]
    fn test_checkpoint_eviction() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = CheckpointManager::new(temp_dir.path(), 3).unwrap();

        let kb = KnowledgeBase::new();

        // Create 5 checkpoints (should evict 2)
        for _ in 0..5 {
            let checkpoint = create_checkpoint(&kb, vec![], HashMap::new(), HashMap::new());
            manager.save(&checkpoint).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // Should only have 3 (max_checkpoints)
        assert_eq!(manager.list_checkpoints().len(), 3);
    }

    #[test]
    fn test_kb_restoration() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = CheckpointManager::new(temp_dir.path(), 5).unwrap();

        let mut kb = KnowledgeBase::new();
        kb.add_fact(Fact::new("test_key", "Test Fact", "Test content"))
            .unwrap();

        let checkpoint = create_checkpoint(&kb, vec![], HashMap::new(), HashMap::new());
        manager.save(&checkpoint).unwrap();

        // Load and restore KB
        let loaded = manager.load(&checkpoint.id).unwrap();
        let mut restored_kb = restore_kb_from_checkpoint(&loaded).unwrap();

        assert_eq!(restored_kb.stats().fact_count, 1);
        assert!(restored_kb.get_fact("test_key").is_ok());
    }

    #[test]
    fn test_checkpoint_with_decompositions() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = CheckpointManager::new(temp_dir.path(), 5).unwrap();

        let kb = KnowledgeBase::new();
        let problem = Problem::new("Test problem");

        let mut active_problems = HashMap::new();
        active_problems.insert(problem.id.clone(), problem);

        let checkpoint = create_checkpoint(&kb, vec![], HashMap::new(), active_problems);
        manager.save(&checkpoint).unwrap();

        let loaded = manager.load(&checkpoint.id).unwrap();
        assert_eq!(loaded.active_problems.len(), 1);
    }

    #[test]
    fn test_clear_all_checkpoints() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = CheckpointManager::new(temp_dir.path(), 5).unwrap();

        let kb = KnowledgeBase::new();

        // Create multiple checkpoints
        for _ in 0..3 {
            let checkpoint = create_checkpoint(&kb, vec![], HashMap::new(), HashMap::new());
            manager.save(&checkpoint).unwrap();
        }

        assert_eq!(manager.list_checkpoints().len(), 3);

        manager.clear_all().unwrap();
        assert_eq!(manager.list_checkpoints().len(), 0);
    }

    #[test]
    fn test_branch_creation() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = CheckpointManager::new(temp_dir.path(), 10).unwrap();

        let mut kb = KnowledgeBase::new();
        kb.add_fact(Fact::new("base_fact", "Base Fact", "Base content"))
            .unwrap();

        // Create base checkpoint
        let base_checkpoint = create_checkpoint(&kb, vec![], HashMap::new(), HashMap::new());
        manager.save(&base_checkpoint).unwrap();

        // Create branch
        let branch_id = manager.branch(&base_checkpoint.id, "test_branch").unwrap();

        // Verify branch exists and has correct metadata
        let branch_meta = manager
            .checkpoint_metadata
            .get(&branch_id)
            .expect("Branch should exist");

        assert_eq!(branch_meta.parent_id, Some(base_checkpoint.id.clone()));
        assert_eq!(branch_meta.branch_name, Some("test_branch".to_string()));
        assert_eq!(branch_meta.num_facts, 1); // Should have same facts as parent
    }

    #[test]
    fn test_list_branches() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = CheckpointManager::new(temp_dir.path(), 10).unwrap();

        let kb = KnowledgeBase::new();
        let base_checkpoint = create_checkpoint(&kb, vec![], HashMap::new(), HashMap::new());
        manager.save(&base_checkpoint).unwrap();

        // Create multiple branches
        manager.branch(&base_checkpoint.id, "branch1").unwrap();
        manager.branch(&base_checkpoint.id, "branch2").unwrap();
        manager.branch(&base_checkpoint.id, "branch3").unwrap();

        let branches = manager.list_branches(&base_checkpoint.id);
        assert_eq!(branches.len(), 3);

        // Verify all branches point to the same parent
        for branch in branches {
            assert_eq!(branch.parent_id, Some(base_checkpoint.id.clone()));
        }
    }

    #[test]
    fn test_branch_independence() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = CheckpointManager::new(temp_dir.path(), 10).unwrap();

        let mut kb = KnowledgeBase::new();
        kb.add_fact(Fact::new("base_fact", "Base Fact", "Base content"))
            .unwrap();

        let base_checkpoint = create_checkpoint(&kb, vec![], HashMap::new(), HashMap::new());
        manager.save(&base_checkpoint).unwrap();

        // Create two branches
        let branch1_id = manager.branch(&base_checkpoint.id, "branch1").unwrap();
        let branch2_id = manager.branch(&base_checkpoint.id, "branch2").unwrap();

        // Load and modify branch 1
        let mut branch1 = manager.load(&branch1_id).unwrap();
        let mut kb1 = restore_kb_from_checkpoint(&branch1).unwrap();
        kb1.add_fact(Fact::new(
            "branch1_fact",
            "Branch 1 Fact",
            "Branch 1 content",
        ))
        .unwrap();
        branch1.kb_snapshot.facts = kb1.get_all_facts();
        manager.save(&branch1).unwrap();

        // Load branch 2 (should not have branch1's fact)
        let branch2 = manager.load(&branch2_id).unwrap();
        assert_eq!(branch2.kb_snapshot.facts.len(), 1); // Only base fact
        assert!(
            branch2
                .kb_snapshot
                .facts
                .iter()
                .any(|f| f.key == "base_fact")
        );
        assert!(
            !branch2
                .kb_snapshot
                .facts
                .iter()
                .any(|f| f.key == "branch1_fact")
        );
    }

    #[test]
    fn test_merge_branches() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = CheckpointManager::new(temp_dir.path(), 10).unwrap();

        let kb = KnowledgeBase::new();
        let base_checkpoint = create_checkpoint(&kb, vec![], HashMap::new(), HashMap::new());
        manager.save(&base_checkpoint).unwrap();

        // Create and modify branch 1
        let branch1_id = manager.branch(&base_checkpoint.id, "branch1").unwrap();
        let mut branch1 = manager.load(&branch1_id).unwrap();
        let mut kb1 = restore_kb_from_checkpoint(&branch1).unwrap();
        kb1.add_fact(Fact::new("fact1", "Fact 1", "Content 1"))
            .unwrap();
        branch1.kb_snapshot.facts = kb1.get_all_facts();
        manager.save(&branch1).unwrap();

        // Create and modify branch 2
        let branch2_id = manager.branch(&base_checkpoint.id, "branch2").unwrap();
        let mut branch2 = manager.load(&branch2_id).unwrap();
        let mut kb2 = restore_kb_from_checkpoint(&branch2).unwrap();
        kb2.add_fact(Fact::new("fact2", "Fact 2", "Content 2"))
            .unwrap();
        branch2.kb_snapshot.facts = kb2.get_all_facts();
        manager.save(&branch2).unwrap();

        // Merge branches
        let merged_id = manager
            .merge_branches(&[&branch1_id, &branch2_id], "merged")
            .unwrap();

        // Verify merged checkpoint has facts from both branches
        let merged = manager.load(&merged_id).unwrap();
        assert_eq!(merged.kb_snapshot.facts.len(), 2);
        assert!(merged.kb_snapshot.facts.iter().any(|f| f.key == "fact1"));
        assert!(merged.kb_snapshot.facts.iter().any(|f| f.key == "fact2"));
    }

    #[test]
    fn test_merge_conflict_resolution() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = CheckpointManager::new(temp_dir.path(), 10).unwrap();

        let kb = KnowledgeBase::new();
        let base_checkpoint = create_checkpoint(&kb, vec![], HashMap::new(), HashMap::new());
        manager.save(&base_checkpoint).unwrap();

        // Branch 1: Add fact with confidence 0.8
        let branch1_id = manager.branch(&base_checkpoint.id, "branch1").unwrap();
        let mut branch1 = manager.load(&branch1_id).unwrap();
        let mut kb1 = restore_kb_from_checkpoint(&branch1).unwrap();
        let mut fact1 = Fact::new("shared_key", "Version 1", "Content 1");
        fact1.confidence = 0.8;
        kb1.add_fact(fact1).unwrap();
        branch1.kb_snapshot.facts = kb1.get_all_facts();
        manager.save(&branch1).unwrap();

        // Branch 2: Add fact with same key but confidence 0.9
        let branch2_id = manager.branch(&base_checkpoint.id, "branch2").unwrap();
        let mut branch2 = manager.load(&branch2_id).unwrap();
        let mut kb2 = restore_kb_from_checkpoint(&branch2).unwrap();
        let mut fact2 = Fact::new("shared_key", "Version 2", "Content 2");
        fact2.confidence = 0.9;
        kb2.add_fact(fact2).unwrap();
        branch2.kb_snapshot.facts = kb2.get_all_facts();
        manager.save(&branch2).unwrap();

        // Merge branches
        let merged_id = manager
            .merge_branches(&[&branch1_id, &branch2_id], "merged")
            .unwrap();

        // Verify higher confidence fact wins
        let merged = manager.load(&merged_id).unwrap();
        assert_eq!(merged.kb_snapshot.facts.len(), 1);
        let merged_fact = &merged.kb_snapshot.facts[0];
        assert_eq!(merged_fact.key, "shared_key");
        assert_eq!(merged_fact.summary, "Version 2"); // Higher confidence
        assert_eq!(merged_fact.confidence, 0.9);
    }

    #[test]
    fn test_branch_tree_structure() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = CheckpointManager::new(temp_dir.path(), 20).unwrap();

        let kb = KnowledgeBase::new();

        // Create base checkpoint
        let base = create_checkpoint(&kb, vec![], HashMap::new(), HashMap::new());
        manager.save(&base).unwrap();

        // Create first level branches
        let branch1 = manager.branch(&base.id, "strategy1").unwrap();
        let branch2 = manager.branch(&base.id, "strategy2").unwrap();

        // Create second level branches (branches of branches)
        let branch1_1 = manager.branch(&branch1, "refinement1").unwrap();
        let branch1_2 = manager.branch(&branch1, "refinement2").unwrap();

        // Verify tree structure
        let base_branches = manager.list_branches(&base.id);
        assert_eq!(base_branches.len(), 2);

        let branch1_branches = manager.list_branches(&branch1);
        assert_eq!(branch1_branches.len(), 2);

        let branch2_branches = manager.list_branches(&branch2);
        assert_eq!(branch2_branches.len(), 0);

        // Verify parent relationships
        let branch1_1_meta = manager.checkpoint_metadata.get(&branch1_1).unwrap();
        assert_eq!(branch1_1_meta.parent_id, Some(branch1.clone()));
    }
}
