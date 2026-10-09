//! Tiered storage for demoted KV cache segments
//!
//! When segments are evicted from the GPU KV cache, they are demoted through
//! storage tiers rather than deleted:
//!
//! ```text
//! GPU (VRAM)  →  CPU (RAM)  →  Disk (RocksDB/SQLite)
//!   hot            warm           cold
//! ```
//!
//! Each demoted segment maintains:
//! - KV tensors (on CPU for RAM tier, serialized for disk tier)
//! - A KnowledgeBase fact with a text summary (for `[KB:key]` placeholder)
//! - Original position IDs (for re-injection at correct RoPE positions)
//!
//! The model can request retrieval of demoted content via `<RETRIEVE:key>`
//! tokens, which triggers promotion back to GPU.

use crate::cache::tensor_codec;
use crate::engine::knowledge_base::{FactKey, KnowledgeBase};
use candlelight::core::{Device, Tensor};
use std::collections::HashMap;
use std::ops::Range;
use std::path::PathBuf;

use super::cache_span::SpanId;

/// Synchronous disk storage trait for KV cache segments.
///
/// This is intentionally simple and synchronous to avoid async complexity
/// in the inference loop. Implementations can use filesystem, RocksDB, etc.
pub trait DiskStore: Send {
    /// Store bytes under a key.
    fn store(&mut self, key: &str, data: &[u8]) -> Result<(), String>;

    /// Load bytes by key.
    fn load(&self, key: &str) -> Result<Vec<u8>, String>;

    /// Delete a key.
    fn delete(&mut self, key: &str) -> Result<(), String>;
}

/// File-system backed disk store.
///
/// Stores each KV segment as a file in a directory, through `persistant`'s blocking `fs`
/// backend (atomic replace via a sibling scratch directory — see `scratch_dir_for`).
pub struct FileDiskStore {
    store: persistant::blocking::Store,
    /// Held for the store's lifetime: proves exclusive ownership of the scratch directory
    /// (see `acquire_scratch_dir`) and is removed on drop so a later store can sweep again.
    scratch_lock: Option<(PathBuf, std::fs::File)>,
}

/// Name of the lock file inside the scratch directory that proves this `FileDiskStore` is
/// the only thing sweeping/using it right now (see `acquire_scratch_dir`).
const SCRATCH_LOCK_NAME: &str = ".filediskstore.lock";

/// Scratch directory for atomic replace, as a sibling of the *resolved* `base_dir` (never
/// nested inside it, and never beside a symlink rather than its target): `persistant`
/// requires `atomic_write_dir` on the same filesystem as `root` and outside it, and a
/// sibling keeps `base_dir`'s own listing exactly what callers put there (see
/// `file_disk_store_composes_with_the_mover` in `model_fuel::policies`, which asserts
/// `base_dir`'s entry count is zero once every key is deleted). `base_dir` must already be
/// canonicalized (absolute, symlinks resolved) — the caller is responsible for that, since
/// canonicalizing requires the directory to exist — or a relative or `.` root derives a
/// scratch path that is not reliably outside it.
fn scratch_dir_for(base_dir: &std::path::Path) -> PathBuf {
    let name = base_dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "disk-store".to_string());
    base_dir
        .parent()
        .unwrap_or(base_dir)
        .join(format!("{name}.atomic-scratch"))
}

/// Claim exclusive use of `scratch_dir` and sweep any temp file an earlier, now-dead holder
/// left behind (a replace interrupted by a crash or kill — `persistant`'s docs note `fs`
/// does not clean its own scratch directory on a dropped, uncommitted write).
///
/// Sweeping a shared directory blindly can delete another live writer's in-progress
/// replacement or unrelated files it happens to contain. To stay safe without coordinating
/// with anyone else, this creates `SCRATCH_LOCK_NAME` with `create_new` (atomic: only one
/// caller can win it) before touching any other entry, and sweeps only on a fresh win —
/// proof nothing else currently holds this directory. If the lock file already exists
/// (held by a still-running store, or left behind by one that crashed without dropping it),
/// this skips the sweep and the open proceeds without it: a stale temp file surviving one
/// more run is strictly safer than risking someone else's live data.
fn acquire_scratch_dir(scratch_dir: &std::path::Path) -> std::io::Result<std::fs::File> {
    let lock_path = scratch_dir.join(SCRATCH_LOCK_NAME);
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock_path)
    {
        Ok(lock) => {
            if let Ok(entries) = std::fs::read_dir(scratch_dir) {
                for entry in entries.flatten() {
                    if entry.path() != lock_path {
                        let _ = std::fs::remove_file(entry.path());
                    }
                }
            }
            Ok(lock)
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            std::fs::OpenOptions::new().write(true).open(&lock_path)
        }
        Err(e) => Err(e),
    }
}

impl FileDiskStore {
    /// Create a new file-based disk store.
    ///
    /// Creates the directory (and its atomic-replace scratch directory) if they don't
    /// exist, and sweeps any temp file left behind by an interrupted replace — see
    /// `acquire_scratch_dir` for when that sweep is skipped to stay safe.
    pub fn new(base_dir: impl Into<PathBuf>) -> Result<Self, String> {
        let base_dir = base_dir.into();
        std::fs::create_dir_all(&base_dir)
            .map_err(|e| format!("Failed to create disk store dir: {}", e))?;
        // Resolve symlinks and relative components (including a bare ".") before deriving
        // the scratch sibling, so it is always outside the real root directory.
        let base_dir = std::fs::canonicalize(&base_dir)
            .map_err(|e| format!("Failed to resolve disk store dir: {}", e))?;
        let scratch_dir = scratch_dir_for(&base_dir);
        std::fs::create_dir_all(&scratch_dir)
            .map_err(|e| format!("Failed to create disk store scratch dir: {}", e))?;
        let scratch_lock = acquire_scratch_dir(&scratch_dir)
            .map_err(|e| format!("Failed to lock disk store scratch dir: {}", e))?;

        let needs = persistant::Needs::new()
            .with(persistant::Need::Read)
            .with(persistant::Need::Write)
            .with(persistant::Need::Delete)
            .with(persistant::Need::AtomicReplace);
        let store = persistant::blocking::Store::open(
            persistant::Config::Fs {
                root: base_dir,
                atomic_write_dir: Some(scratch_dir.clone()),
            },
            needs,
        )
        .map_err(|e| format!("Failed to open disk store: {}", e))?;
        Ok(Self {
            store,
            scratch_lock: Some((scratch_dir, scratch_lock)),
        })
    }
}

impl Drop for FileDiskStore {
    fn drop(&mut self) {
        if let Some((scratch_dir, lock)) = self.scratch_lock.take() {
            drop(lock);
            let _ = std::fs::remove_file(scratch_dir.join(SCRATCH_LOCK_NAME));
        }
    }
}

impl DiskStore for FileDiskStore {
    fn store(&mut self, key: &str, data: &[u8]) -> Result<(), String> {
        // `persistant::Store::replace` takes ownership of the buffer (it crosses an async
        // await boundary that may outlive this call's stack frame), so this clone is
        // structural at the sync-borrow/async-owned boundary, not avoidable without
        // changing the `DiskStore` trait's `&[u8]` signature.
        self.store
            .replace(key, data.to_vec())
            .map_err(|e| format!("Failed to write {}: {}", key, e))
    }

    fn load(&self, key: &str) -> Result<Vec<u8>, String> {
        self.store
            .read(key)
            .map_err(|e| format!("Failed to read {}: {}", key, e))
    }

    fn delete(&mut self, key: &str) -> Result<(), String> {
        self.store
            .delete(key)
            .map_err(|e| format!("Failed to delete {}: {}", key, e))
    }
}

/// Storage tier for a demoted segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageTier {
    /// KV tensors on GPU — active in the KV cache (not demoted).
    Gpu,
    /// KV tensors on CPU RAM — fast to promote back to GPU.
    Ram,
    /// KV tensors serialized to disk — slower but persistent.
    Disk,
}

/// A segment that has been demoted from the GPU KV cache.
#[derive(Debug)]
pub struct DemotedSegment {
    /// Original span ID from the SpanRegistry.
    pub span_id: SpanId,

    /// Current storage tier.
    pub tier: StorageTier,

    /// Which batch slot this segment belonged to.
    pub slot: usize,

    /// Position range in the original KV cache (for re-injection).
    pub position_range: Range<usize>,

    /// Per-layer (K, V) tensors on CPU. Present only for RAM tier.
    /// Cleared when demoted to disk.
    pub cpu_kv_layers: Option<Vec<(Tensor, Tensor)>>,

    /// Storage backend key. Present only for disk tier.
    pub disk_key: Option<String>,

    /// Associated KnowledgeBase fact key (for `[KB:key]` lookup).
    pub fact_key: Option<FactKey>,

    /// Original token IDs (if available, for re-tokenization fallback).
    pub token_ids: Option<Vec<u32>>,
}

/// Manages tiered storage for demoted KV cache segments.
///
/// Coordinates between:
/// - GPU KV cache (via ParallelKvCache / ParallelCacheBuilder)
/// - CPU RAM storage (DemotedSegment with cpu_kv_layers)
/// - Disk storage (via infra-storage StorageBackend)
/// - KnowledgeBase (text summaries for `[KB:key]` placeholders)
pub struct TieredStorageManager {
    /// All demoted segments, keyed by SpanId.
    demoted: HashMap<SpanId, DemotedSegment>,

    /// Reverse index: FactKey → SpanId (for RETRIEVE lookup).
    fact_key_index: HashMap<FactKey, SpanId>,

    /// Knowledge base for storing eviction summaries.
    knowledge_base: KnowledgeBase,

    /// GPU device for tensor promotion.
    gpu_device: Device,

    /// Maximum number of segments to keep in RAM tier.
    max_ram_segments: usize,

    /// Current count of RAM-tier segments.
    ram_segment_count: usize,

    /// Counter for generating unique disk storage keys.
    disk_key_counter: u64,

    /// Optional disk backend for cold-tier storage.
    disk_store: Option<Box<dyn DiskStore>>,
}

impl TieredStorageManager {
    /// Create a new tiered storage manager.
    ///
    /// # Arguments
    ///
    /// * `gpu_device` - The GPU device for promoting tensors back
    /// * `max_ram_segments` - Maximum segments to keep in RAM before disk demotion
    pub fn new(gpu_device: Device, max_ram_segments: usize) -> Self {
        Self {
            demoted: HashMap::new(),
            fact_key_index: HashMap::new(),
            knowledge_base: KnowledgeBase::new(),
            gpu_device,
            max_ram_segments,
            ram_segment_count: 0,
            disk_key_counter: 0,
            disk_store: None,
        }
    }

    /// Set the disk storage backend.
    ///
    /// When set, segments can be demoted from RAM to disk via `auto_demote_to_disk()`.
    pub fn set_disk_store(&mut self, store: Box<dyn DiskStore>) {
        self.disk_store = Some(store);
    }

    /// Demote a segment from GPU to RAM tier.
    ///
    /// Extracts KV tensors from the GPU cache, moves them to CPU,
    /// and creates a KnowledgeBase fact with the provided summary.
    ///
    /// # Arguments
    ///
    /// * `span_id` - SpanId of the segment being demoted
    /// * `slot` - Batch slot index
    /// * `position_range` - Position range in the KV cache
    /// * `kv_layers` - Per-layer (K, V) tensors extracted from the cache (already on CPU)
    /// * `summary` - Text summary for the KB placeholder
    /// * `full_content` - Full text content of the demoted segment
    /// * `token_ids` - Optional original token IDs
    ///
    /// # Returns
    ///
    /// The FactKey of the created KB entry (for `[KB:key]` placeholder).
    pub fn demote_to_ram(
        &mut self,
        span_id: SpanId,
        slot: usize,
        position_range: Range<usize>,
        kv_layers: Vec<(Tensor, Tensor)>,
        summary: String,
        full_content: String,
        token_ids: Option<Vec<u32>>,
    ) -> Result<FactKey, String> {
        // Create KB fact
        let fact_key = self
            .knowledge_base
            .create_fact_from_eviction(
                summary,
                full_content,
                (position_range.start, position_range.end),
                token_ids.clone(),
            )
            .map_err(|e| format!("Failed to create KB fact: {}", e))?;

        // Build reverse index
        self.fact_key_index.insert(fact_key.clone(), span_id);

        // Store demoted segment
        let segment = DemotedSegment {
            span_id,
            tier: StorageTier::Ram,
            slot,
            position_range,
            cpu_kv_layers: Some(kv_layers),
            disk_key: None,
            fact_key: Some(fact_key.clone()),
            token_ids,
        };

        self.demoted.insert(span_id, segment);
        self.ram_segment_count += 1;

        Ok(fact_key)
    }

    /// Demote a segment from RAM to disk tier.
    ///
    /// Serializes the CPU tensors via the tensor codec and stores them
    /// using the provided storage function. Frees CPU memory.
    ///
    /// # Arguments
    ///
    /// * `span_id` - SpanId of the segment to demote further
    /// * `store_fn` - Function that takes (key, bytes) and persists to disk
    ///
    /// # Returns
    ///
    /// The disk storage key, or error if segment not found or not in RAM tier.
    pub fn demote_to_disk<F>(&mut self, span_id: SpanId, store_fn: F) -> Result<String, String>
    where
        F: FnOnce(&str, &[u8]) -> Result<(), String>,
    {
        let segment = self
            .demoted
            .get_mut(&span_id)
            .ok_or_else(|| format!("Segment {} not found in demoted storage", span_id))?;

        if segment.tier != StorageTier::Ram {
            return Err(format!(
                "Segment {} is in {:?} tier, expected Ram",
                span_id, segment.tier
            ));
        }

        let kv_layers = segment
            .cpu_kv_layers
            .take()
            .ok_or_else(|| format!("Segment {} has no CPU tensors", span_id))?;

        // Serialize
        let bytes = tensor_codec::kv_layers_to_bytes(&kv_layers)
            .map_err(|e| format!("Tensor serialization failed: {}", e))?;

        // Generate disk key
        self.disk_key_counter += 1;
        let disk_key = format!("kv_segment_{}", self.disk_key_counter);

        // Store via provided function
        store_fn(&disk_key, &bytes)?;

        // Update segment
        segment.tier = StorageTier::Disk;
        segment.disk_key = Some(disk_key.clone());
        segment.cpu_kv_layers = None; // Free CPU memory
        self.ram_segment_count -= 1;

        Ok(disk_key)
    }

    /// Promote a segment back to GPU.
    ///
    /// Loads tensors from RAM or disk, moves to GPU device, and returns
    /// the per-layer KV tensors ready for re-injection into the cache.
    ///
    /// # Arguments
    ///
    /// * `span_id` - SpanId of the segment to promote
    /// * `load_fn` - Function that loads bytes from disk by key (only needed for disk tier)
    ///
    /// # Returns
    ///
    /// Tuple of (slot, position_range, per-layer KV tensors on GPU).
    pub fn promote_to_gpu<F>(
        &mut self,
        span_id: SpanId,
        load_fn: F,
    ) -> Result<(usize, Range<usize>, Vec<(Tensor, Tensor)>), String>
    where
        F: FnOnce(&str) -> Result<Vec<u8>, String>,
    {
        let segment = self
            .demoted
            .remove(&span_id)
            .ok_or_else(|| format!("Segment {} not found in demoted storage", span_id))?;

        let slot = segment.slot;
        let position_range = segment.position_range.clone();

        // Get KV layers based on tier
        let cpu_kv_layers = match segment.tier {
            StorageTier::Ram => segment
                .cpu_kv_layers
                .ok_or_else(|| "RAM segment missing CPU tensors".to_string())?,
            StorageTier::Disk => {
                let disk_key = segment
                    .disk_key
                    .as_ref()
                    .ok_or_else(|| "Disk segment missing storage key".to_string())?;
                let bytes = load_fn(disk_key)?;
                tensor_codec::kv_layers_from_bytes(&bytes, &Device::Cpu)
                    .map_err(|e| format!("Tensor deserialization failed: {}", e))?
            }
            StorageTier::Gpu => {
                return Err("Segment already on GPU".to_string());
            }
        };

        // Move to GPU
        let mut gpu_kv_layers = Vec::with_capacity(cpu_kv_layers.len());
        for (k, v) in cpu_kv_layers {
            let k_gpu = k
                .to_device(&self.gpu_device)
                .map_err(|e| format!("Failed to move K to GPU: {}", e))?;
            let v_gpu = v
                .to_device(&self.gpu_device)
                .map_err(|e| format!("Failed to move V to GPU: {}", e))?;
            gpu_kv_layers.push((k_gpu, v_gpu));
        }

        // Clean up indices
        if let Some(ref fact_key) = segment.fact_key {
            self.fact_key_index.remove(fact_key);
        }

        if segment.tier == StorageTier::Ram {
            self.ram_segment_count -= 1;
        }

        Ok((slot, position_range, gpu_kv_layers))
    }

    /// Find a demoted segment by its KnowledgeBase fact key.
    ///
    /// Used when the model generates `<RETRIEVE:key>` to find
    /// the corresponding demoted segment for promotion.
    pub fn find_by_fact_key(&self, key: &str) -> Option<SpanId> {
        self.fact_key_index.get(key).copied()
    }

    /// Get a reference to a demoted segment.
    pub fn get_demoted(&self, span_id: SpanId) -> Option<&DemotedSegment> {
        self.demoted.get(&span_id)
    }

    /// Get a reference to the knowledge base.
    pub fn knowledge_base(&self) -> &KnowledgeBase {
        &self.knowledge_base
    }

    /// Get a mutable reference to the knowledge base.
    pub fn knowledge_base_mut(&mut self) -> &mut KnowledgeBase {
        &mut self.knowledge_base
    }

    /// Check if RAM tier is at capacity.
    pub fn is_ram_full(&self) -> bool {
        self.ram_segment_count >= self.max_ram_segments
    }

    /// Auto-demote the oldest RAM segment to disk if RAM is full and disk is available.
    ///
    /// Returns the span_id that was demoted, or None if no demotion needed/possible.
    pub fn auto_demote_oldest_to_disk(&mut self) -> Result<Option<SpanId>, String> {
        if !self.is_ram_full() {
            return Ok(None);
        }

        if self.disk_store.is_none() {
            return Err("RAM full but no disk store configured".to_string());
        }

        // Find the oldest RAM-tier segment (by span_id, which approximates creation order)
        let oldest_ram = self
            .demoted
            .iter()
            .filter(|(_, seg)| seg.tier == StorageTier::Ram)
            .min_by_key(|(id, _)| **id)
            .map(|(&id, _)| id);

        if let Some(span_id) = oldest_ram {
            // Use the stored disk_store
            let segment = self
                .demoted
                .get_mut(&span_id)
                .ok_or_else(|| format!("Segment {} disappeared", span_id))?;

            if segment.tier != StorageTier::Ram {
                return Err(format!("Segment {} is not in RAM tier", span_id));
            }

            let kv_layers = segment
                .cpu_kv_layers
                .take()
                .ok_or_else(|| format!("Segment {} has no CPU tensors", span_id))?;

            let bytes = tensor_codec::kv_layers_to_bytes(&kv_layers)
                .map_err(|e| format!("Serialization failed: {}", e))?;

            self.disk_key_counter += 1;
            let disk_key = format!("kv_segment_{}", self.disk_key_counter);

            self.disk_store.as_mut().unwrap().store(&disk_key, &bytes)?;

            segment.tier = StorageTier::Disk;
            segment.disk_key = Some(disk_key);
            segment.cpu_kv_layers = None;
            self.ram_segment_count -= 1;

            Ok(Some(span_id))
        } else {
            Ok(None)
        }
    }

    /// Promote a segment using the stored disk backend (convenience wrapper).
    ///
    /// Handles both RAM and Disk tiers transparently.
    pub fn promote(
        &mut self,
        span_id: SpanId,
    ) -> Result<(usize, Range<usize>, Vec<(Tensor, Tensor)>), String> {
        self.promote_to_gpu(span_id, |disk_key| {
            // This closure can't borrow self.disk_store because self is already borrowed.
            // We handle this by extracting disk data before calling promote_to_gpu.
            Err(format!(
                "Use promote_with_disk() for disk-tier segments (key: {})",
                disk_key
            ))
        })
    }

    /// Promote a segment, loading from disk store if needed.
    ///
    /// This method handles the borrow-checker constraint by pre-loading
    /// disk data before the promotion call.
    pub fn promote_with_disk(
        &mut self,
        span_id: SpanId,
    ) -> Result<(usize, Range<usize>, Vec<(Tensor, Tensor)>), String> {
        // Check if we need to load from disk first
        let needs_disk_load = self
            .demoted
            .get(&span_id)
            .map(|seg| seg.tier == StorageTier::Disk)
            .unwrap_or(false);

        if needs_disk_load {
            // Pre-load from disk into CPU tensors
            let disk_key = self
                .demoted
                .get(&span_id)
                .and_then(|seg| seg.disk_key.clone())
                .ok_or_else(|| "Disk segment missing key".to_string())?;

            let bytes = self
                .disk_store
                .as_ref()
                .ok_or_else(|| "No disk store configured".to_string())?
                .load(&disk_key)?;

            let cpu_kv = tensor_codec::kv_layers_from_bytes(&bytes, &Device::Cpu)
                .map_err(|e| format!("Deserialization failed: {}", e))?;

            // Update segment to RAM tier with loaded data
            if let Some(segment) = self.demoted.get_mut(&span_id) {
                segment.cpu_kv_layers = Some(cpu_kv);
                segment.tier = StorageTier::Ram;
                self.ram_segment_count += 1;
            }

            // Clean up disk file
            if let Some(ref mut store) = self.disk_store {
                let _ = store.delete(&disk_key);
            }
        }

        // Now promote from RAM (standard path)
        self.promote_to_gpu(span_id, |_| {
            Err("Should not reach disk load — data was pre-loaded".to_string())
        })
    }

    /// Get the number of segments in each tier.
    pub fn tier_counts(&self) -> (usize, usize) {
        let ram = self
            .demoted
            .values()
            .filter(|s| s.tier == StorageTier::Ram)
            .count();
        let disk = self
            .demoted
            .values()
            .filter(|s| s.tier == StorageTier::Disk)
            .count();
        (ram, disk)
    }

    /// Total number of demoted segments.
    pub fn demoted_count(&self) -> usize {
        self.demoted.len()
    }

    /// Get the GPU device.
    pub fn gpu_device(&self) -> &Device {
        &self.gpu_device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_kv_layers(num_layers: usize) -> Vec<(Tensor, Tensor)> {
        let mut layers = Vec::new();
        for i in 0..num_layers {
            let k = Tensor::from_vec(
                vec![(i as f32) * 10.0 + 1.0, (i as f32) * 10.0 + 2.0],
                &[1, 1, 2],
                &Device::Cpu,
            )
            .unwrap();
            let v = Tensor::from_vec(
                vec![(i as f32) * 10.0 + 3.0, (i as f32) * 10.0 + 4.0],
                &[1, 1, 2],
                &Device::Cpu,
            )
            .unwrap();
            layers.push((k, v));
        }
        layers
    }

    #[test]
    fn test_demote_to_ram() {
        let mut manager = TieredStorageManager::new(Device::Cpu, 64);

        let kv = make_test_kv_layers(2);
        let result = manager.demote_to_ram(
            1,
            0,
            10..30,
            kv,
            "Test summary".to_string(),
            "Full content here".to_string(),
            Some(vec![100, 200, 300]),
        );

        assert!(result.is_ok());
        let fact_key = result.unwrap();

        // Verify segment is stored
        let segment = manager.get_demoted(1).unwrap();
        assert_eq!(segment.tier, StorageTier::Ram);
        assert_eq!(segment.slot, 0);
        assert_eq!(segment.position_range, 10..30);
        assert!(segment.cpu_kv_layers.is_some());
        assert_eq!(segment.cpu_kv_layers.as_ref().unwrap().len(), 2);

        // Verify KB fact
        assert!(manager.find_by_fact_key(&fact_key).is_some());
        assert_eq!(manager.find_by_fact_key(&fact_key).unwrap(), 1);

        // Verify tier counts
        assert_eq!(manager.tier_counts(), (1, 0));
    }

    #[test]
    fn test_demote_to_disk() {
        let mut manager = TieredStorageManager::new(Device::Cpu, 64);

        let kv = make_test_kv_layers(2);
        manager
            .demote_to_ram(
                1,
                0,
                10..30,
                kv,
                "Summary".to_string(),
                "Content".to_string(),
                None,
            )
            .unwrap();

        // Simulate disk storage with in-memory HashMap
        let mut disk: HashMap<String, Vec<u8>> = HashMap::new();
        let result = manager.demote_to_disk(1, |key, bytes| {
            disk.insert(key.to_string(), bytes.to_vec());
            Ok(())
        });

        assert!(result.is_ok());
        let disk_key = result.unwrap();

        // Verify segment updated
        let segment = manager.get_demoted(1).unwrap();
        assert_eq!(segment.tier, StorageTier::Disk);
        assert!(segment.cpu_kv_layers.is_none()); // CPU memory freed
        assert_eq!(segment.disk_key.as_ref().unwrap(), &disk_key);

        // Verify disk storage has data
        assert!(disk.contains_key(&disk_key));
        assert!(!disk[&disk_key].is_empty());

        // Verify tier counts
        assert_eq!(manager.tier_counts(), (0, 1));
    }

    #[test]
    fn test_promote_from_ram() {
        let mut manager = TieredStorageManager::new(Device::Cpu, 64);

        let kv = make_test_kv_layers(2);
        let fact_key = manager
            .demote_to_ram(
                1,
                0,
                10..30,
                kv,
                "Summary".to_string(),
                "Content".to_string(),
                None,
            )
            .unwrap();

        // Promote back
        let result =
            manager.promote_to_gpu(1, |_key| panic!("Should not load from disk for RAM tier"));

        assert!(result.is_ok());
        let (slot, range, layers) = result.unwrap();

        assert_eq!(slot, 0);
        assert_eq!(range, 10..30);
        assert_eq!(layers.len(), 2);

        // Verify cleanup
        assert!(manager.get_demoted(1).is_none());
        assert!(manager.find_by_fact_key(&fact_key).is_none());
        assert_eq!(manager.demoted_count(), 0);
    }

    #[test]
    fn test_promote_from_disk() {
        let mut manager = TieredStorageManager::new(Device::Cpu, 64);

        let kv = make_test_kv_layers(2);
        manager
            .demote_to_ram(
                1,
                0,
                10..30,
                kv,
                "Summary".to_string(),
                "Content".to_string(),
                None,
            )
            .unwrap();

        // Demote to disk
        let mut disk: HashMap<String, Vec<u8>> = HashMap::new();
        let disk_key = manager
            .demote_to_disk(1, |key, bytes| {
                disk.insert(key.to_string(), bytes.to_vec());
                Ok(())
            })
            .unwrap();

        // Promote from disk
        let result = manager.promote_to_gpu(1, |key| {
            disk.get(key)
                .cloned()
                .ok_or_else(|| format!("Key {} not found on disk", key))
        });

        assert!(result.is_ok());
        let (slot, range, layers) = result.unwrap();

        assert_eq!(slot, 0);
        assert_eq!(range, 10..30);
        assert_eq!(layers.len(), 2);

        // Verify tensor data survived round-trip
        let k0_data: Vec<f32> = layers[0].0.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(k0_data, vec![1.0, 2.0]);
    }

    #[test]
    fn test_ram_capacity_tracking() {
        let mut manager = TieredStorageManager::new(Device::Cpu, 2);

        for i in 1..=2 {
            let kv = make_test_kv_layers(1);
            manager
                .demote_to_ram(
                    i,
                    0,
                    (i as usize * 10)..(i as usize * 10 + 10),
                    kv,
                    format!("Summary {}", i),
                    format!("Content {}", i),
                    None,
                )
                .unwrap();
        }

        assert!(manager.is_ram_full());
        assert_eq!(manager.tier_counts(), (2, 0));
    }

    #[test]
    fn test_find_by_nonexistent_key() {
        let manager = TieredStorageManager::new(Device::Cpu, 64);
        assert!(manager.find_by_fact_key("nonexistent").is_none());
    }

    #[test]
    fn test_demote_nonexistent_to_disk_fails() {
        let mut manager = TieredStorageManager::new(Device::Cpu, 64);
        let result = manager.demote_to_disk(999, |_, _| Ok(()));
        assert!(result.is_err());
    }

    #[test]
    fn test_promote_nonexistent_fails() {
        let mut manager = TieredStorageManager::new(Device::Cpu, 64);
        let result = manager.promote_to_gpu(999, |_| Ok(vec![]));
        assert!(result.is_err());
    }

    #[test]
    fn test_file_disk_store_delete_of_missing_key_is_ok() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = FileDiskStore::new(dir.path()).unwrap();
        // Old `std::fs`-backed behaviour: deleting a key that was never stored
        // is not an error (persistant documents the same contract on `delete`).
        assert!(store.delete("never-stored").is_ok());
    }

    #[test]
    fn test_file_disk_store_reads_file_written_by_old_plain_fs_writer() {
        let dir = tempfile::tempdir().unwrap();
        // Simulate a pre-existing base_dir populated by the old writer, which
        // wrote each key as a plain file via `std::fs::write` with no sibling
        // scratch directory alongside it.
        std::fs::write(dir.path().join("kv_segment_1"), b"old-writer-bytes").unwrap();

        let store = FileDiskStore::new(dir.path()).unwrap();
        assert_eq!(store.load("kv_segment_1").unwrap(), b"old-writer-bytes");
    }

    #[test]
    fn test_file_disk_store_does_not_sweep_a_scratch_dir_another_store_still_holds() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileDiskStore::new(dir.path()).unwrap();
        let (scratch_dir, _lock) = store.scratch_lock.as_ref().unwrap();

        // A file an in-progress (still-held) replace would have left behind.
        let decoy = scratch_dir.join("decoy-in-progress-write");
        std::fs::write(&decoy, b"not abandoned").unwrap();

        // Opening a second store over the same base_dir finds the lock already held
        // (the first `store` hasn't been dropped) and must not sweep: the decoy survives.
        let _second = FileDiskStore::new(dir.path()).unwrap();
        assert!(
            decoy.exists(),
            "live store's scratch dir was swept out from under it"
        );
    }

    #[test]
    fn test_file_disk_store_sweeps_scratch_dir_once_the_holder_is_gone() {
        let dir = tempfile::tempdir().unwrap();
        let scratch_dir;
        {
            let store = FileDiskStore::new(dir.path()).unwrap();
            scratch_dir = store.scratch_lock.as_ref().unwrap().0.clone();
            std::fs::write(
                scratch_dir.join("abandoned-temp-file"),
                b"crashed mid-write",
            )
            .unwrap();
        } // store dropped: releases the lock, as a crash would not.

        // A fresh store over the same base_dir now sweeps the directory it is first to lock.
        let _reopened = FileDiskStore::new(dir.path()).unwrap();
        assert!(!scratch_dir.join("abandoned-temp-file").exists());
    }

    #[test]
    fn test_scratch_dir_for_a_bare_dot_would_nest_without_resolving_first() {
        // `scratch_dir_for` is purely lexical (`Path::parent`/`file_name`, no filesystem
        // access): fed a literal "." it reads as `file_name() == None` and
        // `parent() == Some("")`, landing the "sibling" back inside "." itself. This
        // characterizes that lexical edge case — it is why `FileDiskStore::new`
        // canonicalizes (resolving "." against the real cwd to an absolute path with a
        // real parent) *before* ever calling `scratch_dir_for`, rather than a claim that
        // `scratch_dir_for` handles "." correctly on its own.
        let scratch = scratch_dir_for(std::path::Path::new("."));
        assert_eq!(
            scratch,
            std::path::PathBuf::from("disk-store.atomic-scratch")
        );
    }

    #[test]
    fn test_file_disk_store_scratch_dir_resolves_a_symlinked_base_dir_to_its_target() {
        let outer = tempfile::tempdir().unwrap();
        let target = outer.path().join("real-store-root");
        std::fs::create_dir_all(&target).unwrap();
        let link = outer.path().join("store-root-symlink");
        #[cfg(windows)]
        let symlinked = std::os::windows::fs::symlink_dir(&target, &link).is_ok();
        #[cfg(not(windows))]
        let symlinked = std::os::unix::fs::symlink(&target, &link).is_ok();
        if !symlinked {
            // No symlink privilege in this environment (e.g. Windows without Developer
            // Mode or admin) — nothing to verify here, not a failure of the code under
            // test.
            return;
        }

        // Before canonicalizing, the naive (lexical) sibling derivation would land the
        // scratch dir beside the *symlink's* own location, not the real target directory
        // it resolves to — which may be a different directory, or a different filesystem
        // entirely (the hazard `scratch_dir_for`'s doc comment names).
        let store = FileDiskStore::new(&link).unwrap();
        let (scratch_dir, _lock) = store.scratch_lock.as_ref().unwrap();
        let canonical_target = std::fs::canonicalize(&target).unwrap();
        assert_eq!(
            scratch_dir,
            &canonical_target
                .parent()
                .unwrap()
                .join("real-store-root.atomic-scratch")
        );
    }
}
