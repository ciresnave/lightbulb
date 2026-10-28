// SPDX-License-Identifier: MIT OR Apache-2.0
//! Synchronous disk storage for the tiered KV cache demotion path.
//!
//! `TieredStorageManager`/`DemotedSegment`/`StorageTier` (the actual
//! Tensor-typed GPU->RAM->disk demotion logic) removed 2026-10-22
//! (candlelight removal, CireSnave: "Remove all dependencies on Candle and
//! Candlelight..."): zero real external callers (model_fuel/policies.rs
//! only referenced them in its own design-notes doc comment). `DiskStore`/
//! `FileDiskStore` below are candlelight-free and survive -- they are a
//! real dependency of model_fuel/policies.rs. See
//! docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md.

use std::path::PathBuf;

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
