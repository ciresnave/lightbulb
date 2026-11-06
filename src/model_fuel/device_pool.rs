// SPDX-License-Identifier: MIT OR Apache-2.0
//! Return a dropped model's device memory to the OS (ROADMAP F2, P2).
//!
//! **Stopgap: Fuel should own this.** Fuel's CUDA backend sets the device's default
//! memory pool to keep freed blocks (`release threshold = u64::MAX`,
//! `fuel-cuda-backend device.rs:533`), so dropping a model's runner frees memory to
//! that pool but not to the OS. Measured 2026-11-05: calling the pool's `trim_to(0)`
//! after the runner thread has exited returned the memory to baseline in 6 of 6
//! trials (SmolLM2-135M; see the F2 design note). Fuel has no public call for it, so
//! this module makes the one call through `baracuda-driver`. When Fuel exposes a
//! release call this module should shrink to a forward to it, then go.
//!
//! **This is the only module that may import `baracuda-driver`** (PM rule, 2026-11-05).
//!
//! # The handle and its cost
//!
//! [`init`] creates a CUDA context and takes the device's default pool **once, at device
//! selection, before any model is loaded**, and keeps both for the life of the process.
//! The context costs VRAM, about 113 MiB (the no-hold baseline in the 2026-11-05 probe).
//! It is created early on purpose: right after an eviction the freed memory sits in
//! Fuel's pool, not with the OS, so a context created then could fail on a full card.
//!
//! Without the `fuel-cuda` feature, or on the CPU, every function here is a no-op.

use anyhow::Result;

/// Bytes the pool holds, in use and reserved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolBytes {
    pub used: u64,
    pub reserved: u64,
}

/// What a release did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Released {
    pub before: PoolBytes,
    pub after: PoolBytes,
}

#[cfg(feature = "fuel-cuda")]
mod imp {
    use super::{PoolBytes, Released};
    use anyhow::{Context as _, Result, anyhow};
    use baracuda_driver::mempool::{MemoryPool, default_pool};
    use baracuda_driver::{Context, Device};
    use std::sync::Mutex;

    struct Handle {
        _context: Context,
        pool: MemoryPool,
    }

    static HANDLE: Mutex<Option<Handle>> = Mutex::new(None);

    pub fn init(index: u32) -> Result<()> {
        let mut guard = HANDLE
            .lock()
            .map_err(|_| anyhow!("device pool lock poisoned"))?;
        if guard.is_some() {
            return Ok(());
        }
        let device = Device::get(index).map_err(|e| anyhow!("cuda device {index}: {e:?}"))?;
        let context =
            Context::new(&device).map_err(|e| anyhow!("creating a cuda context: {e:?}"))?;
        let pool = default_pool(&context, &device)
            .map_err(|e| anyhow!("taking the default memory pool: {e:?}"))?;
        *guard = Some(Handle {
            _context: context,
            pool,
        });
        Ok(())
    }

    fn read(pool: &MemoryPool) -> Result<PoolBytes> {
        Ok(PoolBytes {
            used: pool
                .used_bytes()
                .map_err(|e| anyhow!("pool used_bytes: {e:?}"))?,
            reserved: pool
                .reserved_bytes()
                .map_err(|e| anyhow!("pool reserved_bytes: {e:?}"))?,
        })
    }

    pub fn stats() -> Result<Option<PoolBytes>> {
        let guard = HANDLE
            .lock()
            .map_err(|_| anyhow!("device pool lock poisoned"))?;
        guard.as_ref().map(|h| read(&h.pool)).transpose()
    }

    pub fn release_unused() -> Result<Option<Released>> {
        let guard = HANDLE
            .lock()
            .map_err(|_| anyhow!("device pool lock poisoned"))?;
        let Some(h) = guard.as_ref() else {
            return Ok(None);
        };
        let before = read(&h.pool)?;
        h.pool
            .trim_to(0)
            .map_err(|e| anyhow!("pool trim_to(0): {e:?}"))
            .context("releasing unused device memory")?;
        Ok(Some(Released {
            before,
            after: read(&h.pool)?,
        }))
    }
}

#[cfg(not(feature = "fuel-cuda"))]
mod imp {
    use super::{PoolBytes, Released};
    use anyhow::Result;

    pub fn init(_index: u32) -> Result<()> {
        Ok(())
    }
    pub fn stats() -> Result<Option<PoolBytes>> {
        Ok(None)
    }
    pub fn release_unused() -> Result<Option<Released>> {
        Ok(None)
    }
}

/// Create the process-wide handle for CUDA device `index`. Idempotent. Call at device
/// selection, before any model load (see the module doc for why).
pub fn init(index: u32) -> Result<()> {
    imp::init(index)
}

/// The pool's used and reserved bytes, or `None` when there is no handle (CPU build, CPU
/// device, or [`init`] failed).
pub fn stats() -> Result<Option<PoolBytes>> {
    imp::stats()
}

/// Hand the pool's unused reserve back to the OS (`trim_to(0)`). `None` when there is
/// no handle. Frees only blocks nothing is using: a model still resident keeps its
/// weights; what it pays to re-reserve afterwards is NOT yet measured.
pub fn release_unused_device_memory() -> Result<Option<Released>> {
    imp::release_unused()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Without a handle (CPU build, or `init` never called) both queries answer "no
    /// handle" rather than failing, so the unload path can call them unconditionally.
    #[cfg(not(feature = "fuel-cuda"))]
    #[test]
    fn without_a_handle_everything_is_a_quiet_none() {
        assert!(init(0).is_ok());
        assert_eq!(stats().unwrap(), None);
        assert_eq!(release_unused_device_memory().unwrap(), None);
    }
}
