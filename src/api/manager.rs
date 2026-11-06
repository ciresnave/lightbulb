// SPDX-License-Identifier: MIT OR Apache-2.0
//! Load a model on demand, evicting idle ones to make room (ROADMAP F2, P2b).
//!
//! [`ModelManager::acquire`] answers "give me model `name`, loaded": a loaded model
//! returns at once with a [`Lease`] (its in-flight count, dropped when the request ends);
//! an unloaded one is loaded after [`plan_eviction`] picks which idle models to unload.
//!
//! The 2026-11-06 two-model measurements (design note, "P2a measurements") are why
//! eviction **re-reads free device memory after every unload** instead of trusting the
//! sizes: unloading TinyLlama next to a resident Qwen3 freed ~300 MiB, not ~1.1 GB
//! (issue #298). Evicting one model at a time and asking the device again is the only
//! rule that stays correct when an unload frees less than the estimate.
//!
//! Device access is behind [`ModelLoader`] and a free-memory closure, so the logic here
//! is tested without a GPU. Not yet called by a request handler.

use super::registry::{ModelRegistry, unload_model};
use crate::engine::eviction::{CannotMakeRoom, Plan, Resident, plan_eviction};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Why a model could not be made available.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadError {
    /// No such model on disk (the caller answers 404).
    NotFound(String),
    /// The idle models together do not free enough (the caller answers 503).
    NoRoom(CannotMakeRoom),
    /// The loader failed (the caller answers 500).
    Failed(String),
}

/// Knows where models live and how to start one.
pub trait ModelLoader: Send + Sync {
    /// Device bytes loading `name` needs (the B9 estimate), or `NotFound`.
    fn needed_bytes(&self, name: &str) -> Result<u64, LoadError>;
    /// Start `name` and register its entry in `registry`.
    fn load(&self, registry: &ModelRegistry, name: &str) -> Result<(), LoadError>;
}

/// Free device bytes right now. Called again after every unload.
pub type FreeBytes = Arc<dyn Fn() -> u64 + Send + Sync>;

#[derive(Default)]
struct Book {
    tick: u64,
    models: BTreeMap<String, Tracked>,
}

struct Tracked {
    bytes: u64,
    last_used: u64,
    in_flight: usize,
}

impl Book {
    /// Count a request on `name` if it is loaded; `None` if it is not (or is being evicted).
    fn lease(&mut self, name: &str) -> Option<()> {
        self.tick += 1;
        let tick = self.tick;
        let t = self.models.get_mut(name)?;
        t.last_used = tick;
        t.in_flight += 1;
        Some(())
    }

    fn residents(&self) -> Vec<Resident> {
        self.models
            .iter()
            .map(|(name, t)| Resident {
                name: name.clone(),
                bytes: t.bytes,
                last_used: t.last_used,
                in_flight: t.in_flight,
            })
            .collect()
    }

    /// Stop tracking `name` if nothing is running on it, so no new request can lease it.
    fn claim_for_eviction(&mut self, name: &str) -> bool {
        match self.models.get(name) {
            Some(t) if t.in_flight == 0 => self.models.remove(name).is_some(),
            _ => false,
        }
    }
}

/// One request's hold on a model. While it lives the model is never evicted.
pub struct Lease {
    book: Arc<Mutex<Book>>,
    name: String,
}

impl Lease {
    pub fn model(&self) -> &str {
        &self.name
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        if let Ok(mut b) = self.book.lock()
            && let Some(t) = b.models.get_mut(&self.name)
        {
            t.in_flight = t.in_flight.saturating_sub(1);
        }
    }
}

/// Loads models on demand and evicts idle ones, one at a time, to make room.
pub struct ModelManager {
    registry: ModelRegistry,
    loader: Arc<dyn ModelLoader>,
    free: FreeBytes,
    unload_wait: Duration,
    book: Arc<Mutex<Book>>,
    /// One load or eviction at a time; the fast path for a loaded model does not take it.
    gate: Mutex<()>,
}

impl ModelManager {
    pub fn new(
        registry: ModelRegistry,
        loader: Arc<dyn ModelLoader>,
        free: FreeBytes,
        unload_wait: Duration,
    ) -> Self {
        Self {
            registry,
            loader,
            free,
            unload_wait,
            book: Arc::default(),
            gate: Mutex::new(()),
        }
    }

    /// Track a model that was loaded outside the manager (the startup default).
    pub fn adopt(&self, name: &str, bytes: u64) {
        if let Ok(mut b) = self.book.lock() {
            b.tick += 1;
            let last_used = b.tick;
            b.models.insert(
                name.to_string(),
                Tracked {
                    bytes,
                    last_used,
                    in_flight: 0,
                },
            );
        }
    }

    fn lease_if_loaded(&self, name: &str) -> Option<Lease> {
        let mut b = self.book.lock().ok()?;
        b.lease(name)?;
        Some(Lease {
            book: self.book.clone(),
            name: name.to_string(),
        })
    }

    /// The model `name`, loaded, with a lease on it.
    pub fn acquire(&self, name: &str) -> Result<Lease, LoadError> {
        if let Some(lease) = self.lease_if_loaded(name) {
            return Ok(lease);
        }
        let _gate = self.gate.lock().map_err(poisoned)?;
        // Another request may have loaded it while this one waited for the gate.
        if let Some(lease) = self.lease_if_loaded(name) {
            return Ok(lease);
        }
        let needed = self.loader.needed_bytes(name)?;
        self.make_room(needed)?;
        tracing::info!("loading model '{name}' on demand (needs {needed} bytes)");
        self.loader.load(&self.registry, name)?;
        self.adopt(name, needed);
        self.lease_if_loaded(name)
            .ok_or_else(|| LoadError::Failed(format!("model '{name}' vanished after loading")))
    }

    /// Evict idle models, least recently used first and one at a time, until `needed`
    /// bytes are free. Free memory is re-read after each unload.
    fn make_room(&self, needed: u64) -> Result<(), LoadError> {
        loop {
            let residents = self.book.lock().map_err(poisoned)?.residents();
            match plan_eviction(&residents, needed, (self.free)()) {
                Ok(Plan::Fits) => return Ok(()),
                Err(short) => return Err(LoadError::NoRoom(short)),
                Ok(Plan::Evict(order)) => self.evict_first(&order)?,
            }
        }
    }

    /// Unload the first of `order` that is still idle. If every one has become busy
    /// since the plan was made, return and let the caller plan again.
    fn evict_first(&self, order: &[String]) -> Result<(), LoadError> {
        for name in order {
            let claimed = self.book.lock().map_err(poisoned)?.claim_for_eviction(name);
            if claimed {
                let out = unload_model(&self.registry, name, self.unload_wait)
                    .map_err(|e| LoadError::Failed(format!("unloading '{name}': {e}")))?;
                tracing::info!(
                    "evicted model '{name}' (runner joined: {}, device memory returned: {})",
                    out.joined,
                    out.released.is_some()
                );
                return Ok(());
            }
        }
        Ok(())
    }
}

fn poisoned<T>(_: std::sync::PoisonError<T>) -> LoadError {
    LoadError::Failed("model manager lock poisoned".into())
}

#[cfg(test)]
mod tests;
