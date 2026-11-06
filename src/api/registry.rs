// SPDX-License-Identifier: MIT OR Apache-2.0
//! The set of models this server serves (ROADMAP F2, phase P1).
//!
//! `AppState` used to hold one model's runner sender, control sender and chat
//! template as three loose fields, wired once at startup. They now live in a
//! [`ModelRegistry`] so that "which model does this request run on" has one
//! place to be answered, which is what running several models needs
//! (`docs/superpowers/specs/2026-10-30-f2-multi-model-design.md`).
//!
//! **P1 changes no behaviour.** The registry holds at most one entry, and
//! [`ModelRegistry::entry`] returns it; the handlers still refuse any request
//! whose `model` is not `config.default_model`, exactly as before. Resolving
//! `request.model` to an entry, loading on demand and evicting are later phases.
//!
//! An entry keeps the three pieces independently optional because the code it
//! replaced did: a test server can have a template and no runner, and a control
//! sender is only ever delivered to a live runner.
//!
//! `eos_monitor` and `cache_stats` are per model by nature but stay in
//! `AppState` for now: the runner takes `cache_stats` at start, so moving it
//! belongs with unload (P2).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, RwLock};

use crate::api::chat_template::ResolvedTemplate;
use crate::engine::model_runner::{ControlSender, InferenceJob};

/// What the server holds for one model.
#[derive(Clone, Default)]
pub struct ModelEntry {
    /// Sender to enqueue inference jobs to the model runner thread, if one was
    /// started.
    pub inference_tx: Option<std::sync::mpsc::SyncSender<InferenceJob>>,
    /// Sender for administrative control messages to that runner (`cache_clear`).
    /// Set under the same condition as `inference_tx`: with no runner nothing
    /// listens on the other end.
    pub cache_control: Option<ControlSender>,
    /// The chat template and the special tokens it renders with, resolved once
    /// at startup. `None` when no tier produced one (the handlers then use the
    /// legacy `role: content` join). Held together in one [`ResolvedTemplate`] so
    /// no caller supplies `bos`/`eos` itself.
    pub chat_template: Option<Arc<ResolvedTemplate>>,
    /// The runner thread's join handle (from `ModelRunner::start_with_handle`), so an
    /// unload can wait for the thread to exit before it returns the model's memory.
    /// Shared and taken once: a `JoinHandle` is not `Clone`.
    pub join: Option<JoinSlot>,
}

/// A runner thread's join handle, shared between clones of its entry.
pub type JoinSlot = Arc<Mutex<Option<std::thread::JoinHandle<()>>>>;

/// Models by name. Cloning shares the same set (an `Arc`), so every request's
/// copy of `AppState` sees a change made after startup.
#[derive(Clone, Default)]
pub struct ModelRegistry {
    entries: Arc<RwLock<BTreeMap<String, ModelEntry>>>,
}

impl ModelRegistry {
    /// A registry holding `entry` under `name`.
    pub fn single(name: impl Into<String>, entry: ModelEntry) -> Self {
        let reg = Self::default();
        reg.set(name, entry);
        reg
    }

    /// Put `entry` under `name`, replacing what was there.
    pub fn set(&self, name: impl Into<String>, entry: ModelEntry) {
        if let Ok(mut g) = self.entries.write() {
            g.insert(name.into(), entry);
        }
    }

    /// Change the entry under `name` in place, creating an empty one first if
    /// there is none.
    pub fn update(&self, name: &str, f: impl FnOnce(&mut ModelEntry)) {
        if let Ok(mut g) = self.entries.write() {
            f(g.entry(name.to_string()).or_default());
        }
    }

    /// The entry for `name`, if there is one.
    pub fn get(&self, name: &str) -> Option<ModelEntry> {
        self.entries.read().ok()?.get(name).cloned()
    }

    /// Remove and return the entry for `name`. Requests that arrive afterwards no
    /// longer find it; requests already running keep the sender clones they took.
    pub fn take(&self, name: &str) -> Option<ModelEntry> {
        self.entries.write().ok()?.remove(name)
    }

    /// The one entry this phase holds, or an empty entry when none was ever set
    /// (no model configured, or its runner failed to start). With several
    /// entries (not yet reachable) it is the first by name.
    pub fn entry(&self) -> ModelEntry {
        self.entries
            .read()
            .ok()
            .and_then(|g| g.values().next().cloned())
            .unwrap_or_default()
    }

    /// The names held, sorted.
    pub fn names(&self) -> Vec<String> {
        self.entries
            .read()
            .map(|g| g.keys().cloned().collect())
            .unwrap_or_default()
    }
}

/// What [`unload_model`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Unloaded {
    /// The runner thread exited within the wait. `false` means a request still holds a
    /// sender, so the thread (and the model's memory) lives until that request ends.
    pub joined: bool,
    /// What returning the memory to the OS did; `None` when there is no device handle
    /// (CPU) or the thread had not exited yet.
    pub released: Option<crate::model_fuel::device_pool::Released>,
}

/// Unload `name` (F2 P2): remove its entry so new requests cannot reach it, drop the
/// entry's senders (the runner thread exits when the last sender is gone), wait up to
/// `wait` for the thread to exit, then hand the device's unused memory back to the OS.
///
/// Blocking: call it from `spawn_blocking`. It does not check for requests in flight;
/// the caller decides who to evict ([`crate::engine::eviction::plan_eviction`], which
/// never picks a busy model). If one is running anyway, its sender clone keeps the
/// thread alive and this returns `joined: false` after `wait`, with nothing released.
pub fn unload_model(
    registry: &ModelRegistry,
    name: &str,
    wait: std::time::Duration,
) -> anyhow::Result<Unloaded> {
    let entry = registry
        .take(name)
        .ok_or_else(|| anyhow::anyhow!("model '{name}' is not loaded"))?;
    let join = entry.join.clone();
    drop(entry); // drops the registry's senders
    let handle = join.and_then(|slot| slot.lock().ok().and_then(|mut g| g.take()));
    let Some(handle) = handle else {
        return Ok(Unloaded {
            joined: false,
            released: None,
        });
    };
    let started = std::time::Instant::now();
    while !handle.is_finished() {
        if started.elapsed() >= wait {
            // The handle is dropped here: the thread detaches and exits when the last
            // sender clone does.
            return Ok(Unloaded {
                joined: false,
                released: None,
            });
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    handle
        .join()
        .map_err(|_| anyhow::anyhow!("the runner thread for '{name}' panicked"))?;
    Ok(Unloaded {
        joined: true,
        released: crate::model_fuel::device_pool::release_unused_device_memory()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry_with_runner() -> ModelEntry {
        ModelEntry {
            inference_tx: Some(std::sync::mpsc::sync_channel(1).0),
            ..ModelEntry::default()
        }
    }

    #[test]
    fn an_empty_registry_yields_an_empty_entry_not_a_panic() {
        let reg = ModelRegistry::default();
        let e = reg.entry();
        assert!(e.inference_tx.is_none() && e.cache_control.is_none());
        assert!(e.chat_template.is_none());
        assert!(reg.names().is_empty());
        assert!(reg.get("anything").is_none());
    }

    #[test]
    fn a_set_entry_is_found_by_name_and_as_the_one_entry() {
        let reg = ModelRegistry::single("tiny.gguf", entry_with_runner());
        assert!(reg.get("tiny.gguf").unwrap().inference_tx.is_some());
        assert!(reg.get("other.gguf").is_none(), "control: another name");
        assert!(reg.entry().inference_tx.is_some());
        assert_eq!(reg.names(), vec!["tiny.gguf".to_string()]);
    }

    #[test]
    fn clones_share_one_set_so_a_later_change_reaches_every_copy() {
        let reg = ModelRegistry::default();
        let copy = reg.clone();
        reg.update("m", |e| e.inference_tx = entry_with_runner().inference_tx);
        assert!(
            copy.entry().inference_tx.is_some(),
            "a clone taken before the change must see it"
        );
    }

    #[test]
    fn update_keeps_the_other_pieces_of_the_entry() {
        let reg = ModelRegistry::single("m", entry_with_runner());
        let (ctrl, _rx) = std::sync::mpsc::channel();
        reg.update("m", |e| e.cache_control = Some(ctrl));
        let e = reg.entry();
        assert!(
            e.inference_tx.is_some(),
            "the runner must survive the update"
        );
        assert!(e.cache_control.is_some());
    }

    /// A stand-in for a runner: exits when every sender is gone, and records that it did.
    fn stub_runner() -> (ModelEntry, std::sync::Arc<std::sync::atomic::AtomicBool>) {
        use std::sync::atomic::{AtomicBool, Ordering};
        let (tx, rx) = std::sync::mpsc::sync_channel::<InferenceJob>(1);
        let exited = Arc::new(AtomicBool::new(false));
        let flag = exited.clone();
        let handle = std::thread::spawn(move || {
            while rx.recv().is_ok() {}
            flag.store(true, Ordering::SeqCst);
        });
        let entry = ModelEntry {
            inference_tx: Some(tx),
            join: Some(Arc::new(Mutex::new(Some(handle)))),
            ..ModelEntry::default()
        };
        (entry, exited)
    }

    const WAIT: std::time::Duration = std::time::Duration::from_secs(5);

    #[test]
    fn unload_removes_the_entry_and_waits_for_the_runner_to_exit() {
        use std::sync::atomic::Ordering;
        let (entry, exited) = stub_runner();
        let reg = ModelRegistry::single("m", entry);
        let out = unload_model(&reg, "m", WAIT).unwrap();
        assert!(out.joined, "{out:?}");
        assert!(
            exited.load(Ordering::SeqCst),
            "joined means the thread had really exited"
        );
        assert!(
            reg.get("m").is_none(),
            "new requests must no longer find it"
        );
        assert!(
            unload_model(&reg, "m", WAIT).is_err(),
            "a second unload is an error"
        );
    }

    #[test]
    fn a_request_holding_a_sender_keeps_the_runner_alive_and_unload_says_so() {
        use std::sync::atomic::Ordering;
        let (entry, exited) = stub_runner();
        let in_flight = entry.inference_tx.clone().unwrap(); // a running request's clone
        let reg = ModelRegistry::single("m", entry);
        let out = unload_model(&reg, "m", std::time::Duration::from_millis(150)).unwrap();
        assert!(!out.joined && out.released.is_none(), "{out:?}");
        assert!(reg.get("m").is_none(), "the entry is gone either way");
        assert!(
            !exited.load(Ordering::SeqCst),
            "control: the thread is still up"
        );
        drop(in_flight);
        let t = std::time::Instant::now();
        while !exited.load(Ordering::SeqCst) && t.elapsed() < WAIT {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(
            exited.load(Ordering::SeqCst),
            "it exits once the request ends"
        );
    }

    #[test]
    fn unloading_a_model_that_is_not_loaded_is_an_error() {
        let reg = ModelRegistry::default();
        assert!(unload_model(&reg, "nope", WAIT).is_err());
    }

    #[test]
    fn an_entry_with_no_runner_thread_unloads_without_a_join() {
        let reg = ModelRegistry::single("m", ModelEntry::default());
        let out = unload_model(&reg, "m", WAIT).unwrap();
        assert!(!out.joined && out.released.is_none());
        assert!(reg.get("m").is_none());
    }
}
