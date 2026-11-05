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
use std::sync::{Arc, RwLock};

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
}

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
}
