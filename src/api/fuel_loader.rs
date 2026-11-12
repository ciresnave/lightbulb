// SPDX-License-Identifier: MIT OR Apache-2.0
//! The real [`ModelLoader`]: models live in `models_dir`, one entry per name
//! (ROADMAP F2, P2b-2). Also the one place that starts a runner and registers it, shared
//! with the server's startup so the default model and an on-demand one are started the
//! same way.

use super::chat_template;
use super::manager::{LoadError, ModelLoader};
use super::registry::{ModelEntry, ModelRegistry};
use crate::engine::ModelRunner;
use crate::engine::model_runner::{EngineOptions, SharedCacheStats, SharedScheduler};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Everything a runner start needs besides the model itself.
#[derive(Clone)]
pub struct StartParams {
    pub max_batch_size: usize,
    pub context_length: usize,
    pub max_queue_depth: usize,
    pub options: EngineOptions,
    pub cache_stats: SharedCacheStats,
    pub scheduler: SharedScheduler,
}

impl StartParams {
    pub fn from_state(state: &super::AppState) -> Self {
        Self {
            max_batch_size: state.config.model_max_batch_size,
            context_length: state.config.model_context_length,
            max_queue_depth: state.config.model_max_queue_depth,
            options: EngineOptions {
                prefix_sharing: state.config.model_prefix_sharing,
                allow_oversize_model: state.config.model_allow_oversize,
            },
            cache_stats: state.cache_stats.clone(),
            scheduler: state.scheduler.clone(),
        }
    }
}

/// `name` as a path under `models_dir`, if it names exactly one entry there.
///
/// `name` comes from a request body, so it must not be able to leave `models_dir`: a
/// single plain path component only (no separators, no `..`, no drive or root prefix).
pub fn resolve_in(models_dir: &Path, name: &str) -> Option<PathBuf> {
    let mut parts = Path::new(name).components();
    let (Some(Component::Normal(only)), None) = (parts.next(), parts.next()) else {
        return None;
    };
    if only.to_string_lossy().starts_with('.') {
        return None;
    }
    let path = models_dir.join(only);
    path.exists().then_some(path)
}

/// Start the runner for `path`, resolve its chat template, and register both as `name`.
pub fn start_into(
    registry: &ModelRegistry,
    name: &str,
    path: &Path,
    p: &StartParams,
) -> anyhow::Result<()> {
    let (sender, ctrl_sender, join) = ModelRunner::start_with_handle(
        path,
        p.max_batch_size,
        p.context_length,
        Some("f32".to_string()),
        p.max_queue_depth,
        p.cache_stats.clone(),
        p.scheduler.clone(),
        p.options,
    )?;
    // Resolved here and only here: both halves read files, and a per-request read would
    // put filesystem I/O in the request path for a value that cannot change while the
    // process runs. `resolve_for_serving` owns the "keep it only if a tier produced one"
    // rule.
    let template = chat_template::resolve_for_serving(path);
    if let Some(t) = &template {
        tracing::info!(
            "Chat template for {} resolved via {:?} (bos {:?}, eos {:?})",
            path.display(),
            t.resolved_by(),
            t.tokens.bos,
            t.tokens.eos
        );
    }
    registry.set(
        name,
        ModelEntry {
            inference_tx: Some(sender),
            cache_control: Some(ctrl_sender),
            chat_template: template,
            join: Some(Arc::new(Mutex::new(Some(join)))),
        },
    );
    tracing::info!("Started model runner for {}", path.display());
    Ok(())
}

/// Loads models from a directory.
pub struct FuelLoader {
    models_dir: PathBuf,
    params: StartParams,
}

impl FuelLoader {
    pub fn new(models_dir: impl Into<PathBuf>, params: StartParams) -> Self {
        Self {
            models_dir: models_dir.into(),
            params,
        }
    }

    fn path(&self, name: &str) -> Result<PathBuf, LoadError> {
        resolve_in(&self.models_dir, name).ok_or_else(|| LoadError::NotFound(name.to_string()))
    }
}

impl ModelLoader for FuelLoader {
    fn needed_bytes(&self, name: &str) -> Result<u64, LoadError> {
        let path = self.path(name)?;
        // 0 when the file's shape is unknown: the runner's own preflight still decides.
        Ok(crate::model_fuel::preflight::model_load_bytes(
            &path,
            self.params.max_batch_size,
            self.params.context_length,
        )
        .unwrap_or(0))
    }

    fn load(&self, registry: &ModelRegistry, name: &str) -> Result<(), LoadError> {
        let path = self.path(name)?;
        start_into(registry, name, &path, &self.params)
            .map_err(|e| LoadError::Failed(format!("starting '{name}': {e:#}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir_with(files: &[&str]) -> tempfile::TempDir {
        let d = tempfile::tempdir().unwrap();
        for f in files {
            std::fs::write(d.path().join(f), b"x").unwrap();
        }
        d
    }

    #[test]
    fn a_plain_name_in_the_directory_resolves() {
        let d = dir_with(&["m.gguf"]);
        assert_eq!(
            resolve_in(d.path(), "m.gguf"),
            Some(d.path().join("m.gguf"))
        );
    }

    #[test]
    fn a_missing_name_does_not_resolve() {
        let d = dir_with(&["m.gguf"]);
        assert_eq!(resolve_in(d.path(), "other.gguf"), None);
    }

    #[test]
    fn a_name_cannot_leave_the_directory() {
        let d = dir_with(&["m.gguf"]);
        let outside = d.path().parent().unwrap().join("secret.gguf");
        std::fs::write(&outside, b"x").unwrap();
        // Control: the target exists, so only the name check can refuse it.
        assert!(outside.exists());
        for bad in [
            "../secret.gguf",
            "sub/../m.gguf",
            "./m.gguf",
            "/etc/passwd",
            "C:\\Windows\\win.ini",
            "..\\secret.gguf",
            "",
            ".",
            "..",
            ".hidden",
        ] {
            assert_eq!(resolve_in(d.path(), bad), None, "{bad:?} must not resolve");
        }
        std::fs::remove_file(outside).ok();
    }
}
