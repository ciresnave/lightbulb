// SPDX-License-Identifier: MIT OR Apache-2.0
//! Embedding models on disk and in the registry (ROADMAP F3).
//!
//! A chat model is a `.gguf` FILE in the models directory; an embedding model is a DIRECTORY of Hugging Face
//! files (`model.safetensors`, `tokenizer.json`, `config.json`), because Fuel loads encoders from safetensors
//! (there is no GGUF encoder path, `docs/F3-EMBEDDINGS-DESIGN.md`). Only `nomic_bert` is supported (the
//! pooling-table row exists for nomic-embed-text-v1.5); a directory of any other architecture is not listed
//! and not loaded, rather than served with a guessed pooling.

use std::path::Path;
use std::sync::Arc;

use super::openai::embeddings::LoadedEmbedder;
use super::openai::nomic_embedder::NomicEmbedder;
use super::registry::{ModelEntry, ModelRegistry};

/// Whether `path` is a directory this server can serve embeddings from.
pub fn is_embedding_dir(path: &Path) -> bool {
    path.join("model.safetensors").is_file()
        && path.join("tokenizer.json").is_file()
        && std::fs::read(path.join("config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .and_then(|v| {
                v.get("model_type")
                    .and_then(|t| t.as_str().map(str::to_owned))
            })
            .is_some_and(|t| t == "nomic_bert")
}

/// The servable embedding model directories directly under `dir`, sorted. Hidden entries are skipped (the
/// loader refuses a name that starts with '.').
pub fn embedding_dir_names(dir: &str) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| !n.starts_with('.') && is_embedding_dir(&Path::new(dir).join(n)))
        .collect();
    names.sort();
    names
}

/// Device bytes loading the directory needs: its weights file (safetensors here are f32, which is also what the
/// encoder holds), a lower bound as everywhere else; 0 when unreadable (the load itself then decides).
pub fn weights_bytes(path: &Path) -> u64 {
    std::fs::metadata(path.join("model.safetensors")).map_or(0, |m| m.len())
}

/// Load the encoder in `path` and register it as an embedding entry under `name` (no runner thread, no chat
/// template). Blocking; call it from `spawn_blocking` (the manager does).
pub fn start_embedding_into(
    registry: &ModelRegistry,
    name: &str,
    path: &Path,
) -> anyhow::Result<()> {
    let embedder = NomicEmbedder::load(path)?;
    let spec = embedder.spec().clone();
    registry.set(
        name,
        ModelEntry {
            embedding: Some(Arc::new(LoadedEmbedder {
                spec,
                embedder: Box::new(embedder),
            })),
            ..ModelEntry::default()
        },
    );
    tracing::info!("Loaded embedding model {} as {name}", path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(files: &[(&str, &str)]) -> tempfile::TempDir {
        let d = tempfile::tempdir().unwrap();
        for (f, body) in files {
            std::fs::write(d.path().join(f), body).unwrap();
        }
        d
    }

    #[test]
    fn only_a_complete_nomic_bert_directory_is_an_embedding_model() {
        let ok = dir(&[
            ("model.safetensors", "x"),
            ("tokenizer.json", "{}"),
            ("config.json", r#"{"model_type":"nomic_bert"}"#),
        ]);
        assert!(is_embedding_dir(ok.path()));
        // Each missing or wrong piece is refused (controls against the complete one above).
        let no_tok = dir(&[
            ("model.safetensors", "x"),
            ("config.json", r#"{"model_type":"nomic_bert"}"#),
        ]);
        let other = dir(&[
            ("model.safetensors", "x"),
            ("tokenizer.json", "{}"),
            ("config.json", r#"{"model_type":"bert"}"#),
        ]);
        let bad_cfg = dir(&[
            ("model.safetensors", "x"),
            ("tokenizer.json", "{}"),
            ("config.json", "not json"),
        ]);
        for d in [&no_tok, &other, &bad_cfg] {
            assert!(!is_embedding_dir(d.path()));
        }
    }

    #[test]
    fn the_listing_has_embedding_directories_only_and_skips_hidden_ones() {
        let root = tempfile::tempdir().unwrap();
        let mk = |name: &str, model_type: &str| {
            let d = root.path().join(name);
            std::fs::create_dir(&d).unwrap();
            std::fs::write(d.join("model.safetensors"), b"x").unwrap();
            std::fs::write(d.join("tokenizer.json"), b"{}").unwrap();
            std::fs::write(
                d.join("config.json"),
                format!(r#"{{"model_type":"{model_type}"}}"#),
            )
            .unwrap();
        };
        mk("nomic", "nomic_bert");
        mk(".hidden", "nomic_bert");
        mk("other", "bert");
        std::fs::write(root.path().join("chat.gguf"), b"x").unwrap();
        assert_eq!(
            embedding_dir_names(root.path().to_str().unwrap()),
            vec!["nomic"]
        );
        assert_eq!(weights_bytes(&root.path().join("nomic")), 1);
        assert_eq!(weights_bytes(&root.path().join("missing")), 0);
    }
}
