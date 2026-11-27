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

/// Where `/v1/embeddings` runs the encoder (`LIGHTBULB_EMBEDDING_DEVICE`). CPU is the default and always
/// works. An accelerator is NAMED, and one that cannot be used is an error at startup or at load, never a
/// quiet CPU run: asking for the GPU and measuring the CPU is the failure this type exists to prevent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmbeddingDevice {
    #[default]
    Cpu,
    Cuda(u32),
}

impl EmbeddingDevice {
    /// `""` or `cpu` is the CPU; `cuda` is `cuda:0`; `cuda:<n>` is that card. `vulkan` is recognised and
    /// refused by name (Fuel has the backend, this build does not link it); anything else is an error.
    pub fn parse(raw: &str) -> Result<Self, String> {
        let s = raw.trim().to_ascii_lowercase();
        match s.as_str() {
            "" | "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda(0)),
            "vulkan" => Err(
                "vulkan embeddings are not built into this binary (only cpu and cuda[:n])".into(),
            ),
            _ => match s.strip_prefix("cuda:").map(str::parse::<u32>) {
                Some(Ok(n)) => Ok(Self::Cuda(n)),
                _ => Err(format!(
                    "{raw:?} is not a device: use cpu, cuda or cuda:<n>"
                )),
            },
        }
    }

    /// `"cpu"` or `"cuda:<n>"`, the spelling `/v1/models` reports.
    pub fn label(self) -> String {
        match self {
            Self::Cpu => "cpu".into(),
            Self::Cuda(n) => format!("cuda:{n}"),
        }
    }

    /// Whether this build and machine can run on the device, so startup can refuse a bad setting.
    pub fn check_available(self) -> Result<(), String> {
        match self {
            Self::Cpu => Ok(()),
            Self::Cuda(n) => super::openai::nomic_embedder::open_cuda(n).map(|_| ()),
        }
    }
}

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
    device: EmbeddingDevice,
) -> anyhow::Result<()> {
    let embedder = NomicEmbedder::load_on(path, device)?;
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
    tracing::info!(
        "Loaded embedding model {} as {name} on {}",
        path.display(),
        device.label()
    );
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

#[cfg(test)]
mod device_tests {
    use super::EmbeddingDevice;

    #[test]
    fn the_device_setting_parses_and_refuses_by_name() {
        assert_eq!(EmbeddingDevice::parse("").unwrap(), EmbeddingDevice::Cpu);
        assert_eq!(
            EmbeddingDevice::parse(" CPU ").unwrap(),
            EmbeddingDevice::Cpu
        );
        assert_eq!(
            EmbeddingDevice::parse("cuda").unwrap(),
            EmbeddingDevice::Cuda(0)
        );
        assert_eq!(
            EmbeddingDevice::parse("CUDA:3").unwrap(),
            EmbeddingDevice::Cuda(3)
        );
        assert_eq!(EmbeddingDevice::default(), EmbeddingDevice::Cpu);
        // Refusals: Vulkan by name, and nonsense. Control: each names what it refused.
        let v = EmbeddingDevice::parse("vulkan").unwrap_err();
        assert!(v.contains("vulkan"), "{v}");
        for bad in ["gpu", "cuda:", "cuda:x", "cuda:-1", "cuda0"] {
            let e = EmbeddingDevice::parse(bad).unwrap_err();
            assert!(e.contains(bad), "{bad}: {e}");
        }
        assert_eq!(EmbeddingDevice::Cuda(2).label(), "cuda:2");
        assert!(EmbeddingDevice::Cpu.check_available().is_ok());
    }

    /// Without the `fuel-cuda` feature a CUDA request is an error that says why, from both the startup check and
    /// the loader, so a binary that cannot run on the GPU never answers an embedding from the CPU.
    #[cfg(not(feature = "fuel-cuda"))]
    #[test]
    fn cuda_without_the_feature_is_refused_not_served_from_the_cpu() {
        let e = EmbeddingDevice::Cuda(0).check_available().unwrap_err();
        assert!(e.contains("fuel-cuda"), "{e}");
        let load = crate::api::openai::nomic_embedder::NomicEmbedder::load_on(
            std::path::Path::new("."),
            EmbeddingDevice::Cuda(0),
        );
        let msg = format!("{:#}", load.err().expect("must not load"));
        assert!(msg.contains("fuel-cuda"), "{msg}");
    }
}
