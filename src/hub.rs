//! Hugging Face Hub integration for automatic model downloading
//!
//! This module provides utilities for downloading models from the Hugging Face Hub,
//! supporting both local paths and Hub model IDs like "meta-llama/Llama-2-7b-hf".

use anyhow::{Context, Result};
use hf_hub::api::sync::{Api, ApiRepo};
use hf_hub::{Repo, RepoType};
use std::path::{Path, PathBuf};

/// Model location - either a local path or a Hugging Face Hub model ID
#[derive(Debug, Clone)]
pub enum ModelSource {
    /// Local directory path (e.g., "./models/llama-7b")
    Local(PathBuf),
    /// Hugging Face Hub model ID (e.g., "meta-llama/Llama-2-7b-hf")
    Hub {
        model_id: String,
        revision: Option<String>,
    },
}

impl ModelSource {
    /// Parse a model string into a ModelSource
    ///
    /// If the string contains a '/', it's treated as a Hub model ID.
    /// Otherwise, it's treated as a local path.
    pub fn parse(model: &str) -> Self {
        if model.contains('/') && !model.starts_with('.') && !model.starts_with('/') {
            // Looks like a Hub model ID (org/model)
            Self::Hub {
                model_id: model.to_string(),
                revision: None,
            }
        } else {
            // Treat as local path
            Self::Local(PathBuf::from(model))
        }
    }

    /// Parse with explicit revision
    pub fn parse_with_revision(model: &str, revision: Option<String>) -> Self {
        if model.contains('/') && !model.starts_with('.') && !model.starts_with('/') {
            Self::Hub {
                model_id: model.to_string(),
                revision,
            }
        } else {
            Self::Local(PathBuf::from(model))
        }
    }
}

/// Download a model from Hugging Face Hub
///
/// Returns the local path to the downloaded model directory.
pub fn download_model(model_id: &str, revision: Option<&str>) -> Result<PathBuf> {
    println!("Downloading model {} from Hugging Face Hub...", model_id);
    
    let api = Api::new().context("Failed to initialize Hugging Face Hub API")?;
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.unwrap_or("main").to_string(),
    ));

    // Download required files
    // Most models need: config.json, tokenizer.json, and model weights
    let config_path = repo
        .get("config.json")
        .context("Failed to download config.json")?;
    
    // Get the model directory (parent of config.json)
    let model_dir = config_path
        .parent()
        .context("Invalid config.json path")?
        .to_path_buf();

    println!("Model downloaded to: {}", model_dir.display());
    Ok(model_dir)
}

/// Get a repository handle for downloading model files
pub fn get_repo(model_id: &str, revision: Option<&str>) -> Result<ApiRepo> {
    let api = Api::new().context("Failed to initialize Hugging Face Hub API")?;
    Ok(api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.unwrap_or("main").to_string(),
    )))
}

/// Download safetensors files for a model
///
/// Handles both single-file models (model.safetensors) and sharded models
/// (model.safetensors.index.json + model-00001-of-00002.safetensors, etc.)
pub fn download_safetensors(repo: &ApiRepo) -> Result<Vec<PathBuf>> {
    // Try to get the index file first (for sharded models)
    if let Ok(index_path) = repo.get("model.safetensors.index.json") {
        println!("Found model.safetensors.index.json, loading sharded model...");
        
        // Parse the index to find all shard files
        let index_content = std::fs::read_to_string(&index_path)
            .context("Failed to read model.safetensors.index.json")?;
        let index: serde_json::Value = serde_json::from_str(&index_content)
            .context("Failed to parse model.safetensors.index.json")?;
        
        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .context("No weight_map in model.safetensors.index.json")?;
        
        // Collect unique shard filenames
        let mut shard_files: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str())
            .map(|s| s.to_string())
            .collect();
        shard_files.sort();
        shard_files.dedup();
        
        println!("Downloading {} sharded model files...", shard_files.len());
        
        // Download each shard
        let mut paths = Vec::new();
        for filename in shard_files {
            println!("  - {}", filename);
            let path = repo
                .get(&filename)
                .with_context(|| format!("Failed to download {}", filename))?;
            paths.push(path);
        }
        
        Ok(paths)
    } else {
        // Single file model
        println!("Downloading model.safetensors...");
        let path = repo
            .get("model.safetensors")
            .context("Failed to download model.safetensors")?;
        Ok(vec![path])
    }
}

/// Download tokenizer for a model
pub fn download_tokenizer(repo: &ApiRepo) -> Result<PathBuf> {
    println!("Downloading tokenizer.json...");
    repo.get("tokenizer.json")
        .context("Failed to download tokenizer.json")
}

/// Download config for a model
pub fn download_config(repo: &ApiRepo) -> Result<PathBuf> {
    repo.get("config.json")
        .context("Failed to download config.json")
}

/// Download a GGUF quantized model
pub fn download_gguf(repo: &ApiRepo, filename: &str) -> Result<PathBuf> {
    println!("Downloading {} from Hub...", filename);
    repo.get(filename)
        .with_context(|| format!("Failed to download {}", filename))
}

/// Resolve a model source to a local directory path
///
/// For local paths, verifies the directory exists.
/// For Hub model IDs, downloads the model and returns the cache path.
pub fn resolve_model_path(source: &ModelSource) -> Result<PathBuf> {
    match source {
        ModelSource::Local(path) => {
            if !path.exists() {
                anyhow::bail!("Model path does not exist: {}", path.display());
            }
            if !path.is_dir() {
                anyhow::bail!("Model path is not a directory: {}", path.display());
            }
            Ok(path.clone())
        }
        ModelSource::Hub { model_id, revision } => {
            download_model(model_id, revision.as_deref())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_source_parsing() {
        // Local paths
        assert!(matches!(
            ModelSource::parse("./models/llama"),
            ModelSource::Local(_)
        ));
        assert!(matches!(
            ModelSource::parse("/absolute/path"),
            ModelSource::Local(_)
        ));
        assert!(matches!(
            ModelSource::parse("relative/path"),
            ModelSource::Local(_)
        ));

        // Hub model IDs
        match ModelSource::parse("meta-llama/Llama-2-7b-hf") {
            ModelSource::Hub { model_id, .. } => {
                assert_eq!(model_id, "meta-llama/Llama-2-7b-hf");
            }
            _ => panic!("Should be Hub source"),
        }

        match ModelSource::parse("mistralai/Mistral-7B-v0.1") {
            ModelSource::Hub { model_id, .. } => {
                assert_eq!(model_id, "mistralai/Mistral-7B-v0.1");
            }
            _ => panic!("Should be Hub source"),
        }
    }
}
