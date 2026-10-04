//! Loading a GGUF-quantized Qwen3-shape checkpoint into Fuel.
//!
//! The Llama loader (`loader_gguf.rs`) needs `fuel_loaders::quantized::
//! config_from_gguf::derive_config` plus locally-computed `head_dim` because
//! fuel's own config reader deliberately excludes Llama-specific concerns.
//! Qwen3 has no such detour: `fuel_transformers::models::lazy_quantized_qwen3::
//! qwen3_config_from_gguf_content` reads EVERY `Qwen3Config` field directly
//! from the GGUF's own metadata (`qwen3.attention.head_count`,
//! `qwen3.attention.key_length` for `head_dim`, etc. — see its own doc
//! comment at fuel rev `648e122b`), so this file is a straight read-and-build,
//! not an adaptation. The one piece it still doesn't cover is bos/eos (same
//! `derive_config` exclusion reason), read the same way `loader_gguf.rs` does
//! via `bos_eos_from_metadata` (reused, not duplicated — those GGUF keys are
//! a shared tokenizer convention, not architecture-specific).
//!
//! `require_qwen3_architecture` gates this file the same way
//! `require_llama_architecture` gates the Llama one: refuse loudly on a
//! mismatched `general.architecture`, rather than let a wrong-shaped
//! checkpoint reach code built for a different one. Deliberately a SEPARATE
//! function, not a parameterized version of `require_llama_architecture` —
//! that function's own doc comment catalogs architecture-specific hazards
//! (gemma4's dual attention geometry, partial RoPE, etc.) that a shared
//! "refuse unless X" helper would have to either ignore or duplicate per
//! architecture anyway.
//!
//! Architecture **detection** (reading `general.architecture` to decide
//! Llama vs. Qwen3 BEFORE calling either loader) is deliberately not this
//! file's job either — it belongs at the call site that currently hardcodes
//! the Llama loader for every `.gguf` file
//! (`engine_model.rs::FuelEngineModel::load`,
//! `scheduled_llama_engine.rs::run_fuel_engine`), which is unwired follow-up
//! work, not this PR.

use std::path::Path;

use anyhow::{Context, Result};

use fuel_transformers::models::lazy_llama_full::LlamaEosToks;
use fuel_transformers::models::lazy_quantized_qwen3::{
    QuantizedQwen3Model, qwen3_config_from_gguf_content,
};
use fuel_transformers::models::lazy_qwen3::Qwen3Config;

use super::loader_gguf::bos_eos_from_metadata;

/// A loaded GGUF-quantized Qwen3 model, plus everything a serving loop needs
/// to drive it. Mirrors `loader.rs`'s `LoadedQuantizedLlama` field-for-field.
pub struct LoadedQuantizedQwen3 {
    pub model: QuantizedQwen3Model,
    pub config: Qwen3Config,
    pub tokenizer: tokenizers::Tokenizer,
    pub eos: Option<LlamaEosToks>,
    pub device: fuel::Device,
}

impl LoadedQuantizedQwen3 {
    /// `true` if `tok` ends generation for this checkpoint. Mirrors
    /// `LoadedQuantizedLlama::is_eos`.
    pub fn is_eos(&self, tok: u32) -> bool {
        self.eos.as_ref().is_some_and(|e| e.is_eos(tok))
    }
}

/// Refuse a GGUF whose declared architecture is not `qwen3`. See the module
/// doc for why this is a separate function from `require_llama_architecture`
/// rather than a parameterized shared one.
fn require_qwen3_architecture(
    metadata: &std::collections::HashMap<String, crate::gguf::Value>,
) -> Result<()> {
    use crate::gguf::Value;
    let architecture = match metadata.get("general.architecture") {
        Some(Value::String(s)) => s.clone(),
        _ => anyhow::bail!(
            "this GGUF declares no `general.architecture`, so the architecture cannot be \
             checked before reading qwen3-specific keys. Every GGUF in the reference corpus \
             declares it; a file without it is malformed or truncated."
        ),
    };
    if architecture != "qwen3" {
        anyhow::bail!(
            "this GGUF declares `general.architecture = {architecture:?}`; this loader reads \
             `qwen3` only. Its hyperparameters are under the `{architecture}.` prefix, not \
             `qwen3.`, and the config and tensor mapping built here are qwen3-shaped, so \
             reading them would produce a wrong model rather than a missing key."
        );
    }
    Ok(())
}

/// Derive a `Qwen3Config` by reading the checkpoint's header through fuel's
/// own `gguf_file::Content` — a second, metadata-only parse of the same
/// file (not a second mmap of the weights), same shape as `loader_gguf.rs`'s
/// `derive_llama_full_config`. Split out of [`load_quantized_qwen3_gguf`] to
/// keep that function's own cyclomatic complexity down, same reasoning as
/// `derive_llama_full_config`'s own separation from its caller.
fn derive_qwen3_config(path: &Path) -> Result<Qwen3Config> {
    let mut file = std::fs::File::open(path).with_context(|| {
        format!(
            "opening {} for fuel's own GGUF metadata read",
            path.display()
        )
    })?;
    let fuel_content = fuel_loaders::quantized::gguf_file::Content::read(&mut file)
        .map_err(|e| anyhow::anyhow!("fuel gguf_file::Content::read({}): {e:?}", path.display()))?;
    qwen3_config_from_gguf_content(&fuel_content).map_err(|e| {
        anyhow::anyhow!(
            "fuel qwen3_config_from_gguf_content({}): {e}",
            path.display()
        )
    })
}

/// Log the same "no declared EOS" warning `loader_gguf.rs`'s Llama loader
/// does, when `no_eos` is true. Split out purely to keep
/// [`load_quantized_qwen3_gguf`]'s own cyclomatic complexity down.
fn warn_if_no_eos(path: &Path, no_eos: bool) {
    if no_eos {
        tracing::warn!(
            "GGUF file {} declares no tokenizer.ggml.eos_token_id; generation will stop \
             only on max_new_tokens",
            path.display()
        );
    }
}

/// Load a GGUF-quantized Qwen3-shape checkpoint from a single `.gguf` file.
///
/// Tokenizer comes from the GGUF file's own embedded vocabulary, same as
/// `loader_gguf.rs`'s Llama loader — no separate `tokenizer.json` needed.
///
/// # Safety
///
/// Memory-maps or reads the checkpoint's header up to three times: once
/// through `crate::gguf::Content` (tokenizer + bos/eos), once through fuel's
/// own `gguf_file::Content` (via `derive_qwen3_config`, private — not an
/// intra-doc link here since this function is `pub`), and once inside
/// `QuantizedQwen3Model::from_gguf` (the actual weights) — the OS shares page
/// cache across all three. Mutating the file while any is alive is undefined
/// behaviour. Mirrors `load_quantized_llama_gguf`'s own safety note exactly.
pub fn load_quantized_qwen3_gguf(path: &Path) -> Result<LoadedQuantizedQwen3> {
    let content = crate::gguf::Content::read(path)
        .with_context(|| format!("reading GGUF metadata from {}", path.display()))?;
    require_qwen3_architecture(content.metadata())?;

    let tokenizer = content
        .extract_tokenizer()
        .with_context(|| format!("extracting tokenizer from {}", path.display()))?;

    let cfg = derive_qwen3_config(path)?;

    let (_bos, eos) = bos_eos_from_metadata(&content);
    warn_if_no_eos(path, eos.is_none());

    let device = super::device::select();
    let model = QuantizedQwen3Model::from_gguf(path, &cfg).map_err(|e| {
        anyhow::anyhow!(
            "building QuantizedQwen3Model from {}: {e:?}",
            path.display()
        )
    })?;

    Ok(LoadedQuantizedQwen3 {
        model,
        config: cfg,
        tokenizer,
        eos,
        device,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn require_qwen3_architecture_accepts_qwen3() {
        let mut md = std::collections::HashMap::new();
        md.insert(
            "general.architecture".to_string(),
            crate::gguf::Value::String("qwen3".to_string()),
        );
        assert!(require_qwen3_architecture(&md).is_ok());
    }

    #[test]
    fn require_qwen3_architecture_refuses_other_architectures() {
        let mut md = std::collections::HashMap::new();
        md.insert(
            "general.architecture".to_string(),
            crate::gguf::Value::String("llama".to_string()),
        );
        let err = require_qwen3_architecture(&md).unwrap_err().to_string();
        assert!(err.contains("qwen3"));
        assert!(err.contains("llama"));
    }

    #[test]
    fn require_qwen3_architecture_refuses_a_missing_declaration() {
        let md = std::collections::HashMap::new();
        let err = require_qwen3_architecture(&md).unwrap_err().to_string();
        assert!(err.contains("general.architecture"));
    }
}
