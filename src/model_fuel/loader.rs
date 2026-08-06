//! Loading a Llama-shape checkpoint into Fuel, from a local directory.
//!
//! Fuel's own `Llama2cModel::from_hub` goes through `hf_hub`. Lightbulb needs to
//! load from a path it already has — a served model directory, or a warm HF
//! cache snapshot — so this reimplements that path over local files using the
//! same three pieces Fuel uses internally.
//!
//! # dtype: this uses Fuel's stock loader, which is NOT the oracle's dtype
//!
//! `LlamaWeights::load_from_mmapped` loads embeddings as f32 but **deliberately
//! preserves bf16 for projection matrices** — its own comment: *"bf16 source
//! files stay bf16 on-device (halving weight memory on this layer)."*
//!
//! That means a bf16 checkpoint lands in the **promoted regime**: Fuel's
//! dtype-reconciliation pass inserts a value-lossless-but-not-
//! accumulation-preserving upcast at every mixed matmul. On TinyLlama-1.1B that
//! is **155 promoting casts per realize**, measured by Fuel.
//!
//! This is fine for *proving the path works* and is deliberately what this
//! function does — it is the exact route `llama-lazy` exercises, so a failure
//! here is a Lightbulb integration bug rather than a loader bug.
//!
//! **It is not fine for the parity oracle**, which requires all-f32 weights so
//! the matmul key stays `[F32, F32, F32]`, no fixup runs, and goldens are
//! captured on a path the pass never touches. That needs a loader built on
//! `load_transposed_matrix` (the f32 variant, which also handles the HF
//! `[out, in]` → Fuel `[in, out]` transpose) rather than
//! `load_transposed_matrix_preserve_dtype`. `WeightStorage` has no dtype
//! conversion, so it is a separate loader rather than a post-pass — tracked as
//! the next step, not an oversight.

use anyhow::{Context, Result};
use std::path::Path;

use fuel::lazy::{LlamaModel, LlamaWeights};
use fuel::lazy_llama2c::Llama2cConfig;

/// A loaded Llama-shape model plus the config it was built from.
pub struct LoadedLlama {
    pub model: LlamaModel,
    /// Retained because callers need `vocab_size` to slice logits and
    /// `n_layers`/`n_kv_heads`/`head_dim` to size a KV cache.
    pub config: fuel::lazy::LlamaConfig,
}

impl LoadedLlama {
    /// The dtype every projection weight was materialized at, or `None` when
    /// they disagree.
    ///
    /// # Why this exists
    ///
    /// The tier-3 golden records `provenance.loader` to pin that a fixture was
    /// captured on the all-f32 path — rule 2, and not pedantry: bf16
    /// projections give a matmul key of `[F32, BF16, F32]`, which no CPU kernel
    /// serves, so the optimizer inserts a promoting cast that is value-lossless
    /// but **not accumulation-preserving**. Numbers captured through it are not
    /// comparable with numbers captured without it.
    ///
    /// That check could not fail. `build_provenance` writes the `REQUIRED_LOADER`
    /// constant and every call site compares against the same constant, so it is
    /// constant-versus-constant: switch the capture to the bf16 loader and the
    /// fixture still claims f32. This method makes the loader's **consequence**
    /// observable, so the golden can assert what the weights actually are rather
    /// than what a string says they should be.
    ///
    /// Returns `None` on a mixed set rather than picking a representative,
    /// because "some projections are bf16" is exactly the state the check exists
    /// to catch, and a representative sample could miss it — the same
    /// passing-sample-is-not-a-passing-class trap this project keeps meeting.
    pub fn projection_dtype(&self) -> Option<fuel::DType> {
        let w: &LlamaWeights = &self.model.weights;
        let mut dt = w.output.dtype();
        for layer in &w.layers {
            for s in [
                &layer.attn_q,
                &layer.attn_k,
                &layer.attn_v,
                &layer.attn_o,
                &layer.ffn_gate,
                &layer.ffn_up,
                &layer.ffn_down,
            ] {
                if s.dtype() != dt {
                    return None;
                }
                dt = s.dtype();
            }
        }
        Some(dt)
    }
}

/// Load a Llama-shape checkpoint from a local directory containing
/// `config.json` and `model.safetensors`.
///
/// Single-file checkpoints only for now. Sharded ones advertise their parts in
/// `model.safetensors.index.json`; `MmapedSafetensors::multi` already accepts a
/// path list, so extending this is reading the index rather than new machinery.
///
/// # Safety
///
/// Memory-maps the checkpoint. Mutating the file while this is alive is
/// undefined behaviour — the same contract Fuel's own loader carries.
pub fn load_llama_from_dir(dir: &Path) -> Result<LoadedLlama> {
    let config_path = dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;

    // Llama2cConfig::from_hf_json_str parses HF-native field names
    // (hidden_size → dim, num_hidden_layers → n_layers, ...) and is documented
    // compatible with TinyLlama, Llama-2, Llama-3 and Mistral.
    let cfg_2c = Llama2cConfig::from_hf_json_str(&config_str)
        .map_err(|e| anyhow::anyhow!("parsing config.json: {e:?}"))?;
    let config = cfg_2c.to_llama_config();

    let weights_path = dir.join("model.safetensors");
    if !weights_path.is_file() {
        anyhow::bail!(
            "no model.safetensors in {} (sharded checkpoints are not handled yet)",
            dir.display()
        );
    }

    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&[weights_path]) }
        .map_err(|e| anyhow::anyhow!("mmapping safetensors: {e:?}"))?;

    let weights = LlamaWeights::load_from_mmapped(&st, &config)
        .map_err(|e| anyhow::anyhow!("building LlamaWeights: {e:?}"))?;

    Ok(LoadedLlama {
        model: LlamaModel {
            config: config.clone(),
            weights,
        },
        config,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The warm HF cache snapshot `llama-lazy` already populated. Skipped rather
    /// than failed when absent, so the suite stays green on a machine without it.
    fn tinyllama_dir() -> Option<std::path::PathBuf> {
        let p = std::path::PathBuf::from(
            "C:/Users/cires/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        );
        p.join("model.safetensors").is_file().then_some(p)
    }

    /// Loads real TinyLlama weights into Fuel from a local directory and checks
    /// the config against the values `llama-lazy` printed on its verified run —
    /// so a silently-wrong parse is caught rather than assumed correct.
    ///
    /// `#[ignore]` because it needs a 2.2 GB checkpoint on disk.
    /// Run: `cargo test --lib model_fuel -- --ignored --nocapture`
    #[test]
    #[ignore = "needs the TinyLlama checkpoint in the HF cache"]
    fn loads_tinyllama_from_local_dir() -> Result<()> {
        let Some(dir) = tinyllama_dir() else {
            eprintln!("skipping: no TinyLlama snapshot");
            return Ok(());
        };

        let loaded = load_llama_from_dir(&dir)?;
        let c = &loaded.config;

        // These are the values llama-lazy printed on the run that produced
        // "Once upon a time, in a land far far away," — a known-good reference.
        assert_eq!(c.dim, 2048, "hidden size");
        assert_eq!(c.n_layers, 22, "layer count");
        assert_eq!(c.n_heads, 32, "attention heads");
        assert_eq!(c.n_kv_heads, 4, "KV heads (GQA)");
        assert_eq!(c.vocab_size, 32000, "vocab");

        eprintln!(
            "loaded TinyLlama: dim={} layers={} heads={} kv_heads={} vocab={}",
            c.dim, c.n_layers, c.n_heads, c.n_kv_heads, c.vocab_size
        );
        Ok(())
    }
}
