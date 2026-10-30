// SPDX-License-Identifier: MIT OR Apache-2.0
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
use mlmf_hf_layout::shards::ShardIndex;
use std::path::{Path, PathBuf};

use fuel_model_llama::{LlamaConfig, LlamaModel, LlamaWeights};

/// A loaded GGUF-quantized Llama-shape model, plus everything a serving loop
/// needs to drive it — the `QuantizedLlama3Model` counterpart to
/// [`LoadedLlama`]. See `loader_gguf.rs` for how this is built.
///
/// `eos` uses `fuel_transformers::models::lazy_llama_full::LlamaEosToks`
/// directly (not `Option<LlamaEosToks>` wrapped again) since
/// `LlamaFullConfig::eos_token_id` already carries that `Option`.
pub struct LoadedQuantizedLlama {
    pub model: fuel_transformers::models::lazy_quantized_llama::QuantizedLlama3Model,
    pub config: LlamaConfig,
    pub tokenizer: tokenizers::Tokenizer,
    pub eos: Option<fuel_transformers::models::lazy_llama_full::LlamaEosToks>,
    pub device: fuel::Device,
}

impl LoadedQuantizedLlama {
    /// `true` if `tok` ends generation for this checkpoint. Mirrors
    /// [`LoadedLlama::is_eos`].
    pub fn is_eos(&self, tok: u32) -> bool {
        self.eos.as_ref().is_some_and(|e| e.is_eos(tok))
    }
}

/// A loaded Llama-shape model, plus everything a serving loop needs to drive it.
///
/// `config` is retained because callers need `vocab_size` to slice logits and
/// `n_layers`/`n_kv_heads`/`head_dim` to size a KV cache.
///
/// `tokenizer`, `eos` and `device` were added when this path was wired into the
/// engine. All three load EAGERLY: a server that accepts a request and only
/// then discovers it cannot detokenize has converted a startup error into a
/// per-request one.
pub struct LoadedLlama {
    pub model: LlamaModel,
    pub config: LlamaConfig,
    pub tokenizer: tokenizers::Tokenizer,
    /// `None` when `config.json` carries no `eos_token_id`. Generation then
    /// stops only on `max_new_tokens`, which is why the loader logs a warning.
    pub eos: Option<fuel_transformers::models::lazy_llama_full::LlamaEosToks>,
    /// The device the weights were loaded onto. A `SessionState` built against
    /// a different one is a byte-count error deep inside `realize`.
    pub device: fuel::Device,
}

impl LoadedLlama {
    /// `true` if `tok` ends generation for this checkpoint.
    ///
    /// `false` when the checkpoint declares no EOS — generation then runs to
    /// `max_new_tokens`, which is the honest behaviour for a model that never
    /// declared a stop.
    pub fn is_eos(&self, tok: u32) -> bool {
        self.eos.as_ref().is_some_and(|e| e.is_eos(tok))
    }

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

/// The safetensors files holding a checkpoint directory's weights, in a stable
/// order: `model.safetensors` if it exists, otherwise the distinct shards named by
/// `model.safetensors.index.json` (parsed by `mlmf-hf-layout`), sorted.
///
/// The index is data from the model directory, so each shard name is checked
/// before it is joined onto `dir`: it must be a single path component ending in
/// `.safetensors`. `mlmf-hf-layout` turns bytes into structure and does no path
/// checking, so an index naming `../x` or `/etc/x` would otherwise be mapped.
pub(crate) fn safetensors_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let single = dir.join("model.safetensors");
    if single.is_file() {
        return Ok(vec![single]);
    }
    let index_path = dir.join("model.safetensors.index.json");
    if !index_path.is_file() {
        anyhow::bail!(
            "neither model.safetensors nor model.safetensors.index.json in {}",
            dir.display()
        );
    }
    let bytes =
        std::fs::read(&index_path).with_context(|| format!("reading {}", index_path.display()))?;
    let index = ShardIndex::parse(&bytes)
        .map_err(|e| anyhow::anyhow!("parsing {}: {e}", index_path.display()))?;
    let names = index.shards();
    if names.is_empty() {
        anyhow::bail!("{} lists no shard files", index_path.display());
    }
    names
        .into_iter()
        .map(|name| shard_path(dir, &index_path, name))
        .collect()
}

/// `name` is a single path component ending in `.safetensors`: no separator,
/// no `..`/`.`, no drive prefix, not empty.
fn is_plain_safetensors_name(name: &str) -> bool {
    Path::new(name)
        .file_name()
        .is_some_and(|f| f == std::ffi::OsStr::new(name))
        && !name.contains(['/', '\\'])
        && name.ends_with(".safetensors")
}

/// One shard named by the index: checked as a plain file name, then required to
/// exist in `dir`.
fn shard_path(dir: &Path, index_path: &Path, name: &str) -> Result<PathBuf> {
    if !is_plain_safetensors_name(name) {
        anyhow::bail!(
            "{} lists the shard {name:?}, which is not a plain *.safetensors file name",
            index_path.display()
        );
    }
    let path = dir.join(name);
    if !path.is_file() {
        anyhow::bail!(
            "{} lists the shard {name}, which is not in {}",
            index_path.display(),
            dir.display()
        );
    }
    Ok(path)
}

/// Load a Llama-shape checkpoint from a local directory containing
/// `config.json` and either `model.safetensors` or a sharded set described by
/// `model.safetensors.index.json` (see `safetensors_files`).
///
/// # Safety
///
/// Memory-maps the checkpoint. Mutating the file while this is alive is
/// undefined behaviour — the same contract Fuel's own loader carries.
pub fn load_llama_from_dir(dir: &Path) -> Result<LoadedLlama> {
    let config_path = dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;

    // `LlamaFullConfig`, NOT `Llama2cConfig`. Both parse HF-native field names
    // (hidden_size → dim, num_hidden_layers → n_layers, ...) into the same
    // `LlamaConfig` via `to_lazy_config()`, but only this one retains
    // `eos_token_id`. Without it the engine can stop only on max_new_tokens.
    let full =
        fuel_transformers::models::lazy_llama_full::LlamaFullConfig::from_hf_json_str(&config_str)
            .map_err(|e| anyhow::anyhow!("parsing config.json: {e:?}"))?;
    let config = full.to_lazy_config();

    let weight_files = safetensors_files(dir)?;

    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&weight_files) }
        .map_err(|e| anyhow::anyhow!("mmapping safetensors: {e:?}"))?;

    let weights = LlamaWeights::load_from_mmapped(&st, &config)
        .map_err(|e| anyhow::anyhow!("building LlamaWeights: {e:?}"))?;

    let tokenizer_path = dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("loading {}: {e}", tokenizer_path.display()))?;
    if full.eos_token_id.is_none() {
        tracing::warn!(
            "config.json in {} declares no eos_token_id; generation will stop \
             only on max_new_tokens",
            dir.display()
        );
    }

    Ok(LoadedLlama {
        model: LlamaModel {
            config: config.clone(),
            weights,
        },
        config,
        tokenizer,
        eos: full.eos_token_id,
        device: super::device::select(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, body: &str) {
        std::fs::write(dir.join(name), body).unwrap();
    }
    fn index(entries: &[(&str, &str)]) -> String {
        let map: Vec<String> = entries
            .iter()
            .map(|(t, f)| format!("\"{t}\": \"{f}\""))
            .collect();
        format!(
            "{{\"metadata\": {{\"total_size\": 1}}, \"weight_map\": {{{}}}}}",
            map.join(", ")
        )
    }

    #[test]
    fn a_single_file_checkpoint_wins_even_when_an_index_is_present() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "model.safetensors", "x");
        write(
            d.path(),
            "model.safetensors.index.json",
            &index(&[("a", "s1.safetensors")]),
        );
        assert_eq!(
            safetensors_files(d.path()).unwrap(),
            vec![d.path().join("model.safetensors")]
        );
    }

    #[test]
    fn a_sharded_checkpoint_resolves_to_its_sorted_distinct_shards() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "model-00002-of-00002.safetensors", "x");
        write(d.path(), "model-00001-of-00002.safetensors", "x");
        write(
            d.path(),
            "model.safetensors.index.json",
            &index(&[
                ("t.c", "model-00002-of-00002.safetensors"),
                ("t.a", "model-00001-of-00002.safetensors"),
                ("t.b", "model-00001-of-00002.safetensors"),
            ]),
        );
        assert_eq!(
            safetensors_files(d.path()).unwrap(),
            vec![
                d.path().join("model-00001-of-00002.safetensors"),
                d.path().join("model-00002-of-00002.safetensors")
            ]
        );
    }

    #[test]
    fn neither_file_is_an_error_naming_both() {
        let d = tempfile::tempdir().unwrap();
        let e = format!("{:#}", safetensors_files(d.path()).unwrap_err());
        assert!(
            e.contains("model.safetensors") && e.contains("model.safetensors.index.json"),
            "{e}"
        );
    }

    #[test]
    fn a_listed_shard_that_is_missing_is_an_error_naming_it() {
        let d = tempfile::tempdir().unwrap();
        write(
            d.path(),
            "model.safetensors.index.json",
            &index(&[("a", "gone.safetensors")]),
        );
        let e = format!("{:#}", safetensors_files(d.path()).unwrap_err());
        assert!(e.contains("gone.safetensors"), "{e}");
    }

    /// The refused names point at files that EXIST, so only the name check can
    /// reject them -- not the later "shard is missing" check.
    /// The refused names point at files that EXIST, so only the name check can
    /// reject them -- not the later "shard is missing" check.
    #[test]
    fn shard_names_that_are_not_plain_safetensors_file_names_are_refused() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("model");
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(root.path().join("evil.safetensors"), "x").unwrap();
        std::fs::write(dir.join("sub").join("x.safetensors"), "x").unwrap();
        std::fs::write(dir.join("notes.txt"), "x").unwrap();
        std::fs::write(dir.join("x.safetensors.bak"), "x").unwrap();
        let absolute = root
            .path()
            .join("evil.safetensors")
            .to_string_lossy()
            .replace('\u{5c}', "/");
        let backslash = format!("sub{}x.safetensors", '\u{5c}');
        for bad in [
            "../evil.safetensors".to_string(),
            "sub/x.safetensors".to_string(),
            backslash,
            absolute,
            "..".to_string(),
            ".".to_string(),
            String::new(),
            "notes.txt".to_string(),
            "x.safetensors.bak".to_string(),
        ] {
            write(&dir, "model.safetensors.index.json", &index(&[("a", &bad)]));
            assert!(safetensors_files(&dir).is_err(), "{bad:?} was accepted");
        }
    }

    #[test]
    fn an_empty_weight_map_is_an_error() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "model.safetensors.index.json", &index(&[]));
        assert!(safetensors_files(d.path()).is_err());
    }

    #[test]
    fn a_malformed_or_weightless_index_is_an_error() {
        for body in ["not json", "[]", "{\"metadata\": {}}"] {
            let d = tempfile::tempdir().unwrap();
            write(d.path(), "model.safetensors.index.json", body);
            assert!(safetensors_files(d.path()).is_err(), "{body:?}");
        }
    }

    /// The warm HF cache snapshot `llama-lazy` already populated. Skipped rather
    /// than failed when absent, so the suite stays green on a machine without it.
    fn tinyllama_dir() -> Option<std::path::PathBuf> {
        // one locator, not twelve copies of an absolute path into one home directory
        crate::test_models::tinyllama_dir()
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
            crate::test_notice::skip_unless_required(
                "LIGHTBULB_REQUIRE_MODEL",
                "no TinyLlama snapshot",
            );
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
