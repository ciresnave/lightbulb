//! Loading a GGUF-quantized Llama-shape checkpoint into Fuel.
//!
//! # Provenance and removal condition — read before touching this file
//!
//! `fuel::QuantizedLlama3Model::from_gguf` takes a pre-built
//! `fuel::lazy_llama_full::LlamaFullConfig` as an argument; it does not derive
//! one from the GGUF file itself. Checked against fuel's `origin/main` on
//! 2026-09-24 (`gh api` code search over `repo:ciresnave/fuel` for the
//! `llama.*` GGUF metadata keys): the only place fuel derives this config is
//! ~80 hand-written lines inside `fuel-examples/examples/quantized/main.rs`'s
//! `llama_config_from_gguf`, and no reusable helper exists anywhere in the
//! repo. **This is a missing API on fuel's side, filed with the fuel lane the
//! same day** (see `docs/FUEL-PORT-STATUS-2026-09-24.md` §6), with
//! `fuel-loaders` named as the obvious home (its stated job is already
//! "HF `config.json` resolution", and this is the same job for GGUF).
//!
//! **What this file actually is: NOT a copy of that ~80-line function.**
//! Lightbulb already has its own independently-tested GGUF metadata reader
//! (`crate::gguf`, `metadata_u64`/`metadata_f32`, used by the candlelight GGUF
//! loader in `src/loaders/mod.rs`), including a fallback chain for
//! `vocab_size` that fuel's example does not need (fuel derives vocab_size
//! from `token_embd.weight`'s tensor shape instead — a different, also-valid
//! derivation this file does not need to duplicate because the metadata-key
//! chain already handles the one local checkpoint this was tested against).
//! What IS taken from fuel's example is the **field mapping** — which
//! `llama.*` GGUF keys correspond to which `LlamaFullConfig` field — because
//! that mapping is not documented anywhere else fuel or Lightbulb has it
//! written down.
//!
//! **DELETE THIS FILE'S CONFIG-BUILDING FUNCTION (`llama_full_config_from_gguf`)
//! THE MOMENT FUEL SHIPS A `from_gguf`-adjacent config derivation** (in
//! `fuel-loaders` or elsewhere) and call that instead. Until then this is
//! Lightbulb's own code, reusing Lightbulb's own already-tested reader, not a
//! fork of fuel's.

use std::path::Path;

use anyhow::{Context, Result};

use fuel::lazy_llama_full::{LlamaEosToks, LlamaFullConfig};
use fuel::lazy_quantized_llama::QuantizedLlama3Model;

use super::loader::LoadedQuantizedLlama;

/// Build a `LlamaFullConfig` from a GGUF file's own metadata, using
/// Lightbulb's existing GGUF metadata reader (`crate::gguf`), not fuel's.
///
/// See the module doc for why this exists and when to delete it.
fn llama_full_config_from_gguf(content: &crate::gguf::Content) -> Result<LlamaFullConfig> {
    crate::gguf::require_llama_architecture(content.metadata())?;

    let get_u64 = |key: &str| -> Result<u64> { crate::gguf::metadata_u64(content.metadata(), key) };
    let get_f32 = |key: &str| -> Result<f32> { crate::gguf::metadata_f32(content.metadata(), key) };

    let hidden_size = get_u64("llama.embedding_length")? as usize;
    let intermediate_size = get_u64("llama.feed_forward_length")? as usize;
    let num_hidden_layers = get_u64("llama.block_count")? as usize;
    let num_attention_heads = get_u64("llama.attention.head_count")? as usize;
    let num_key_value_heads = get_u64("llama.attention.head_count_kv")? as usize;
    let head_dim = hidden_size / num_attention_heads.max(1);

    use crate::gguf::Value;

    // Same fallback chain as `src/loaders/mod.rs::extract_llama_config_from_metadata`
    // — `llama.vocab_size` is absent on this repo's own TinyLlama GGUF, measured
    // 2026-09-05 over the local corpus. Kept in sync deliberately: both loaders
    // solve the same problem on the same files.
    let vocab_size = get_u64("llama.vocab_size")
        .or_else(|_| get_u64("llama.n_vocab"))
        .map(|v| v as usize)
        .or_else(|_| match content.metadata().get("tokenizer.ggml.tokens") {
            Some(Value::Array(tokens)) => Ok(tokens.len()),
            _ => anyhow::bail!(
                "could not determine vocab_size: tried llama.vocab_size, llama.n_vocab, \
                 and counting tokenizer.ggml.tokens"
            ),
        })?;

    let rms_norm_eps = get_f32("llama.attention.layer_norm_rms_epsilon").unwrap_or(1e-5) as f64;
    let rope_theta = get_f32("llama.rope.freq_base").unwrap_or(10_000.0) as f64;
    let max_position_embeddings = get_u64("llama.context_length").unwrap_or(2048) as usize;

    let bos_token_id = content
        .metadata()
        .get("tokenizer.ggml.bos_token_id")
        .and_then(|v| match v {
            Value::U32(id) => Some(*id),
            _ => None,
        });
    let eos_token_id = content
        .metadata()
        .get("tokenizer.ggml.eos_token_id")
        .and_then(|v| match v {
            Value::U32(id) => Some(*id),
            _ => None,
        })
        .map(LlamaEosToks::Single);

    Ok(LlamaFullConfig {
        hidden_size,
        intermediate_size,
        vocab_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        rms_norm_eps,
        rope_theta,
        max_position_embeddings,
        bos_token_id,
        eos_token_id,
        rope_scaling: None,
        tie_word_embeddings: false,
    })
}

/// Load a GGUF-quantized Llama-shape checkpoint from a single `.gguf` file.
///
/// Tokenizer comes from the GGUF file's own embedded vocabulary
/// (`crate::gguf::Content::extract_tokenizer`, the same extractor the
/// candlelight GGUF path uses) — no separate `tokenizer.json` needed, unlike
/// the SafeTensors loader in `loader.rs`/`loader_f32.rs`.
///
/// # Safety
///
/// Memory-maps the checkpoint twice: once through `crate::gguf::Content` (for
/// config + tokenizer) and once inside `QuantizedLlama3Model::from_gguf` (for
/// weights) — mirroring `fuel-examples/examples/quantized/main.rs`, which
/// notes the OS shares page cache across the two. Mutating the file while
/// either mapping is alive is undefined behaviour.
pub fn load_quantized_llama_gguf(path: &Path) -> Result<LoadedQuantizedLlama> {
    let content = crate::gguf::Content::read(path)
        .with_context(|| format!("reading GGUF metadata from {}", path.display()))?;

    let tokenizer = content
        .extract_tokenizer()
        .with_context(|| format!("extracting tokenizer from {}", path.display()))?;

    let full = llama_full_config_from_gguf(&content)?;
    if full.eos_token_id.is_none() {
        tracing::warn!(
            "GGUF file {} declares no tokenizer.ggml.eos_token_id; generation will stop \
             only on max_new_tokens",
            path.display()
        );
    }

    let device = super::device::select();
    let model = QuantizedLlama3Model::from_gguf(path, &full).map_err(|e| {
        anyhow::anyhow!(
            "building QuantizedLlama3Model from {}: {e:?}",
            path.display()
        )
    })?;

    Ok(LoadedQuantizedLlama {
        eos: full.eos_token_id.clone(),
        config: full.to_lazy_config(),
        model,
        tokenizer,
        device,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same convention as `tests/gguf_serving_e2e.rs`'s `gguf_path()`:
    /// `LIGHTBULB_GGUF` env override, else the local corpus location this was
    /// developed and last run against.
    fn tinyllama_gguf_path() -> Option<std::path::PathBuf> {
        if let Some(p) = std::env::var_os("LIGHTBULB_GGUF") {
            let p = std::path::PathBuf::from(p);
            return p.is_file().then_some(p);
        }
        let p = std::path::PathBuf::from(
            "C:/Models/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
        );
        p.is_file().then_some(p)
    }

    /// Loads the real TinyLlama Q4_0 GGUF checkpoint through
    /// `QuantizedLlama3Model::from_gguf` and runs one prefill + a few decode
    /// steps, asserting the completion is coherent English (contains
    /// "Paris") rather than just checking for a non-panic — the same
    /// standard `tests/fuel_engine_http.rs` holds itself to and for the same
    /// reason (a wiring bug can still produce plausible-shaped garbage).
    ///
    /// CPU-only: Q4_0 is served by `fuel-quantized`'s CPU kernels
    /// (`k_quants.rs`/`avx.rs`/`neon.rs`), no GPU needed — unlike
    /// `fuel_engine_http.rs`'s f32 SafeTensors path, this one does not wait
    /// on the GPU desktop CireSnave is standing up.
    ///
    /// `#[ignore]`d for two reasons, both dated 2026-09-24 and both run BY
    /// HAND that day against the local corpus path above:
    ///
    /// 1. It needs a 638 MB checkpoint on disk that CI does not have —
    ///    same shape as every other checkpoint-gated test in this crate.
    /// 2. ⚠️ **It currently FAILS, and not from a Lightbulb bug.** The one
    ///    local TinyLlama Q4_0 GGUF file stores `output.weight` as Q6_K
    ///    (llama.cpp's ordinary practice: the output/embedding tensors stay
    ///    at higher precision even in a "Q4_0" file). Fuel's
    ///    `lazy_quantized_llama::dequant_bytes_to_f32`
    ///    (fuel-transformers/src/models/lazy_quantized_llama.rs:497-553,
    ///    checked against fuel's `origin/main` 2026-09-24) wires ONLY
    ///    `F32`/`F16`/`BF16`/`Q4_0` — everything else, including Q6_K,
    ///    hits its catch-all error arm — even though `fuel-quantized`'s
    ///    `k_quants.rs` already implements full K-quant dequant
    ///    (`BlockQ6K: GgmlType`) one layer down. **The capability exists in
    ///    fuel, just not wired into this specific model loader.** Filed
    ///    with the fuel lane the same day this test was added; this repo
    ///    cannot fix it. The wiring above this call (config extraction,
    ///    tokenizer, `FuelDecoder` dispatch) is confirmed reached — the
    ///    failure is exactly at, and only at, fuel's own dequant dispatch.
    ///
    /// **Trigger to re-run:** the fuel lane wires K-quant dequant into
    /// `lazy_quantized_llama`, OR a Q4_0-throughout (no K-quant tensors at
    /// all) GGUF becomes available locally. Either removes this ignore.
    /// Run: `cargo test --features fuel-engine --lib
    /// model_fuel::loader_gguf -- --ignored --nocapture`.
    #[test]
    #[ignore = "needs a 638 MB checkpoint AND currently fails on it: fuel's lazy_quantized_llama doesn't dequant this file's Q6_K output.weight (filed 2026-09-24, not a Lightbulb bug — see doc comment)"]
    fn quantized_llama_gguf_serves_a_coherent_completion() -> anyhow::Result<()> {
        let Some(path) = tinyllama_gguf_path() else {
            panic!(
                "no TinyLlama GGUF checkpoint — this test asserts numerical behaviour, \
                 so it fails rather than skipping"
            );
        };

        let loaded = load_quantized_llama_gguf(&path)?;

        let prompt = "The capital of France is";
        let ids: Vec<u32> = loaded
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?
            .get_ids()
            .to_vec();

        let max_seq_len = ids.len() + 24;
        let mut st =
            super::super::session::SessionState::new(&loaded.config, max_seq_len, &loaded.device)?;

        use crate::model_fuel::decoder::FuelDecoder;
        let mut logits = loaded.model.prefill(&ids, &mut st)?;
        let mut generated = Vec::new();
        for _ in 0..24 {
            let tok = crate::model_fuel::generate::argmax(&logits);
            if loaded.is_eos(tok) {
                break;
            }
            generated.push(tok);
            logits = loaded.model.step(tok, &mut st)?;
        }

        let text = loaded
            .tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow::anyhow!("detokenizing: {e}"))?;
        eprintln!("GGUF completion: {text:?}");
        assert!(
            text.to_lowercase().contains("paris"),
            "expected the continuation to name Paris, got {text:?} — the \
             quantized wiring is producing nonsense"
        );
        Ok(())
    }
}
