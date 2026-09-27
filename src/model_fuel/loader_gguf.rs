//! Loading a GGUF-quantized Llama-shape checkpoint into Fuel.
//!
//! # Provenance — read before touching this file
//!
//! `fuel::QuantizedLlama3Model::from_gguf` takes a pre-built
//! `LlamaFullConfig` as an argument; it does not derive one from the GGUF
//! file itself. That gap is now filled by
//! `fuel_loaders::quantized::config_from_gguf::derive_config(content,
//! architecture) -> Result<GgufDerivedConfig>` (`fuel#246`), reachable since
//! lightbulb#97's rev bump (PR #100). This file's job is now just the
//! adaptation: `GgufDerivedConfig` deliberately excludes `head_dim`,
//! `bos_token_id`, `eos_token_id`, `rope_scaling` and `tie_word_embeddings`
//! (per its own doc — "Llama-specific config concerns, not GGUF-metadata
//! extraction"), so `derive_llama_full_config` computes `head_dim` locally
//! and reads bos/eos through Lightbulb's own already-tested `crate::gguf`
//! reader, same as before.
//!
//! `require_llama_architecture` still gates this file (hardcoded to the
//! literal architecture `"llama"`) because nothing downstream of it —
//! `QuantizedLlama3Model`, this file's `LlamaFullConfig` construction — can
//! serve any other architecture yet. Widening this to fuel's other
//! `Architecture` variants (`Qwen3` named specifically, board item 71) is
//! lightbulb#97's steps 3-4, deliberately not done here: routing to a
//! different fuel model type per detected architecture is new construction,
//! not a config-source swap, and narrowing the gate before that construction
//! exists would let an unsupported file through to code that cannot build
//! it — worse than the blanket refusal it would replace.

use std::path::Path;

use anyhow::{Context, Result};

use fuel_transformers::models::lazy_llama_full::{LlamaEosToks, LlamaFullConfig};
use fuel_transformers::models::lazy_quantized_llama::QuantizedLlama3Model;

use super::loader::LoadedQuantizedLlama;

/// Build a `LlamaFullConfig` from a GGUF file's own metadata: the eight core
/// fields from fuel's `derive_config`, `head_dim` computed locally, and
/// bos/eos read through Lightbulb's own `crate::gguf` reader (`content` is
/// already open for tokenizer extraction — see the caller).
///
/// See the module doc for why this file still exists at all.
fn derive_llama_full_config(
    content: &crate::gguf::Content,
    path: &Path,
) -> Result<LlamaFullConfig> {
    crate::gguf::require_llama_architecture(content.metadata())?;

    // `derive_config` takes fuel's OWN GGUF reader, not `crate::gguf::Content`
    // — a second, metadata-only parse of the same file (not a second mmap of
    // the weights; `Content::read` reads only the header + tensor-info table).
    let mut file = std::fs::File::open(path).with_context(|| {
        format!(
            "opening {} for fuel's own GGUF metadata read",
            path.display()
        )
    })?;
    let fuel_content = fuel_loaders::quantized::gguf_file::Content::read(&mut file)
        .map_err(|e| anyhow::anyhow!("fuel gguf_file::Content::read({}): {e:?}", path.display()))?;
    let derived = fuel_loaders::quantized::config_from_gguf::derive_config(
        &fuel_content,
        fuel_loaders::quantized::arch::Architecture::Llama,
    )
    .map_err(|e| anyhow::anyhow!("fuel derive_config({}): {e}", path.display()))?;

    let head_dim = derived.hidden_size / derived.n_heads.max(1);
    let (bos_token_id, eos_token_id) = bos_eos_from_metadata(content);

    Ok(LlamaFullConfig {
        hidden_size: derived.hidden_size,
        intermediate_size: derived.intermediate_size,
        vocab_size: derived.vocab_size,
        num_hidden_layers: derived.n_layers,
        num_attention_heads: derived.n_heads,
        num_key_value_heads: derived.n_kv_heads,
        head_dim,
        rms_norm_eps: derived.rms_norm_eps,
        rope_theta: derived.rope_theta,
        max_position_embeddings: derived.max_position_embeddings,
        bos_token_id,
        eos_token_id,
        rope_scaling: None,
        tie_word_embeddings: false,
    })
}

/// Read `tokenizer.ggml.{bos,eos}_token_id` — the two fields `derive_config`
/// deliberately excludes — through Lightbulb's own `crate::gguf` reader.
fn bos_eos_from_metadata(content: &crate::gguf::Content) -> (Option<u32>, Option<LlamaEosToks>) {
    use crate::gguf::Value;

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

    (bos_token_id, eos_token_id)
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
/// Memory-maps or reads the checkpoint's header up to three times: once
/// through `crate::gguf::Content` (tokenizer + bos/eos), once through fuel's
/// own `gguf_file::Content` (the metadata `derive_config` reads), and once
/// inside `QuantizedLlama3Model::from_gguf` (the actual weights) — the OS
/// shares page cache across all three. Mutating the file while any is alive
/// is undefined behaviour.
pub fn load_quantized_llama_gguf(path: &Path) -> Result<LoadedQuantizedLlama> {
    let content = crate::gguf::Content::read(path)
        .with_context(|| format!("reading GGUF metadata from {}", path.display()))?;

    let tokenizer = content
        .extract_tokenizer()
        .with_context(|| format!("extracting tokenizer from {}", path.display()))?;

    let full = derive_llama_full_config(&content, path)?;
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

    /// `derive_llama_full_config` against the real corpus file, before and
    /// after lightbulb#97 step 2's swap to fuel's `derive_config` -- "same
    /// file in, identical `LlamaFullConfig` out", per the PM's own framing:
    /// if any field moved, that is the finding, not a nuisance. Values are
    /// this file's standard TinyLlama-1.1B-Chat architecture numbers, hand-
    /// verified against the file's own metadata (`hidden_size` /
    /// `num_attention_heads` = 2048 / 32 = 64 = `head_dim`) rather than
    /// copied from the deleted hand-rolled parser's output blind.
    #[test]
    fn derive_llama_full_config_matches_the_pre_swap_baseline() {
        let Some(path) = tinyllama_gguf_path() else {
            eprintln!("no TinyLlama GGUF on disk; skipping (see tinyllama_gguf_path)");
            return;
        };
        let content = crate::gguf::Content::read(&path).expect("reading GGUF metadata");
        let full = derive_llama_full_config(&content, &path).expect("deriving LlamaFullConfig");

        assert_eq!(full.hidden_size, 2048);
        assert_eq!(full.intermediate_size, 5632);
        assert_eq!(full.vocab_size, 32000);
        assert_eq!(full.num_hidden_layers, 22);
        assert_eq!(full.num_attention_heads, 32);
        assert_eq!(full.num_key_value_heads, 4);
        assert_eq!(full.head_dim, 64);
        assert!((full.rms_norm_eps - 1e-5).abs() < 1e-9);
        assert_eq!(full.rope_theta, 10000.0);
        assert_eq!(full.max_position_embeddings, 2048);
        assert_eq!(full.bos_token_id, Some(1));
        assert!(matches!(full.eos_token_id, Some(ref e) if e.is_eos(2)));
        assert_eq!(full.rope_scaling, None);
        assert!(!full.tie_word_embeddings);
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
