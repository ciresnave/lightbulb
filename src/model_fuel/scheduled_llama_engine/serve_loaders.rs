// SPDX-License-Identifier: MIT OR Apache-2.0
//! Per-format model loading and serving entry points for the scheduled engine.

use super::*;

/// The `.gguf` branch of [`run_fuel_engine`]: reads the file's own
/// `general.architecture` declaration FIRST — through a throwaway
/// `crate::gguf::Content` read, same pattern the loaders' own "up to three
/// times" safety note already accepts — and picks the loader that actually
/// matches, instead of hardcoding Llama and discovering a Qwen3 checkpoint's
/// mismatch as a refusal deep inside `require_llama_architecture`.
pub(super) fn serve_gguf(
    model_path: &Path,
    context_length: usize,
    config: BatchedServeConfig,
    rx: Receiver<InferenceJob>,
) {
    let content = match crate::gguf::Content::read(model_path) {
        Ok(c) => c,
        Err(e) => return load_failed(model_path, e, rx),
    };
    match detect_architecture(content.metadata()) {
        Ok(GgufArchitecture::Llama) => serve_gguf_llama(model_path, context_length, config, rx),
        Ok(GgufArchitecture::Qwen3) => serve_gguf_qwen3(model_path, context_length, config, rx),
        Ok(GgufArchitecture::Qwen35) => {
            load_failed(model_path, qwen35_unserved(content.metadata()), rx)
        }
        Err(e) => load_failed(model_path, e, rx),
    }
}

/// The Llama-architecture half of [`serve_gguf`] — split out so that
/// function stays short; see its own doc for the load/route/batch shape both
/// architecture branches repeat.
pub(super) fn serve_gguf_llama(
    model_path: &Path,
    context_length: usize,
    config: BatchedServeConfig,
    rx: Receiver<InferenceJob>,
) {
    let loaded = match super::super::loader_gguf::load_quantized_llama_gguf(model_path) {
        Ok(l) => l,
        Err(e) => return load_failed(model_path, e, rx),
    };
    println!("Fuel model loaded at {}", model_path.display());
    let eos_ids = eos_ids_from(loaded.eos.as_ref());
    run_batched(
        &loaded.model,
        config,
        eos_ids,
        context_length,
        loaded.tokenizer,
        rx,
        ServingDriverInner::QuantizedGguf,
    );
}

/// The Qwen3-architecture half of [`serve_gguf`] — see [`serve_gguf_llama`].
pub(super) fn serve_gguf_qwen3(
    model_path: &Path,
    context_length: usize,
    config: BatchedServeConfig,
    rx: Receiver<InferenceJob>,
) {
    let loaded = match super::super::loader_gguf_qwen3::load_quantized_qwen3_gguf(model_path) {
        Ok(l) => l,
        Err(e) => return load_failed(model_path, e, rx),
    };
    println!("Fuel model loaded at {}", model_path.display());
    let eos_ids = eos_ids_from(loaded.eos.as_ref());
    run_batched(
        &loaded.model,
        config,
        eos_ids,
        context_length,
        loaded.tokenizer,
        rx,
        ServingDriverInner::QuantizedGgufQwen3,
    );
}

/// The SafeTensors-directory branch of [`run_fuel_engine`] — see [`serve_gguf`].
pub(super) fn serve_f32(
    model_path: &Path,
    context_length: usize,
    config: BatchedServeConfig,
    rx: Receiver<InferenceJob>,
) {
    let loaded = match super::super::loader_f32::load_llama_f32_from_dir(model_path) {
        Ok(l) => l,
        Err(e) => return load_failed(model_path, e, rx),
    };
    println!("Fuel model loaded at {}", model_path.display());
    let eos_ids = eos_ids_from(loaded.eos.as_ref());
    run_batched(
        &loaded.model,
        config,
        eos_ids,
        context_length,
        loaded.tokenizer,
        rx,
        ServingDriverInner::F32,
    );
}

/// Why a `qwen35` GGUF is recognised but not served: a specific refusal rather
/// than the generic unsupported-architecture one, so the reader is not sent
/// looking for a defect in a file that is fine.
const QWEN35_UNSERVED: &str = "this GGUF declares `general.architecture = \"qwen35\"` \
    (Qwen3.5/3.8: Gated DeltaNet layers interleaved with full attention). Lightbulb \
    recognises it and its tokenizer is supported, but serving it needs a hybrid decode \
    model with per-session recurrent state and K-quants kept quantized, which Fuel does \
    not provide yet (ROADMAP F10).";

/// [`QWEN35_UNSERVED`] plus the file's own geometry, or why it could not be read.
pub(super) fn qwen35_unserved(
    md: &std::collections::HashMap<String, crate::gguf::Value>,
) -> anyhow::Error {
    match super::super::qwen35_config::Qwen35Config::from_metadata(md) {
        Ok(c) => anyhow::anyhow!("{QWEN35_UNSERVED} This file: {}.", c.describe()),
        Err(e) => anyhow::anyhow!("{QWEN35_UNSERVED} Its geometry could not be read: {e:#}."),
    }
}

/// Report a load failure the same way both branches of [`run_fuel_engine`]
/// need to, then drain `rx` so no job is left waiting on a model that will
/// never load.
pub(super) fn load_failed(model_path: &Path, e: anyhow::Error, rx: Receiver<InferenceJob>) {
    eprintln!(
        "Failed to load Fuel model at {}: {e:#}",
        model_path.display()
    );
    drain_with_error(rx, &format!("model load failed: {e:#}"));
}
