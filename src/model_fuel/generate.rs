//! The decode loop — Lightbulb driving Fuel, one token at a time.
//!
//! # Why Lightbulb owns this loop
//!
//! Fuel ships `generate_streaming_with_kv_context`, which takes a
//! `SamplingStrategy` and runs the whole generation itself. `llama-lazy` uses it.
//! **Lightbulb deliberately does not.**
//!
//! Per Fuel's own consumer contract (§15), *"selection among different math —
//! which optimizer, which sampling strategy"* is the **consumer's**, and Fuel's
//! `DecodeModel` trait pointedly leaves `SamplingStrategy` out for that reason.
//! Lightbulb has its own sampler (`crate::sampling`) and its own constrained
//! generation (`crate::contracts`) layered on top; a Fuel-driven loop would take
//! both out of the picture and leave the engine unable to schedule, batch, or
//! apply an output contract.
//!
//! So the split this module demonstrates is the one the port is built on:
//!
//! | | |
//! | --- | --- |
//! | **Fuel** | the forward pass, the KV cache, the graph, the arm choice |
//! | **Lightbulb** | which token to pick, when to stop, whose request runs |
//!
//! `forward_with_kv_context` returns **`Vec<f32>`** — already realized. That is
//! the seam: everything upstream of it is a lazy graph Fuel optimizes, everything
//! downstream is ordinary host code that Lightbulb owns.
//!
//! # Status
//!
//! Greedy only, single sequence, no batching. This is the vertical slice that
//! proves the path end to end; the engine's real scheduling, batching and
//! sampling policy are ported on top of it, not into it.

use anyhow::Result;

use fuel::inference_context::{InferenceContext, KvCache};
use fuel::Device;

use super::loader::LoadedLlama;

/// Greedy argmax over a realized logits row.
///
/// Deliberately written here rather than reaching for `crate::sampling`: this is
/// the slice's proof that token selection happens on **realized host values**
/// under Lightbulb's control. The real sampler — temperature, top-k, top-p, the
/// seeded RNG, and `contracts`' constrained generation — replaces this, and the
/// replacement is a host-side change that Fuel never sees.
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// Prefill `prompt_tokens`, then greedily decode up to `max_new` tokens.
///
/// Stops early on `eos`, if given. Returns only the newly generated tokens.
///
/// The loop shape matters and is the one the port keeps: **prefill is one call
/// with the whole prompt; each decode step is one call with a single token**,
/// mutating the same `KvCache` and `InferenceContext` in place. That is what
/// makes the step capture-shaped — the graph is stable across steps, so nothing
/// here rebuilds it per token.
pub fn generate_greedy(
    loaded: &LoadedLlama,
    prompt_tokens: &[u32],
    max_new: usize,
    eos: Option<u32>,
) -> Result<Vec<u32>> {
    // The KV cache is pre-allocated, so its capacity is fixed up front and the
    // caller's budget has to cover prompt + generation.
    let max_seq_len = prompt_tokens.len() + max_new + 1;
    let c = &loaded.config;
    let device = Device::cpu();

    // `with_capacity`, NOT `with_dims` — and this is a rule-3 decision, not a
    // detail. `with_dims` grows the cache by REPLACEMENT, which rebuilds the
    // graph every step and defeats capture. `with_capacity` pre-allocates
    // `[1, n_kv_heads, max_seq_len, head_dim]` per layer and appends via
    // `Op::WriteSlice` at a runtime offset — a stable graph across steps, which
    // is exactly what `CapturedRun` requires.
    //
    // Fuel enforces this: `forward_with_kv_context` rejects a `with_dims` cache
    // outright rather than silently taking the slow path. Good error, and it
    // means the capture-shape decision is made at construction and cannot be
    // quietly lost later.
    //
    // Cost is real and worth stating: n_layers * 2 * n_kv_heads * max_seq_len *
    // head_dim * dtype_size. For TinyLlama at 2048 f32 that is
    // 22 * 2 * 4 * 2048 * 64 * 4 ≈ 92 MiB, allocated up front.
    let mut cache = KvCache::with_capacity(
        c.n_layers,
        c.n_kv_heads,
        c.head_dim,
        max_seq_len,
        fuel::DType::F32,
        &device,
    )
    .map_err(|e| anyhow::anyhow!("allocating KV cache: {e:?}"))?;
    let mut ctx = InferenceContext::new(device);

    // `forward_with_kv_context_PERSISTENT`, not the plain variant — and this is
    // rule 3 in practice, not a micro-optimisation.
    //
    // The plain `forward_with_kv_context` re-optimises the graph on EVERY token.
    // The persistent one optimises ONCE: the first call builds and caches an
    // `OptimizedGraph` plus stable NodeIds into `session`; each later token
    // rewrites the per-token host bytes (token id, RoPE tables at the new
    // position, the shifted mask boundary) into the held device Arcs and calls
    // `realize_prebuilt_as_with_env`, which SKIPS optimize. The KV Arcs are bound
    // once and mutate in place via `Op::WriteSlice`, and the data `Const`s persist
    // across tokens rather than being re-emitted.
    //
    // Measured: 8.92s/token on the plain path vs ~3.55s/token for Fuel's own
    // reference generate — a ~2.5x gap that was entirely per-token
    // re-optimisation. Fuel documents the persistent path as byte-identical on
    // the same prefix, so this is free.
    //
    // `session` is what makes it plan-once; it must live across the whole loop.
    let mut session: Option<fuel::inference_context::DecodeSession> = None;

    // Prefill: the whole prompt in one forward.
    let mut logits = loaded
        .model
        .forward_with_kv_context_persistent(prompt_tokens, &mut cache, &mut ctx, &mut session)
        .map_err(|e| anyhow::anyhow!("prefill forward: {e:?}"))?;

    let mut generated = Vec::with_capacity(max_new);
    for _ in 0..max_new {
        // Sampling: Lightbulb's, over realized values.
        let next = argmax(&logits);
        generated.push(next);
        if Some(next) == eos {
            break;
        }
        // Decode: one token, same cache and context, mutated in place.
        logits = loaded
            .model
            .forward_with_kv_context_persistent(&[next], &mut cache, &mut ctx, &mut session)
            .map_err(|e| anyhow::anyhow!("decode forward: {e:?}"))?;
    }

    Ok(generated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tinyllama_dir() -> Option<PathBuf> {
        let p = PathBuf::from(
            "C:/Users/cires/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        );
        p.join("model.safetensors").is_file().then_some(p)
    }

    /// **The vertical slice**: Lightbulb loads a real checkpoint into Fuel,
    /// prefills, decodes with its own sampler, and detokenizes — end to end,
    /// inside Lightbulb's own crate.
    ///
    /// Asserts the output is *coherent English continuing the prompt*, not
    /// merely that nothing panicked. A model wired up wrongly — a transposed
    /// projection, a mis-set RoPE base — still produces tokens; it just produces
    /// nonsense. Only reading the text catches that.
    ///
    /// `#[ignore]`: needs a 2.2 GB checkpoint and is slow on CPU.
    ///
    /// **Run it in release.** Measured progression on this machine, 6 tokens
    /// from a 5-token prompt, CPU:
    ///
    /// | build | forward | s/token |
    /// | --- | --- | --- |
    /// | debug | plain | 87.9 |
    /// | release | plain | 8.92 |
    /// | release | **persistent** | **4.52** |
    ///
    /// The debug→release step is ~10× and is the build profile, not Fuel —
    /// numeric code is where debug overhead is worst, so a debug timing is not a
    /// performance signal. The plain→persistent step is a further ~2× and was a
    /// real bug in this file: the plain forward re-optimises the graph every
    /// token.
    ///
    /// Fuel's own `llama-lazy` reference measures ~3.55s/token over 8 tokens.
    /// The remaining gap is largely prefill amortised over fewer tokens here, so
    /// these are close rather than identical — not a claim of parity.
    ///
    /// Run: `cargo test --release --lib model_fuel -- --ignored --nocapture`
    #[test]
    #[ignore = "needs the TinyLlama checkpoint; ~3.5s/token on CPU"]
    fn generates_coherent_text_end_to_end() -> Result<()> {
        let Some(dir) = tinyllama_dir() else {
            eprintln!("skipping: no TinyLlama snapshot");
            return Ok(());
        };

        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

        let loaded = super::super::loader::load_llama_from_dir(&dir)?;

        let prompt = "The capital of France is";
        let prompt_ids: Vec<u32> = tok
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?
            .get_ids()
            .to_vec();

        let started = std::time::Instant::now();
        let generated = generate_greedy(&loaded, &prompt_ids, 6, Some(2))?;
        let elapsed = started.elapsed();

        let text = tok
            .decode(&generated, true)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))?;

        eprintln!("prompt:    {prompt:?}");
        eprintln!("generated: {text:?}");
        eprintln!(
            "{} tokens in {:.1?} ({:.2?}/token)",
            generated.len(),
            elapsed,
            elapsed / generated.len().max(1) as u32
        );

        assert!(!generated.is_empty(), "no tokens generated");
        // Greedy on this prompt should reach "Paris". Asserting on content is
        // the point — token count alone would pass on garbage.
        assert!(
            text.to_lowercase().contains("paris"),
            "expected the continuation to name Paris, got {text:?} — the model \
             runs but is producing nonsense, which points at the wiring \
             (projection layout, RoPE base, norm placement) rather than the plumbing"
        );
        Ok(())
    }
}
