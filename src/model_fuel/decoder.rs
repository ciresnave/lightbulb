//! Which Fuel model type decodes — abstracted, so a second one can arrive.
//!
//! # Why a trait for a single implementor
//!
//! Because the second implementor exists and is next.
//!
//! GGUF/quantized Llama is `QuantizedLlama3Model::from_gguf`, which wraps
//! `Llama3Model`. When this plan was written `Llama3Model` had no `KvCache` and
//! no `forward_with_kv_context*` at all, so serving through it would have
//! re-run the whole prefix per token. Lightbulb filed that gap; Fuel landed
//! `Llama3Model::forward_with_kv_context_persistent`
//! (`lazy_llama_full.rs:382`) the same day, and the quantized wrapper inherits
//! it through its `inner`.
//!
//! So quantized support is now `impl FuelDecoder for Llama3Model` and nothing
//! else — no change to the session, the engine model, or the runner. It is
//! deliberately NOT in this plan: the runner should be proven end-to-end on one
//! model path before a second one is debugged alongside it.
//!
//! **Two things to know when that impl is written.** Fuel's paged tier does not
//! yet thread the LLaMA-3.1 frequency override, so `PagedDecodeModel` is a
//! separate supertrait and handing a `Llama3Model` to `PagedSessionScheduler` is
//! a compile error rather than a silent wrong answer. And a scaled-vs-unscaled
//! parity test on a short prompt proves nothing: the RoPE band edges sit at
//! `original_max_position_embeddings / *_freq_factor` = 8192, and at positions
//! 0..6 the two differ by 1.1e-7.

use anyhow::Result;

use fuel::lazy::LlamaModel;

use super::session::SessionState;

/// A model that can prefill a prompt and then decode one token at a time
/// against a held [`SessionState`].
pub trait FuelDecoder {
    /// Consume the whole prompt in one forward. Returns the final logit row.
    fn prefill(&self, tokens: &[u32], st: &mut SessionState) -> Result<Vec<f32>>;

    /// Consume one token. Returns the resulting logit row.
    fn step(&self, token: u32, st: &mut SessionState) -> Result<Vec<f32>>;
}

impl FuelDecoder for LlamaModel {
    fn prefill(&self, tokens: &[u32], st: &mut SessionState) -> Result<Vec<f32>> {
        // `forward_decode_step` for BOTH prefill and decode, deliberately.
        // At `seq != 1` it falls back to the rebuild path without building a
        // plan; the first `seq == 1` token builds it; later tokens rebind and
        // skip optimize. Output is byte-identical either way, so there is no
        // reason for the two arms to call different entry points — and one
        // entry point is one fewer place to get the fast path wrong.
        let (cache, ctx) = st.parts();
        let logits = self
            .forward_decode_step(tokens, cache, ctx)
            .map_err(|e| anyhow::anyhow!("prefill forward: {e:?}"))?;
        st.advance(tokens.len());
        Ok(logits)
    }

    fn step(&self, token: u32, st: &mut SessionState) -> Result<Vec<f32>> {
        let (cache, ctx) = st.parts();
        let logits = self
            .forward_decode_step(&[token], cache, ctx)
            .map_err(|e| anyhow::anyhow!("decode forward: {e:?}"))?;
        st.advance(1);
        Ok(logits)
    }
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

    /// Stepping through `FuelDecoder` produces bit-identical logits to the
    /// original `generate_greedy` loop.
    ///
    /// **Asserted on logits with maxdiff == 0, deliberately not on tokens.**
    /// This test's sole purpose is detecting numerical drift, and argmax is the
    /// operation that hides it — a token comparison would pass whether or not
    /// the decode maths changed, which is evidence carrying no information.
    #[test]
    #[ignore = "needs the TinyLlama checkpoint; slow on CPU"]
    fn stepwise_logits_match_the_reference_loop_exactly() -> anyhow::Result<()> {
        let Some(dir) = tinyllama_dir() else {
            panic!("no TinyLlama snapshot — this test asserts numerical behaviour, so it fails rather than skipping");
        };
        let loaded = crate::model_fuel::loader_f32::load_llama_f32_from_dir(&dir)?;
        let prompt: Vec<u32> = vec![1, 450, 7483, 310, 3444, 338];

        // Reference: the shape generate_greedy uses — one persistent session
        // across prefill and every decode step.
        let mut ref_st =
            SessionState::new(&loaded.config, prompt.len() + 4, &loaded.device)?;
        let mut reference = loaded.model.prefill(&prompt, &mut ref_st)?;
        for _ in 0..3 {
            let next = crate::model_fuel::generate::argmax(&reference);
            reference = loaded.model.step(next, &mut ref_st)?;
        }

        // Under test: an independently constructed session doing the same.
        let mut st = SessionState::new(&loaded.config, prompt.len() + 4, &loaded.device)?;
        let mut got = loaded.model.prefill(&prompt, &mut st)?;
        for _ in 0..3 {
            let next = crate::model_fuel::generate::argmax(&got);
            got = loaded.model.step(next, &mut st)?;
        }

        assert_eq!(got.len(), reference.len(), "logit row width changed");
        let maxdiff = got
            .iter()
            .zip(reference.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(
            maxdiff, 0.0,
            "logits drifted by {maxdiff:e} — decode is not reproducible step-for-step"
        );
        assert_eq!(st.position(), ref_st.position(), "position accounting diverged");
        Ok(())
    }
}
