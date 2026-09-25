//! One sequence's decode state — and the reason it is a type rather than three
//! locals.
//!
//! # Why this exists
//!
//! To keep the three things one sequence needs — KV cache, inference context,
//! position — as a single unit, and to make `with_capacity` the only way a
//! cache gets built here.
//!
//! `with_capacity`, not `with_dims`, is the load-bearing part. `with_dims`
//! grows the cache by REPLACEMENT, which rebuilds the graph every step and
//! defeats both plan reuse and capture.
//!
//! # A hazard this type no longer has to defend against
//!
//! Fuel originally exposed plan reuse only via
//! `forward_with_kv_context_persistent`, whose `&mut Option<DecodeSession>`
//! fourth argument you could not write unless you already knew `DecodeSession`
//! existed. The call a consumer naturally writes was the slow one — measured on
//! CUDA at **5,901 ms/token** against **26.47 ms/token**, though note that
//! comparison also toggled capture, so it bounds rather than measures either.
//!
//! Fuel diagnosed it as a REACHABILITY defect and shipped `forward_decode_step`
//! (`lazy.rs:8675`), which carries the held plan on the `InferenceContext` and
//! takes the plain `(tokens, cache, ctx)` shape. There is no fourth argument to
//! forget, so this type does not need to own one.

use anyhow::Result;

use fuel::inference_context::{DecodeSession, InferenceContext, KvCache};
use fuel::lazy::LlamaConfig;
use fuel::{DType, Device};

/// One sequence's KV cache, inference context, and position.
///
/// The decode plan is NOT held here for the plain `LlamaModel` path — it rides
/// on the `InferenceContext`, which is what `forward_decode_step` reads and
/// writes. `quantized_session` is the exception: `Llama3Model`/
/// `QuantizedLlama3Model` (fuel-transformers) expose only
/// `forward_with_kv_context_persistent`, which takes the plan explicitly as a
/// `&mut Option<DecodeSession>` fourth argument — the pre-`forward_decode_step`
/// shape this module's own doc comment above describes as a hazard. There is
/// no `forward_decode_step`-equivalent on `Llama3Model` to avoid it with (see
/// `docs/FUEL-PORT-STATUS-2026-09-24.md` §6 and the filed fuel finding), so the
/// quantized `FuelDecoder` impl in `decoder.rs` has to hold one. `None` and
/// unused for the plain `LlamaModel` path.
pub struct SessionState {
    cache: KvCache,
    ctx: InferenceContext,
    position: usize,
    quantized_session: Option<DecodeSession>,
}

impl SessionState {
    /// Allocate a session able to hold `max_seq_len` tokens.
    ///
    /// `with_capacity`, NOT `with_dims` — and this is load-bearing. `with_dims`
    /// grows the cache by REPLACEMENT, rebuilding the graph every step and
    /// defeating both plan reuse and capture. `with_capacity` pre-allocates
    /// `[1, n_kv_heads, max_seq_len, head_dim]` per layer and appends via
    /// `Op::WriteSlice` at a runtime offset, giving a graph that is stable
    /// across steps.
    ///
    /// Fuel enforces this: the persistent forward rejects a `with_dims` cache
    /// outright rather than silently taking the slow path.
    ///
    /// Memory is real and allocated up front:
    /// `n_layers * 2 * n_kv_heads * max_seq_len * head_dim * sizeof(dtype)`.
    /// TinyLlama at 2048 f32 is ~92 MiB.
    pub fn new(config: &LlamaConfig, max_seq_len: usize, device: &Device) -> Result<Self> {
        let cache = KvCache::with_capacity(
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            max_seq_len,
            DType::F32,
            device,
        )
        .map_err(|e| anyhow::anyhow!("allocating KV cache: {e:?}"))?;
        Ok(Self {
            cache,
            ctx: InferenceContext::new(device.clone()),
            position: 0,
            quantized_session: None,
        })
    }

    /// Tokens committed to the cache so far.
    pub fn position(&self) -> usize {
        self.position
    }

    /// The two pieces a forward needs, borrowed together.
    ///
    /// `pub(crate)` and returned as a pair so `decoder.rs` can drive a forward
    /// without the fields becoming public.
    pub(crate) fn parts(&mut self) -> (&mut KvCache, &mut InferenceContext) {
        (&mut self.cache, &mut self.ctx)
    }

    /// The three pieces `forward_with_kv_context_persistent` needs, borrowed
    /// together — the quantized (`QuantizedLlama3Model`) decode path only.
    /// See the struct doc for why this differs from `parts()`.
    pub(crate) fn parts_persistent(
        &mut self,
    ) -> (
        &mut KvCache,
        &mut InferenceContext,
        &mut Option<DecodeSession>,
    ) {
        (&mut self.cache, &mut self.ctx, &mut self.quantized_session)
    }

    /// Record that `n` tokens were committed.
    pub(crate) fn advance(&mut self, n: usize) {
        self.position += n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh session starts at position 0 with an allocated cache.
    ///
    /// Weak by design — the interesting assertions about `SessionState` are
    /// numerical and live in Task 5, which compares logits. This one exists so
    /// `new()` cannot silently start mid-sequence.
    #[test]
    fn new_session_starts_at_position_zero() {
        let cfg = fuel::lazy::LlamaConfig {
            vocab_size: 32,
            dim: 8,
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let dev = crate::model_fuel::device::select();
        let st = SessionState::new(&cfg, 16, &dev).expect("allocating a tiny KV cache");
        assert_eq!(st.position(), 0);
    }
}
