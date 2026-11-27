// SPDX-License-Identifier: MIT OR Apache-2.0
//! F3 slice 2: an [`Embedder`] over Fuel's `lazy_nomic_bert` (`nomic-ai/nomic-embed-text-v1.5`, safetensors).
//!
//! One sequence per call (Fuel has no padded batch, `docs/F3-EMBEDDINGS-DESIGN.md`), so there is no padding and
//! no attention mask: mean pooling is a plain mean over the sequence. The task prefixes (`search_document:`,
//! `search_query:`) are the CLIENT's to send; this module never adds one.
//!
//! NOT yet proven against reference vectors: that gate needs the checkpoint and sentence-transformers output
//! (ROADMAP F3). Until it exists, the pooling arithmetic and the spec row are tested; the model forward is not.

use std::path::Path;

use fuel_transformers::models::lazy_nomic_bert::{
    NomicBertConfig, NomicBertModel, NomicBertWeights,
};
use tokenizers::Tokenizer;

use super::embeddings::{
    Embedded, Embedder, EmbeddingError, EmbeddingSpec, Pooling, bad, internal,
};

/// The pooling-table row for `nomic-embed-text-v1.5` (design note: mean, normalised, Matryoshka widths
/// 768/512/256/128/64). `max_tokens` is the model's `n_positions`; `max_items` is the per-request array cap.
pub fn nomic_v1_5_spec() -> EmbeddingSpec {
    let cfg = NomicBertConfig::nomic_embed_text_v1_5();
    EmbeddingSpec {
        pooling: Pooling::Mean,
        normalize: true,
        dim: cfg.n_embd,
        matryoshka: vec![512, 256, 128, 64],
        max_tokens: cfg.n_positions,
        max_items: 128,
    }
}

/// Pool a row-major `[seq, dim]` hidden-state block into one `dim` vector. Not normalised.
pub fn pool(
    hidden: &[f32],
    seq: usize,
    dim: usize,
    pooling: Pooling,
) -> Result<Vec<f32>, EmbeddingError> {
    if seq == 0 || dim == 0 || hidden.len() != seq * dim {
        return Err(internal(format!(
            "hidden states hold {} values, expected {seq} x {dim}",
            hidden.len()
        )));
    }
    let row = |i: usize| &hidden[i * dim..(i + 1) * dim];
    Ok(match pooling {
        Pooling::Cls => row(0).to_vec(),
        Pooling::LastToken => row(seq - 1).to_vec(),
        Pooling::Mean => {
            let mut out = vec![0.0f32; dim];
            for i in 0..seq {
                for (o, v) in out.iter_mut().zip(row(i)) {
                    *o += v;
                }
            }
            out.iter_mut().for_each(|o| *o /= seq as f32);
            out
        }
    })
}

pub struct NomicEmbedder {
    model: NomicBertModel,
    tokenizer: Tokenizer,
    spec: EmbeddingSpec,
}

impl NomicEmbedder {
    /// Load from a Hugging Face directory holding `model.safetensors` and `tokenizer.json`.
    pub fn load(dir: &Path) -> anyhow::Result<Self> {
        let cfg = NomicBertConfig::nomic_embed_text_v1_5();
        let weights_path = dir.join("model.safetensors");
        // SAFETY: the file is opened read-only and is not modified while mapped; the same contract as
        // every other safetensors load in `model_fuel`.
        let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&[weights_path]) }
            .map_err(|e| anyhow::anyhow!("mapping model.safetensors: {e:?}"))?;
        let weights = NomicBertWeights::load_from_mmapped(&st, &cfg)
            .map_err(|e| anyhow::anyhow!("loading nomic-bert weights: {e:?}"))?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("loading tokenizer.json: {e}"))?;
        Ok(Self {
            model: NomicBertModel {
                config: cfg,
                weights,
            },
            tokenizer,
            spec: nomic_v1_5_spec(),
        })
    }

    pub fn spec(&self) -> &EmbeddingSpec {
        &self.spec
    }
}

impl NomicEmbedder {
    /// Per-token hidden states of `text`, row-major `[seq, dim]`, and `seq`: the encoder output before pooling
    /// (public so a test can pool it a second way as a control).
    pub fn hidden_states(&self, text: &str) -> Result<(Vec<f32>, usize), EmbeddingError> {
        let ids = self.token_ids(text)?;
        // `forward` asserts a non-empty sequence and errors past `n_positions`; both are refused here
        // as typed errors (a tokenizer.json that adds no special tokens can yield an empty row).
        if ids.is_empty() {
            return Err(bad("input", "the tokenizer produced no tokens"));
        }
        if ids.len() > self.spec.max_tokens {
            return Err(bad(
                "input",
                format!(
                    "{} tokens is over the model's window of {}",
                    ids.len(),
                    self.spec.max_tokens
                ),
            ));
        }
        let hidden = self
            .model
            .forward(&ids, None, None)
            .map_err(|e| internal(format!("nomic-bert forward: {e:?}")))?
            .realize_f32();
        Ok((hidden, ids.len()))
    }

    fn token_ids(&self, text: &str) -> Result<Vec<u32>, EmbeddingError> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| bad("input", format!("tokenizing: {e}")))?;
        Ok(enc.get_ids().to_vec())
    }
}

impl Embedder for NomicEmbedder {
    fn count_tokens(&self, text: &str) -> Result<usize, EmbeddingError> {
        Ok(self.token_ids(text)?.len())
    }

    fn embed(&self, text: &str) -> Result<Embedded, EmbeddingError> {
        let (hidden, seq) = self.hidden_states(text)?;
        let vector = pool(&hidden, seq, self.spec.dim, self.spec.pooling)?;
        Ok(Embedded {
            vector,
            tokens: seq,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Two tokens of width 3: row 0 = [1,2,3], row 1 = [3,4,9].
    const H: [f32; 6] = [1.0, 2.0, 3.0, 3.0, 4.0, 9.0];

    #[test]
    fn mean_cls_and_last_token_differ_on_the_same_block() {
        let mean = pool(&H, 2, 3, Pooling::Mean).unwrap();
        let cls = pool(&H, 2, 3, Pooling::Cls).unwrap();
        let last = pool(&H, 2, 3, Pooling::LastToken).unwrap();
        assert_eq!(mean, vec![2.0, 3.0, 6.0]);
        assert_eq!(cls, vec![1.0, 2.0, 3.0]);
        assert_eq!(last, vec![3.0, 4.0, 9.0]);
        // The control the design note asks for: swapping mean for CLS must be visible.
        assert_ne!(mean, cls);
    }

    #[test]
    fn a_misshapen_block_is_an_error_not_a_panic() {
        assert!(pool(&H, 2, 4, Pooling::Mean).is_err());
        assert!(pool(&[], 0, 3, Pooling::Mean).is_err());
        assert!(pool(&H, 2, 0, Pooling::Mean).is_err());
    }

    #[test]
    fn the_spec_row_matches_the_published_model() {
        let s = nomic_v1_5_spec();
        assert_eq!((s.dim, s.max_tokens), (768, 8192));
        assert_eq!(s.pooling, Pooling::Mean);
        assert!(s.normalize);
        assert_eq!(s.matryoshka, vec![512, 256, 128, 64]);
        // Every Matryoshka width is narrower than the native one.
        assert!(s.matryoshka.iter().all(|&d| d < s.dim));
    }
}
