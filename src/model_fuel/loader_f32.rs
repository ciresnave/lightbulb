//! All-f32 weight loading — the parity oracle's dtype.
//!
//! # Why this exists separately
//!
//! Fuel's `LlamaWeights::load_from_mmapped` loads embeddings and norm gains as
//! f32 but **deliberately preserves bf16 for the projection matrices** — its own
//! comment: *"bf16 source files stay bf16 on-device (halving weight memory on
//! this layer)."* That is a sensible default and the wrong one for us.
//!
//! With bf16 projections, every matmul has key `[F32, BF16, F32]`, which no CPU
//! kernel serves. Fuel's optimizer inserts a promoting cast — **155 of them per
//! realize on TinyLlama-1.1B**, measured by Fuel. That cast is *value-lossless
//! but not accumulation-preserving*: the promoted op accumulates in higher
//! precision, so the same graph produces different numbers depending on which
//! arm ran. Fine for serving. **Fatal for an oracle**, whose entire job is to be
//! the fixed point everything else is compared against.
//!
//! Loading f32 keeps the key at `[F32, F32, F32]` — natively supported, **no
//! fixup runs**, and goldens are captured on a path the dtype pass never touches.
//!
//! `WeightStorage` has no dtype conversion, so this is a separate loader rather
//! than a post-pass over the stock one.
//!
//! # The cost, stated
//!
//! f32 projections are **2× the weight memory** of bf16. For TinyLlama-1.1B that
//! is roughly 4.4 GB against 2.2 GB. That is the price of an oracle that isn't
//! subject to arm-dependent numerics, and it is why this is the *oracle's*
//! loader and not the serving default.

use anyhow::{Context, Result};
use std::path::Path;
use std::sync::Arc;

use fuel::lazy::{
    load_tensor_as_f32, load_transposed_matrix, LayerWeights, LlamaConfig, LlamaModel,
    LlamaWeights, WeightStorage,
};
use fuel::lazy_llama2c::Llama2cConfig;
use fuel::safetensors::MmapedSafetensors;

use super::loader::LoadedLlama;

/// Load a Llama-shape checkpoint with **every** weight as f32.
///
/// Same directory layout as [`super::loader::load_llama_from_dir`]: `config.json`
/// plus a single `model.safetensors`.
///
/// # Safety
///
/// Memory-maps the checkpoint; mutating the file while this is alive is UB.
pub fn load_llama_f32_from_dir(dir: &Path) -> Result<LoadedLlama> {
    let config_str = std::fs::read_to_string(dir.join("config.json"))
        .with_context(|| format!("reading config.json in {}", dir.display()))?;
    let config: LlamaConfig = Llama2cConfig::from_hf_json_str(&config_str)
        .map_err(|e| anyhow::anyhow!("parsing config.json: {e:?}"))?
        .to_llama_config();

    let weights_path = dir.join("model.safetensors");
    if !weights_path.is_file() {
        anyhow::bail!("no model.safetensors in {}", dir.display());
    }
    let st = unsafe { MmapedSafetensors::multi(&[weights_path]) }
        .map_err(|e| anyhow::anyhow!("mmapping safetensors: {e:?}"))?;

    let dim = config.dim;
    let ffn_dim = config.ffn_dim;
    let kv_dim = config.n_kv_heads * config.head_dim;

    // f32 helper: HF stores linear weights [out, in]; Fuel wants [in, out] for
    // `x @ W`. `load_transposed_matrix` does both the f32 conversion and the
    // transpose, which is exactly the pair we need.
    let mat = |name: &str, out: usize, inp: usize| -> Result<WeightStorage> {
        let v = load_transposed_matrix(&st, name, out, inp)
            .map_err(|e| anyhow::anyhow!("loading {name}: {e:?}"))?;
        Ok(WeightStorage::F32(Arc::from(v)))
    };
    let vecf = |name: &str| -> Result<Arc<[f32]>> {
        let v = load_tensor_as_f32(&st, name)
            .map_err(|e| anyhow::anyhow!("loading {name}: {e:?}"))?;
        Ok(Arc::from(v))
    };

    let token_embedding = vecf("model.embed_tokens.weight")?;

    let mut layers = Vec::with_capacity(config.n_layers);
    for i in 0..config.n_layers {
        layers.push(LayerWeights {
            attn_q: mat(&format!("model.layers.{i}.self_attn.q_proj.weight"), dim, dim)?,
            attn_q_bias: None,
            attn_k: mat(&format!("model.layers.{i}.self_attn.k_proj.weight"), kv_dim, dim)?,
            attn_k_bias: None,
            attn_v: mat(&format!("model.layers.{i}.self_attn.v_proj.weight"), kv_dim, dim)?,
            attn_v_bias: None,
            attn_o: mat(&format!("model.layers.{i}.self_attn.o_proj.weight"), dim, dim)?,
            ffn_gate: mat(&format!("model.layers.{i}.mlp.gate_proj.weight"), ffn_dim, dim)?,
            ffn_up: mat(&format!("model.layers.{i}.mlp.up_proj.weight"), ffn_dim, dim)?,
            ffn_down: mat(&format!("model.layers.{i}.mlp.down_proj.weight"), dim, ffn_dim)?,
            attn_norm_gain: vecf(&format!("model.layers.{i}.input_layernorm.weight"))?,
            ffn_norm_gain: vecf(&format!(
                "model.layers.{i}.post_attention_layernorm.weight"
            ))?,
        });
    }

    let final_norm_gain = vecf("model.norm.weight")?;

    // `lm_head` is absent when the checkpoint ties the output head to the
    // embedding table. Handled explicitly rather than by an unwrap: TinyLlama
    // ships an untied head, but Llama-2-7B and many small models do not, and a
    // missing-tensor panic would be a confusing way to discover that.
    let output = match mat("lm_head.weight", config.vocab_size, dim) {
        Ok(w) => w,
        Err(_) => {
            // Tied: the head is the embedding table transposed, [vocab, dim] →
            // [dim, vocab].
            let mut t = vec![0f32; config.vocab_size * dim];
            for v in 0..config.vocab_size {
                for d in 0..dim {
                    t[d * config.vocab_size + v] = token_embedding[v * dim + d];
                }
            }
            WeightStorage::F32(Arc::from(t))
        }
    };

    let weights = LlamaWeights {
        token_embedding,
        layers,
        final_norm_gain,
        output,
    };

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
    use std::path::PathBuf;

    fn tinyllama_dir() -> Option<PathBuf> {
        let p = PathBuf::from(
            "C:/Users/cires/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        );
        p.join("model.safetensors").is_file().then_some(p)
    }

    /// Every projection must be f32 — that is the whole point of this loader,
    /// and it is checkable directly rather than inferred from the absence of
    /// promotion warnings.
    #[test]
    #[ignore = "needs the TinyLlama checkpoint; loads ~4.4 GB as f32"]
    fn every_weight_is_f32() -> Result<()> {
        let Some(dir) = tinyllama_dir() else {
            eprintln!("skipping: no TinyLlama snapshot");
            return Ok(());
        };

        let loaded = load_llama_f32_from_dir(&dir)?;
        let w = &loaded.model.weights;

        assert_eq!(w.layers.len(), loaded.config.n_layers);
        for (i, l) in w.layers.iter().enumerate() {
            for (name, ws) in [
                ("attn_q", &l.attn_q),
                ("attn_k", &l.attn_k),
                ("attn_v", &l.attn_v),
                ("attn_o", &l.attn_o),
                ("ffn_gate", &l.ffn_gate),
                ("ffn_up", &l.ffn_up),
                ("ffn_down", &l.ffn_down),
            ] {
                assert!(
                    matches!(ws, WeightStorage::F32(_)),
                    "layer {i} {name} is {:?}, not F32 — this loader's only job is \
                     keeping the matmul key at [F32,F32,F32] so no promoting cast runs",
                    ws.dtype()
                );
            }
        }
        assert!(
            matches!(w.output, WeightStorage::F32(_)),
            "lm_head is not F32"
        );

        eprintln!(
            "all-f32: {} layers x 7 projections + lm_head",
            w.layers.len()
        );
        Ok(())
    }

    /// The f32 model must generate the same text as the stock (bf16-projection)
    /// loader. Different dtype, same answer — if these disagree, one of the two
    /// load paths is wrong, and this says which case to investigate rather than
    /// leaving it to be discovered as bad output later.
    #[test]
    #[ignore = "needs the checkpoint; runs two full generations on CPU"]
    fn f32_and_stock_loaders_agree() -> Result<()> {
        let Some(dir) = tinyllama_dir() else {
            eprintln!("skipping: no TinyLlama snapshot");
            return Ok(());
        };
        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let ids: Vec<u32> = tok
            .encode("The capital of France is", true)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?
            .get_ids()
            .to_vec();

        let stock = super::super::loader::load_llama_from_dir(&dir)?;
        let a = super::super::generate::generate_greedy(&stock, &ids, 3, Some(2))?;
        drop(stock);

        let f32m = load_llama_f32_from_dir(&dir)?;
        let b = super::super::generate::generate_greedy(&f32m, &ids, 3, Some(2))?;

        eprintln!("stock (bf16 proj): {a:?}");
        eprintln!("f32 loader:        {b:?}");
        assert_eq!(
            a, b,
            "the two loaders disagree — greedy decode should be identical unless \
             one of them mis-maps a tensor name, shape, or transpose"
        );
        Ok(())
    }
}
