// Shared test support: a tiny synthetic Llama-shaped checkpoint.
//
// # Why this exists
//
// See `docs/CHECKPOINT_FIXTURE_FEASIBILITY.md`. A large class of
// `#[ignore]`d tests only need a *loadable, architecturally valid*
// checkpoint -- not one with correct trained values. This writes exactly
// that: a `config.json` + `model.safetensors` pair small enough to generate
// and load in milliseconds, in the same HuggingFace directory layout
// [`load_llama_f32_from_dir`]/[`load_llama_from_dir`] already read.
//
// Random weights, not zeros or ones: a model whose activations are
// real-valued (not identically zero, not saturated) exercises the same
// numerical code paths a trained checkpoint would, which matters for tests
// that check shapes/determinism/consistency through actual forward passes.
// Not cryptographically random, and not meant to be -- a small fixed-seed
// generator (not `rand`, to avoid a dependency for four lines of math) so
// regenerating the fixture is reproducible.
//
// Reached two ways: integration tests under `tests/` declare
// `#[path = "support/mod.rs"] mod support;`; `#[cfg(test)]` unit tests under
// `src/` reach the same file via `include!` from `src/test_models.rs`
// (`#[cfg(test)]`-gated there, since `safetensors` is a dev-dependency,
// unavailable to the crate's normal non-test build). One physical file,
// included both ways, not two copies to keep in sync.
//
// Plain `//` comments throughout, not `//!`/`///` doc comments: inner doc
// comments are only valid as the first items of a file loaded via
// `mod foo;`, not when spliced into an enclosing block by `include!` --
// using them here failed to compile with E0753 ("expected outer doc
// comment") the first time this was written.

use std::collections::HashMap;
use std::path::Path;

use safetensors::serialize_to_file;
use safetensors::tensor::{Dtype, TensorView};

/// Architecture of the synthetic checkpoint. Deliberately tiny -- large
/// enough that every tensor has a real shape to check, small enough that
/// generating and loading it costs milliseconds, not minutes.
pub struct SyntheticArchitecture {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
}

impl Default for SyntheticArchitecture {
    /// Values chosen to be unambiguously different from each other (no two
    /// dimensions share a value) so a test that mixes up two axes -- e.g.
    /// transposes dim and ffn_dim, or confuses num_layers with num_heads --
    /// fails loudly instead of accidentally still producing the right shape.
    fn default() -> Self {
        Self {
            hidden_size: 16,
            intermediate_size: 40,
            num_hidden_layers: 3,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            vocab_size: 97,
            max_position_embeddings: 64,
            bos_token_id: 0,
            eos_token_id: 1,
        }
    }
}

impl SyntheticArchitecture {
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim()
    }

    /// The HuggingFace `config.json` this architecture renders to.
    fn config_json(&self) -> String {
        format!(
            r#"{{
  "hidden_size": {},
  "intermediate_size": {},
  "num_hidden_layers": {},
  "num_attention_heads": {},
  "num_key_value_heads": {},
  "vocab_size": {},
  "max_position_embeddings": {},
  "rms_norm_eps": 1e-5,
  "rope_theta": 10000.0,
  "bos_token_id": {},
  "eos_token_id": {},
  "tie_word_embeddings": false,
  "model_type": "llama"
}}
"#,
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.vocab_size,
            self.max_position_embeddings,
            self.bos_token_id,
            self.eos_token_id,
        )
    }

    // The loaders also require a `tokenizer.json` next to the checkpoint
    // (`tokenizers::Tokenizer::from_file`). A `WordLevel` model is the
    // simplest valid shape that crate accepts -- a flat token->id map and an
    // `unk_token`, no merges/BPE rules to get right. `<s>`/`</s>` are pinned
    // to `bos_token_id`/`eos_token_id` so the config and tokenizer agree;
    // the rest are arbitrary placeholder tokens filling out `vocab_size`.
    fn tokenizer_json(&self) -> String {
        let mut vocab_entries = vec![
            format!(r#""<unk>": {}"#, self.vocab_size.saturating_sub(1)),
            format!(r#""<s>": {}"#, self.bos_token_id),
            format!(r#""</s>": {}"#, self.eos_token_id),
        ];
        let reserved = [self.bos_token_id, self.eos_token_id];
        let mut next_filler = 0u32;
        while vocab_entries.len() < self.vocab_size {
            while reserved.contains(&next_filler)
                || next_filler == (self.vocab_size.saturating_sub(1)) as u32
            {
                next_filler += 1;
            }
            vocab_entries.push(format!(r#""tok{next_filler}": {next_filler}"#));
            next_filler += 1;
        }
        format!(
            r#"{{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {{"type": "Whitespace"}},
  "post_processor": null,
  "decoder": null,
  "model": {{
    "type": "WordLevel",
    "vocab": {{{}}},
    "unk_token": "<unk>"
  }}
}}
"#,
            vocab_entries.join(", "),
        )
    }
}

/// A small, fixed-seed, dependency-free PRNG (xorshift64*) -- not
/// cryptographic, not meant to be; just deterministic and fast.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// A small f32 in roughly [-0.1, 0.1] -- small enough that a few dozen
    /// matmul layers stay numerically sane (no overflow/NaN) without any
    /// normalization tricks.
    fn next_small_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32; // 24 bits of entropy
        let unit = (bits as f32) / (1u32 << 24) as f32; // [0, 1)
        (unit - 0.5) * 0.2
    }
}

fn random_vec(rng: &mut Xorshift64, len: usize) -> Vec<f32> {
    (0..len).map(|_| rng.next_small_f32()).collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Writes a synthetic Llama-shaped checkpoint (`config.json` +
/// `model.safetensors`) into `dir`, which must already exist.
///
/// Every tensor [`load_llama_f32_from_dir`]/[`load_llama_from_dir`] reads is
/// written, using the standard HuggingFace Llama tensor names
/// (`model.embed_tokens.weight`, `model.layers.{i}.self_attn.*`,
/// `model.layers.{i}.mlp.*`, `model.layers.{i}.*_layernorm.weight`,
/// `model.norm.weight`, `lm_head.weight`).
pub fn write_synthetic_checkpoint(
    dir: &Path,
    arch: &SyntheticArchitecture,
) -> anyhow::Result<()> {
    std::fs::write(dir.join("config.json"), arch.config_json())?;
    std::fs::write(dir.join("tokenizer.json"), arch.tokenizer_json())?;

    let dim = arch.hidden_size;
    let ffn_dim = arch.intermediate_size;
    let kv_dim = arch.kv_dim();
    let vocab = arch.vocab_size;

    let mut rng = Xorshift64::new(0x5EED_0000_F00D_0001);

    // Buffers must outlive the TensorViews borrowing from them, so collect
    // everything first, then build views over the finished Vec<Vec<u8>>.
    let mut data: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    let mut push = |name: String, shape: Vec<usize>, len: usize, rng: &mut Xorshift64| {
        data.push((name, shape, f32_bytes(&random_vec(rng, len))));
    };

    push(
        "model.embed_tokens.weight".into(),
        vec![vocab, dim],
        vocab * dim,
        &mut rng,
    );

    for i in 0..arch.num_hidden_layers {
        push(
            format!("model.layers.{i}.self_attn.q_proj.weight"),
            vec![dim, dim],
            dim * dim,
            &mut rng,
        );
        push(
            format!("model.layers.{i}.self_attn.k_proj.weight"),
            vec![kv_dim, dim],
            kv_dim * dim,
            &mut rng,
        );
        push(
            format!("model.layers.{i}.self_attn.v_proj.weight"),
            vec![kv_dim, dim],
            kv_dim * dim,
            &mut rng,
        );
        push(
            format!("model.layers.{i}.self_attn.o_proj.weight"),
            vec![dim, dim],
            dim * dim,
            &mut rng,
        );
        push(
            format!("model.layers.{i}.mlp.gate_proj.weight"),
            vec![ffn_dim, dim],
            ffn_dim * dim,
            &mut rng,
        );
        push(
            format!("model.layers.{i}.mlp.up_proj.weight"),
            vec![ffn_dim, dim],
            ffn_dim * dim,
            &mut rng,
        );
        push(
            format!("model.layers.{i}.mlp.down_proj.weight"),
            vec![dim, ffn_dim],
            dim * ffn_dim,
            &mut rng,
        );
        push(
            format!("model.layers.{i}.input_layernorm.weight"),
            vec![dim],
            dim,
            &mut rng,
        );
        push(
            format!("model.layers.{i}.post_attention_layernorm.weight"),
            vec![dim],
            dim,
            &mut rng,
        );
    }

    push("model.norm.weight".into(), vec![dim], dim, &mut rng);
    push(
        "lm_head.weight".into(),
        vec![vocab, dim],
        vocab * dim,
        &mut rng,
    );

    let views: Vec<(String, TensorView)> = data
        .iter()
        .map(|(name, shape, bytes)| {
            let view = TensorView::new(Dtype::F32, shape.clone(), bytes)
                .unwrap_or_else(|e| panic!("building TensorView for {name}: {e:?}"));
            (name.clone(), view)
        })
        .collect();

    serialize_to_file(views, None::<HashMap<String, String>>, &dir.join("model.safetensors"))?;

    Ok(())
}
