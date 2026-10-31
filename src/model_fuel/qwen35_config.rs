// SPDX-License-Identifier: MIT OR Apache-2.0
//! `qwen35` (Qwen3.5 / Qwen3.8) model geometry from a GGUF's own metadata
//! (ROADMAP F10, L1).
//!
//! The architecture is hybrid: Gated DeltaNet ("linear") layers with a
//! fixed-size recurrent state, and every `full_attention_interval`-th layer a
//! full-attention layer with a KV cache. No loader exists yet (it needs Fuel's
//! hybrid decode model), but the geometry can be read and checked now, and the
//! engine's refusal quotes it so the reader sees what the file is.
//!
//! The key names are the ones read from `unsloth/Qwen3.5-0.8B-GGUF`'s header
//! (ROADMAP F10); values here are checked against that model's `config.json`
//! (24 layers, `full_attention_interval` 4). Not yet read from a 27B file.

use std::collections::HashMap;

use anyhow::{Result, bail};

use crate::gguf::{Value, metadata_f32, metadata_u64};

/// Which mixer a layer uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayerKind {
    /// Gated DeltaNet: constant-size recurrent state, no KV cache.
    Linear,
    /// Full attention with a KV cache.
    Full,
}

/// Gated DeltaNet (`ssm.*`) hyperparameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SsmConfig {
    pub conv_kernel: u64,
    pub state_size: u64,
    pub group_count: u64,
    pub time_step_rank: u64,
    pub inner_size: u64,
}

/// Geometry of a `qwen35` checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Qwen35Config {
    pub layers: u64,
    pub context_length: u64,
    pub hidden_size: u64,
    pub ffn_size: u64,
    pub heads: u64,
    pub kv_heads: u64,
    pub key_length: u64,
    pub value_length: u64,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub rope_dimensions: u64,
    pub rope_sections: Vec<u64>,
    pub full_attention_interval: u64,
    pub ssm: SsmConfig,
}

impl Qwen35Config {
    /// Read and validate the geometry. Every key is required: a hybrid
    /// layout guessed from defaults would be a wrong model, not a missing key.
    pub(crate) fn from_metadata(md: &HashMap<String, Value>) -> Result<Self> {
        match md.get("general.architecture") {
            Some(Value::String(a)) if a == "qwen35" => {}
            other => bail!("not a `qwen35` GGUF (general.architecture = {other:?})"),
        }
        let u = |k: &str| metadata_u64(md, &format!("qwen35.{k}"));
        let cfg = Self {
            layers: u("block_count")?,
            context_length: u("context_length")?,
            hidden_size: u("embedding_length")?,
            ffn_size: u("feed_forward_length")?,
            heads: u("attention.head_count")?,
            kv_heads: u("attention.head_count_kv")?,
            key_length: u("attention.key_length")?,
            value_length: u("attention.value_length")?,
            rms_eps: metadata_f32(md, "qwen35.attention.layer_norm_rms_epsilon")?,
            rope_theta: metadata_f32(md, "qwen35.rope.freq_base")?,
            rope_dimensions: u("rope.dimension_count")?,
            rope_sections: int_array(md, "qwen35.rope.dimension_sections")?,
            full_attention_interval: u("full_attention_interval")?,
            ssm: SsmConfig {
                conv_kernel: u("ssm.conv_kernel")?,
                state_size: u("ssm.state_size")?,
                group_count: u("ssm.group_count")?,
                time_step_rank: u("ssm.time_step_rank")?,
                inner_size: u("ssm.inner_size")?,
            },
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        if self.full_attention_interval == 0
            || !self.layers.is_multiple_of(self.full_attention_interval)
        {
            bail!(
                "qwen35: {} layers is not a whole number of groups of \
                 full_attention_interval = {} (each group is interval-1 linear layers then \
                 one full-attention layer)",
                self.layers,
                self.full_attention_interval
            );
        }
        if self.kv_heads == 0 || !self.heads.is_multiple_of(self.kv_heads) {
            bail!(
                "qwen35: attention.head_count {} is not a multiple of head_count_kv {}",
                self.heads,
                self.kv_heads
            );
        }
        Ok(())
    }

    /// Layer `i` (zero-based) is full attention when it ends a group of
    /// `full_attention_interval`: interval 4 gives linear, linear, linear, full.
    pub(crate) fn layer_kind(&self, i: u64) -> LayerKind {
        if (i + 1).is_multiple_of(self.full_attention_interval) {
            LayerKind::Full
        } else {
            LayerKind::Linear
        }
    }

    /// Number of full-attention layers (the only ones with a KV cache).
    pub(crate) fn full_layers(&self) -> u64 {
        self.layers / self.full_attention_interval
    }

    /// Number of Gated DeltaNet layers.
    pub(crate) fn linear_layers(&self) -> u64 {
        self.layers - self.full_layers()
    }

    /// One line for the engine's refusal message.
    pub(crate) fn describe(&self) -> String {
        format!(
            "{} layers ({} Gated DeltaNet + {} full attention), hidden {}, {} heads / {} KV heads \
             of {}, context {}",
            self.layers,
            self.linear_layers(),
            self.full_layers(),
            self.hidden_size,
            self.heads,
            self.kv_heads,
            self.key_length,
            self.context_length
        )
    }
}

/// An integer-array metadata key, refusing a wrong type or a negative entry.
fn int_array(md: &HashMap<String, Value>, key: &str) -> Result<Vec<u64>> {
    let Some(Value::Array(items)) = md.get(key) else {
        bail!("`{key}` is not declared as an array by this GGUF");
    };
    items
        .iter()
        .map(|v| match v {
            Value::U8(n) => Ok(u64::from(*n)),
            Value::U16(n) => Ok(u64::from(*n)),
            Value::U32(n) => Ok(u64::from(*n)),
            Value::U64(n) => Ok(*n),
            Value::I8(n) => u64::try_from(*n).map_err(Into::into),
            Value::I16(n) => u64::try_from(*n).map_err(Into::into),
            Value::I32(n) => u64::try_from(*n).map_err(Into::into),
            Value::I64(n) => u64::try_from(*n).map_err(Into::into),
            other => bail!("`{key}` holds a non-integer entry: {other:?}"),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Metadata as `unsloth/Qwen3.5-0.8B-GGUF` declares it, with the values of
    /// `Qwen/Qwen3.5-0.8B`'s `config.json`.
    fn md_0_8b() -> HashMap<String, Value> {
        let mut m = HashMap::new();
        let mut u32k = |k: &str, v: u32| {
            m.insert(format!("qwen35.{k}"), Value::U32(v));
        };
        u32k("block_count", 24);
        u32k("context_length", 262_144);
        u32k("embedding_length", 1024);
        u32k("feed_forward_length", 3584);
        u32k("attention.head_count", 8);
        u32k("attention.head_count_kv", 2);
        u32k("attention.key_length", 256);
        u32k("attention.value_length", 256);
        u32k("rope.dimension_count", 64);
        u32k("full_attention_interval", 4);
        u32k("ssm.conv_kernel", 4);
        u32k("ssm.state_size", 128);
        u32k("ssm.group_count", 16);
        u32k("ssm.time_step_rank", 16);
        u32k("ssm.inner_size", 2048);
        m.insert(
            "qwen35.attention.layer_norm_rms_epsilon".into(),
            Value::F32(1e-6),
        );
        m.insert("qwen35.rope.freq_base".into(), Value::F32(10_000_000.0));
        m.insert(
            "qwen35.rope.dimension_sections".into(),
            Value::Array(vec![
                Value::I32(11),
                Value::I32(11),
                Value::I32(10),
                Value::I32(0),
            ]),
        );
        m.insert(
            "general.architecture".into(),
            Value::String("qwen35".into()),
        );
        m
    }

    #[test]
    fn reads_the_0_8b_geometry_and_its_18_plus_6_layer_split() {
        let c = Qwen35Config::from_metadata(&md_0_8b()).unwrap();
        assert_eq!(
            (c.layers, c.heads, c.kv_heads, c.key_length),
            (24, 8, 2, 256)
        );
        assert_eq!(c.rope_sections, vec![11, 11, 10, 0]);
        assert_eq!((c.linear_layers(), c.full_layers()), (18, 6));
        // `layer_types` of the model's config.json: three linear, then one full.
        let kinds: Vec<_> = (0..8).map(|i| c.layer_kind(i)).collect();
        use LayerKind::{Full, Linear};
        assert_eq!(
            kinds,
            [Linear, Linear, Linear, Full, Linear, Linear, Linear, Full]
        );
        let full = (0..c.layers).filter(|i| c.layer_kind(*i) == Full).count() as u64;
        assert_eq!(full, c.full_layers());
    }

    #[test]
    fn a_27b_shape_has_16_full_layers_of_64() {
        let mut m = md_0_8b();
        m.insert("qwen35.block_count".into(), Value::U32(64));
        let c = Qwen35Config::from_metadata(&m).unwrap();
        assert_eq!((c.linear_layers(), c.full_layers()), (48, 16));
        assert!(
            c.describe()
                .starts_with("64 layers (48 Gated DeltaNet + 16 full attention)")
        );
    }

    #[test]
    fn every_key_is_required_and_a_wrong_layout_is_refused() {
        // Control: the full set parses (above). Dropping any one key must refuse.
        let full = md_0_8b();
        for key in full.keys().filter(|k| k.starts_with("qwen35.")) {
            let mut m = full.clone();
            m.remove(key);
            assert!(
                Qwen35Config::from_metadata(&m).is_err(),
                "{key} was optional"
            );
        }
        let mut m = full.clone();
        m.insert("qwen35.block_count".into(), Value::U32(25));
        let e = Qwen35Config::from_metadata(&m).unwrap_err().to_string();
        assert!(e.contains("25 layers"), "{e}");
        let mut m = full.clone();
        m.insert("qwen35.full_attention_interval".into(), Value::U32(0));
        assert!(Qwen35Config::from_metadata(&m).is_err());
        let mut m = full.clone();
        m.insert("qwen35.attention.head_count_kv".into(), Value::U32(3));
        assert!(Qwen35Config::from_metadata(&m).is_err());
        let mut m = full;
        m.insert("general.architecture".into(), Value::String("qwen3".into()));
        assert!(Qwen35Config::from_metadata(&m).is_err());
    }

    #[test]
    fn a_negative_rope_section_is_refused_not_wrapped() {
        let mut m = md_0_8b();
        m.insert(
            "qwen35.rope.dimension_sections".into(),
            Value::Array(vec![Value::I32(-1)]),
        );
        assert!(Qwen35Config::from_metadata(&m).is_err());
    }
}
