//! Speculative Decoding Memory Estimation
//!
//! Memory accounting for dual-model (draft + target) speculative decoding scenarios.

use super::{ActivationMemory, KvCacheMemory, MemoryEstimate, WeightMemory};
use crate::memory::utils::{estimate_parameters, format_bytes};
use candlelight::core::DType;

/// Memory estimate for speculative decoding (dual-model)
#[derive(Debug, Clone)]
pub struct SpeculativeMemoryEstimate {
    /// Target model estimate
    pub target: MemoryEstimate,

    /// Draft model estimate
    pub draft: MemoryEstimate,

    /// Shared resources (if any, e.g., embeddings)
    pub shared_bytes: usize,
}

impl SpeculativeMemoryEstimate {
    /// Total memory for both models
    pub fn total_bytes(&self) -> usize {
        self.target.total_bytes() + self.draft.total_bytes() - self.shared_bytes
    }

    /// Display breakdown
    pub fn display(&self) -> String {
        format!(
            "Target: {}\nDraft: {}\nShared: {}\nTotal: {}",
            self.target.display(),
            self.draft.display(),
            format_bytes(self.shared_bytes),
            format_bytes(self.total_bytes()),
        )
    }

    /// Check if both models fit in available memory
    pub fn fits_in(&self, available_bytes: usize) -> bool {
        self.total_bytes() < available_bytes
    }

    /// Calculate memory utilization
    pub fn utilization(&self, available_bytes: usize) -> f64 {
        self.total_bytes() as f64 / available_bytes as f64
    }
}

/// Configuration for quantization (AWQ, GPTQ, Marlin)
#[derive(Debug, Clone)]
pub struct QuantConfig {
    pub bits: i32,
    pub group_size: i32,
    pub quant_method: String, // "awq", "gptq", "marlin"
}

/// Simple transformer config for memory estimation
#[derive(Debug, Clone)]
pub struct SimpleTransformerConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub dtype: Option<DType>,
}

impl SimpleTransformerConfig {
    /// Calculate head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// Create speculative estimate from configs
pub fn estimate_from_configs(
    target_config: &SimpleTransformerConfig,
    target_quant: Option<&QuantConfig>,
    draft_config: &SimpleTransformerConfig,
    draft_quant: Option<&QuantConfig>,
    batch_size: usize,
    max_seq_len: usize,
) -> SpeculativeMemoryEstimate {
    // Estimate target model
    let target_weights = create_weight_memory(target_config, target_quant);

    let target = MemoryEstimate {
        weights: target_weights,
        kv_cache: KvCacheMemory {
            batch_size,
            max_seq_len,
            num_layers: target_config.num_hidden_layers,
            num_kv_heads: target_config.num_key_value_heads,
            head_dim: target_config.head_dim(),
            dtype: target_config.dtype.unwrap_or(DType::F16),
        },
        activations: ActivationMemory {
            batch_size,
            seq_len: 256, // Typical generation batch size
            hidden_size: target_config.hidden_size,
            intermediate_size: target_config.intermediate_size,
            num_layers: target_config.num_hidden_layers,
            dtype: target_config.dtype.unwrap_or(DType::F16),
        },
        overhead_bytes: 100 * 1024 * 1024, // 100MB overhead
    };

    // Estimate draft model
    let draft_weights = create_weight_memory(draft_config, draft_quant);

    let draft = MemoryEstimate {
        weights: draft_weights,
        kv_cache: KvCacheMemory {
            batch_size: 1, // Draft model runs single sequence
            max_seq_len,
            num_layers: draft_config.num_hidden_layers,
            num_kv_heads: draft_config.num_key_value_heads,
            head_dim: draft_config.head_dim(),
            dtype: draft_config.dtype.unwrap_or(DType::F16),
        },
        activations: ActivationMemory {
            batch_size: 1,
            seq_len: 256,
            hidden_size: draft_config.hidden_size,
            intermediate_size: draft_config.intermediate_size,
            num_layers: draft_config.num_hidden_layers,
            dtype: draft_config.dtype.unwrap_or(DType::F16),
        },
        overhead_bytes: 50 * 1024 * 1024, // 50MB overhead
    };

    // Check for shared embeddings
    let shared_bytes = if can_share_embeddings(target_config, draft_config) {
        let dtype = target_config.dtype.unwrap_or(DType::F16);
        crate::memory::utils::estimate_embedding_size(
            target_config.vocab_size,
            target_config.hidden_size,
            dtype,
        )
    } else {
        0
    };

    SpeculativeMemoryEstimate {
        target,
        draft,
        shared_bytes,
    }
}

/// Create WeightMemory from config and quantization
fn create_weight_memory(
    config: &SimpleTransformerConfig,
    quant_config: Option<&QuantConfig>,
) -> WeightMemory {
    let num_params = estimate_parameters(
        config.vocab_size,
        config.hidden_size,
        config.intermediate_size,
        config.num_hidden_layers,
    );

    if let Some(qc) = quant_config {
        WeightMemory::Quantized {
            original_dtype: config.dtype.unwrap_or(DType::F16),
            bits: qc.bits as usize,
            num_parameters: num_params,
            metadata_bytes: WeightMemory::estimate_metadata(
                num_params,
                qc.group_size,
                &qc.quant_method,
            ),
        }
    } else {
        WeightMemory::Unquantized {
            dtype: config.dtype.unwrap_or(DType::F16),
            num_parameters: num_params,
        }
    }
}

/// Check if models can share embeddings
fn can_share_embeddings(target: &SimpleTransformerConfig, draft: &SimpleTransformerConfig) -> bool {
    target.vocab_size == draft.vocab_size && target.hidden_size == draft.hidden_size
}

#[cfg(test)]
mod tests {
    use super::*;

    fn llama_7b_config() -> SimpleTransformerConfig {
        SimpleTransformerConfig {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            dtype: Some(DType::F16),
        }
    }

    fn llama_1b_config() -> SimpleTransformerConfig {
        SimpleTransformerConfig {
            vocab_size: 32000,
            hidden_size: 2048,
            intermediate_size: 5632,
            num_hidden_layers: 24,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            dtype: Some(DType::F16),
        }
    }

    fn awq_config() -> QuantConfig {
        QuantConfig {
            bits: 4,
            group_size: 128,
            quant_method: "awq".to_string(),
        }
    }

    #[test]
    fn test_speculative_estimate_fp16_target_awq_draft() {
        let target_config = llama_7b_config();
        let draft_config = llama_1b_config();
        let draft_quant = awq_config();

        let estimate = estimate_from_configs(
            &target_config,
            None,
            &draft_config,
            Some(&draft_quant),
            1,    // batch_size
            2048, // max_seq_len
        );

        // Target: ~15GB (14GB weights + 1GB cache)
        // Draft: ~0.7GB (0.5GB weights + 0.13GB cache)
        // Total: ~15.7GB

        assert!(estimate.total_bytes() > 15_000_000_000);
        assert!(estimate.total_bytes() < 17_000_000_000);
    }

    #[test]
    fn test_shared_embeddings() {
        let target_config = llama_7b_config();
        let draft_config = llama_1b_config();

        // Both have vocab_size=32000, hidden_size differs
        assert!(!can_share_embeddings(&target_config, &draft_config));

        // Make them compatible
        let mut draft_compatible = draft_config.clone();
        draft_compatible.hidden_size = 4096;

        assert!(can_share_embeddings(&target_config, &draft_compatible));
    }

    #[test]
    fn test_fits_in_24gb_gpu() {
        let target_config = llama_7b_config();
        let draft_config = llama_1b_config();
        let draft_quant = awq_config();

        let estimate = estimate_from_configs(
            &target_config,
            None,
            &draft_config,
            Some(&draft_quant),
            1,
            2048,
        );

        let gpu_24gb = 24 * 1024 * 1024 * 1024; // 24GB
        assert!(estimate.fits_in(gpu_24gb));
        assert!(estimate.utilization(gpu_24gb) < 0.7); // < 70% utilization
    }

    #[test]
    fn test_display_format() {
        let target_config = llama_7b_config();
        let draft_config = llama_1b_config();
        let draft_quant = awq_config();

        let estimate = estimate_from_configs(
            &target_config,
            None,
            &draft_config,
            Some(&draft_quant),
            1,
            2048,
        );

        let display = estimate.display();
        assert!(display.contains("Target:"));
        assert!(display.contains("Draft:"));
        assert!(display.contains("Shared:"));
        assert!(display.contains("Total:"));
    }
}
