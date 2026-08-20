//! Memory Utility Functions
//!
//! Helper functions for memory estimation and formatting.

/// Format bytes as human-readable string (B, KB, MB, GB)
pub fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Estimate total parameters for a transformer model
///
/// # Arguments
///
/// * `vocab_size` - Vocabulary size
/// * `hidden_size` - Hidden dimension
/// * `intermediate_size` - MLP intermediate dimension
/// * `num_layers` - Number of transformer layers
///
/// # Returns
///
/// Estimated number of parameters
pub fn estimate_parameters(
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_layers: usize,
) -> usize {
    // Embeddings
    let embedding_params = vocab_size * hidden_size;

    // Per-layer parameters
    let layer_params = estimate_layer_parameters(hidden_size, intermediate_size);

    // Output head (lm_head)
    let output_params = hidden_size * vocab_size;

    embedding_params + (layer_params * num_layers) + output_params
}

/// Estimate parameters in a single transformer layer
fn estimate_layer_parameters(hidden_size: usize, intermediate_size: usize) -> usize {
    // Attention: Q, K, V, O projections (all hidden x hidden)
    let attn_params = hidden_size * hidden_size * 4;

    // MLP: gate_proj, up_proj, down_proj
    // gate: hidden -> intermediate
    // up: hidden -> intermediate
    // down: intermediate -> hidden
    let mlp_params = (hidden_size * intermediate_size * 2) + (intermediate_size * hidden_size);

    // Layer norms (typically 2 per layer, each has hidden_size params)
    let ln_params = hidden_size * 2;

    attn_params + mlp_params + ln_params
}

/// Estimate embedding layer size in bytes
///
/// # Arguments
///
/// * `vocab_size` - Vocabulary size
/// * `hidden_size` - Embedding dimension
/// * `dtype` - Data type
pub fn estimate_embedding_size(
    vocab_size: usize,
    hidden_size: usize,
    dtype: candlelight::core::DType,
) -> usize {
    let bytes_per_param = match dtype {
        candlelight::core::DType::F32 => 4,
        candlelight::core::DType::F16 | candlelight::core::DType::BF16 => 2,
        _ => 2,
    };
    vocab_size * hidden_size * bytes_per_param
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1_048_576), "1.00 MB");
        assert_eq!(format_bytes(1_073_741_824), "1.00 GB");
        assert_eq!(format_bytes(7_516_192_768), "7.00 GB");
    }

    #[test]
    fn test_estimate_parameters() {
        // Llama-7B config
        let params = estimate_parameters(
            32000, // vocab_size
            4096,  // hidden_size
            11008, // intermediate_size
            32,    // num_layers
        );

        // Should be approximately 7B parameters
        assert!(params > 6_500_000_000);
        assert!(params < 7_500_000_000);
    }

    #[test]
    fn test_estimate_layer_parameters() {
        // Llama-7B layer
        let params = estimate_layer_parameters(4096, 11008);

        // Attention: 4096 × 4096 × 4 = 67,108,864
        // MLP: 4096 × 11008 × 3 = 135,266,304
        // LN: 4096 × 2 = 8,192
        // Total ≈ 202,383,360

        assert!(params > 200_000_000);
        assert!(params < 210_000_000);
    }

    #[test]
    fn test_estimate_embedding_size() {
        use candlelight::core::DType;

        // Llama-7B embeddings
        let size_fp16 = estimate_embedding_size(32000, 4096, DType::F16);
        let size_fp32 = estimate_embedding_size(32000, 4096, DType::F32);

        // FP16: 32000 × 4096 × 2 = 262,144,000 bytes
        assert_eq!(size_fp16, 262_144_000);

        // FP32: 32000 × 4096 × 4 = 524,288,000 bytes
        assert_eq!(size_fp32, 524_288_000);
    }
}
