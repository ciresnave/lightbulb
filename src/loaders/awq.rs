//! AWQ (Activation-aware Weight Quantization) model loader
//!
//! This module loads AWQ-quantized models from safetensors files and integrates
//! with Marlin CUDA kernels for efficient 4-bit inference.
//!
//! AWQ format stores weights in three tensors per layer:
//! - `qweight`: 4-bit quantized weights (packed)
//! - `qzeros`: Zero points for quantization groups
//! - `scales`: Per-group quantization scales (typically group_size=128)

use anyhow::{Context, Result};
use candlelight::core::{DType, Device, Tensor};
use candlelight::nn::VarBuilder;
use std::collections::HashMap;
use std::path::Path;

/// AWQ quantization configuration
#[derive(Debug, Clone)]
pub struct AwqConfig {
    /// Number of bits for quantization (typically 4)
    pub bits: usize,

    /// Group size for quantization (typically 128)
    pub group_size: usize,

    /// AWQ version ("gemm" or "gemv")
    pub version: String,

    /// Whether to use zero-point quantization
    pub zero_point: bool,

    /// Modules to skip quantization
    pub modules_to_not_convert: Option<Vec<String>>,
}

impl AwqConfig {
    /// Extract AWQ config from model config.json
    pub fn from_config_json(config: &serde_json::Value) -> Result<Self> {
        let quant_config = config
            .get("quantization_config")
            .context("No quantization_config found in config.json")?;

        Ok(Self {
            bits: quant_config
                .get("bits")
                .and_then(|v| v.as_u64())
                .context("Missing or invalid 'bits' in quantization_config")?
                as usize,
            group_size: quant_config
                .get("group_size")
                .and_then(|v| v.as_u64())
                .context("Missing or invalid 'group_size' in quantization_config")?
                as usize,
            version: quant_config
                .get("version")
                .and_then(|v| v.as_str())
                .unwrap_or("gemm")
                .to_string(),
            zero_point: quant_config
                .get("zero_point")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
            modules_to_not_convert: quant_config
                .get("modules_to_not_convert")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                }),
        })
    }
}

/// AWQ quantized linear layer (stored as qweight/qzeros/scales)
#[derive(Debug, Clone)]
pub struct AwqLinear {
    /// Quantized weights (4-bit packed)
    pub qweight: Tensor,

    /// Zero points (per group)
    pub qzeros: Tensor,

    /// Scales (per group)
    pub scales: Tensor,

    /// Input features
    pub in_features: usize,

    /// Output features
    pub out_features: usize,

    /// Group size for quantization
    pub group_size: usize,

    /// Number of bits
    pub bits: usize,
}
impl AwqLinear {
    /// Create new AWQ linear layer (loads from VarBuilder)
    pub fn new(in_features: usize, out_features: usize, vb: VarBuilder) -> Result<Self> {
        // Default AWQ parameters (will be overridden by actual config)
        let group_size = 128;
        let bits = 4;

        let qweight = vb
            .get((in_features / 8, out_features), "qweight")
            .context("Failed to load qweight")?;

        let qzeros = vb
            .get((in_features / group_size, out_features / 8), "qzeros")
            .context("Failed to load qzeros")?;

        let scales = vb
            .get((in_features / group_size, out_features), "scales")
            .context("Failed to load scales")?;

        Ok(Self {
            qweight,
            qzeros,
            scales,
            in_features,
            out_features,
            group_size,
            bits,
        })
    }

    /// Load AWQ linear layer from VarBuilder (alternative constructor with explicit params)
    pub fn load(
        vb: VarBuilder,
        in_features: usize,
        out_features: usize,
        group_size: usize,
        bits: usize,
    ) -> Result<Self> {
        let qweight = vb
            .get((in_features / 8, out_features), "qweight")
            .context("Failed to load qweight")?;

        let qzeros = vb
            .get((in_features / group_size, out_features / 8), "qzeros")
            .context("Failed to load qzeros")?;

        let scales = vb
            .get((in_features / group_size, out_features), "scales")
            .context("Failed to load scales")?;

        Ok(Self {
            qweight,
            qzeros,
            scales,
            in_features,
            out_features,
            group_size,
            bits,
        })
    }

    /// Forward pass using Marlin kernels
    #[cfg(feature = "cuda")]
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        use crate::backend::marlin::{MarlinMatMul, Precision, QuantFormat};

        // Determine precision from input dtype
        let precision = match input.dtype() {
            candlelight::core::DType::F16 => Precision::F16,
            candlelight::core::DType::BF16 => Precision::BF16,
            _ => anyhow::bail!(
                "AWQ forward only supports F16/BF16, got {:?}",
                input.dtype()
            ),
        };

        // Create Marlin MatMul operation
        let marlin_op = MarlinMatMul::new(precision, QuantFormat::AWQ, self.group_size);

        // Apply quantized matmul
        input
            .apply_op3(&self.qweight, &self.scales, &marlin_op)
            .context("Marlin AWQ matmul failed")
    }

    /// Forward pass (CPU fallback - dequantize then matmul)
    #[cfg(not(feature = "cuda"))]
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Dequantize weights (simplified - actual AWQ dequant is more complex)
        anyhow::bail!("AWQ CPU inference not yet implemented. Use CUDA or convert to GGUF.")
    }
}

// Implement Module trait for AwqLinear to enable use with .apply()
impl candlelight::core::Module for AwqLinear {
    fn forward(&self, input: &Tensor) -> candlelight::core::Result<Tensor> {
        Self::forward(self, input)
            .map_err(|e| candlelight::core::Error::Msg(format!("AWQ forward failed: {}", e)))
    }
}

/// Load AWQ model metadata from directory
pub fn load_awq_metadata(model_dir: &Path) -> Result<(serde_json::Value, AwqConfig)> {
    let config_path = model_dir.join("config.json");
    let config_bytes = std::fs::read(&config_path)
        .with_context(|| format!("Failed to read config.json from {:?}", config_path))?;

    let config: serde_json::Value =
        serde_json::from_slice(&config_bytes).context("Failed to parse config.json")?;

    let awq_config = AwqConfig::from_config_json(&config)?;

    // Validate AWQ config
    if awq_config.bits != 4 {
        anyhow::bail!(
            "Only 4-bit AWQ is currently supported, got {} bits",
            awq_config.bits
        );
    }

    println!("✓ Detected AWQ configuration:");
    println!("  - Bits: {}", awq_config.bits);
    println!("  - Group size: {}", awq_config.group_size);
    println!("  - Version: {}", awq_config.version);
    println!("  - Zero point: {}", awq_config.zero_point);

    Ok((config, awq_config))
}

/// Check if a tensor name should be quantized
pub fn should_quantize(name: &str, skip_modules: &Option<Vec<String>>) -> bool {
    if let Some(skip) = skip_modules {
        for module in skip {
            if name.contains(module) {
                return false;
            }
        }
    }

    // Linear layers in attention and MLP are quantized
    name.contains(".q_proj.")
        || name.contains(".k_proj.")
        || name.contains(".v_proj.")
        || name.contains(".o_proj.")
        || name.contains(".gate_proj.")
        || name.contains(".up_proj.")
        || name.contains(".down_proj.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_quantize() {
        assert!(should_quantize(
            "model.layers.0.self_attn.q_proj.qweight",
            &None
        ));
        assert!(should_quantize(
            "model.layers.0.mlp.gate_proj.qweight",
            &None
        ));
        assert!(!should_quantize("model.embed_tokens.weight", &None));
        assert!(!should_quantize("lm_head.weight", &None));
    }

    #[test]
    fn test_should_quantize_with_skip() {
        let skip = Some(vec!["lm_head".to_string()]);
        assert!(!should_quantize("lm_head.qweight", &skip));
        assert!(should_quantize("model.layers.0.q_proj.qweight", &skip));
    }
}
