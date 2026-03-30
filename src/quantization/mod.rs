//! Quantization utilities for GGUF models and post-quantization optimizations
//!
//! This module provides tools for working with quantized weights in GGUF format,
//! including dequantization, requantization, format conversion, and accuracy
//! recovery techniques like Norm Tweaking.
//!
//! # Supported Quantization Formats
//!
//! - **Q4_0**: 4-bit weights with single scale per block (32 values)
//! - **Q4_1**: 4-bit weights with scale + minimum per block
//! - **Q5_0**: 5-bit weights with single scale per block
//! - **Q5_1**: 5-bit weights with scale + minimum per block
//! - **Q8_0**: 8-bit weights with single scale per block
//! - **Q8_1**: 8-bit weights with scale + minimum per block
//! - **Q2K-Q6K**: K-quants (complex multi-level quantization)
//! - **Q8K**: 8-bit K-quants
//!
//! # Post-Quantization Optimizations
//!
//! - **Norm Tweaking**: LayerNorm/RMSNorm calibration for 1.5-3% accuracy recovery
//!
//! # Examples
//!
//! ```rust,no_run
//! use lightbulb::quantization::{dequantize_tensor, quantize_tensor, GgmlDType};
//! use lightbulb::quantization::{NormTweaker, NormTweakingConfig};
//! use std::path::Path;
//!
//! // Dequantize Q4_K weights to F32
//! let weights_f32 = dequantize_tensor(quantized_data, GgmlDType::Q4K)?;
//!
//! // Re-quantize to Q8_0 (higher quality)
//! let weights_q8 = quantize_tensor(&weights_f32, GgmlDType::Q8_0)?;
//!
//! // Apply Norm Tweaking for accuracy recovery
//! let config = NormTweakingConfig::default();
//! let tweaker = NormTweaker::new(config, device);
//! let adjustments = tweaker.calibrate(&calibration_data, &layer_stats)?;
//! ```

pub mod gguf_ops;
pub mod norm_tweaking;

// Re-export commonly used types from ggml-quants
pub use ggml_quants::{
    Q2K, Q3K, Q4_0, Q4_1, Q4K, Q5_0, Q5_1, Q5K, Q6K, Q8_0, Q8_1, Q8K, Quantize, QuantizeError,
    bf16, f16,
};

// Re-export Norm Tweaking types
pub use norm_tweaking::{
    LayerAdjustments, LayerStats, NormTweaker, NormTweakingConfig, apply_norm_tweaking,
};

use anyhow::Result;

/// GGML quantization data types (matches GGUF format)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgmlDType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
}

impl GgmlDType {
    /// Convert from raw GGUF type value
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            15 => Some(Self::Q8K),
            _ => None,
        }
    }

    /// Get block size for this quantization type
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
        }
    }

    /// Get type size in bytes for one block
    pub fn type_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 => 18, // 16 bytes (4-bit × 32) + 2 bytes (scale)
            Self::Q4_1 => 20, // 16 bytes + 2 bytes (scale) + 2 bytes (min)
            Self::Q5_0 => 22, // 20 bytes (5-bit × 32) + 2 bytes (scale)
            Self::Q5_1 => 24, // 20 bytes + 2 bytes (scale) + 2 bytes (min)
            Self::Q8_0 => 34, // 32 bytes (8-bit × 32) + 2 bytes (scale)
            Self::Q8_1 => 36, // 32 bytes + 2 bytes (scale) + 2 bytes (min)
            Self::Q2K => 82,  // Complex block structure
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
        }
    }

    /// Human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
        }
    }
}

/// Dequantize a tensor from GGML quantized format to F32
///
/// # Arguments
/// * `data` - Raw quantized bytes
/// * `dtype` - Quantization format
/// * `elem_count` - Expected number of F32 elements
///
/// # Returns
/// Vector of F32 values
pub fn dequantize_tensor(data: &[u8], dtype: GgmlDType, elem_count: usize) -> Result<Vec<f32>> {
    use ggml_quants::DataBlock; // Import trait for COUNT constant

    match dtype {
        GgmlDType::Q4_0 => {
            // Q4_0 has 32 elements per block
            let block_size = <Q4_0 as DataBlock>::COUNT;
            let num_blocks = (elem_count + block_size - 1) / block_size;
            let expected_bytes = num_blocks * std::mem::size_of::<Q4_0>();

            if data.len() < expected_bytes {
                anyhow::bail!(
                    "Insufficient data for Q4_0: got {} bytes, need {} bytes",
                    data.len(),
                    expected_bytes
                );
            }

            let mut result = Vec::with_capacity(elem_count);
            let blocks_ptr = data.as_ptr() as *const Q4_0;

            unsafe {
                for i in 0..num_blocks {
                    let block = &*blocks_ptr.add(i);
                    let deq = block.dequantize();
                    result.extend_from_slice(&deq[..block_size.min(elem_count - result.len())]);
                }
            }

            result.truncate(elem_count);
            Ok(result)
        }
        GgmlDType::Q4K => {
            // Q4K has 256 elements per block
            let block_size = <Q4K as DataBlock>::COUNT;
            let num_blocks = (elem_count + block_size - 1) / block_size;
            let expected_bytes = num_blocks * std::mem::size_of::<Q4K>();

            if data.len() < expected_bytes {
                anyhow::bail!(
                    "Insufficient data for Q4K: got {} bytes, need {} bytes",
                    data.len(),
                    expected_bytes
                );
            }

            let mut result = Vec::with_capacity(elem_count);
            let blocks_ptr = data.as_ptr() as *const Q4K;

            unsafe {
                for i in 0..num_blocks {
                    let block = &*blocks_ptr.add(i);
                    let deq = block.dequantize();
                    result.extend_from_slice(&deq[..block_size.min(elem_count - result.len())]);
                }
            }

            result.truncate(elem_count);
            Ok(result)
        }
        GgmlDType::Q8_0 => {
            let block_size = <Q8_0 as DataBlock>::COUNT;
            let num_blocks = (elem_count + block_size - 1) / block_size;
            let expected_bytes = num_blocks * std::mem::size_of::<Q8_0>();

            if data.len() < expected_bytes {
                anyhow::bail!(
                    "Insufficient data for Q8_0: got {} bytes, need {} bytes",
                    data.len(),
                    expected_bytes
                );
            }

            let mut result = Vec::with_capacity(elem_count);
            let blocks_ptr = data.as_ptr() as *const Q8_0;

            unsafe {
                for i in 0..num_blocks {
                    let block = &*blocks_ptr.add(i);
                    let deq = block.dequantize();
                    result.extend_from_slice(&deq[..block_size.min(elem_count - result.len())]);
                }
            }

            result.truncate(elem_count);
            Ok(result)
        }
        _ => anyhow::bail!("Dequantization for {:?} not yet implemented", dtype),
    }
}

/// Quantize F32 tensor to GGML quantized format
///
/// # Arguments
/// * `data` - F32 values to quantize
/// * `dtype` - Target quantization format
///
/// # Returns
/// Vector of quantized bytes
pub fn quantize_tensor(data: &[f32], dtype: GgmlDType) -> Result<Vec<u8>> {
    use ggml_quants::DataBlock; // Import trait for COUNT constant

    match dtype {
        GgmlDType::Q4_0 => {
            let block_size = <Q4_0 as DataBlock>::COUNT;
            let num_blocks = (data.len() + block_size - 1) / block_size;
            let mut result = Vec::with_capacity(num_blocks * std::mem::size_of::<Q4_0>());

            for chunk in data.chunks(block_size) {
                // Pad chunk to block_size if needed
                let mut padded = [0.0f32; 32]; // Q4_0::COUNT
                padded[..chunk.len()].copy_from_slice(chunk);

                let block = Q4_0::quantize(&padded);
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        &block as *const Q4_0 as *const u8,
                        std::mem::size_of::<Q4_0>(),
                    )
                };
                result.extend_from_slice(bytes);
            }

            Ok(result)
        }
        GgmlDType::Q4K => {
            let block_size = <Q4K as DataBlock>::COUNT;
            let num_blocks = (data.len() + block_size - 1) / block_size;
            let mut result = Vec::with_capacity(num_blocks * std::mem::size_of::<Q4K>());

            for chunk in data.chunks(block_size) {
                let mut padded = [0.0f32; 256]; // Q4K::COUNT
                padded[..chunk.len()].copy_from_slice(chunk);

                let block = Q4K::quantize(&padded);
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        &block as *const Q4K as *const u8,
                        std::mem::size_of::<Q4K>(),
                    )
                };
                result.extend_from_slice(bytes);
            }

            Ok(result)
        }
        GgmlDType::Q8_0 => {
            let block_size = <Q8_0 as DataBlock>::COUNT;
            let num_blocks = (data.len() + block_size - 1) / block_size;
            let mut result = Vec::with_capacity(num_blocks * std::mem::size_of::<Q8_0>());

            for chunk in data.chunks(block_size) {
                let mut padded = [0.0f32; 32]; // Q8_0::COUNT
                padded[..chunk.len()].copy_from_slice(chunk);

                let block = Q8_0::quantize(&padded);
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        &block as *const Q8_0 as *const u8,
                        std::mem::size_of::<Q8_0>(),
                    )
                };
                result.extend_from_slice(bytes);
            }

            Ok(result)
        }
        _ => anyhow::bail!("Quantization for {:?} not yet implemented", dtype),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_trip_q4_0() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32) / 256.0).collect();
        let quantized = quantize_tensor(&data, GgmlDType::Q4_0).unwrap();
        let dequantized = dequantize_tensor(&quantized, GgmlDType::Q4_0, 256).unwrap();

        assert_eq!(data.len(), dequantized.len());

        // Check approximate equality (quantization loses precision)
        for (orig, deq) in data.iter().zip(dequantized.iter()) {
            let error = (orig - deq).abs();
            assert!(error < 0.1, "Error too large: {} vs {}", orig, deq);
        }
    }

    #[test]
    fn test_round_trip_q8_0() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32) / 256.0).collect();
        let quantized = quantize_tensor(&data, GgmlDType::Q8_0).unwrap();
        let dequantized = dequantize_tensor(&quantized, GgmlDType::Q8_0, 256).unwrap();

        assert_eq!(data.len(), dequantized.len());

        // Q8_0 should have much better precision than Q4_0
        for (orig, deq) in data.iter().zip(dequantized.iter()) {
            let error = (orig - deq).abs();
            assert!(error < 0.01, "Error too large: {} vs {}", orig, deq);
        }
    }

    #[test]
    fn test_dtype_properties() {
        assert_eq!(GgmlDType::Q4_0.block_size(), 32);
        assert_eq!(GgmlDType::Q4K.block_size(), 256);
        assert_eq!(GgmlDType::Q4_0.name(), "Q4_0");
        assert_eq!(GgmlDType::from_u32(12), Some(GgmlDType::Q4K));
    }
}
