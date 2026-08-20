//! Core Memory Estimation Types
//!
//! Provides structs for calculating memory usage of models, caches, and activations.

use candlelight::core::DType;

/// Memory usage for model weights
#[derive(Debug, Clone)]
pub enum WeightMemory {
    /// Unquantized weights (FP32, FP16, BF16)
    Unquantized { dtype: DType, num_parameters: usize },

    /// Quantized weights (AWQ, GPTQ, Marlin)
    Quantized {
        /// Original dtype before quantization
        original_dtype: DType,
        /// Number of bits per weight (4, 8)
        bits: usize,
        /// Total parameters
        num_parameters: usize,
        /// Quantization format overhead (scales, zero-points, etc.)
        metadata_bytes: usize,
    },
}

impl WeightMemory {
    /// Calculate weight memory in bytes
    pub fn bytes(&self) -> usize {
        match self {
            WeightMemory::Unquantized {
                dtype,
                num_parameters,
            } => {
                let bytes_per_param = match dtype {
                    DType::F32 => 4,
                    DType::F16 | DType::BF16 => 2,
                    DType::U8 => 1,
                    _ => 4, // Conservative default
                };
                num_parameters * bytes_per_param
            }

            WeightMemory::Quantized {
                bits,
                num_parameters,
                metadata_bytes,
                ..
            } => {
                // Quantized weights (packed)
                let weight_bytes = (num_parameters * bits) / 8;

                // Add metadata (scales, zeros, group indices)
                weight_bytes + metadata_bytes
            }
        }
    }

    /// Estimate metadata size for quantized format
    ///
    /// # Arguments
    ///
    /// * `num_parameters` - Total number of parameters
    /// * `group_size` - Quantization group size (-1 for per-channel)
    /// * `quant_method` - "awq", "gptq", or "marlin"
    pub fn estimate_metadata(num_parameters: usize, group_size: i32, quant_method: &str) -> usize {
        if group_size <= 0 {
            // Per-channel quantization
            let num_channels = (num_parameters as f64).sqrt() as usize;
            num_channels * 6 // 2 bytes scale + 4 bytes for overhead
        } else {
            // Group quantization
            let num_groups = num_parameters / group_size as usize;
            let scale_bytes = num_groups * 2; // FP16 scales
            let zero_bytes = if quant_method == "gptq" {
                (num_groups * 4) / 8 // 4-bit zeros, packed
            } else {
                0 // AWQ doesn't use zero-points
            };
            scale_bytes + zero_bytes
        }
    }
}

/// Memory usage for KV cache
#[derive(Debug, Clone)]
pub struct KvCacheMemory {
    /// Batch size (number of concurrent sequences)
    pub batch_size: usize,

    /// Maximum sequence length
    pub max_seq_len: usize,

    /// Number of layers
    pub num_layers: usize,

    /// Number of KV heads
    pub num_kv_heads: usize,

    /// Head dimension
    pub head_dim: usize,

    /// Data type (F16, BF16, F32)
    pub dtype: DType,
}

impl KvCacheMemory {
    /// Calculate KV cache memory in bytes
    pub fn bytes(&self) -> usize {
        let bytes_per_element = match self.dtype {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            _ => 2, // Default to FP16
        };

        // Cache shape: [num_layers, 2 (K+V), batch_size, num_kv_heads, max_seq_len, head_dim]
        self.num_layers
            * 2 // K and V
            * self.batch_size
            * self.num_kv_heads
            * self.max_seq_len
            * self.head_dim
            * bytes_per_element
    }
}

/// Memory usage for activations during forward pass
#[derive(Debug, Clone)]
pub struct ActivationMemory {
    /// Batch size
    pub batch_size: usize,

    /// Sequence length
    pub seq_len: usize,

    /// Hidden size
    pub hidden_size: usize,

    /// Intermediate size (MLP)
    pub intermediate_size: usize,

    /// Number of layers
    pub num_layers: usize,

    /// Data type
    pub dtype: DType,
}

impl ActivationMemory {
    /// Estimate activation memory (peak usage during forward pass)
    pub fn bytes(&self) -> usize {
        let bytes_per_element = match self.dtype {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            _ => 2,
        };

        // Peak activations (conservative estimate):
        // - Input embeddings: batch × seq × hidden
        // - Attention QKV: batch × seq × hidden × 3
        // - Attention output: batch × seq × hidden
        // - MLP intermediate: batch × seq × intermediate × 2
        // - MLP output: batch × seq × hidden
        // Only one layer active at a time (no layer parallelism)

        let input_embed = self.batch_size * self.seq_len * self.hidden_size;
        let attn_qkv = self.batch_size * self.seq_len * self.hidden_size * 3;
        let attn_out = self.batch_size * self.seq_len * self.hidden_size;
        let mlp_intermediate = self.batch_size * self.seq_len * self.intermediate_size * 2;
        let mlp_out = self.batch_size * self.seq_len * self.hidden_size;

        let peak_elements = input_embed + attn_qkv + attn_out + mlp_intermediate + mlp_out;

        peak_elements * bytes_per_element
    }
}

/// Complete memory estimate for a model
#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    /// Model weight memory
    pub weights: WeightMemory,

    /// KV cache memory
    pub kv_cache: KvCacheMemory,

    /// Activation memory (peak during inference)
    pub activations: ActivationMemory,

    /// Additional overhead (buffers, workspace tensors, etc.)
    pub overhead_bytes: usize,
}

impl MemoryEstimate {
    /// Total memory required in bytes
    pub fn total_bytes(&self) -> usize {
        self.weights.bytes()
            + self.kv_cache.bytes()
            + self.activations.bytes()
            + self.overhead_bytes
    }

    /// Format as human-readable string
    pub fn display(&self) -> String {
        format!(
            "Weights: {}, KV Cache: {}, Activations: {}, Overhead: {}, Total: {}",
            crate::memory::utils::format_bytes(self.weights.bytes()),
            crate::memory::utils::format_bytes(self.kv_cache.bytes()),
            crate::memory::utils::format_bytes(self.activations.bytes()),
            crate::memory::utils::format_bytes(self.overhead_bytes),
            crate::memory::utils::format_bytes(self.total_bytes()),
        )
    }

    /// Check if estimate fits in available memory
    pub fn fits_in(&self, available_bytes: usize) -> bool {
        self.total_bytes() < available_bytes
    }

    /// Calculate memory utilization (percentage of available memory used)
    pub fn utilization(&self, available_bytes: usize) -> f64 {
        self.total_bytes() as f64 / available_bytes as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_memory_fp16() {
        let weights = WeightMemory::Unquantized {
            dtype: DType::F16,
            num_parameters: 7_000_000_000,
        };
        assert_eq!(weights.bytes(), 14_000_000_000); // 14GB
    }

    #[test]
    fn test_weight_memory_awq() {
        let weights = WeightMemory::Quantized {
            original_dtype: DType::F16,
            bits: 4,
            num_parameters: 7_000_000_000,
            metadata_bytes: 109_000_000, // ~109MB scales
        };
        // 7B × 0.5 bytes + 109MB ≈ 3.609GB
        assert!((weights.bytes() as f64 - 3.609e9).abs() < 1e8);
    }

    #[test]
    fn test_kv_cache_memory() {
        let cache = KvCacheMemory {
            batch_size: 1,
            max_seq_len: 2048,
            num_layers: 32,
            num_kv_heads: 32,
            head_dim: 128,
            dtype: DType::F16,
        };
        // 32 × 2 × 1 × 32 × 2048 × 128 × 2 = 1,073,741,824 bytes ≈ 1GB
        assert_eq!(cache.bytes(), 1_073_741_824);
    }

    #[test]
    fn test_activation_memory() {
        let activations = ActivationMemory {
            batch_size: 1,
            seq_len: 128,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_layers: 32,
            dtype: DType::F16,
        };
        // Should be around 12MB for these dimensions
        let bytes = activations.bytes();
        assert!(bytes > 10_000_000); // > 10MB
        assert!(bytes < 15_000_000); // < 15MB
    }

    #[test]
    fn test_memory_estimate_utilization() {
        let estimate = MemoryEstimate {
            weights: WeightMemory::Unquantized {
                dtype: DType::F16,
                num_parameters: 1_000_000_000,
            },
            kv_cache: KvCacheMemory {
                batch_size: 1,
                max_seq_len: 2048,
                num_layers: 24,
                num_kv_heads: 32,
                head_dim: 128,
                dtype: DType::F16,
            },
            activations: ActivationMemory {
                batch_size: 1,
                seq_len: 128,
                hidden_size: 2048,
                intermediate_size: 5632,
                num_layers: 24,
                dtype: DType::F16,
            },
            overhead_bytes: 100 * 1024 * 1024, // 100MB
        };

        let available = 8 * 1024 * 1024 * 1024; // 8GB
        assert!(estimate.fits_in(available));
        assert!(estimate.utilization(available) < 0.5); // < 50% utilization
    }
}
