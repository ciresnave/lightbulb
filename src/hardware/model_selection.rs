//! Adaptive model selection based on hardware capabilities
//!
//! This module maps hardware profiles to optimal model choices, automatically
//! selecting model size, quantization level, and backend based on available resources.
//!
//! # Model Recommendations
//!
//! - **4GB RAM**: TinyLlama 1.1B (Q4 quantization)
//! - **8GB RAM**: Phi-3 Mini (Q4/Q8 quantization)
//! - **16GB RAM**: Mistral 7B (F16/Q8 quantization)
//! - **32GB+ RAM**: Llama 3.2 11B or Llama 3.3 70B (F16 quantization)
//!
//! # Backend Selection
//!
//! - **NVIDIA GPU**: CUDA backend (highest priority)
//! - **AMD GPU**: ROCm backend (if available)
//! - **Apple Silicon**: Metal backend
//! - **CPU only**: Optimized CPU execution with quantization

use crate::hardware::{HardwareProfile, InferenceBackend};
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Recommended model configuration for detected hardware
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRecommendation {
    /// Model name/identifier
    pub model_name: String,
    /// Model parameter count (in billions)
    pub param_count_b: f64,
    /// Recommended data type
    pub dtype: DataType,
    /// Recommended backend
    pub backend: InferenceBackend,
    /// Estimated inference throughput (tokens/sec)
    pub estimated_throughput: f64,
    /// Confidence score (0.0-1.0) for this recommendation
    pub confidence: f64,
    /// Human-readable explanation
    pub rationale: String,
}

/// Supported data types for model weights
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    /// 32-bit floating point (highest precision, most memory)
    F32,
    /// 16-bit floating point (good balance)
    F16,
    /// Brain float 16 (mixed precision training format)
    BF16,
    /// 8-bit quantized (Q8)
    Q8,
    /// 4-bit quantized (Q4 - most compact)
    Q4,
}

impl DataType {
    /// Get bytes per parameter for this data type
    pub fn bytes_per_param(&self) -> f64 {
        match self {
            Self::F32 => 4.0,
            Self::F16 | Self::BF16 => 2.0,
            Self::Q8 => 1.0,
            Self::Q4 => 0.5,
        }
    }

    /// Convert to Candle dtype string
    pub fn to_candle_str(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::Q8 | Self::Q4 => "f16", // Quantized models still use f16 for compute
        }
    }
}

/// Model size category
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSize {
    /// ~1B parameters (TinyLlama, Phi-2)
    Tiny,
    /// ~3-4B parameters (Phi-3, Llama 3B)
    Small,
    /// ~7-8B parameters (Mistral, Llama 2 7B)
    Medium,
    /// ~11-13B parameters (Llama 3.2 11B, Vicuna 13B)
    Large,
    /// ~70B+ parameters (Llama 3.3 70B, Mixtral 8x7B)
    Huge,
}

impl ModelSize {
    /// Get parameter count for this size category
    pub fn param_count_b(&self) -> f64 {
        match self {
            Self::Tiny => 1.1,
            Self::Small => 3.0,
            Self::Medium => 7.0,
            Self::Large => 11.0,
            Self::Huge => 70.0,
        }
    }

    /// Estimate memory required for model weights (in bytes)
    pub fn memory_bytes(&self, dtype: DataType) -> u64 {
        let params = self.param_count_b() * 1_000_000_000.0;
        (params * dtype.bytes_per_param()) as u64
    }

    /// Get recommended model name for this size
    pub fn recommended_model(&self) -> &'static str {
        match self {
            Self::Tiny => "TinyLlama-1.1B-Chat",
            Self::Small => "Phi-3-Mini-4K",
            Self::Medium => "Mistral-7B-Instruct",
            Self::Large => "Llama-3.2-11B-Instruct",
            Self::Huge => "Llama-3.3-70B-Instruct",
        }
    }
}

/// Select optimal model based on hardware profile
///
/// This implements the core decision tree:
///
/// 1. **Memory constraints**: Filter models that fit in available RAM/VRAM
/// 2. **Backend optimization**: Prefer GPU if available, quantize for CPU
/// 3. **Throughput estimation**: Predict tokens/sec based on hardware
/// 4. **Confidence scoring**: Rate recommendation reliability
///
/// # Example
///
/// ```ignore
/// let profile = HardwareProfile::detect()?;
/// let recommendation = recommend_model(&profile)?;
/// println!("Recommended: {} ({:.1}B params, {:?})",
///     recommendation.model_name,
///     recommendation.param_count_b,
///     recommendation.dtype);
/// ```
pub fn recommend_model(profile: &HardwareProfile) -> Result<ModelRecommendation> {
    // Determine available memory (prefer GPU VRAM if available)
    let available_memory_gb = if let Some(gpu) = &profile.gpu {
        gpu.vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    } else {
        profile.memory.available_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    };

    // Select backend
    let backend = profile.recommended_backend();
    let use_gpu = backend != InferenceBackend::Cpu;

    // Model selection logic based on available memory
    let (model_size, dtype) = if available_memory_gb < 5.0 {
        // 4GB: TinyLlama with Q4 quantization
        (ModelSize::Tiny, DataType::Q4)
    } else if available_memory_gb < 9.0 {
        // 8GB: Phi-3 with Q4/Q8
        (
            ModelSize::Small,
            if use_gpu { DataType::Q8 } else { DataType::Q4 },
        )
    } else if available_memory_gb < 18.0 {
        // 16GB: Mistral 7B with F16/Q8
        (
            ModelSize::Medium,
            if use_gpu { DataType::F16 } else { DataType::Q8 },
        )
    } else if available_memory_gb < 35.0 {
        // 32GB: Llama 11B with F16
        (ModelSize::Large, DataType::F16)
    } else {
        // 64GB+: Llama 70B with F16
        (ModelSize::Huge, DataType::F16)
    };

    // Estimate throughput based on hardware
    let estimated_throughput = estimate_throughput(profile, model_size, dtype);

    // Calculate confidence based on hardware suitability
    let confidence = calculate_confidence(profile, model_size, dtype);

    // Build rationale
    let rationale = format!(
        "{:.1} GB {} available, {} CPU cores, ML score {:.1}/10. {} backend {}.",
        available_memory_gb,
        if use_gpu { "VRAM" } else { "RAM" },
        profile.cpu.physical_cores,
        profile.ml_score,
        if use_gpu { "GPU" } else { "CPU" },
        if use_gpu {
            "with hardware acceleration"
        } else {
            "with quantization"
        }
    );

    Ok(ModelRecommendation {
        model_name: model_size.recommended_model().to_string(),
        param_count_b: model_size.param_count_b(),
        dtype,
        backend,
        estimated_throughput,
        confidence,
        rationale,
    })
}

/// Estimate inference throughput for given hardware and model
///
/// Returns estimated tokens/second based on:
/// - Hardware capabilities (CPU cores, GPU FLOPS)
/// - Model size (parameter count)
/// - Data type (precision)
/// - Backend (CPU vs GPU acceleration)
fn estimate_throughput(profile: &HardwareProfile, model_size: ModelSize, dtype: DataType) -> f64 {
    // Base throughput factors (tokens/sec per billion parameters)
    let base_cpu_throughput = 0.5; // CPU: ~0.5 tok/s per billion params
    let base_gpu_throughput = 5.0; // GPU: ~5 tok/s per billion params

    // Model size penalty (larger models are slower per param)
    let size_factor = match model_size {
        ModelSize::Tiny => 2.0, // Smaller models have overhead
        ModelSize::Small => 1.5,
        ModelSize::Medium => 1.0,
        ModelSize::Large => 0.8,
        ModelSize::Huge => 0.5, // Large models memory-bound
    };

    // Quantization speedup
    let quant_factor = match dtype {
        DataType::F32 => 1.0,
        DataType::F16 | DataType::BF16 => 1.5,
        DataType::Q8 => 2.0,
        DataType::Q4 => 2.5,
    };

    // CPU core scaling (diminishing returns)
    let cpu_scale = (profile.cpu.physical_cores as f64).sqrt();

    // Calculate throughput
    if profile.gpu.is_some() {
        // GPU path
        let gpu_boost = 1.0 + (profile.ml_score / 10.0); // ML score affects GPU utilization
        base_gpu_throughput * size_factor * quant_factor * gpu_boost
    } else {
        // CPU path
        base_cpu_throughput * size_factor * quant_factor * cpu_scale
    }
}

/// Calculate confidence score for recommendation
///
/// Higher confidence when:
/// - Hardware ML score is high
/// - Memory headroom is ample (>30%)
/// - GPU available for larger models
fn calculate_confidence(profile: &HardwareProfile, model_size: ModelSize, dtype: DataType) -> f64 {
    let mut confidence = 0.5; // Base confidence

    // ML score contributes (0-4 points → 0-0.4 confidence)
    confidence += profile.ml_score / 10.0 * 0.4;

    // Memory headroom check
    let model_memory = model_size.memory_bytes(dtype);
    let available_memory = if let Some(gpu) = &profile.gpu {
        gpu.vram_bytes
    } else {
        profile.memory.available_bytes
    };

    let memory_ratio = model_memory as f64 / available_memory as f64;
    if memory_ratio < 0.5 {
        confidence += 0.2; // Plenty of headroom
    } else if memory_ratio < 0.7 {
        confidence += 0.1; // Adequate headroom
    } else {
        confidence -= 0.1; // Tight fit
    }

    // GPU availability for large models
    if model_size == ModelSize::Large || model_size == ModelSize::Huge {
        if profile.gpu.is_some() {
            confidence += 0.1; // GPU helps with large models
        } else {
            confidence -= 0.1; // CPU struggles with large models
        }
    }

    confidence.clamp(0.0, 1.0)
}

/// Get all viable model options for hardware (not just top recommendation)
///
/// Returns a sorted list of models that could run on the hardware,
/// ordered by estimated performance.
pub fn list_viable_models(profile: &HardwareProfile) -> Result<Vec<ModelRecommendation>> {
    let mut recommendations = Vec::new();

    // Try each model size
    for model_size in [
        ModelSize::Tiny,
        ModelSize::Small,
        ModelSize::Medium,
        ModelSize::Large,
        ModelSize::Huge,
    ] {
        // Try each quantization level
        for dtype in [DataType::Q4, DataType::Q8, DataType::F16, DataType::F32] {
            let model_memory = model_size.memory_bytes(dtype);
            let available = if let Some(gpu) = &profile.gpu {
                gpu.vram_bytes
            } else {
                profile.memory.available_bytes
            };

            // Only include if model fits with headroom
            if model_memory < (available as f64 * 0.7) as u64 {
                let backend = profile.recommended_backend();
                let throughput = estimate_throughput(profile, model_size, dtype);
                let confidence = calculate_confidence(profile, model_size, dtype);

                recommendations.push(ModelRecommendation {
                    model_name: model_size.recommended_model().to_string(),
                    param_count_b: model_size.param_count_b(),
                    dtype,
                    backend,
                    estimated_throughput: throughput,
                    confidence,
                    rationale: format!(
                        "{:.1}B params, {:?}, {:.1} tok/s estimated",
                        model_size.param_count_b(),
                        dtype,
                        throughput
                    ),
                });
            }
        }
    }

    // Sort by estimated throughput (highest first)
    recommendations.sort_by(|a, b| {
        b.estimated_throughput
            .partial_cmp(&a.estimated_throughput)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(recommendations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{CpuInfo, GpuBackend, GpuInfo, MemoryInfo};

    fn make_test_profile(ram_gb: u64, has_gpu: bool, gpu_vram_gb: u64) -> HardwareProfile {
        HardwareProfile {
            cpu: CpuInfo {
                physical_cores: 8,
                logical_cores: 16,
                architecture: "x86_64".to_string(),
                model_name: "Test CPU".to_string(),
            },
            memory: MemoryInfo {
                total_bytes: ram_gb * 1024 * 1024 * 1024,
                available_bytes: (ram_gb as f64 * 0.75) as u64 * 1024 * 1024 * 1024,
                bandwidth_gbs: Some(40.0),
            },
            gpu: if has_gpu {
                Some(GpuInfo {
                    name: "Test GPU".to_string(),
                    vram_bytes: gpu_vram_gb * 1024 * 1024 * 1024,
                    backend: GpuBackend::Cuda,
                    compute_capability: Some("8.0".to_string()),
                })
            } else {
                None
            },
            ml_score: if has_gpu { 8.0 } else { 5.0 },
        }
    }

    #[test]
    fn test_model_recommendations() {
        // Test different hardware profiles
        let profiles = vec![
            ("4GB RAM, no GPU", make_test_profile(4, false, 0)),
            ("8GB RAM, no GPU", make_test_profile(8, false, 0)),
            ("16GB RAM, no GPU", make_test_profile(16, false, 0)),
            ("32GB RAM, 24GB GPU", make_test_profile(32, true, 24)),
            ("64GB RAM, 80GB GPU", make_test_profile(64, true, 80)),
        ];

        for (name, profile) in profiles {
            let rec = recommend_model(&profile).unwrap();
            println!("\n{}", name);
            println!(
                "  Model: {} ({:.1}B params)",
                rec.model_name, rec.param_count_b
            );
            println!("  Type: {:?}, Backend: {:?}", rec.dtype, rec.backend);
            println!("  Throughput: {:.1} tok/s", rec.estimated_throughput);
            println!("  Confidence: {:.0}%", rec.confidence * 100.0);
            println!("  Rationale: {}", rec.rationale);

            // Verify recommendations make sense
            assert!(rec.confidence > 0.3); // Reasonable confidence
            assert!(rec.estimated_throughput > 0.0);
        }
    }

    #[test]
    fn test_memory_scaling() {
        // Verify model selection scales with available memory
        let profile_4gb = make_test_profile(4, false, 0);
        let profile_16gb = make_test_profile(16, false, 0);
        let profile_64gb = make_test_profile(64, true, 80);

        let rec_4gb = recommend_model(&profile_4gb).unwrap();
        let rec_16gb = recommend_model(&profile_16gb).unwrap();
        let rec_64gb = recommend_model(&profile_64gb).unwrap();

        // Should recommend larger models with more memory
        assert!(rec_4gb.param_count_b < rec_16gb.param_count_b);
        assert!(rec_16gb.param_count_b < rec_64gb.param_count_b);

        println!("\nMemory scaling:");
        println!("  4GB: {:.1}B params", rec_4gb.param_count_b);
        println!("  16GB: {:.1}B params", rec_16gb.param_count_b);
        println!("  64GB: {:.1}B params", rec_64gb.param_count_b);
    }

    #[test]
    fn test_viable_models_list() {
        let profile = make_test_profile(16, false, 0);
        let viable = list_viable_models(&profile).unwrap();

        assert!(!viable.is_empty());
        println!("\nViable models for 16GB RAM:");
        for (i, rec) in viable.iter().take(5).enumerate() {
            println!(
                "  {}. {} ({:.1}B, {:?}) - {:.1} tok/s",
                i + 1,
                rec.model_name,
                rec.param_count_b,
                rec.dtype,
                rec.estimated_throughput
            );
        }

        // Verify sorting (highest throughput first)
        for i in 1..viable.len() {
            assert!(viable[i - 1].estimated_throughput >= viable[i].estimated_throughput);
        }
    }

    #[test]
    fn test_quantization_benefits() {
        // Same model, different quantization levels
        let model_size = ModelSize::Medium; // 7B

        let f32_mem = model_size.memory_bytes(DataType::F32);
        let f16_mem = model_size.memory_bytes(DataType::F16);
        let q8_mem = model_size.memory_bytes(DataType::Q8);
        let q4_mem = model_size.memory_bytes(DataType::Q4);

        println!("\n7B model memory requirements:");
        println!(
            "  F32: {:.1} GB",
            f32_mem as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        println!(
            "  F16: {:.1} GB",
            f16_mem as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        println!("  Q8: {:.1} GB", q8_mem as f64 / (1024.0 * 1024.0 * 1024.0));
        println!("  Q4: {:.1} GB", q4_mem as f64 / (1024.0 * 1024.0 * 1024.0));

        // Verify quantization reduces memory (F32 > F16 > Q8 > Q4)
        assert!(f32_mem > f16_mem);
        assert!(f16_mem > q8_mem);
        assert!(q8_mem > q4_mem);
    }
}
