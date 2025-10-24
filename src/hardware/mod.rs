//! Hardware detection and adaptive configuration
//!
//! This module provides hardware capability detection and automatic configuration
//! for optimal model performance. It uses the `system-analysis` crate for
//! comprehensive hardware profiling.
//!
//! # Features
//!
//! - CPU core count and architecture detection
//! - Memory availability and bandwidth assessment
//! - GPU detection (CUDA, ROCm) with VRAM capacity
//! - Automatic batch size calculation based on available resources
//! - Model size recommendations based on hardware constraints
//! - Backend selection (CPU/CUDA/ROCm) based on detected capabilities

pub mod batch_sizing;
pub mod model_selection;

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Hardware profile summary for inference configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    /// CPU information
    pub cpu: CpuInfo,
    /// Memory information
    pub memory: MemoryInfo,
    /// GPU information (if available)
    pub gpu: Option<GpuInfo>,
    /// Overall AI/ML workload suitability score (0-10)
    pub ml_score: f64,
}

/// CPU capability information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    /// Number of physical CPU cores
    pub physical_cores: usize,
    /// Number of logical cores (with hyperthreading)
    pub logical_cores: usize,
    /// CPU architecture (x86_64, aarch64, etc.)
    pub architecture: String,
    /// CPU model name
    pub model_name: String,
}

/// Memory capability information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    /// Total system RAM in bytes
    pub total_bytes: u64,
    /// Available RAM in bytes (free + buffers/cache)
    pub available_bytes: u64,
    /// Estimated memory bandwidth in GB/s (if detectable)
    pub bandwidth_gbs: Option<f64>,
}

/// GPU capability information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    /// GPU name/model
    pub name: String,
    /// VRAM capacity in bytes
    pub vram_bytes: u64,
    /// Backend type (CUDA, ROCm, Metal, etc.)
    pub backend: GpuBackend,
    /// GPU compute capability/generation (if applicable)
    pub compute_capability: Option<String>,
}

/// Supported GPU backends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuBackend {
    /// NVIDIA CUDA
    Cuda,
    /// AMD ROCm
    Rocm,
    /// Apple Metal
    Metal,
    /// Vulkan compute
    Vulkan,
}

impl HardwareProfile {
    /// Get total system memory (platform-specific implementation)
    #[cfg(target_os = "windows")]
    fn get_total_memory() -> u64 {
        // On Windows, estimate based on environment
        // TODO: Use proper Windows API calls
        16 * 1024 * 1024 * 1024 // Default to 16GB estimate
    }

    #[cfg(target_os = "linux")]
    fn get_total_memory() -> u64 {
        // On Linux, read from /proc/meminfo
        use std::fs;
        if let Ok(contents) = fs::read_to_string("/proc/meminfo") {
            for line in contents.lines() {
                if let Some(mem_str) = line.strip_prefix("MemTotal:") {
                    if let Some(kb_str) = mem_str.trim().split_whitespace().next() {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            return kb * 1024; // Convert KB to bytes
                        }
                    }
                }
            }
        }
        16 * 1024 * 1024 * 1024 // Fallback to 16GB
    }

    #[cfg(target_os = "macos")]
    fn get_total_memory() -> u64 {
        // On macOS, use sysctl
        // TODO: Implement proper sysctl call
        16 * 1024 * 1024 * 1024 // Default to 16GB estimate
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    fn get_total_memory() -> u64 {
        16 * 1024 * 1024 * 1024 // Default to 16GB estimate
    }

    /// Detect current hardware capabilities
    ///
    /// Uses the `system-analysis` crate to probe CPU, memory, and GPU resources.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let profile = HardwareProfile::detect()?;
    /// println!("CPU: {} cores, RAM: {} GB",
    ///     profile.cpu.physical_cores,
    ///     profile.memory.total_bytes / (1024*1024*1024));
    /// ```
    pub fn detect() -> Result<Self> {
        use system_analysis::SystemAnalyzer;

        // Create analyzer and gather system information
        let mut analyzer = SystemAnalyzer::new();

        // Run async analysis in a blocking context
        let system_profile = tokio::runtime::Runtime::new()
            .map_err(|e| anyhow::anyhow!("Failed to create tokio runtime: {}", e))?
            .block_on(analyzer.analyze_system())
            .map_err(|e| anyhow::anyhow!("Failed to analyze system: {}", e))?;

        // Access the system_info from the profile
        let sys_info = &system_profile.system_info;

        // Extract CPU info
        let cpu = CpuInfo {
            physical_cores: sys_info.cpu_info.physical_cores,
            logical_cores: sys_info.cpu_info.logical_cores,
            architecture: sys_info.cpu_info.architecture.clone(),
            model_name: sys_info.cpu_info.brand.clone(),
        };

        // Extract memory info
        let memory = MemoryInfo {
            total_bytes: sys_info.memory_info.total_ram * 1024 * 1024, // Convert MB to bytes
            available_bytes: sys_info.memory_info.available_ram * 1024 * 1024, // Convert MB to bytes
            bandwidth_gbs: sys_info.memory_info.memory_speed.map(|speed_mhz| {
                // Estimate bandwidth from memory speed (rough approximation)
                // DDR transfers data on both clock edges, typical 64-bit bus
                (speed_mhz as f64 * 2.0 * 8.0) / 1000.0
            }),
        };

        // Extract GPU info (if available)
        let gpu = if !sys_info.gpu_info.is_empty() {
            let g = &sys_info.gpu_info[0]; // Use first GPU

            let backend = if g.vendor.to_lowercase().contains("nvidia") || g.cuda_support {
                GpuBackend::Cuda
            } else if g.vendor.to_lowercase().contains("amd") {
                GpuBackend::Rocm
            } else if g.vendor.to_lowercase().contains("apple") {
                GpuBackend::Metal
            } else {
                GpuBackend::Vulkan
            };

            Some(GpuInfo {
                name: g.name.clone(),
                vram_bytes: g.vram_size.unwrap_or(0) * 1024 * 1024, // Convert MB to bytes
                backend,
                compute_capability: g.compute_capability.clone(),
            })
        } else {
            None
        };

        // Use the AI workload score as ML score
        let ml_score = system_profile.ai_workload_score;

        Ok(Self {
            cpu,
            memory,
            gpu,
            ml_score,
        })
    }

    /// Get recommended backend for inference
    ///
    /// Returns the optimal backend based on detected hardware:
    /// - CUDA if NVIDIA GPU available
    /// - ROCm if AMD GPU available
    /// - Metal if Apple GPU available
    /// - CPU otherwise
    pub fn recommended_backend(&self) -> InferenceBackend {
        if let Some(gpu) = &self.gpu {
            match gpu.backend {
                GpuBackend::Cuda => InferenceBackend::Cuda,
                GpuBackend::Rocm => InferenceBackend::Rocm,
                GpuBackend::Metal => InferenceBackend::Metal,
                GpuBackend::Vulkan => InferenceBackend::Vulkan,
            }
        } else {
            InferenceBackend::Cpu
        }
    }

    /// Get human-readable hardware summary
    pub fn summary(&self) -> String {
        let mut s = String::new();

        s.push_str(&format!(
            "CPU: {} ({} cores, {} threads)\n",
            self.cpu.model_name, self.cpu.physical_cores, self.cpu.logical_cores
        ));

        s.push_str(&format!(
            "RAM: {:.1} GB total, {:.1} GB available\n",
            self.memory.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            self.memory.available_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        ));

        if let Some(gpu) = &self.gpu {
            s.push_str(&format!(
                "GPU: {} ({:.1} GB VRAM, {:?})\n",
                gpu.name,
                gpu.vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                gpu.backend
            ));
        } else {
            s.push_str("GPU: None\n");
        }

        s.push_str(&format!("ML Score: {:.1}/10", self.ml_score));

        s
    }
}

/// Inference backend options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InferenceBackend {
    /// CPU-only execution
    Cpu,
    /// NVIDIA CUDA GPU
    Cuda,
    /// AMD ROCm GPU
    Rocm,
    /// Apple Metal GPU
    Metal,
    /// Vulkan compute GPU
    Vulkan,
}

impl InferenceBackend {
    /// Convert to Candle device
    pub fn to_device(&self) -> candle_core::Device {
        match self {
            Self::Cpu => candle_core::Device::Cpu,
            Self::Cuda => {
                candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu)
            }
            Self::Rocm => candle_core::Device::Cpu, // TODO: Add ROCm support to Candle
            Self::Metal => candle_core::Device::new_metal(0).unwrap_or(candle_core::Device::Cpu),
            Self::Vulkan => candle_core::Device::Cpu, // TODO: Add Vulkan support
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_detection() {
        let profile = HardwareProfile::detect().expect("Hardware detection failed");

        // Basic sanity checks
        assert!(profile.cpu.physical_cores > 0);
        assert!(profile.cpu.logical_cores >= profile.cpu.physical_cores);
        assert!(profile.memory.total_bytes > 0);
        assert!(profile.memory.available_bytes > 0);
        assert!(profile.memory.available_bytes <= profile.memory.total_bytes);
        assert!(profile.ml_score >= 0.0 && profile.ml_score <= 10.0);

        println!("Detected hardware:\n{}", profile.summary());
    }

    #[test]
    fn test_backend_recommendation() {
        let profile = HardwareProfile::detect().expect("Hardware detection failed");
        let backend = profile.recommended_backend();

        println!("Recommended backend: {:?}", backend);

        // Should never recommend a backend we can't use
        if profile.gpu.is_none() {
            assert_eq!(backend, InferenceBackend::Cpu);
        }
    }
}
