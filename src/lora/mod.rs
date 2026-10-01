//! LoRA (Low-Rank Adaptation) Support with Architecture-Aware Name Mapping
//!
//! This module provides support for loading, validating, and merging LoRA adapters
//! with automatic format detection and architecture-aware tensor name mapping.
//!
//! # Features
//!
//! - **Format Auto-Detection**: Automatically detects HuggingFace, Custom, and PEFT formats
//! - **Architecture-Aware Mapping**: Uses TensorNameMapper to handle varied naming conventions
//! - **Validation**: Checks shape compatibility before merging
//! - **Flexible Merging**: Configurable scaling factor for LoRA weight application
//!
//! # Example
//!
//! ```rust,ignore
//! use lightbulb::lora::LoraAdapter;
//!
//! // Load LoRA adapter with automatic format detection
//! let adapter = LoraAdapter::load(
//!     "path/to/lora",
//!     &base_model_tensor_names,
//! )?;
//!
//! // Validate compatibility
//! let report = adapter.validate(&base_model)?;
//! println!("Compatible components: {}", report.compatible.len());
//!
//! // Merge with scaling
//! adapter.merge_into(&mut base_model, 0.8)?;
//! ```

use anyhow::{Result, bail};
use candlelight::core::{Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

use crate::pruning::name_mapping::{ComponentType, ModelArchitecture, TensorNameMapper};

/// LoRA adapter with architecture-aware name mapping
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    /// Base model name mapper for architecture-aware translation
    base_mapper: TensorNameMapper,

    /// LoRA weights mapped to base model component names
    /// Key: base model tensor name, Value: LoRA weight matrices
    adapters: HashMap<String, LoraWeight>,

    /// Detected LoRA format
    format: LoraFormat,

    /// Configuration
    config: LoraConfig,
}

/// LoRA weight matrices (A and B) with scaling factor
#[derive(Debug, Clone)]
pub struct LoraWeight {
    /// Low-rank matrix A (r × d_in)
    pub a: Tensor,

    /// Low-rank matrix B (d_out × r)
    pub b: Tensor,

    /// Scaling factor (typically rank or learned value)
    pub alpha: f32,

    /// Rank dimension
    pub rank: usize,
}

/// The flat host arrays `fuel_nn::modules::lora::LoraLinear::new` expects,
/// bridged from [`LoraWeight`]'s HF/PyTorch-layout tensors by
/// [`LoraWeight::to_fuel_lora_arrays`].
pub struct FuelLoraArrays {
    /// `[in_features, rank]`, row-major.
    pub lora_a: std::sync::Arc<[f32]>,
    /// `[rank, out_features]`, row-major.
    pub lora_b: std::sync::Arc<[f32]>,
    pub rank: usize,
    pub in_features: usize,
    pub out_features: usize,
}

impl LoraWeight {
    /// Bridges this adapter's A/B matrices to the flat host arrays
    /// `fuel_nn::modules::lora::LoraLinear::new` expects (board item 106,
    /// PR 17 — the narrow piece of `lora/mod.rs` confirmed independent of PR
    /// 16's mlmf gating, since it depends only on `LoraLinear::new`'s INPUT
    /// shape, never on how `LoraAdapter::load` populates `self.a`/`self.b`).
    ///
    /// This crate stores `a`/`b` in HF/PyTorch's native layout (`a: [rank,
    /// in_features]`, `b: [out_features, rank]` — confirmed against
    /// `merge_into`'s own shape checks, not assumed). Fuel's `LoraLinear`
    /// wants each TRANSPOSED (`lora_a: [in_features, rank]`, `lora_b: [rank,
    /// out_features]` — confirmed against `fuel-nn/src/modules/lora.rs`'s own
    /// doc comment, which states this explicitly).
    pub fn to_fuel_lora_arrays(&self) -> Result<FuelLoraArrays> {
        let a_dims = self.a.dims();
        let b_dims = self.b.dims();
        if a_dims.len() != 2 || b_dims.len() != 2 {
            bail!(
                "LoraWeight::to_fuel_lora_arrays: expected 2D A/B matrices, got {:?}/{:?}",
                a_dims,
                b_dims
            );
        }
        let (rank, in_features) = (a_dims[0], a_dims[1]);
        let (out_features, b_rank) = (b_dims[0], b_dims[1]);
        if b_rank != rank {
            bail!("LoraWeight::to_fuel_lora_arrays: A rank {rank} does not match B rank {b_rank}");
        }

        let lora_a: Vec<f32> = self.a.t()?.contiguous()?.flatten_all()?.to_vec1()?;
        let lora_b: Vec<f32> = self.b.t()?.contiguous()?.flatten_all()?.to_vec1()?;

        Ok(FuelLoraArrays {
            lora_a: lora_a.into(),
            lora_b: lora_b.into(),
            rank,
            in_features,
            out_features,
        })
    }
}

/// LoRA format variants with different naming conventions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoraFormat {
    /// HuggingFace format: "base_model.model.layers.0.self_attn.q_proj.lora_A"
    HuggingFace,

    /// Custom GGUF format: "lora.blk.0.attn_q.A"
    Custom,

    /// PEFT format: "peft.q_proj.adapter_A"
    Peft,

    /// Unknown format (requires manual mapping)
    Unknown,
}

/// LoRA adapter configuration
#[derive(Debug, Clone)]
pub struct LoraConfig {
    /// Default scaling factor when merging (0.0 to 1.0)
    pub default_scale: f32,

    /// Whether to validate shapes before merging
    pub validate_shapes: bool,

    /// Whether to log detailed mapping information
    pub verbose: bool,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            default_scale: 1.0,
            validate_shapes: true,
            verbose: false,
        }
    }
}

/// Validation report after checking LoRA compatibility
#[derive(Debug, Default, Clone)]
pub struct ValidationReport {
    /// Components that are compatible
    pub compatible: Vec<String>,

    /// Components with shape mismatches
    pub shape_mismatch: Vec<(String, String)>, // (component, reason)

    /// Components missing in base model
    pub missing_in_base: Vec<String>,

    /// Components in base but not adapted
    pub not_adapted: Vec<String>,
}

impl ValidationReport {
    /// Check if validation passed (no errors)
    pub fn is_valid(&self) -> bool {
        self.shape_mismatch.is_empty() && self.missing_in_base.is_empty()
    }

    /// Get total number of components checked
    pub fn total_components(&self) -> usize {
        self.compatible.len() + self.shape_mismatch.len() + self.missing_in_base.len()
    }
}

impl LoraAdapter {
    /// Create a new LoRA adapter from base model tensors
    ///
    /// This is the main entry point for loading LoRA adapters with automatic
    /// format detection and architecture-aware name mapping.
    ///
    /// # Arguments
    ///
    /// * `adapter_path` - Path to LoRA safetensors or GGUF file
    /// * `base_model_tensors` - List of base model tensor names for mapping
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let adapter = LoraAdapter::load(
    ///     "adapter.safetensors",
    ///     &model.tensor_names(),
    /// )?;
    /// ```
    pub fn load(adapter_path: &Path, base_model_tensors: &[String]) -> Result<Self> {
        Self::load_with_config(adapter_path, base_model_tensors, LoraConfig::default())
    }

    /// Load LoRA adapter with custom configuration
    pub fn load_with_config(
        adapter_path: &Path,
        base_model_tensors: &[String],
        config: LoraConfig,
    ) -> Result<Self> {
        // Create name mapper from base model tensors
        let base_mapper = TensorNameMapper::from_tensor_names(base_model_tensors)?;

        if config.verbose {
            println!("📦 Loading LoRA adapter from: {}", adapter_path.display());
            println!(
                "🏗️  Detected base model architecture: {:?}",
                base_mapper.architecture
            );
        }

        // Load LoRA tensors from file
        let lora_tensors = Self::load_lora_tensors(adapter_path)?;

        if config.verbose {
            println!("🔍 Found {} LoRA tensors", lora_tensors.len());
        }

        // Detect LoRA format from tensor names
        let format = Self::detect_lora_format(&lora_tensors)?;

        if config.verbose {
            println!("📋 Detected LoRA format: {:?}", format);
        }

        // Map LoRA tensor names to base model components
        let mut adapters = HashMap::new();
        let mut mapped_count = 0;
        let mut skipped_count = 0;

        for (lora_name, tensor) in lora_tensors {
            // Check if this is an A or B matrix
            let (base_name, is_a_matrix) = Self::parse_lora_tensor_name(&lora_name, format)?;

            // Map to base model component name
            if let Some(base_component) = Self::map_lora_to_base(&base_name, &base_mapper, format) {
                // Get or create LoRA weight entry
                let weight = adapters.entry(base_component.clone()).or_insert_with(|| {
                    // Initialize with placeholder rank
                    LoraWeight {
                        a: tensor.clone(),
                        b: tensor.clone(),
                        alpha: 1.0,
                        rank: 8, // Will be updated from actual tensors
                    }
                });

                // Assign A or B matrix
                if is_a_matrix {
                    weight.a = tensor;
                    weight.rank = weight.a.dims()[0]; // Rank is first dim of A
                } else {
                    weight.b = tensor;
                }

                mapped_count += 1;

                if config.verbose {
                    println!("  ✓ Mapped: {} -> {}", lora_name, base_component);
                }
            } else {
                skipped_count += 1;
                if config.verbose {
                    println!("  ⊘ Skipped: {} (no mapping)", lora_name);
                }
            }
        }

        if config.verbose {
            println!(
                "✅ Successfully mapped {}/{} LoRA tensors",
                mapped_count,
                mapped_count + skipped_count
            );
        }

        Ok(Self {
            base_mapper,
            adapters,
            format,
            config,
        })
    }

    /// Detect LoRA format from tensor names
    fn detect_lora_format(tensors: &HashMap<String, Tensor>) -> Result<LoraFormat> {
        let names: Vec<&str> = tensors.keys().map(|s| s.as_str()).collect();

        // Count patterns
        let mut hf_score = 0;
        let mut custom_score = 0;
        let mut peft_score = 0;

        for name in &names {
            // HuggingFace: "base_model.model.layers.N.component.lora_A/B"
            if name.contains("base_model.model.layers") && name.contains("lora_") {
                hf_score += 1;
            }

            // Custom GGUF: "lora.blk.N.component.A/B"
            if name.starts_with("lora.blk.") {
                custom_score += 1;
            }

            // PEFT: "peft.component.adapter_A/B"
            if name.starts_with("peft.") && name.contains("adapter_") {
                peft_score += 1;
            }
        }

        // Return format with highest score
        let max_score = hf_score.max(custom_score).max(peft_score);

        if max_score == 0 {
            Ok(LoraFormat::Unknown)
        } else if hf_score == max_score {
            Ok(LoraFormat::HuggingFace)
        } else if custom_score == max_score {
            Ok(LoraFormat::Custom)
        } else {
            Ok(LoraFormat::Peft)
        }
    }

    /// Parse LoRA tensor name to extract base name and matrix type
    ///
    /// Returns: (base_component_name, is_a_matrix)
    fn parse_lora_tensor_name(lora_name: &str, format: LoraFormat) -> Result<(String, bool)> {
        match format {
            LoraFormat::HuggingFace => {
                // "base_model.model.layers.5.self_attn.q_proj.lora_A" -> ("layers.5.self_attn.q_proj", true)
                if let Some(stripped) = lora_name.strip_prefix("base_model.model.") {
                    let is_a = stripped.ends_with(".lora_A");
                    let base = stripped
                        .strip_suffix(".lora_A")
                        .or_else(|| stripped.strip_suffix(".lora_B"))
                        .unwrap_or(stripped);
                    Ok((base.to_string(), is_a))
                } else {
                    bail!("Invalid HuggingFace LoRA name: {}", lora_name)
                }
            }
            LoraFormat::Custom => {
                // "lora.blk.0.attn_q.A" -> ("blk.0.attn_q", true)
                if let Some(stripped) = lora_name.strip_prefix("lora.") {
                    let is_a = stripped.ends_with(".A");
                    let base = stripped
                        .strip_suffix(".A")
                        .or_else(|| stripped.strip_suffix(".B"))
                        .unwrap_or(stripped);
                    Ok((base.to_string(), is_a))
                } else {
                    bail!("Invalid Custom LoRA name: {}", lora_name)
                }
            }
            LoraFormat::Peft => {
                // "peft.q_proj.adapter_A" -> ("q_proj", true)
                if let Some(stripped) = lora_name.strip_prefix("peft.") {
                    let is_a = stripped.ends_with(".adapter_A");
                    let base = stripped
                        .strip_suffix(".adapter_A")
                        .or_else(|| stripped.strip_suffix(".adapter_B"))
                        .unwrap_or(stripped);
                    Ok((base.to_string(), is_a))
                } else {
                    bail!("Invalid PEFT LoRA name: {}", lora_name)
                }
            }
            LoraFormat::Unknown => {
                bail!("Cannot parse unknown LoRA format: {}", lora_name)
            }
        }
    }

    /// Map LoRA component name to base model tensor name
    ///
    /// Uses TensorNameMapper to handle architecture-specific naming conventions
    fn map_lora_to_base(
        lora_component: &str,
        base_mapper: &TensorNameMapper,
        format: LoraFormat,
    ) -> Option<String> {
        match format {
            LoraFormat::HuggingFace => {
                // "layers.5.self_attn.q_proj" -> extract layer and component
                Self::map_huggingface_to_base(lora_component, base_mapper)
            }
            LoraFormat::Custom => {
                // "blk.0.attn_q" -> already close to base format
                Self::map_custom_to_base(lora_component, base_mapper)
            }
            LoraFormat::Peft => {
                // PEFT format is tricky - may need additional context
                // For now, return None (requires manual mapping)
                None
            }
            LoraFormat::Unknown => None,
        }
    }

    /// Map HuggingFace LoRA name to base model name
    fn map_huggingface_to_base(
        lora_component: &str,
        base_mapper: &TensorNameMapper,
    ) -> Option<String> {
        // Parse: "layers.5.self_attn.q_proj" -> layer=5, component="q_proj"
        let parts: Vec<&str> = lora_component.split('.').collect();

        // Find layer index
        let layer_idx = parts
            .iter()
            .position(|&p| p == "layers")
            .and_then(|i| parts.get(i + 1))
            .and_then(|s| s.parse::<usize>().ok())?;

        // Determine component type from the projection name
        let component_type = if lora_component.contains("q_proj") {
            ComponentType::AttentionQuery
        } else if lora_component.contains("k_proj") {
            ComponentType::AttentionKey
        } else if lora_component.contains("v_proj") {
            ComponentType::AttentionValue
        } else if lora_component.contains("o_proj") {
            ComponentType::AttentionOutput
        } else if lora_component.contains("gate_proj") {
            ComponentType::FfnGate
        } else if lora_component.contains("up_proj") {
            ComponentType::FfnUp
        } else if lora_component.contains("down_proj") {
            ComponentType::FfnDown
        } else {
            return None;
        };

        // Look up the actual tensor name from the mapping
        base_mapper
            .mappings
            .get(&(layer_idx, component_type))
            .cloned()
    }

    /// Map custom GGUF LoRA name to base model name
    fn map_custom_to_base(lora_component: &str, base_mapper: &TensorNameMapper) -> Option<String> {
        // "blk.0.attn_q" -> extract layer and component
        let parts: Vec<&str> = lora_component.split('.').collect();

        // Find layer index
        let layer_idx = parts
            .iter()
            .position(|&p| p == "blk")
            .and_then(|i| parts.get(i + 1))
            .and_then(|s| s.parse::<usize>().ok())?;

        // Determine component type from the component name
        let component_type = if lora_component.contains("attn_q") {
            ComponentType::AttentionQuery
        } else if lora_component.contains("attn_k") {
            ComponentType::AttentionKey
        } else if lora_component.contains("attn_v") {
            ComponentType::AttentionValue
        } else if lora_component.contains("attn_output") {
            ComponentType::AttentionOutput
        } else if lora_component.contains("ffn_gate") {
            ComponentType::FfnGate
        } else if lora_component.contains("ffn_up") {
            ComponentType::FfnUp
        } else if lora_component.contains("ffn_down") {
            ComponentType::FfnDown
        } else {
            return None;
        };

        // Look up the actual tensor name from the mapping
        base_mapper
            .mappings
            .get(&(layer_idx, component_type))
            .cloned()
    }

    /// Load LoRA tensors from safetensors file
    ///
    /// TODO: Add GGUF support when needed
    fn load_lora_tensors(path: &Path) -> Result<HashMap<String, Tensor>> {
        use candlelight::core::safetensors::load as load_safetensors;

        // Load safetensors file
        let tensors = load_safetensors(path, &Device::Cpu)?;

        Ok(tensors)
    }

    /// Validate LoRA adapter compatibility with base model
    ///
    /// Checks shape compatibility and reports any issues
    pub fn validate(&self, base_model_tensors: &HashMap<String, Tensor>) -> ValidationReport {
        let mut report = ValidationReport::default();

        // Check each adapted component
        for (component, lora_weight) in &self.adapters {
            if let Some(base_tensor) = base_model_tensors.get(component) {
                // Check shape compatibility
                if let Err(reason) = Self::check_shape_compatibility(base_tensor, lora_weight) {
                    report
                        .shape_mismatch
                        .push((component.clone(), reason.to_string()));
                } else {
                    report.compatible.push(component.clone());
                }
            } else {
                report.missing_in_base.push(component.clone());
            }
        }

        // Find components not adapted
        for component in base_model_tensors.keys() {
            if !self.adapters.contains_key(component) {
                report.not_adapted.push(component.clone());
            }
        }

        report
    }

    /// Check if LoRA weight shapes are compatible with base tensor
    fn check_shape_compatibility(base_tensor: &Tensor, lora_weight: &LoraWeight) -> Result<()> {
        let base_shape = base_tensor.dims();

        // Base tensor should be 2D (or can be flattened to 2D)
        if base_shape.len() > 2 {
            bail!(
                "Base tensor has unsupported shape: {:?} (expected 2D)",
                base_shape
            );
        }

        let (d_out, d_in) = if base_shape.len() == 2 {
            (base_shape[0], base_shape[1])
        } else {
            bail!("Base tensor must be 2D, got: {:?}", base_shape);
        };

        // Check A matrix: should be (rank × d_in)
        let a_shape = lora_weight.a.dims();
        if a_shape.len() != 2 || a_shape[1] != d_in {
            bail!(
                "LoRA A matrix shape mismatch: expected (rank, {}), got {:?}",
                d_in,
                a_shape
            );
        }

        // Check B matrix: should be (d_out × rank)
        let b_shape = lora_weight.b.dims();
        if b_shape.len() != 2 || b_shape[0] != d_out || b_shape[1] != a_shape[0] {
            bail!(
                "LoRA B matrix shape mismatch: expected ({}, {}), got {:?}",
                d_out,
                a_shape[0],
                b_shape
            );
        }

        Ok(())
    }

    /// Merge LoRA weights into base model tensors
    ///
    /// Formula: W' = W + (B @ A) * (alpha / rank) * scale
    ///
    /// # Arguments
    ///
    /// * `base_tensors` - Mutable reference to base model tensors
    /// * `scale` - Scaling factor (0.0 to 1.0, typically 0.8-1.0)
    pub fn merge_into(&self, base_tensors: &mut HashMap<String, Tensor>, scale: f32) -> Result<()> {
        if self.config.validate_shapes {
            let report = self.validate(base_tensors);
            if !report.is_valid() {
                bail!(
                    "LoRA validation failed: {} shape mismatches, {} missing components",
                    report.shape_mismatch.len(),
                    report.missing_in_base.len()
                );
            }
        }

        for (component, lora_weight) in &self.adapters {
            if let Some(base_tensor) = base_tensors.get_mut(component) {
                // Compute LoRA delta: (B @ A) * (alpha / rank) * scale
                let delta = lora_weight.b.matmul(&lora_weight.a)?;
                let scaling_factor = (lora_weight.alpha / lora_weight.rank as f32) * scale;
                let scaled_delta = (delta * scaling_factor as f64)?;

                // Add to base tensor: W' = W + scaled_delta
                *base_tensor = base_tensor.add(&scaled_delta)?;

                if self.config.verbose {
                    println!(
                        "  ✓ Merged LoRA into: {} (scale={:.3})",
                        component, scaling_factor
                    );
                }
            }
        }

        Ok(())
    }

    /// Get detected LoRA format
    pub fn format(&self) -> LoraFormat {
        self.format
    }

    /// Get base model architecture
    pub fn architecture(&self) -> &ModelArchitecture {
        &self.base_mapper.architecture
    }

    /// Get number of adapted components
    pub fn num_adapters(&self) -> usize {
        self.adapters.len()
    }

    /// Get list of adapted component names
    pub fn adapted_components(&self) -> Vec<&String> {
        self.adapters.keys().collect()
    }

    /// Get reference to base name mapper
    pub fn name_mapper(&self) -> &TensorNameMapper {
        &self.base_mapper
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candlelight::core::DType;

    #[test]
    fn test_detect_lora_format_huggingface() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "base_model.model.layers.0.self_attn.q_proj.lora_A".to_string(),
            Tensor::zeros((8, 4096), DType::F32, &Device::Cpu).unwrap(),
        );
        tensors.insert(
            "base_model.model.layers.0.self_attn.q_proj.lora_B".to_string(),
            Tensor::zeros((4096, 8), DType::F32, &Device::Cpu).unwrap(),
        );

        let format = LoraAdapter::detect_lora_format(&tensors).unwrap();
        assert_eq!(format, LoraFormat::HuggingFace);
    }

    #[test]
    fn test_detect_lora_format_custom() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "lora.blk.0.attn_q.A".to_string(),
            Tensor::zeros((8, 4096), DType::F32, &Device::Cpu).unwrap(),
        );
        tensors.insert(
            "lora.blk.0.attn_q.B".to_string(),
            Tensor::zeros((4096, 8), DType::F32, &Device::Cpu).unwrap(),
        );

        let format = LoraAdapter::detect_lora_format(&tensors).unwrap();
        assert_eq!(format, LoraFormat::Custom);
    }

    #[test]
    fn test_parse_huggingface_name() {
        let (base, is_a) = LoraAdapter::parse_lora_tensor_name(
            "base_model.model.layers.5.self_attn.q_proj.lora_A",
            LoraFormat::HuggingFace,
        )
        .unwrap();

        assert_eq!(base, "layers.5.self_attn.q_proj");
        assert!(is_a);
    }

    #[test]
    fn test_parse_custom_name() {
        let (base, is_a) =
            LoraAdapter::parse_lora_tensor_name("lora.blk.0.attn_q.A", LoraFormat::Custom).unwrap();

        assert_eq!(base, "blk.0.attn_q");
        assert!(is_a);
    }

    #[test]
    fn test_validation_report() {
        let report = ValidationReport {
            compatible: vec!["layer1".to_string()],
            shape_mismatch: vec![("layer2".to_string(), "wrong shape".to_string())],
            missing_in_base: vec!["layer3".to_string()],
            not_adapted: vec![],
        };

        assert!(!report.is_valid());
        assert_eq!(report.total_components(), 3);
    }

    /// `to_fuel_lora_arrays` bridges `LoraLinear::new`'s INPUT shape, not
    /// `LoraAdapter::load`'s storage format (board item 106, PR 17 — the
    /// narrow piece confirmed independent of PR 16's mlmf gating). Fuel's own
    /// doc comment on `lora_a`/`lora_b` says each is the TRANSPOSE of HF's
    /// native layout; this test pins that with distinct per-cell values so a
    /// missing transpose fails loudly, not coincidentally.
    #[test]
    fn to_fuel_lora_arrays_transposes_a_and_b_into_fuels_layout() {
        // rank=2, in_features=3: a is HF-native [rank, in_features].
        let a =
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &Device::Cpu).unwrap();
        // out_features=4, rank=2: b is HF-native [out_features, rank].
        let b = Tensor::from_vec(
            vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            (4, 2),
            &Device::Cpu,
        )
        .unwrap();
        let weight = LoraWeight {
            a,
            b,
            alpha: 16.0,
            rank: 2,
        };

        let arrays = weight.to_fuel_lora_arrays().unwrap();

        assert_eq!(
            (arrays.rank, arrays.in_features, arrays.out_features),
            (2, 3, 4)
        );
        // a transposed: [3,2] row-major = [[1,4],[2,5],[3,6]] flattened.
        assert_eq!(arrays.lora_a.as_ref(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        // b transposed: [2,4] row-major = [[10,30,50,70],[20,40,60,80]] flattened.
        assert_eq!(
            arrays.lora_b.as_ref(),
            &[10.0, 30.0, 50.0, 70.0, 20.0, 40.0, 60.0, 80.0]
        );
    }

    #[test]
    fn to_fuel_lora_arrays_rejects_a_b_rank_mismatch() {
        let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu).unwrap(); // rank=2
        let b = Tensor::zeros((4, 5), DType::F32, &Device::Cpu).unwrap(); // rank=5
        let weight = LoraWeight {
            a,
            b,
            alpha: 16.0,
            rank: 2,
        };
        assert!(weight.to_fuel_lora_arrays().is_err());
    }
}
