//! Dynamic tensor name mapping for GGUF models
//!
//! This module provides intelligent mapping between abstract layer identifiers
//! (e.g., "layer_0", "attention.query") and concrete GGUF tensor names
//! (e.g., "blk.0.attn_q.weight"). It automatically detects model architecture
//! patterns and supports multiple model families.

use anyhow::Result;
use regex::Regex;
use std::collections::HashMap;

/// Component type within a transformer layer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComponentType {
    /// Attention query projection
    AttentionQuery,
    /// Attention key projection
    AttentionKey,
    /// Attention value projection
    AttentionValue,
    /// Attention output projection
    AttentionOutput,
    /// FFN gate projection (for gated FFNs like SwiGLU)
    FfnGate,
    /// FFN up projection
    FfnUp,
    /// FFN down projection
    FfnDown,
    /// Attention normalization
    AttentionNorm,
    /// FFN normalization
    FfnNorm,
}

/// Detected model architecture family
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArchitecture {
    /// LLaMA family (uses "blk.{N}.attn_q" pattern)
    LLaMA,
    /// GPT family (uses "h.{N}.attn.c_attn" pattern)
    GPT,
    /// Mistral/Mixtral (similar to LLaMA but may have MoE)
    Mistral,
    /// Generic/Unknown architecture
    Unknown,
}

/// Dynamic tensor name mapper
#[derive(Debug, Clone)]
pub struct TensorNameMapper {
    /// Detected architecture
    pub architecture: ModelArchitecture,

    /// Mapping from (layer_index, component) to GGUF tensor name
    pub mappings: HashMap<(usize, ComponentType), String>,

    /// All detected layer indices
    pub layer_indices: Vec<usize>,

    /// Compiled regex patterns for this architecture
    patterns: ArchitecturePatterns,
}

/// Regex patterns for parsing tensor names
#[derive(Debug, Clone)]
struct ArchitecturePatterns {
    layer_pattern: Regex,
    component_patterns: HashMap<ComponentType, Regex>,
}

impl TensorNameMapper {
    /// Create a mapper by auto-detecting architecture from GGUF tensor names
    pub fn from_tensor_names(tensor_names: &[String]) -> Result<Self> {
        let architecture = Self::detect_architecture(tensor_names)?;
        let patterns = ArchitecturePatterns::for_architecture(architecture)?;

        let mut mappings = HashMap::new();
        let mut layer_indices = Vec::new();

        // Parse all tensor names and build mapping
        for tensor_name in tensor_names {
            if let Some((layer_idx, component)) = patterns.parse_tensor_name(tensor_name) {
                mappings.insert((layer_idx, component), tensor_name.clone());

                if !layer_indices.contains(&layer_idx) {
                    layer_indices.push(layer_idx);
                }
            }
        }

        layer_indices.sort();

        Ok(Self {
            architecture,
            mappings,
            layer_indices,
            patterns,
        })
    }

    /// Detect model architecture from tensor names
    fn detect_architecture(tensor_names: &[String]) -> Result<ModelArchitecture> {
        // Check for LLaMA pattern: "blk.N.attn_q.weight"
        if tensor_names
            .iter()
            .any(|name| name.contains("blk.") && name.contains(".attn_q"))
        {
            return Ok(ModelArchitecture::LLaMA);
        }

        // Check for GPT pattern: "h.N.attn.c_attn"
        if tensor_names
            .iter()
            .any(|name| name.contains("h.") && name.contains(".attn.c_attn"))
        {
            return Ok(ModelArchitecture::GPT);
        }

        // Check for Mistral (similar to LLaMA but with possible MoE)
        if tensor_names
            .iter()
            .any(|name| name.contains("layers.") && name.contains(".self_attn."))
        {
            return Ok(ModelArchitecture::Mistral);
        }

        // Default to unknown
        Ok(ModelArchitecture::Unknown)
    }

    /// Map abstract layer identifier to concrete GGUF tensor name
    ///
    /// # Arguments
    /// * `abstract_name` - Generic name like "layer_0", "layer_1.attention.query"
    ///
    /// # Returns
    /// Concrete GGUF tensor name, or None if no mapping exists
    pub fn map_name(&self, abstract_name: &str) -> Option<String> {
        // Parse abstract name
        let (layer_idx, component) = Self::parse_abstract_name(abstract_name)?;

        // Look up in mappings
        self.mappings.get(&(layer_idx, component)).cloned()
    }

    /// Map all tensors for a specific layer
    pub fn map_layer(&self, layer_idx: usize) -> Vec<(ComponentType, String)> {
        let mut results = Vec::new();

        for &component in &[
            ComponentType::AttentionQuery,
            ComponentType::AttentionKey,
            ComponentType::AttentionValue,
            ComponentType::AttentionOutput,
            ComponentType::FfnGate,
            ComponentType::FfnUp,
            ComponentType::FfnDown,
        ] {
            if let Some(name) = self.mappings.get(&(layer_idx, component)) {
                results.push((component, name.clone()));
            }
        }

        results
    }

    /// Parse abstract name like "layer_0" or "layer_5.attention.query"
    fn parse_abstract_name(name: &str) -> Option<(usize, ComponentType)> {
        // Simple pattern: "layer_{N}" maps to all components of layer N
        if let Some(stripped) = name.strip_prefix("layer_") {
            if let Ok(idx) = stripped.parse::<usize>() {
                // Default to attention query if no component specified
                return Some((idx, ComponentType::AttentionQuery));
            }
        }

        // Extended pattern: "layer_{N}.{component}"
        let parts: Vec<&str> = name.split('.').collect();
        if parts.len() >= 2 {
            if let Some(layer_part) = parts[0].strip_prefix("layer_") {
                if let Ok(idx) = layer_part.parse::<usize>() {
                    let component = Self::parse_component(&parts[1..])?;
                    return Some((idx, component));
                }
            }
        }

        None
    }

    /// Parse component type from string
    fn parse_component(parts: &[&str]) -> Option<ComponentType> {
        match parts {
            ["attention", "query"] | ["attn", "q"] => Some(ComponentType::AttentionQuery),
            ["attention", "key"] | ["attn", "k"] => Some(ComponentType::AttentionKey),
            ["attention", "value"] | ["attn", "v"] => Some(ComponentType::AttentionValue),
            ["attention", "output"] | ["attn", "output"] => Some(ComponentType::AttentionOutput),
            ["ffn", "gate"] => Some(ComponentType::FfnGate),
            ["ffn", "up"] => Some(ComponentType::FfnUp),
            ["ffn", "down"] => Some(ComponentType::FfnDown),
            _ => None,
        }
    }

    /// Get all tensor names for a layer (for batch processing)
    pub fn get_layer_tensors(&self, layer_idx: usize) -> Vec<String> {
        self.mappings
            .iter()
            .filter_map(|((idx, _component), name)| {
                if *idx == layer_idx {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect()
    }
}

impl ArchitecturePatterns {
    /// Create patterns for a specific architecture
    fn for_architecture(arch: ModelArchitecture) -> Result<Self> {
        match arch {
            ModelArchitecture::LLaMA => Self::llama_patterns(),
            ModelArchitecture::GPT => Self::gpt_patterns(),
            ModelArchitecture::Mistral => Self::mistral_patterns(),
            ModelArchitecture::Unknown => Self::generic_patterns(),
        }
    }

    /// LLaMA-style patterns (TinyLlama, LLaMA 2/3, Qwen, etc.)
    fn llama_patterns() -> Result<Self> {
        let layer_pattern = Regex::new(r"blk\.(\d+)\.")?;

        let mut component_patterns = HashMap::new();
        component_patterns.insert(
            ComponentType::AttentionQuery,
            Regex::new(r"blk\.\d+\.attn_q\.weight")?,
        );
        component_patterns.insert(
            ComponentType::AttentionKey,
            Regex::new(r"blk\.\d+\.attn_k\.weight")?,
        );
        component_patterns.insert(
            ComponentType::AttentionValue,
            Regex::new(r"blk\.\d+\.attn_v\.weight")?,
        );
        component_patterns.insert(
            ComponentType::AttentionOutput,
            Regex::new(r"blk\.\d+\.attn_output\.weight")?,
        );
        component_patterns.insert(
            ComponentType::FfnGate,
            Regex::new(r"blk\.\d+\.ffn_gate\.weight")?,
        );
        component_patterns.insert(
            ComponentType::FfnUp,
            Regex::new(r"blk\.\d+\.ffn_up\.weight")?,
        );
        component_patterns.insert(
            ComponentType::FfnDown,
            Regex::new(r"blk\.\d+\.ffn_down\.weight")?,
        );

        Ok(Self {
            layer_pattern,
            component_patterns,
        })
    }

    /// GPT-style patterns
    fn gpt_patterns() -> Result<Self> {
        let layer_pattern = Regex::new(r"h\.(\d+)\.")?;

        let mut component_patterns = HashMap::new();
        component_patterns.insert(
            ComponentType::AttentionQuery,
            Regex::new(r"h\.\d+\.attn\.c_attn\.weight")?,
        );
        // GPT often fuses Q/K/V into c_attn, so we'll map all three to it
        component_patterns.insert(
            ComponentType::AttentionKey,
            Regex::new(r"h\.\d+\.attn\.c_attn\.weight")?,
        );
        component_patterns.insert(
            ComponentType::AttentionValue,
            Regex::new(r"h\.\d+\.attn\.c_attn\.weight")?,
        );
        component_patterns.insert(
            ComponentType::AttentionOutput,
            Regex::new(r"h\.\d+\.attn\.c_proj\.weight")?,
        );
        component_patterns.insert(
            ComponentType::FfnUp,
            Regex::new(r"h\.\d+\.mlp\.c_fc\.weight")?,
        );
        component_patterns.insert(
            ComponentType::FfnDown,
            Regex::new(r"h\.\d+\.mlp\.c_proj\.weight")?,
        );

        Ok(Self {
            layer_pattern,
            component_patterns,
        })
    }

    /// Mistral-style patterns
    fn mistral_patterns() -> Result<Self> {
        let layer_pattern = Regex::new(r"layers\.(\d+)\.")?;

        let mut component_patterns = HashMap::new();
        component_patterns.insert(
            ComponentType::AttentionQuery,
            Regex::new(r"layers\.\d+\.self_attn\.q_proj\.weight")?,
        );
        component_patterns.insert(
            ComponentType::AttentionKey,
            Regex::new(r"layers\.\d+\.self_attn\.k_proj\.weight")?,
        );
        component_patterns.insert(
            ComponentType::AttentionValue,
            Regex::new(r"layers\.\d+\.self_attn\.v_proj\.weight")?,
        );
        component_patterns.insert(
            ComponentType::AttentionOutput,
            Regex::new(r"layers\.\d+\.self_attn\.o_proj\.weight")?,
        );
        component_patterns.insert(
            ComponentType::FfnGate,
            Regex::new(r"layers\.\d+\.mlp\.gate_proj\.weight")?,
        );
        component_patterns.insert(
            ComponentType::FfnUp,
            Regex::new(r"layers\.\d+\.mlp\.up_proj\.weight")?,
        );
        component_patterns.insert(
            ComponentType::FfnDown,
            Regex::new(r"layers\.\d+\.mlp\.down_proj\.weight")?,
        );

        Ok(Self {
            layer_pattern,
            component_patterns,
        })
    }

    /// Generic fallback patterns
    fn generic_patterns() -> Result<Self> {
        // Try to match common patterns
        Self::llama_patterns()
    }

    /// Parse a tensor name into (layer_index, component)
    fn parse_tensor_name(&self, name: &str) -> Option<(usize, ComponentType)> {
        // Extract layer index
        let layer_idx = self
            .layer_pattern
            .captures(name)?
            .get(1)?
            .as_str()
            .parse::<usize>()
            .ok()?;

        // Find matching component
        for (&component, pattern) in &self.component_patterns {
            if pattern.is_match(name) {
                return Some((layer_idx, component));
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama_architecture_detection() {
        let tensor_names = vec![
            "blk.0.attn_q.weight".to_string(),
            "blk.0.attn_k.weight".to_string(),
            "blk.1.ffn_gate.weight".to_string(),
        ];

        let mapper = TensorNameMapper::from_tensor_names(&tensor_names).unwrap();
        assert_eq!(mapper.architecture, ModelArchitecture::LLaMA);
        assert_eq!(mapper.layer_indices, vec![0, 1]);
    }

    #[test]
    fn test_name_mapping() {
        let tensor_names = vec![
            "blk.0.attn_q.weight".to_string(),
            "blk.0.attn_k.weight".to_string(),
            "blk.1.attn_q.weight".to_string(),
        ];

        let mapper = TensorNameMapper::from_tensor_names(&tensor_names).unwrap();

        // Test simple mapping
        let mapped = mapper.map_name("layer_0").unwrap();
        assert_eq!(mapped, "blk.0.attn_q.weight");

        // Test component-specific mapping
        assert_eq!(
            mapper.mappings.get(&(0, ComponentType::AttentionKey)),
            Some(&"blk.0.attn_k.weight".to_string())
        );
    }

    #[test]
    fn test_layer_tensors() {
        let tensor_names = vec![
            "blk.0.attn_q.weight".to_string(),
            "blk.0.attn_k.weight".to_string(),
            "blk.0.ffn_gate.weight".to_string(),
            "blk.1.attn_q.weight".to_string(),
        ];

        let mapper = TensorNameMapper::from_tensor_names(&tensor_names).unwrap();
        let layer_0_tensors = mapper.get_layer_tensors(0);

        assert_eq!(layer_0_tensors.len(), 3);
        assert!(layer_0_tensors.contains(&"blk.0.attn_q.weight".to_string()));
    }
}
