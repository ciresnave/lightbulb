//! Tool Registry with Architecture-Aware Capability Detection
//!
//! This module provides automatic detection of model capabilities (text, vision,
//! function calling) based on tensor names and architecture, enabling dynamic
//! tool registration for multi-modal and function-calling models.
//!
//! # Features
//!
//! - **Capability Auto-Detection**: Detects vision, function calling, context length
//! - **Dynamic Tool Registration**: Auto-registers tools based on detected capabilities
//! - **Query-Based Tool Selection**: Recommends tools for user queries
//! - **Extensible**: Support for custom tool registration
//!
//! # Example
//!
//! ```rust,ignore
//! use lightbulb::tools::ToolRegistry;
//!
//! // Create registry with auto-detected capabilities
//! let registry = ToolRegistry::from_tensor_names(&model.tensor_names())?;
//!
//! // Check capabilities
//! if registry.capabilities().supports_vision {
//!     println!("Model supports vision!");
//! }
//!
//! // Get recommended tools for query
//! let tools = registry.get_tools_for_query("analyze this image");
//! ```

use anyhow::{bail, Result};
use std::collections::HashMap;

use crate::pruning::name_mapping::{ModelArchitecture, TensorNameMapper};

/// Tool registry with architecture-aware capability detection
#[derive(Debug, Clone)]
pub struct ToolRegistry {
    /// Registered tools (name → tool)
    tools: HashMap<String, Tool>,

    /// Detected model capabilities
    capabilities: ModelCapabilities,

    /// Architecture from name mapper
    architecture: ModelArchitecture,
}

/// Detected model capabilities
#[derive(Debug, Clone, Default)]
pub struct ModelCapabilities {
    /// Supports text generation (all transformer models)
    pub supports_text: bool,

    /// Supports vision/image understanding
    pub supports_vision: bool,

    /// Supports function calling / tool use
    pub supports_function_calling: bool,

    /// Maximum context length (from positional embeddings)
    pub max_context: usize,

    /// Number of layers detected
    pub num_layers: usize,
}

/// Tool definition
#[derive(Debug, Clone)]
pub struct Tool {
    /// Tool name (e.g., "text_generation", "image_understanding")
    pub name: String,

    /// Tool description
    pub description: String,

    /// Keywords for query matching
    pub keywords: Vec<String>,

    /// Whether tool requires vision capability
    pub requires_vision: bool,

    /// Whether tool requires function calling capability
    pub requires_function_calling: bool,
}

impl Tool {
    /// Create text generation tool
    pub fn text_generation() -> Self {
        Self {
            name: "text_generation".to_string(),
            description: "Generate text based on a prompt".to_string(),
            keywords: vec![
                "generate".to_string(),
                "write".to_string(),
                "create".to_string(),
                "text".to_string(),
            ],
            requires_vision: false,
            requires_function_calling: false,
        }
    }

    /// Create image understanding tool
    pub fn image_understanding() -> Self {
        Self {
            name: "image_understanding".to_string(),
            description: "Analyze and understand images".to_string(),
            keywords: vec![
                "image".to_string(),
                "picture".to_string(),
                "photo".to_string(),
                "analyze".to_string(),
                "see".to_string(),
                "look".to_string(),
            ],
            requires_vision: true,
            requires_function_calling: false,
        }
    }

    /// Create function calling tool
    pub fn function_call() -> Self {
        Self {
            name: "function_call".to_string(),
            description: "Call external functions or APIs".to_string(),
            keywords: vec![
                "function".to_string(),
                "call".to_string(),
                "api".to_string(),
                "tool".to_string(),
                "execute".to_string(),
            ],
            requires_vision: false,
            requires_function_calling: true,
        }
    }
}

impl ToolRegistry {
    /// Create tool registry from tensor names with auto-detection
    ///
    /// Automatically detects model capabilities and registers appropriate tools.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let registry = ToolRegistry::from_tensor_names(&model.tensor_names())?;
    /// println!("Detected {} tools", registry.num_tools());
    /// ```
    pub fn from_tensor_names(tensor_names: &[String]) -> Result<Self> {
        // Create name mapper for architecture detection
        let name_mapper = TensorNameMapper::from_tensor_names(tensor_names)?;

        // Detect capabilities
        let capabilities = Self::detect_capabilities(&name_mapper, tensor_names)?;

        // Auto-register tools based on capabilities
        let mut tools = HashMap::new();

        if capabilities.supports_text {
            let tool = Tool::text_generation();
            tools.insert(tool.name.clone(), tool);
        }

        if capabilities.supports_vision {
            let tool = Tool::image_understanding();
            tools.insert(tool.name.clone(), tool);
        }

        if capabilities.supports_function_calling {
            let tool = Tool::function_call();
            tools.insert(tool.name.clone(), tool);
        }

        Ok(Self {
            tools,
            capabilities,
            architecture: name_mapper.architecture,
        })
    }

    /// Detect model capabilities from architecture and tensor names
    fn detect_capabilities(
        mapper: &TensorNameMapper,
        tensor_names: &[String],
    ) -> Result<ModelCapabilities> {
        let mut capabilities = ModelCapabilities::default();

        // All transformer models support text
        capabilities.supports_text = true;

        // Check for vision encoder (vision transformers, CLIP, multimodal models)
        let has_vision = tensor_names.iter().any(|name| {
            name.contains("vision")
                || name.contains("image")
                || name.contains("visual")
                || name.contains("patch_embed")
                || name.contains("clip")
        });
        capabilities.supports_vision = has_vision;

        // Check for function calling head (tool-use models)
        let has_function_head = tensor_names.iter().any(|name| {
            name.contains("function")
                || name.contains("tool")
                || name.contains("tool_call")
                || name.contains("function_call")
        });
        capabilities.supports_function_calling = has_function_head;

        // Detect max context length from positional embeddings
        capabilities.max_context = Self::detect_max_context(tensor_names);

        // Get layer count from name mapper
        capabilities.num_layers = mapper.layer_indices.len();

        Ok(capabilities)
    }

    /// Detect maximum context length from positional embedding tensors
    fn detect_max_context(tensor_names: &[String]) -> usize {
        // Look for positional embedding tensors like "pos_embd" or "position_embedding"
        for name in tensor_names {
            if name.contains("pos") && name.contains("emb") {
                // Common context lengths: 2048, 4096, 8192, 16384, 32768
                // Default to 2048 if we detect positional embeddings
                return 2048;
            }
        }

        // Default fallback
        2048
    }

    /// Register a custom tool
    ///
    /// Validates that the tool is compatible with model capabilities
    /// before registering.
    pub fn register_tool(&mut self, tool: Tool) -> Result<()> {
        // Validate tool requirements against capabilities
        if tool.requires_vision && !self.capabilities.supports_vision {
            bail!(
                "Tool '{}' requires vision but model doesn't support it",
                tool.name
            );
        }

        if tool.requires_function_calling && !self.capabilities.supports_function_calling {
            bail!(
                "Tool '{}' requires function calling but model doesn't support it",
                tool.name
            );
        }

        self.tools.insert(tool.name.clone(), tool);
        Ok(())
    }

    /// Get recommended tools for a user query
    ///
    /// Uses simple keyword matching to find relevant tools.
    /// Returns tools sorted by relevance (number of keyword matches).
    pub fn get_tools_for_query(&self, query: &str) -> Vec<&Tool> {
        let query_lower = query.to_lowercase();

        let mut matches: Vec<(&Tool, usize)> = self
            .tools
            .values()
            .map(|tool| {
                let match_count = tool
                    .keywords
                    .iter()
                    .filter(|kw| query_lower.contains(kw.as_str()))
                    .count();
                (tool, match_count)
            })
            .filter(|(_, count)| *count > 0)
            .collect();

        // Sort by match count (descending)
        matches.sort_by_key(|(_, count)| std::cmp::Reverse(*count));

        matches.into_iter().map(|(tool, _)| tool).collect()
    }

    /// Get all registered tools
    pub fn tools(&self) -> &HashMap<String, Tool> {
        &self.tools
    }

    /// Get number of registered tools
    pub fn num_tools(&self) -> usize {
        self.tools.len()
    }

    /// Get tool by name
    pub fn get_tool(&self, name: &str) -> Option<&Tool> {
        self.tools.get(name)
    }

    /// Get model capabilities
    pub fn capabilities(&self) -> &ModelCapabilities {
        &self.capabilities
    }

    /// Get detected architecture
    pub fn architecture(&self) -> &ModelArchitecture {
        &self.architecture
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candlelight::core::{DType, Device, Tensor};

    #[test]
    fn test_text_only_model() -> Result<()> {
        // LLaMA-like model (text only)
        let mut tensor_names = Vec::new();
        for layer_idx in 0..32 {
            tensor_names.push(format!("blk.{}.attn_q.weight", layer_idx));
            tensor_names.push(format!("blk.{}.ffn_gate.weight", layer_idx));
        }
        tensor_names.push("output.weight".to_string());

        let registry = ToolRegistry::from_tensor_names(&tensor_names)?;

        assert!(registry.capabilities().supports_text);
        assert!(!registry.capabilities().supports_vision);
        assert!(!registry.capabilities().supports_function_calling);
        assert_eq!(registry.capabilities().num_layers, 32);
        assert_eq!(registry.num_tools(), 1); // Only text_generation

        Ok(())
    }

    #[test]
    fn test_vision_model() -> Result<()> {
        // Vision model (e.g., CLIP, vision transformer)
        let mut tensor_names = Vec::new();
        tensor_names.push("vision.patch_embed.weight".to_string());
        tensor_names.push("vision.encoder.0.attn_q.weight".to_string());
        for layer_idx in 0..24 {
            tensor_names.push(format!("blk.{}.attn_q.weight", layer_idx));
        }

        let registry = ToolRegistry::from_tensor_names(&tensor_names)?;

        assert!(registry.capabilities().supports_text);
        assert!(registry.capabilities().supports_vision);
        assert!(!registry.capabilities().supports_function_calling);
        assert_eq!(registry.num_tools(), 2); // text_generation + image_understanding

        Ok(())
    }

    #[test]
    fn test_function_calling_model() -> Result<()> {
        // Model with function calling (e.g., GPT-4 with tools)
        let mut tensor_names = Vec::new();
        for layer_idx in 0..40 {
            tensor_names.push(format!("blk.{}.attn_q.weight", layer_idx));
        }
        tensor_names.push("function_call.head.weight".to_string());
        tensor_names.push("tool_use.classifier.weight".to_string());

        let registry = ToolRegistry::from_tensor_names(&tensor_names)?;

        assert!(registry.capabilities().supports_text);
        assert!(!registry.capabilities().supports_vision);
        assert!(registry.capabilities().supports_function_calling);
        assert_eq!(registry.num_tools(), 2); // text_generation + function_call

        Ok(())
    }

    #[test]
    fn test_multimodal_model() -> Result<()> {
        // Multimodal model with all capabilities
        let mut tensor_names = Vec::new();
        tensor_names.push("vision.encoder.0.attn_q.weight".to_string());
        for layer_idx in 0..32 {
            tensor_names.push(format!("blk.{}.attn_q.weight", layer_idx));
        }
        tensor_names.push("function_head.weight".to_string());

        let registry = ToolRegistry::from_tensor_names(&tensor_names)?;

        assert!(registry.capabilities().supports_text);
        assert!(registry.capabilities().supports_vision);
        assert!(registry.capabilities().supports_function_calling);
        assert_eq!(registry.num_tools(), 3); // All 3 tools

        Ok(())
    }

    #[test]
    fn test_query_tool_matching() -> Result<()> {
        let mut tensor_names = Vec::new();
        tensor_names.push("vision.patch_embed.weight".to_string());
        for layer_idx in 0..16 {
            tensor_names.push(format!("blk.{}.attn_q.weight", layer_idx));
        }

        let registry = ToolRegistry::from_tensor_names(&tensor_names)?;

        // Test image-related query
        let tools = registry.get_tools_for_query("analyze this image");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "image_understanding");

        // Test text-related query
        let tools = registry.get_tools_for_query("write a story");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "text_generation");

        // Test query matching multiple tools
        let tools = registry.get_tools_for_query("generate text and analyze picture");
        assert_eq!(tools.len(), 2);

        Ok(())
    }

    #[test]
    fn test_custom_tool_registration() -> Result<()> {
        let tensor_names = vec!["blk.0.attn_q.weight".to_string()];
        let mut registry = ToolRegistry::from_tensor_names(&tensor_names)?;

        // Register custom tool
        let custom_tool = Tool {
            name: "code_generation".to_string(),
            description: "Generate code".to_string(),
            keywords: vec!["code".to_string(), "program".to_string()],
            requires_vision: false,
            requires_function_calling: false,
        };

        registry.register_tool(custom_tool)?;
        assert_eq!(registry.num_tools(), 2); // text_generation + code_generation

        Ok(())
    }

    #[test]
    fn test_incompatible_tool_registration() {
        let tensor_names = vec!["blk.0.attn_q.weight".to_string()];
        let mut registry = ToolRegistry::from_tensor_names(&tensor_names).unwrap();

        // Try to register vision tool on text-only model
        let vision_tool = Tool::image_understanding();
        let result = registry.register_tool(vision_tool);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("requires vision"));
    }

    #[test]
    fn test_architecture_detection() -> Result<()> {
        let mut tensor_names = Vec::new();
        for layer_idx in 0..24 {
            tensor_names.push(format!("blk.{}.attn_q.weight", layer_idx));
        }

        let registry = ToolRegistry::from_tensor_names(&tensor_names)?;

        assert_eq!(
            registry.architecture(),
            &crate::pruning::name_mapping::ModelArchitecture::LLaMA
        );
        assert_eq!(registry.capabilities().num_layers, 24);

        Ok(())
    }
}
