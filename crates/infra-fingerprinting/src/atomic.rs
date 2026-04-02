//! Atomic-level fingerprinting for individual tokens/concepts within dimensional context
//!
//! This module provides fingerprinting capabilities for the smallest units of meaning
//! in the DynAniML system - individual tokens and concepts that exist within
//! specific dimensional contexts.

use crate::{Fingerprint, FingerprintLevel, FingerprintMetadata, FingerprintResult};
use blake3::Hasher;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Trait for types that can be fingerprinted at the atomic level
pub trait AtomicFingerprintable {
    /// Extract atomic components for fingerprinting
    fn get_atomic_components(&self) -> AtomicComponents;
}

/// Components needed for atomic-level fingerprinting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomicComponents {
    /// The core token/concept value
    pub token: String,
    /// Dimensional context in which this token exists
    pub dimensional_context: Vec<DimensionalContext>,
    /// Position information within the knowledge structure
    pub position: Option<PositionInfo>,
    /// Type information about the token
    pub token_type: TokenType,
}

/// Context within a specific dimension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionalContext {
    /// Name of the dimension (e.g., "entity", "temporal", "semantic")
    pub dimension: String,
    /// Value or position within this dimension
    pub value: String,
    /// Weight or importance of this dimensional context
    pub weight: f32,
}

/// Position information for tokens
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionInfo {
    /// Absolute position in the source
    pub absolute_position: usize,
    /// Relative position within parent structure
    pub relative_position: Option<usize>,
    /// Depth in hierarchical structure
    pub depth: Option<usize>,
}

/// Types of tokens for fingerprinting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TokenType {
    /// Plain text token
    Text,
    /// Numeric value
    Numeric,
    /// Identifier or symbol
    Identifier,
    /// Operator or function
    Operator,
    /// Structural marker
    Structural,
    /// Custom type with metadata
    Custom(HashMap<String, String>),
}

/// Engine for computing atomic-level fingerprints
pub struct AtomicFingerprintEngine {
    /// Whether to include position information in fingerprints
    include_position: bool,
    /// Whether to normalize tokens before fingerprinting
    normalize: bool,
    /// Weight given to dimensional context vs token content
    context_weight: f32,
}

impl Default for AtomicFingerprintEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl AtomicFingerprintEngine {
    /// Create a new atomic fingerprint engine with default settings
    pub fn new() -> Self {
        Self {
            include_position: true,
            normalize: true,
            context_weight: 0.3, // 30% context, 70% content
        }
    }

    /// Create an engine with custom settings
    pub fn with_config(include_position: bool, normalize: bool, context_weight: f32) -> Self {
        Self {
            include_position,
            normalize,
            context_weight: context_weight.clamp(0.0, 1.0),
        }
    }

    /// Compute an atomic-level fingerprint for the given input
    pub fn compute_fingerprint<T>(&self, input: &T) -> FingerprintResult<Fingerprint>
    where
        T: AtomicFingerprintable,
    {
        let components = input.get_atomic_components();
        let hash = self.hash_components(&components)?;

        let metadata = FingerprintMetadata {
            algorithm: "atomic-blake3".to_string(),
            parameters: self.get_parameters(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            confidence: self.calculate_confidence(&components),
        };

        Ok(Fingerprint::with_metadata(
            hash,
            FingerprintLevel::Atomic,
            metadata,
        ))
    }

    /// Hash the atomic components into a fingerprint
    fn hash_components(&self, components: &AtomicComponents) -> FingerprintResult<[u8; 32]> {
        let mut hasher = Hasher::new();

        // Hash the core token
        let token = if self.normalize {
            self.normalize_token(&components.token)
        } else {
            components.token.clone()
        };
        hasher.update(token.as_bytes());

        // Hash dimensional context
        let mut context_parts = Vec::new();
        for context in &components.dimensional_context {
            context_parts.push(format!(
                "{}:{}:{}",
                context.dimension, context.value, context.weight
            ));
        }
        context_parts.sort(); // Ensure order independence

        for part in context_parts {
            hasher.update(part.as_bytes());
        }

        // Optionally include position information
        if self.include_position {
            if let Some(pos) = &components.position {
                hasher.update(&pos.absolute_position.to_le_bytes());
                if let Some(rel_pos) = pos.relative_position {
                    hasher.update(&rel_pos.to_le_bytes());
                }
                if let Some(depth) = pos.depth {
                    hasher.update(&depth.to_le_bytes());
                }
            }
        }

        // Hash token type
        let type_str = match &components.token_type {
            TokenType::Text => "text".to_string(),
            TokenType::Numeric => "numeric".to_string(),
            TokenType::Identifier => "identifier".to_string(),
            TokenType::Operator => "operator".to_string(),
            TokenType::Structural => "structural".to_string(),
            TokenType::Custom(metadata) => {
                let mut meta_parts: Vec<_> = metadata.iter().collect();
                meta_parts.sort_by_key(|(k, _)| *k);
                format!("custom:{:?}", meta_parts)
            }
        };
        hasher.update(type_str.as_bytes());

        Ok(hasher.finalize().into())
    }

    /// Normalize a token for consistent fingerprinting
    fn normalize_token(&self, token: &str) -> String {
        token.trim().to_lowercase()
    }

    /// Calculate confidence score for this fingerprint
    fn calculate_confidence(&self, components: &AtomicComponents) -> f32 {
        let mut confidence = 1.0;

        // Reduce confidence for very short tokens
        if components.token.len() < 2 {
            confidence *= 0.7;
        }

        // Increase confidence with more dimensional context
        if !components.dimensional_context.is_empty() {
            confidence *= 1.0 + (components.dimensional_context.len() as f32 * 0.1);
        }

        // Reduce confidence if position is missing when expected
        if self.include_position && components.position.is_none() {
            confidence *= 0.9;
        }

        confidence.min(1.0)
    }

    /// Get current engine parameters
    fn get_parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert(
            "include_position".to_string(),
            self.include_position.to_string(),
        );
        params.insert("normalize".to_string(), self.normalize.to_string());
        params.insert(
            "context_weight".to_string(),
            self.context_weight.to_string(),
        );
        params
    }
}

/// Helper function to create atomic components for simple text tokens
pub fn create_simple_atomic_components(token: &str) -> AtomicComponents {
    AtomicComponents {
        token: token.to_string(),
        dimensional_context: Vec::new(),
        position: None,
        token_type: TokenType::Text,
    }
}

/// Helper function to create atomic components with dimensional context
pub fn create_contextual_atomic_components(
    token: &str,
    contexts: Vec<(String, String, f32)>,
) -> AtomicComponents {
    let dimensional_context = contexts
        .into_iter()
        .map(|(dimension, value, weight)| DimensionalContext {
            dimension,
            value,
            weight,
        })
        .collect();

    AtomicComponents {
        token: token.to_string(),
        dimensional_context,
        position: None,
        token_type: TokenType::Text,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestToken {
        token: String,
        contexts: Vec<DimensionalContext>,
    }

    impl AtomicFingerprintable for TestToken {
        fn get_atomic_components(&self) -> AtomicComponents {
            AtomicComponents {
                token: self.token.clone(),
                dimensional_context: self.contexts.clone(),
                position: Some(PositionInfo {
                    absolute_position: 42,
                    relative_position: Some(7),
                    depth: Some(2),
                }),
                token_type: TokenType::Text,
            }
        }
    }

    #[test]
    fn test_simple_fingerprinting() {
        let engine = AtomicFingerprintEngine::new();
        let token = TestToken {
            token: "hello".to_string(),
            contexts: vec![],
        };

        let fingerprint = engine.compute_fingerprint(&token).unwrap();
        assert_eq!(fingerprint.level, FingerprintLevel::Atomic);
        assert!(fingerprint.metadata.is_some());
    }

    #[test]
    fn test_contextual_fingerprinting() {
        let engine = AtomicFingerprintEngine::new();
        let token = TestToken {
            token: "variable".to_string(),
            contexts: vec![
                DimensionalContext {
                    dimension: "entity".to_string(),
                    value: "identifier".to_string(),
                    weight: 0.8,
                },
                DimensionalContext {
                    dimension: "scope".to_string(),
                    value: "local".to_string(),
                    weight: 0.6,
                },
            ],
        };

        let fingerprint = engine.compute_fingerprint(&token).unwrap();
        assert_eq!(fingerprint.level, FingerprintLevel::Atomic);

        // Should be different from a token without context
        let simple_token = TestToken {
            token: "variable".to_string(),
            contexts: vec![],
        };
        let simple_fingerprint = engine.compute_fingerprint(&simple_token).unwrap();
        assert_ne!(fingerprint.hash, simple_fingerprint.hash);
    }

    #[test]
    fn test_normalization() {
        let engine = AtomicFingerprintEngine::with_config(false, true, 0.5);

        let token1 = TestToken {
            token: "Hello".to_string(),
            contexts: vec![],
        };
        let token2 = TestToken {
            token: "hello".to_string(),
            contexts: vec![],
        };

        let fp1 = engine.compute_fingerprint(&token1).unwrap();
        let fp2 = engine.compute_fingerprint(&token2).unwrap();

        // Should be the same due to normalization
        assert_eq!(fp1.hash, fp2.hash);
    }
}
