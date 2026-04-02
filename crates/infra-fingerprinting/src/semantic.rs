//! Semantic fingerprinting for functional capabilities and meaning
//!
//! This module provides fingerprinting capabilities for the semantic content
//! and functional capabilities of knowledge modules, focusing on what they
//! enable rather than how they're structured.

use crate::{Fingerprint, FingerprintLevel, FingerprintMetadata, FingerprintResult};
use std::collections::HashMap;

/// Trait for types that can be fingerprinted at the semantic level
pub trait SemanticFingerprintable {
    /// Extract semantic components for fingerprinting
    fn get_semantic_components(&self) -> SemanticComponents;
}

/// Components needed for semantic-level fingerprinting
#[derive(Debug, Clone)]
pub struct SemanticComponents {
    /// Functional capabilities provided by this component
    pub capabilities: Vec<Capability>,
    /// Semantic relationships with other components
    pub relationships: Vec<SemanticRelationship>,
    /// Intent or purpose of this component
    pub intent: Option<String>,
    /// Domain or context information
    pub domain: Option<String>,
}

/// A functional capability
#[derive(Debug, Clone)]
pub struct Capability {
    /// Name of the capability
    pub name: String,
    /// Description of what this capability does
    pub description: String,
    /// Input types this capability accepts
    pub inputs: Vec<String>,
    /// Output types this capability produces
    pub outputs: Vec<String>,
    /// Confidence in this capability definition
    pub confidence: f32,
}

/// Semantic relationship between components
#[derive(Debug, Clone)]
pub struct SemanticRelationship {
    /// Type of relationship (e.g., "uses", "extends", "implements")
    pub relationship_type: String,
    /// Target of the relationship
    pub target: String,
    /// Strength of the relationship (0.0 to 1.0)
    pub strength: f32,
}

/// Engine for computing semantic-level fingerprints
pub struct SemanticFingerprintEngine {
    /// Whether to include capabilities in fingerprints
    _include_capabilities: bool,
    /// Whether to include relationships in fingerprints
    include_relationships: bool,
    /// Weight given to different semantic aspects
    aspect_weights: HashMap<String, f32>,
}

impl Default for SemanticFingerprintEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl SemanticFingerprintEngine {
    /// Create a new semantic fingerprint engine with default settings
    pub fn new() -> Self {
        let mut weights = HashMap::new();
        weights.insert("capabilities".to_string(), 0.4);
        weights.insert("relationships".to_string(), 0.3);
        weights.insert("intent".to_string(), 0.2);
        weights.insert("domain".to_string(), 0.1);
        Self {
            _include_capabilities: true,
            include_relationships: true,
            aspect_weights: weights,
        }
    }

    /// Compute a semantic-level fingerprint for the given input
    pub fn compute_fingerprint<T>(&self, input: &T) -> FingerprintResult<Fingerprint>
    where
        T: SemanticFingerprintable,
    {
        let components = input.get_semantic_components();

        // Advanced semantic fingerprinting algorithm with capability clustering
        let mut hash_input = Vec::new();

        // 1. Semantic capability clustering and vectorization
        let capability_vector = self.vectorize_capabilities(&components.capabilities);
        hash_input.extend_from_slice(&capability_vector);

        // 2. Relationship strength matrix computation
        if self.include_relationships {
            let relationship_matrix = self.compute_relationship_matrix(&components.relationships);
            hash_input.extend_from_slice(&relationship_matrix);
        }

        // 3. Domain and intent semantic encoding
        if let Some(domain_weight) = self.aspect_weights.get("domain") {
            if let Some(ref domain) = components.domain {
                hash_input.extend_from_slice(domain.as_bytes());
                hash_input.extend_from_slice(&domain_weight.to_le_bytes());
            }
        }
        if let Some(intent_weight) = self.aspect_weights.get("intent") {
            if let Some(ref intent) = components.intent {
                hash_input.extend_from_slice(intent.as_bytes());
                hash_input.extend_from_slice(&intent_weight.to_le_bytes());
            }
        }

        // 4. Semantic coherence factor
        let coherence = self.calculate_semantic_coherence(&components);
        hash_input.extend_from_slice(&coherence.to_le_bytes());

        // 5. Advanced hash computation with semantic layering
        let hash = blake3::hash(&hash_input);
        let hash_bytes: [u8; 32] = *hash.as_bytes();

        let metadata = FingerprintMetadata {
            algorithm: "semantic-advanced-v1".to_string(),
            parameters: self.get_parameters(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            confidence: 0.5,
        };

        Ok(Fingerprint::with_metadata(
            hash_bytes,
            FingerprintLevel::Semantic,
            metadata,
        ))
    }

    /// Vectorize capabilities into a normalized vector representation
    fn vectorize_capabilities(&self, capabilities: &[Capability]) -> Vec<u8> {
        let mut vector = Vec::new();

        // Create a capability type frequency map
        let mut type_counts = std::collections::HashMap::new();
        for capability in capabilities {
            *type_counts.entry(&capability.name).or_insert(0u32) += 1;
        }

        // Sort capabilities by type for consistent ordering
        let mut sorted_types: Vec<_> = type_counts.keys().collect();
        sorted_types.sort();

        // Encode capability frequencies with weights
        if let Some(cap_weight) = self.aspect_weights.get("capabilities") {
            for &cap_type in &sorted_types {
                let count = type_counts[cap_type];
                let weighted_count = (count as f32 * cap_weight) as u32;
                vector.extend_from_slice(&weighted_count.to_le_bytes());

                // Add semantic hash of capability type
                let type_hash = blake3::hash(cap_type.as_bytes());
                vector.extend_from_slice(&type_hash.as_bytes()[0..8]); // First 8 bytes
            }
        }

        // Add capability complexity measures
        for capability in capabilities {
            let complexity = capability.inputs.len() + capability.outputs.len();
            vector.extend_from_slice(&(complexity as u32).to_le_bytes());

            // Confidence weighting
            let weighted_confidence = (capability.confidence * 255.0) as u8;
            vector.push(weighted_confidence);
        }

        vector
    }

    /// Get parameters used for fingerprint computation
    fn get_parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert(
            "include_relationships".to_string(),
            self.include_relationships.to_string(),
        );
        for (key, value) in &self.aspect_weights {
            params.insert(format!("weight_{}", key), value.to_string());
        }
        params
    }

    /// Calculate semantic coherence between components
    fn calculate_semantic_coherence(&self, components: &SemanticComponents) -> f32 {
        let mut coherence_score = 0.0;
        let mut total_factors = 0;

        // Capability-domain coherence
        if let Some(ref domain) = components.domain {
            let domain_coherence = components
                .capabilities
                .iter()
                .map(|cap| self.calculate_domain_alignment(&cap.name, domain))
                .sum::<f32>()
                / components.capabilities.len().max(1) as f32;
            coherence_score += domain_coherence;
            total_factors += 1;
        }

        // Intent-capability coherence
        if let Some(ref intent) = components.intent {
            let intent_coherence = components
                .capabilities
                .iter()
                .map(|cap| self.calculate_intent_alignment(&cap.description, intent))
                .sum::<f32>()
                / components.capabilities.len().max(1) as f32;
            coherence_score += intent_coherence;
            total_factors += 1;
        }

        // Relationship consistency
        if !components.relationships.is_empty() {
            let relationship_coherence =
                self.calculate_relationship_coherence(&components.relationships);
            coherence_score += relationship_coherence;
            total_factors += 1;
        }

        if total_factors > 0 {
            coherence_score / total_factors as f32
        } else {
            0.5 // Default coherence when no factors available
        }
    }
    /// Calculate domain alignment score for a capability
    fn calculate_domain_alignment(&self, capability_name: &str, domain: &str) -> f32 {
        // Simple semantic similarity based on common words/stems
        let cap_lower = capability_name.to_lowercase();
        let domain_lower = domain.to_lowercase();
        let cap_words: Vec<&str> = cap_lower.split_whitespace().collect();
        let domain_words: Vec<&str> = domain_lower.split_whitespace().collect();

        let common_words = cap_words
            .iter()
            .filter(|word| domain_words.contains(word))
            .count();

        if cap_words.is_empty() || domain_words.is_empty() {
            0.0
        } else {
            (common_words as f32) / (cap_words.len().max(domain_words.len()) as f32)
        }
    }

    /// Calculate intent alignment score for a capability
    fn calculate_intent_alignment(&self, capability_desc: &str, intent: &str) -> f32 {
        // Simple semantic similarity based on common words
        let desc_lower = capability_desc.to_lowercase();
        let intent_lower = intent.to_lowercase();
        let desc_words: Vec<&str> = desc_lower.split_whitespace().collect();
        let intent_words: Vec<&str> = intent_lower.split_whitespace().collect();

        let common_words = desc_words
            .iter()
            .filter(|word| intent_words.contains(word))
            .count();

        if desc_words.is_empty() || intent_words.is_empty() {
            0.0
        } else {
            (common_words as f32) / (desc_words.len().max(intent_words.len()) as f32)
        }
    }

    /// Calculate relationship coherence score
    fn calculate_relationship_coherence(&self, relationships: &[SemanticRelationship]) -> f32 {
        if relationships.is_empty() {
            return 1.0;
        }

        // Calculate variance in relationship strengths
        let avg_strength =
            relationships.iter().map(|r| r.strength).sum::<f32>() / relationships.len() as f32;
        let variance = relationships
            .iter()
            .map(|r| (r.strength - avg_strength).powi(2))
            .sum::<f32>()
            / relationships.len() as f32;

        // Lower variance = higher coherence
        1.0 - variance.sqrt().min(1.0)
    }

    /// Compute relationship strength matrix
    fn compute_relationship_matrix(&self, relationships: &[SemanticRelationship]) -> Vec<u8> {
        let mut matrix_data = Vec::new();

        if let Some(rel_weight) = self.aspect_weights.get("relationships") {
            for relationship in relationships {
                matrix_data.extend_from_slice(relationship.target.as_bytes());
                matrix_data.extend_from_slice(&relationship.strength.to_le_bytes());
                matrix_data.extend_from_slice(&rel_weight.to_le_bytes());
            }
        }

        matrix_data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestSemanticInput;

    impl SemanticFingerprintable for TestSemanticInput {
        fn get_semantic_components(&self) -> SemanticComponents {
            SemanticComponents {
                capabilities: vec![],
                relationships: vec![],
                intent: None,
                domain: None,
            }
        }
    }

    #[test]
    fn test_semantic_fingerprint_engine_creation() {
        let engine = SemanticFingerprintEngine::new();
        assert!(engine.include_relationships);
    }

    #[test]
    fn test_semantic_fingerprint_computation() {
        let engine = SemanticFingerprintEngine::new();
        let input = TestSemanticInput;

        let result = engine.compute_fingerprint(&input);
        assert!(result.is_ok());

        let fingerprint = result.unwrap();
        assert_eq!(fingerprint.level, FingerprintLevel::Semantic);
    }
}
