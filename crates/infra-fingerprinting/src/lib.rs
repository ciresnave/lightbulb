//! Multi-Level Fingerprinting System for DynAniML
//!
//! This crate provides a comprehensive fingerprinting system that operates at multiple levels:
//! - Atomic level: Individual concepts/tokens within dimensional context
//! - Relational level: Graph isomorphism signatures for relationship patterns  
//! - Structural level: Knowledge chunk topology hashing
//! - Semantic level: Functional capability fingerprinting
//!
//! The system enables efficient deduplication, similarity matching, and knowledge transfer
//! learning across the DynAniML ecosystem.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

pub mod atomic;
pub mod deduplication;
pub mod relational;
pub mod semantic;
pub mod similarity;
pub mod structural;

/// Core fingerprint type that can represent different levels of abstraction
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fingerprint {
    /// The fingerprint value as a 256-bit hash
    pub hash: [u8; 32],
    /// The level at which this fingerprint was computed
    pub level: FingerprintLevel,
    /// Optional metadata about how this fingerprint was computed
    pub metadata: Option<FingerprintMetadata>,
}

/// Different levels at which fingerprints can be computed
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FingerprintLevel {
    /// Individual tokens/concepts within dimensional context
    Atomic,
    /// Relationship patterns and graph structures
    Relational,
    /// Knowledge chunk topology and organization
    Structural,
    /// Functional capabilities and semantic meaning
    Semantic,
}

/// Metadata about fingerprint computation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FingerprintMetadata {
    /// Algorithm used for computation
    pub algorithm: String,
    /// Parameters used in computation
    pub parameters: HashMap<String, String>,
    /// Timestamp of computation
    pub timestamp: u64,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
}

/// Configuration for fingerprint computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FingerprintConfig {
    /// Which levels to compute fingerprints for
    pub levels: Vec<FingerprintLevel>,
    /// Whether to include metadata in fingerprints
    pub include_metadata: bool,
    /// Similarity threshold for deduplication (0.0 to 1.0)
    pub similarity_threshold: f32,
    /// Whether to use parallel processing
    pub parallel: bool,
}

/// Errors that can occur during fingerprinting operations
#[derive(Error, Debug)]
pub enum FingerprintError {
    #[error("Invalid input data: {0}")]
    InvalidInput(String),

    #[error("Computation failed: {0}")]
    ComputationFailed(String),

    #[error("Unsupported fingerprint level: {level:?}")]
    UnsupportedLevel { level: FingerprintLevel },

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type for fingerprinting operations
pub type FingerprintResult<T> = Result<T, FingerprintError>;

/// Main fingerprinting engine that coordinates all levels
pub struct FingerprintEngine {
    config: FingerprintConfig,
    atomic_engine: atomic::AtomicFingerprintEngine,
    relational_engine: relational::RelationalFingerprintEngine,
    structural_engine: structural::StructuralFingerprintEngine,
    semantic_engine: semantic::SemanticFingerprintEngine,
}

impl Default for FingerprintConfig {
    fn default() -> Self {
        Self {
            levels: vec![
                FingerprintLevel::Atomic,
                FingerprintLevel::Relational,
                FingerprintLevel::Structural,
                FingerprintLevel::Semantic,
            ],
            include_metadata: true,
            similarity_threshold: 0.85,
            parallel: true,
        }
    }
}

impl Fingerprint {
    /// Create a new fingerprint with the given hash and level
    pub fn new(hash: [u8; 32], level: FingerprintLevel) -> Self {
        Self {
            hash,
            level,
            metadata: None,
        }
    }

    /// Create a fingerprint with metadata
    pub fn with_metadata(
        hash: [u8; 32],
        level: FingerprintLevel,
        metadata: FingerprintMetadata,
    ) -> Self {
        Self {
            hash,
            level,
            metadata: Some(metadata),
        }
    }

    /// Get the fingerprint as a hex string
    pub fn to_hex(&self) -> String {
        hex::encode(self.hash)
    }

    /// Create a fingerprint from a hex string
    pub fn from_hex(hex_str: &str, level: FingerprintLevel) -> FingerprintResult<Self> {
        let bytes = hex::decode(hex_str)
            .map_err(|e| FingerprintError::InvalidInput(format!("Invalid hex: {}", e)))?;

        if bytes.len() != 32 {
            return Err(FingerprintError::InvalidInput(
                "Hash must be exactly 32 bytes".to_string(),
            ));
        }

        let mut hash = [0u8; 32];
        hash.copy_from_slice(&bytes);

        Ok(Self::new(hash, level))
    }
}

impl FingerprintEngine {
    /// Create a new fingerprint engine with the given configuration
    pub fn new(config: FingerprintConfig) -> Self {
        Self {
            atomic_engine: atomic::AtomicFingerprintEngine::new(),
            relational_engine: relational::RelationalFingerprintEngine::new(),
            structural_engine: structural::StructuralFingerprintEngine::new(),
            semantic_engine: semantic::SemanticFingerprintEngine::new(),
            config,
        }
    }

    /// Create a fingerprint engine with default configuration
    pub fn default() -> Self {
        Self::new(FingerprintConfig::default())
    }

    /// Compute fingerprints for the given input at all configured levels
    pub fn compute_fingerprints<T>(&self, input: &T) -> FingerprintResult<Vec<Fingerprint>>
    where
        T: atomic::AtomicFingerprintable
            + relational::RelationalFingerprintable
            + structural::StructuralFingerprintable
            + semantic::SemanticFingerprintable,
    {
        let mut results = Vec::new();

        for level in &self.config.levels {
            let fingerprint = match level {
                FingerprintLevel::Atomic => self.atomic_engine.compute_fingerprint(input)?,
                FingerprintLevel::Relational => {
                    self.relational_engine.compute_fingerprint(input)?
                }
                FingerprintLevel::Structural => {
                    self.structural_engine.compute_fingerprint(input)?
                }
                FingerprintLevel::Semantic => self.semantic_engine.compute_fingerprint(input)?,
            };

            results.push(fingerprint);
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint_creation() {
        let hash = [1u8; 32];
        let fp = Fingerprint::new(hash, FingerprintLevel::Atomic);

        assert_eq!(fp.hash, hash);
        assert_eq!(fp.level, FingerprintLevel::Atomic);
        assert!(fp.metadata.is_none());
    }

    #[test]
    fn test_fingerprint_hex_conversion() {
        let hash = [0xAB; 32];
        let fp = Fingerprint::new(hash, FingerprintLevel::Semantic);

        let hex_str = fp.to_hex();
        let fp2 = Fingerprint::from_hex(&hex_str, FingerprintLevel::Semantic).unwrap();

        assert_eq!(fp.hash, fp2.hash);
        assert_eq!(fp.level, fp2.level);
    }

    #[test]
    fn test_default_config() {
        let config = FingerprintConfig::default();

        assert_eq!(config.levels.len(), 4);
        assert!(config.include_metadata);
        assert_eq!(config.similarity_threshold, 0.85);
        assert!(config.parallel);
    }
}
