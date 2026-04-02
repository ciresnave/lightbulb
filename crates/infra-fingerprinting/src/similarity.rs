//! Similarity computation between fingerprints
//!
//! This module provides algorithms for computing similarity scores between
//! fingerprints at different levels, enabling fuzzy matching and deduplication.

use crate::{Fingerprint, FingerprintError, FingerprintLevel, FingerprintResult};
use std::collections::HashMap;

/// Similarity score between 0.0 (completely different) and 1.0 (identical)
pub type SimilarityScore = f32;

/// Similarity computation engine
pub struct SimilarityEngine {
    /// Weights for different fingerprint levels when computing overall similarity
    level_weights: HashMap<FingerprintLevel, f32>,
}

impl Default for SimilarityEngine {
    fn default() -> Self {
        let mut level_weights = HashMap::new();
        level_weights.insert(FingerprintLevel::Atomic, 0.2);
        level_weights.insert(FingerprintLevel::Relational, 0.3);
        level_weights.insert(FingerprintLevel::Structural, 0.3);
        level_weights.insert(FingerprintLevel::Semantic, 0.2);

        Self { level_weights }
    }
}

impl SimilarityEngine {
    /// Create a new similarity engine with custom level weights
    pub fn with_weights(level_weights: HashMap<FingerprintLevel, f32>) -> Self {
        Self { level_weights }
    }

    /// Compute similarity between two fingerprints
    pub fn compute_similarity(
        &self,
        fp1: &Fingerprint,
        fp2: &Fingerprint,
    ) -> FingerprintResult<SimilarityScore> {
        if fp1.level != fp2.level {
            return Err(FingerprintError::InvalidInput(
                "Cannot compare fingerprints of different levels".to_string(),
            ));
        }

        match fp1.level {
            FingerprintLevel::Atomic => self.compute_atomic_similarity(fp1, fp2),
            FingerprintLevel::Relational => self.compute_relational_similarity(fp1, fp2),
            FingerprintLevel::Structural => self.compute_structural_similarity(fp1, fp2),
            FingerprintLevel::Semantic => self.compute_semantic_similarity(fp1, fp2),
        }
    }

    /// Compute overall similarity between two sets of fingerprints
    pub fn compute_overall_similarity(
        &self,
        fps1: &[Fingerprint],
        fps2: &[Fingerprint],
    ) -> FingerprintResult<SimilarityScore> {
        let mut total_weight = 0.0;
        let mut weighted_similarity = 0.0;

        for level in [
            FingerprintLevel::Atomic,
            FingerprintLevel::Relational,
            FingerprintLevel::Structural,
            FingerprintLevel::Semantic,
        ] {
            let fp1 = fps1.iter().find(|fp| fp.level == level);
            let fp2 = fps2.iter().find(|fp| fp.level == level);

            if let (Some(fp1), Some(fp2)) = (fp1, fp2) {
                let weight = self.level_weights.get(&level).unwrap_or(&0.25);
                let similarity = self.compute_similarity(fp1, fp2)?;

                weighted_similarity += similarity * weight;
                total_weight += weight;
            }
        }

        if total_weight > 0.0 {
            Ok(weighted_similarity / total_weight)
        } else {
            Err(FingerprintError::InvalidInput(
                "No matching fingerprint levels found".to_string(),
            ))
        }
    }

    /// Compute Hamming distance based similarity for atomic fingerprints
    fn compute_atomic_similarity(
        &self,
        fp1: &Fingerprint,
        fp2: &Fingerprint,
    ) -> FingerprintResult<SimilarityScore> {
        let distance = hamming_distance(&fp1.hash, &fp2.hash);
        let max_distance = fp1.hash.len() * 8; // 8 bits per byte
        let similarity = 1.0 - (distance as f32 / max_distance as f32);
        Ok(similarity)
    }
    /// Compute similarity for relational fingerprints using graph-aware metrics
    fn compute_relational_similarity(
        &self,
        fp1: &Fingerprint,
        fp2: &Fingerprint,
    ) -> FingerprintResult<SimilarityScore> {
        // Use a combination of Hamming distance and graph structure similarity
        let hamming_sim = self.compute_atomic_similarity(fp1, fp2)?;

        // Additional graph-aware similarity using Jaccard index
        let jaccard_sim = self.jaccard_similarity(&fp1.hash, &fp2.hash);

        // Weighted combination favoring structural relationships
        let combined_score = 0.4 * hamming_sim + 0.6 * jaccard_sim;

        Ok(combined_score)
    }

    /// Compute similarity for structural fingerprints using topology-aware metrics
    fn compute_structural_similarity(
        &self,
        fp1: &Fingerprint,
        fp2: &Fingerprint,
    ) -> FingerprintResult<SimilarityScore> {
        // Use a combination of Hamming distance and structural pattern similarity
        let hamming_sim = self.compute_atomic_similarity(fp1, fp2)?;

        // Additional structural similarity using cosine similarity
        let cosine_sim = self.cosine_similarity(&fp1.hash, &fp2.hash);

        // Weighted combination emphasizing structural patterns
        let combined_score = 0.3 * hamming_sim + 0.7 * cosine_sim;

        Ok(combined_score)
    }

    /// Compute similarity for semantic fingerprints using semantic distance metrics
    fn compute_semantic_similarity(
        &self,
        fp1: &Fingerprint,
        fp2: &Fingerprint,
    ) -> FingerprintResult<SimilarityScore> {
        // Use a combination of multiple semantic similarity metrics
        let hamming_sim = self.compute_atomic_similarity(fp1, fp2)?;
        let jaccard_sim = self.jaccard_similarity(&fp1.hash, &fp2.hash);
        let cosine_sim = self.cosine_similarity(&fp1.hash, &fp2.hash);

        // Weighted combination optimized for semantic content
        let semantic_score = 0.2 * hamming_sim + 0.4 * jaccard_sim + 0.4 * cosine_sim;

        Ok(semantic_score)
    }

    /// Compute Jaccard similarity between two hash arrays
    fn jaccard_similarity(&self, hash1: &[u8; 32], hash2: &[u8; 32]) -> f32 {
        let mut intersection = 0;
        let mut union = 0;

        for i in 0..32 {
            let byte1 = hash1[i];
            let byte2 = hash2[i];

            for bit in 0..8 {
                let bit1 = (byte1 >> bit) & 1;
                let bit2 = (byte2 >> bit) & 1;

                if bit1 == 1 || bit2 == 1 {
                    union += 1;
                    if bit1 == 1 && bit2 == 1 {
                        intersection += 1;
                    }
                }
            }
        }

        if union == 0 {
            1.0 // Both hashes are all zeros, consider them identical
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Compute cosine similarity between two hash arrays
    fn cosine_similarity(&self, hash1: &[u8; 32], hash2: &[u8; 32]) -> f32 {
        let mut dot_product = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;

        for i in 0..32 {
            let val1 = hash1[i] as f32;
            let val2 = hash2[i] as f32;

            dot_product += val1 * val2;
            norm1 += val1 * val1;
            norm2 += val2 * val2;
        }

        let norm1 = norm1.sqrt();
        let norm2 = norm2.sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1 * norm2)
        }
    }
}

/// Compute Hamming distance between two byte arrays
fn hamming_distance(a: &[u8], b: &[u8]) -> usize {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones() as usize)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FingerprintLevel;
    #[test]
    fn test_hamming_distance() {
        let a = [0b11110000, 0b00001111];
        let b = [0b11110000, 0b11110000];
        // First byte: same (0 differences)
        // Second byte: 0b00001111 ^ 0b11110000 = 0b11111111 (8 differences)
        assert_eq!(hamming_distance(&a, &b), 8);
    }

    #[test]
    fn test_identical_fingerprints() {
        let engine = SimilarityEngine::default();
        let hash = [0u8; 32];
        let fp1 = Fingerprint::new(hash, FingerprintLevel::Atomic);
        let fp2 = Fingerprint::new(hash, FingerprintLevel::Atomic);

        let similarity = engine.compute_similarity(&fp1, &fp2).unwrap();
        assert_eq!(similarity, 1.0);
    }

    #[test]
    fn test_different_level_fingerprints() {
        let engine = SimilarityEngine::default();
        let hash = [0u8; 32];
        let fp1 = Fingerprint::new(hash, FingerprintLevel::Atomic);
        let fp2 = Fingerprint::new(hash, FingerprintLevel::Semantic);

        assert!(engine.compute_similarity(&fp1, &fp2).is_err());
    }
}
