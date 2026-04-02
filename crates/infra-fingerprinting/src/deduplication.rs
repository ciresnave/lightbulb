//! Deduplication system using multi-level fingerprints
//!
//! This module provides deduplication capabilities across all fingerprint levels,
//! enabling efficient identification and removal of duplicate content.

use crate::{Fingerprint, FingerprintLevel};
use std::collections::{HashMap, HashSet};

/// Deduplication engine that operates across multiple fingerprint levels
pub struct DeduplicationEngine {
    /// Threshold for considering fingerprints similar (0.0 to 1.0)
    similarity_threshold: f32,
    /// Cache of known fingerprints
    fingerprint_cache: HashMap<FingerprintLevel, HashSet<[u8; 32]>>,
}

impl Default for DeduplicationEngine {
    fn default() -> Self {
        Self::new(0.85)
    }
}

impl DeduplicationEngine {
    /// Create a new deduplication engine with the given similarity threshold
    pub fn new(similarity_threshold: f32) -> Self {
        Self {
            similarity_threshold: similarity_threshold.clamp(0.0, 1.0),
            fingerprint_cache: HashMap::new(),
        }
    }

    /// Check if a fingerprint represents duplicate content
    pub fn is_duplicate(&self, fingerprint: &Fingerprint) -> bool {
        if let Some(cache) = self.fingerprint_cache.get(&fingerprint.level) {
            cache.contains(&fingerprint.hash)
        } else {
            false
        }
    }

    /// Add a fingerprint to the cache
    pub fn add_fingerprint(&mut self, fingerprint: &Fingerprint) {
        self.fingerprint_cache
            .entry(fingerprint.level)
            .or_insert_with(HashSet::new)
            .insert(fingerprint.hash);
    }

    /// Remove a fingerprint from the cache
    pub fn remove_fingerprint(&mut self, fingerprint: &Fingerprint) -> bool {
        if let Some(cache) = self.fingerprint_cache.get_mut(&fingerprint.level) {
            cache.remove(&fingerprint.hash)
        } else {
            false
        }
    }

    /// Advanced pattern-based duplicate detection using similarity thresholds
    pub fn is_similar_duplicate(
        &self,
        fingerprint: &Fingerprint,
        existing_fingerprints: &[Fingerprint],
    ) -> Option<usize> {
        for (index, existing) in existing_fingerprints.iter().enumerate() {
            if existing.level == fingerprint.level {
                let similarity = self.calculate_similarity_score(&fingerprint.hash, &existing.hash);
                if similarity >= self.similarity_threshold {
                    return Some(index);
                }
            }
        }
        None
    }

    /// Calculate similarity score between two hash arrays using multiple metrics
    fn calculate_similarity_score(&self, hash1: &[u8; 32], hash2: &[u8; 32]) -> f32 {
        // Combine multiple similarity metrics for robust comparison
        let hamming_sim = self.hamming_similarity(hash1, hash2);
        let jaccard_sim = self.jaccard_similarity(hash1, hash2);
        let cosine_sim = self.cosine_similarity(hash1, hash2);

        // Weighted combination of similarity metrics
        0.5 * hamming_sim + 0.3 * jaccard_sim + 0.2 * cosine_sim
    }

    /// Hamming similarity (1 - normalized hamming distance)
    fn hamming_similarity(&self, hash1: &[u8; 32], hash2: &[u8; 32]) -> f32 {
        let different_bits = hash1
            .iter()
            .zip(hash2.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum::<u32>();

        let total_bits = 32 * 8; // 256 bits total
        1.0 - (different_bits as f32 / total_bits as f32)
    }

    /// Jaccard similarity for byte arrays
    fn jaccard_similarity(&self, hash1: &[u8; 32], hash2: &[u8; 32]) -> f32 {
        let set1: HashSet<u8> = hash1.iter().cloned().collect();
        let set2: HashSet<u8> = hash2.iter().cloned().collect();

        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();

        if union == 0 {
            1.0 // Both are empty
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Cosine similarity treating byte arrays as vectors
    fn cosine_similarity(&self, hash1: &[u8; 32], hash2: &[u8; 32]) -> f32 {
        let dot_product: f32 = hash1
            .iter()
            .zip(hash2.iter())
            .map(|(a, b)| (*a as f32) * (*b as f32))
            .sum();

        let norm1: f32 = hash1
            .iter()
            .map(|x| (*x as f32).powi(2))
            .sum::<f32>()
            .sqrt();
        let norm2: f32 = hash2
            .iter()
            .map(|x| (*x as f32).powi(2))
            .sum::<f32>()
            .sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1 * norm2)
        }
    }
    /// Perform batch deduplication on a collection of fingerprints
    pub fn deduplicate_batch(&mut self, fingerprints: Vec<Fingerprint>) -> Vec<Fingerprint> {
        let mut deduplicated = Vec::new();
        let mut seen_hashes: HashMap<FingerprintLevel, Vec<Fingerprint>> = HashMap::new();

        for fingerprint in fingerprints {
            let mut is_duplicate = false;

            // Check exact duplicates first
            if self.is_duplicate(&fingerprint) {
                continue;
            }

            // Check similarity-based duplicates
            if let Some(similar_hashes) = seen_hashes.get(&fingerprint.level) {
                if let Some(_similar_index) =
                    self.is_similar_duplicate(&fingerprint, similar_hashes)
                {
                    is_duplicate = true;
                }
            }

            if !is_duplicate {
                // Add to seen hashes for this level
                seen_hashes
                    .entry(fingerprint.level)
                    .or_insert_with(Vec::new)
                    .push(fingerprint.clone());

                // Add to cache and result
                self.add_fingerprint(&fingerprint);
                deduplicated.push(fingerprint);
            }
        }

        deduplicated
    }

    /// Get statistics about cached fingerprints
    pub fn get_statistics(&self) -> DeduplicationStatistics {
        let mut total_fingerprints = 0;
        let mut level_counts = HashMap::new();

        for (level, cache) in &self.fingerprint_cache {
            let count = cache.len();
            level_counts.insert(*level, count);
            total_fingerprints += count;
        }

        DeduplicationStatistics {
            total_fingerprints,
            level_counts,
            similarity_threshold: self.similarity_threshold,
        }
    }

    /// Clear all cached fingerprints
    pub fn clear_cache(&mut self) {
        self.fingerprint_cache.clear();
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> HashMap<FingerprintLevel, usize> {
        self.fingerprint_cache
            .iter()
            .map(|(level, cache)| (*level, cache.len()))
            .collect()
    }

    /// Clear cache for a specific fingerprint level
    pub fn clear_level_cache(&mut self, level: FingerprintLevel) {
        self.fingerprint_cache.remove(&level);
    }

    /// Get total number of cached fingerprints across all levels
    pub fn total_cached_fingerprints(&self) -> usize {
        self.fingerprint_cache
            .values()
            .map(|cache| cache.len())
            .sum()
    }
}

/// Statistics about the deduplication system
#[derive(Debug, Clone)]
pub struct DeduplicationStatistics {
    /// Total number of cached fingerprints
    pub total_fingerprints: usize,
    /// Number of fingerprints per level
    pub level_counts: HashMap<FingerprintLevel, usize>,
    /// Current similarity threshold
    pub similarity_threshold: f32,
}
