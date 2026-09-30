//! Sampling strategies (top-k, temperature) and utilities

use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Apply temperature scaling to logits in-place
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if temperature <= 0.0 || (temperature - 1.0).abs() < f32::EPSILON {
        return;
    }
    let inv_t = 1.0 / temperature;
    logits.iter_mut().for_each(|l| *l *= inv_t);
}

/// Keep top-k logits (others set to very low value)
pub fn top_k_filter(logits: &mut [f32], k: usize) {
    if k == 0 || k >= logits.len() {
        return; // no-op
    }
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    let threshold = logits[idx[k - 1]];
    for l in logits.iter_mut() {
        if *l < threshold {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// Derive one token's sampling seed from the request it belongs to and its
/// position within that request.
///
/// NOT a shared counter on the model. A single counter incremented across
/// every request makes one request's sampled tokens depend on how much
/// traffic the process served before it — the same prompt at
/// `temperature: 0.7` would return different text depending on what ran
/// earlier. Harmless only by accident on a serial decode loop, since nothing
/// interleaves; it becomes real cross-tenant coupling the moment a genuine
/// batch has two requests stepping in the same call and sharing the
/// counter's sequence. Hashing `(request_id, token_index)` instead makes
/// each request's seed sequence depend only on itself.
///
/// Shared by both decode paths (`model_fuel::engine_model` and
/// `model::parallel_model_manager`) rather than duplicated — this is the
/// one thing both need to get right identically, and a second copy is a
/// second thing to keep in sync.
pub fn seed_for(request_id: &str, token_index: usize) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    request_id.hash(&mut hasher);
    token_index.hash(&mut hasher);
    hasher.finish()
}

/// Sample an index from (filtered) logits using a seeded RNG for reproducibility
pub fn sample_from_logits(logits: &[f32], seed: u64) -> usize {
    // Convert logits to probabilities via softmax
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|l| (l - max_l).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum == 0.0 || !sum.is_finite() {
        // fallback: uniform
        let mut rng = StdRng::seed_from_u64(seed);
        return rng.gen_range(0..logits.len());
    }
    for p in &mut probs {
        *p /= sum;
    }
    let dist = WeightedIndex::new(&probs).expect("valid probs");
    let mut rng = StdRng::seed_from_u64(seed);
    dist.sample(&mut rng)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_k() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        top_k_filter(&mut logits, 2);
        // Only the two largest should remain finite
        assert!(logits[3].is_finite() && logits[2].is_finite());
        assert!(logits[1].is_infinite() && logits[0].is_infinite());
    }

    #[test]
    fn test_sample() {
        let logits = vec![0.0, 0.0, 10.0];
        let idx = sample_from_logits(&logits, 123);
        assert_eq!(idx, 2);
    }
}
