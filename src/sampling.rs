//! Sampling strategies (top-k, top-p, temperature) and utilities

use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            seed: 42,
        }
    }
}

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

/// Keep smallest set of logits whose softmax mass >= p (nucleus sampling)
pub fn top_p_filter(logits: &mut [f32], p: f32) {
    if !(0.0..=1.0).contains(&p) || p >= 1.0 {
        return;
    }
    // sort by logit desc, track cumulative softmax probs
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    // compute softmax of sorted logits in a numerically stable way
    let max_l = logits[idx[0]];
    let mut exp_sum = 0.0f32;
    let mut exps = vec![0.0f32; logits.len()];
    for &i in &idx {
        let e = (logits[i] - max_l).exp();
        exp_sum += e;
        exps[i] = e;
    }
    let mut cum = 0.0f32;
    let mut keep = vec![false; logits.len()];
    for &i in &idx {
        let prob = exps[i] / exp_sum;
        cum += prob;
        keep[i] = true;
        if cum >= p {
            break;
        }
    }
    for (i, l) in logits.iter_mut().enumerate() {
        if !keep[i] {
            *l = f32::NEG_INFINITY;
        }
    }
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
