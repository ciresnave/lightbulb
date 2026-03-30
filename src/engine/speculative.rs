//! Speculative Decoding
//!
//! This module implements speculative decoding, where a small "draft" model
//! generates multiple tokens quickly, and a larger "target" model verifies them.
//! Accepted tokens provide speedup; rejected tokens fall back to standard generation.
//!
//! # Algorithm
//!
//! 1. Draft model generates K speculative tokens (typically 4-8)
//! 2. Target model evaluates all K tokens in parallel
//! 3. Accept longest prefix where draft and target agree
//! 4. If rejection occurs, use target's token and repeat
//!
//! # Speedup
//!
//! - Best case: K tokens accepted → K× speedup (rare)
//! - Typical case: 2-4 tokens accepted → 1.3-2× speedup
//! - Worst case: 0 tokens accepted → small overhead from draft model
//!
//! # References
//!
//! - "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., 2023)
//! - "SpecInfer: Accelerating Generative LLM Serving with Speculative Inference" (Miao et al., 2023)

use anyhow::{Context, Result};
use candlelight::core::{Device, Tensor};

/// Configuration for speculative decoding
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of speculative tokens to generate per iteration
    pub num_speculative_tokens: usize,

    /// Minimum acceptance rate before falling back to standard decoding
    /// (e.g., 0.3 = require 30% token acceptance rate)
    pub min_acceptance_rate: f64,

    /// Enable/disable speculative decoding
    pub enabled: bool,

    /// Fallback to standard decoding if acceptance rate drops below threshold
    pub auto_fallback: bool,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            num_speculative_tokens: 5,
            min_acceptance_rate: 0.3,
            enabled: true,
            auto_fallback: true,
        }
    }
}

/// Statistics tracking for speculative decoding performance
#[derive(Debug, Clone, Default)]
pub struct SpeculativeStats {
    /// Total tokens generated
    pub total_tokens: usize,

    /// Tokens accepted from speculation
    pub accepted_tokens: usize,

    /// Tokens rejected (had to use target model)
    pub rejected_tokens: usize,

    /// Number of speculative rounds
    pub speculation_rounds: usize,

    /// Total time spent in draft model (microseconds)
    pub draft_time_us: u64,

    /// Total time spent in target model (microseconds)
    pub target_time_us: u64,
}

impl SpeculativeStats {
    /// Calculate acceptance rate (0.0 to 1.0)
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_tokens == 0 {
            return 0.0;
        }
        self.accepted_tokens as f64 / self.total_tokens as f64
    }

    /// Calculate effective speedup vs standard decoding
    pub fn speedup(&self) -> f64 {
        if self.speculation_rounds == 0 {
            return 1.0;
        }
        // Speedup = total_tokens / (rounds + rejected_tokens)
        // Each round generates 1 token minimum (from target)
        let effective_target_calls = self.speculation_rounds + self.rejected_tokens;
        if effective_target_calls == 0 {
            return 1.0;
        }
        self.total_tokens as f64 / effective_target_calls as f64
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Record a speculation round result
    pub fn record_round(&mut self, accepted: usize, rejected: usize, draft_us: u64, target_us: u64) {
        self.speculation_rounds += 1;
        self.accepted_tokens += accepted;
        self.rejected_tokens += rejected;
        self.total_tokens += accepted + 1; // +1 for the token we always get from target
        self.draft_time_us += draft_us;
        self.target_time_us += target_us;
    }
}

/// Trait for models that can participate in speculative decoding
///
/// Both draft and target models implement this interface.
pub trait SpeculativeModel: Send + Sync {
    /// Generate next token logits given input tokens
    ///
    /// # Arguments
    ///
    /// * `tokens` - Input token sequence
    /// * `position` - Current position in sequence
    ///
    /// # Returns
    ///
    /// Logits tensor of shape [vocab_size]
    fn forward_logits(&mut self, tokens: &[u32], position: usize) -> Result<Tensor>;

    /// Get the device this model runs on
    fn device(&self) -> &Device;

    /// Get vocabulary size
    fn vocab_size(&self) -> usize;

    /// Reset KV cache (for new sequence)
    fn reset_cache(&mut self);
}

/// Speculative decoder that orchestrates draft and target models
pub struct SpeculativeDecoder {
    /// Configuration
    config: SpeculativeConfig,

    /// Performance statistics
    stats: SpeculativeStats,

    /// Whether we've fallen back to standard decoding
    fallback_active: bool,
}

impl SpeculativeDecoder {
    /// Create a new speculative decoder
    pub fn new(config: SpeculativeConfig) -> Self {
        Self {
            config,
            stats: SpeculativeStats::default(),
            fallback_active: false,
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }

    /// Check if we should use speculative decoding or fall back to standard
    pub fn should_speculate(&self) -> bool {
        if !self.config.enabled {
            return false;
        }

        if self.fallback_active {
            return false;
        }

        // If we have enough history, check acceptance rate
        if self.stats.speculation_rounds >= 10 {
            if self.stats.acceptance_rate() < self.config.min_acceptance_rate {
                return false;
            }
        }

        true
    }

    /// Generate one or more tokens using speculative decoding
    ///
    /// # Arguments
    ///
    /// * `draft_model` - Small, fast draft model
    /// * `target_model` - Large, accurate target model
    /// * `context_tokens` - Current token sequence
    /// * `sampler` - Function to sample from logits
    ///
    /// # Returns
    ///
    /// Vector of accepted tokens (length 1 to num_speculative_tokens+1)
    pub fn generate_tokens<F>(
        &mut self,
        draft_model: &mut dyn SpeculativeModel,
        target_model: &mut dyn SpeculativeModel,
        context_tokens: &[u32],
        mut sampler: F,
    ) -> Result<Vec<u32>>
    where
        F: FnMut(&Tensor) -> Result<u32>,
    {
        if !self.should_speculate() {
            // Fallback to standard decoding
            return self.standard_generate(target_model, context_tokens, sampler);
        }

        let start = std::time::Instant::now();

        // Phase 1: Draft model generates K speculative tokens
        let mut draft_tokens = Vec::with_capacity(self.config.num_speculative_tokens);
        let mut current_context = context_tokens.to_vec();

        let draft_start = std::time::Instant::now();
        for _ in 0..self.config.num_speculative_tokens {
            let logits = draft_model
                .forward_logits(&current_context, current_context.len())
                .context("Draft model forward pass")?;

            let token = sampler(&logits).context("Sampling from draft model")?;
            draft_tokens.push(token);
            current_context.push(token);
        }
        let draft_time = draft_start.elapsed().as_micros() as u64;

        // Phase 2: Target model verifies all speculative tokens in parallel
        let target_start = std::time::Instant::now();

        // Extend context with all draft tokens for target verification
        let mut verify_context = context_tokens.to_vec();
        verify_context.extend_from_slice(&draft_tokens);

        // Get target model's opinion on each position
        let mut target_tokens = Vec::with_capacity(draft_tokens.len());
        for i in 0..draft_tokens.len() {
            let position = context_tokens.len() + i;
            let logits = target_model
                .forward_logits(&verify_context[..position + 1], position)
                .context("Target model verification")?;

            let token = sampler(&logits).context("Sampling from target model")?;
            target_tokens.push(token);
        }
        let target_time = target_start.elapsed().as_micros() as u64;

        // Phase 3: Find longest matching prefix
        let mut accepted_count = 0;
        for (draft_tok, target_tok) in draft_tokens.iter().zip(target_tokens.iter()) {
            if draft_tok == target_tok {
                accepted_count += 1;
            } else {
                break; // First mismatch - stop accepting
            }
        }

        let rejected_count = draft_tokens.len() - accepted_count;

        // Record statistics
        self.stats.record_round(accepted_count, rejected_count, draft_time, target_time);

        // Check if we should fall back
        if self.config.auto_fallback && self.stats.speculation_rounds >= 10 {
            if self.stats.acceptance_rate() < self.config.min_acceptance_rate {
                self.fallback_active = true;
            }
        }

        // Return accepted tokens plus one corrected token from target
        let mut result = draft_tokens[..accepted_count].to_vec();
        if accepted_count < target_tokens.len() {
            result.push(target_tokens[accepted_count]);
        }

        Ok(result)
    }

    /// Standard single-token generation (fallback path)
    fn standard_generate<F>(
        &mut self,
        target_model: &mut dyn SpeculativeModel,
        context_tokens: &[u32],
        mut sampler: F,
    ) -> Result<Vec<u32>>
    where
        F: FnMut(&Tensor) -> Result<u32>,
    {
        let logits = target_model
            .forward_logits(context_tokens, context_tokens.len())
            .context("Target model forward pass")?;

        let token = sampler(&logits).context("Sampling from target model")?;

        // Record as a round with 0 accepted speculative tokens
        self.stats.record_round(0, 0, 0, 0);

        Ok(vec![token])
    }

    /// Reset decoder state for a new sequence
    pub fn reset(&mut self) {
        self.stats.reset();
        self.fallback_active = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_acceptance_rate() {
        let mut stats = SpeculativeStats::default();
        assert_eq!(stats.acceptance_rate(), 0.0);

        stats.record_round(3, 2, 100, 200); // 3 accepted, 2 rejected → 4 total (3 + 1 from target)
        // acceptance_rate = accepted / total = 3 / 4 = 0.75
        assert!((stats.acceptance_rate() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_stats_speedup() {
        let mut stats = SpeculativeStats::default();
        assert_eq!(stats.speedup(), 1.0);

        // Round 1: 3 accepted, 2 rejected
        stats.record_round(3, 2, 100, 200);
        // total_tokens = 4 (3 accepted + 1 from target)
        // effective_target_calls = 1 round + 2 rejected = 3
        // speedup = 4 / 3 = 1.33
        assert!((stats.speedup() - 1.333).abs() < 0.01);
    }

    #[test]
    fn test_config_defaults() {
        let config = SpeculativeConfig::default();
        assert_eq!(config.num_speculative_tokens, 5);
        assert_eq!(config.min_acceptance_rate, 0.3);
        assert!(config.enabled);
        assert!(config.auto_fallback);
    }

    #[test]
    fn test_should_speculate() {
        let config = SpeculativeConfig::default();
        let mut decoder = SpeculativeDecoder::new(config);

        // Should speculate initially
        assert!(decoder.should_speculate());

        // Simulate poor acceptance rate
        for _ in 0..15 {
            decoder.stats.record_round(0, 5, 100, 200); // 0% acceptance
        }

        // Should not speculate after poor performance
        assert!(!decoder.should_speculate());
    }

    #[test]
    fn test_fallback_activation() {
        let config = SpeculativeConfig {
            auto_fallback: true,
            min_acceptance_rate: 0.3,
            ..Default::default()
        };
        let mut decoder = SpeculativeDecoder::new(config);

        // Record 10+ rounds with poor acceptance
        for _ in 0..12 {
            decoder.stats.record_round(0, 5, 100, 200);
        }

        // Calling should_speculate() checks acceptance and activates fallback
        assert!(!decoder.should_speculate());
    }
}
