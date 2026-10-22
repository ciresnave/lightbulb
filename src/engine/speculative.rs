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
use anyhow::Result;

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
    pub fn record_round(
        &mut self,
        accepted: usize,
        rejected: usize,
        draft_us: u64,
        target_us: u64,
    ) {
        self.speculation_rounds += 1;
        self.accepted_tokens += accepted;
        self.rejected_tokens += rejected;
        self.total_tokens += accepted + 1; // +1 for the token we always get from target
        self.draft_time_us += draft_us;
        self.target_time_us += target_us;
    }
}

// `SpeculativeModel`/`SpeculativeDecoder` (the Tensor-typed draft/target-model trait and
// the actual accept/reject algorithm) removed 2026-10-22 (candlelight removal, CireSnave:
// "Remove all dependencies on Candle and Candlelight..."): candlelight-typed throughout,
// zero external callers beyond this config/stats pair. See
// `docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md`.
