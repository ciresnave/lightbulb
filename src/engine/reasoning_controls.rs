//! Reasoning Efficiency Controls
//!
//! Budget-aware decoding with chain-of-thought management, overthinking detection,
//! and adaptive reasoning depth control for efficient multi-step reasoning.
//!
//! # Features
//!
//! - **Budget-Aware Decoding**: Limit max reasoning chains/samples
//! - **Overthinking Detection**: Identify when model is repeating or degrading
//! - **Adaptive Depth**: Dynamically adjust reasoning steps based on complexity
//! - **Thought Termination**: Early stopping policies for efficient reasoning
//!
//! # Example
//!
//! ```rust,ignore
//! use lightbulb::engine::reasoning_controls::{ReasoningBudget, OverthinkingDetector};
//!
//! // Configure reasoning budget
//! let budget = ReasoningBudget::new()
//!     .max_chains(3)
//!     .max_steps_per_chain(10)
//!     .max_total_tokens(500);
//!
//! // Detect overthinking
//! let detector = OverthinkingDetector::new();
//! if detector.is_overthinking(&logits_history) {
//!     // Terminate early
//! }
//! ```

use std::collections::VecDeque;

/// Configuration for reasoning budget constraints
#[derive(Debug, Clone)]
pub struct ReasoningBudget {
    /// Maximum number of reasoning chains to explore
    pub max_chains: usize,

    /// Maximum steps per reasoning chain
    pub max_steps_per_chain: usize,

    /// Maximum total tokens across all chains
    pub max_total_tokens: usize,

    /// Enable budget enforcement
    pub enabled: bool,

    /// Verifier frequency (verify every N steps, 0 = disabled)
    pub verifier_frequency: usize,
}

impl Default for ReasoningBudget {
    fn default() -> Self {
        Self {
            max_chains: 3,
            max_steps_per_chain: 10,
            max_total_tokens: 500,
            enabled: true,
            verifier_frequency: 5,
        }
    }
}

impl ReasoningBudget {
    /// Create new budget with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum number of chains
    pub fn max_chains(mut self, max: usize) -> Self {
        self.max_chains = max;
        self
    }

    /// Set maximum steps per chain
    pub fn max_steps_per_chain(mut self, max: usize) -> Self {
        self.max_steps_per_chain = max;
        self
    }

    /// Set maximum total tokens
    pub fn max_total_tokens(mut self, max: usize) -> Self {
        self.max_total_tokens = max;
        self
    }

    /// Set verifier frequency
    pub fn verifier_frequency(mut self, freq: usize) -> Self {
        self.verifier_frequency = freq;
        self
    }

    /// Check if budget allows another chain
    pub fn can_start_chain(&self, current_chains: usize) -> bool {
        !self.enabled || current_chains < self.max_chains
    }

    /// Check if budget allows another step in current chain
    pub fn can_continue_chain(&self, current_steps: usize) -> bool {
        !self.enabled || current_steps < self.max_steps_per_chain
    }

    /// Check if total token budget is exhausted
    pub fn can_generate_tokens(&self, current_tokens: usize, new_tokens: usize) -> bool {
        !self.enabled || (current_tokens + new_tokens) <= self.max_total_tokens
    }

    /// Should run verifier at this step
    pub fn should_verify(&self, step: usize) -> bool {
        self.verifier_frequency > 0 && step % self.verifier_frequency == 0
    }
}

/// Overthinking detection using entropy and variance analysis
#[derive(Debug, Clone)]
pub struct OverthinkingDetector {
    /// Window size for entropy/variance tracking
    window_size: usize,

    /// Entropy threshold for detecting low-confidence loops
    entropy_threshold: f32,

    /// Variance threshold for detecting repetition
    variance_threshold: f32,

    /// Enable detection
    enabled: bool,
}

impl Default for OverthinkingDetector {
    fn default() -> Self {
        Self {
            window_size: 5,
            entropy_threshold: 0.5,
            variance_threshold: 0.1,
            enabled: true,
        }
    }
}

impl OverthinkingDetector {
    /// Create new detector with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set window size for analysis
    pub fn window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Set entropy threshold
    pub fn entropy_threshold(mut self, threshold: f32) -> Self {
        self.entropy_threshold = threshold;
        self
    }

    /// Set variance threshold
    pub fn variance_threshold(mut self, threshold: f32) -> Self {
        self.variance_threshold = threshold;
        self
    }

    /// Detect overthinking from logits history
    ///
    /// Returns true if:
    /// - Entropy is consistently low (model is uncertain)
    /// - Variance is low (model is repeating similar outputs)
    pub fn is_overthinking(&self, logits_history: &[Vec<f32>]) -> bool {
        if !self.enabled || logits_history.len() < self.window_size {
            return false;
        }

        // Analyze recent window
        let recent = &logits_history[logits_history.len() - self.window_size..];

        // Calculate average entropy
        let avg_entropy = recent
            .iter()
            .map(|logits| self.calculate_entropy(logits))
            .sum::<f32>()
            / self.window_size as f32;

        // Calculate variance across top predictions
        let top_probs: Vec<f32> = recent
            .iter()
            .map(|logits| self.get_top_prob(logits))
            .collect();
        let variance = self.calculate_variance(&top_probs);

        // Overthinking if entropy is low AND variance is low
        avg_entropy < self.entropy_threshold && variance < self.variance_threshold
    }

    /// Calculate entropy of logits (higher = more uncertain)
    fn calculate_entropy(&self, logits: &[f32]) -> f32 {
        // Convert logits to probabilities using softmax
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();

        let mut entropy = 0.0;
        for &logit in logits {
            let prob = (logit - max_logit).exp() / exp_sum;
            if prob > 1e-10 {
                // Avoid log(0)
                entropy -= prob * prob.ln();
            }
        }
        entropy
    }

    /// Get top probability from logits
    fn get_top_prob(&self, logits: &[f32]) -> f32 {
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
        (max_logit - max_logit).exp() / exp_sum
    }

    /// Calculate variance of values
    fn calculate_variance(&self, values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }

        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
        variance
    }

    /// Detect repetition in token sequence
    pub fn is_repeating(&self, tokens: &[u32]) -> bool {
        if !self.enabled || tokens.len() < self.window_size * 2 {
            return false;
        }

        // Check if recent tokens match earlier pattern
        let recent = &tokens[tokens.len() - self.window_size..];
        let earlier = &tokens[tokens.len() - self.window_size * 2..tokens.len() - self.window_size];

        recent == earlier
    }
}

/// Thought termination policy
#[derive(Debug, Clone, PartialEq)]
pub enum TerminationPolicy {
    /// Fixed number of steps
    FixedSteps(usize),

    /// Terminate when confidence exceeds threshold
    ConfidenceThreshold(f32),

    /// Terminate when improvement plateaus
    ConvergenceDetection { window: usize, threshold: f32 },

    /// Terminate on special token (e.g., `[DONE]`)
    SpecialToken(u32),

    /// Combined policies (any match triggers termination)
    Combined(Vec<TerminationPolicy>),
}

impl TerminationPolicy {
    /// Check if should terminate
    pub fn should_terminate(
        &self,
        step: usize,
        confidence: f32,
        recent_scores: &[f32],
        token: u32,
    ) -> bool {
        match self {
            TerminationPolicy::FixedSteps(max_steps) => step >= *max_steps,

            TerminationPolicy::ConfidenceThreshold(threshold) => confidence >= *threshold,

            TerminationPolicy::ConvergenceDetection { window, threshold } => {
                if recent_scores.len() < *window {
                    return false;
                }
                let recent = &recent_scores[recent_scores.len() - window..];
                let variance = Self::calculate_variance(recent);
                variance < *threshold
            }

            TerminationPolicy::SpecialToken(special) => token == *special,

            TerminationPolicy::Combined(policies) => policies
                .iter()
                .any(|p| p.should_terminate(step, confidence, recent_scores, token)),
        }
    }

    fn calculate_variance(values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32
    }
}

/// Reasoning chain tracker
#[derive(Debug, Clone)]
pub struct ReasoningChain {
    /// Chain ID
    pub id: usize,

    /// Steps in this chain
    pub steps: usize,

    /// Tokens generated
    pub tokens: Vec<u32>,

    /// Confidence scores per step
    pub confidences: Vec<f32>,

    /// Whether chain completed successfully
    pub completed: bool,

    /// Termination reason
    pub termination_reason: Option<String>,
}

impl ReasoningChain {
    /// Create new reasoning chain
    pub fn new(id: usize) -> Self {
        Self {
            id,
            steps: 0,
            tokens: Vec::new(),
            confidences: Vec::new(),
            completed: false,
            termination_reason: None,
        }
    }

    /// Add step to chain
    pub fn add_step(&mut self, token: u32, confidence: f32) {
        self.steps += 1;
        self.tokens.push(token);
        self.confidences.push(confidence);
    }

    /// Mark chain as completed
    pub fn complete(&mut self, reason: String) {
        self.completed = true;
        self.termination_reason = Some(reason);
    }

    /// Get average confidence
    pub fn average_confidence(&self) -> f32 {
        if self.confidences.is_empty() {
            return 0.0;
        }
        self.confidences.iter().sum::<f32>() / self.confidences.len() as f32
    }
}

/// Reasoning controller that manages budget and termination
pub struct ReasoningController {
    /// Budget configuration
    budget: ReasoningBudget,

    /// Overthinking detector
    overthinking_detector: OverthinkingDetector,

    /// Termination policy
    termination_policy: TerminationPolicy,

    /// Active reasoning chains
    chains: Vec<ReasoningChain>,

    /// Total tokens generated
    total_tokens: usize,

    /// Logits history for overthinking detection
    logits_history: VecDeque<Vec<f32>>,

    /// Maximum logits history size
    max_history_size: usize,
}

impl ReasoningController {
    /// Create new reasoning controller
    pub fn new(
        budget: ReasoningBudget,
        overthinking_detector: OverthinkingDetector,
        termination_policy: TerminationPolicy,
    ) -> Self {
        Self {
            budget,
            overthinking_detector,
            termination_policy,
            chains: Vec::new(),
            total_tokens: 0,
            logits_history: VecDeque::new(),
            max_history_size: 20,
        }
    }

    /// Start a new reasoning chain
    pub fn start_chain(&mut self) -> Result<usize, String> {
        if !self.budget.can_start_chain(self.chains.len()) {
            return Err(format!(
                "Budget exceeded: max chains {} reached",
                self.budget.max_chains
            ));
        }

        let chain_id = self.chains.len();
        self.chains.push(ReasoningChain::new(chain_id));
        Ok(chain_id)
    }

    /// Record a reasoning step
    pub fn record_step(
        &mut self,
        chain_id: usize,
        token: u32,
        confidence: f32,
        logits: Vec<f32>,
    ) -> Result<bool, String> {
        let chain = self
            .chains
            .get_mut(chain_id)
            .ok_or_else(|| format!("Invalid chain ID: {}", chain_id))?;

        // Check budget
        if !self.budget.can_continue_chain(chain.steps) {
            chain.complete("Budget: max steps reached".to_string());
            return Ok(true); // Should terminate
        }

        if !self.budget.can_generate_tokens(self.total_tokens, 1) {
            chain.complete("Budget: max tokens reached".to_string());
            return Ok(true);
        }

        // Add to history
        self.logits_history.push_back(logits);
        if self.logits_history.len() > self.max_history_size {
            self.logits_history.pop_front();
        }

        // Record step
        chain.add_step(token, confidence);
        self.total_tokens += 1;

        // Check overthinking
        let logits_vec: Vec<Vec<f32>> = self.logits_history.iter().cloned().collect();
        if self.overthinking_detector.is_overthinking(&logits_vec) {
            chain.complete("Overthinking detected".to_string());
            return Ok(true);
        }

        // Check repetition
        if self.overthinking_detector.is_repeating(&chain.tokens) {
            chain.complete("Repetition detected".to_string());
            return Ok(true);
        }

        // Check termination policy
        if self.termination_policy.should_terminate(
            chain.steps,
            confidence,
            &chain.confidences,
            token,
        ) {
            chain.complete("Termination policy triggered".to_string());
            return Ok(true);
        }

        Ok(false) // Continue
    }

    /// Get current chain
    pub fn get_chain(&self, chain_id: usize) -> Option<&ReasoningChain> {
        self.chains.get(chain_id)
    }

    /// Get all chains
    pub fn get_chains(&self) -> &[ReasoningChain] {
        &self.chains
    }

    /// Get statistics
    pub fn stats(&self) -> ReasoningStats {
        ReasoningStats {
            total_chains: self.chains.len(),
            completed_chains: self.chains.iter().filter(|c| c.completed).count(),
            total_tokens: self.total_tokens,
            avg_steps_per_chain: if self.chains.is_empty() {
                0.0
            } else {
                self.chains.iter().map(|c| c.steps).sum::<usize>() as f32 / self.chains.len() as f32
            },
            avg_confidence: if self.chains.is_empty() {
                0.0
            } else {
                self.chains
                    .iter()
                    .map(|c| c.average_confidence())
                    .sum::<f32>()
                    / self.chains.len() as f32
            },
        }
    }

    /// Reset controller for new request
    pub fn reset(&mut self) {
        self.chains.clear();
        self.total_tokens = 0;
        self.logits_history.clear();
    }
}

/// Reasoning statistics
#[derive(Debug, Clone)]
pub struct ReasoningStats {
    /// Total chains started
    pub total_chains: usize,

    /// Chains that completed
    pub completed_chains: usize,

    /// Total tokens generated
    pub total_tokens: usize,

    /// Average steps per chain
    pub avg_steps_per_chain: f32,

    /// Average confidence across chains
    pub avg_confidence: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reasoning_budget_defaults() {
        let budget = ReasoningBudget::default();
        assert_eq!(budget.max_chains, 3);
        assert_eq!(budget.max_steps_per_chain, 10);
        assert_eq!(budget.max_total_tokens, 500);
        assert!(budget.enabled);
    }

    #[test]
    fn test_reasoning_budget_builder() {
        let budget = ReasoningBudget::new()
            .max_chains(5)
            .max_steps_per_chain(20)
            .max_total_tokens(1000)
            .verifier_frequency(3);

        assert_eq!(budget.max_chains, 5);
        assert_eq!(budget.max_steps_per_chain, 20);
        assert_eq!(budget.max_total_tokens, 1000);
        assert_eq!(budget.verifier_frequency, 3);
    }

    #[test]
    fn test_budget_constraints() {
        let budget = ReasoningBudget::new().max_chains(2).max_steps_per_chain(5);

        assert!(budget.can_start_chain(0));
        assert!(budget.can_start_chain(1));
        assert!(!budget.can_start_chain(2));

        assert!(budget.can_continue_chain(0));
        assert!(budget.can_continue_chain(4));
        assert!(!budget.can_continue_chain(5));
    }

    #[test]
    fn test_overthinking_detector() {
        let detector = OverthinkingDetector::new()
            .window_size(3)
            .entropy_threshold(1.0) // Increased threshold
            .variance_threshold(0.001); // Lower variance threshold

        // Low entropy, low variance = overthinking
        let logits_history = vec![
            vec![10.0, 1.0, 1.0, 1.0], // Very high confidence, low entropy
            vec![10.0, 1.0, 1.0, 1.0],
            vec![10.0, 1.0, 1.0, 1.0],
        ];

        assert!(detector.is_overthinking(&logits_history));
    }

    #[test]
    fn test_repetition_detection() {
        let detector = OverthinkingDetector::new().window_size(3);

        // Repeating pattern
        let tokens = vec![1, 2, 3, 1, 2, 3];
        assert!(detector.is_repeating(&tokens));

        // Non-repeating
        let tokens2 = vec![1, 2, 3, 4, 5, 6];
        assert!(!detector.is_repeating(&tokens2));
    }

    #[test]
    fn test_termination_policy_fixed_steps() {
        let policy = TerminationPolicy::FixedSteps(10);

        assert!(!policy.should_terminate(5, 0.9, &[], 0));
        assert!(!policy.should_terminate(9, 0.9, &[], 0));
        assert!(policy.should_terminate(10, 0.9, &[], 0));
    }

    #[test]
    fn test_termination_policy_confidence() {
        let policy = TerminationPolicy::ConfidenceThreshold(0.95);

        assert!(!policy.should_terminate(5, 0.9, &[], 0));
        assert!(policy.should_terminate(5, 0.96, &[], 0));
    }

    #[test]
    fn test_termination_policy_convergence() {
        let policy = TerminationPolicy::ConvergenceDetection {
            window: 3,
            threshold: 0.0001, // Very low threshold for this test
        };

        // High variance - not converged
        let scores1 = vec![0.5, 0.7, 0.6, 0.8];
        assert!(!policy.should_terminate(4, 0.8, &scores1, 0));

        // Low variance - converged
        let scores2 = vec![0.9, 0.900001, 0.899999, 0.9];
        assert!(policy.should_terminate(4, 0.90, &scores2, 0));
    }

    #[test]
    fn test_reasoning_chain() {
        let mut chain = ReasoningChain::new(0);

        chain.add_step(10, 0.9);
        chain.add_step(20, 0.85);
        chain.add_step(30, 0.95);

        assert_eq!(chain.steps, 3);
        assert_eq!(chain.tokens.len(), 3);
        assert!((chain.average_confidence() - 0.9).abs() < 0.01);

        chain.complete("Success".to_string());
        assert!(chain.completed);
        assert_eq!(chain.termination_reason, Some("Success".to_string()));
    }

    #[test]
    fn test_reasoning_controller() {
        let budget = ReasoningBudget::new().max_chains(2).max_steps_per_chain(3);
        let detector = OverthinkingDetector::new();
        let policy = TerminationPolicy::FixedSteps(3);

        let mut controller = ReasoningController::new(budget, detector, policy);

        // Start first chain
        let chain_id = controller.start_chain().unwrap();
        assert_eq!(chain_id, 0);

        // Add steps
        let logits = vec![5.0, 4.0, 3.0, 2.0];
        assert!(
            !controller
                .record_step(chain_id, 10, 0.9, logits.clone())
                .unwrap()
        );
        assert!(
            !controller
                .record_step(chain_id, 20, 0.85, logits.clone())
                .unwrap()
        );
        assert!(controller.record_step(chain_id, 30, 0.95, logits).unwrap()); // Should terminate at step 3

        let stats = controller.stats();
        assert_eq!(stats.total_chains, 1);
        assert_eq!(stats.completed_chains, 1);
        assert_eq!(stats.total_tokens, 3);
    }

    #[test]
    fn test_controller_budget_limits() {
        let budget = ReasoningBudget::new().max_chains(1);
        let detector = OverthinkingDetector::new();
        let policy = TerminationPolicy::FixedSteps(100);

        let mut controller = ReasoningController::new(budget, detector, policy);

        // First chain succeeds
        assert!(controller.start_chain().is_ok());

        // Second chain fails (budget exceeded)
        assert!(controller.start_chain().is_err());
    }
}
