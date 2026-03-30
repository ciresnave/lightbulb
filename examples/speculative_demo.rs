//! Speculative Decoding Demonstration
//!
//! This example demonstrates speculative decoding with mock models
//! to show the verification and acceptance mechanics.
//!
//! Run with:
//! ```bash
//! cargo run --example speculative_demo
//! ```

use anyhow::Result;
use candlelight::core::{Device, Tensor};
use lightbulb::engine::speculative::{SpeculativeConfig, SpeculativeDecoder, SpeculativeModel};

/// Mock model for demonstration purposes
struct MockModel {
    vocab_size: usize,
    device: Device,
    /// Deterministic token sequence for testing
    token_sequence: Vec<u32>,
    call_count: usize,
}

impl MockModel {
    fn new(vocab_size: usize, token_sequence: Vec<u32>) -> Self {
        Self {
            vocab_size,
            device: Device::Cpu,
            token_sequence,
            call_count: 0,
        }
    }
}

impl SpeculativeModel for MockModel {
    fn forward_logits(&mut self, _tokens: &[u32], _position: usize) -> Result<Tensor> {
        // Return logits that will sample to our predetermined sequence
        let mut logits_vec = vec![0.0f32; self.vocab_size];

        // Make the next token in our sequence have highest probability
        let next_token = self.token_sequence[self.call_count % self.token_sequence.len()];
        logits_vec[next_token as usize] = 10.0; // High logit → high probability

        self.call_count += 1;

        Ok(Tensor::new(logits_vec, &self.device)?)
    }
    fn device(&self) -> &Device {
        &self.device
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn reset_cache(&mut self) {
        self.call_count = 0;
    }
}

fn main() -> Result<()> {
    println!("🎯 Speculative Decoding Demonstration\n");
    println!("This demo shows how speculative decoding works with different");
    println!("acceptance scenarios using mock models.\n");

    // Simple greedy sampler (argmax)
    let sampler = |logits: &Tensor| -> Result<u32> {
        let logits_vec = logits.to_vec1::<f32>()?;
        let max_idx = logits_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        Ok(max_idx as u32)
    };

    // Scenario 1: Perfect Agreement (100% acceptance)
    println!("📊 Scenario 1: Perfect Agreement");
    println!("Draft and target models produce identical tokens\n");

    let config = SpeculativeConfig {
        num_speculative_tokens: 5,
        min_acceptance_rate: 0.3,
        enabled: true,
        auto_fallback: false,
    };

    let mut decoder = SpeculativeDecoder::new(config.clone());

    // Both models will generate: [1, 2, 3, 4, 5, ...]
    let mut draft = MockModel::new(100, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    let mut target = MockModel::new(100, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

    let context = vec![10u32]; // Starting token (safe value)
    let tokens = decoder.generate_tokens(&mut draft, &mut target, &context, sampler)?;

    println!("  Generated tokens: {:?}", tokens);
    println!(
        "  Acceptance rate: {:.1}%",
        decoder.stats().acceptance_rate() * 100.0
    );
    println!("  Speedup: {:.2}x\n", decoder.stats().speedup());

    // Scenario 2: Partial Agreement (60% acceptance)
    println!("📊 Scenario 2: Partial Agreement");
    println!("Draft model diverges after 3 tokens\n");

    decoder.reset();

    // Draft: [1, 2, 3, 4, 5], Target: [1, 2, 3, 90, 91]
    let mut draft = MockModel::new(100, vec![1, 2, 3, 4, 5]);
    let mut target = MockModel::new(100, vec![1, 2, 3, 90, 91]);
    let tokens = decoder.generate_tokens(&mut draft, &mut target, &context, sampler)?;

    println!("  Generated tokens: {:?}", tokens);
    println!(
        "  Accepted: {} tokens (first 3 matched, then target corrects)",
        tokens.len() - 1
    );
    println!(
        "  Acceptance rate: {:.1}%",
        decoder.stats().acceptance_rate() * 100.0
    );
    println!("  Speedup: {:.2}x\n", decoder.stats().speedup());

    // Scenario 3: Immediate Divergence (0% acceptance)
    println!("📊 Scenario 3: Immediate Divergence");
    println!("Draft and target disagree on first token\n");

    decoder.reset();

    // Draft: [1, 2, 3], Target: [99, 98, 97]
    let mut draft = MockModel::new(100, vec![1, 2, 3, 4, 5]);
    let mut target = MockModel::new(100, vec![99, 98, 97, 96, 95]);

    let tokens = decoder.generate_tokens(&mut draft, &mut target, &context, sampler)?;

    println!("  Generated tokens: {:?}", tokens);
    println!("  Accepted: 0 speculative tokens (immediate rejection)");
    println!("  Returned: 1 token from target model");
    println!(
        "  Acceptance rate: {:.1}%",
        decoder.stats().acceptance_rate() * 100.0
    );
    println!(
        "  Speedup: {:.2}x (overhead from draft model)\n",
        decoder.stats().speedup()
    );

    // Scenario 4: Multiple Rounds with Auto-Fallback
    println!("📊 Scenario 4: Auto-Fallback Mechanism");
    println!("Testing fallback after poor acceptance rate\n");

    let config = SpeculativeConfig {
        num_speculative_tokens: 5,
        min_acceptance_rate: 0.3,
        enabled: true,
        auto_fallback: true, // Enable auto-fallback
    };

    let mut decoder = SpeculativeDecoder::new(config);

    // Run 15 rounds with poor acceptance (always disagree)
    for round in 1..=15 {
        let mut draft = MockModel::new(100, vec![1, 2, 3]);
        let mut target = MockModel::new(100, vec![99, 98, 97]);

        let _tokens = decoder.generate_tokens(&mut draft, &mut target, &context, sampler)?;

        if round == 10 || round == 15 {
            println!(
                "  Round {}: Acceptance rate = {:.1}%, Speculating = {}",
                round,
                decoder.stats().acceptance_rate() * 100.0,
                decoder.should_speculate()
            );
        }
    }

    println!("\n  After 15 rounds with 0% acceptance:");
    println!("  Auto-fallback activated: {}", !decoder.should_speculate());
    println!("  System will now use standard decoding (no speculation overhead)\n");

    // Summary
    println!("✅ Demonstration Complete!\n");
    println!("Key Takeaways:");
    println!("  • Speculative decoding accelerates generation when draft matches target");
    println!("  • Longest matching prefix is always accepted");
    println!("  • Target model provides correction on first mismatch");
    println!("  • Auto-fallback prevents overhead when acceptance is low");
    println!("  • Real speedup depends on draft model quality and hardware\n");

    Ok(())
}
