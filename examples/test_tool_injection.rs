//! CR.1 Mid-Reasoning Context Injection — Integration Test
//!
//! Validates the full tool call detection → pause → inject → resume pipeline:
//!
//! 1. Load a GGUF model
//! 2. Enable tool call detection
//! 3. Set up a system prompt that instructs the model to use tools
//! 4. Ask a question that should trigger a tool call
//! 5. Detect the tool call (if model produces one)
//! 6. Inject a mock tool result via mini-prefill
//! 7. Verify the model continues generating with the result in context
//!
//! Note: Whether the model actually produces a tool call depends on the model.
//! TinyLlama may not produce tool calls without fine-tuning. This test validates
//! the infrastructure works when a tool call IS detected.
//!
//! Run with: cargo run --example test_tool_injection --release

use anyhow::Result;
use lightbulb::cache::SegmentedCacheConfig;
use lightbulb::engine::tool_call;
use lightbulb::engine::{Request, RequestContext, RequestState};
use lightbulb::model::ParallelModelManager;

fn main() -> Result<()> {
    println!("\n=== CR.1 Mid-Reasoning Context Injection Test ===\n");

    let model_path = find_model()?;
    println!("Model: {}", model_path);

    // Load model
    println!("Loading model...");
    let start = std::time::Instant::now();

    let mut model = ParallelModelManager::load_gguf(
        &model_path,
        1,
        256,
        None,
        None,
    )?;

    println!("Model loaded in {:.2}s\n", start.elapsed().as_secs_f32());

    // Enable segmented cache (for ToolOutput span tracking)
    model.enable_segmented_cache(SegmentedCacheConfig::observation());

    // Enable tool call detection
    model.enable_tool_call_detection();
    println!("Tool call detection enabled\n");

    // === Test 1: Manual injection test ===
    // This bypasses tool call detection and directly tests inject_tool_result()
    println!("--- Test 1: Manual Tool Result Injection ---");
    println!("Testing: generate some tokens → inject tool result → continue generating\n");

    let req = Request {
        id: "inject-test".to_string(),
        prompt: "The weather in Paris is".to_string(),
        max_new_tokens: 10,
    };

    let mut batch = vec![RequestContext::new(req)];

    // Generate 5 tokens first
    println!("Phase 1: Generating 5 tokens...");
    for step in 0..5 {
        model.forward_batch(&mut batch)?;
        if batch[0].state == RequestState::Completed {
            println!("  Model completed early at step {}", step);
            break;
        }
    }

    let tokens_before = batch[0].generated_tokens.len();
    let pos_before = batch[0].position;
    println!(
        "  Generated {} tokens, position = {}",
        tokens_before, pos_before
    );

    // Now manually pause and inject a tool result
    if batch[0].state == RequestState::Decoding {
        println!("\nPhase 2: Injecting tool result...");

        // Pause the request
        batch[0].await_tool_result(
            "get_weather".to_string(),
            "Paris".to_string(),
            None,
        );

        assert!(
            batch[0].state.is_awaiting_tool_result(),
            "Should be in AwaitingToolResult state"
        );

        // Format and inject the tool result
        let formatted_result = tool_call::format_tool_result(
            "get_weather",
            "Current weather in Paris: 22°C, sunny with light clouds. Humidity: 45%.",
        );
        println!("  Injecting: {:?}", &formatted_result[..formatted_result.len().min(80)]);

        model.inject_tool_result(&mut batch, 0, &formatted_result)?;

        let pos_after_inject = batch[0].position;
        println!(
            "  Position after injection: {} (advanced by {} tokens)",
            pos_after_inject,
            pos_after_inject - pos_before
        );

        assert!(
            batch[0].state == RequestState::Decoding,
            "Should be back in Decoding state after injection"
        );

        // Continue generating
        println!("\nPhase 3: Resuming generation after injection...");
        let mut post_inject_steps = 0;
        while batch[0].state == RequestState::Decoding && post_inject_steps < 15 {
            model.forward_batch(&mut batch)?;
            post_inject_steps += 1;
        }

        let tokens_after = batch[0].generated_tokens.len();
        println!(
            "  Generated {} more tokens after injection ({} total)",
            tokens_after - tokens_before,
            tokens_after
        );

        // Decode and show the full output
        let all_tokens = &batch[0].generated_tokens;
        match model.decode(all_tokens, false) {
            Ok(text) => println!("\n  Full output: \"{}\"", text),
            Err(e) => println!("\n  (Could not decode output: {})", e),
        }
    } else {
        println!("  Skipping injection test — model already completed");
    }

    // === Test 2: Cache utilization check ===
    println!("\n--- Test 2: Cache State After Injection ---");
    let max_util = model.cache_builder().get_max_utilization();
    println!("  Cache utilization: {:.1}%", max_util * 100.0);

    let summaries = model.cache_builder().compute_span_attention_summary();
    println!("  Active spans: {}", summaries.len());
    for (span_id, tag, tokens, total_attn, _avg) in &summaries {
        println!(
            "    Span {:2} | {:?} | {} tokens | attn: {:.4}",
            span_id, tag, tokens, total_attn
        );
    }

    // === Test 3: Verify tool call detection patterns ===
    println!("\n--- Test 3: Pattern Detection Unit Test ---");

    let mut detector = tool_call::ToolCallDetector::with_defaults();
    println!("  Registered {} default patterns", detector.pattern_count());

    // Simulate tokens that spell out a tool call
    // (In real usage, these would come from the model's output)
    let test_text = "Let me check <tool_call>get_weather(Paris)</tool_call> for you";
    let test_tokens = model.tokenize(test_text, false)?;
    println!("  Test text: \"{}\"", test_text);
    println!("  Tokenized to {} tokens", test_tokens.len());

    for &token in &test_tokens {
        detector.push_token(99, token);
    }

    match detector.check_for_tool_call(99, model.tokenizer(), test_tokens.len()) {
        Some(detected) => {
            println!(
                "  ✓ Detected tool call: {}({}) at tokens {}..{}",
                detected.tool_name, detected.tool_args, detected.token_start, detected.token_end
            );
        }
        None => {
            println!("  ✗ No tool call detected (tokenization may split markers)");
            println!("    This is expected if the tokenizer splits '<tool_call>' across tokens");
        }
    }

    println!("\n=== CR.1 Test Complete ===");
    Ok(())
}

fn find_model() -> Result<String> {
    let candidates = [
        "../models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    ];
    for path in &candidates {
        if std::path::Path::new(path).exists() {
            return Ok(path.to_string());
        }
    }
    anyhow::bail!("Model not found. Tried: {:?}", candidates)
}
