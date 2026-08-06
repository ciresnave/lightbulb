//! KV Cache Insertion Demo - RAG Context Injection
//!
//! This example demonstrates mid-conversation context insertion for RAG use cases.
//! When a user asks "as we discussed earlier...", the system can inject the relevant
//! prior conversation context without re-processing the entire conversation history.
//!
//! ## Use Cases
//!
//! 1. **RAG Retrieval Injection**: Insert retrieved documents mid-conversation
//! 2. **Tool Output Insertion**: Inject function call results into context
//! 3. **Context Restoration**: Bring back earlier conversation snippets
//!
//! ## Performance Target
//!
//! < 20% overhead vs full re-prefill by using KV-only computation for evicted content

use lightbulb::cache::parallel_cache_builder::ParallelCacheBuilder;
use lightbulb::cache::cache_span::CacheTag;

/// Simulates a RAG scenario: mid-conversation context injection
fn main() {
    println!("=== KV Cache Insertion Demo ===\n");
    
    // Scenario: Long conversation where user references earlier topic
    let scenario = r#"
    Conversation History:
    User: "Tell me about climate change impacts on agriculture"
    Assistant: [Long response with citations - 500 tokens]
    User: "What about renewable energy?"
    Assistant: [Response about renewables - 300 tokens]
    User: "Going back to agriculture, how does that relate to what we discussed?"
    
    Problem: Need to inject agriculture context without re-processing 800 tokens
    "#;
    
    println!("{}", scenario);
    println!("\n--- Setup ---");
    
    // Create cache builder (typical config)
    let batch_size = 1;
    let context_length = 8192;
    let mut cache_builder = ParallelCacheBuilder::new(
        batch_size,
        context_length,
        candlelight::core::DType::F16,
        &candlelight::core::Device::Cpu,
    ).unwrap();
    
    println!("Cache: {} slots × {} positions", batch_size, context_length);
    
    // Simulate conversation state
    let slot_id = 0;
    let conversation_tokens = simulate_conversation(&mut cache_builder);
    println!("Conversation cached: {} tokens", conversation_tokens.len());
    
    // Simulate that we've actually processed these tokens (advance position)
    cache_builder.set_position(slot_id, conversation_tokens.len());
    
    // User asks about agriculture - need to inject earlier agriculture discussion
    println!("\n--- RAG Retrieval Triggered ---");
    let mut agriculture_context = vec![
        // Mock token IDs for: "Earlier we discussed: [agriculture_section]"
        100, 200, 300, 400, 500, 600, 700, 800, // "Earlier we discussed:"
        // ... agriculture content retrieved from vector DB (~100 tokens)
    ];
    for i in 0..100 {
        agriculture_context.push(1000 + i);
    }
    println!("Retrieved agriculture context: {} tokens", agriculture_context.len());
    
    // Insert at current position (after renewable energy discussion)
    let slot_id = 0;
    
    // We want to insert BEFORE the current end, to simulate injecting into mid-conversation
    // Let's insert after the agriculture discussion ended (~508 tokens)
    let insertion_pos = 508; 
    
    println!("\n--- Performing Insertion ---");
    println!("Total cached tokens: {}", cache_builder.get_position(slot_id));
    println!("Inserting {} tokens at position {} (after agriculture, before renewables)", 
             agriculture_context.len(), insertion_pos);
    
    // Call insert_context_at API
    match cache_builder.insert_context_at(slot_id, insertion_pos) {
        Ok(eviction_result) => {
            println!("✓ Eviction successful");
            println!("  Spans affected: {}", eviction_result.spans_affected.len());
            println!("  Positions evicted: {} ranges", eviction_result.positions_evicted.len());
            
            // Reconstruct sequence for re-processing
            let (full_seq, reprocess_start) = ParallelCacheBuilder::reconstruct_after_insertion(
                &conversation_tokens,
                insertion_pos,
                &agriculture_context,
            );
            
            println!("\n--- Re-processing Plan ---");
            println!("Full sequence length: {}", full_seq.len());
            println!("Cached prefix: {} tokens (skip)", reprocess_start);
            println!("Need to process: {} tokens (KV-only)", full_seq.len() - reprocess_start);
            
            let overhead_pct = ((full_seq.len() - reprocess_start) as f32 / full_seq.len() as f32) * 100.0;
            println!("\nOverhead: {:.1}% of full prefill", overhead_pct);
            
            if overhead_pct < 20.0 {
                println!("✓ Within <20% target!");
            } else {
                println!("⚠ Exceeds 20% target");
                println!("  Note: Overhead depends on insertion position.");
                println!("  Early insertions = less to reprocess, lower overhead.");
                println!("  This demo inserts mid-conversation, so more eviction needed.");
                println!("  For RAG at conversation end: overhead approaches ~10-15%");
            }
        }
        Err(e) => {
            eprintln!("✗ Insertion failed: {}", e);
        }
    }
    
    println!("\n--- Complete Workflow ---");
    print_workflow_summary();
}

/// Simulates a conversation with multiple turns
fn simulate_conversation(cache_builder: &mut ParallelCacheBuilder) -> Vec<u32> {
    let mut all_tokens = Vec::new();
    
    // User: "Tell me about climate change impacts on agriculture"
    let user_q1 = vec![1, 2, 3, 4, 5, 6, 7, 8]; // Mock tokens
    all_tokens.extend_from_slice(&user_q1);
    
    // Register span for first question
    let _ = cache_builder.tag_region(
        0,
        0,
        all_tokens.len(),
        CacheTag::UserInput,
        Some("agriculture_question".to_string()),
    );
    
    // Assistant: Long response about agriculture (500 tokens)
    let agriculture_response: Vec<u32> = (100..600).collect();
    all_tokens.extend_from_slice(&agriculture_response);
    
    let _ = cache_builder.tag_region(
        0,
        user_q1.len(),
        all_tokens.len(),
        CacheTag::ToolOutput, // Use existing tag for important facts
        Some("agriculture_response".to_string()),
    );
    
    // User: "What about renewable energy?"
    let user_q2 = vec![10, 11, 12, 13, 14];
    all_tokens.extend_from_slice(&user_q2);
    
    // Assistant: Response about renewables (300 tokens)
    let renewable_response: Vec<u32> = (600..900).collect();
    all_tokens.extend_from_slice(&renewable_response);
    
    all_tokens
}

fn print_workflow_summary() {
    println!(r#"
Complete KV Insertion Workflow:

1. **Detection Phase**
   User: "Going back to agriculture..."
   → Trigger RAG retrieval for "agriculture" topic
   
2. **Retrieval Phase**
   → Vector DB query: "agriculture discussion"
   → Returns earlier conversation snippet (100-500 tokens)
   
3. **Insertion Phase**
   cache_builder.insert_context_at(slot, current_pos)
   → Evicts everything after insertion point
   → Returns EvictionResult with affected spans
   
4. **Reconstruction Phase**
   (seq, start) = reconstruct_after_insertion(cached, pos, retrieved)
   → Builds: [cached_prefix] + [retrieved] + [evicted_suffix]
   
5. **Re-processing Phase**
   model.forward_kv_only(&seq[start..])
   → Compute only K/V for inserted + evicted content
   → Skip sampling/logits (not generating new tokens)
   → Target: <20% overhead vs full prefill
   
6. **Continue Generation**
   model.forward(&user_continuation)
   → Cache now has full context including injected retrieval
   → Generate response with proper context awareness

Benefits:
- No full conversation re-prefill (saves ~80% compute)
- Maintains causal structure (correct attention masks)
- Preserves span metadata for future evictions
- Enables long-context RAG without quadratic costs
"#);
}
