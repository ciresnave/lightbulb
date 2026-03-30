//! Cache Span Management Demonstration
//!
//! This example demonstrates the Phase 2 span tagging system for semantic
//! KV cache management. It shows:
//!
//! 1. **External Token Storage** - Tokens stored separately from cache metadata
//! 2. **Span Lifecycle** - Creating, tracking, and managing semantic regions
//! 3. **Parent-Child Dependencies** - Auto-cascade eviction
//! 4. **Importance Scoring** - Priority-based eviction
//! 5. **Tag-Based Queries** - Finding and evicting by semantic category
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │ ParallelCacheBuilder (Metadata Only)        │
//! │  - SpanRegistry (spans, names, tags)        │
//! │  - No token storage                         │
//! └─────────────────────────────────────────────┘
//!                     │
//!                     │ SpanId
//!                     ↓
//! ┌─────────────────────────────────────────────┐
//! │ External Token Storage (Caller's Choice)    │
//! │  - HashMap<SpanId, Vec<u32>>                │
//! │  - Vector DB for semantic search            │
//! │  - Disk for cold storage                    │
//! └─────────────────────────────────────────────┘
//! ```
//!
//! Run with: `cargo run --example span_management_demo`

use candlelight::core::{DType, Device, Result};
use lightbulb::cache::{CacheTag, CacheUsageInfo, EvictionImpact, ParallelCacheBuilder, SpanId};
use std::collections::HashMap;

/// External token storage (caller's responsibility)
///
/// In a real application, this could be:
/// - HashMap for in-memory storage
/// - Vector DB for semantic search (Phase 3)
/// - Disk archive for cold storage
struct TokenStorage {
    tokens: HashMap<SpanId, Vec<u32>>,
}

impl TokenStorage {
    fn new() -> Self {
        Self {
            tokens: HashMap::new(),
        }
    }

    fn store(&mut self, span_id: SpanId, tokens: Vec<u32>) {
        self.tokens.insert(span_id, tokens);
    }

    fn retrieve(&self, span_id: SpanId) -> Option<&Vec<u32>> {
        self.tokens.get(&span_id)
    }

    fn remove(&mut self, span_id: SpanId) -> Option<Vec<u32>> {
        self.tokens.remove(&span_id)
    }

    fn archive_to_vector_db(&mut self, span_id: SpanId) -> Result<()> {
        if let Some(tokens) = self.remove(span_id) {
            println!(
                "  📦 Archived {} tokens from span {} to vector DB",
                tokens.len(),
                span_id
            );
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    println!("🚀 Cache Span Management Demo\n");

    // Initialize cache builder (2 slots, 512 context)
    let device = Device::Cpu;
    let mut builder = ParallelCacheBuilder::new(2, 512, DType::F32, &device)?;
    let mut storage = TokenStorage::new();

    println!("✅ Created cache builder: 2 slots, 512 context\n");

    // === SCENARIO 1: System Prompt ===
    println!("📝 Scenario 1: System Prompt");
    println!("─────────────────────────────");

    let sys_tokens = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]; // Mock tokens
    let sys_span = builder.begin_span(0, CacheTag::SystemPrompt, Some("sys_v1".into()))?;
    storage.store(sys_span, sys_tokens);

    builder.set_position(0, 10); // Advance position
    builder.end_span(sys_span)?;
    builder.set_span_importance(sys_span, 1.0)?; // Maximum importance!

    println!("✓ Created system prompt span (ID: {})", sys_span);
    println!("  - Position: 0-10");
    println!("  - Importance: 1.0 (never evict)");
    println!("  - Tokens stored externally\n");

    // === SCENARIO 2: User Input ===
    println!("📝 Scenario 2: User Input");
    println!("─────────────────────────────");

    let user_tokens = vec![11, 12, 13, 14, 15];
    let user_span = builder.tag_region(0, 10, 15, CacheTag::UserInput, Some("user_q1".into()))?;
    storage.store(user_span, user_tokens);
    builder.set_span_importance(user_span, 0.9)?;

    println!("✓ Created user input span (ID: {})", user_span);
    println!("  - Position: 10-15");
    println!("  - Importance: 0.9\n");

    // === SCENARIO 3: Tool Execution with Auto-Generated Context ===
    println!("📝 Scenario 3: Tool Output + Generated Context");
    println!("─────────────────────────────────────────────────");

    // File content from tool
    let file_tokens = vec![16, 17, 18, 19, 20, 21, 22, 23, 24, 25];
    let file_span = builder.tag_region(
        0,
        15,
        25,
        CacheTag::ToolOutput,
        Some("file:report.pdf".into()),
    )?;
    storage.store(file_span, file_tokens);
    builder.set_span_importance(file_span, 0.8)?;

    println!("✓ Created tool output span (ID: {})", file_span);
    println!("  - Name: file:report.pdf");
    println!("  - Position: 15-25");

    // Auto-generated context explaining the file
    let ctx_tokens = vec![26, 27, 28, 29, 30];
    let ctx_span =
        builder.tag_region(0, 25, 30, CacheTag::Custom, Some("ctx:report.pdf".into()))?;
    storage.store(ctx_span, ctx_tokens);

    // Make context a child of file (dependency)
    builder.set_span_parent(ctx_span, file_span)?;

    println!("✓ Created context span (ID: {}) as child of file", ctx_span);
    println!("  - Parent-child dependency established");
    println!("  - Evicting parent will cascade to child\n");

    // === SCENARIO 4: Model Generation ===
    println!("📝 Scenario 4: Model Generation");
    println!("─────────────────────────────────");

    let gen_tokens = vec![31, 32, 33, 34, 35, 36, 37, 38, 39, 40];
    let gen_span = builder.tag_region(
        0,
        30,
        40,
        CacheTag::ModelGeneration,
        Some("answer_1".into()),
    )?;
    storage.store(gen_span, gen_tokens);
    builder.set_span_importance(gen_span, 0.5)?;

    println!("✓ Created model generation span (ID: {})", gen_span);
    println!("  - Position: 30-40");
    println!("  - Importance: 0.5 (ephemeral)\n");

    // === STATUS CHECK ===
    println!("📊 Current Cache Status");
    println!("───────────────────────");

    let info = builder.get_cache_usage();
    print_cache_usage(&info);

    // === SCENARIO 5: Cache Pressure - Evict Old Tool Outputs ===
    println!("\n⚠️  Scenario 5: Cache Pressure");
    println!("────────────────────────────────");
    println!("Need to free space - evicting file:report.pdf\n");

    let result = builder.evict_named("file:report.pdf")?;

    println!("✓ Eviction complete:");
    for (span_id, impact) in &result.spans_affected {
        match impact {
            EvictionImpact::FullyEvicted => {
                println!("  - Span {} fully evicted", span_id);

                // Archive to vector DB
                storage.archive_to_vector_db(*span_id)?;
            }
            EvictionImpact::PartiallyEvicted { remaining } => {
                println!("  - Span {} partially evicted", span_id);
                println!("    Remaining ranges: {:?}", remaining);
            }
            EvictionImpact::Unchanged => {
                println!("  - Span {} unchanged", span_id);
            }
        }
    }

    println!(
        "\n💡 Notice: Context span {} was auto-evicted (child cascade)",
        ctx_span
    );

    // === FINAL STATUS ===
    println!("\n📊 Final Cache Status");
    println!("──────────────────────");

    let info = builder.get_cache_usage();
    print_cache_usage(&info);

    println!("\n✅ Demo complete!");
    println!("\n📚 Key Takeaways:");
    println!("  1. Spans store metadata only - caller manages tokens");
    println!("  2. Parent-child dependencies enable auto-cascade eviction");
    println!("  3. Importance scores guide eviction priorities");
    println!("  4. Tags enable semantic grouping and queries");
    println!("  5. External storage can be HashMap, Vector DB, or disk");

    Ok(())
}

fn print_cache_usage(info: &CacheUsageInfo) {
    println!("  Total slots: {}", info.total_slots);
    println!("  Active spans: {}", info.active_span_count);
    println!("  Slot positions: {:?}", info.per_slot_positions);
    println!("  Spans by tag:");
    for (tag, count) in &info.per_tag_counts {
        println!("    - {:?}: {}", tag, count);
    }
}
