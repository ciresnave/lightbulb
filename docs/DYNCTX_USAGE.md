# DynCtx Usage Guide

## Overview

`dynctx` (Dynamic Context) is a production-ready arena-based memory management system originally developed for DynAniML and integrated into Lightbulb. It provides high-performance token management, relationship tracking, and persistence capabilities optimized for ML workloads.

## Table of Contents

- [Core Concepts](#core-concepts)
- [SlotArena API](#slotarena-api)
- [Rope Operations](#rope-operations)
- [Relationship System](#relationship-system)
- [Snapshot and Persistence](#snapshot-and-persistence)
- [Audit Logging](#audit-logging)
- [Best Practices](#best-practices)
- [Performance Tips](#performance-tips)

---

## Core Concepts

### Arena-Based Memory Management

Instead of allocating tokens individually on the heap, `dynctx` uses an arena allocator that pre-allocates a large block of memory and manages it efficiently:

**Benefits:**

- O(1) allocation and deallocation
- Better cache locality
- Reduced memory fragmentation
- Predictable performance
- Easy bulk cleanup

### Hot/Cold Data Separation

The arena separates frequently-accessed data (token IDs, positions) from rarely-accessed data (prev/next pointers) to improve cache performance:

```rust
pub struct SlotArena {
    hot_data: Vec<TokenNode>,  // Frequently accessed: token_id + rel_pos
    cold_data: Vec<LinkNode>,  // Rarely accessed: prev + next pointers
    free: Vec<u32>,            // Free slot tracking
    head: Option<NodeKey>,
    tail: Option<NodeKey>,
}
```

### Trust-the-Caller Optimization

DynCtx uses `Vec<TokenNode>` instead of `Vec<Option<TokenNode>>`, saving 8 bytes per slot and eliminating Option unwrapping overhead. Debug builds validate all operations; release builds trust the caller for maximum performance.

---

## SlotArena API

### Creating an Arena

```rust
use dynctx::{SlotArena, TokenNode, NodeKey};

// Create new arena with default capacity (32,768 slots)
let mut arena = SlotArena::new();
```

### Adding Tokens

```rust
// Append token to end of sequence
let key1 = arena.append(TokenNode::new(token_id, relative_pos))?;

// Insert token after specific position
let key2 = arena.insert_after(key1, TokenNode::new(next_id, next_pos))?;

// Insert token before specific position
let key3 = arena.insert_before(key2, TokenNode::new(prev_id, prev_pos))?;
```

### Accessing Tokens

```rust
// Get token by key
if let Some(token) = arena.get(key1) {
    println!("Token ID: {}, Position: {}", token.tok_id, token.rel_pos);
}

// Iterate through all tokens in sequence
for key in arena.iter_keys() {
    if let Some(token) = arena.get(key) {
        // Process token...
    }
}
```

### Removing Tokens

```rust
// Remove single token
arena.remove(key1)?;

// Remove range of tokens
arena.remove_range(start_key, end_key)?;

// Clear entire arena
arena.clear();
```

### Query Operations

```rust
// Get arena statistics
let stats = arena.stats();
println!("Used slots: {}, Free slots: {}", stats.used, stats.free);

// Check if key exists
if arena.contains(key) {
    // Key is valid
}

// Get head and tail
if let Some(head_key) = arena.head() {
    // First token
}

if let Some(tail_key) = arena.tail() {
    // Last token
}
```

### Complete Example

```rust
use dynctx::{SlotArena, TokenNode, NodeKey, ArenaError};

fn example_token_sequence() -> Result<(), ArenaError> {
    let mut arena = SlotArena::new();
    
    // Build token sequence: "The quick brown fox"
    let tokens = vec![
        (1, "The"),
        (2, "quick"),
        (3, "brown"),
        (4, "fox"),
    ];
    
    let mut keys = Vec::new();
    for (idx, (token_id, _text)) in tokens.iter().enumerate() {
        let key = arena.append(TokenNode::new(*token_id, idx as u32))?;
        keys.push(key);
    }
    
    // Access tokens
    for key in &keys {
        if let Some(token) = arena.get(*key) {
            println!("Token: ID={}, Pos={}", token.tok_id, token.rel_pos);
        }
    }
    
    // Remove middle token ("quick")
    arena.remove(keys[1])?;
    
    println!("After removal: {} tokens", arena.len());
    
    Ok(())
}
```

---

## Rope Operations

Rope operations provide efficient position tracking and dense view construction for processing.

### Building Dense Views

```rust
use dynctx::rope;

// Build dense view for efficient sequential access
let dense_view = rope::build_dense_view(&arena)?;

// Access token data through dense arrays
for idx in 0..dense_view.ids.len() {
    let token_id = dense_view.ids[idx];
    let abs_pos = dense_view.abs_pos[idx];
    let node_key = dense_view.mapping[idx];
    
    println!("Token {} at position {} (key: {:?})", token_id, abs_pos, node_key);
}
```

### Position Calculations

```rust
use dynctx::rope;

// Calculate cumulative positions from relative positions
let relative_positions = vec![0, 1, 2, 3];
let absolute_positions = rope::cumulative_positions(0, &relative_positions)?;

// Result: [0, 1, 3, 6]
```

### Performance Characteristics

- **Dense view build**: <1ms for 5,000 tokens (release), <10ms (debug)
- **Memory usage**: 3 × sizeof(u32) × num_tokens
- **Use case**: Batch processing, serialization, visualization

---

## Relationship System

The 7-layer relationship system enables rich semantic modeling between tokens.

### Relationship Types

1. **Structural**: Syntactic structure (dependencies, constituents, etc.)
2. **Linguistic**: Linguistic properties (agreement, government, etc.)
3. **Discourse**: Discourse relationships (coreference, topic, etc.)
4. **Temporal**: Temporal ordering and relationships
5. **Pragmatic**: Pragmatic and social context
6. **Knowledge**: Knowledge graph relationships
7. **Cognitive**: Cognitive and reasoning relationships

### Using Relationships

```rust
use dynctx::{
    RelationshipManager,
    RelationshipType,
    relationship_types::{
        StructuralType, StructuralMetadata,
        LinguisticType, LinguisticFeatures,
        DiscourseType, DiscourseProperties,
    }
};

// Create relationship manager
let mut mgr = RelationshipManager::new();

// Add structural dependency
let structural = RelationshipType::Structural {
    kind: StructuralType::Dependency,
    metadata: Some(StructuralMetadata {
        direction: DependencyDirection::HeadToDependent,
        strength: 1.0,
    }),
};
mgr.add_relationship(head_key, dependent_key, structural);

// Add linguistic agreement
let linguistic = RelationshipType::Linguistic {
    kind: LinguisticType::Agreement,
    features: Some(LinguisticFeatures {
        person: Some(3),
        number: Some(Number::Singular),
        gender: Some(Gender::Masculine),
    }),
};
mgr.add_relationship(subj_key, verb_key, linguistic);

// Add discourse coreference
let discourse = RelationshipType::Discourse {
    kind: DiscourseType::Coreference,
    properties: Some(DiscourseProperties {
        salience: 0.9,
        topic_continuity: true,
    }),
};
mgr.add_relationship(pronoun_key, antecedent_key, discourse);
```

### Querying Relationships

```rust
// Get all relationships from a source
let outgoing = mgr.get_relationships_from(source_key);

// Get all relationships to a target
let incoming = mgr.get_relationships_to(target_key);

// Find specific relationship type
for rel in mgr.get_relationships_from(key) {
    match rel.rel_type {
        RelationshipType::Structural { kind, .. } => {
            println!("Structural relationship: {:?}", kind);
        }
        RelationshipType::Linguistic { kind, .. } => {
            println!("Linguistic relationship: {:?}", kind);
        }
        _ => {}
    }
}
```

### Example: Dependency Parse

```rust
fn build_dependency_tree(arena: &SlotArena, sentence: &[(u32, &str, &str)]) -> RelationshipManager {
    let mut mgr = RelationshipManager::new();
    
    // sentence: [(token_id, word, head_relation)]
    // Example: [(1, "The", "det"), (2, "cat", "nsubj"), (3, "sat", "ROOT")]
    
    for (i, (token_id, _word, dep_type)) in sentence.iter().enumerate() {
        if *dep_type != "ROOT" {
            let child_key = arena.get_key_by_position(i as u32).unwrap();
            // Find head (simplified: assume previous token)
            if i > 0 {
                let head_key = arena.get_key_by_position((i - 1) as u32).unwrap();
                
                let rel = RelationshipType::Structural {
                    kind: StructuralType::from_dep_label(dep_type),
                    metadata: Some(StructuralMetadata {
                        direction: DependencyDirection::HeadToDependent,
                        strength: 1.0,
                    }),
                };
                
                mgr.add_relationship(head_key, child_key, rel);
            }
        }
    }
    
    mgr
}
```

---

## Snapshot and Persistence

### Creating Snapshots

```rust
use dynctx::snapshot;
use std::path::Path;

// Create snapshot (fast: direct memory dump)
let snapshot = snapshot::create_snapshot(&arena)?;

// Write to file
snapshot::write_to_file(&snapshot, Path::new("arena.snapshot"))?;
```

### Loading Snapshots

```rust
use dynctx::snapshot;

// Load snapshot (memory-mapped, very fast)
let snapshot = snapshot::load_from_file(Path::new("arena.snapshot"))?;

// Restore arena
let restored_arena = snapshot::restore_arena(&snapshot)?;
```

### Snapshot Format

- **Fixed size**: 1MB per snapshot (configurable)
- **Direct memory copy**: No serialization overhead
- **Memory-mapped loading**: Instant loading via mmap
- **Metadata rebuild**: Links and indices reconstructed on load

### Example: Checkpoint System

```rust
use dynctx::snapshot;
use std::path::PathBuf;

struct CheckpointManager {
    checkpoint_dir: PathBuf,
    checkpoint_count: u64,
}

impl CheckpointManager {
    fn save_checkpoint(&mut self, arena: &SlotArena) -> Result<(), ArenaError> {
        let path = self.checkpoint_dir.join(format!("checkpoint_{}.snapshot", self.checkpoint_count));
        let snapshot = snapshot::create_snapshot(arena)?;
        snapshot::write_to_file(&snapshot, &path)?;
        self.checkpoint_count += 1;
        Ok(())
    }
    
    fn restore_latest(&self) -> Result<SlotArena, ArenaError> {
        if self.checkpoint_count == 0 {
            return Err(ArenaError::NoCheckpoints);
        }
        
        let path = self.checkpoint_dir.join(format!("checkpoint_{}.snapshot", self.checkpoint_count - 1));
        let snapshot = snapshot::load_from_file(&path)?;
        snapshot::restore_arena(&snapshot)
    }
}
```

---

## Audit Logging

### Writing Audit Logs

```rust
use dynctx::log::{LogWriter, LogEntry, OpType};

// Create log writer
let mut writer = LogWriter::new("audit.log")?;

// Log operations
writer.write_entry(LogEntry {
    seq: 1,
    op: OpType::InsertAfter as u16,
    payload: serialize_insert_operation(&insert_data)?,
})?;

writer.write_entry(LogEntry {
    seq: 2,
    op: OpType::DropRange as u16,
    payload: serialize_drop_operation(&drop_data)?,
})?;

// Flush to disk
writer.flush()?;
```

### Reading Audit Logs

```rust
use dynctx::log::{LogReader, LogEntry};

// Open log for reading
let reader = LogReader::open("audit.log")?;

// Iterate through entries
for entry in reader.iter_entries() {
    match OpType::from_u16(entry.op) {
        Some(OpType::InsertAfter) => {
            let data = deserialize_insert_operation(&entry.payload)?;
            println!("Insert at seq {}: {:?}", entry.seq, data);
        }
        Some(OpType::DropRange) => {
            let data = deserialize_drop_operation(&entry.payload)?;
            println!("Drop at seq {}: {:?}", entry.seq, data);
        }
        _ => {}
    }
}
```

### Tamper Detection

```rust
use dynctx::log::LogReader;

// Verify log integrity
let reader = LogReader::open("audit.log")?;

match reader.verify_integrity() {
    Ok(()) => println!("Log integrity verified"),
    Err(e) => println!("Log tampering detected: {}", e),
}
```

### Log Replay

```rust
use dynctx::log::LogReader;

// Replay log to rebuild arena
let reader = LogReader::open("audit.log")?;
let rebuilt_arena = reader.replay_log_to_arena()?;
```

---

## Best Practices

### Memory Management

**DO:**

- Use arena for related tokens that have similar lifetimes
- Clear arena in bulk when done with token set
- Use dense views for batch processing

**DON'T:**

- Mix short-lived and long-lived tokens in same arena
- Keep arena alive longer than necessary
- Create too many small arenas (overhead)

### Performance Optimization

**DO:**

- Build dense views once, reuse for multiple operations
- Use batch operations (insert_many, remove_range)
- Leverage hot/cold separation (avoid modifying links)
- Profile with benchmarks before optimizing

**DON'T:**

- Build dense views repeatedly
- Mix random access with sequential processing
- Ignore capacity planning (see MAX_SLOTS)

### Error Handling

**DO:**

- Handle `ArenaError` with proper error propagation
- Use `?` operator for cleaner error handling
- Log errors for debugging
- Implement graceful degradation

**DON'T:**

- Unwrap arena operations without checking
- Ignore capacity errors
- Continue after corruption detection

### Relationship Modeling

**DO:**

- Use appropriate relationship types for semantics
- Add metadata for rich relationships
- Query relationships efficiently
- Document relationship semantics

**DON'T:**

- Overuse relationships (performance cost)
- Create circular dependencies without care
- Ignore relationship strength/confidence

---

## Performance Tips

### Benchmarking Results

From DynAniML benchmarks (5,000 tokens):

```
| Operation            | Release | Debug  |
| -------------------- | ------- | ------ |
| Arena append         | 50ns    | 200ns  |
| Dense view build     | <1ms    | <10ms  |
| Snapshot write       | ~100μs  | ~150μs |
| Snapshot load (mmap) | ~10μs   | ~15μs  |
```

### Optimization Checklist

1. **Use release builds** for performance testing
2. **Profile first** before optimizing
3. **Batch operations** when possible
4. **Reuse dense views** for multiple passes
5. **Consider capacity** and pre-allocate if known
6. **Use snapshots** for fast persistence
7. **Minimize relationship queries** in hot paths

### Memory Usage

```rust
// Memory per token (approximate)
// - Hot data: 8 bytes (token_id: u32, rel_pos: u32)
// - Cold data: 8 bytes (prev: u32, next: u32)
// - Total: 16 bytes per token
// - Arena overhead: ~100 bytes
//
// For 10,000 tokens: ~160KB
// For 100,000 tokens: ~1.6MB
// For 1,000,000 tokens: ~16MB
```

### Capacity Planning

```rust
// Check current capacity
const CAPACITY: usize = dynctx::constants::MAX_SLOTS;

// Estimate memory usage
fn estimate_memory(num_tokens: usize) -> usize {
    num_tokens * 16 + 100  // bytes
}

// Plan for growth
if estimated_tokens > CAPACITY * 0.9 {
    // Consider increasing MAX_SLOTS or using multiple arenas
}
```

---

## Error Handling Guide

### Common Errors

```rust
use dynctx::ArenaError;

match operation_result {
    Err(ArenaError::SlotNotFound(key)) => {
        // Handle invalid key
    }
    Err(ArenaError::ArenaFull) => {
        // Handle capacity exceeded
    }
    Err(ArenaError::InvalidOperation(msg)) => {
        // Handle invalid operation
    }
    Err(ArenaError::CorruptedSnapshot) => {
        // Handle snapshot corruption
    }
    Ok(result) => {
        // Success
    }
}
```

### Error Recovery

```rust
fn robust_token_processing(arena: &mut SlotArena, tokens: &[Token]) -> Result<(), ArenaError> {
    for token in tokens {
        match arena.append(TokenNode::new(token.id, token.pos)) {
            Ok(key) => {
                // Success
            }
            Err(ArenaError::ArenaFull) => {
                // Recovery: create new arena or flush current
                flush_arena(arena)?;
            }
            Err(e) => {
                // Other errors: log and continue or fail
                log::error!("Token processing error: {:?}", e);
                return Err(e);
            }
        }
    }
    Ok(())
}
```

---

## Integration Examples

### Example 1: Token Stream Processing

```rust
use dynctx::{SlotArena, TokenNode, rope};

fn process_token_stream(tokens: Vec<u32>) -> Result<Vec<u32>, ArenaError> {
    let mut arena = SlotArena::new();
    
    // Load tokens
    for (idx, &token_id) in tokens.iter().enumerate() {
        arena.append(TokenNode::new(token_id, idx as u32))?;
    }
    
    // Build dense view for efficient processing
    let dense = rope::build_dense_view(&arena)?;
    
    // Process tokens (example: filter even IDs)
    let filtered: Vec<u32> = dense.ids.iter()
        .filter(|&&id| id % 2 == 0)
        .copied()
        .collect();
    
    Ok(filtered)
}
```

### Example 2: Attention Mask Generation

```rust
use dynctx::{SlotArena, RelationshipManager, RelationshipType};

fn generate_attention_mask(
    arena: &SlotArena,
    relationships: &RelationshipManager,
) -> Vec<Vec<f32>> {
    let size = arena.len();
    let mut mask = vec![vec![0.0; size]; size];
    
    // Self-attention
    for i in 0..size {
        mask[i][i] = 1.0;
    }
    
    // Relationship-based attention
    for (source_idx, source_key) in arena.iter_keys().enumerate() {
        for rel in relationships.get_relationships_from(source_key) {
            if let Some(target_idx) = arena.get_position(rel.target) {
                // Weight by relationship strength
                let weight = match &rel.rel_type {
                    RelationshipType::Structural { metadata, .. } => {
                        metadata.as_ref().map(|m| m.strength).unwrap_or(0.5)
                    }
                    _ => 0.5,
                };
                mask[source_idx][target_idx as usize] = weight;
            }
        }
    }
    
    mask
}
```

### Example 3: Cached Token Embeddings

```rust
use dynctx::{SlotArena, TokenNode};
use std::collections::HashMap;

struct CachedEmbeddings {
    arena: SlotArena,
    embeddings: HashMap<NodeKey, Vec<f32>>,
}

impl CachedEmbeddings {
    fn new() -> Self {
        CachedEmbeddings {
            arena: SlotArena::new(),
            embeddings: HashMap::new(),
        }
    }
    
    fn add_token(&mut self, token_id: u32, embedding: Vec<f32>) -> Result<NodeKey, ArenaError> {
        let key = self.arena.append(TokenNode::new(token_id, 0))?;
        self.embeddings.insert(key, embedding);
        Ok(key)
    }
    
    fn get_embedding(&self, key: NodeKey) -> Option<&Vec<f32>> {
        self.embeddings.get(&key)
    }
    
    fn clear_cache(&mut self) {
        self.arena.clear();
        self.embeddings.clear();
    }
}
```

---

## Further Reading

- **DynCtx Source Code**: `crates/dynctx/src/` - Comprehensive implementation
- **Integration Guide**: `docs/DYNANIML_INTEGRATION.md` - High-level integration overview
- **Code Analysis**: `docs/DYNANIML_CODE_ANALYSIS.md` - Detailed analysis of DynAniML components
- **Lightbulb Roadmap**: `ROADMAP.md` - How dynctx fits into Lightbulb development

---

## Support and Contributing

### Getting Help

1. Review inline documentation in source files
2. Check test files for usage examples: `crates/dynctx/src/tests/`
3. Run benchmarks to understand performance: `cargo bench -p dynctx`
4. Consult integration documentation

### Reporting Issues

When reporting issues with dynctx:

1. Include arena size and operation sequence
2. Provide minimal reproduction case
3. Note release vs debug build differences
4. Include error messages and stack traces

### Contributing Improvements

Potential improvement areas:

- Configurable MAX_SLOTS via features
- Additional relationship types
- Performance optimizations
- Enhanced error messages
- More usage examples
