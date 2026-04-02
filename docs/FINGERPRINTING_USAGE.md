# Fingerprinting System Usage Guide

## Overview

The `infra-fingerprinting` crate provides a multi-level fingerprinting system for content deduplication, similarity matching, and integrity verification. It supports four fingerprinting levels: Atomic, Relational, Structural, and Semantic.

## Table of Contents

- [Core Concepts](#core-concepts)
- [Fingerprinting Levels](#fingerprinting-levels)
- [Deduplication](#deduplication)
- [Similarity Matching](#similarity-matching)
- [Integration with Lightbulb](#integration-with-lightbulb)
- [Performance](#performance)
- [Best Practices](#best-practices)

---

## Core Concepts

### What is Fingerprinting?

Fingerprinting creates compact hash representations of content that enable:

- **Fast equality checking**: Compare fingerprints instead of full content
- **Deduplication**: Identify duplicate content efficiently
- **Similarity detection**: Find similar content using proximity measures
- **Integrity verification**: Detect content changes or corruption
- **Cache keys**: Use fingerprints as cache lookup keys

### Fingerprint Structure

```rust
pub struct Fingerprint {
    hash: [u8; 32],      // 256-bit hash
    level: FingerprintLevel,
    metadata: Option<FingerprintMetadata>,
}

pub enum FingerprintLevel {
    Atomic,      // Token-level
    Relational,  // Relationship patterns
    Structural,  // Hierarchical structure
    Semantic,    // Content meaning
}
```

---

## Fingerprinting Levels

### Level 1: Atomic Fingerprinting

**Purpose:** Hash individual tokens or small content units

**Use Cases:**
- Token deduplication
- Exact match detection
- Fast cache lookups
- Token-level integrity

**Example:**

```rust
use infra_fingerprinting::{AtomicFingerprinter, Fingerprint};

let fingerprinter = AtomicFingerprinter::new();

// Fingerprint single token
let fp1 = fingerprinter.fingerprint_token(42);

// Fingerprint token sequence
let tokens = vec![1, 2, 3, 4, 5];
let fp2 = fingerprinter.fingerprint_sequence(&tokens);

// Compare fingerprints
if fp1 == fp2 {
    println!("Identical tokens!");
}
```

**Algorithm:** BLAKE3 (fastest cryptographic hash)

**Performance:** ~100ns per token

### Level 2: Relational Fingerprinting

**Purpose:** Capture relationship patterns between elements

**Use Cases:**
- Dependency pattern matching
- Relationship-aware deduplication
- Graph structure comparison
- Attention pattern caching

**Example:**

```rust
use infra_fingerprinting::RelationalFingerprinter;
use dynctx::{RelationshipManager, RelationshipType};

let fingerprinter = RelationalFingerprinter::new();

// Fingerprint relationship graph
let relationships: Vec<(u32, u32, RelationshipType)> = vec![
    (1, 2, RelationshipType::Structural { /* ... */ }),
    (2, 3, RelationshipType::Linguistic { /* ... */ }),
];

let fp = fingerprinter.fingerprint_relationships(&relationships);
```

**Algorithm:** Relationship-aware hashing with type encoding

**Performance:** ~1μs per relationship set

### Level 3: Structural Fingerprinting

**Purpose:** Hash hierarchical structures and layouts

**Use Cases:**
- Document structure comparison
- Parse tree deduplication
- AST fingerprinting
- Template matching

**Example:**

```rust
use infra_fingerprinting::StructuralFingerprinter;

let fingerprinter = StructuralFingerprinter::new();

// Define hierarchical structure
struct Document {
    sections: Vec<Section>,
}

struct Section {
    title: String,
    paragraphs: Vec<String>,
}

// Fingerprint structure
let fp = fingerprinter.fingerprint_structure(&document)?;
```

**Algorithm:** Merkle tree hashing with structural encoding

**Performance:** ~10μs per structure

### Level 4: Semantic Fingerprinting

**Purpose:** Content-based similarity hashing

**Use Cases:**
- Semantic deduplication
- Near-duplicate detection
- Content similarity search
- Fuzzy matching

**Example:**

```rust
use infra_fingerprinting::SemanticFingerprinter;

let fingerprinter = SemanticFingerprinter::new();

// Fingerprint content for similarity
let content1 = "The quick brown fox jumps over the lazy dog";
let content2 = "A quick brown fox leaps over a lazy dog";

let fp1 = fingerprinter.fingerprint(content1);
let fp2 = fingerprinter.fingerprint(content2);

// Compute similarity
let similarity = SemanticFingerprinter::similarity(&fp1, &fp2);
println!("Similarity: {:.2}%", similarity * 100.0);
```

**Algorithm:** Locality-sensitive hashing (LSH) with shingling

**Performance:** ~1ms per document

---

## Deduplication

### Basic Deduplication

```rust
use infra_fingerprinting::{DeduplicationEngine, FingerprintLevel};

let mut dedup = DeduplicationEngine::new(FingerprintLevel::Atomic);

let items = vec![
    "item1", "item2", "item1", "item3", "item2", "item4"
];

let unique_items: Vec<&str> = items.into_iter()
    .filter(|item| dedup.is_unique(item))
    .collect();

// Result: ["item1", "item2", "item3", "item4"]
```

### Multi-Level Deduplication

```rust
use infra_fingerprinting::{MultiLevelDeduplication, FingerprintLevel};

let mut dedup = MultiLevelDeduplication::new(vec![
    FingerprintLevel::Atomic,
    FingerprintLevel::Structural,
    FingerprintLevel::Semantic,
]);

// Check uniqueness at multiple levels
let is_unique = dedup.check_all_levels(&content);

if is_unique {
    // Truly unique content
    process_content(&content);
}
```

### Deduplication with Metadata

```rust
use infra_fingerprinting::{DeduplicationEngine, DuplicateInfo};

let mut dedup = DeduplicationEngine::with_metadata();

for (idx, content) in contents.iter().enumerate() {
    match dedup.check_duplicate(content) {
        None => {
            // Unique content
            println!("New content at index {}", idx);
        }
        Some(DuplicateInfo { original_index, count }) => {
            // Duplicate found
            println!("Duplicate of index {} (seen {} times)", original_index, count);
        }
    }
}
```

---

## Similarity Matching

### Exact Similarity

```rust
use infra_fingerprinting::{SimilarityMatcher, AtomicFingerprinter};

let fingerprinter = AtomicFingerprinter::new();
let matcher = SimilarityMatcher::new();

let fp1 = fingerprinter.fingerprint(&content1);
let fp2 = fingerprinter.fingerprint(&content2);

// Exact match (0.0 or 1.0)
let exact_match = matcher.exact_match(&fp1, &fp2);
```

### Fuzzy Similarity

```rust
use infra_fingerprinting::{SimilarityMatcher, SemanticFingerprinter};

let fingerprinter = SemanticFingerprinter::new();
let matcher = SimilarityMatcher::with_threshold(0.8);

let query_fp = fingerprinter.fingerprint(&query);

// Find similar items
let similar_items: Vec<(usize, f64)> = corpus.iter()
    .enumerate()
    .map(|(idx, content)| {
        let fp = fingerprinter.fingerprint(content);
        let similarity = matcher.compute_similarity(&query_fp, &fp);
        (idx, similarity)
    })
    .filter(|(_, sim)| *sim >= 0.8)
    .collect();

for (idx, similarity) in similar_items {
    println!("Item {}: {:.2}% similar", idx, similarity * 100.0);
}
```

### K-Nearest Neighbors

```rust
use infra_fingerprinting::{KNNMatcher, SemanticFingerprinter};

let fingerprinter = SemanticFingerprinter::new();
let mut knn = KNNMatcher::new(5); // Find 5 nearest neighbors

// Index corpus
for content in corpus {
    let fp = fingerprinter.fingerprint(content);
    knn.add(fp);
}

// Query
let query_fp = fingerprinter.fingerprint(&query);
let neighbors = knn.find_nearest(&query_fp);

for (idx, distance) in neighbors {
    println!("Neighbor {}: distance = {:.4}", idx, distance);
}
```

---

## Integration with Lightbulb

### Use Case 1: Cache Key Generation (M5)

```rust
use infra_fingerprinting::{AtomicFingerprinter, StructuralFingerprinter};
use dynctx::SlotArena;

struct CacheKey {
    content_hash: Fingerprint,
    structure_hash: Fingerprint,
    tier: CacheTier,
}

impl CacheKey {
    fn from_arena(arena: &SlotArena, tier: CacheTier) -> Self {
        let atomic_fp = AtomicFingerprinter::new();
        let structural_fp = StructuralFingerprinter::new();
        
        // Build dense view
        let dense = rope::build_dense_view(arena).unwrap();
        
        // Fingerprint content
        let content_hash = atomic_fp.fingerprint_sequence(&dense.ids);
        
        // Fingerprint structure
        let structure_hash = structural_fp.fingerprint_positions(&dense.abs_pos);
        
        CacheKey { content_hash, structure_hash, tier }
    }
    
    fn lookup_key(&self) -> String {
        format!("{:x}:{:x}", self.content_hash, self.structure_hash)
    }
}
```

### Use Case 2: Token Graph Integrity (M6)

```rust
use infra_fingerprinting::RelationalFingerprinter;
use dynctx::{RelationshipManager, SlotArena};

struct GraphIntegrity {
    fingerprinter: RelationalFingerprinter,
}

impl GraphIntegrity {
    fn compute_graph_hash(
        &self,
        arena: &SlotArena,
        relationships: &RelationshipManager,
    ) -> Fingerprint {
        let rel_list = relationships.get_all_relationships();
        self.fingerprinter.fingerprint_relationships(&rel_list)
    }
    
    fn verify_integrity(
        &self,
        arena: &SlotArena,
        relationships: &RelationshipManager,
        expected_hash: &Fingerprint,
    ) -> bool {
        let current_hash = self.compute_graph_hash(arena, relationships);
        current_hash == *expected_hash
    }
}
```

### Use Case 3: Federated Content Verification (M7)

```rust
use infra_fingerprinting::{MultiLevelFingerprinting, FingerprintLevel};

struct FederatedVerifier {
    multi_fp: MultiLevelFingerprinting,
}

impl FederatedVerifier {
    fn new() -> Self {
        FederatedVerifier {
            multi_fp: MultiLevelFingerprinting::new(vec![
                FingerprintLevel::Atomic,
                FingerprintLevel::Structural,
                FingerprintLevel::Semantic,
            ]),
        }
    }
    
    fn verify_content(&self, local: &Content, remote: &Content) -> VerificationResult {
        let local_fps = self.multi_fp.fingerprint_all_levels(local);
        let remote_fps = self.multi_fp.fingerprint_all_levels(remote);
        
        VerificationResult {
            atomic_match: local_fps.atomic == remote_fps.atomic,
            structural_match: local_fps.structural == remote_fps.structural,
            semantic_similarity: SimilarityMatcher::compute_similarity(
                &local_fps.semantic,
                &remote_fps.semantic
            ),
        }
    }
}
```

---

## Performance

### Benchmarking Results

```text
| Fingerprinting Level | Operation          | Time   |
| -------------------- | ------------------ | ------ |
| Atomic               | Single token       | ~100ns |
| Atomic               | 1000 tokens        | ~100μs |
| Relational           | 100 relationships  | ~1μs   |
| Structural           | Medium document    | ~10μs  |
| Semantic             | 1KB text           | ~1ms   |
| Deduplication        | 10K items (atomic) | ~50ms  |
| Similarity (fuzzy)   | 1K comparisons     | ~100ms |
```

### Performance Tuning

```rust
// Fast path: Atomic fingerprinting
let fast_fp = AtomicFingerprinter::new();
let fp = fast_fp.fingerprint_quick(&small_content); // ~100ns

// Balanced: Structural fingerprinting
let balanced_fp = StructuralFingerprinter::new();
let fp = balanced_fp.fingerprint(&medium_content); // ~10μs

// Accurate: Semantic fingerprinting
let accurate_fp = SemanticFingerprinter::new();
let fp = accurate_fp.fingerprint(&large_content); // ~1ms
```

### Memory Usage

```rust
// Fingerprint size: 32 bytes (256-bit hash)
// Metadata (optional): ~16 bytes
// Total per fingerprint: 32-48 bytes

// Deduplication engine memory:
// - 10K items: ~320KB-480KB
// - 100K items: ~3.2MB-4.8MB
// - 1M items: ~32MB-48MB
```

---

## Best Practices

### Choosing Fingerprinting Levels

**Use Atomic when:**
- Exact match detection is needed
- Working with small, discrete units (tokens)
- Performance is critical
- Content rarely changes

**Use Relational when:**
- Relationship patterns matter
- Graph structure is important
- Dependencies must be preserved
- Comparing complex structures

**Use Structural when:**
- Document layout matters
- Hierarchical organization is key
- Template matching is needed
- Format consistency is important

**Use Semantic when:**
- Finding similar content
- Near-duplicate detection
- Fuzzy matching required
- Content meaning matters more than exact text

### Deduplication Strategy

```rust
// Strategy 1: Fast exact deduplication (Atomic)
let mut fast_dedup = DeduplicationEngine::new(FingerprintLevel::Atomic);

// Strategy 2: Structure-aware deduplication (Structural)
let mut structural_dedup = DeduplicationEngine::new(FingerprintLevel::Structural);

// Strategy 3: Multi-level deduplication (Comprehensive)
let mut comprehensive_dedup = MultiLevelDeduplication::new(vec![
    FingerprintLevel::Atomic,
    FingerprintLevel::Structural,
    FingerprintLevel::Semantic,
]);

// Choose based on requirements:
// - Fast: Use Atomic
// - Accurate: Use Semantic
// - Balanced: Use Structural
// - Comprehensive: Use Multi-level
```

### Similarity Thresholds

```rust
// Recommended thresholds
const EXACT_MATCH: f64 = 1.0;         // 100% identical
const VERY_SIMILAR: f64 = 0.9;        // 90%+ similar
const SIMILAR: f64 = 0.75;            // 75%+ similar
const SOMEWHAT_SIMILAR: f64 = 0.5;    // 50%+ similar
const DIFFERENT: f64 = 0.0;           // Completely different

// Usage
let similarity = matcher.compute_similarity(&fp1, &fp2);

if similarity >= VERY_SIMILAR {
    // Treat as duplicates
} else if similarity >= SIMILAR {
    // Suggest as related
} else if similarity >= SOMEWHAT_SIMILAR {
    // Show as possibly related
} else {
    // Completely different
}
```

### Error Handling

```rust
use infra_fingerprinting::FingerprintError;

match fingerprinter.fingerprint(&content) {
    Ok(fp) => {
        // Success
    }
    Err(FingerprintError::InvalidInput) => {
        // Handle invalid input
    }
    Err(FingerprintError::HashingFailed(msg)) => {
        // Handle hashing error
    }
    Err(e) => {
        // Handle other errors
        log::error!("Fingerprinting failed: {:?}", e);
    }
}
```

---

## Advanced Usage

### Custom Fingerprinting

```rust
use infra_fingerprinting::{Fingerprinter, Fingerprint};
use blake3::Hasher;

struct CustomFingerprinter {
    hasher_factory: fn() -> Hasher,
}

impl Fingerprinter for CustomFingerprinter {
    fn fingerprint(&self, content: &[u8]) -> Result<Fingerprint, FingerprintError> {
        let mut hasher = (self.hasher_factory)();
        
        // Custom preprocessing
        let processed = preprocess(content);
        
        // Hash
        hasher.update(&processed);
        let hash = hasher.finalize();
        
        Ok(Fingerprint::from_hash(hash.as_bytes()))
    }
}
```

### Incremental Fingerprinting

```rust
use infra_fingerprinting::IncrementalFingerprinter;

let mut incremental = IncrementalFingerprinter::new();

// Add content incrementally
incremental.update(&chunk1);
incremental.update(&chunk2);
incremental.update(&chunk3);

// Finalize fingerprint
let fp = incremental.finalize();
```

### Fingerprint Caching

```rust
use std::collections::HashMap;

struct FingerprintCache {
    cache: HashMap<ContentId, Fingerprint>,
    fingerprinter: AtomicFingerprinter,
}

impl FingerprintCache {
    fn get_or_compute(&mut self, id: ContentId, content: &Content) -> Fingerprint {
        self.cache.entry(id)
            .or_insert_with(|| self.fingerprinter.fingerprint(content))
            .clone()
    }
    
    fn invalidate(&mut self, id: ContentId) {
        self.cache.remove(&id);
    }
}
```

---

## Testing and Validation

### Unit Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_atomic_fingerprinting() {
        let fp = AtomicFingerprinter::new();
        
        let content1 = b"test content";
        let content2 = b"test content";
        let content3 = b"different content";
        
        let fp1 = fp.fingerprint(content1).unwrap();
        let fp2 = fp.fingerprint(content2).unwrap();
        let fp3 = fp.fingerprint(content3).unwrap();
        
        assert_eq!(fp1, fp2);
        assert_ne!(fp1, fp3);
    }
    
    #[test]
    fn test_deduplication() {
        let mut dedup = DeduplicationEngine::new(FingerprintLevel::Atomic);
        
        assert!(dedup.is_unique(b"item1"));
        assert!(dedup.is_unique(b"item2"));
        assert!(!dedup.is_unique(b"item1")); // Duplicate
    }
}
```

### Benchmarking

```rust
#[cfg(test)]
mod benches {
    use criterion::{black_box, Criterion};
    
    fn bench_atomic_fingerprinting(c: &mut Criterion) {
        let fp = AtomicFingerprinter::new();
        let content = vec![0u8; 1000];
        
        c.bench_function("atomic_fingerprint_1000_bytes", |b| {
            b.iter(|| fp.fingerprint(black_box(&content)))
        });
    }
}
```

---

## Further Reading

- **Source Code**: `crates/infra-fingerprinting/src/` - Implementation details
- **Integration Guide**: `docs/DYNANIML_INTEGRATION.md` - High-level integration
- **Benchmarks**: `crates/infra-fingerprinting/benches/` - Performance benchmarks
- **Lightbulb Roadmap**: `ROADMAP.md` - How fingerprinting supports M5, M6, M7
