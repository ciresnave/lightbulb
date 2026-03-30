//! Knowledge Base Construction and Management
//!
//! Implements a knowledge base that integrates with KV cache eviction to extend
//! effective context beyond cache limits. When tokens are evicted, they're summarized
//! and stored in the KB with lookup keys, allowing the LLM to retrieve them on demand.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    KV Cache (Limited)                       │
//! ├─────────────────────────────────────────────────────────────┤
//! │ System prompt: [KB instructions - EVICT LAST]               │
//! │ User prompt: "What is the capital of France?"               │
//! │ [KB:paris_capital] Paris is capital of France, pop 2.1M     │ ← Evicted summary
//! │ Assistant: "The capital is Paris..."                        │
//! └─────────────────────────────────────────────────────────────┘
//!                              ↓ Eviction
//! ┌─────────────────────────────────────────────────────────────┐
//! │              Knowledge Base (Unlimited)                     │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Key: "paris_capital"                                        │
//! │ Summary: "Paris is capital of France, pop 2.1M"             │
//! │ Full Content: "Paris is the capital of France. It has..."   │
//! │ Source Tokens: [1234, 5678, ...]                            │
//! │ Confidence: 0.95                                            │
//! └─────────────────────────────────────────────────────────────┘
//!                              ↓ LLM requests: <RETRIEVE:paris_capital>
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Re-inject into Context                   │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage Flow
//!
//! 1. **System prompt includes KB instructions** (marked as evict-last)
//! 2. **KV cache fills up** → Eviction policy selects tokens
//! 3. **Evicted tokens → KB fact**:
//!    - Generate summary (1 sentence)
//!    - Create lookup key (semantic or hash-based)
//!    - Store in KB with full content
//! 4. **Replace in cache**: `[KB:key] summary`
//! 5. **LLM sees placeholder** and can request: `<RETRIEVE:key>`
//! 6. **System intercepts** retrieve token and injects full content
//!
//! # Future Enhancements (See ROADMAP.md)
//!
//! - Semantic search with embeddings (M6+)
//! - NLI-based consistency checking
//! - Graph-based fact relationships
//! - Multi-level eviction (KB facts can be archived)
//! - Persistent KB across sessions

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

/// Unique identifier for a knowledge base fact
pub type FactKey = String;

/// Category of knowledge fact
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum FactCategory {
    /// Factual knowledge (Paris is capital of France)
    Factual,

    /// Numerical data (GDP = $25.5T)
    Numerical,

    /// Temporal information (Event X happened in 2023)
    Temporal,

    /// Relational (X is related to Y)
    Relational,

    /// Procedural (How to do X)
    Procedural,

    /// General/uncategorized
    General,
}

impl Default for FactCategory {
    fn default() -> Self {
        FactCategory::General
    }
}

/// A single fact in the knowledge base
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Fact {
    /// Unique lookup key
    pub key: FactKey,

    /// One-sentence summary (what LLM sees in placeholder)
    pub summary: String,

    /// Full content from evicted tokens
    pub full_content: String,

    /// Original tokens (optional, for re-injection)
    pub source_tokens: Option<Vec<u32>>,

    /// Category of knowledge
    pub category: FactCategory,

    /// Confidence score (0.0-1.0)
    pub confidence: f64,

    /// When this fact was added
    pub timestamp: u64,

    /// Source of this fact (e.g., "evicted_position_150-200")
    pub source: String,

    /// Number of times this fact has been retrieved
    pub retrieval_count: usize,
}

impl Fact {
    /// Create a new fact
    pub fn new(
        key: impl Into<String>,
        summary: impl Into<String>,
        full_content: impl Into<String>,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            key: key.into(),
            summary: summary.into(),
            full_content: full_content.into(),
            source_tokens: None,
            category: FactCategory::General,
            confidence: 1.0,
            timestamp: now,
            source: String::new(),
            retrieval_count: 0,
        }
    }

    /// Set source tokens
    pub fn with_tokens(mut self, tokens: Vec<u32>) -> Self {
        self.source_tokens = Some(tokens);
        self
    }

    /// Set category
    pub fn with_category(mut self, category: FactCategory) -> Self {
        self.category = category;
        self
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set source
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }

    /// Format as placeholder for KV cache
    pub fn as_placeholder(&self) -> String {
        format!("[KB:{}] {}", self.key, self.summary)
    }

    /// Increment retrieval count
    pub fn mark_retrieved(&mut self) {
        self.retrieval_count += 1;
    }
}

/// Record of an eviction event
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EvictionRecord {
    /// Key of the fact that replaced evicted content
    pub fact_key: FactKey,

    /// Position range in KV cache that was evicted
    pub position_range: (usize, usize),

    /// What was placed in the cache as replacement
    pub placeholder_text: String,

    /// When this eviction occurred
    pub timestamp: u64,
}

/// Statistics about KB usage
#[derive(Debug, Clone, Default)]
pub struct KnowledgeBaseStats {
    /// Total facts stored
    pub fact_count: usize,

    /// Total evictions that created facts
    pub eviction_count: usize,

    /// Total retrieval requests
    pub retrieval_count: usize,

    /// Facts by category
    pub facts_by_category: HashMap<FactCategory, usize>,

    /// Average confidence score
    pub avg_confidence: f64,

    /// Most retrieved fact
    pub most_retrieved_key: Option<FactKey>,
}

/// Errors that can occur in KB operations
#[derive(Debug, Error)]
pub enum KnowledgeBaseError {
    #[error("Fact with key {0} not found")]
    FactNotFound(FactKey),

    #[error("Invalid key format: {0}")]
    InvalidKey(String),

    #[error("Fact with key {0} already exists")]
    DuplicateKey(FactKey),

    #[error("Summary generation failed: {0}")]
    SummaryGenerationFailed(String),
}

/// Configuration for knowledge base behavior
#[derive(Debug, Clone)]
pub struct KnowledgeBaseConfig {
    /// Maximum number of facts to store (0 = unlimited)
    pub max_facts: usize,

    /// Automatically generate keys if not provided
    pub auto_generate_keys: bool,

    /// Track eviction history
    pub track_evictions: bool,

    /// Enable retrieval statistics
    pub enable_stats: bool,
}

impl Default for KnowledgeBaseConfig {
    fn default() -> Self {
        Self {
            max_facts: 0, // Unlimited
            auto_generate_keys: true,
            track_evictions: true,
            enable_stats: true,
        }
    }
}

/// Knowledge base for storing facts from evicted KV cache content
pub struct KnowledgeBase {
    /// All facts indexed by key
    facts: HashMap<FactKey, Fact>,

    /// Eviction history
    evictions: Vec<EvictionRecord>,

    /// Configuration
    config: KnowledgeBaseConfig,

    /// Statistics
    stats: KnowledgeBaseStats,

    /// Key generation counter
    key_counter: u64,
}

impl KnowledgeBase {
    /// Create a new knowledge base
    pub fn new() -> Self {
        Self::with_config(KnowledgeBaseConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: KnowledgeBaseConfig) -> Self {
        Self {
            facts: HashMap::new(),
            evictions: Vec::new(),
            config,
            stats: KnowledgeBaseStats::default(),
            key_counter: 0,
        }
    }

    /// Add a fact to the knowledge base
    pub fn add_fact(&mut self, fact: Fact) -> Result<(), KnowledgeBaseError> {
        // Check if key already exists
        if self.facts.contains_key(&fact.key) {
            return Err(KnowledgeBaseError::DuplicateKey(fact.key.clone()));
        }

        // Check max facts limit
        if self.config.max_facts > 0 && self.facts.len() >= self.config.max_facts {
            // TODO: Implement eviction policy for KB itself (future enhancement)
            // For now, reject new facts when full
            return Err(KnowledgeBaseError::SummaryGenerationFailed(
                "Knowledge base is full".to_string(),
            ));
        }

        // Update stats
        if self.config.enable_stats {
            *self
                .stats
                .facts_by_category
                .entry(fact.category)
                .or_insert(0) += 1;
            self.stats.fact_count += 1;
        }

        // Store fact
        self.facts.insert(fact.key.clone(), fact);

        Ok(())
    }

    /// Create fact from evicted content
    pub fn create_fact_from_eviction(
        &mut self,
        summary: impl Into<String>,
        full_content: impl Into<String>,
        position_range: (usize, usize),
        tokens: Option<Vec<u32>>,
    ) -> Result<FactKey, KnowledgeBaseError> {
        // Generate key
        let key = self.generate_key();

        // Create fact
        let mut fact = Fact::new(key.clone(), summary, full_content)
            .with_source(format!("evicted_{}_{}", position_range.0, position_range.1));

        if let Some(tokens) = tokens {
            fact = fact.with_tokens(tokens);
        }

        // Create eviction record
        if self.config.track_evictions {
            let record = EvictionRecord {
                fact_key: key.clone(),
                position_range,
                placeholder_text: fact.as_placeholder(),
                timestamp: fact.timestamp,
            };
            self.evictions.push(record);
            self.stats.eviction_count += 1;
        }

        // Add to KB
        self.add_fact(fact)?;

        Ok(key)
    }

    /// Retrieve a fact by key
    pub fn get_fact(&mut self, key: &str) -> Result<&Fact, KnowledgeBaseError> {
        let fact = self
            .facts
            .get_mut(key)
            .ok_or_else(|| KnowledgeBaseError::FactNotFound(key.to_string()))?;

        // Update stats
        fact.mark_retrieved();
        if self.config.enable_stats {
            self.stats.retrieval_count += 1;
        }

        Ok(self.facts.get(key).unwrap())
    }

    /// Get fact without marking as retrieved (for inspection)
    pub fn peek_fact(&self, key: &str) -> Option<&Fact> {
        self.facts.get(key)
    }

    /// Get all facts by category
    pub fn get_facts_by_category(&self, category: FactCategory) -> Vec<&Fact> {
        self.facts
            .values()
            .filter(|f| f.category == category)
            .collect()
    }

    /// Get all facts
    pub fn all_facts(&self) -> Vec<&Fact> {
        self.facts.values().collect()
    }

    /// Get all facts as owned values (for serialization)
    pub fn get_all_facts(&self) -> Vec<Fact> {
        self.facts.values().cloned().collect()
    }

    /// Format all facts as context string for prompt injection
    pub fn to_context_string(&self) -> String {
        if self.facts.is_empty() {
            return String::new();
        }

        let mut context = String::from("=== Knowledge Base ===\n");
        for fact in self.facts.values() {
            context.push_str(&format!(
                "[{}] {} (confidence: {:.2})\n",
                fact.key, fact.summary, fact.confidence
            ));
        }
        context.push_str("======================\n");
        context
    }

    /// Generate system instructions for LLM
    pub fn system_instructions() -> &'static str {
        r#"KNOWLEDGE BASE SYSTEM:
You have access to a knowledge base that extends beyond your immediate context window.

When you see: [KB:key] summary_text
- This indicates information that was removed from context to save memory
- The summary gives you a brief overview
- If you need the full details, output: <RETRIEVE:key>
- The system will inject the complete information

Example:
You see: [KB:paris_facts] Paris is capital of France, population 2.1M
If needed: <RETRIEVE:paris_facts>
System provides: "Paris is the capital of France. It has a population of 2.1 million people and covers an area of 105 km²..."

IMPORTANT: These instructions should be kept in context as long as possible."#
    }

    /// Get statistics
    pub fn stats(&self) -> &KnowledgeBaseStats {
        &self.stats
    }

    /// Get eviction history
    pub fn eviction_history(&self) -> &[EvictionRecord] {
        &self.evictions
    }

    /// Generate a unique key
    fn generate_key(&mut self) -> FactKey {
        if self.config.auto_generate_keys {
            self.key_counter += 1;
            format!("fact_{}", self.key_counter)
        } else {
            // Fallback to timestamp-based
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            format!("fact_{}", now)
        }
    }

    /// Clear all facts (useful for testing or session resets)
    pub fn clear(&mut self) {
        self.facts.clear();
        self.evictions.clear();
        self.stats = KnowledgeBaseStats::default();
        self.key_counter = 0;
    }

    /// Get fact count
    pub fn len(&self) -> usize {
        self.facts.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }
}

impl Default for KnowledgeBase {
    fn default() -> Self {
        Self::new()
    }
}

/// Convergence detection for iterative workflows
#[derive(Debug, Clone)]
pub struct ConvergenceDetector {
    /// Maximum iterations before forced stop
    max_iterations: usize,

    /// Minimum new facts per iteration to continue
    min_new_facts_threshold: usize,

    /// Track facts per iteration
    facts_per_iteration: Vec<usize>,

    /// Current iteration
    current_iteration: usize,

    /// Whether convergence has been detected
    converged: bool,

    /// Reason for convergence
    convergence_reason: Option<String>,
}

impl ConvergenceDetector {
    /// Create a new convergence detector
    pub fn new(max_iterations: usize, min_new_facts_threshold: usize) -> Self {
        Self {
            max_iterations,
            min_new_facts_threshold,
            facts_per_iteration: Vec::new(),
            current_iteration: 0,
            converged: false,
            convergence_reason: None,
        }
    }

    /// Record facts added in current iteration
    pub fn record_iteration(&mut self, new_facts: usize) {
        self.facts_per_iteration.push(new_facts);
        self.current_iteration += 1;

        // Check for convergence
        if self.current_iteration >= self.max_iterations {
            self.converged = true;
            self.convergence_reason =
                Some(format!("Max iterations ({}) reached", self.max_iterations));
        } else if new_facts < self.min_new_facts_threshold {
            self.converged = true;
            self.convergence_reason = Some(format!(
                "Fact saturation detected (only {} new facts, threshold: {})",
                new_facts, self.min_new_facts_threshold
            ));
        }
    }

    /// Check if converged
    pub fn has_converged(&self) -> bool {
        self.converged
    }

    /// Get convergence reason
    pub fn convergence_reason(&self) -> Option<&str> {
        self.convergence_reason.as_deref()
    }

    /// Get current iteration
    pub fn current_iteration(&self) -> usize {
        self.current_iteration
    }

    /// Get total facts across all iterations
    pub fn total_facts(&self) -> usize {
        self.facts_per_iteration.iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fact_creation() {
        let fact = Fact::new(
            "paris_capital",
            "Paris is capital of France",
            "Paris is the capital of France. It has a population of 2.1 million...",
        )
        .with_category(FactCategory::Factual)
        .with_confidence(0.95);

        assert_eq!(fact.key, "paris_capital");
        assert_eq!(fact.category, FactCategory::Factual);
        assert_eq!(fact.confidence, 0.95);
        assert_eq!(fact.retrieval_count, 0);
    }

    #[test]
    fn test_fact_placeholder() {
        let fact = Fact::new("test_key", "Test summary", "Full content");
        assert_eq!(fact.as_placeholder(), "[KB:test_key] Test summary");
    }

    #[test]
    fn test_knowledge_base_add_retrieve() {
        let mut kb = KnowledgeBase::new();

        let fact = Fact::new("key1", "Summary 1", "Full content 1");
        kb.add_fact(fact).unwrap();

        assert_eq!(kb.len(), 1);

        let retrieved = kb.get_fact("key1").unwrap();
        assert_eq!(retrieved.summary, "Summary 1");
        assert_eq!(retrieved.retrieval_count, 1);

        // Retrieve again
        let retrieved2 = kb.get_fact("key1").unwrap();
        assert_eq!(retrieved2.retrieval_count, 2);
    }

    #[test]
    fn test_create_fact_from_eviction() {
        let mut kb = KnowledgeBase::new();

        let key = kb
            .create_fact_from_eviction(
                "Paris is capital",
                "Paris is the capital of France...",
                (100, 150),
                Some(vec![1, 2, 3]),
            )
            .unwrap();

        assert_eq!(kb.len(), 1);
        assert_eq!(kb.eviction_history().len(), 1);

        let fact = kb.peek_fact(&key).unwrap();
        assert!(fact.source_tokens.is_some());
        assert_eq!(fact.source_tokens.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_facts_by_category() {
        let mut kb = KnowledgeBase::new();

        kb.add_fact(Fact::new("fact1", "Factual", "Content").with_category(FactCategory::Factual))
            .unwrap();
        kb.add_fact(
            Fact::new("fact2", "Numerical", "Content").with_category(FactCategory::Numerical),
        )
        .unwrap();
        kb.add_fact(Fact::new("fact3", "Factual2", "Content").with_category(FactCategory::Factual))
            .unwrap();

        let factual = kb.get_facts_by_category(FactCategory::Factual);
        assert_eq!(factual.len(), 2);

        let numerical = kb.get_facts_by_category(FactCategory::Numerical);
        assert_eq!(numerical.len(), 1);
    }

    #[test]
    fn test_duplicate_key_error() {
        let mut kb = KnowledgeBase::new();

        kb.add_fact(Fact::new("key1", "Summary", "Content"))
            .unwrap();

        let result = kb.add_fact(Fact::new("key1", "Summary2", "Content2"));
        assert!(matches!(result, Err(KnowledgeBaseError::DuplicateKey(_))));
    }

    #[test]
    fn test_fact_not_found_error() {
        let mut kb = KnowledgeBase::new();
        let result = kb.get_fact("nonexistent");
        assert!(matches!(result, Err(KnowledgeBaseError::FactNotFound(_))));
    }

    #[test]
    fn test_convergence_detector() {
        let mut detector = ConvergenceDetector::new(5, 2);

        detector.record_iteration(10);
        assert!(!detector.has_converged());

        detector.record_iteration(5);
        assert!(!detector.has_converged());

        detector.record_iteration(1); // Below threshold
        assert!(detector.has_converged());
        assert!(
            detector
                .convergence_reason()
                .unwrap()
                .contains("saturation")
        );
    }

    #[test]
    fn test_convergence_max_iterations() {
        let mut detector = ConvergenceDetector::new(3, 2);

        detector.record_iteration(10);
        detector.record_iteration(10);
        detector.record_iteration(10);

        assert!(detector.has_converged());
        assert!(
            detector
                .convergence_reason()
                .unwrap()
                .contains("Max iterations")
        );
    }

    #[test]
    fn test_system_instructions() {
        let instructions = KnowledgeBase::system_instructions();
        assert!(instructions.contains("KNOWLEDGE BASE SYSTEM"));
        assert!(instructions.contains("[KB:key]"));
        assert!(instructions.contains("<RETRIEVE:key>"));
    }

    #[test]
    fn test_to_context_string() {
        let mut kb = KnowledgeBase::new();

        kb.add_fact(Fact::new("key1", "Summary 1", "Content 1").with_confidence(0.95))
            .unwrap();
        kb.add_fact(Fact::new("key2", "Summary 2", "Content 2").with_confidence(0.88))
            .unwrap();

        let context = kb.to_context_string();
        assert!(context.contains("Knowledge Base"));
        assert!(context.contains("[key1]"));
        assert!(context.contains("[key2]"));
        assert!(context.contains("0.95"));
    }

    #[test]
    fn test_stats() {
        let mut kb = KnowledgeBase::new();

        kb.add_fact(Fact::new("f1", "S1", "C1").with_category(FactCategory::Factual))
            .unwrap();
        kb.add_fact(Fact::new("f2", "S2", "C2").with_category(FactCategory::Numerical))
            .unwrap();

        kb.get_fact("f1").unwrap();
        kb.get_fact("f1").unwrap();
        kb.get_fact("f2").unwrap();

        let stats = kb.stats();
        assert_eq!(stats.fact_count, 2);
        assert_eq!(stats.retrieval_count, 3);
        assert_eq!(
            *stats.facts_by_category.get(&FactCategory::Factual).unwrap(),
            1
        );
    }
}
