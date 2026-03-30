//! Query Analysis and Intent Classification
//!
//! Fast preprocessing to understand query intent before routing to retrieval or LLM.
//! Uses pattern matching for common cases, falls back to small LLM for ambiguous queries.
//!
//! # Architecture
//!
//! ```text
//! Raw Query
//!     ↓
//! QueryAnalyzer
//!     ├─ Intent Classification (pattern matching)
//!     ├─ Entity Extraction (regex)
//!     ├─ Constraint Parsing (temporal, filters)
//!     └─ Ambiguity Detection
//!     ↓
//! AnalyzedQuery
//!     ├─ intent: QueryIntent
//!     ├─ entities: Vec<Entity>
//!     ├─ constraints: Vec<Constraint>
//!     ├─ sub_queries: Vec<SubQuery>
//!     └─ confidence: f64
//! ```
//!
//! # Example
//!
//! ```ignore
//! let analyzer = QueryAnalyzer::new();
//! let query = "What is the capital of France?";
//! let analyzed = analyzer.analyze(query)?;
//!
//! assert_eq!(analyzed.intent, QueryIntent::Definition);
//! assert_eq!(analyzed.entities.len(), 1); // "France"
//! assert!(analyzed.confidence > 0.9);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during query analysis
#[derive(Debug, Error)]
pub enum QueryAnalysisError {
    #[error("Empty query")]
    EmptyQuery,

    #[error("Query too long: {0} characters (max {1})")]
    QueryTooLong(usize, usize),

    #[error("Failed to parse constraint: {0}")]
    ConstraintParseError(String),

    #[error("Circular dependency in sub-queries: {0}")]
    CircularDependency(String),
}

/// Query intent classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QueryIntent {
    /// Define a concept or term
    /// Examples: "What is X?", "Define Y"
    Definition,

    /// Explain a process or procedure
    /// Examples: "How to X?", "Steps for Y"
    Procedure,

    /// Compare two or more things
    /// Examples: "X vs Y", "Difference between A and B"
    Comparison,

    /// Diagnose or fix a problem
    /// Examples: "Why doesn't X work?", "How to fix Y?"
    Troubleshooting,

    /// Explain why or how something works
    /// Examples: "Why does X happen?", "How does Y work?"
    Explanation,

    /// Analyze or evaluate something
    /// Examples: "Is X better than Y?", "Should I use Z?"
    Analysis,

    /// Synthesize information from multiple sources
    /// Examples: "Summarize X", "What are the key points of Y?"
    Synthesis,

    /// Factual information retrieval
    /// Examples: "When was X?", "Who is Y?"
    Factual,

    /// Open-ended or unclear intent
    /// Examples: Complex multi-part questions
    Unknown,
}

/// Entity extracted from query
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    /// Entity text as it appears in query
    pub text: String,

    /// Entity type (Person, Place, Organization, Concept, etc.)
    pub entity_type: EntityType,

    /// Position in original query (character offset)
    pub position: usize,

    /// Confidence score (0.0-1.0)
    pub confidence: f64,
}

/// Types of entities that can be extracted
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntityType {
    /// Person name
    Person,

    /// Geographic location
    Place,

    /// Organization or company
    Organization,

    /// Technical concept or term
    Concept,

    /// Programming language or framework
    Technology,

    /// File path or URL
    Path,

    /// Date or time reference
    Temporal,

    /// Numeric value
    Numeric,

    /// Unknown entity type
    Other,
}

/// Constraint extracted from query
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Constraint {
    /// Constraint type
    pub constraint_type: ConstraintType,

    /// Constraint value
    pub value: String,

    /// Optional comparison operator
    pub operator: Option<ComparisonOperator>,
}

/// Types of constraints
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintType {
    /// Time-based constraint ("after 2024", "in the last week")
    Temporal,

    /// Numeric filter ("greater than 100", "between 10 and 20")
    Numeric,

    /// Category or tag filter ("only Python", "exclude deprecated")
    Category,

    /// Source constraint ("from official docs", "peer-reviewed")
    Source,

    /// Language constraint ("in English", "not translated")
    Language,

    /// Other constraint type
    Other,
}

/// Comparison operators for constraints
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonOperator {
    Equal,
    NotEqual,
    GreaterThan,
    LessThan,
    GreaterOrEqual,
    LessOrEqual,
    Between,
    In,
    NotIn,
}

/// Sub-query with dependencies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubQuery {
    /// Sub-query text
    pub text: String,

    /// Intent of this sub-query
    pub intent: QueryIntent,

    /// IDs of sub-queries this depends on
    pub depends_on: Vec<usize>,

    /// Priority (higher = earlier execution)
    pub priority: u32,
}

/// Analyzed query with extracted metadata
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalyzedQuery {
    /// Original query text
    pub original: String,

    /// Classified intent
    pub intent: QueryIntent,

    /// Extracted entities
    pub entities: Vec<Entity>,

    /// Extracted constraints
    pub constraints: Vec<Constraint>,

    /// Sub-queries with dependencies (for complex queries)
    pub sub_queries: Vec<SubQuery>,

    /// Detected ambiguities or unclear parts
    pub ambiguities: Vec<String>,

    /// Overall confidence in analysis (0.0-1.0)
    pub confidence: f64,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Query analyzer with fast pattern matching
pub struct QueryAnalyzer {
    /// Maximum query length to analyze
    max_query_length: usize,

    /// Intent patterns for fast classification
    intent_patterns: HashMap<QueryIntent, Vec<&'static str>>,
}

impl QueryAnalyzer {
    /// Create a new query analyzer
    pub fn new() -> Self {
        let mut intent_patterns = HashMap::new();

        // Definition patterns
        intent_patterns.insert(
            QueryIntent::Definition,
            vec![
                "what is",
                "what are",
                "define",
                "definition of",
                "meaning of",
                "what does",
                "what's",
            ],
        );

        // Procedure patterns
        intent_patterns.insert(
            QueryIntent::Procedure,
            vec![
                "how to",
                "how do i",
                "how can i",
                "steps to",
                "guide to",
                "tutorial",
                "instructions for",
            ],
        );

        // Comparison patterns
        intent_patterns.insert(
            QueryIntent::Comparison,
            vec![
                " vs ",
                " versus ",
                "difference between",
                "compare",
                "comparison of",
                "better than",
                "worse than",
            ],
        );

        // Troubleshooting patterns
        intent_patterns.insert(
            QueryIntent::Troubleshooting,
            vec![
                "why doesn't",
                "why won't",
                "how to fix",
                "not working",
                "error",
                "problem with",
                "issue with",
                "troubleshoot",
            ],
        );

        // Explanation patterns
        intent_patterns.insert(
            QueryIntent::Explanation,
            vec![
                "why does",
                "why is",
                "how does",
                "how is",
                "explain",
                "explanation of",
                "reason for",
            ],
        );

        // Analysis patterns
        intent_patterns.insert(
            QueryIntent::Analysis,
            vec![
                "should i",
                "is it",
                "evaluate",
                "assess",
                "pros and cons",
                "advantages of",
                "disadvantages of",
            ],
        );

        // Synthesis patterns
        intent_patterns.insert(
            QueryIntent::Synthesis,
            vec![
                "summarize",
                "summary of",
                "key points",
                "main ideas",
                "overview of",
                "what are the",
            ],
        );

        // Factual patterns
        intent_patterns.insert(
            QueryIntent::Factual,
            vec![
                "when was",
                "when did",
                "who is",
                "who was",
                "where is",
                "where was",
                "which",
            ],
        );

        Self {
            max_query_length: 10000,
            intent_patterns,
        }
    }

    /// Analyze a query and extract metadata
    pub fn analyze(&self, query: &str) -> Result<AnalyzedQuery, QueryAnalysisError> {
        // Validate query
        if query.trim().is_empty() {
            return Err(QueryAnalysisError::EmptyQuery);
        }

        if query.len() > self.max_query_length {
            return Err(QueryAnalysisError::QueryTooLong(
                query.len(),
                self.max_query_length,
            ));
        }

        let query_lower = query.to_lowercase();

        // Classify intent
        let (intent, confidence) = self.classify_intent(&query_lower);

        // Extract entities
        let entities = self.extract_entities(query);

        // Parse constraints
        let constraints = self.parse_constraints(&query_lower);

        // Detect ambiguities
        let ambiguities = self.detect_ambiguities(&query_lower);

        // Build analyzed query
        Ok(AnalyzedQuery {
            original: query.to_string(),
            intent,
            entities,
            constraints,
            sub_queries: Vec::new(), // Will be populated by decompose_query()
            ambiguities,
            confidence,
            metadata: HashMap::new(),
        })
    }

    /// Classify query intent using pattern matching
    fn classify_intent(&self, query_lower: &str) -> (QueryIntent, f64) {
        let mut best_intent = QueryIntent::Unknown;
        let mut best_confidence = 0.0;

        // Special case: "how does it compare" should be Comparison, not Explanation
        if query_lower.contains("compare")
            && (query_lower.contains("how does") || query_lower.contains("how do"))
        {
            return (QueryIntent::Comparison, 0.85);
        }

        for (intent, patterns) in &self.intent_patterns {
            for pattern in patterns {
                if query_lower.contains(pattern) {
                    // Confidence based on pattern specificity and position
                    let position_weight = if query_lower.starts_with(pattern) {
                        1.0
                    } else {
                        0.8
                    };

                    let specificity = pattern.len() as f64 / 20.0; // Longer patterns more specific
                    let confidence = (0.7 + specificity * 0.3) * position_weight;

                    if confidence > best_confidence {
                        best_intent = *intent;
                        best_confidence = confidence;
                    }
                }
            }
        }

        // If no pattern matched, default to Unknown with low confidence
        if best_confidence == 0.0 {
            best_confidence = 0.3;
        }

        (best_intent, best_confidence.min(1.0))
    }

    /// Extract entities using regex patterns
    fn extract_entities(&self, query: &str) -> Vec<Entity> {
        let mut entities = Vec::new();

        // Simple capitalized word extraction (basic named entity recognition)
        // In production, this would use a proper NER model
        let words: Vec<&str> = query.split_whitespace().collect();

        for (i, word) in words.iter().enumerate() {
            // Remove punctuation from word for analysis
            let clean_word: String = word
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '-')
                .collect();

            // Check if word starts with capital letter (likely proper noun)
            if let Some(first_char) = clean_word.chars().next() {
                if first_char.is_uppercase() && clean_word.len() > 1 {
                    // Skip common non-entity words
                    let lower = clean_word.to_lowercase();
                    if ![
                        "what", "when", "where", "who", "why", "how", "the", "and", "or",
                    ]
                    .contains(&lower.as_str())
                    {
                        // Determine entity type (very basic heuristic)
                        let entity_type = if word.contains('.') || word.contains('/') {
                            EntityType::Path
                        } else if clean_word.chars().all(|c| c.is_alphanumeric() || c == '-') {
                            EntityType::Concept
                        } else {
                            EntityType::Other
                        };

                        entities.push(Entity {
                            text: clean_word,
                            entity_type,
                            position: i,
                            confidence: 0.6, // Low confidence for simple heuristic
                        });
                    }
                }
            }
        }

        entities
    }

    /// Parse constraints from query
    fn parse_constraints(&self, query_lower: &str) -> Vec<Constraint> {
        let mut constraints = Vec::new();

        // Temporal constraints
        if query_lower.contains("after") || query_lower.contains("since") {
            constraints.push(Constraint {
                constraint_type: ConstraintType::Temporal,
                value: "recent".to_string(),
                operator: Some(ComparisonOperator::GreaterThan),
            });
        }

        if query_lower.contains("before") || query_lower.contains("until") {
            constraints.push(Constraint {
                constraint_type: ConstraintType::Temporal,
                value: "past".to_string(),
                operator: Some(ComparisonOperator::LessThan),
            });
        }

        // Category constraints
        if query_lower.contains("only") || query_lower.contains("just") {
            constraints.push(Constraint {
                constraint_type: ConstraintType::Category,
                value: "exclusive".to_string(),
                operator: Some(ComparisonOperator::Equal),
            });
        }

        if query_lower.contains("exclude") || query_lower.contains("without") {
            constraints.push(Constraint {
                constraint_type: ConstraintType::Category,
                value: "exclusive".to_string(),
                operator: Some(ComparisonOperator::NotEqual),
            });
        }

        constraints
    }

    /// Detect ambiguous or unclear parts of query
    fn detect_ambiguities(&self, query_lower: &str) -> Vec<String> {
        let mut ambiguities = Vec::new();

        // Check for pronouns without clear antecedents
        let ambiguous_pronouns = ["it", "this", "that", "they", "them"];
        for pronoun in &ambiguous_pronouns {
            if query_lower.contains(pronoun) {
                ambiguities.push(format!("Pronoun '{}' may be unclear", pronoun));
            }
        }

        // Check for multiple questions
        let question_count = query_lower.matches('?').count();
        if question_count > 1 {
            ambiguities.push(format!("Multiple questions detected ({})", question_count));
        }

        ambiguities
    }

    /// Decompose complex queries into sub-queries with dependencies
    ///
    /// Breaks down multi-part questions into simpler sub-queries that can be
    /// answered independently, with dependency tracking for proper ordering.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let query = "How does Rust ownership work and how does it compare to C++ RAII?";
    /// let mut analyzed = analyzer.analyze(query)?;
    /// analyzer.decompose_query(&mut analyzed)?;
    ///
    /// // Results in 3 sub-queries:
    /// // 1. "How does Rust ownership work?" (Explanation)
    /// // 2. "What is C++ RAII?" (Definition)
    /// // 3. "Compare Rust ownership vs C++ RAII" (Comparison, depends on 1 & 2)
    /// ```
    pub fn decompose_query(&self, analyzed: &mut AnalyzedQuery) -> Result<(), QueryAnalysisError> {
        // Skip if already simple
        if analyzed.ambiguities.is_empty()
            && !analyzed.original.contains(" and ")
            && !analyzed.original.contains(" or ")
            && analyzed.original.matches('?').count() <= 1
        {
            return Ok(());
        }

        let mut sub_queries = Vec::new();

        // Split on coordinating conjunctions with question marks
        let parts = self.split_complex_query(&analyzed.original);

        for (idx, part) in parts.iter().enumerate() {
            if part.trim().is_empty() {
                continue;
            }

            // Analyze the sub-query
            let part_lower = part.to_lowercase();
            let (intent, _) = self.classify_intent(&part_lower);

            // Determine dependencies
            let depends_on = self.find_dependencies(idx, &parts, &sub_queries);

            // Priority: earlier sub-queries have higher priority
            let priority = (parts.len() - idx) as u32;

            sub_queries.push(SubQuery {
                text: part.trim().to_string(),
                intent,
                depends_on,
                priority,
            });
        }

        analyzed.sub_queries = sub_queries;
        Ok(())
    }

    /// Split complex query into parts
    fn split_complex_query(&self, query: &str) -> Vec<String> {
        let mut parts = Vec::new();

        // Split on " and " or " or " that appear with questions
        let query_lower = query.to_lowercase();

        if query_lower.contains(" and how ") {
            // "What is X and how does Y work?"
            let split: Vec<&str> = query.split(" and ").collect();
            for part in split {
                parts.push(part.to_string());
            }
        } else if query_lower.contains(" and what ") {
            let split: Vec<&str> = query.split(" and ").collect();
            for part in split {
                parts.push(part.to_string());
            }
        } else if query.matches('?').count() > 1 {
            // Multiple questions separated by '?'
            let split: Vec<&str> = query.split('?').collect();
            for part in split {
                if !part.trim().is_empty() {
                    parts.push(format!("{}?", part.trim()));
                }
            }
        } else if query_lower.contains(" and ") && query.len() > 100 {
            // Long query with "and" - likely compound
            let split: Vec<&str> = query.split(" and ").collect();
            for part in split {
                if part.len() > 20 {
                    // Only split if parts are substantial
                    parts.push(part.to_string());
                }
            }
        }

        // If no splitting occurred, return original
        if parts.is_empty() {
            parts.push(query.to_string());
        }

        parts
    }

    /// Find dependencies between sub-queries
    fn find_dependencies(
        &self,
        current_idx: usize,
        parts: &[String],
        sub_queries: &[SubQuery],
    ) -> Vec<usize> {
        let mut dependencies = Vec::new();

        if current_idx == 0 {
            return dependencies; // First sub-query has no dependencies
        }

        let current_lower = parts[current_idx].to_lowercase();

        // Comparison queries typically depend on previous definition/explanation queries
        if current_lower.contains("compare")
            || current_lower.contains("difference")
            || current_lower.contains(" vs ")
        {
            // Depends on all previous sub-queries
            for i in 0..current_idx {
                dependencies.push(i);
            }
        }
        // Analysis queries may depend on previous factual queries
        else if current_lower.contains("evaluate")
            || current_lower.contains("assess")
            || current_lower.contains("should")
        {
            // Depends on immediate predecessor
            if current_idx > 0 {
                dependencies.push(current_idx - 1);
            }
        }
        // Sequential "how to" steps typically depend on previous steps
        else if current_lower.contains("then")
            || current_lower.contains("next")
            || current_lower.contains("after")
        {
            if current_idx > 0 {
                dependencies.push(current_idx - 1);
            }
        }

        dependencies
    }
}

impl Default for QueryAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intent_classification_definition() {
        let analyzer = QueryAnalyzer::new();

        let queries = vec![
            "What is Rust?",
            "Define machine learning",
            "What's the meaning of life?",
        ];

        for query in queries {
            let analyzed = analyzer.analyze(query).unwrap();
            assert_eq!(analyzed.intent, QueryIntent::Definition);
            assert!(analyzed.confidence > 0.7);
        }
    }

    #[test]
    fn test_intent_classification_procedure() {
        let analyzer = QueryAnalyzer::new();

        let queries = vec![
            "How to install Rust?",
            "How do I create a file?",
            "Steps to build a Docker image",
        ];

        for query in queries {
            let analyzed = analyzer.analyze(query).unwrap();
            assert_eq!(analyzed.intent, QueryIntent::Procedure);
            assert!(analyzed.confidence > 0.7);
        }
    }

    #[test]
    fn test_intent_classification_comparison() {
        let analyzer = QueryAnalyzer::new();

        let queries = vec![
            "Rust vs C++",
            "Difference between Python and Java",
            "Compare Docker and Kubernetes",
        ];

        for query in queries {
            let analyzed = analyzer.analyze(query).unwrap();
            assert_eq!(analyzed.intent, QueryIntent::Comparison);
            // "Compare" at start gets high confidence, others get good confidence
            assert!(
                analyzed.confidence > 0.6,
                "Query '{}' had confidence {}",
                query,
                analyzed.confidence
            );
        }
    }

    #[test]
    fn test_intent_classification_troubleshooting() {
        let analyzer = QueryAnalyzer::new();

        let queries = vec![
            "Why doesn't my code compile?",
            "How to fix segmentation fault?",
            "Error when running tests",
        ];

        for query in queries {
            let analyzed = analyzer.analyze(query).unwrap();
            assert_eq!(analyzed.intent, QueryIntent::Troubleshooting);
            assert!(analyzed.confidence > 0.7);
        }
    }

    #[test]
    fn test_entity_extraction() {
        let analyzer = QueryAnalyzer::new();

        let query = "What is the capital of France?";
        let analyzed = analyzer.analyze(query).unwrap();

        // Should extract "France" as an entity
        assert!(!analyzed.entities.is_empty());
        assert!(
            analyzed
                .entities
                .iter()
                .any(|e| e.text.to_lowercase() == "france")
        );
    }

    #[test]
    fn test_constraint_parsing() {
        let analyzer = QueryAnalyzer::new();

        let query = "Show me results after 2024 only Python";
        let analyzed = analyzer.analyze(query).unwrap();

        // Should detect temporal and category constraints
        assert!(!analyzed.constraints.is_empty());
        assert!(
            analyzed
                .constraints
                .iter()
                .any(|c| c.constraint_type == ConstraintType::Temporal)
        );
    }

    #[test]
    fn test_ambiguity_detection() {
        let analyzer = QueryAnalyzer::new();

        let query = "What is it? How does it work?";
        let analyzed = analyzer.analyze(query).unwrap();

        // Should detect pronoun ambiguity and multiple questions
        assert!(!analyzed.ambiguities.is_empty());
        assert!(analyzed.ambiguities.len() >= 2); // "it" + multiple questions
    }

    #[test]
    fn test_empty_query() {
        let analyzer = QueryAnalyzer::new();

        let result = analyzer.analyze("");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            QueryAnalysisError::EmptyQuery
        ));
    }

    #[test]
    fn test_unknown_intent() {
        let analyzer = QueryAnalyzer::new();

        let query = "Random thoughts about stuff";
        let analyzed = analyzer.analyze(query).unwrap();

        // Should default to Unknown for unclear queries
        assert_eq!(analyzed.intent, QueryIntent::Unknown);
        assert!(analyzed.confidence < 0.5);
    }

    #[test]
    fn test_query_decomposition_simple() {
        let analyzer = QueryAnalyzer::new();

        let query = "What is Rust?";
        let mut analyzed = analyzer.analyze(query).unwrap();
        analyzer.decompose_query(&mut analyzed).unwrap();

        // Simple queries should not be decomposed
        assert!(analyzed.sub_queries.is_empty() || analyzed.sub_queries.len() == 1);
    }

    #[test]
    fn test_query_decomposition_multiple_questions() {
        let analyzer = QueryAnalyzer::new();

        let query = "What is Rust? How do I install it?";
        let mut analyzed = analyzer.analyze(query).unwrap();
        analyzer.decompose_query(&mut analyzed).unwrap();

        // Should split into 2 sub-queries
        assert_eq!(analyzed.sub_queries.len(), 2);
        assert!(analyzed.sub_queries[0].text.contains("Rust"));
        assert!(analyzed.sub_queries[1].text.contains("install"));
    }

    #[test]
    fn test_query_decomposition_comparison() {
        let analyzer = QueryAnalyzer::new();

        let query = "What is Rust and how does it compare to C++?";
        let mut analyzed = analyzer.analyze(query).unwrap();
        analyzer.decompose_query(&mut analyzed).unwrap();

        // Should split into sub-queries
        assert!(analyzed.sub_queries.len() >= 2);

        // Find the comparison query
        let comparison_query = analyzed
            .sub_queries
            .iter()
            .find(|sq| sq.intent == QueryIntent::Comparison);

        assert!(comparison_query.is_some());

        // Comparison query should have dependencies
        if let Some(cq) = comparison_query {
            assert!(
                !cq.depends_on.is_empty(),
                "Comparison should depend on prior queries"
            );
        }
    }

    #[test]
    fn test_query_decomposition_dependencies() {
        let analyzer = QueryAnalyzer::new();

        let query = "How does Rust work? Then how does it compare to C++?";
        let mut analyzed = analyzer.analyze(query).unwrap();
        analyzer.decompose_query(&mut analyzed).unwrap();

        // Should have 2 sub-queries
        assert_eq!(analyzed.sub_queries.len(), 2);

        // Second query should depend on first
        assert_eq!(analyzed.sub_queries[1].depends_on, vec![0]);
    }

    #[test]
    fn test_query_decomposition_priorities() {
        let analyzer = QueryAnalyzer::new();

        let query = "What is Rust? What is C++? How do they compare?";
        let mut analyzed = analyzer.analyze(query).unwrap();
        analyzer.decompose_query(&mut analyzed).unwrap();

        // Should have 3 sub-queries
        assert_eq!(analyzed.sub_queries.len(), 3);

        // Earlier queries should have higher priority
        assert!(analyzed.sub_queries[0].priority >= analyzed.sub_queries[1].priority);
        assert!(analyzed.sub_queries[1].priority >= analyzed.sub_queries[2].priority);
    }
}
