// M4 Integration Tests: End-to-End Pipeline
//
// Tests the complete M4 query analysis and context pipeline:
// M4.D (QueryAnalyzer) → M4.E (RelevanceSearch) → M4.F (ContextInjection) → M4.A (MetadataScheduling)
//
// Validates:
// - Query analysis feeds into search strategies
// - Search results inform context providers
// - Context injections respect priority and position
// - Metadata scheduling routes based on analyzed queries
// - Complete pipeline executes in <50ms

use lightbulb::engine::{
    ConstraintValidator,
    ContextInjection,
    // M4.F: Context Injection
    ContextManager,
    ContextProvider,
    CrateApiProvider,

    CrossEncoderReranker,
    // M4.E: Relevance Search
    Document,
    DocumentMetadata,
    EthicalFlag,
    HybridSearchPipeline,
    HydeSearch,
    InjectionPosition,
    MetadataFilter,

    MetadataRoutingPolicy,
    // M4.A: Metadata Scheduling
    MetadataScheduler,
    // M4.D: Query Analysis
    QueryAnalyzer,
    QueryIntent,

    RequestMetadata,
    RequestPriority,
    RequestTag,
    SchedulingPipelineId,
    SearchConfig,
    SearchStrategy,
    SemanticSearch,
    StaticContextProvider,
};

use std::sync::Arc;
use std::time::Instant;

#[test]
fn test_m4d_to_m4e_integration() {
    // M4.D: Analyze query
    let analyzer = QueryAnalyzer::new();
    let query = "Compare Rust ownership vs C++ RAII";
    let analyzed = analyzer.analyze(query).unwrap();

    // Verify intent analysis (should be Comparison due to "compare" keyword)
    assert_eq!(analyzed.intent, QueryIntent::Comparison);

    // M4.E: Use analysis to guide search
    let search = SemanticSearch::new(128);

    // Create test documents
    let doc1 = Document::new(
        "Rust ownership ensures memory safety at compile time".to_string(),
        DocumentMetadata::new(
            "doc1".to_string(),
            "documentation".to_string(),
            "conceptual".to_string(),
        ),
    )
    .with_embedding(vec![1.0; 128]);

    let doc2 = Document::new(
        "C++ RAII uses destructors for resource management".to_string(),
        DocumentMetadata::new(
            "doc2".to_string(),
            "documentation".to_string(),
            "conceptual".to_string(),
        ),
    )
    .with_embedding(vec![0.9; 128]);

    let documents = vec![doc1, doc2];
    let config = SearchConfig::default();

    let results = search.search(query, &documents, &config).unwrap();

    // Should find relevant documents
    assert!(!results.is_empty());
    assert_eq!(results[0].strategy, "semantic_search");
}

#[test]
fn test_m4d_to_m4f_integration() {
    // M4.D: Analyze query
    let analyzer = QueryAnalyzer::new();
    let query = "How do I use tokio for async programming?";
    let analyzed = analyzer.analyze(query).unwrap();

    // Verify procedural intent
    assert_eq!(analyzed.intent, QueryIntent::Procedure);

    // M4.F: Use analysis to activate context providers
    let mut manager = ContextManager::new();

    // Add crate API provider
    let crate_provider = CrateApiProvider::new().add_crate_docs(
        "tokio".to_string(),
        "Tokio is an async runtime for Rust. Use tokio::main macro to create async main."
            .to_string(),
    );
    manager.register_provider(Arc::new(crate_provider));

    // Provide contexts
    let results = manager.provide_contexts(&analyzed, query);

    assert_eq!(results.len(), 1);
    assert!(results[0].success);
    assert!(!results[0].injections.is_empty());
    assert!(results[0].injections[0].content.contains("tokio"));
    assert!(results[0].execution_time_ms < 50); // Fast execution
}

#[test]
fn test_m4d_to_m4a_integration() {
    // M4.D: Analyze query
    let analyzer = QueryAnalyzer::new();
    let query = "Compare Rust and Python for web development";
    let analyzed = analyzer.analyze(query).unwrap();

    // M4.A: Create metadata from analysis
    let metadata = RequestMetadata::from_query("req1".to_string(), &analyzed);

    // Should have comparison and reasoning tags
    assert!(metadata.tags.contains(&RequestTag::Comparison));
    assert!(metadata.tags.contains(&RequestTag::Reasoning));

    // Create routing policy
    let policy = MetadataRoutingPolicy::new(
        "test_policy".to_string(),
        SchedulingPipelineId("default".to_string()),
    )
    .add_rule(
        vec![RequestTag::Comparison],
        SchedulingPipelineId("comparison_pipeline".to_string()),
        100,
    );

    let validator = ConstraintValidator::new();
    let scheduler = MetadataScheduler::new(policy, validator);

    // Schedule request
    let decision = scheduler.schedule(&metadata).unwrap();

    assert_eq!(decision.pipeline_id.0, "comparison_pipeline");
    assert!(decision.confidence > 0.5);
    assert!(decision.matched_tags.contains(&RequestTag::Comparison));
}

#[test]
fn test_full_m4_pipeline() {
    let start = Instant::now();

    // Step 1: M4.D Query Analysis
    let analyzer = QueryAnalyzer::new();
    let query = "What is Rust and how does it compare to C++?";
    let analyzed = analyzer.analyze(query).unwrap();

    assert_eq!(analyzed.intent, QueryIntent::Comparison);

    // Step 2: M4.E Search with intent-aware strategy
    let search = if analyzed.intent == QueryIntent::Comparison {
        // Use HyDE for comparison queries
        Box::new(HydeSearch::new(128)) as Box<dyn SearchStrategy>
    } else {
        Box::new(SemanticSearch::new(128)) as Box<dyn SearchStrategy>
    };

    let doc = Document::new(
        "Rust is a systems programming language focused on safety and performance".to_string(),
        DocumentMetadata::new(
            "doc1".to_string(),
            "documentation".to_string(),
            "conceptual".to_string(),
        ),
    )
    .with_embedding(vec![1.0; 128]);

    let documents = vec![doc];
    let config = SearchConfig::default();
    let search_results = search.search(query, &documents, &config).unwrap();

    assert!(!search_results.is_empty());

    // Step 3: M4.F Context Injection
    let mut manager = ContextManager::new();

    let rust_context = ContextInjection::new(
        "Rust is memory-safe without garbage collection".to_string(),
        InjectionPosition::BeforePrompt,
        "rust_docs".to_string(),
    )
    .with_priority(80);

    let provider = StaticContextProvider::new("static".to_string())
        .add_context("rust".to_string(), rust_context);

    manager.register_provider(Arc::new(provider));

    let context_results = manager.provide_contexts(&analyzed, query);
    assert!(!context_results.is_empty());

    let merged = manager.merge_contexts(context_results).unwrap();
    assert!(merged.contains_key(&InjectionPosition::BeforePrompt));

    // Step 4: M4.A Metadata Scheduling
    let metadata = RequestMetadata::from_query("req1".to_string(), &analyzed)
        .with_priority(RequestPriority::High);

    let policy = MetadataRoutingPolicy::new(
        "production".to_string(),
        SchedulingPipelineId("default".to_string()),
    )
    .add_rule(
        vec![RequestTag::Comparison, RequestTag::Reasoning],
        SchedulingPipelineId("advanced_reasoning".to_string()),
        100,
    );

    let validator = ConstraintValidator::new();
    let scheduler = MetadataScheduler::new(policy, validator);

    let decision = scheduler.schedule(&metadata).unwrap();
    assert_eq!(decision.pipeline_id.0, "advanced_reasoning");

    // Verify total pipeline time
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_millis() < 100,
        "Pipeline took {}ms (target: <100ms)",
        elapsed.as_millis()
    );
}

#[test]
fn test_m4_pipeline_with_hybrid_search() {
    // M4.D: Analyze
    let analyzer = QueryAnalyzer::new();
    let query = "Explain Rust ownership system";
    let analyzed = analyzer.analyze(query).unwrap();

    assert_eq!(analyzed.intent, QueryIntent::Explanation);

    // M4.E: Hybrid search pipeline
    let semantic = SemanticSearch::new(128);
    let reranker = CrossEncoderReranker::new(512);

    let pipeline = HybridSearchPipeline::new(Box::new(semantic))
        .with_reranker(Box::new(reranker))
        .with_retrieval_top_k(10);

    let doc = Document::new(
        "Rust ownership ensures memory safety through borrowing and lifetimes".to_string(),
        DocumentMetadata::new(
            "doc1".to_string(),
            "tutorial".to_string(),
            "procedural".to_string(),
        ),
    )
    .with_embedding(vec![1.0; 128]);

    let documents = vec![doc];
    let config = SearchConfig::default();

    let results = pipeline.search(query, &documents, &config).unwrap();

    assert!(!results.is_empty());
    assert!(results[0].strategy.contains("semantic_search"));
    assert!(results[0].strategy.contains("cross_encoder_rerank"));
    assert!(results[0].sub_scores.contains_key("retrieval_score"));
    assert!(results[0].sub_scores.contains_key("rerank_score"));
}

#[test]
fn test_m4_pipeline_with_metadata_filtering() {
    // M4.D: Analyze
    let analyzer = QueryAnalyzer::new();
    let query = "Show me recent Rust tutorials";
    let analyzed = analyzer.analyze(query).unwrap();

    // M4.E: Search with metadata filter
    let search = SemanticSearch::new(128);

    let doc1 = Document::new(
        "Rust tutorial from 2024".to_string(),
        DocumentMetadata {
            id: "doc1".to_string(),
            doc_type: "tutorial".to_string(),
            info_type: "procedural".to_string(),
            created_at: Some(1704067200000), // 2024
            modified_at: None,
            source: "blog".to_string(),
            custom: std::collections::HashMap::new(),
        },
    )
    .with_embedding(vec![1.0; 128]);

    let doc2 = Document::new(
        "Old Rust tutorial from 2020".to_string(),
        DocumentMetadata {
            id: "doc2".to_string(),
            doc_type: "tutorial".to_string(),
            info_type: "procedural".to_string(),
            created_at: Some(1577836800000), // 2020
            modified_at: None,
            source: "blog".to_string(),
            custom: std::collections::HashMap::new(),
        },
    )
    .with_embedding(vec![0.9; 128]);

    let documents = vec![doc1, doc2];

    // Filter for recent documents (2024+)
    let mut filter = MetadataFilter::new();
    filter.min_created_at = Some(1704067200000);
    filter.doc_types.push("tutorial".to_string());

    let config = SearchConfig::default().with_metadata_filter(filter);

    let results = search.search(query, &documents, &config).unwrap();

    // ASSERT THE COLLECTION BEFORE ASSERTING OVER IT.
    //
    // The loop below is the whole test. Over an empty `results` it runs zero
    // times and this passes, so it cannot tell "only the recent document was
    // returned" from "nothing was returned" — and the second is what a
    // too-strict filter produces, which is the bug this test exists to catch.
    assert!(
        !results.is_empty(),
        "the 2024 tutorial must survive the filter, or the loop below asserts nothing"
    );
    // Should only return recent document
    for result in &results {
        assert!(result.document.metadata.created_at.unwrap() >= 1704067200000);
    }
}

#[test]
fn test_m4_pipeline_ethical_constraints() {
    // M4.D: Analyze potentially harmful query
    let analyzer = QueryAnalyzer::new();
    let query = "How to create malware?";
    let analyzed = analyzer.analyze(query).unwrap();

    // M4.A: Create metadata with ethical flag
    let metadata = RequestMetadata::from_query("req1".to_string(), &analyzed)
        .add_ethical_flag(EthicalFlag::Harmful);

    let policy = MetadataRoutingPolicy::new(
        "test".to_string(),
        SchedulingPipelineId("default".to_string()),
    );

    let validator = ConstraintValidator::new(); // Ethical enforcement enabled by default
    let scheduler = MetadataScheduler::new(policy, validator);

    // Should reject harmful content
    let result = scheduler.schedule(&metadata);
    assert!(result.is_err());
}

#[test]
fn test_m4_pipeline_context_priority() {
    // M4.F: Multiple providers with different priorities
    let mut manager = ContextManager::new();

    let high_priority = ContextInjection::new(
        "High priority context".to_string(),
        InjectionPosition::BeforePrompt,
        "high".to_string(),
    )
    .with_priority(100);

    let low_priority = ContextInjection::new(
        "Low priority context".to_string(),
        InjectionPosition::BeforePrompt,
        "low".to_string(),
    )
    .with_priority(10);

    let provider1 = StaticContextProvider::new("provider1".to_string())
        .add_context("test".to_string(), high_priority);

    let provider2 = StaticContextProvider::new("provider2".to_string())
        .add_context("test".to_string(), low_priority);

    manager.register_provider(Arc::new(provider1));
    manager.register_provider(Arc::new(provider2));

    let analyzer = QueryAnalyzer::new();
    let analyzed = analyzer.analyze("test query").unwrap();

    let results = manager.provide_contexts(&analyzed, "test query");
    let merged = manager.merge_contexts(results).unwrap();

    let before_prompt = merged.get(&InjectionPosition::BeforePrompt).unwrap();

    // High priority should come first
    assert_eq!(before_prompt[0].priority, 100);
    assert_eq!(before_prompt[1].priority, 10);
}

#[test]
fn test_m4_pipeline_performance() {
    // Verify entire pipeline executes quickly
    let iterations = 100;
    let mut total_time = std::time::Duration::new(0, 0);

    for i in 0..iterations {
        let start = Instant::now();

        // M4.D
        let analyzer = QueryAnalyzer::new();
        let query = format!("What is Rust? (iteration {})", i);
        let analyzed = analyzer.analyze(&query).unwrap();

        // M4.E
        let search = SemanticSearch::new(128);
        let doc = Document::new(
            "Rust content".to_string(),
            DocumentMetadata::new("doc".to_string(), "doc".to_string(), "factual".to_string()),
        )
        .with_embedding(vec![1.0; 128]);
        let documents = vec![doc];
        let config = SearchConfig::default();
        let _results = search.search(&query, &documents, &config).unwrap();

        // M4.F
        let manager = ContextManager::new();
        let _contexts = manager.provide_contexts(&analyzed, &query);

        // M4.A
        let metadata = RequestMetadata::from_query(format!("req{}", i), &analyzed);
        let policy = MetadataRoutingPolicy::new(
            "test".to_string(),
            SchedulingPipelineId("default".to_string()),
        );
        let validator = ConstraintValidator::new();
        let scheduler = MetadataScheduler::new(policy, validator);
        let _decision = scheduler.schedule(&metadata).unwrap();

        total_time += start.elapsed();
    }

    let avg_time = total_time / iterations;
    println!("Average M4 pipeline time: {:?}", avg_time);

    // Should average well under 50ms per request
    assert!(
        avg_time.as_millis() < 50,
        "Average time {}ms exceeds 50ms target",
        avg_time.as_millis()
    );
}
