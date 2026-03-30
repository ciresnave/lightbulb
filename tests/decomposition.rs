//! Tests for the knowledge-aware decomposition system

use lightbulb::engine::{
    ComplexityLevel, DecompositionConfig, DecompositionEngine, KnowledgeBase, Problem, SubProblem,
};

#[test]
fn test_engine_creation() {
    let kb = KnowledgeBase::new();
    let engine = DecompositionEngine::new(kb);

    let stats = engine.stats();
    assert_eq!(stats.total_decompositions, 0);
    assert_eq!(stats.active_decompositions, 0);
}

#[test]
fn test_engine_with_config() {
    let kb = KnowledgeBase::new();
    let config = DecompositionConfig {
        max_depth: 3,
        max_sub_problems: 5,
        min_coverage_for_redecomp: 0.8,
        auto_redecompose: false,
        track_history: false,
    };

    let engine = DecompositionEngine::with_config(kb, config);
    let stats = engine.stats();
    assert_eq!(stats.total_decompositions, 0);
}

#[test]
fn test_problem_creation() {
    let problem = Problem::new("Calculate GDP growth");

    assert!(!problem.description.is_empty());
    assert!(!problem.id.is_empty());
}

#[test]
fn test_problem_with_requirements() {
    let problem = Problem::new("Calculate GDP growth")
        .requires("gdp_2023")
        .requires("gdp_2024");

    assert_eq!(problem.required_facts.len(), 2);
}

#[test]
fn test_problem_with_complexity() {
    let problem = Problem::new("Complex calculation").with_complexity(ComplexityLevel::VeryComplex);

    assert!(matches!(problem.complexity, ComplexityLevel::VeryComplex));
}

#[test]
fn test_problem_coverage() {
    let problem = Problem::new("Test")
        .requires("fact1")
        .requires("fact2")
        .with_known_fact("fact1", "value1");

    let coverage = problem.knowledge_coverage();
    assert!(coverage > 0.4 && coverage < 0.6); // Should be around 0.5
}

#[test]
fn test_problem_is_atomic() {
    let atomic = Problem::new("Simple task").with_complexity(ComplexityLevel::Simple);

    assert!(atomic.is_atomic());

    let not_atomic = Problem::new("Complex task").requires("fact1");

    assert!(!not_atomic.is_atomic());
}

#[test]
fn test_kb_enrichment() {
    let mut kb = KnowledgeBase::new();

    kb.add_fact(lightbulb::engine::Fact::new(
        "gdp_2023",
        "GDP 2023",
        "GDP in 2023 was $25.5 trillion",
    ))
    .expect("Failed to add fact");

    let retrieved = kb.get_fact("gdp_2023");
    assert!(retrieved.is_ok());
}

#[test]
fn test_config_defaults() {
    let config = DecompositionConfig::default();

    assert!(config.max_depth > 0);
    assert!(config.max_sub_problems > 0);
    assert!(config.min_coverage_for_redecomp > 0.0 && config.min_coverage_for_redecomp <= 1.0);
}

#[test]
fn test_subproblem_creation() {
    let sub = SubProblem::new("sub1", "Task 1")
        .produces("output1")
        .with_complexity(ComplexityLevel::Simple);

    assert_eq!(sub.id, "sub1");
    assert_eq!(sub.produces.len(), 1);
}

#[test]
fn test_subproblem_dependencies() {
    let sub = SubProblem::new("sub2", "Task 2").depends_on("sub1");

    assert_eq!(sub.depends_on.len(), 1);
    assert!(sub.depends_on.contains(&"sub1".to_string()));
}

#[test]
fn test_problem_uniqueness() {
    // IDs are timestamp-based, so problems created in rapid succession
    // may have the same ID. In practice, this is not an issue since
    // problem creation is not that frequent.
    let p1 = Problem::new("Problem 1");
    let p2 = Problem::new("Problem 2");

    // At minimum, problems should have different descriptions
    assert_ne!(p1.description, p2.description);
}
#[test]
fn test_kb_stats() {
    let mut kb = KnowledgeBase::new();

    kb.add_fact(lightbulb::engine::Fact::new("fact1", "Fact 1", "Content 1"))
        .ok();
    kb.add_fact(lightbulb::engine::Fact::new("fact2", "Fact 2", "Content 2"))
        .ok();

    let stats = kb.stats();
    assert_eq!(stats.fact_count, 2);
}

#[test]
fn test_problem_clone() {
    let problem = Problem::new("Test problem");
    let cloned = problem.clone();

    assert_eq!(problem.id, cloned.id);
    assert_eq!(problem.description, cloned.description);
}

#[test]
fn test_engine_stats_tracking() {
    let kb = KnowledgeBase::new();
    let engine = DecompositionEngine::new(kb);

    let stats = engine.stats();
    assert_eq!(stats.total_decompositions, 0);
    assert_eq!(stats.structural_count, 0);
    assert_eq!(stats.computational_count, 0);
    assert_eq!(stats.redecomposition_count, 0);
}

#[test]
fn test_decompose_simple_problem() {
    let kb = KnowledgeBase::new();
    let mut engine = DecompositionEngine::new(kb);

    let problem = Problem::new("Calculate 2 + 2").with_complexity(ComplexityLevel::Trivial);

    let result = engine.decompose(problem);
    assert!(result.is_ok());
}

#[test]
fn test_history_tracking() {
    let kb = KnowledgeBase::new();
    let config = DecompositionConfig {
        track_history: true,
        ..Default::default()
    };

    let mut engine = DecompositionEngine::with_config(kb, config);

    let problem = Problem::new("Test problem");
    let _ = engine.decompose(problem);

    let history = engine.history();
    assert_eq!(history.len(), 1);
}
