// SPDX-License-Identifier: MIT OR Apache-2.0
//! Re-exports of the context, query and knowledge modules at `crate::engine::*`.

pub use super::adaptive_selection::{
    ProviderMetrics, ProviderSelector, RegisteredProvider, SelectionConfig, SelectionStrategy,
};
pub use super::consistency_checking::{
    CheckerConfig, CheckerStats, Conflict, ConflictType, ConsistencyChecker, ConsistencyError,
    ValidationResult,
};
pub use super::context_compression::{
    CompressionConfig, CompressionResult, CompressionStrategy, ContextCompressor,
};
pub use super::context_injection::{
    ContextError, ContextInjection, ContextManager, ContextProvider, CrateApiProvider,
    InjectionPosition, ProviderConfig, ProviderResult, StaticContextProvider,
};
pub use super::conversation_history::{
    ConversationConfig, ConversationHistory, ConversationTurn, Role,
};
pub use super::decomposition::{
    ComplexityLevel, Decomposition, DecompositionConfig, DecompositionEngine, DecompositionError,
    DecompositionHistory, DecompositionStats, DecompositionStrategy, Problem, ProblemId,
    SubProblem,
};
pub use super::knowledge_base::{
    ConvergenceDetector, EvictionRecord, Fact, FactCategory, FactKey, KnowledgeBase,
    KnowledgeBaseConfig, KnowledgeBaseError, KnowledgeBaseStats,
};
pub use super::query_analysis::{
    AnalyzedQuery, ComparisonOperator, Constraint, ConstraintType, Entity, EntityType,
    QueryAnalysisError, QueryAnalyzer, QueryIntent, SubQuery,
};
pub use super::relevance_search::{
    CrossEncoderReranker, Document, DocumentMetadata, HybridSearchPipeline, HydeSearch,
    MetadataFilter, SearchConfig, SearchError, SearchResult, SearchStrategy, SemanticSearch,
};
