//! Inference Engine Components
//!
//! This module contains the core scheduling and execution infrastructure
//! for continuous batching and request management.

pub mod adaptive_selection;
pub mod consistency_checking;
pub mod context_compression;
pub mod context_injection;
pub mod conversation_history;
pub mod decomposition;
pub mod knowledge_base;
pub mod memory_aware_scheduler;
pub mod metadata_scheduling;
pub mod mixed_precision;
pub mod model_runner;
pub mod moe_router;
pub mod performance_optimization;
pub mod pipeline;
pub mod production_providers;
pub mod query_analysis;
pub mod reasoning_controls;
pub mod relevance_search;
pub mod slot_monitor;
pub mod slot_pool;
pub mod speculative;
pub mod state_persistence;
pub mod streaming_context;
pub mod tool_call;

pub use adaptive_selection::{
    ProviderMetrics, ProviderSelector, RegisteredProvider, SelectionConfig, SelectionStrategy,
};
pub use consistency_checking::{
    CheckerConfig, CheckerStats, Conflict, ConflictType, ConsistencyChecker, ConsistencyError,
    ValidationResult,
};
pub use context_compression::{
    CompressionConfig, CompressionResult, CompressionStrategy, ContextCompressor,
};
pub use context_injection::{
    ContextError, ContextInjection, ContextManager, ContextProvider, CrateApiProvider,
    InjectionPosition, ProviderConfig, ProviderResult, StaticContextProvider,
};
pub use conversation_history::{ConversationConfig, ConversationHistory, ConversationTurn, Role};
pub use decomposition::{
    ComplexityLevel, Decomposition, DecompositionConfig, DecompositionEngine, DecompositionError,
    DecompositionHistory, DecompositionStats, DecompositionStrategy, Problem, ProblemId,
    SubProblem,
};
pub use knowledge_base::{
    ConvergenceDetector, EvictionRecord, Fact, FactCategory, FactKey, KnowledgeBase,
    KnowledgeBaseConfig, KnowledgeBaseError, KnowledgeBaseStats,
};
pub use memory_aware_scheduler::{MemoryAwareConfig, MemoryAwareScheduler, MemoryStats, Priority};
pub use metadata_scheduling::{
    ConstraintError, ConstraintValidator, EthicalFlag, MetadataScheduler,
    PipelineId as SchedulingPipelineId, Priority as RequestPriority, RequestMetadata, RequestTag,
    RoutingDecision, RoutingPolicy as MetadataRoutingPolicy,
};
pub use model_runner::{InferenceJob, InferenceRequestSender, ModelRunner};
pub use moe_router::{
    ExpertId, MoeConfig, MoeRouter, RoutingError, RoutingPolicy, RoutingResult, RoutingStats,
    TokenAssignment, TokenIndex,
};
pub use performance_optimization::{
    AsyncContextManager, CacheConfig, CacheStats, ConnectionPool, EmbeddingModel,
    EmbeddingModelConfig, LazyEmbeddingModel, PoolStats,
};
pub use pipeline::{
    CacheRef, Pipeline, PipelineBuilder, PipelineConfig, PipelineError, PipelineId, PipelineStats,
    Stage, StageData, StageId, StageResult, StageState, StageType,
};
pub use production_providers::{
    EmbeddingConfig, EmbeddingProvider, FileChange, FileChangeType, FileWatcherConfig,
    FileWatcherProvider, SearchResult as WebSearchResult, WebSearchConfig, WebSearchProvider,
};
pub use query_analysis::{
    AnalyzedQuery, ComparisonOperator, Constraint, ConstraintType, Entity, EntityType,
    QueryAnalysisError, QueryAnalyzer, QueryIntent, SubQuery,
};
pub use relevance_search::{
    CrossEncoderReranker, Document, DocumentMetadata, HybridSearchPipeline, HydeSearch,
    MetadataFilter, SearchConfig, SearchError, SearchResult, SearchStrategy, SemanticSearch,
};
pub use slot_monitor::{AdjustmentConfig, MemoryStatistics, SlotPoolMonitor};
pub use slot_pool::{
    CompletedToken, Request as SlotPoolRequest, RequestId, SlotId, SlotPool, SlotPoolError,
    SlotPoolStats, SlotState,
};
pub use speculative::{SpeculativeConfig, SpeculativeDecoder, SpeculativeModel, SpeculativeStats};
pub use state_persistence::{
    CheckpointId, CheckpointManager, CheckpointMetadata, InferenceCheckpoint,
    KnowledgeBaseSnapshot, KvCacheSnapshot, PipelineSnapshot, create_checkpoint,
    restore_kb_from_checkpoint,
};
pub use streaming_context::{
    CodeCompletionStreamProvider, ContextStream, StreamConfig, StreamingContextProvider,
    WebSearchStreamProvider,
};

// Legacy types for backward compatibility (will be migrated to SlotPool)
use anyhow::Result;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;

/// Legacy Request type (to be replaced by SlotPoolRequest)
#[derive(Debug, Clone)]
pub struct Request {
    pub id: String,
    pub prompt: String,
    pub max_new_tokens: usize,
}

/// Legacy RequestState enum
#[derive(Debug, Clone)]
pub enum RequestState {
    Pending,
    Decoding,
    /// Generation paused, waiting for tool result. KV cache is preserved.
    /// Call `inject_tool_result()` to resume generation with the result.
    AwaitingToolResult {
        tool_name: String,
        tool_args: String,
        /// Cache position at which the tool call was detected.
        cache_position: usize,
        /// Attention snapshot at the moment of the tool call (if captured).
        attention_snapshot: Option<tool_call::AttentionSnapshot>,
    },
    Completed,
}

impl PartialEq for RequestState {
    fn eq(&self, other: &Self) -> bool {
        // Compare variant discriminants only (ignore data fields)
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }
}

impl Eq for RequestState {}

impl RequestState {
    /// Check if this state is AwaitingToolResult (any variant data).
    pub fn is_awaiting_tool_result(&self) -> bool {
        matches!(self, RequestState::AwaitingToolResult { .. })
    }
}

/// Legacy RequestContext (to be replaced by SlotPool's internal state)
#[derive(Debug, Clone)]
pub struct RequestContext {
    pub request: Request,
    pub state: RequestState,
    pub tokens_generated: usize,
    pub position: usize,
    pub cache_index: Option<usize>,
    pub generated_tokens: Vec<u32>,
    /// Sampling temperature carried from the originating `InferenceJob`.
    /// `0.0` means greedy. Defaults to `0.0` so existing constructors are
    /// unchanged and deterministic.
    pub temperature: f64,
}

impl RequestContext {
    pub fn new(request: Request) -> Self {
        Self {
            request,
            state: RequestState::Pending,
            tokens_generated: 0,
            position: 0,
            cache_index: None,
            generated_tokens: Vec::new(),
            temperature: 0.0,
        }
    }

    pub fn request_temperature(&self) -> f64 {
        self.temperature
    }

    pub fn assign_cache_index(&mut self, index: usize) {
        self.cache_index = Some(index);
    }

    pub fn start_decoding(&mut self) {
        self.state = RequestState::Decoding;
    }

    pub fn record_token(&mut self) {
        self.tokens_generated += 1;
        self.position += 1;
    }

    pub fn complete(&mut self) {
        self.state = RequestState::Completed;
    }

    pub fn should_continue(&self) -> bool {
        matches!(self.state, RequestState::Decoding)
            && self.tokens_generated < self.request.max_new_tokens
    }

    /// Transition to AwaitingToolResult state, preserving KV cache position.
    pub fn await_tool_result(
        &mut self,
        tool_name: String,
        tool_args: String,
        attention_snapshot: Option<tool_call::AttentionSnapshot>,
    ) {
        let cache_position = self.position;
        self.state = RequestState::AwaitingToolResult {
            tool_name,
            tool_args,
            cache_position,
            attention_snapshot,
        };
    }

    /// Resume decoding after tool result injection.
    pub fn resume_decoding(&mut self) {
        self.state = RequestState::Decoding;
    }
}

/// Legacy RequestQueue
#[derive(Clone)]
pub struct RequestQueue {
    pending: Arc<Mutex<VecDeque<RequestContext>>>,
}

impl RequestQueue {
    pub fn new() -> Self {
        Self {
            pending: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    pub fn submit(&self, req: Request) -> Result<String> {
        let id = req.id.clone();
        let ctx = RequestContext::new(req);
        self.pending.lock().push_back(ctx);
        Ok(id)
    }

    pub fn pop(&self) -> Option<RequestContext> {
        self.pending.lock().pop_front()
    }

    pub fn len(&self) -> usize {
        self.pending.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.lock().is_empty()
    }
}

impl Default for RequestQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Legacy BatchConfig
#[derive(Debug, Clone, Copy)]
pub struct BatchConfig {
    pub max_batch_size: usize,
    pub max_batch_tokens: usize,
}

impl BatchConfig {
    pub fn new(max_batch_size: usize, max_batch_tokens: usize) -> Self {
        Self {
            max_batch_size,
            max_batch_tokens,
        }
    }
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_batch_tokens: 2048,
        }
    }
}

/// Legacy BatchAssembler
pub struct BatchAssembler {
    config: BatchConfig,
}

impl BatchAssembler {
    pub fn new(config: BatchConfig) -> Self {
        Self { config }
    }

    pub fn assemble_batch(&self, queue: &RequestQueue) -> Vec<RequestContext> {
        let mut batch = Vec::new();
        let mut total_tokens = 0;
        let mut overflow = Vec::new();

        while batch.len() < self.config.max_batch_size {
            let ctx = match queue.pop() {
                Some(ctx) => ctx,
                None => break,
            };

            let request_tokens = ctx.request.max_new_tokens;

            if total_tokens + request_tokens > self.config.max_batch_tokens {
                overflow.push(ctx);
                break;
            }

            total_tokens += request_tokens;
            batch.push(ctx);
        }

        for ctx in overflow {
            let req = ctx.request;
            let _ = queue.submit(req);
        }

        batch
    }
}

// Re-export cache types for direct use throughout the codebase
pub use crate::cache::parallel_cache_builder::{
    IndicesAndMask, ParallelCacheBuilder, ParallelKvCache,
};
