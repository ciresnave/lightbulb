# API Exposure Analysis - Lightbulb Features

**Date**: October 30, 2025  
**Purpose**: Ensure all user/LLM-accessible features have proper API interfaces  
**Status**: ANALYSIS COMPLETE, IMPLEMENTATION NEEDED

## Feature Classification

### ✅ Internal Only (No API Needed)
These features are automatic optimizations or development tools:

1. **Model Loading** (M0) - Internal: Automatic on startup
2. **Continuous Batching** (M1) - Internal: Automatic scheduling optimization
3. **Paged KV Cache** (M1) - Internal: Automatic memory management
4. **Prefix KV Caching** (M2) - Internal: Automatic optimization
5. **FlashAttention** (M3.4) - Internal: Automatic when CUDA available
6. **Kernel Fusion** (M3.3) - Internal: Automatic optimization
7. **Dynamic Batch Sizing** (M1.5) - Internal: Automatic adjustment
8. **Chunk Size Optimization** (M1.5) - Internal: Automatic tuning
9. **Distributed KV Cache** (M3.6) - Internal: Automatic coordination
10. **Tensor Name Mapping** (M3.7) - Internal: Automatic architecture detection
11. **MoE Routing** (M4) - Internal: Automatic routing decisions
12. **Mixed-Precision Profiling** (M5) - Internal: Automatic profiling at startup
13. **Load Testing Infrastructure** (M3.5) - Internal: Dev/ops tool
14. **Correctness Validation** (M3.5) - Internal: Testing framework
15. **Regression Detection** (M3.5) - Internal: CI/CD tool

### 🔴 USER API NEEDED (Missing)
Features users should configure/monitor but currently lack APIs:

#### Monitoring & Health
16. **Observability/Metrics** (M1) - MISSING: Health check API, metrics endpoint
    - Needed: `GET /health`, `GET /metrics` (Prometheus format)
    - Returns: TTFT, tokens/sec, active requests, KV bytes, cache stats

17. **Hardware Detection** (M1.5) - MISSING: System info API
    - Needed: `GET /system/info`, `GET /system/capabilities`
    - Returns: CPU/GPU specs, memory, detected backends

18. **Architecture Detection** (M3.7) - MISSING: Model info API
    - Needed: `GET /model/info`, `GET /model/architecture`
    - Returns: Architecture type, layer count, context length

#### Configuration
19. **StreamingLLM Policy** (M2) - MISSING: Cache config API
    - Needed: `POST /config/cache`, `GET /config/cache`
    - Settings: Window size, attention sinks, eviction policy

20. **Intelligent Cache Management** (M2) - MISSING: Cache control API
    - Needed: `POST /cache/policy`, `GET /cache/stats`, `POST /cache/clear`
    - Settings: Eviction policy selection, aggregation weights

21. **Speculative Decoding** (M3) - MISSING: Speculation config API
    - Needed: `POST /config/speculation`, `GET /config/speculation`
    - Settings: Enable/disable, draft model, acceptance threshold

22. **Multi-GPU Configuration** (M3.6) - MISSING: GPU allocation API
    - Needed: `POST /config/gpu`, `GET /gpu/status`
    - Settings: Parallelism strategy (tensor/pipeline), layer distribution

23. **KV Cache Compression** (M5) - MISSING: Compression config API
    - Needed: `POST /config/compression`, `GET /compression/stats`
    - Settings: Strategy selection (H2O, KIVI, R-KV), compression ratio

24. **Pruning Utilities** (M5) - MISSING: Offline pruning tool
    - Needed: CLI command `lightbulb prune --model <path> --ratio 0.25 --strategy wanda`
    - Not HTTP API (offline operation)

25. **Reasoning Controls** (M5) - MISSING: Budget/policy API
    - Needed: `POST /reasoning/budget`, `POST /reasoning/policy`, `GET /reasoning/stats`
    - Settings: Max chains, termination policy, overthinking detection

### 🟡 LLM TOOL API NEEDED (Missing)
Features LLMs should access via function calling:

#### Query & Analysis
26. **Query Analysis** (M4.D) - MISSING: Tool definition
    - Tool: `analyze_query(query: str) -> AnalyzedQuery`
    - Returns: Intent, entities, constraints, sub-queries

27. **Relevance Search** (M4.E) - MISSING: Tool definition
    - Tool: `search_documents(query: str, top_k: int) -> Vec<Document>`
    - Returns: Ranked documents with relevance scores

28. **Context Injection** (M4.F) - MISSING: Tool definition
    - Tool: `inject_context(provider: str, prompt: str) -> Context`
    - Returns: Injected context from provider (crate docs, notifications, files)

#### Reasoning & Planning
29. **Multi-Stage Pipelines** (M4) - MISSING: Tool definition
    - Tool: `create_pipeline(stages: Vec<Stage>) -> PipelineId`
    - Tool: `execute_pipeline(id: PipelineId) -> PipelineResult`
    - Returns: Pipeline execution results with stage outputs

30. **Metadata Scheduling** (M4) - MISSING: Tool definition
    - Tool: `set_request_metadata(priority: Priority, tags: Vec<Tag>) -> RequestId`
    - Returns: Request ID with metadata applied

31. **State Persistence** (M4.B) - MISSING: Tool definition
    - Tool: `save_state(label: str) -> StateId`
    - Tool: `restore_state(id: StateId) -> State`
    - Tool: `branch_state(id: StateId, label: str) -> StateId`
    - Returns: State snapshots for rollback/branching

32. **Convergence Detection** (M4) - MISSING: Tool definition
    - Tool: `check_convergence(window: int) -> ConvergenceStatus`
    - Returns: Converged/NotConverged with fact saturation metrics

33. **Problem Decomposition** (M4.C) - MISSING: Tool definition
    - Tool: `decompose_problem(description: str) -> DecompositionResult`
    - Tool: `get_subproblem(id: ProblemId) -> SubProblem`
    - Tool: `solve_subproblem(id: ProblemId, solution: str) -> Result`
    - Returns: Problem tree with dependencies and solutions

#### Knowledge Management
34. **Knowledge Base** (M4.5) - MISSING: Tool definition
    - Tool: `add_fact(content: str, category: Category) -> FactKey`
    - Tool: `query_facts(category: Option<Category>) -> Vec<Fact>`
    - Tool: `get_kb_stats() -> KBStats`
    - Returns: Facts with confidence scores and sources

35. **Consistency Checking** (M4) - MISSING: Tool definition
    - Tool: `validate_fact(fact: str, existing: Vec<str>) -> ValidationResult`
    - Returns: Valid/invalid with detected conflicts

#### Context Management
36. **Streaming Context** (M5) - MISSING: Tool definition
    - Tool: `get_conversation_history(limit: int) -> Vec<Turn>`
    - Tool: `add_turn(role: Role, content: str) -> TurnId`
    - Tool: `clear_history() -> Result`
    - Returns: Conversation turns with roles and content

37. **Adaptive Selection** (M5) - MISSING: Tool definition
    - Tool: `select_provider(query: str, strategy: Strategy) -> Provider`
    - Returns: Best provider for query with selection reasoning

## Implementation Plan

### Phase 1: Admin REST API (Week 1)
**File**: `src/api/admin.rs`

```rust
// User-facing HTTP endpoints for configuration and monitoring
pub mod admin {
    // Health & Metrics
    GET  /health          -> HealthStatus
    GET  /metrics         -> PrometheusMetrics
    GET  /system/info     -> SystemInfo
    GET  /system/capabilities -> HardwareCapabilities
    
    // Model Info
    GET  /model/info      -> ModelInfo
    GET  /model/architecture -> ArchitectureInfo
    
    // Configuration
    POST /config/cache    -> CacheConfig
    GET  /config/cache    -> CacheConfig
    POST /cache/policy    -> PolicyConfig
    GET  /cache/stats     -> CacheStats
    POST /cache/clear     -> ClearResult
    
    POST /config/speculation -> SpeculationConfig
    GET  /config/speculation -> SpeculationConfig
    
    POST /config/gpu      -> MultiGPUConfig
    GET  /gpu/status      -> GPUStatus
    
    POST /config/compression -> CompressionConfig
    GET  /compression/stats  -> CompressionStats
    
    POST /reasoning/budget -> ReasoningBudget
    POST /reasoning/policy -> TerminationPolicy
    GET  /reasoning/stats  -> ReasoningStats
}
```

### Phase 2: LLM Tool Definitions (Week 1)
**File**: `src/api/tools.rs`

```rust
// Function calling schemas for LLM access
pub mod tools {
    // Query & Analysis
    Tool: analyze_query(query: String) -> AnalyzedQuery
    Tool: search_documents(query: String, top_k: usize) -> Vec<Document>
    Tool: inject_context(provider: String, prompt: String) -> Context
    
    // Reasoning & Planning
    Tool: create_pipeline(stages: Vec<Stage>) -> PipelineId
    Tool: execute_pipeline(id: PipelineId) -> PipelineResult
    Tool: set_request_metadata(priority: Priority, tags: Vec<Tag>) -> RequestId
    
    // State Management
    Tool: save_state(label: String) -> StateId
    Tool: restore_state(id: StateId) -> State
    Tool: branch_state(id: StateId, label: String) -> StateId
    
    // Knowledge Management
    Tool: add_fact(content: String, category: Category) -> FactKey
    Tool: query_facts(category: Option<Category>) -> Vec<Fact>
    Tool: validate_fact(fact: String, existing: Vec<String>) -> ValidationResult
    
    // Problem Decomposition
    Tool: decompose_problem(description: String) -> DecompositionResult
    Tool: get_subproblem(id: ProblemId) -> SubProblem
    Tool: solve_subproblem(id: ProblemId, solution: String) -> Result
    
    // Convergence
    Tool: check_convergence(window: usize) -> ConvergenceStatus
    
    // Context
    Tool: get_conversation_history(limit: usize) -> Vec<Turn>
    Tool: add_turn(role: Role, content: String) -> TurnId
    
    // Adaptive Selection
    Tool: select_provider(query: String, strategy: Strategy) -> Provider
}
```

### Phase 3: CLI Commands (Week 2)
**File**: `src/cli/mod.rs`

```rust
// Command-line interface for users
lightbulb serve --model <path> --port 8080
lightbulb health --endpoint <url>
lightbulb metrics --endpoint <url>
lightbulb system-info
lightbulb model-info --model <path>

lightbulb config cache --window 2048 --sinks 4
lightbulb config speculation --enable --draft <model>
lightbulb config gpu --strategy tensor-parallel --gpus 0,1
lightbulb config compression --strategy h2o --ratio 0.5
lightbulb config reasoning --max-chains 3 --policy convergence

lightbulb prune --model <path> --ratio 0.25 --strategy wanda --output <path>
lightbulb benchmark --model <path> --batch-size 32
lightbulb convert --input <path> --format gguf --output <path>
```

### Phase 4: Web Server Integration (Week 2)
**File**: `src/server/mod.rs`

Use `web-server-abstraction` crate for production server:

```rust
use web_server_abstraction::{WebServer, Request, Response};

let server = WebServer::with_axum_adapter()
    .register_route("/health", health_handler)
    .register_route("/metrics", metrics_handler)
    .register_route("/v1/completions", completions_handler)
    .register_route("/v1/chat/completions", chat_handler)
    .register_route("/api/admin/*", admin_handler)
    .register_route("/api/tools/*", tools_handler)
    .with_cors()
    .with_compression()
    .with_rate_limiting(100)
    .build()?;

server.start("0.0.0.0:8080").await?;
```

## Success Criteria

### User API (Admin Endpoints)
- ✅ All configuration options exposed via REST API
- ✅ Health check endpoint with detailed status
- ✅ Prometheus metrics export
- ✅ System/model info queries
- ✅ Real-time cache/GPU/reasoning stats

### LLM API (Tool Definitions)
- ✅ All 12 LLM-accessible features exposed as tools
- ✅ JSON schemas for function calling
- ✅ OpenAI-compatible tool format
- ✅ Comprehensive parameter validation
- ✅ Detailed error messages for tool failures

### CLI
- ✅ All user configuration via CLI commands
- ✅ Offline tools (prune, convert, benchmark)
- ✅ Remote control via `--endpoint` flag
- ✅ Interactive prompts for complex configs
- ✅ Clear help text and examples

### Integration
- ✅ Admin API secured with API keys
- ✅ Rate limiting on all endpoints
- ✅ CORS support for web UIs
- ✅ Compression for large responses
- ✅ WebSocket support for streaming

## Priority Order

1. **HIGH PRIORITY** (v1.0 blockers):
   - Health/metrics endpoints (observability)
   - Basic CLI (serve, health, system-info)
   - Tool definitions for core features (KB, query analysis, pipelines)

2. **MEDIUM PRIORITY** (v1.1):
   - Full admin API (all config endpoints)
   - Complete CLI (all commands)
   - All LLM tool definitions

3. **LOW PRIORITY** (v1.2+):
   - Web UI dashboard
   - Advanced admin features
   - Tool marketplace integration

## Next Steps

1. Implement `src/api/admin.rs` - User-facing REST API
2. Implement `src/api/tools.rs` - LLM function calling schemas
3. Implement `src/cli/mod.rs` - Command-line interface
4. Implement `src/server/mod.rs` - Web server with `web-server-abstraction`
5. Update ROADMAP.md M5.5 with detailed API specifications
6. Add API documentation to docs/
7. Create integration tests for all APIs
8. Add examples/ for API usage

## References

- `web-server-abstraction` docs: https://docs.rs/web-server-abstraction/latest/
- OpenAI function calling: https://platform.openai.com/docs/guides/function-calling
- Prometheus metrics: https://prometheus.io/docs/instrumenting/exposition_formats/
- REST API best practices: https://restfulapi.net/
