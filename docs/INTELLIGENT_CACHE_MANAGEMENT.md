# Intelligent KV Cache Management - lightbulb Implementation

## Status: Design Phase / Phased Implementation

This document outlines an advanced KV cache management system for lightbulb, inspired by hierarchical memory architectures and combining multiple eviction policies with explicit model control.

## Vision

Treat LLM context management like a memory hierarchy:
- **KV Cache** = Working memory (fast, limited, managed intelligently)
- **RAG/External** = Long-term memory (slower, unlimited) [future]
- **Model** = Can explicitly manage its own context via tools

## Core Principles (lightbulb-aligned)

✅ **Portable**: Works on CPU/GPU, no special hardware  
✅ **Measurable**: Clear metrics (cache efficiency, memory savings, hit rates)  
✅ **Practical**: Solves real problems (long conversations, limited memory)  
✅ **Compatible**: Works with existing pre-trained models, no retraining  

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   User Request                          │
└────────────────────┬────────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │  Multi-Policy       │
          │  Eviction System    │
          │  ┌──────────────┐   │
          │  │ H2O Policy   │   │  Track attention scores
          │  │ StreamingLLM │   │  Attention sinks + window
          │  │ Recency      │   │  Time-based decay
          │  │ Tool Tags    │   │  Explicit priorities
          │  └──────────────┘   │
          │  Voting Aggregator  │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │  Enhanced KV Cache  │
          │  with Metadata      │
          │  ┌──────────────┐   │
          │  │ Token Data   │   │  K/V tensors
          │  │ Attention    │   │  Cumulative scores
          │  │ Tags         │   │  ToolOutput, User, etc.
          │  │ Timestamps   │   │  For recency tracking
          │  │ Importance   │   │  Explicit priorities
          │  └──────────────┘   │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │  Cache Control      │
          │  Tools (exposed     │
          │  to model)          │
          │  ┌──────────────┐   │
          │  │ get_cache_   │   │  Inspect usage
          │  │   usage()    │   │
          │  │ tag_region() │   │  Mark important
          │  │ evict_tagged │   │  Explicit cleanup
          │  └──────────────┘   │
          └─────────────────────┘
```

## Implementation Phases

### Phase 1: Multi-Policy Eviction (HIGH PRIORITY) ✅ StreamingLLM done

**Status**: StreamingLLM ✅ Complete, H2O and Voting System → Next

**Components:**

1. **H2O (Heavy Hitters Oracle)** - `src/cache/h2o_policy.rs`
   ```rust
   pub struct H2OPolicy {
       attention_scores: Vec<Vec<f32>>,  // [batch][position]
       decay_factor: f32,                 // Decay old scores over time
   }
   
   impl H2OPolicy {
       pub fn update_scores(&mut self, attention_weights: &Tensor);
       pub fn score_for_eviction(&self, position: usize) -> f32;
   }
   ```
   
   - Track cumulative attention scores per token
   - Modify `custom_attention.rs` to accumulate weights
   - Store in per-slot metadata
   - Evict tokens with lowest cumulative attention

2. **Policy Abstraction** - `src/cache/eviction_policy.rs`
   ```rust
   pub trait EvictionPolicy {
       /// Score tokens for eviction (0.0 = keep, 1.0 = evict)
       fn score_tokens(&self, slot: usize, positions: &[usize]) -> Vec<f32>;
       
       /// Update policy state after forward pass
       fn update(&mut self, attention_data: Option<&Tensor>);
   }
   
   pub struct StreamingLLMPolicy { config: StreamingConfig }
   pub struct H2OPolicy { /* attention tracking */ }
   pub struct RecencyPolicy { decay_rate: f32 }
   pub struct TagBasedPolicy { tag_priorities: HashMap<String, f32> }
   ```

3. **Voting System** - `src/cache/eviction_manager.rs`
   ```rust
   pub struct EvictionManager {
       policies: Vec<(Box<dyn EvictionPolicy>, f32)>,  // (policy, weight)
       strategy: VotingStrategy,
   }
   
   pub enum VotingStrategy {
       WeightedAverage,
       Threshold { min_votes: usize },
       Veto { any_keep_wins: bool },
   }
   
   impl EvictionManager {
       pub fn decide_eviction(&self, slot: usize) -> Vec<usize> {
           // Collect votes from all policies
           // Aggregate based on strategy
           // Return positions to evict
       }
   }
   ```

**Acceptance Criteria:**
- ✅ StreamingLLM working (done!)
- ⏳ H2O tracks attention, evicts low-attention tokens
- ⏳ Voting system combines policies measurably better than single policy
- ⏳ Configurable policy weights and strategies
- ⏳ Benchmarks show improved cache efficiency

**Benefits:**
- Semantic awareness (H2O sees what model actually uses)
- Stability (StreamingLLM preserves attention sinks)
- Flexibility (voting adapts to different workloads)
- No model retraining needed

### Phase 2: Enhanced Tagging & Tool Integration (MEDIUM PRIORITY)

**Status**: Design phase

**Components:**

1. **Token Metadata** - Extend `ParallelCacheBuilder`
   ```rust
   pub struct TokenMetadata {
       pub tag: Option<String>,           // "tool:read_file", "user", "system"
       pub timestamp: SystemTime,
       pub importance: f32,               // 0.0-1.0, explicit priority
       pub tool_context: Option<String>,  // Tool call ID or context
       pub attention_score: f32,          // From H2O
   }
   
   pub struct EnhancedCacheBuilder {
       base: ParallelCacheBuilder,
       metadata: Vec<Vec<TokenMetadata>>,  // [batch][position]
   }
   ```

2. **Cache Control Tools** - `src/tools/cache_control.rs`
   ```rust
   #[derive(Serialize, Deserialize)]
   pub struct GetCacheUsage {
       // Returns: used, total, percentage, breakdown by tag
   }
   
   #[derive(Serialize, Deserialize)]
   pub struct TagRegion {
       pub start: usize,
       pub end: usize,
       pub tag: String,
       pub importance: f32,
   }
   
   #[derive(Serialize, Deserialize)]
   pub struct EvictTagged {
       pub tag: String,  // Evict all tokens with this tag
   }
   ```

3. **Tool Output Auto-Tagging**
   - When `read_file()` executes, tag output as `tool:read_file:filename`
   - When model generates, tag as `model:generation`
   - User input tagged as `user:input`
   - System prompts tagged as `system:prompt` with `importance=1.0`

**Usage Example:**
```python
# Model can explicitly manage context
tools = [
    {"name": "get_cache_usage", "description": "Check context usage"},
    {"name": "read_file", "description": "Load file into context"},
    {"name": "close_file", "description": "Remove file from context"},
]

# Model reasoning:
# "Cache at 85%, need to load auth.py. Let me close_file(test.py) first."
```

**Acceptance Criteria:**
- ⏳ Metadata tracked for all tokens
- ⏳ Tools exposed and functional
- ⏳ Model can inspect and control cache
- ⏳ Automatic tagging for tool outputs
- ⏳ Explicit eviction by tag works correctly

**Benefits:**
- Transparency: Model knows what's in context
- Control: Model can manage its memory explicitly
- Efficiency: Strategic loading/unloading of content
- Novel: No other framework offers this

### Phase 3: Hierarchical Memory (FUTURE/EXPERIMENTAL)

**Status**: Design only, deferred

This phase involves a small "memory controller" model and RAG integration. While intellectually compelling, it adds significant complexity:

**Challenges:**
- ⚠️ Latency overhead (small model inference every turn)
- ⚠️ RAG infrastructure not currently in lightbulb
- ⚠️ Coordination complexity (2 models)
- ⚠️ Reliability concerns (small model makes decisions)

**Recommendation:**
- Document the design thoroughly (this file)
- Feature-gate as experimental (`--features hierarchical-memory`)
- Implement simplified version first (scoring only, no RAG)
- Gather real-world usage data before full implementation
- Consider as research contribution / paper topic

**If implemented, start with:**
1. Small model analyzes incoming prompt
2. Scores current cache content for relevance
3. Makes eviction recommendations (not automatic)
4. Logs decisions for analysis
5. Gradually increase autonomy based on reliability data

## Performance Targets

### Phase 1 (Multi-Policy)
- **Memory savings**: 30-50% vs. full cache (benchmark on long conversations)
- **Cache hit rate**: >80% for relevant content
- **Overhead**: <5ms per token for policy decisions
- **Quality**: No degradation on standard benchmarks

### Phase 2 (Tool Integration)
- **Model control accuracy**: >90% of explicit evictions correct
- **Tool call overhead**: <10ms per cache control operation
- **User experience**: Transparently better context management

### Phase 3 (Hierarchical)
- **Small model latency**: <200ms per turn
- **Retrieval relevance**: >85% of retrieved content actually used
- **Fact preservation**: >95% accuracy in summarization
- **End-to-end latency**: <500ms overhead vs. baseline

## Testing Strategy

### Unit Tests
- [x] StreamingLLM policy (6/6 tests passing)
- [ ] H2O attention tracking and scoring
- [ ] Voting system with known inputs
- [ ] Tag-based eviction
- [ ] Tool call handling

### Integration Tests
- [ ] Multi-policy coordination on synthetic workload
- [ ] Tool + eviction interaction
- [ ] Long conversation (1000+ turns) memory stability
- [ ] Mixed content types (code, prose, data)

### Benchmarks
- [ ] Memory usage over time (various policies)
- [ ] Cache hit rates (multi-policy vs. single)
- [ ] Latency per policy (H2O, StreamingLLM, Recency)
- [ ] Quality preservation (perplexity, task accuracy)

### Real-World Scenarios
- [ ] Code review session (file context management)
- [ ] Research paper writing (section eviction/retrieval)
- [ ] Debugging workflow (stack trace preservation)
- [ ] Multi-file refactoring (strategic file loading)

## Open Questions

1. **Attention weight extraction**: How expensive is storing full attention during inference?
   - **Answer needed**: Profile overhead, consider sampling strategies

2. **Voting strategy tunables**: What are good default weights?
   - **Answer needed**: Empirical testing across workloads

3. **Tool adoption**: Will models actually use cache control tools effectively?
   - **Answer needed**: User studies, prompt engineering

4. **Hierarchical reliability**: Can small model make good eviction decisions?
   - **Answer needed**: Extensive testing, fallback mechanisms

## Related Work

- **StreamingLLM**: Xiao et al., "Efficient Streaming Language Models with Attention Sinks"
- **H2O**: Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference"
- **Scissorhands**: Liu et al., "Scissorhands: Exploiting the Persistence of Importance Hypothesis"
- **RAG**: Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"

## Implementation Timeline

**M2 (Current) - Performance Enablers:**
- ✅ StreamingLLM policy
- ⏳ Lightning GGUF Phase 3
- ⏳ Flash attention integration

**M3 - Advanced Cache Management:**
- H2O policy
- Multi-policy voting system
- Enhanced tagging

**M4 - Tool Integration:**
- Cache control tools
- Automatic tagging
- Model-driven memory management

**M5+ - Research/Experimental:**
- Hierarchical memory (feature-gated)
- Small model controller
- RAG integration

## Contributing

This is a complex, multi-faceted system. Contributions welcome in:
- Policy implementations (new eviction strategies)
- Benchmarking and testing
- Real-world use case validation
- Documentation and examples

See `CONTRIBUTING.md` for guidelines.

---

## Design Credits

This architecture synthesizes ideas from recent research (H2O, StreamingLLM, Scissorhands) with novel concepts (tool integration, hierarchical memory) specifically adapted for lightbulb's portable, measurable, practical philosophy.

**Key Insight**: Rather than pick one eviction strategy, combine multiple approaches with voting, and give the model explicit control over its memory. This flexibility enables optimization for diverse workloads without model retraining.
