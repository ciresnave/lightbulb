# CR.1: Mid-Reasoning Context Injection — Design Document

**Author:** Eric (ciresnave@gmail.com), with design assistance from Claude
**Date:** April 2, 2026
**Status:** Design complete, ready for implementation
**Dependencies:** Segmented KV cache (COMPLETE), attention weight capture (COMPLETE)

---

## 1. The Problem

Every current tool use system works like this:

```
Generate tokens → detect tool call → STOP → run tool →
start fresh forward pass → read tool result as new prompt text
```

The reasoning state that made the tool call meaningful is **gone** by the time the result arrives. The model reads tool results as narrative reports about something that happened in the past, not as sensory feedback arriving into active reasoning.

This is equivalent to a human asking a question, then getting amnesia, then reading the answer written on a piece of paper with no memory of why they asked.

## 2. What We Can Do Differently

Lightbulb's `ParallelKvCache` stores the full attention state — every token's key and value vectors across all layers. When we detect a tool call, we can:

1. **Preserve the KV cache** — don't clear it, don't reset positions
2. **Append the tool result as new tokens** — process them against the preserved cache
3. **Resume generation** — the model continues from where it left off, with the tool result integrated into its full attentional context

The model experiences the tool result as information arriving mid-thought, not as a fresh prompt.

## 3. How It Maps to Our Infrastructure

### 3.1 What Already Exists

| Component | Status | Role in CR.1 |
|-----------|--------|--------------|
| `ParallelKvCache` | ✅ Complete | Stores K/V tensors; survives across calls naturally |
| `ParallelCacheBuilder` | ✅ Complete | Tracks positions; just don't reset them |
| `CacheSpan` + `SpanRegistry` | ✅ Complete | Tag injected content as `ToolOutput` spans |
| `capture_attention` | ✅ Complete | Capture what model was attending to at tool call moment |
| `<RETRIEVE:key>` interception | ✅ Complete | Prototype for tool call detection pattern |
| `RequestContext` | ✅ Complete | Tracks generation state per request |
| `ToolRegistry` | ✅ Complete | Detects model capabilities, stores tool schemas |
| `KnowledgeBase` | ✅ Complete | Could store tool results for later retrieval |

### 3.2 What's Missing

1. **Tool call detection** — generalized pattern matching beyond `<RETRIEVE:key>`
2. **RequestState::AwaitingToolResult** — new state for paused generation
3. **Token injection into active cache** — append tool result tokens at current position
4. **Generation resumption** — continue decoding after injection
5. **Attention state capture at tool call moment** — save what the model was focusing on
6. **Tool execution interface** — callback mechanism for running tools

## 4. Detailed Design

### 4.1 Request State Extension

```rust
pub enum RequestState {
    Pending,           // Waiting for prefill
    Decoding,          // Actively generating tokens
    AwaitingToolResult {  // NEW: Paused, waiting for tool result
        tool_name: String,
        tool_args: String,
        cache_position: usize,       // Where we paused in the cache
        attention_snapshot: Option<AttentionSnapshot>,  // What model was attending to
    },
    Completed,         // Generation finished
}
```

### 4.2 Attention Snapshot

At the moment of the tool call, the last layer's attention weights tell us exactly what the model was focusing on. This is already computed and available via `capture_attention`.

```rust
pub struct AttentionSnapshot {
    /// Attention weights from the tool-call-generating token.
    /// Shape: [num_heads, context_len] — how much each head attended to each position.
    pub weights: Vec<Vec<f32>>,

    /// Which spans received the most attention (pre-computed).
    /// Sorted by total attention descending.
    pub attended_spans: Vec<(SpanId, f32)>,

    /// The position at which the tool call was generated.
    pub position: usize,
}
```

This snapshot serves two purposes:
1. **Diagnostic** — understand what the model was reasoning about when it called the tool
2. **Future use (CR.1.2)** — weight the tool result processing based on what the model expected

### 4.3 Tool Call Detection

Generalize the existing `<RETRIEVE:key>` pattern into a configurable detector:

```rust
pub struct ToolCallDetector {
    /// Registered tool call patterns.
    patterns: Vec<ToolCallPattern>,

    /// Rolling token buffer per request (last N tokens).
    token_buffers: HashMap<usize, Vec<u32>>,  // cache_idx → buffer

    /// Buffer size (how many tokens to keep for pattern matching).
    buffer_size: usize,  // default: 30
}

pub struct ToolCallPattern {
    /// Start marker (e.g., "<tool_call>", "[TOOL_CALL]")
    pub start_marker: String,

    /// End marker (e.g., "</tool_call>", "[/TOOL_CALL]")
    pub end_marker: String,

    /// Parser function: extract tool name and arguments from the matched text
    pub parser: Box<dyn Fn(&str) -> Option<(String, String)> + Send>,
}

pub struct DetectedToolCall {
    pub tool_name: String,
    pub tool_args: String,
    pub token_range: (usize, usize),  // start..end in generated_tokens
}
```

### 4.4 Token Injection Flow

When a tool result is ready, inject it into the active KV cache:

```
BEFORE injection:
  Cache: [sys_prompt | user_input | model_gen... | <tool_call>get_weather("Paris")</tool_call>]
  Position: 85

INJECTION:
  1. Tokenize tool result: "The weather in Paris is 22°C and sunny"
  2. Create prefill-style input from result tokens
  3. Forward pass ONLY for result tokens, against preserved cache
  4. Cache now contains: [...prior context... | tool_call | tool_result]
  5. Position: 85 + len(result_tokens)
  6. Create ToolOutput span for the injected range

AFTER injection:
  Cache: [sys_prompt | user_input | model_gen | tool_call | TOOL_RESULT]
  Model resumes generation from position 85 + result_len
  Full attentional context preserved — model "remembers" why it called the tool
```

### 4.5 The Key Mechanism: Mini-Prefill for Injection

The tool result is processed as a **mini-prefill** — a forward pass where:
- Input = tokenized tool result (multiple tokens)
- Cache = preserved from before the tool call (not reset)
- Starting position = where we paused
- The model processes the result tokens and updates the KV cache
- No token generation during this pass — just cache population

This is identical to how prefill works, except:
- The cache already has content (not empty)
- We start at a non-zero position
- The result is much shorter than a typical prompt

Our chunked prefill infrastructure already handles exactly this — processing tokens at arbitrary positions against an existing cache.

### 4.6 Integration into the Decode Loop

```
forward_batch() decode phase:
  for each request:
    generate next_token via forward pass
    push to generated_tokens

    // Tool call detection (replaces existing RETRIEVE check)
    if tool_detector.check(cache_idx, &generated_tokens):
      detected = tool_detector.extract_call(...)

      // Capture attention snapshot
      snapshot = capture_attention_state(last_attn_weights, cache_idx)

      // Transition to waiting state
      ctx.state = AwaitingToolResult {
        tool_name: detected.tool_name,
        tool_args: detected.tool_args,
        cache_position: current_position,
        attention_snapshot: Some(snapshot),
      }

      // Signal caller that tool execution is needed
      // (via channel, callback, or return value)
      results[idx] = ToolCallRequested(detected)
      continue  // Skip EOS check, skip further generation

    // Normal EOS check
    if is_eos(next_token) || max_tokens_reached:
      ctx.complete()
```

### 4.7 Tool Result Injection (Called After Tool Execution)

```rust
impl ParallelModelManager {
    /// Inject a tool result into an active request's KV cache.
    ///
    /// The request must be in AwaitingToolResult state. The tool result
    /// is tokenized and processed as a mini-prefill against the preserved
    /// cache, then the request transitions back to Decoding state.
    pub fn inject_tool_result(
        &mut self,
        request_idx: usize,
        tool_result: &str,
    ) -> Result<()> {
        // 1. Verify request is in AwaitingToolResult state
        // 2. Tokenize tool result (with appropriate formatting)
        // 3. Create a ToolOutput span
        // 4. Forward pass: mini-prefill of result tokens against preserved cache
        // 5. Advance positions by result token count
        // 6. End ToolOutput span
        // 7. Transition request back to Decoding state
    }
}
```

### 4.8 Formatting the Tool Result

The injected tokens need formatting that the model can distinguish from its own output. Common patterns:

```
[TOOL_RESULT]
The weather in Paris is 22°C and sunny.
[/TOOL_RESULT]
```

The format should be:
- Configurable per model (different models expect different formats)
- Clearly delimited so the model knows where tool output begins and ends
- Brief — tool results should be concise to minimize cache consumption

### 4.9 Integration with ModelRunner

ModelRunner needs to handle the new `AwaitingToolResult` state:

```rust
// In the streaming loop:
match manager.forward_batch(&mut batch) {
    Ok(results) => {
        for result in results {
            match result {
                Some(token) => stream_tx.send(Ok(token_text)),

                ToolCallRequested(call) => {
                    // Execute tool (sync or async)
                    let tool_result = execute_tool(&call.tool_name, &call.tool_args)?;

                    // Inject result back into the model's reasoning
                    manager.inject_tool_result(request_idx, &tool_result)?;

                    // Continue generation — model resumes with tool result in context
                }

                None => { /* completed */ }
            }
        }
    }
}
```

### 4.10 Integration with Segmented KV Cache

Tool results are first-class segments:
- **CacheTag::ToolOutput** — already defined in the CacheSpan system
- Tool result spans get their own attention tracking
- The eviction system can reason about tool results independently
- High-value tool results (frequently attended to) survive eviction
- Stale tool results (from early in conversation) can be demoted like any other segment

## 5. What This Enables Beyond Basic Tool Use

### 5.1 Async Tool Completion
Long-running tools (database queries, API calls) can complete asynchronously. The model continues reasoning about other aspects while the tool runs, and the result is injected when ready.

### 5.2 Multi-Tool Parallelism
Multiple tool calls can be in flight simultaneously. Each result injects independently into the preserved cache at the appropriate position.

### 5.3 User Interruption Without Context Loss
A user can inject a clarification mid-generation. The clarification is processed against the model's current reasoning state rather than requiring a full restart.

### 5.4 Inter-LLM Communication
One LLM's output can be injected into another's active reasoning. This is the foundation for the multi-agent communication described in the original segmented KV cache design.

### 5.5 Memory Retrieval Mid-Thought
When the segmented eviction system detects the model is "searching" for demoted content (via attention patterns on the KB placeholder), it can proactively inject the retrieved content rather than waiting for an explicit `<RETRIEVE:key>` token.

## 6. Implementation Phases

### Phase 1: Tool Call Detection + State Pause (1 week)
- `ToolCallDetector` with configurable patterns
- `RequestState::AwaitingToolResult` with attention snapshot
- Detection integrated into decode loop (generalizes RETRIEVE pattern)
- No actual injection yet — just detection and pause

### Phase 2: Token Injection via Mini-Prefill (1 week)
- `inject_tool_result()` on ParallelModelManager
- Mini-prefill of result tokens against preserved cache
- ToolOutput span creation
- Request state transition back to Decoding
- End-to-end test: detect → pause → inject → resume → verify coherent output

### Phase 3: ModelRunner Integration (3 days)
- Handle ToolCallRequested in streaming and complete modes
- Synchronous tool execution callback
- Wire into API layer

### Phase 4: Attention Snapshot Analysis (1 week)
- Capture full attention state at tool call moment
- Analyze what the model was attending to (which spans, which positions)
- Log attention snapshot alongside tool call for diagnostics
- Foundation for CR.1.2 (attention-weighted result processing)

### Phase 5: Async Tool Execution (future)
- Non-blocking tool execution with injection on completion
- Multiple concurrent tool calls
- Priority-based injection ordering

## 7. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Mini-prefill for injection | Reuses existing chunked prefill infrastructure. No new tensor operations needed. |
| KV cache preserved by default | "Freezing" is just not advancing positions. No explicit save/restore. |
| Token injection, not activation injection | Buildable today without model modification. Activations are a future enhancement (CR.1.2). |
| Configurable tool call patterns | Different models use different formats. Lightbulb shouldn't hard-code any. |
| ToolOutput as CacheSpan tag | Integrates naturally with eviction system. Tool results are first-class segments. |
| Attention snapshot is optional | Useful for diagnostics and future work but not required for basic injection. |

## 8. What's Different From Every Other System

**Current systems (LangChain, OpenAI, Claude, etc.):**
```
Generate → stop → run tool → new prompt with tool result appended → fresh forward pass
```
- Reasoning state lost at the stop boundary
- Tool result is narrative context, not mid-thought information
- Model must "re-derive" what it was thinking from textual clues

**Lightbulb with CR.1:**
```
Generate → detect tool call → preserve KV cache → run tool →
inject result tokens into preserved cache → resume generation
```
- Reasoning state fully preserved in KV cache
- Tool result processed in the attentional context of the reasoning that requested it
- Model experiences the result as information arriving mid-thought
- No re-derivation needed — the original reasoning is still live

The difference is analogous to:
- **Current:** Writing down your question, mailing it, forgetting you asked, receiving a letter with the answer
- **CR.1:** Asking a question, holding your thought, hearing the answer, continuing your reasoning

## 9. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Tool result tokens consume cache space | ToolOutput spans participate in eviction like any other segment |
| Model confused by injected tokens | Use clear delimiters; test with multiple model architectures |
| Position discontinuity | No discontinuity — result tokens occupy sequential positions after the tool call |
| Tool call detection false positives | Require both start and end markers; configurable pattern strictness |
| Long tool results overwhelm context | Summarize/truncate tool results before injection; configurable max length |

## 10. Connection to CR.2 (Continuous Reasoning)

CR.1 is the prerequisite for CR.2. Once we can inject information mid-reasoning without losing state:
- The model can run continuously, receiving injections as events occur
- Confidence gating (CR.2.1) determines when to surface output
- The model never stops — it's always reasoning, always integrable

CR.1 transforms Lightbulb from an inference engine into a **reasoning substrate** that can receive and integrate information at any time.
