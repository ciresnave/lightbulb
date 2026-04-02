# Thought-local ("lightning-strike") Workspace — Design Notes

Status: draft

This document sketches an API and design for a thought-local workspace: a short-lived, in-memory workspace that captures the data and intermediate outputs relevant to a single inference or "thought". It complements the existing arena in `dynctx` and is intended for exploration and prototyping.

Goals

- Provide a focused, high-throughput temporary store for intermediate values during a computation.
- Track provenance and access traces (tracking tokens) for later analysis and selective persistence.
- Avoid global bottlenecks by scoping access to the active "thought" and enabling efficient lookups for modules participating in that thought.
- Provide simple primitives that can be implemented quickly and evolve into more sophisticated versions.

Concepts

- ThoughtId: a short-lived UUID for each thought/inference.
- TokenHandle: an opaque handle for stored items (references into the thought-local workspace or into `dynctx` arena).
- TrackingToken: lightweight provenance data attached to TokenHandle captures which modules accessed/produced the value and timestamps.
- ModuleContext: per-module ephemeral state (caching hints, attention scores, metadata) for the duration of the thought.

High-level API (pseudo-Rust)

```rust
/// Unique identifier for a thought
pub struct ThoughtId(pub uuid::Uuid);

/// A short-lived workspace created for the duration of an inference
pub struct ThoughtWorkspace {
    id: ThoughtId,
    storage: HashMap<TokenId, Value>,
    traces: Vec<AccessTrace>,
}

impl ThoughtWorkspace {
    /// Create a new thought workspace
    pub fn new() -> Self { /* ... */ }

    /// Insert a value and receive a TokenHandle
    pub fn insert(&mut self, value: Value, producer: ModuleId) -> TokenHandle { /* ... */ }

    /// Read a Token by handle (records access trace)
    pub fn read(&mut self, handle: &TokenHandle, reader: ModuleId) -> Option<&Value> { /* ... */ }

    /// Persist select Tokens to longer-term storage (arena or CapabilityStore)
    pub async fn persist(&mut self, tokens: &[TokenHandle]) -> Result<()> { /* ... */ }

    /// Retrieve access traces for analysis
    pub fn traces(&self) -> &[AccessTrace] { &self.traces }
}
```

Tracking and provenance

- Each insert/read updates an AccessTrace record: (TokenId, ModuleId, action {read|write}, timestamp, attention_score?).
- Traces are compact and can be flushed to a log or used locally to train the sequence-aware predictor.

Integration with `dynctx`

- Values that should persist can be copied into the `dynctx` arena via a `persist` helper.
- The ThoughtWorkspace can store arena Positions (references) for larger objects to avoid duplication.

Access patterns and concurrency

- ThoughtWorkspaces are single-writer (the owning computation) but support concurrent read helpers for worker threads.
- Use lock-free queues or small RwLock around the traces vector for low-overhead recording.

Example usage

1. Router creates a ThoughtWorkspace and records initial inputs.
2. Router selects modules (using attention); modules read input tokens, produce outputs inserted into the workspace.
3. Router records attention scores and module activations in traces.
4. At completion, persist a subset of tokens (selected by fingerprinting / heuristics) into `dynctx` or capability store.

Security and privacy

- Thought workspaces are ephemeral and should be zeroed before deallocation.
- Sensitive inputs must be annotated; persistence requires explicit authorization.

Next steps (prototype)

1. Add a minimal in-memory ThoughtWorkspace implementation in `crates/dynctx/examples/` for experimentation.
2. Add helper functions to serialize traces for the sequence-predictor training pipeline.
3. Add unit tests for insertion/read/persist semantics and trace correctness.

---
