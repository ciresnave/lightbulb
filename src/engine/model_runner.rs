use anyhow::Result;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, SyncSender};
use tokio::sync::{mpsc as async_mpsc, oneshot};

use crate::engine::{Request, RequestContext};

/// Why generation stopped, in OpenAI's vocabulary.
///
/// OpenAI also defines `content_filter`; nothing in Lightbulb produces it,
/// and an enum carrying a variant no code can construct is worse than one
/// that grows when a producer appears. `ToolCalls` is the one exception,
/// added together with its producer: `api::openai::chat::extract_tool_calls`
/// parsing Lightbulb's own injected `<tool_call>` tag out of the complete
/// generated text (see `docs/TOOL-CALLING-DESIGN-2026-09-25.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// The model emitted an end-of-sequence token.
    Stop,
    /// `max_new_tokens` was reached with no EOS. Continuing is meaningful,
    /// which is why a client needs to be able to tell this from `Stop`.
    Length,
    /// The response carries one or more `tool_calls` instead of (or ahead
    /// of) a normal answer. Set by the handler after parsing generated
    /// text, never by the runner: the runner has no notion of `tools`.
    ToolCalls,
}

impl FinishReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::ToolCalls => "tool_calls",
        }
    }
}

/// Everything a completion response needs, reported by the runner rather than
/// guessed by the handler.
#[derive(Debug)]
pub struct CompletionResult {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: FinishReason,
}

/// One item on the streaming channel.
///
/// `Done` carries ONLY the finish reason. Adding `completion_tokens` for
/// symmetry with `CompletionResult` is deliberately rejected: OpenAI omits
/// `usage` from stream chunks, so nothing would consume it, and a field that
/// exists because it looked symmetrical is one the next reader has to work out
/// is unused.
#[derive(Debug)]
pub enum StreamItem {
    Token(String),
    Done { finish_reason: FinishReason },
}

/// Classify how a finished request ended.
///
/// Reads `ctx.stopped_on_eos` rather than `ctx.state`, because
/// `RequestState::Completed` cannot answer this question: every backend
/// completes a request on `is_eos || tokens_generated >= max_new_tokens`
/// (EOS-only completion would leave an ordinary budget-exhausted request
/// stuck in `Decoding` forever, leaking its KV cache), so by the time a
/// request reaches `Completed` the two exits are already indistinguishable
/// from `state` alone. `stopped_on_eos` is recorded by the backend at the
/// point where it still knows which condition fired, so there is no
/// ordering trap here to get wrong — unlike the old `state`-based version,
/// which had to check `Completed` before the token count precisely because
/// EOS landing on the final allowed token satisfies both at once.
///
/// A free function over `RequestContext` rather than a method, so it is
/// unit-testable without a model, a checkpoint, or a runner thread.
pub fn finish_reason_for(ctx: &RequestContext) -> FinishReason {
    if ctx.stopped_on_eos {
        FinishReason::Stop
    } else {
        FinishReason::Length
    }
}

/// Response mode for inference jobs
#[derive(Debug)]
pub enum ResponseMode {
    /// Complete response - send final text when done
    Complete(oneshot::Sender<anyhow::Result<CompletionResult>>),
    /// Streaming response - send tokens as they are generated
    Streaming(async_mpsc::UnboundedSender<Result<StreamItem>>),
}

/// Job submitted to the model runner thread
pub struct InferenceJob {
    pub id: String,
    pub prompt: String,
    pub max_new_tokens: usize,
    pub temperature: f64,
    /// Whether the backend must ask its tokenizer to add special tokens when
    /// prefilling `prompt`.
    ///
    /// Deliberately **not** `#[serde(default)]`-shaped or `Option`: every
    /// construction site must state it, because the right answer depends on
    /// something only the prompt's builder knows — whether a chat template was
    /// applied. See [`crate::engine::RequestContext::add_special_tokens`] for
    /// what goes wrong when a templated prompt gets `true`.
    ///
    /// `true` for a raw prompt (`/v1/completions`, and the legacy
    /// `role: content` join used when no tier resolved a template); `false` for
    /// a prompt that came out of `ResolvedTemplate::render` — which now
    /// includes every contract-executor attempt.
    pub add_special_tokens: bool,
    /// M4 slice 5.4c (Option B): the priority this job was submitted with,
    /// read from the `X-Lightbulb-Priority` HTTP header (never the OpenAI-
    /// compatible JSON body -- see `docs/API-EXTENSIONS.md`), defaulting to
    /// `Priority::Normal` when the header is absent or unparseable.
    /// Reaches the shared scheduler via `admit_pending`'s
    /// `submit_priority_request` call, not `submit_request` (which always
    /// submits `Normal` regardless of what is passed to it).
    pub priority: crate::engine::Priority,
    /// Response mode (complete or streaming)
    pub response_mode: ResponseMode,
}

/// Sender type used by API handlers to enqueue inference jobs.
///
/// `SyncSender`, not the unbounded `Sender` — the channel `ModelRunner::start`
/// creates has a fixed capacity, and a caller that wants to enqueue without
/// blocking the async request-handling task uses `try_send`, which returns
/// [`std::sync::mpsc::TrySendError::Full`] once that capacity is reached
/// rather than queuing without bound.
pub type InferenceRequestSender = SyncSender<InferenceJob>;

/// Shared, cross-thread-readable snapshot of the prefix cache's stats — the
/// `cache_stats` admin endpoint's data source (board item M4 slice 1).
///
/// GAP since candlelight's removal (2026-10-22): the candlelight path's
/// `run_jobs` was the only writer; `ModelRunner::start`'s Fuel implementation
/// accepts this parameter for signature compatibility and never writes it,
/// so the admin endpoint now always reads the zero value it was constructed
/// with. See `docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md`.
pub type SharedCacheStats = std::sync::Arc<std::sync::Mutex<crate::cache::PrefixCacheStats>>;

/// Shared, lockable handle on the admission scheduler (M4 slice 5.4a).
///
/// GAP since candlelight's removal (2026-10-22): the candlelight path's
/// `run_jobs` was the only writer (real admission/departure mirroring,
/// slices 5.4a/5.4b/5.4c); `ModelRunner::start`'s Fuel implementation accepts
/// this parameter for signature compatibility and never writes it, so the
/// `scheduler_stats`/`scheduler_queue` admin endpoints now always read an
/// empty scheduler. See `docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md`.
pub type SharedScheduler = std::sync::Arc<std::sync::Mutex<crate::engine::MemoryAwareScheduler>>;

/// Live count of inference jobs currently enqueued or in flight — the
/// `/metrics` endpoint's queue-depth gauge (Phase 4 observability). Plain
/// `Arc<AtomicUsize>` rather than threading a counter through the channel
/// itself: `std::sync::mpsc::sync_channel`'s `SyncSender`/`Receiver` expose
/// no `len()` at all, so there is nothing to read off the channel directly.
pub type SharedQueueDepth = std::sync::Arc<std::sync::atomic::AtomicUsize>;

/// RAII handle on one job's lifetime in `SharedQueueDepth`: increments on
/// construction, decrements on drop, covering every exit path (a completed
/// job, a job whose client disconnected, a job that errored, a job whose
/// `try_send` never actually reached the channel) without each call site
/// having to remember to decrement on its own way out.
pub struct QueueDepthGuard(SharedQueueDepth);

impl QueueDepthGuard {
    pub fn new(counter: SharedQueueDepth) -> Self {
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self(counter)
    }
}

impl Drop for QueueDepthGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Returned by an enqueue attempt that found the inference queue at capacity.
///
/// A distinct type, rather than a bare `anyhow!(...)`, so a handler can tell
/// "the server is temporarily busy" (HTTP 503, safe to retry) apart from
/// every other enqueue failure (e.g. the runner thread died — HTTP 500,
/// retrying the identical request will fail identically forever) without
/// matching on the error's rendered text.
#[derive(Debug)]
pub struct QueueFullError;

impl std::fmt::Display for QueueFullError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "inference queue is full; try again shortly")
    }
}

impl std::error::Error for QueueFullError {}

/// Returned when an enqueued job's response didn't arrive within
/// `ApiConfig::request_timeout_ms`. A distinct type, same reasoning as
/// [`QueueFullError`]: a handler maps this to HTTP 504 (the request may
/// still be running server-side when the client gives up on it; retrying
/// is not obviously safe the way a 503 retry is) rather than folding it
/// into the generic 500 every other failure gets.
#[derive(Debug)]
pub struct RequestTimeoutError;

impl std::fmt::Display for RequestTimeoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "inference request timed out")
    }
}

impl std::error::Error for RequestTimeoutError {}

/// What the job loop needs from a model, and nothing more.
///
/// **Deliberately tensor-free.** No Fuel or candlelight type appears here, which
/// is what keeps `engine/` free of the tensor layer — the property the port
/// design names as the actual differentiator worth protecting.
///
/// Already batch-shaped, so adding real batching changes the job-draining loop
/// and the implementations, not this trait.

/// Lightweight model runner that owns the Fuel-backed model on a dedicated thread
/// and services incoming inference jobs.
///
/// Supports both SafeTensors (directory) and GGUF (single file) model formats,
/// auto-detected from the model path. The candlelight-backed implementation
/// (`ParallelModelManager`, `run_jobs`) was removed 2026-10-22; see
/// `docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md` for what that cost.
pub struct ModelRunner;

impl ModelRunner {
    /// Start the model runner thread, serving through the Fuel path.
    ///
    /// `dtype` is accepted and IGNORED: this path loads all weights as f32 by
    /// construction. Fuel's CPU backend has no `[F32, BF16, F32]` matmul
    /// kernel, and while the optimizer will insert a promoting cast, that cast
    /// is value-lossless but NOT accumulation-preserving. Silently honouring a
    /// "bf16" request would change numerics against every golden captured on
    /// the f32 path. The parameter stays for signature compatibility.
    ///
    /// `cache_stats` and `scheduler` are accepted for signature compatibility with
    /// callers and never written: this path's scheduling (board item 97) has no
    /// prefix-cache or admission-tracking concept of its own today. See the GAP
    /// notes on `SharedCacheStats`/`SharedScheduler` above.
    pub fn start(
        model_path: impl Into<PathBuf>,
        max_batch_size: usize,
        context_length: usize,
        dtype: Option<String>,
        max_queue_depth: usize,
        _cache_stats: SharedCacheStats,
        _scheduler: SharedScheduler,
    ) -> Result<InferenceRequestSender> {
        let model_path = model_path.into();
        let (tx, rx): (SyncSender<InferenceJob>, Receiver<InferenceJob>) =
            std::sync::mpsc::sync_channel(max_queue_depth);

        if let Some(d) = dtype.as_deref() {
            if d != "f32" {
                tracing::warn!(
                    "fuel-engine ignores dtype={d:?} and loads f32; see the module docs on \
                     accumulation-preserving casts"
                );
            }
        }

        std::thread::spawn(move || {
            // `scheduled_llama_engine::run_fuel_engine` (board item 97) loads
            // the checkpoint — dispatching on `.gguf` vs a SafeTensors
            // directory, same as before — and architecture (Llama vs Qwen3,
            // from `general.architecture`) — then drives it through the
            // batched `FuelSchedulerDriver`/`run_scheduled_jobs` path (the
            // multi-session scheduler). The old serial `FuelEngineModel`/
            // `run_jobs` fallback for multi-EOS checkpoints was deleted once
            // fuel#307 widened `eos_ids` to `Vec<u32>` (PR #122); `run_jobs`
            // itself (the candlelight `ParallelModelManager` path's job loop)
            // was removed entirely 2026-10-22 along with candlelight.
            // GGUF-quantized loading via `fuel::QuantizedLlama3Model::
            // from_gguf` landed 2026-09-24; see `src/model_fuel/
            // loader_gguf.rs` for the loader and
            // `docs/FUEL-PORT-STATUS-2026-09-24.md` §6 for how that was
            // scoped. RE-MEASURED 2026-09-26 against fuel's `main` `580540f`
            // (2026-09-25T22:46Z fuel#244) — fuel's dequant dispatch now
            // wires 14 of 15 `GgmlDType` variants, GGUF's `Q6_K`
            // (`GgmlDType::Q6K`) included. The remaining gaps: `Q8_1` (typed
            // error, GAP-125) and GGUF's IQ/TQ/MXFP4/NVFP4 families, which
            // `GgmlDType` doesn't model at all — per its own doc comment
            // (`fuel-ir/src/quantized.rs:14`), not a GAP number (`GAP-339`/
            // `fuel#247` is a different, unrelated sub-byte hazard; cited
            // wrongly here once and retracted). See
            // `docs/FUEL-CAPABILITY-AUDIT-2026-09-24.md`'s superseded-by
            // note for the full history of this claim.
            println!("Loading Fuel model from {}", model_path.display());
            crate::model_fuel::scheduled_llama_engine::run_fuel_engine(
                &model_path,
                max_batch_size,
                context_length,
                rx,
            );
        });

        Ok(tx)
    }
}

/// Answer every queued job with an error, then exit.
///
/// Shared by `ModelRunner::start`'s own load-failure path and
/// `scheduled_llama_engine::run_fuel_engine`'s: a load failure must not leave
/// clients blocked on a channel that will never produce.
pub(crate) fn drain_with_error(rx: Receiver<InferenceJob>, msg: &str) {
    while let Ok(job) = rx.recv() {
        match job.response_mode {
            ResponseMode::Complete(resp_tx) => {
                let _ = resp_tx.send(Err(anyhow::anyhow!("{}", msg)));
            }
            ResponseMode::Streaming(stream_tx) => {
                let _ = stream_tx.send(Err(anyhow::anyhow!("{}", msg)));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Request, RequestContext};

    #[test]
    fn queue_depth_guard_increments_on_construction_and_decrements_on_drop() {
        let counter: SharedQueueDepth = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let guard = QueueDepthGuard::new(counter.clone());
        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 1);

        drop(guard);
        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 0);
    }

    /// Two overlapping jobs must both be counted -- not just that a single
    /// guard's own increment/decrement is paired correctly.
    #[test]
    fn queue_depth_guard_counts_multiple_overlapping_jobs() {
        let counter: SharedQueueDepth = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let first = QueueDepthGuard::new(counter.clone());
        let second = QueueDepthGuard::new(counter.clone());
        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 2);

        drop(first);
        assert_eq!(
            counter.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "dropping one of two overlapping jobs must leave the other counted"
        );

        drop(second);
        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 0);
    }

    fn ctx_with(max_new: usize, generated: usize, stopped_on_eos: bool) -> RequestContext {
        let mut c = RequestContext::new(Request {
            id: "t".to_string(),
            prompt: "p".to_string(),
            max_new_tokens: max_new,
        });
        c.start_decoding();
        for _ in 0..generated {
            c.record_token();
        }
        c.stopped_on_eos = stopped_on_eos;
        c.complete();
        c
    }

    /// Hitting the cap without EOS must report Length. This is the original
    /// defect: both backends call `ctx.complete()` on `tokens_generated >=
    /// max_new_tokens` alone — EOS-only completion would leak a KV-cache
    /// session on every ordinary budget-exhausted request — so `state`
    /// becomes `Completed` here too, exactly like the EOS case. Only
    /// `stopped_on_eos` (false here) tells them apart; a `finish_reason_for`
    /// that read `state` instead would report `Stop` and fail this test.
    #[test]
    fn hitting_the_cap_without_eos_is_length() {
        let c = ctx_with(8, 8, false);
        assert_eq!(finish_reason_for(&c), FinishReason::Length);
    }

    /// EOS must report Stop. Guards the inverse error — an implementation that
    /// hardcoded Length would pass the test above and fail this one.
    ///
    /// **The old ordering trap dissolved rather than moving, which is why
    /// there is no separate "EOS on the final allowed token" case here.** A
    /// state-based classifier had to check `Completed` before the token count,
    /// because an EOS landing on the last allowed token satisfies both at
    /// once; get the order wrong and that generation reports Length.
    /// `finish_reason_for` reads exactly one field, so `ctx_with(8, 8, true)`
    /// and `ctx_with(8, 3, true)` differ only in fields it never looks at —
    /// the same test twice. The `tokens_generated == max_new_tokens && EOS`
    /// case is therefore covered by this one, not by omission.
    ///
    /// What this canNOT cover: `stopped_on_eos` is assigned directly above,
    /// the way a backend assigns it, so nothing here exercises a producer.
    /// `tests/api_result_metadata.rs::eos_terminated_generation_reports_stop`
    /// is what catches a backend that stops setting the flag.
    #[test]
    fn completing_before_the_cap_is_stop() {
        let c = ctx_with(8, 3, true);
        assert_eq!(finish_reason_for(&c), FinishReason::Stop);
    }
}
