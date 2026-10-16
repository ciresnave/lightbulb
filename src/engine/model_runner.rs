use anyhow::Result;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, SyncSender};
use tokio::sync::{mpsc as async_mpsc, oneshot};

use crate::engine::{Request, RequestContext};
// Only the `not(fuel-engine)` `ModelRunner::start` names this bare — the
// `impl EngineModel for crate::model::ParallelModelManager` below spells the
// type out fully qualified, so it does not keep this import alive on its
// own. Under `fuel-engine`, nothing in this file uses the bare name, so an
// ungated import is an unused-import warning on that build.
#[cfg(not(feature = "fuel-engine"))]
use crate::model::ParallelModelManager;

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
/// `cache_stats` admin endpoint's data source (board item M4 slice 1). The
/// model-runner thread owns the real `ParallelModelManager`/`PrefixKvCache`
/// and never shares it; this is the one piece of state that crosses that
/// boundary outside the per-job `mpsc` channel, written by `run_jobs` after
/// each job and read by the HTTP layer without going through it.
pub type SharedCacheStats = std::sync::Arc<std::sync::Mutex<crate::cache::PrefixCacheStats>>;

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
pub(crate) trait EngineModel {
    /// Advance every request in `batch` by at most one token.
    ///
    /// Returns one entry per request, **positionally aligned with `batch`**.
    /// Errors are the `Err` arm and abort the whole batch. That much holds;
    /// see the next section before relying on anything about the `Ok` value.
    ///
    /// # Return value: currently unread, and implementors disagree about it
    ///
    /// This doc used to claim `Some(tok)` means "produced `tok` this step"
    /// and `None` means "produced no token and that is not an error — mid-
    /// chunked-prefill or already stopped." That description is not honoured
    /// by either side of the trait as it stands:
    ///
    /// - The only caller, `run_jobs` in this file, matches `Ok(_)` on the
    ///   whole `Vec` and never reads an individual entry — it checks
    ///   `batch[0].state`/`batch[0].generated_tokens` instead. Nothing today
    ///   depends on what the `Vec`'s elements contain.
    /// - `ParallelModelManager::step_batch`
    ///   (`model/parallel_model_manager.rs:1513`) sets an entry to `None` on
    ///   the SAME step it pushes a token to `generated_tokens` and completes
    ///   the request — using `None` for "produced a token and is now done",
    ///   not "produced no token", contradicting this doc's original wording.
    ///
    /// Do not add a caller that relies on the per-entry `Option` meaning
    /// anything specific until this is reconciled — treat it as an unread
    /// implementation detail for now. (Reconciling `ParallelModelManager`'s
    /// behaviour is out of scope here; this paragraph only corrects the doc
    /// to match reality.)
    ///
    /// # Caller precondition: `batch` must be every live request, every call
    ///
    /// Every request that is still live (not yet completed/errored/dropped)
    /// MUST appear in `batch` on every `step_batch` call for as long as it
    /// stays live. An implementation is entitled to treat a live request's
    /// ABSENCE from `batch` as termination and free whatever per-request
    /// state it holds (e.g. a KV cache) as a result.
    ///
    /// This holds today only because the current sole caller (`run_jobs`)
    /// builds one `batch = vec![ctx]` per job and re-steps that same
    /// single-request batch for the request's entire lifetime — so an
    /// implementor that sweeps per-request state on every call, dropping
    /// anything whose id is not in the current `batch` (the now-deleted
    /// `FuelEngineModel::step_batch` did exactly this), is safe: "absent ⇒
    /// dead" and "absent ⇒ still live but not scheduled this round" are
    /// indistinguishable to that sweep, and only the single-request-per-batch
    /// caller shape makes the former the only real case. A future scheduler
    /// that steps a SUBSET of live requests per call (e.g. true
    /// multi-request batching) would violate this precondition silently: the
    /// sweep would free the KV cache of a request that is still running, and
    /// the next step for that request would either error on a missing
    /// session or, worse, get handed a fresh session and silently restart
    /// from position 0 — a cross-tenant correctness bug, not a leak. Any such
    /// scheduler must
    /// either keep every live request in every `batch`, or `step_batch`
    /// implementations that free on absence must be changed first.
    fn step_batch(&mut self, batch: &mut [RequestContext]) -> Result<Vec<Option<u32>>>;

    /// Detokenize.
    fn decode_text(&self, tokens: &[u32], skip_special: bool) -> Result<String>;

    /// Snapshot of the prefix cache's current stats, for diagnostics (the
    /// `cache_stats` admin endpoint — board item M4 slice 1). Default-provided
    /// and empty for an implementor with no such cache, so the test stubs in
    /// this module's own tests need no change.
    fn prefix_cache_stats(&self) -> crate::cache::PrefixCacheStats {
        crate::cache::PrefixCacheStats::default()
    }
}

impl EngineModel for crate::model::ParallelModelManager {
    fn step_batch(&mut self, batch: &mut [RequestContext]) -> Result<Vec<Option<u32>>> {
        self.forward_batch(batch)
    }

    fn decode_text(&self, tokens: &[u32], skip_special: bool) -> Result<String> {
        self.decode(tokens, skip_special)
    }

    fn prefix_cache_stats(&self) -> crate::cache::PrefixCacheStats {
        // Resolves to the inherent method of the same name
        // (`ParallelModelManager::prefix_cache_stats`), which Rust prefers
        // over recursing into this trait method.
        self.prefix_cache_stats()
    }
}

/// Lightweight model runner that owns a ParallelModelManager on a dedicated thread
/// and services incoming inference jobs via a blocking mpsc receiver.
///
/// Supports both SafeTensors (directory) and GGUF (single file) model formats,
/// auto-detected from the model path.
pub struct ModelRunner;

impl ModelRunner {
    /// Start the model runner thread. Returns a Sender used to submit inference jobs.
    ///
    /// This function loads the model on the spawned thread and blocks on the
    /// receiver to process jobs synchronously with the ParallelModelManager.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to either a directory containing `model.safetensors`
    ///   and `tokenizer.json`, or a `.gguf` file
    /// * `max_batch_size` - Maximum number of concurrent requests
    /// * `context_length` - Context window size
    /// * `dtype` - Data type for SafeTensors models ("f32", "f16", "bf16").
    ///   Ignored for GGUF models (dtype embedded in quantization format).
    /// * `max_queue_depth` - Maximum number of jobs the channel will hold
    ///   (queued plus the one being processed) before `try_send` on the
    ///   returned sender starts reporting [`std::sync::mpsc::TrySendError::Full`].
    /// * `cache_stats` - Shared handle `run_jobs` writes the prefix cache's
    ///   stats into after every job, for the `cache_stats` admin endpoint to
    ///   read without going through the inference channel.
    #[cfg(not(feature = "fuel-engine"))]
    pub fn start(
        model_path: impl Into<PathBuf>,
        max_batch_size: usize,
        context_length: usize,
        dtype: Option<String>,
        max_queue_depth: usize,
        cache_stats: SharedCacheStats,
    ) -> Result<InferenceRequestSender> {
        let model_path = model_path.into();
        let (tx, rx): (SyncSender<InferenceJob>, Receiver<InferenceJob>) =
            std::sync::mpsc::sync_channel(max_queue_depth);

        // Move a cloned dtype into the thread closure as an owned String (if provided)
        let thread_dtype = dtype.clone();

        std::thread::spawn(move || {
            // Auto-detect model format from path
            let is_gguf = model_path
                .extension()
                .map_or(false, |ext| ext.eq_ignore_ascii_case("gguf"));

            // Load model on this thread (blocking)
            let load_result = if is_gguf {
                println!("Loading GGUF model from {}", model_path.display());
                ParallelModelManager::load_gguf(
                    &model_path,
                    max_batch_size,
                    context_length,
                    None, // chunked_prefill_config (use defaults)
                    None, // device (auto-detect)
                )
            } else {
                let dtype_ref = thread_dtype.as_deref();
                println!("Loading SafeTensors model from {}", model_path.display());
                ParallelModelManager::load(
                    &model_path,
                    max_batch_size,
                    context_length,
                    dtype_ref,
                    None, // chunked_prefill_config (use defaults)
                )
            };

            match load_result {
                Ok(manager) => {
                    println!(
                        "Model loaded at {} (format: {})",
                        model_path.display(),
                        if is_gguf { "GGUF" } else { "SafeTensors" }
                    );

                    run_jobs(manager, rx, cache_stats);
                }
                Err(e) => {
                    // Model failed to load; consume any pending jobs and return error message
                    eprintln!("Failed to load model at {}: {:#}", model_path.display(), e);
                    drain_with_error(rx, &format!("model load failed: {e}"));
                }
            }
        });

        Ok(tx)
    }

    /// Start the model runner thread, serving through the Fuel path.
    ///
    /// `dtype` is accepted and IGNORED: this path loads all weights as f32 by
    /// construction. Fuel's CPU backend has no `[F32, BF16, F32]` matmul
    /// kernel, and while the optimizer will insert a promoting cast, that cast
    /// is value-lossless but NOT accumulation-preserving. Silently honouring a
    /// "bf16" request would change numerics against every golden captured on
    /// the f32 path. The parameter stays for signature compatibility.
    ///
    /// `cache_stats` is accepted for signature compatibility with the
    /// candlelight path and never written: this path's scheduling (board
    /// item 97) doesn't go through `run_jobs`, which is the only writer,
    /// and has no prefix-cache concept of its own today.
    #[cfg(feature = "fuel-engine")]
    pub fn start(
        model_path: impl Into<PathBuf>,
        max_batch_size: usize,
        context_length: usize,
        dtype: Option<String>,
        max_queue_depth: usize,
        _cache_stats: SharedCacheStats,
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
            // itself survives only as the candlelight (`ParallelModelManager`)
            // path's job loop, above.
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
/// Shared by both backends: a load failure must not leave clients blocked on a
/// channel that will never produce.
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

/// Write `model`'s current prefix-cache stats into the shared handle, for the
/// `cache_stats` admin endpoint. Pulled out of `run_jobs` so this slice's own
/// contribution to that already-large function's complexity count is a
/// single call, not a branch.
fn publish_cache_stats<M: EngineModel>(model: &M, cache_stats: &SharedCacheStats) {
    if let Ok(mut guard) = cache_stats.lock() {
        *guard = model.prefix_cache_stats();
    }
}

/// The job loop. Identical for every backend, which is why it is generic rather
/// than duplicated per `cfg`.
pub(crate) fn run_jobs<M: EngineModel>(
    mut model: M,
    rx: Receiver<InferenceJob>,
    cache_stats: SharedCacheStats,
) {
    // Process incoming jobs
    while let Ok(job) = rx.recv() {
        let req = Request {
            id: job.id.clone(),
            prompt: job.prompt.clone(),
            max_new_tokens: job.max_new_tokens,
        };

        let mut ctx = RequestContext::new(req);
        ctx.temperature = job.temperature;
        // Carried, not re-derived. The runner cannot tell a templated prompt
        // from a raw one by looking at it — `<s>` is ordinary text — so the
        // handler that built the prompt is the only place the answer exists.
        ctx.add_special_tokens = job.add_special_tokens;
        let mut batch = vec![ctx];

        // Process based on response mode (move out of job to avoid borrow issues)
        match job.response_mode {
            ResponseMode::Streaming(stream_tx) => {
                // Streaming mode: decode and send tokens incrementally.
                //
                // `finished_normally` distinguishes the loop's four `break`
                // paths: a dropped receiver, a `decode_text` failure, a
                // `step_batch` failure, and `should_continue()` going false.
                // Only the last is a normal completion — the other three have
                // either already sent `Err` or can no longer be received by
                // anyone — so it is the only path that sets the flag before
                // breaking.
                let mut finished_normally = false;
                loop {
                    match model.step_batch(&mut batch) {
                        Ok(_) => {
                            // Get the most recent token and decode it
                            if let Some(&last_token) = batch[0].generated_tokens.last() {
                                match model.decode_text(&[last_token], false) {
                                    Ok(token_text) => {
                                        if stream_tx
                                            .send(Ok(StreamItem::Token(token_text)))
                                            .is_err()
                                        {
                                            // Receiver dropped, stop generation
                                            break;
                                        }
                                    }
                                    Err(e) => {
                                        let _ = stream_tx.send(Err(e));
                                        break;
                                    }
                                }
                            }

                            if !batch[0].should_continue() {
                                finished_normally = true;
                                break;
                            }
                        }
                        Err(e) => {
                            let _ = stream_tx.send(Err(e));
                            break;
                        }
                    }
                }

                // Exactly one Done, carrying what the runner observed rather
                // than what the handler would otherwise assume. Sent on the
                // normal exit only — an error path already sent Err, and a
                // dropped receiver cannot receive this either.
                if finished_normally {
                    let _ = stream_tx.send(Ok(StreamItem::Done {
                        finish_reason: finish_reason_for(&batch[0]),
                    }));
                }
            }
            ResponseMode::Complete(resp_tx) => {
                // Complete mode: generate all tokens then send final text.
                //
                // Unlike Streaming, nothing here sends anything until the job
                // is done, so a dropped client would otherwise run to
                // completion unnoticed — burning a generation's worth of
                // compute for a `oneshot::Sender::send` whose `Err` nobody
                // reads. `resp_tx.is_closed()` is checked before every step:
                // it goes true the moment the handler's `resp_rx` is dropped,
                // which happens when the client disconnects and the request
                // future awaiting it is cancelled (see
                // `chat::run_inference_once`).
                let mut final_result: Option<anyhow::Result<CompletionResult>> = None;
                let mut cancelled = false;

                loop {
                    if resp_tx.is_closed() {
                        cancelled = true;
                        break;
                    }
                    match model.step_batch(&mut batch) {
                        Ok(_) => {
                            if !batch[0].should_continue() {
                                break;
                            }
                        }
                        Err(e) => {
                            final_result = Some(Err(e));
                            break;
                        }
                    }
                }

                // Decode all generated tokens at once. Skipped on
                // cancellation: there is no receiver left to decode for, and
                // `decode_text` is not free.
                if !cancelled && final_result.is_none() {
                    let generated_tokens = batch[0].generated_tokens.clone();
                    match model.decode_text(&generated_tokens, false) {
                        Ok(text) => {
                            final_result = Some(Ok(CompletionResult {
                                text,
                                prompt_tokens: batch[0].prompt_tokens,
                                completion_tokens: batch[0].tokens_generated,
                                finish_reason: finish_reason_for(&batch[0]),
                            }))
                        }
                        Err(e) => final_result = Some(Err(e)),
                    }
                }

                if !cancelled && let Some(res) = final_result {
                    let _ = resp_tx.send(res);
                }
            }
        }

        // Once per job, regardless of mode -- the one piece of state this
        // thread shares outside the per-job channel (board item M4 slice 1).
        publish_cache_stats(&model, &cache_stats);
    }

    println!("Model runner thread exiting (receiver closed)");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Request, RequestContext};

    /// Builds a context that has already been marked `Completed` — every
    /// real backend does that unconditionally on `is_eos ||
    /// tokens_generated >= max_new_tokens`, so `state` alone never
    /// distinguishes the two exits. `stopped_on_eos` is set directly here,
    /// the same way a real backend sets it: at the point completion is
    /// decided, not reconstructed afterward.
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

    /// Records what `run_jobs` handed the backend, then completes the request
    /// immediately so the loop terminates without a model.
    struct SpyModel {
        seen: std::sync::Arc<std::sync::Mutex<Vec<bool>>>,
    }

    impl EngineModel for SpyModel {
        fn step_batch(&mut self, batch: &mut [RequestContext]) -> Result<Vec<Option<u32>>> {
            // This is the value a real backend passes to `encode`.
            self.seen.lock().unwrap().push(batch[0].add_special_tokens);
            batch[0].complete();
            Ok(vec![None])
        }

        fn decode_text(&self, _tokens: &[u32], _skip_special: bool) -> Result<String> {
            Ok(String::new())
        }
    }

    /// `InferenceJob::add_special_tokens` must reach `RequestContext`.
    ///
    /// The hop is one assignment in `run_jobs`, and dropping it is invisible
    /// everywhere else: `RequestContext::new` defaults the field to `true`, so
    /// a runner that never copies the job's value serves every templated prompt
    /// with a doubled BOS while every other test stays green. Both values are
    /// asserted because a mutation to a constant passes on one of them.
    #[test]
    fn the_jobs_tokenizer_flag_reaches_the_request_context() {
        for want in [false, true] {
            let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            let (tx, rx) = std::sync::mpsc::channel();
            let (resp_tx, _resp_rx) = oneshot::channel();
            tx.send(InferenceJob {
                id: "spy".to_string(),
                prompt: "p".to_string(),
                max_new_tokens: 1,
                temperature: 0.0,
                add_special_tokens: want,
                response_mode: ResponseMode::Complete(resp_tx),
            })
            .unwrap();
            drop(tx); // so `run_jobs` returns rather than blocking forever

            run_jobs(
                SpyModel { seen: seen.clone() },
                rx,
                std::sync::Arc::new(std::sync::Mutex::new(
                    crate::cache::PrefixCacheStats::default(),
                )),
            );

            let seen = seen.lock().unwrap();
            assert_eq!(
                seen.as_slice(),
                &[want],
                "the backend saw add_special_tokens {:?} for a job that asked for {want}",
                seen.as_slice()
            );
        }
    }

    /// Counts `step_batch` calls and advances the request exactly like a real
    /// backend would (`start_decoding` on the first step, `record_token` on
    /// every step, a real token pushed onto `generated_tokens`) — so a job
    /// that is NOT cancelled would keep going for `max_new_tokens` steps,
    /// and the test can tell "never started" apart from "started and then
    /// got cut off early" rather than from a model that completes on its
    /// own regardless of cancellation.
    ///
    /// Pushing a token onto `generated_tokens` matters beyond realism: the
    /// streaming arm's disconnect check (`stream_tx.send(...).is_err()`,
    /// `model_runner.rs`) is reached only via `if let Some(&last_token) =
    /// batch[0].generated_tokens.last()` -- a model that never pushes a
    /// token leaves that branch permanently unreached, and the loop would
    /// then exit only through `should_continue()`'s normal-completion path
    /// after the full `max_new_tokens` budget, silently never exercising
    /// the disconnect check at all. Caught exactly this way: the streaming
    /// disconnect test below first failed with `calls == 5` (the full
    /// budget) instead of `1`, before this method pushed a token.
    struct CountingModel {
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl EngineModel for CountingModel {
        fn step_batch(&mut self, batch: &mut [RequestContext]) -> Result<Vec<Option<u32>>> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if batch[0].tokens_generated == 0 {
                batch[0].start_decoding();
            }
            batch[0].record_token();
            batch[0].generated_tokens.push(0);
            Ok(vec![None])
        }

        fn decode_text(&self, _tokens: &[u32], _skip_special: bool) -> Result<String> {
            Ok(String::new())
        }
    }

    /// A client that disconnected before the job even reached the runner
    /// must not be generated for at all: `run_jobs` must see
    /// `resp_tx.is_closed()` before the first `step_batch` call, not only
    /// between calls that already happened.
    ///
    /// Without the fix, `run_jobs` has no way to learn the receiver is gone
    /// and steps `CountingModel` for the job's full `max_new_tokens` budget
    /// (5 here) before giving up on sending a result nobody can read.
    #[test]
    fn a_dropped_complete_mode_receiver_stops_generation_before_it_starts() {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (tx, rx) = std::sync::mpsc::channel();
        let (resp_tx, resp_rx) = oneshot::channel::<anyhow::Result<CompletionResult>>();
        drop(resp_rx); // the client is already gone by the time the job is picked up

        tx.send(InferenceJob {
            id: "cancelled".to_string(),
            prompt: "p".to_string(),
            max_new_tokens: 5,
            temperature: 0.0,
            add_special_tokens: true,
            response_mode: ResponseMode::Complete(resp_tx),
        })
        .unwrap();
        drop(tx);

        run_jobs(
            CountingModel {
                calls: calls.clone(),
            },
            rx,
            std::sync::Arc::new(std::sync::Mutex::new(
                crate::cache::PrefixCacheStats::default(),
            )),
        );

        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "a job whose client already disconnected must not step the model at all"
        );
    }

    /// The streaming counterpart of the test above — and the reason it's a
    /// SEPARATE test rather than one more case of it: `ResponseMode::
    /// Streaming`'s loop (unlike `Complete`'s) has no proactive
    /// `is_closed()` check before stepping. It only learns the receiver is
    /// gone from `stream_tx.send(...).is_err()`, which it cannot call
    /// before producing a token to send — so a disconnected streaming
    /// client costs exactly ONE wasted `step_batch` call, not zero, before
    /// generation stops. Asserting `== 0` here would be the wrong
    /// expectation for this mode; asserting `< max_new_tokens` is what the
    /// roadmap's "Request cancellation support" item actually needed
    /// proven for streaming (board audit, #148): that a dropped client
    /// stops generation at all, not that it stops with zero overhead.
    #[test]
    fn a_dropped_streaming_receiver_stops_generation_after_one_wasted_step() {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (tx, rx) = std::sync::mpsc::channel();
        let (stream_tx, stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<anyhow::Result<StreamItem>>();
        drop(stream_rx); // the client is already gone by the time the job is picked up

        tx.send(InferenceJob {
            id: "cancelled".to_string(),
            prompt: "p".to_string(),
            max_new_tokens: 5,
            temperature: 0.0,
            add_special_tokens: true,
            response_mode: ResponseMode::Streaming(stream_tx),
        })
        .unwrap();
        drop(tx);

        run_jobs(
            CountingModel {
                calls: calls.clone(),
            },
            rx,
            std::sync::Arc::new(std::sync::Mutex::new(
                crate::cache::PrefixCacheStats::default(),
            )),
        );

        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "a dropped streaming receiver must stop generation after the first step's \
             send fails, not run to the job's full max_new_tokens budget (5 here)"
        );
    }
}
