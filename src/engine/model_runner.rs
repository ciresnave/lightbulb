use anyhow::Result;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use tokio::sync::{mpsc as async_mpsc, oneshot};

use crate::engine::{Request, RequestContext, RequestState};
// Only the `not(fuel-engine)` `ModelRunner::start` names this bare — the
// `impl EngineModel for crate::model::ParallelModelManager` below spells the
// type out fully qualified, so it does not keep this import alive on its
// own. Under `fuel-engine`, nothing in this file uses the bare name, so an
// ungated import is an unused-import warning on that build.
#[cfg(not(feature = "fuel-engine"))]
use crate::model::ParallelModelManager;

/// Why generation stopped, in OpenAI's vocabulary.
///
/// Two variants only. OpenAI also defines `content_filter` and `tool_calls`;
/// nothing in Lightbulb produces either, and an enum carrying variants no
/// code can construct is worse than one that grows when a producer appears.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// The model emitted an end-of-sequence token.
    Stop,
    /// `max_new_tokens` was reached with no EOS. Continuing is meaningful,
    /// which is why a client needs to be able to tell this from `Stop`.
    Length,
}

impl FinishReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
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
    /// Response mode (complete or streaming)
    pub response_mode: ResponseMode,
}

/// Sender type used by API handlers to enqueue inference jobs
pub type InferenceRequestSender = Sender<InferenceJob>;

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
    ///   not "produced no token". `FuelEngineModel::step_batch`
    ///   (`model_fuel/engine_model.rs`) uses `Some`/`None` closer to the
    ///   original wording. The two implementations do not agree with each
    ///   other, let alone with this doc.
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
    /// single-request batch for the request's entire lifetime — so
    /// `FuelEngineModel::step_batch`'s per-call session sweep (see
    /// `model_fuel/engine_model.rs`), which drops any session whose id is not
    /// in the current `batch`, is safe: "absent ⇒ dead" and "absent ⇒ still
    /// live but not scheduled this round" are indistinguishable to that
    /// sweep, and only the single-request-per-batch caller shape makes the
    /// former the only real case. A future scheduler that steps a SUBSET of
    /// live requests per call (e.g. true multi-request batching) would
    /// violate this precondition silently: the sweep would free the KV cache
    /// of a request that is still running, and the next step for that
    /// request would either error on a missing session or, worse, get
    /// handed a fresh session and silently restart from position 0 — a
    /// cross-tenant correctness bug, not a leak. Any such scheduler must
    /// either keep every live request in every `batch`, or `step_batch`
    /// implementations that free on absence must be changed first.
    fn step_batch(&mut self, batch: &mut [RequestContext]) -> Result<Vec<Option<u32>>>;

    /// Detokenize.
    fn decode_text(&self, tokens: &[u32], skip_special: bool) -> Result<String>;
}

impl EngineModel for crate::model::ParallelModelManager {
    fn step_batch(&mut self, batch: &mut [RequestContext]) -> Result<Vec<Option<u32>>> {
        self.forward_batch(batch)
    }

    fn decode_text(&self, tokens: &[u32], skip_special: bool) -> Result<String> {
        self.decode(tokens, skip_special)
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
    #[cfg(not(feature = "fuel-engine"))]
    pub fn start(
        model_path: impl Into<PathBuf>,
        max_batch_size: usize,
        context_length: usize,
        dtype: Option<String>,
    ) -> Result<InferenceRequestSender> {
        let model_path = model_path.into();
        let (tx, rx): (Sender<InferenceJob>, Receiver<InferenceJob>) = std::sync::mpsc::channel();

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

                    run_jobs(manager, rx);
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
    #[cfg(feature = "fuel-engine")]
    pub fn start(
        model_path: impl Into<PathBuf>,
        _max_batch_size: usize,
        context_length: usize,
        dtype: Option<String>,
    ) -> Result<InferenceRequestSender> {
        let model_path = model_path.into();
        let (tx, rx): (Sender<InferenceJob>, Receiver<InferenceJob>) = std::sync::mpsc::channel();

        if let Some(d) = dtype.as_deref() {
            if d != "f32" {
                tracing::warn!(
                    "fuel-engine ignores dtype={d:?} and loads f32; see the module docs on \
                     accumulation-preserving casts"
                );
            }
        }

        std::thread::spawn(move || {
            // The candlelight arm above auto-detects `.gguf` and routes to
            // `load_gguf`. This arm has no GGUF loader at all — Fuel decode
            // support is SafeTensors-only today (`load_llama_f32_from_dir`
            // expects a directory with `model.safetensors`). Without this
            // check, a GGUF-configured server rebuilt with `fuel-engine`
            // would silently hand the `.gguf` file path to the SafeTensors
            // loader and fail with a confusing "missing model.safetensors"
            // error that names the wrong problem. Fail explicitly instead.
            let is_gguf = model_path
                .extension()
                .map_or(false, |ext| ext.eq_ignore_ascii_case("gguf"));
            if is_gguf {
                let msg = format!(
                    "fuel-engine does not yet support GGUF models (got {}); \
                     the Fuel path currently loads SafeTensors only via \
                     load_llama_f32_from_dir. Quantized/GGUF support is \
                     pending `impl FuelDecoder for Llama3Model`. Rebuild \
                     without --features fuel-engine to use this model, or \
                     point at a SafeTensors checkpoint directory.",
                    model_path.display()
                );
                eprintln!("{msg}");
                drain_with_error(rx, &msg);
                return;
            }

            println!("Loading Fuel model from {}", model_path.display());
            match crate::model_fuel::engine_model::FuelEngineModel::load(
                &model_path,
                context_length,
            ) {
                Ok(model) => {
                    println!("Fuel model loaded at {}", model_path.display());
                    run_jobs(model, rx);
                }
                Err(e) => {
                    eprintln!("Failed to load Fuel model at {}: {:#}", model_path.display(), e);
                    drain_with_error(rx, &format!("model load failed: {e}"));
                }
            }
        });

        Ok(tx)
    }
}

/// Answer every queued job with an error, then exit.
///
/// Shared by both backends: a load failure must not leave clients blocked on a
/// channel that will never produce.
fn drain_with_error(rx: Receiver<InferenceJob>, msg: &str) {
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

/// The job loop. Identical for every backend, which is why it is generic rather
/// than duplicated per `cfg`.
fn run_jobs<M: EngineModel>(mut model: M, rx: Receiver<InferenceJob>) {
    // Process incoming jobs
    while let Ok(job) = rx.recv() {
        let req = Request {
            id: job.id.clone(),
            prompt: job.prompt.clone(),
            max_new_tokens: job.max_new_tokens,
        };

        let mut ctx = RequestContext::new(req);
        ctx.temperature = job.temperature;
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
                            // Check if a tool call was detected (CR.1)
                            if batch[0].state.is_awaiting_tool_result() {
                                // Tool call detected — KV cache is preserved.
                                // For now, log and resume without tool execution.
                                // Future: execute tool via callback, then inject result.
                                if let RequestState::AwaitingToolResult {
                                    ref tool_name,
                                    ref tool_args,
                                    ..
                                } = batch[0].state
                                {
                                    eprintln!(
                                        "Tool call detected (no executor registered): {}({})",
                                        tool_name, tool_args
                                    );
                                }
                                // Resume decoding — model continues without tool result
                                // (graceful degradation when no tool executor is available)
                                batch[0].resume_decoding();
                                continue;
                            }

                            // Get the most recent token and decode it
                            if let Some(&last_token) =
                                batch[0].generated_tokens.last()
                            {
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
                // Complete mode: generate all tokens then send final text
                let mut final_result: Option<anyhow::Result<CompletionResult>> = None;

                loop {
                    match model.step_batch(&mut batch) {
                        Ok(_) => {
                            // Check if a tool call was detected (CR.1)
                            if batch[0].state.is_awaiting_tool_result() {
                                // Graceful degradation: resume without tool result
                                batch[0].resume_decoding();
                                continue;
                            }

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

                // Decode all generated tokens at once
                if final_result.is_none() {
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

                if let Some(res) = final_result {
                    let _ = resp_tx.send(res);
                }
            }
        }
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
    #[test]
    fn completing_before_the_cap_is_stop() {
        let c = ctx_with(8, 3, true);
        assert_eq!(finish_reason_for(&c), FinishReason::Stop);
    }

    /// EOS on the very last allowed token is Stop, not Length. Both
    /// `stopped_on_eos` and `tokens_generated == max_new_tokens` hold at
    /// once here; this guards that the flag — not the token count — is what
    /// decides the answer, unlike the old state-based version where the
    /// check order decided it.
    #[test]
    fn eos_on_the_final_allowed_token_is_stop() {
        let c = ctx_with(8, 8, true);
        assert_eq!(finish_reason_for(&c), FinishReason::Stop);
    }
}
