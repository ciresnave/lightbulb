use anyhow::Result;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use tokio::sync::{mpsc as async_mpsc, oneshot};

use crate::engine::{Request, RequestContext, RequestState};
use crate::model::ParallelModelManager;

/// Response mode for inference jobs
#[derive(Debug)]
pub enum ResponseMode {
    /// Complete response - send final text when done
    Complete(oneshot::Sender<anyhow::Result<String>>),
    /// Streaming response - send tokens as they are generated
    Streaming(async_mpsc::UnboundedSender<Result<String>>),
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
    ///
    /// `Some(tok)` means that request produced `tok` this step. **`None` means
    /// it produced no token and that is not an error** — it is mid-chunked-
    /// prefill or already stopped. Errors are the `Err` arm and abort the whole
    /// batch. An implementation returning `Err` where `None` is meant turns
    /// ordinary prefill progress into a failed request.
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
                // Streaming mode: decode and send tokens incrementally
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
                                        if stream_tx.send(Ok(token_text)).is_err() {
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
                                break;
                            }
                        }
                        Err(e) => {
                            let _ = stream_tx.send(Err(e));
                            break;
                        }
                    }
                }
            }
            ResponseMode::Complete(resp_tx) => {
                // Complete mode: generate all tokens then send final text
                let mut final_result: Option<anyhow::Result<String>> = None;

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
                        Ok(text) => final_result = Some(Ok(text)),
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
