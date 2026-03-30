use anyhow::Result;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use tokio::sync::{mpsc as async_mpsc, oneshot};

use crate::engine::{Request, RequestContext};
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
                Ok(mut manager) => {
                    println!(
                        "Model loaded at {} (format: {})",
                        model_path.display(),
                        if is_gguf { "GGUF" } else { "SafeTensors" }
                    );

                    // Process incoming jobs
                    while let Ok(job) = rx.recv() {
                        let req = Request {
                            id: job.id.clone(),
                            prompt: job.prompt.clone(),
                            max_new_tokens: job.max_new_tokens,
                        };

                        let ctx = RequestContext::new(req);
                        let mut batch = vec![ctx];

                        // Process based on response mode (move out of job to avoid borrow issues)
                        match job.response_mode {
                            ResponseMode::Streaming(stream_tx) => {
                                // Streaming mode: decode and send tokens incrementally
                                loop {
                                    match manager.forward_batch(&mut batch) {
                                        Ok(_) => {
                                            // Get the most recent token and decode it
                                            if let Some(&last_token) =
                                                batch[0].generated_tokens.last()
                                            {
                                                match manager.decode(&[last_token], false) {
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
                                    match manager.forward_batch(&mut batch) {
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

                                // Decode all generated tokens at once
                                if final_result.is_none() {
                                    let generated_tokens = batch[0].generated_tokens.clone();
                                    match manager.decode(&generated_tokens, false) {
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
                Err(e) => {
                    // Model failed to load; consume any pending jobs and return error message
                    eprintln!("Failed to load model at {}: {:#}", model_path.display(), e);
                    while let Ok(job) = rx.recv() {
                        let error_msg = format!("model load failed: {}", e);
                        match job.response_mode {
                            ResponseMode::Complete(resp_tx) => {
                                let _ = resp_tx.send(Err(anyhow::anyhow!("{}", error_msg)));
                            }
                            ResponseMode::Streaming(stream_tx) => {
                                let _ = stream_tx.send(Err(anyhow::anyhow!("{}", error_msg)));
                            }
                        }
                    }
                }
            }
        });

        Ok(tx)
    }
}
