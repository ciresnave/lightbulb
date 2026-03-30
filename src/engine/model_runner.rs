use anyhow::Result;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use tokio::sync::{mpsc as async_mpsc, oneshot};

use crate::engine::{Request, RequestContext};
use crate::model::ModelManager;

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

/// Lightweight model runner that owns a ModelManager on a dedicated thread
/// and services incoming inference jobs via a blocking mpsc receiver.
pub struct ModelRunner;

impl ModelRunner {
    /// Start the model runner thread. Returns a Sender used to submit inference jobs.
    ///
    /// This function loads the model on the spawned thread and blocks on the
    /// receiver to process jobs synchronously with the ModelManager.
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
            // Load model on this thread (blocking)
            let dtype_ref = thread_dtype.as_deref();
            match ModelManager::load(&model_path, max_batch_size, context_length, dtype_ref) {
                Ok(mut manager) => {
                    println!("Model loaded at {}", model_path.display());

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
