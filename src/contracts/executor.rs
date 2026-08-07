//! Generic contract executor — the retry/fallback loop.
//!
//! This module contains `execute_contract`, which drives the full
//! structured-output loop:
//!
//! 1. Inject contract instructions into the message list.
//! 2. Convert messages → prompt and call the inference closure.
//! 3. Try to parse the response; if it fails, append model output as context,
//!    add a tightening message, and retry up to `max_attempts` times.
//! 4. On primary contract exhaustion, repeat with each fallback contract.
//! 5. Return a [`ContractExecutionResult`] either way.
//!
//! # Testability
//!
//! The inference callable is a generic `Fn(String) -> impl Future<Output =
//! anyhow::Result<String>>`.  Production code passes a closure that wraps the
//! `mpsc::Sender` to the model runner thread.  Tests pass a closure that
//! returns scripted responses, so every branch of this loop can be exercised
//! without a real model.

use std::future::Future;

use super::{ContractOutput, ContractResult, OutputContractSpec};
use super::validation::{self, RawMessage};

// ─── Convenience wrapper for ModelRunner-backed inference ──────────────────

/// Run `execute_contract` using a live [`ModelRunner`] sender.
///
/// This is the public entry point for code that already holds an
/// `mpsc::Sender<InferenceJob>` (e.g. integration tests, CLI tools) and
/// doesn't want to wire up the closure manually.
///
/// [`ModelRunner`]: crate::engine::model_runner::ModelRunner
pub async fn execute_contract_with_runner(
    tx: std::sync::mpsc::Sender<crate::engine::model_runner::InferenceJob>,
    messages: &[RawMessage],
    model_id: &str,
    primary: &OutputContractSpec,
    max_attempts: u32,
    fallbacks: &[OutputContractSpec],
    max_new_tokens: usize,
    temperature: f64,
) -> anyhow::Result<ContractExecutionResult> {
    execute_contract(
        messages,
        model_id,
        primary,
        max_attempts,
        fallbacks,
        move |prompt| {
            let tx = tx.clone();
            async move {
                use crate::engine::model_runner::{InferenceJob, ResponseMode};
                let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
                let job = InferenceJob {
                    id: uuid::Uuid::new_v4().to_string(),
                    prompt,
                    max_new_tokens,
                    temperature,
                    response_mode: ResponseMode::Complete(resp_tx),
                };
                tx.send(job)
                    .map_err(|e| anyhow::anyhow!("failed to send inference job: {}", e))?;
                resp_rx
                    .await
                    .map_err(|e| anyhow::anyhow!("inference runner dropped: {}", e))?
                    .map(|r| r.text)
            }
        },
    )
    .await
}

// ─── Public result type ────────────────────────────────────────────────────

/// Everything the caller needs after the loop is done.
#[derive(Debug)]
pub struct ContractExecutionResult {
    /// The structured (or raw-fallback) output.
    pub result: ContractResult,
    /// The final raw text from the model (used as the `choices[0].content`
    /// in the HTTP response).
    pub final_text: String,
}

// ─── Core loop ─────────────────────────────────────────────────────────────

/// Drive the contract retry/fallback loop.
///
/// # Parameters
///
/// - `messages`     — the caller's original message list (will be cloned /
///                    mutated per-contract internally; caller's copy is untouched).
/// - `model_id`     — used only for logging and the returned `ContractResult`.
/// - `primary`      — the first contract to attempt.
/// - `max_attempts` — how many inference calls to make per contract before
///                    moving to the next one.
/// - `fallbacks`    — additional contracts tried (in order) after `primary`
///                    exhausts all attempts.
/// - `infer`        — async callable `(prompt: String) -> Result<String>`.
///                    Called once per attempt.
pub async fn execute_contract<F, Fut>(
    messages: &[RawMessage],
    model_id: &str,
    primary: &OutputContractSpec,
    max_attempts: u32,
    fallbacks: &[OutputContractSpec],
    infer: F,
) -> anyhow::Result<ContractExecutionResult>
where
    F: Fn(String) -> Fut,
    Fut: Future<Output = anyhow::Result<String>>,
{
    let start = std::time::Instant::now();
    let mut total_attempts: u32 = 0;
    let mut last_raw = String::new();

    let contracts: Vec<&OutputContractSpec> = std::iter::once(primary)
        .chain(fallbacks.iter())
        .collect();

    for contract in &contracts {
        // Fresh message list for each contract so fallbacks start clean.
        let mut msgs: Vec<RawMessage> = messages
            .iter()
            .map(|m| RawMessage { role: m.role.clone(), content: m.content.clone() })
            .collect();

        validation::inject_contract_instruction(&mut msgs, contract);

        for attempt in 0..max_attempts {
            total_attempts += 1;

            if attempt > 0 {
                msgs.push(RawMessage {
                    role: "user".to_string(),
                    content: validation::tightening_message(attempt, contract),
                });
            }

            let prompt = validation::messages_to_prompt(&msgs);

            let attempt_start = std::time::Instant::now();
            let raw_text = infer(prompt).await?;
            let attempt_ms = attempt_start.elapsed().as_millis() as u64;

            last_raw = raw_text.clone();

            let parsed = validation::try_parse(&raw_text, contract);
            let parse_ok = parsed
                .as_ref()
                .map(|o| validation::is_parse_successful(o, contract))
                .unwrap_or(false);

            validation::log_attempt(
                contract.type_name(),
                total_attempts,
                parse_ok,
                model_id,
                attempt_ms,
            );

            if parse_ok {
                let output = parsed.unwrap();
                let latency_ms = start.elapsed().as_millis() as u64;
                return Ok(ContractExecutionResult {
                    final_text: raw_text,
                    result: ContractResult {
                        contract_type: contract.type_name().to_string(),
                        attempts: total_attempts,
                        parse_success: true,
                        model_id: model_id.to_string(),
                        latency_ms,
                        output,
                    },
                });
            }

            // Append bad assistant response as context before the next retry.
            if attempt + 1 < max_attempts {
                msgs.push(RawMessage {
                    role: "assistant".to_string(),
                    content: raw_text,
                });
            }
        }
    }

    // Everything failed.
    let latency_ms = start.elapsed().as_millis() as u64;
    tracing::error!(
        model = model_id,
        total_attempts,
        latency_ms,
        "all contract attempts failed; returning raw fallback"
    );

    Ok(ContractExecutionResult {
        final_text: last_raw.clone(),
        result: ContractResult {
            contract_type: primary.type_name().to_string(),
            attempts: total_attempts,
            parse_success: false,
            model_id: model_id.to_string(),
            latency_ms,
            output: ContractOutput::Raw { text: last_raw },
        },
    })
}
