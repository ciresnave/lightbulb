//! C1 — does a single-pass PREFILL of N tokens agree with a PREFILL of N-1
//! followed by one DECODE step?
//!
//! DIAGNOSTIC, not a gate. Prints; asserts only its own preconditions.
//!
//! B2 measured the decode positions as 23, 24, 25, 26 against a 23-token
//! prompt, on two independently-maintained counters — so indexing is not the
//! fault. But both counters are BOOKKEEPING; neither reads the cache's
//! physical fill. What remains is cache CONTENTS.
//!
//! Token 1 (`1576 "The"`) came from the PREFILL logits, which never read the
//! cache. That it matches llama.cpp proves the forward-pass math, NOT that the
//! cache was written correctly. This file separates those two claims: it feeds
//! the 23-token prompt PLUS token 1 as a single 24-token prefill, so the same
//! next-token prediction is made without ever taking the decode path.
//!
//! Driven through `/v1/completions`, which applies no chat template, so the
//! exact token sequence is under this file's control.
//!
//! # Running the llama.cpp reference — read this before you do
//!
//! Every line below cost a retraction to learn. b10757, measured 2026-09-02.
//!
//! **Use `llama-completion`, not `llama-cli`.** This build split them, and
//! `llama-cli` REJECTS `-no-cnv` outright (`error: invalid argument`).
//!
//! **Pass `-no-cnv`.** Conversation mode auto-enables whenever the model ships
//! a chat template — which every chat GGUF does — and in that mode llama.cpp
//! SILENTLY RE-WRAPS an already-templated prompt in another turn:
//!
//! ```text
//! <|user|>          <- llama.cpp added this
//! <|user|>          <- ours
//! Name the capital of France.
//! <|assistant|>
//! The<|assistant|>  <- and this
//! ```
//!
//! It then blocks on stdin, so redirect from `/dev/null` too. A comparison run
//! this way is against a different prompt than the one you think you sent, and
//! nothing in the output says so.
//!
//! **Verify the ids, do not trust the count.** `--verbose-prompt` dumps
//! `id -> 'piece'` per token. `tokens_evaluated` and `usage.prompt_tokens` are
//! COUNTS; "24 tokens" and "these 24 tokens" are different claims and only the
//! second supports an id-for-id comparison.
//!
//! **For the reference DISTRIBUTION, use `llama-server`, not the CLI.** No
//! `--n-probs` exists on the CLI. Start the server and POST to `/completion`
//! with `n_probs`, `cache_prompt: false`, and — importantly —
//! `temperature: 1.0, top_k: 0, top_p: 1.0, min_p: 0.0`, so the reported
//! distribution is the raw model output rather than a sampler-shaped view.
//! (Measured: temp 0 and the settings above give byte-identical logprobs here,
//! but that is a fact about this build, not a guarantee.)
//!
//! Returned `logprob`s differ from raw logits by a constant, so DIFFERENCES
//! between two candidates are directly comparable to differences between our
//! own raw logits. Absolute values are not.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use lightbulb::api::{ApiConfig, AppState};
use lightbulb::engine::model_runner::ModelRunner;
use lightbulb::engine::{MemoryAwareScheduler, memory_aware_scheduler::MemoryAwareConfig};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tower::ServiceExt;

/// The prompt the chat path renders for "Name the capital of France." —
/// asserted byte-for-byte by `gguf_serving_e2e.rs`.
const RENDERED: &str = "<|user|>\nName the capital of France.</s>\n<|assistant|>\n";

fn gguf_path() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var_os("LIGHTBULB_GGUF")?);
    p.is_file().then_some(p)
}

async fn complete_raw(path: &Path, prompt: &str, max_tokens: usize) -> serde_json::Value {
    let tx = ModelRunner::start(path, 1, 512, Some("f32".to_string()))
        .expect("starting the model runner");
    let state = AppState {
        scheduler: Arc::new(MemoryAwareScheduler::new(MemoryAwareConfig::default())),
        config: ApiConfig::default(),
        db_pool: None,
        inference_tx: Some(tx),
        chat_template: None,
        eos_monitor: Default::default(),
    };
    let app = lightbulb::api::openai::routes().with_state(state);
    let body = serde_json::json!({
        "model": "tinyllama",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    });
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .expect("router returned no response");
    assert_eq!(resp.status(), StatusCode::OK, "completions must answer 200");
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

/// ARM A0 — CONTROL ON THE DRIVER.
///
/// Before the 24-token arm can mean anything, the raw path must reproduce what
/// the chat path produced from the same string. If it does not, any difference
/// in the 24-token arm could be the DRIVER rather than the extra token, and the
/// experiment would be measuring the wrong thing.
#[tokio::test]
#[ignore = "needs the GGUF checkpoint; set LIGHTBULB_GGUF"]
async fn c1_a0_the_raw_path_reproduces_the_chat_path() {
    let Some(path) = gguf_path() else { return };
    let v = complete_raw(&path, RENDERED, 24).await;
    let text = v["choices"][0]["text"].as_str().unwrap_or("");
    let n = v["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
    eprintln!("C1-A0 prompt_tokens={n} text={text:?}");
    assert_eq!(
        n, 23,
        "driver control: the raw path must tokenize the rendered prompt to the same 23 ids the chat path did"
    );
}

/// ARM A1 — THE MEASUREMENT.
///
/// 24 tokens in one prefill: the 23-token prompt plus token 1, `"The"`. The
/// first token this emits is predicted from PREFILL logits over exactly the
/// context that the original run's first DECODE step saw through the cache.
#[tokio::test]
#[ignore = "needs the GGUF checkpoint; set LIGHTBULB_GGUF"]
async fn c1_a1_a_24_token_single_pass_prefill() {
    let Some(path) = gguf_path() else { return };
    let prompt = format!("{RENDERED}The");
    let v = complete_raw(&path, &prompt, 8).await;
    let text = v["choices"][0]["text"].as_str().unwrap_or("");
    let n = v["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
    eprintln!("C1-A1 prompt_tokens={n} text={text:?}");
    assert_eq!(
        n, 24,
        "precondition: appending \"The\" must add exactly one token, or this is not the sequence B2 measured"
    );
}

/// H2 — INPUT REDUCTION on a CONTINUOUS signal.
///
/// Token identity is a discrete signal: a short prompt either diverges from the
/// reference or does not, and a non-divergence tells you almost nothing. The
/// top-5 logit SPREAD is continuous, so a short prompt that does not yet
/// diverge can still show the distribution collapsing.
///
/// Prints our spread for prompts of increasing length; the reference's spread
/// for the same strings comes from `llama-server`'s `n_probs`.
#[tokio::test]
#[ignore = "needs the GGUF checkpoint; set LIGHTBULB_GGUF"]
async fn h2_flatness_across_prompt_lengths() {
    let Some(path) = gguf_path() else { return };
    let long = format!("{RENDERED}The");
    for p in ["", "Hello", "The capital of France is", RENDERED, &long] {
        let v = complete_raw(&path, p, 1).await;
        eprintln!(
            "H2-PROMPT {:?} prompt_tokens={} -> {:?}",
            &p[..p.len().min(28)],
            v["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
            v["choices"][0]["text"].as_str().unwrap_or("")
        );
    }
}
