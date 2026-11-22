// SPDX-License-Identifier: MIT OR Apache-2.0
//! The GGUF serving path end to end on a SYNTHETIC GGUF (ROADMAP C1): header and metadata parse, tokenizer
//! rebuild, weight load into Fuel, the runner, generation, and the OpenAI response, with no downloaded model.
//!
//! The file is `support::gguf::constant_output_gguf_bytes`: tiny, F32, and its greedy output is one fixed token,
//! so the assertions are exact (a trained model's text is not). It proves the PATH works and reports honestly
//! (`finish_reason`, `usage`); it says nothing about a real checkpoint's quality, and nothing about quantised
//! dtypes (the tensors are F32).

use std::path::Path;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use lightbulb::api::{ApiConfig, AppState};
use lightbulb::engine::{MemoryAwareConfig, MemoryAwareScheduler, ModelRunner};

#[path = "support/gguf.rs"]
#[allow(dead_code)]
mod gguf_writer;

fn app_state_for(path: &Path) -> AppState {
    let scheduler = Arc::new(std::sync::Mutex::new(MemoryAwareScheduler::new(
        MemoryAwareConfig::default(),
    )));
    let (tx, ctrl_tx) = ModelRunner::start(
        path,
        1,
        512,
        Some("f32".to_string()),
        8,
        Arc::new(std::sync::Mutex::new(
            lightbulb::cache::PrefixCacheStats::default(),
        )),
        scheduler.clone(),
    )
    .expect("starting the model runner on the synthetic GGUF");
    AppState {
        scheduler,
        config: ApiConfig::default(),
        pg_backend: None,
        manager: None,
        models: lightbulb::api::ModelRegistry::single(
            "test-model",
            lightbulb::api::ModelEntry {
                inference_tx: Some(tx),
                cache_control: Some(ctrl_tx),
                chat_template: lightbulb::api::chat_template::resolve_for_serving(path),
                ..Default::default()
            },
        ),
        eos_monitor: Default::default(),
        security_sinks: Vec::new(),
        cache_stats: Arc::new(std::sync::Mutex::new(
            lightbulb::cache::PrefixCacheStats::default(),
        )),
        live_system: lightbulb::api::new_shared_system_metrics(),
        queue_depth: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
    }
}

/// POST a chat request for a synthetic GGUF whose output token is `token`; return the parsed body.
async fn chat(token: u32, max_tokens: u32) -> serde_json::Value {
    let body = chat_body(max_tokens, serde_json::json!({}));
    let bytes = post_chat(token, body).await;
    serde_json::from_slice(&bytes).expect("response was not JSON")
}

fn chat_body(max_tokens: u32, extra: serde_json::Value) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": "lightbulb-default",
        "messages": [{"role": "user", "content": "ab cd"}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    });
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    body
}

/// The response body (200 asserted) of `body` posted to a server over the synthetic GGUF whose output is `token`.
async fn post_chat(token: u32, body: serde_json::Value) -> Vec<u8> {
    post_chat_gguf(gguf_writer::constant_output_gguf_bytes(token), body).await
}

/// [`post_chat`] over the given GGUF file contents.
async fn post_chat_gguf(gguf: Vec<u8>, body: serde_json::Value) -> Vec<u8> {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("synthetic.gguf");
    std::fs::write(&path, gguf).expect("writing the gguf");
    let app = lightbulb::api::openai::routes().with_state(app_state_for(&path));
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .expect("router returned no response");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    bytes.to_vec()
}

/// A GGUF whose model never emits the EOS runs to its cap and is served as such: `finish_reason: "length"`,
/// `completion_tokens` exactly the cap, and the content is the favoured token's text, six times (token 7 is
/// `&`), so the tokenizer rebuilt from the file's own vocabulary decoded what the weights produced.
#[tokio::test]
async fn a_synthetic_gguf_is_served_to_its_cap() {
    let v = chat(7, 6).await;
    assert_eq!(v["choices"][0]["finish_reason"], "length", "{v}");
    assert_eq!(v["usage"]["completion_tokens"], 6, "{v}");
    assert_eq!(v["choices"][0]["message"]["content"], "&&&&&&", "{v}");
}

/// The stop arm: the same file with the EOS as its output token stops on the first token.
#[tokio::test]
async fn a_synthetic_gguf_whose_model_emits_eos_is_served_as_stopped() {
    let v = chat(1, 32).await;
    assert_eq!(v["choices"][0]["finish_reason"], "stop", "{v}");
    let completion = v["usage"]["completion_tokens"]
        .as_u64()
        .expect("completion_tokens");
    assert!(
        (1..32).contains(&completion),
        "stopped early, not at the cap: {v}"
    );
}

/// Logprobs end to end through real Fuel on the synthetic GGUF, non-streaming: one entry per generated token, the
/// token text decoded from the file's own vocabulary, and, because the model's output row for `&` dominates
/// (logit 16 against at most 0.8), a logprob within a hair of 0 whose top alternative is that token.
#[tokio::test]
async fn a_synthetic_gguf_returns_logprobs() {
    let body = chat_body(4, serde_json::json!({"logprobs": true, "top_logprobs": 2}));
    let v: serde_json::Value =
        serde_json::from_slice(&post_chat(7, body).await).expect("response was not JSON");
    let content = v["choices"][0]["logprobs"]["content"]
        .as_array()
        .expect("logprobs.content");
    assert_eq!(content.len(), 4, "one entry per generated token: {v}");
    for entry in content {
        assert_eq!(entry["token"], "&", "{v}");
        let logprob = entry["logprob"].as_f64().expect("logprob");
        assert!(
            (-1e-3..=0.0).contains(&logprob),
            "a dominant token has logprob ~0: {v}"
        );
        assert_eq!(
            entry["top_logprobs"]
                .as_array()
                .expect("top_logprobs")
                .len(),
            2,
            "{v}"
        );
        assert_eq!(entry["top_logprobs"][0]["token"], "&", "{v}");
    }
}

/// The same file with every matrix quantised to Q8_0 (norms F32): the quantised load and matmul path serves it,
/// and the output is the same fixed token with the same logprob shape (a dominant token, logprob near 0), so the
/// dequantised weights are close enough that the decision does not move. Fails if the file does not load, or the
/// quantised path decodes anything but `&`.
#[tokio::test]
async fn a_q8_0_synthetic_gguf_is_served_and_decodes_the_same_token() {
    let body = chat_body(4, serde_json::json!({"logprobs": true, "top_logprobs": 1}));
    let bytes = post_chat_gguf(gguf_writer::constant_output_gguf_q8_0_bytes(7), body).await;
    let v: serde_json::Value = serde_json::from_slice(&bytes).expect("response was not JSON");
    assert_eq!(v["choices"][0]["finish_reason"], "length", "{v}");
    assert_eq!(v["choices"][0]["message"]["content"], "&&&&", "{v}");
    let first = &v["choices"][0]["logprobs"]["content"][0];
    assert!(first["logprob"].as_f64().expect("logprob") > -1e-2, "{v}");
}

/// Q4_0, the format of this repo's own real TinyLlama file: same file, 4-bit matrices, still the same fixed token
/// (the dominant row survives 4-bit quantisation; the logprob is looser than Q8_0's, so the bound is too).
#[tokio::test]
async fn a_q4_0_synthetic_gguf_is_served_and_decodes_the_same_token() {
    let body = chat_body(4, serde_json::json!({"logprobs": true, "top_logprobs": 1}));
    let bytes = post_chat_gguf(gguf_writer::constant_output_gguf_q4_0_bytes(7), body).await;
    let v: serde_json::Value = serde_json::from_slice(&bytes).expect("response was not JSON");
    assert_eq!(v["choices"][0]["finish_reason"], "length", "{v}");
    assert_eq!(v["choices"][0]["message"]["content"], "&&&&", "{v}");
    let first = &v["choices"][0]["logprobs"]["content"][0];
    assert!(first["logprob"].as_f64().expect("logprob") > -5e-2, "{v}");
}

/// The quantisers are not no-ops: the Q8_0 file is under half the F32 one (34 bytes per 32 values against 128 for
/// the matrices) and the Q4_0 file smaller still (18), so the tests above ran different files, not the F32 one under
/// other names.
#[test]
fn the_quantised_files_are_smaller_than_the_f32_one() {
    let f32_len = gguf_writer::constant_output_gguf_bytes(7).len();
    let q8_len = gguf_writer::constant_output_gguf_q8_0_bytes(7).len();
    let q4_len = gguf_writer::constant_output_gguf_q4_0_bytes(7).len();
    assert!(q8_len * 2 < f32_len, "q8_0 {q8_len} vs f32 {f32_len} bytes");
    assert!(q4_len < q8_len, "q4_0 {q4_len} vs q8_0 {q8_len} bytes");
}
