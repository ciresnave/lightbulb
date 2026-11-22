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
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("synthetic.gguf");
    std::fs::write(&path, gguf_writer::constant_output_gguf_bytes(token))
        .expect("writing the gguf");
    let app = lightbulb::api::openai::routes().with_state(app_state_for(&path));
    let body = serde_json::json!({
        "model": "lightbulb-default",
        "messages": [{"role": "user", "content": "ab cd"}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    });
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
    serde_json::from_slice(&bytes).expect("response was not JSON")
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
