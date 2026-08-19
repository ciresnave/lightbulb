//! Serving a real GGUF end to end.
//!
//! Known-red baseline, measured 2026-08-14 BEFORE this work, on
//! `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` @ `52e7645b`, Q4_0, over HTTP at
//! `temperature: 0.0`:
//!
//! ```text
//! content: "| ass istant | <0x0A> | ass istant | ass istant | | ass istant |"
//! Chat template … resolved via Registry (bos "", eos "")
//! ```
//!
//! The file declares `tokenizer.chat_template` and ids 1/2 → `<s>`/`</s>`;
//! resolution simply never looked inside it. Re-running this test against a
//! commit before that fix must reproduce the garbage above — that is what makes
//! this gate a gate rather than a description.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use lightbulb::api::chat_template::{self, Resolution};
use lightbulb::api::{ApiConfig, AppState};
use lightbulb::engine::model_runner::ModelRunner;
use lightbulb::engine::{MemoryAwareScheduler, memory_aware_scheduler::MemoryAwareConfig};
use std::path::PathBuf;
use std::sync::Arc;
use tower::ServiceExt;

const MISSING: &str = "no GGUF checkpoint. Set LIGHTBULB_GGUF to a .gguf file.";

fn gguf_path() -> Option<PathBuf> {
    let p = match std::env::var_os("LIGHTBULB_GGUF") {
        Some(v) => PathBuf::from(v),
        None => PathBuf::from(
            "C:/Users/cires/.cache/huggingface/hub/models--TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF/snapshots/52e7645ba7c309695bec7ac98f4f005b139cf465/tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
        ),
    };
    p.is_file().then_some(p)
}

#[tokio::test]
#[ignore = "needs the GGUF checkpoint"]
async fn a_gguf_is_served_with_its_own_template() {
    let path = gguf_path().expect(MISSING);

    // 1. The file's own declaration is what resolution used. Exact strings, not
    //    `!is_empty()`: empty is precisely what the defect produced.
    let tk = chat_template::special_tokens(&path);
    assert_eq!(tk.bos, "<s>", "BOS did not come from GGUF metadata");
    assert_eq!(tk.eos, "</s>", "EOS did not come from GGUF metadata");
    assert_eq!(
        chat_template::resolve(&path).resolved_by,
        Resolution::GgufMetadata,
        "resolution fell through to a guess instead of reading the file"
    );

    // 2. And the model answers, rather than free-associating.
    let tx = ModelRunner::start(&path, 1, 512, Some("f32".to_string()))
        .expect("starting the model runner");
    let state = AppState {
        scheduler: Arc::new(MemoryAwareScheduler::new(MemoryAwareConfig::default())),
        config: ApiConfig::default(),
        db_pool: None,
        inference_tx: Some(tx),
        chat_template: chat_template::resolve_for_serving(&path),
        eos_monitor: Default::default(),
    };
    let app = lightbulb::api::openai::routes().with_state(state);
    let body = serde_json::json!({
        "model": "tinyllama",
        "messages": [{"role": "user", "content": "Name the capital of France."}],
        "max_tokens": 24,
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
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let content = v["choices"][0]["message"]["content"].as_str().unwrap();
    eprintln!("completion: {content:?}");

    assert!(
        content.contains("Paris"),
        "the model did not answer the question: {content:?}"
    );
    // The recorded pre-fix output; named so a regression is recognisable rather
    // than just a failed `contains`.
    assert!(
        !content.contains("ass istant"),
        "this is the pre-fix free-association output: {content:?}"
    );
}
