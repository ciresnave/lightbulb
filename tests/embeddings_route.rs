// SPDX-License-Identifier: MIT OR Apache-2.0
//! `POST /v1/embeddings` over the real router with a STUB encoder (ROADMAP F3, design note test plan 5): status
//! mapping, wrong-kind models, formats, usage, and that an embedding entry never answers chat. No model is
//! downloaded; the Fuel-backed encoder's vectors are a separate gate.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use lightbulb::api::openai::embeddings::{
    Embedded, Embedder, EmbeddingError, EmbeddingSpec, LoadedEmbedder, Pooling,
};
use lightbulb::api::{ApiConfig, AppState, ModelEntry, ModelRegistry};
use lightbulb::engine::{MemoryAwareConfig, MemoryAwareScheduler};

const NATIVE: usize = 4;

/// Token count = byte length; the vector is `[len, 1, 2, 3]`, or NaN-poisoned for the text "nan".
struct Stub;
impl Embedder for Stub {
    fn count_tokens(&self, t: &str) -> Result<usize, EmbeddingError> {
        Ok(t.len())
    }
    fn embed(&self, t: &str) -> Result<Embedded, EmbeddingError> {
        let first = if t == "nan" { f32::NAN } else { t.len() as f32 };
        Ok(Embedded {
            vector: vec![first, 1.0, 2.0, 3.0],
            tokens: t.len(),
        })
    }
}

fn state() -> AppState {
    let models = ModelRegistry::default();
    // Sorts BEFORE "chat": if `entry()` did not skip it, the chat lookup below would return it.
    models.set(
        "a-embed",
        ModelEntry {
            embedding: Some(Arc::new(LoadedEmbedder {
                spec: EmbeddingSpec {
                    pooling: Pooling::Mean,
                    normalize: true,
                    dim: NATIVE,
                    matryoshka: vec![2],
                    max_tokens: 8,
                    max_items: 3,
                },
                embedder: Box::new(Stub),
            })),
            ..Default::default()
        },
    );
    models.set(
        "chat",
        ModelEntry {
            chat_template: None,
            ..Default::default()
        },
    );
    AppState {
        scheduler: Arc::new(std::sync::Mutex::new(MemoryAwareScheduler::new(
            MemoryAwareConfig::default(),
        ))),
        config: ApiConfig::default(),
        pg_backend: None,
        manager: None,
        models,
        eos_monitor: Default::default(),
        security_sinks: Vec::new(),
        cache_stats: Arc::new(std::sync::Mutex::new(
            lightbulb::cache::PrefixCacheStats::default(),
        )),
        live_system: lightbulb::api::new_shared_system_metrics(),
        queue_depth: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
    }
}

async fn post(body: serde_json::Value) -> (StatusCode, serde_json::Value) {
    let app = lightbulb::api::openai::routes().with_state(state());
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
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
    (status, serde_json::from_slice(&bytes).unwrap_or_default())
}

#[tokio::test]
async fn a_string_and_an_array_are_embedded_in_order_with_usage() {
    let (s, v) = post(serde_json::json!({"model":"a-embed","input":["abc","de"]})).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["object"], "list");
    assert_eq!(v["model"], "a-embed");
    assert_eq!(v["data"].as_array().unwrap().len(), 2);
    assert_eq!(v["data"][0]["index"], 0);
    assert_eq!(v["data"][1]["index"], 1);
    assert_eq!(v["usage"]["prompt_tokens"], 5);
    let e = v["data"][0]["embedding"].as_array().unwrap();
    assert_eq!(e.len(), NATIVE);
    let norm: f64 = e.iter().map(|x| x.as_f64().unwrap().powi(2)).sum();
    assert!((norm - 1.0).abs() < 1e-5, "unit norm, got {norm}");
}

#[tokio::test]
async fn base64_and_dimensions_are_honoured_and_a_bad_width_is_a_400() {
    let (s, v) = post(
        serde_json::json!({"model":"a-embed","input":"ab","encoding_format":"base64","dimensions":2}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert!(v["data"][0]["embedding"].is_string());
    // base64 of 2 little-endian f32 = 8 bytes = 12 chars with padding
    assert_eq!(v["data"][0]["embedding"].as_str().unwrap().len(), 12);

    let (s, v) = post(serde_json::json!({"model":"a-embed","input":"ab","dimensions":3})).await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
    assert_eq!(v["param"], "dimensions");
}

#[tokio::test]
async fn the_wrong_model_is_a_400_that_names_what_is_wrong() {
    let (s, v) = post(serde_json::json!({"model":"nope","input":"x"})).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert!(v["error"].as_str().unwrap().contains("not loaded"), "{v}");

    let (s, v) = post(serde_json::json!({"model":"chat","input":"x"})).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert!(
        v["error"]
            .as_str()
            .unwrap()
            .contains("not an embedding model"),
        "{v}"
    );
}

#[tokio::test]
async fn bad_inputs_are_400s_never_silent_truncation() {
    for body in [
        serde_json::json!({"model":"a-embed","input":[1,2,3]}),
        serde_json::json!({"model":"a-embed","input":""}),
        serde_json::json!({"model":"a-embed","input":[]}),
        serde_json::json!({"model":"a-embed","input":"123456789"}),
        serde_json::json!({"model":"a-embed","input":["a","b","c","d"]}),
        serde_json::json!({"model":"a-embed","input":"x","encoding_format":"hex"}),
    ] {
        let (s, v) = post(body.clone()).await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{body} -> {v}");
    }
    // Control: the same shape within the limits is served.
    let (s, _) = post(serde_json::json!({"model":"a-embed","input":"12345678"})).await;
    assert_eq!(s, StatusCode::OK);
}

#[tokio::test]
async fn a_model_that_returns_nan_is_a_500_not_a_vector() {
    let (s, v) = post(serde_json::json!({"model":"a-embed","input":"nan"})).await;
    assert_eq!(s, StatusCode::INTERNAL_SERVER_ERROR, "{v}");
    assert_eq!(v["param"], "model");
}

#[test]
fn an_embedding_entry_never_answers_the_chat_lookup() {
    let st = state();
    assert!(
        st.model().embedding.is_none(),
        "the chat lookup returned the embedding entry"
    );
    // Control: the registry does hold the embedding entry, under the name that sorts first.
    assert_eq!(st.models.names(), vec!["a-embed", "chat"]);
    assert!(st.models.get("a-embed").unwrap().embedding.is_some());
}
