// SPDX-License-Identifier: MIT OR Apache-2.0
//! F3 acceptance gate (docs/F3-EMBEDDINGS-DESIGN.md test plan 1-4): `NomicEmbedder` over Fuel's `lazy_nomic_bert`
//! against reference vectors produced by an INDEPENDENT implementation, onnxruntime running the model's own fp32
//! ONNX export (`tools/gen-nomic-reference.py`, fixture `tests/fixtures/nomic-embed-text-v1.5-reference.json`).
//!
//! Model: nomic-ai/nomic-embed-text-v1.5 (Apache-2.0, Nomic AI,
//! <https://huggingface.co/nomic-ai/nomic-embed-text-v1.5>), kept OUTSIDE every repo; `PROVENANCE.txt` beside it
//! records the revision and sha256. Located by `$NOMIC_EMBED_DIR`, else `<$LIGHTBULB_MODELS_DIR | $MODELS_DIR>/
//! nomic-embed-text-v1.5`, else `C:\lightbulb-models\nomic-embed-text-v1.5`. Absent: a loud skip, or a FAILURE
//! when `LIGHTBULB_REQUIRE_MODEL=1` (the repo's `test_notice` convention), never a silent pass.
//!
//! What this does NOT test: the tokenizer. The reference ids come from the same `tokenizer.json` through the
//! `tokenizers` library, so tokenizer agreement shares its method and is only checked by token COUNT here.

use std::path::{Path, PathBuf};

use lightbulb::api::openai::embeddings::{
    Embedder, EmbeddingRequest, Pooling, build_response, finish, validate,
};
use lightbulb::api::openai::nomic_embedder::{NomicEmbedder, pool};

const DIR_NAME: &str = "nomic-embed-text-v1.5";

/// Where the checkpoint is, decided so that a skip can never be mistaken for a pass:
/// - `NOMIC_EMBED_DIR` SET: that directory is required. Missing files FAIL the run (never a skip), because
///   whoever set the variable declared the model provisioned.
/// - UNSET: the models chain and `C:\lightbulb-models`; nothing found is a printed skip (a failure under
///   `LIGHTBULB_REQUIRE_MODEL=1`).
#[derive(Debug, PartialEq, Eq)]
enum Located {
    Found(PathBuf),
    SetButIncomplete(PathBuf),
    Absent,
}

fn locate(
    env: &dyn Fn(&str) -> Option<std::ffi::OsString>,
    complete: &dyn Fn(&Path) -> bool,
) -> Located {
    if let Some(dir) = env("NOMIC_EMBED_DIR") {
        let dir = PathBuf::from(dir);
        return if complete(&dir) {
            Located::Found(dir)
        } else {
            Located::SetButIncomplete(dir)
        };
    }
    let chain = [
        env("LIGHTBULB_MODELS_DIR").map(|d| PathBuf::from(d).join(DIR_NAME)),
        env("MODELS_DIR").map(|d| PathBuf::from(d).join(DIR_NAME)),
        Some(PathBuf::from(r"C:\lightbulb-models").join(DIR_NAME)),
    ];
    chain
        .into_iter()
        .flatten()
        .find(|d| complete(d))
        .map_or(Located::Absent, Located::Found)
}

fn is_complete(d: &Path) -> bool {
    d.join("model.safetensors").is_file() && d.join("tokenizer.json").is_file()
}

fn load() -> Option<NomicEmbedder> {
    match locate(&|k| std::env::var_os(k), &is_complete) {
        Located::Found(dir) => {
            Some(NomicEmbedder::load(&dir).expect("loading the nomic-embed-text-v1.5 checkpoint"))
        }
        Located::SetButIncomplete(dir) => panic!(
            "NOMIC_EMBED_DIR is set to {} but model.safetensors and tokenizer.json are not both there:              the gate refuses to skip when the model was declared provisioned",
            dir.display()
        ),
        Located::Absent => {
            lightbulb::test_notice::skip_unless_required(
                "LIGHTBULB_REQUIRE_MODEL",
                "the nomic-embed-text-v1.5 checkpoint (model.safetensors + tokenizer.json); see PROVENANCE.txt",
            );
            None
        }
    }
}

#[derive(serde::Deserialize)]
struct Row {
    text: String,
    tokens: usize,
    embedding: Vec<f32>,
    matryoshka: std::collections::BTreeMap<String, Vec<f32>>,
}

#[derive(serde::Deserialize)]
struct Fixture {
    rows: Vec<Row>,
}

fn fixture() -> Fixture {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/nomic-embed-text-v1.5-reference.json"
    );
    serde_json::from_slice(&std::fs::read(path).expect("reading the reference fixture"))
        .expect("parsing the reference fixture")
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

#[test]
fn every_reference_sentence_matches_onnxruntime_at_cosine_0_999() {
    let Some(emb) = load() else { return };
    let fx = fixture();
    assert_eq!(fx.rows.len(), 5, "the fixture holds five sentences");
    for r in &fx.rows {
        assert_eq!(
            emb.count_tokens(&r.text).unwrap(),
            r.tokens,
            "token count for {:?}",
            r.text
        );
        let e = emb.embed(&r.text).unwrap();
        assert_eq!(e.tokens, r.tokens);
        let ours = finish(e.vector, emb.spec(), None);
        let c = cosine(&ours, &r.embedding);
        eprintln!("cosine {c:.6}  {} tokens  {:?}", r.tokens, r.text);
        assert!(c >= 0.999, "cosine {c} < 0.999 for {:?}", r.text);
    }
}

/// The control the design note asks for: a wrong pooling must be visible. The same hidden states pooled as CLS
/// or last-token must fall BELOW the tolerance the mean passes, otherwise the gate cannot tell them apart.
#[test]
fn the_gate_tells_mean_pooling_from_cls_and_last_token() {
    let Some(emb) = load() else { return };
    let fx = fixture();
    let dim = emb.spec().dim;
    for r in &fx.rows {
        let (hidden, seq) = emb.hidden_states(&r.text).unwrap();
        let mean = pool(&hidden, seq, dim, Pooling::Mean).unwrap();
        let cls = pool(&hidden, seq, dim, Pooling::Cls).unwrap();
        let last = pool(&hidden, seq, dim, Pooling::LastToken).unwrap();
        let (cm, cc, cl) = (
            cosine(&mean, &r.embedding),
            cosine(&cls, &r.embedding),
            cosine(&last, &r.embedding),
        );
        eprintln!("mean {cm:.4}  cls {cc:.4}  last {cl:.4}  {:?}", r.text);
        assert!(cm >= 0.999, "mean {cm}");
        assert!(
            cc < 0.99,
            "CLS pooling reached {cc}: the gate cannot tell it from mean"
        );
        assert!(
            cl < 0.99,
            "last-token pooling reached {cl}: the gate cannot tell it from mean"
        );
    }
}

#[test]
fn a_sentence_alone_and_inside_an_array_give_the_same_vector_and_unit_norm() {
    let Some(emb) = load() else { return };
    let fx = fixture();
    let spec = emb.spec();
    let req = |input: serde_json::Value| -> EmbeddingRequest {
        serde_json::from_value(serde_json::json!({"model": "m", "input": input})).unwrap()
    };
    let alone = {
        let v = validate(&req(serde_json::json!(fx.rows[1].text)), spec).unwrap();
        build_response("m", &v, spec, &emb).unwrap()
    };
    let all: Vec<&str> = fx.rows.iter().map(|r| r.text.as_str()).collect();
    let together = {
        let v = validate(&req(serde_json::json!(all)), spec).unwrap();
        build_response("m", &v, spec, &emb).unwrap()
    };
    let get = |r: &lightbulb::api::openai::embeddings::EmbeddingResponse, i: usize| -> Vec<f32> {
        match &r.data[i].embedding {
            lightbulb::api::openai::embeddings::EmbeddingValue::Float(v) => v.clone(),
            other => panic!("expected floats, got {other:?}"),
        }
    };
    let a = get(&alone, 0);
    let b = get(&together, 1);
    assert!(cosine(&a, &b) >= 0.9999, "cosine {}", cosine(&a, &b));
    let norm: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "norm {norm}");
    // Usage is the sum of the real token counts, in order.
    let want: usize = fx.rows.iter().map(|r| r.tokens).sum();
    assert_eq!(together.usage.prompt_tokens, want);
}

/// Matryoshka widths against the card's recipe (layer_norm over the full vector, slice, L2), computed in the
/// scratch venv from the ONNX output (`tools/gen-nomic-reference.py`; model card README.md at the pinned
/// revision, lines 2676-2734).
///
/// WHAT THIS CAN AND CANNOT PROVE. Measured: on these five sentences the layer norm moves the vector by only
/// about 1e-5 in cosine (the pooled vectors have a near-zero per-component mean, and the L2 that follows removes
/// the scale), so the recipe WITHOUT the layer norm also passes the 0.999 bar. This test therefore shows the
/// output is right at every width, not that the layer-norm step is present; that step is pinned by the
/// hand-computed `a_reduced_width_layer_norms_the_full_vector_before_truncating` in `embeddings/tests.rs`, whose
/// input has a clearly nonzero mean. It only asserts ours is at least as close as the plain recipe.
#[test]
fn matryoshka_widths_match_the_cards_layer_norm_then_slice_recipe() {
    let Some(emb) = load() else { return };
    let fx = fixture();
    let spec = emb.spec().clone();
    let mut plain = spec.clone();
    plain.matryoshka_layer_norm = false;
    for r in &fx.rows {
        let pooled = emb.embed(&r.text).unwrap().vector;
        for d in [64usize, 128, 256, 512] {
            assert!(spec.matryoshka.contains(&d), "the spec offers {d}");
            let want = &r.matryoshka[&d.to_string()];
            let ours = finish(pooled.clone(), &spec, Some(d));
            assert_eq!(ours.len(), d);
            let c = cosine(&ours, want);
            let c_plain = cosine(&finish(pooled.clone(), &plain, Some(d)), want);
            eprintln!(
                "d={d:3} with layer_norm {c:.6}  without {c_plain:.4}  {:?}",
                r.text
            );
            assert!(c >= 0.999, "d={d}: cosine {c} < 0.999 for {:?}", r.text);
            assert!(
                c >= c_plain - 1e-6,
                "d={d}: the plain recipe ({c_plain}) is closer to the reference than ours ({c})"
            );
        }
    }
}

#[test]
fn a_set_but_incomplete_env_dir_is_never_a_skip() {
    let none = |_: &str| None;
    let set = |k: &str| (k == "NOMIC_EMBED_DIR").then(|| "X:/nowhere".into());
    // Control: unset and nothing found is the (loud) skip path.
    assert_eq!(locate(&none, &|_| false), Located::Absent);
    // Set but incomplete: a failure path, never Absent.
    assert_eq!(
        locate(&set, &|_| false),
        Located::SetButIncomplete(PathBuf::from("X:/nowhere"))
    );
    // Set and complete: found, and the models chain is not consulted.
    assert_eq!(
        locate(&set, &|_| true),
        Located::Found(PathBuf::from("X:/nowhere"))
    );
    // Unset: the chain finds a complete directory.
    let chain = |k: &str| (k == "MODELS_DIR").then(|| "M:/models".into());
    assert_eq!(
        locate(&chain, &|d| d.starts_with("M:/models")),
        Located::Found(PathBuf::from("M:/models").join(DIR_NAME))
    );
}

use axum::body::Body;
use axum::http::{Request, StatusCode};
use lightbulb::api::fuel_loader::{FuelLoader, StartParams};
use lightbulb::api::manager::{FreeBytes, ModelManager};
use lightbulb::api::{ApiConfig, AppState, ModelRegistry};
use lightbulb::engine::{MemoryAwareConfig, MemoryAwareScheduler};
use tower::ServiceExt;

/// An `AppState` whose manager loads from the models directory `parent`, as the server builds it at startup.
fn managed_state(parent: &Path) -> AppState {
    let models = ModelRegistry::default();
    let mut state = AppState {
        scheduler: std::sync::Arc::new(std::sync::Mutex::new(MemoryAwareScheduler::new(
            MemoryAwareConfig::default(),
        ))),
        config: ApiConfig {
            models_dir: Some(parent.to_string_lossy().into_owned()),
            ..ApiConfig::default()
        },
        pg_backend: None,
        manager: None,
        models: models.clone(),
        eos_monitor: Default::default(),
        security_sinks: Vec::new(),
        cache_stats: std::sync::Arc::new(std::sync::Mutex::new(
            lightbulb::cache::PrefixCacheStats::default(),
        )),
        live_system: lightbulb::api::new_shared_system_metrics(),
        queue_depth: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
    };
    let free: FreeBytes = std::sync::Arc::new(|| u64::MAX);
    state.manager = Some(std::sync::Arc::new(ModelManager::new(
        models,
        std::sync::Arc::new(FuelLoader::new(parent, StartParams::from_state(&state))),
        free,
        std::time::Duration::from_secs(30),
    )));
    state
}

async fn call(
    state: &AppState,
    method: &str,
    uri: &str,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    let resp = lightbulb::api::openai::routes()
        .with_state(state.clone())
        .oneshot(
            Request::builder()
                .method(method)
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 22)
        .await
        .unwrap();
    (status, serde_json::from_slice(&bytes).unwrap_or_default())
}

fn floats(v: &serde_json::Value) -> Vec<f32> {
    v["data"][0]["embedding"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

async fn listed(state: &AppState, name: &str) -> serde_json::Value {
    let (s, v) = call(state, "GET", "/v1/models", serde_json::json!({})).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    v["data"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["id"] == name)
        .unwrap_or_else(|| panic!("{name} is not listed: {v}"))
        .clone()
}

/// A Matryoshka width over HTTP against the card-recipe reference, and a width the model was not trained for as
/// a typed 400 by name (not a wrong vector).
async fn dimensions_over_http(state: &AppState, name: &str, row: &Row) {
    let ask = |d: u32| serde_json::json!({"model": name, "input": row.text, "dimensions": d});
    let (s, v) = call(state, "POST", "/v1/embeddings", ask(256)).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(floats(&v).len(), 256);
    assert!(cosine(&floats(&v), &row.matryoshka["256"]) >= 0.999);
    let (s, v) = call(state, "POST", "/v1/embeddings", ask(300)).await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
    assert_eq!(v["param"], "dimensions");
}

/// The whole server path with the real checkpoint: a models directory holding the embedding model, the
/// manager loading it on demand through `FuelLoader`, `POST /v1/embeddings` answering, and `GET /v1/models`
/// marking it. The vectors are held to the same reference as the encoder tests.
#[tokio::test]
async fn served_through_the_manager_loader_and_route_matches_the_reference() {
    let dir = match locate(&|k| std::env::var_os(k), &is_complete) {
        Located::Found(d) => d,
        Located::SetButIncomplete(d) => panic!("NOMIC_EMBED_DIR={} is incomplete", d.display()),
        Located::Absent => {
            lightbulb::test_notice::skip_unless_required(
                "LIGHTBULB_REQUIRE_MODEL",
                "the nomic-embed-text-v1.5 checkpoint; see PROVENANCE.txt",
            );
            return;
        }
    };
    let name = dir.file_name().unwrap().to_string_lossy().into_owned();
    let state = managed_state(dir.parent().expect("the model directory has a parent"));
    let row = &fixture().rows[1];

    // Listed before it is loaded: marked "embedding", not loaded.
    let before = listed(&state, &name).await;
    assert_eq!(before["lightbulb_kind"], "embedding");
    assert_eq!(before["lightbulb_loaded"], false);

    // The first request loads it on demand through the manager.
    let body = |extra: serde_json::Value| {
        let mut b = serde_json::json!({"model": name, "input": row.text});
        b.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        b
    };
    let (s, v) = call(
        &state,
        "POST",
        "/v1/embeddings",
        body(serde_json::json!({})),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["usage"]["prompt_tokens"], row.tokens);
    let c = cosine(&floats(&v), &row.embedding);
    eprintln!("served cosine {c:.6}");
    assert!(c >= 0.999, "cosine {c}");

    dimensions_over_http(&state, &name, row).await;

    assert_eq!(listed(&state, &name).await["lightbulb_loaded"], true);
}
