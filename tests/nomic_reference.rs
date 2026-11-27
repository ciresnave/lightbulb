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

use std::path::PathBuf;

use lightbulb::api::openai::embeddings::{
    Embedder, EmbeddingRequest, Pooling, build_response, finish, validate,
};
use lightbulb::api::openai::nomic_embedder::{NomicEmbedder, pool};

const DIR_NAME: &str = "nomic-embed-text-v1.5";

fn model_dir() -> Option<PathBuf> {
    let candidates = [
        std::env::var_os("NOMIC_EMBED_DIR").map(PathBuf::from),
        std::env::var_os("LIGHTBULB_MODELS_DIR").map(|d| PathBuf::from(d).join(DIR_NAME)),
        std::env::var_os("MODELS_DIR").map(|d| PathBuf::from(d).join(DIR_NAME)),
        Some(PathBuf::from(r"C:\lightbulb-models").join(DIR_NAME)),
    ];
    candidates
        .into_iter()
        .flatten()
        .find(|d| d.join("model.safetensors").is_file() && d.join("tokenizer.json").is_file())
}

fn load() -> Option<NomicEmbedder> {
    let Some(dir) = model_dir() else {
        lightbulb::test_notice::skip_unless_required(
            "LIGHTBULB_REQUIRE_MODEL",
            "the nomic-embed-text-v1.5 checkpoint (model.safetensors + tokenizer.json); see PROVENANCE.txt",
        );
        return None;
    };
    Some(NomicEmbedder::load(&dir).expect("loading the nomic-embed-text-v1.5 checkpoint"))
}

#[derive(serde::Deserialize)]
struct Row {
    text: String,
    tokens: usize,
    embedding: Vec<f32>,
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
