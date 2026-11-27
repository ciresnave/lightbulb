// SPDX-License-Identifier: MIT OR Apache-2.0
//! F3 GPU acceptance: the nomic-embed-text-v1.5 encoder realized on CUDA through Fuel's `realize_f32_cuda`,
//! held to the same ONNX reference and the same cosine >= 0.999 bar as the CPU gate (`tests/nomic_reference.rs`).
//!
//! Run under the machine GPU lock, compiling OUTSIDE it:
//! `cargo test --features fuel-cuda --test nomic_gpu --no-run`, then
//! `pwsh C:/Projects/tools/gpu-run/gpu-run.ps1 -Project lightbulb -- <the test binary> --nocapture`.
//! Like the other GPU harnesses this FAILS, never skips, on a missing device or checkpoint.
#![cfg(feature = "fuel-cuda")]

use std::path::PathBuf;

use lightbulb::api::openai::embeddings::{Pooling, finish};
use lightbulb::api::openai::nomic_embedder::{NomicEmbedder, pool};

#[derive(serde::Deserialize)]
struct Row {
    text: String,
    embedding: Vec<f32>,
}

#[derive(serde::Deserialize)]
struct Fixture {
    rows: Vec<Row>,
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let n = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    dot / (n(a) * n(b))
}

fn model_dir() -> PathBuf {
    std::env::var_os("NOMIC_EMBED_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\lightbulb-models\nomic-embed-text-v1.5"))
}

#[test]
fn nomic_on_cuda_matches_onnxruntime_at_cosine_0_999() {
    let dir = model_dir();
    let emb =
        NomicEmbedder::load(&dir).unwrap_or_else(|e| panic!("loading {}: {e}", dir.display()));
    let cuda = fuel_cuda_backend::CudaDevice::new(0)
        .expect("no CUDA device: this gate measures the GPU, so absence is a failure");
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/nomic-embed-text-v1.5-reference.json"
    );
    let fx: Fixture = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(fx.rows.len(), 5);
    let dim = emb.spec().dim;
    for r in &fx.rows {
        let (hidden, seq) = emb.hidden_states_cuda(&r.text, &cuda).expect("GPU forward");
        let pooled = pool(&hidden, seq, dim, Pooling::Mean).unwrap();
        let c = cosine(&finish(pooled, emb.spec(), None), &r.embedding);
        // Control: the same hidden states pooled as CLS must fall below the bar, so a GPU vector that merely
        // resembles a unit vector cannot pass.
        let cls = cosine(
            &finish(
                pool(&hidden, seq, dim, Pooling::Cls).unwrap(),
                emb.spec(),
                None,
            ),
            &r.embedding,
        );
        eprintln!(
            "GPU cosine {c:.6} (cls control {cls:.4})  {seq} tokens  {:?}",
            r.text
        );
        assert!(c >= 0.999, "GPU cosine {c} < 0.999 for {:?}", r.text);
        assert!(
            cls < 0.99,
            "CLS control reached {cls}: the gate cannot discriminate"
        );
    }
}
