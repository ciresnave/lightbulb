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

/// Text of roughly `n` tokens (counted by the model's own tokenizer, so the number printed is the real one).
fn text_of_about(emb: &NomicEmbedder, n: usize) -> String {
    use lightbulb::api::openai::embeddings::Embedder;
    let unit = "the quick brown fox jumps over the lazy dog and then rests by the river. ";
    let per = emb.count_tokens(unit).unwrap();
    format!("search_document: {}", unit.repeat(n / per))
}

/// Closes the gap the accuracy test leaves: a cosine of 1.0 would also come out of a silent CPU fallback. Here
/// the SAME forward is timed on CPU and on CUDA, short and at about the model's 2048-token training context.
/// Run with `nvidia-smi --query-gpu=utilization.gpu,memory.used -lms 200` sampling beside it. Prints timings;
/// asserts the two devices agree (cosine) and that the GPU run is not the CPU run (timing ratio printed, and the
/// sampled memory/utilisation is the device evidence).
#[test]
fn nomic_cuda_vs_cpu_timing_short_and_long() {
    let dir = model_dir();
    let emb =
        NomicEmbedder::load(&dir).unwrap_or_else(|e| panic!("loading {}: {e}", dir.display()));
    let cuda = fuel_cuda_backend::CudaDevice::new(0).expect("no CUDA device");
    let dim = emb.spec().dim;
    for (label, text) in [
        (
            "short",
            "search_query: what is the capital of France?".to_string(),
        ),
        ("long ", text_of_about(&emb, 2048)),
    ] {
        use lightbulb::api::openai::embeddings::Embedder;
        let toks = emb.count_tokens(&text).unwrap();
        let _warm = emb.hidden_states_cuda(&text, &cuda).unwrap();
        let t = std::time::Instant::now();
        let mut g = None;
        for _ in 0..3 {
            g = Some(emb.hidden_states_cuda(&text, &cuda).unwrap());
        }
        let gpu_ms = t.elapsed().as_secs_f64() * 1000.0 / 3.0;
        let t = std::time::Instant::now();
        let c = emb.hidden_states(&text).unwrap();
        let cpu_ms = t.elapsed().as_secs_f64() * 1000.0;
        let (g, c) = (g.unwrap(), c);
        let cos = cosine(
            &pool(&g.0, g.1, dim, Pooling::Mean).unwrap(),
            &pool(&c.0, c.1, dim, Pooling::Mean).unwrap(),
        );
        eprintln!(
            "TIMING {label} {toks:5} tokens: cuda {gpu_ms:9.1} ms (mean of 3, warm)  cpu {cpu_ms:9.1} ms (1 run)  cuda/cpu cosine {cos:.6}"
        );
        assert!(cos >= 0.999, "{label}: CUDA and CPU disagree, cosine {cos}");
    }
}
