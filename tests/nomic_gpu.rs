// SPDX-License-Identifier: MIT OR Apache-2.0
//! F3 GPU acceptance: the nomic-embed-text-v1.5 encoder realized on CUDA through Fuel's `realize_f32_cuda`,
//! held to the same ONNX reference and the same cosine >= 0.999 bar as the CPU gate (`tests/nomic_reference.rs`).
//!
//! Run under the machine GPU lock, compiling OUTSIDE it:
//! `cargo test --features fuel-cuda --test nomic_gpu --no-run`, then
//! `pwsh C:/Projects/tools/gpu-run/gpu-run.ps1 -Project lightbulb -- <the test binary> --nocapture`.
//! The exe needs `cudnn64_9.dll` (`C:\Program Files\NVIDIA\CUDNN\v9.23\bin\13.3\x64`) and the CUDA toolkit
//! `bin\x64` on PATH, else it exits 0xC0000135, which gpu-run reports as 53 with no test output.
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

/// F3 batching fact: the encoder takes ONE sequence per call (no padded batch in Fuel), so N sequences are N
/// calls. Measures tokens/s for 1, 8 and 32 short sequences on CUDA and CPU; if a batch were really amortised
/// the per-sequence time would fall with N, and it does not (printed, read it).
#[test]
fn nomic_one_vs_many_sequences_tokens_per_second() {
    use lightbulb::api::openai::embeddings::Embedder;
    let dir = model_dir();
    let emb =
        NomicEmbedder::load(&dir).unwrap_or_else(|e| panic!("loading {}: {e}", dir.display()));
    let cuda = fuel_cuda_backend::CudaDevice::new(0).expect("no CUDA device");
    let texts: Vec<String> = (0..32)
        .map(|i| format!("search_document: sentence number {i} about the weather in Paris today."))
        .collect();
    let toks: usize = texts.iter().map(|t| emb.count_tokens(t).unwrap()).sum();
    let _warm = emb.hidden_states_cuda(&texts[0], &cuda).unwrap();
    for n in [1usize, 8, 32] {
        let tok = toks * n / 32;
        let t = std::time::Instant::now();
        for s in &texts[..n] {
            emb.hidden_states_cuda(s, &cuda).unwrap();
        }
        let g = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        for s in &texts[..n] {
            emb.hidden_states(s).unwrap();
        }
        let c = t.elapsed().as_secs_f64();
        eprintln!(
            "BATCH n={n:2} (~{tok:4} tokens)  cuda {g:7.2} s = {:6.1} tok/s ({:6.0} ms/seq)  cpu {c:7.2} s = {:6.1} tok/s ({:6.0} ms/seq)",
            tok as f64 / g,
            g * 1000.0 / n as f64,
            tok as f64 / c,
            c * 1000.0 / n as f64
        );
    }
}

// ---- F3 device option: `LIGHTBULB_EMBEDDING_DEVICE` through the real server path --------------------------------

mod served {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    use axum::body::Body;
    use axum::http::Request;
    use lightbulb::api::embedding_models::EmbeddingDevice;
    use lightbulb::api::fuel_loader::{FuelLoader, StartParams};
    use lightbulb::api::manager::{FreeBytes, ModelManager};
    use lightbulb::api::{ApiConfig, AppState, ModelRegistry};
    use lightbulb::engine::{MemoryAwareConfig, MemoryAwareScheduler};
    use tower::ServiceExt;

    pub fn state(parent: &std::path::Path, device: EmbeddingDevice) -> AppState {
        let models = ModelRegistry::default();
        let mut state = AppState {
            scheduler: Arc::new(std::sync::Mutex::new(MemoryAwareScheduler::new(
                MemoryAwareConfig::default(),
            ))),
            config: ApiConfig {
                models_dir: Some(parent.to_string_lossy().into_owned()),
                embedding_device: device,
                ..ApiConfig::default()
            },
            pg_backend: None,
            manager: None,
            models: models.clone(),
            eos_monitor: Default::default(),
            security_sinks: Vec::new(),
            cache_stats: Arc::new(std::sync::Mutex::new(
                lightbulb::cache::PrefixCacheStats::default(),
            )),
            live_system: lightbulb::api::new_shared_system_metrics(),
            queue_depth: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        };
        let free: FreeBytes = Arc::new(|| u64::MAX);
        state.manager = Some(Arc::new(ModelManager::new(
            models,
            Arc::new(FuelLoader::new(parent, StartParams::from_state(&state))),
            free,
            std::time::Duration::from_secs(30),
        )));
        state
    }

    pub async fn call(
        state: &AppState,
        method: &str,
        uri: &str,
        body: serde_json::Value,
    ) -> (u16, serde_json::Value) {
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
        let status = resp.status().as_u16();
        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 24)
            .await
            .unwrap();
        (status, serde_json::from_slice(&bytes).unwrap_or_default())
    }

    fn gpu_used_mib() -> u64 {
        std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().lines().next().and_then(|l| l.trim().parse().ok()))
            .expect("nvidia-smi must report memory.used: this gate measures the GPU")
    }

    /// GPU memory used while `f` runs: (baseline MiB, peak MiB), sampled every 100 ms on another thread.
    pub async fn watched<F: std::future::Future>(f: F) -> (F::Output, u64, u64) {
        let base = gpu_used_mib();
        let peak = Arc::new(AtomicU64::new(base));
        let stop = Arc::new(AtomicBool::new(false));
        let (p, s) = (peak.clone(), stop.clone());
        let t = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                p.fetch_max(gpu_used_mib(), Ordering::Relaxed);
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        });
        let out = f.await;
        stop.store(true, Ordering::Relaxed);
        t.join().unwrap();
        (out, base, peak.load(Ordering::Relaxed))
    }
}

fn floats(v: &serde_json::Value) -> Vec<f32> {
    v["data"][0]["embedding"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

async fn listed_device(state: &lightbulb::api::AppState, name: &str) -> String {
    let (_, v) = served::call(state, "GET", "/v1/models", serde_json::json!({})).await;
    v["data"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["id"] == name)
        .unwrap()["lightbulb_device"]
        .as_str()
        .unwrap()
        .to_string()
}

/// The request really runs on the card, and the same request configured for the CPU does not touch it. Device
/// evidence is `nvidia-smi` memory sampled across the request (the CUDA run must add the model's weights on
/// the device, the CPU control must add nothing like that), the encoder's own `device_label` via `/v1/models`,
/// and the cosine gate on the vector that came back over HTTP. Either arm alone could be satisfied by the wrong device.
#[tokio::test]
async fn served_on_cuda_uses_the_card_and_matches_the_reference_and_cpu_does_not() {
    use lightbulb::api::embedding_models::EmbeddingDevice;
    let dir = model_dir();
    let name = dir.file_name().unwrap().to_string_lossy().into_owned();
    let parent = dir.parent().unwrap().to_path_buf();
    let fx: Fixture = serde_json::from_slice(
        &std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/nomic-embed-text-v1.5-reference.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let long = {
        let emb = NomicEmbedder::load(&dir).unwrap();
        text_of_about(&emb, 300)
    };

    let mut peaks = Vec::new();
    for device in [EmbeddingDevice::Cuda(0), EmbeddingDevice::Cpu] {
        let state = served::state(&parent, device);
        let ((s_short, v_short, s_long), base, peak) = served::watched(async {
            let (s1, v1) = served::call(
                &state,
                "POST",
                "/v1/embeddings",
                serde_json::json!({"model": name, "input": fx.rows[1].text}),
            )
            .await;
            let (s2, _) = served::call(
                &state,
                "POST",
                "/v1/embeddings",
                serde_json::json!({"model": name, "input": long}),
            )
            .await;
            (s1, v1, s2)
        })
        .await;
        assert_eq!((s_short, s_long), (200, 200), "{v_short}");
        let c = cosine(&floats(&v_short), &fx.rows[1].embedding);
        let label = listed_device(&state, &name).await;
        eprintln!(
            "DEVICE {device:?}: gpu memory base {base} MiB peak {peak} MiB (+{}), listed {label}, cosine {c:.6}",
            peak - base
        );
        assert_eq!(
            label,
            device.label(),
            "/v1/models must report the device the encoder really holds"
        );
        assert!(c >= 0.999, "{device:?}: cosine {c}");
        peaks.push((device, peak - base));
    }
    let (cuda_delta, cpu_delta) = (peaks[0].1, peaks[1].1);
    assert!(
        cuda_delta >= 200,
        "CUDA run added only {cuda_delta} MiB of GPU memory: it did not run on the card"
    );
    assert!(
        cpu_delta < 100,
        "the CPU control added {cpu_delta} MiB of GPU memory: the control touched the card"
    );
}

/// A device that is not there is an error the caller sees, never a CPU answer.
#[tokio::test]
async fn an_unavailable_cuda_device_is_a_typed_error_not_a_cpu_run() {
    use lightbulb::api::embedding_models::EmbeddingDevice;
    let dir = model_dir();
    let name = dir.file_name().unwrap().to_string_lossy().into_owned();
    let state = served::state(dir.parent().unwrap(), EmbeddingDevice::Cuda(99));
    let (s, v) = served::call(
        &state,
        "POST",
        "/v1/embeddings",
        serde_json::json!({"model": name, "input": "x"}),
    )
    .await;
    eprintln!("unavailable device -> {s} {v}");
    assert_ne!(
        s, 200,
        "a missing card must not be answered from the CPU: {v}"
    );
    assert!(
        v.to_string().contains("99"),
        "the error must name the device: {v}"
    );
    // Startup-style check, the same refusal.
    let e = EmbeddingDevice::Cuda(99).check_available().unwrap_err();
    assert!(e.contains("99"), "{e}");
}
