// SPDX-License-Identifier: MIT OR Apache-2.0
//! F3 known-limit measurement (ROADMAP F3, HANDOFF): what does unloading an EMBEDDING entry return, and to whom?
//!
//! Prints process memory (working set and private bytes, from `sysinfo`) and GPU memory (`nvidia-smi`) at each
//! phase of load -> serve -> unload through the real manager path. It ASSERTS only the structural facts (the
//! unload reports `joined: false` and `released: None` because an embedding entry has no runner thread, and the
//! entry is gone afterwards); the memory figures are a measurement to read, not a pass/fail bar.
//! Checkpoint: `$NOMIC_EMBED_DIR` set means it is REQUIRED (missing files fail); unset means the default
//! `C:\lightbulb-models` location, and absent there is a printed skip (a failure under `LIGHTBULB_REQUIRE_MODEL=1`),
//! so hosted CI skips it and a provisioned runner runs it.

use std::path::PathBuf;

use axum::body::Body;
use axum::http::Request;
use lightbulb::api::fuel_loader::{FuelLoader, StartParams};
use lightbulb::api::manager::{FreeBytes, ModelManager};
use lightbulb::api::{ApiConfig, AppState, ModelRegistry};
use lightbulb::engine::{MemoryAwareConfig, MemoryAwareScheduler};
use tower::ServiceExt;

fn mib(b: u64) -> f64 {
    b as f64 / 1048576.0
}

fn proc_mem() -> (f64, f64) {
    let pid = sysinfo::get_current_pid().unwrap();
    let mut sys = sysinfo::System::new();
    sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
    let p = sys.process(pid).expect("own process");
    (mib(p.memory()), mib(p.virtual_memory()))
}

fn gpu_mib() -> String {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map_or("n/a".into(), |s| s.trim().to_string())
}

fn phase(label: &str) {
    // Let the allocator and the driver settle before reading.
    std::thread::sleep(std::time::Duration::from_secs(2));
    let (ws, private) = proc_mem();
    eprintln!(
        "MEM {label:<28} working_set {ws:8.1} MiB  private {private:8.1} MiB  gpu_used {} MiB",
        gpu_mib()
    );
}

/// The checkpoint directory, or `None` for a skip (never a silent pass: `skip_unless_required` prints it).
fn checkpoint() -> Option<PathBuf> {
    let set = std::env::var_os("NOMIC_EMBED_DIR").map(PathBuf::from);
    let dir = set
        .clone()
        .unwrap_or_else(|| PathBuf::from(r"C:\lightbulb-models\nomic-embed-text-v1.5"));
    if dir.join("model.safetensors").is_file() {
        return Some(dir);
    }
    assert!(
        set.is_none(),
        "NOMIC_EMBED_DIR={} has no model.safetensors",
        dir.display()
    );
    lightbulb::test_notice::skip_unless_required(
        "LIGHTBULB_REQUIRE_MODEL",
        "the nomic-embed-text-v1.5 checkpoint (model.safetensors); see PROVENANCE.txt",
    );
    None
}

fn managed(parent: &std::path::Path) -> (AppState, std::sync::Arc<ModelManager>, ModelRegistry) {
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
    let manager = std::sync::Arc::new(ModelManager::new(
        models.clone(),
        std::sync::Arc::new(FuelLoader::new(parent, StartParams::from_state(&state))),
        free,
        std::time::Duration::from_secs(30),
    ));
    state.manager = Some(manager.clone());
    (state, manager, models)
}

async fn post(state: &AppState, name: &str, text: &str) -> u16 {
    let body = serde_json::json!({"model": name, "input": text});
    let req = Request::builder()
        .method("POST")
        .uri("/v1/embeddings")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    let app = lightbulb::api::openai::routes().with_state(state.clone());
    app.oneshot(req).await.unwrap().status().as_u16()
}

/// Unload through the manager and assert the structural facts (no runner to join, entry gone).
async fn unload_and_check(
    manager: &std::sync::Arc<ModelManager>,
    models: &ModelRegistry,
    name: &str,
) {
    let (m, n) = (manager.clone(), name.to_string());
    let out = tokio::task::spawn_blocking(move || m.unload(&n))
        .await
        .unwrap()
        .expect("unload");
    eprintln!(
        "UNLOAD result: joined={} released={:?}",
        out.joined, out.released
    );
    assert!(
        !out.joined && out.released.is_none(),
        "an embedding entry has no runner to join: {out:?}"
    );
    assert_eq!(
        out.runner.describe(),
        "n/a (no runner thread: embedding entry)"
    );
    assert!(
        !models.names().contains(&name.to_string()),
        "the entry must be gone"
    );
}

#[tokio::test]
async fn embedding_entry_unload_memory_return() {
    let Some(dir) = checkpoint() else { return };
    let name = dir.file_name().unwrap().to_string_lossy().into_owned();
    let (state, manager, models) = managed(dir.parent().unwrap());

    phase("baseline (nothing loaded)");
    assert_eq!(
        post(
            &state,
            &name,
            "search_query: what is the capital of France?"
        )
        .await,
        200
    );
    phase("loaded + 1 request");
    for _ in 0..5 {
        assert_eq!(post(&state, &name, "search_query: again").await, 200);
    }
    phase("after 5 more requests");
    unload_and_check(&manager, &models, &name).await;
    phase("after unload");
    // Control: a second load puts the memory back, so the unload's drop is not an artefact of never loading.
    assert_eq!(post(&state, &name, "search_query: reload").await, 200);
    phase("reloaded + 1 request");
}
