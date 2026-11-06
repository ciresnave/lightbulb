// SPDX-License-Identifier: MIT OR Apache-2.0
//! BENCH, not a test: the measurements that gate the F2 P2 code
//! (`docs/superpowers/specs/2026-10-30-f2-multi-model-design.md`, "P2 decision").
//!
//! Two real runners are loaded through the registry; model A is unloaded with
//! `unload_model` (drop senders, join the thread, `device_pool` release) while model B
//! stays resident. Per trial it prints device-wide used memory (`nvidia-smi`) after each
//! step, B's request latency before the unload and for the first three requests after it
//! (the re-reserve cost the note could not state), and a positive control: the same unload
//! with a request's sender clone retained must NOT free A.
//!
//! The release handle is created by `device::try_select()` (as the server does), before
//! the baseline is read, so its context (~113 MiB) is part of the baseline.
//!
//! ```text
//! LB_MODEL_A=<gguf> LB_MODEL_B=<gguf> LIGHTBULB_DEVICE=cuda:0 \
//!   cargo run --release --features fuel-cuda --example unload_two_models_bench
//! ```
//! Knobs: `LB_MODEL_A`, `LB_MODEL_B` (required), `LB_TRIALS` (3), `LB_CTX` (512),
//! `LB_WAIT_S` (30). GPU runs go through `fuel/scripts/gpu-run.ps1 -Project lightbulb`.

use lightbulb::api::registry::{ModelEntry, ModelRegistry, unload_model};
use lightbulb::engine::model_runner::{
    EngineOptions, InferenceJob, InferenceRequestSender, ModelRunner, ResponseMode,
};
use lightbulb::engine::{
    MemoryAwareScheduler, Priority, memory_aware_scheduler::MemoryAwareConfig,
};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Device-wide used memory in MiB, from nvidia-smi.
fn used_mib() -> i64 {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .expect("running nvidia-smi");
    assert!(out.status.success(), "nvidia-smi failed: {out:?}");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()
        .and_then(|l| l.trim().parse().ok())
        .expect("nvidia-smi memory.used was not an integer")
}

/// Wait until used memory stops changing (two reads within 8 MiB), up to `wait`.
fn settled_mib(wait: Duration) -> i64 {
    let t0 = Instant::now();
    let mut prev = used_mib();
    loop {
        std::thread::sleep(Duration::from_millis(700));
        let now = used_mib();
        if (now - prev).abs() <= 8 || t0.elapsed() >= wait {
            return now;
        }
        prev = now;
    }
}

/// Send one 8-token job and wait for the text; returns (text, latency).
fn serve(tx: &InferenceRequestSender, prompt: &str) -> (String, Duration) {
    let (otx, orx) = tokio::sync::oneshot::channel();
    let t = Instant::now();
    tx.send(InferenceJob {
        id: "bench".into(),
        prompt: prompt.into(),
        max_new_tokens: 8,
        temperature: 0.0,
        add_special_tokens: true,
        priority: Priority::Normal,
        response_mode: ResponseMode::Complete(otx),
    })
    .expect("queueing a job");
    let result = orx
        .blocking_recv()
        .expect("runner dropped the job")
        .expect("the job failed (model did not load?)");
    (result.text, t.elapsed())
}

/// Start a runner for `path`, register it as `name`, and serve one job so the weights
/// and KV pool really exist before memory is read.
fn load(registry: &ModelRegistry, name: &str, path: &str, ctx: usize) -> i64 {
    let scheduler = Arc::new(Mutex::new(MemoryAwareScheduler::new(
        MemoryAwareConfig::default(),
    )));
    let stats = Arc::new(Mutex::new(lightbulb::cache::PrefixCacheStats::default()));
    let (tx, ctrl, join) = ModelRunner::start_with_handle(
        path,
        4,
        ctx,
        Some("f32".into()),
        16,
        stats,
        scheduler,
        EngineOptions::default(),
    )
    .expect("starting a runner");
    registry.set(
        name,
        ModelEntry {
            inference_tx: Some(tx.clone()),
            cache_control: Some(ctrl),
            join: Some(Arc::new(Mutex::new(Some(join)))),
            ..ModelEntry::default()
        },
    );
    let t = Instant::now();
    let _ = serve(&tx, "The quick brown fox");
    println!(
        "  {name} loaded and answered in {:.1}s",
        t.elapsed().as_secs_f64()
    );
    settled_mib(Duration::from_secs(20))
}

fn sender(registry: &ModelRegistry, name: &str) -> InferenceRequestSender {
    registry
        .get(name)
        .and_then(|e| e.inference_tx)
        .unwrap_or_else(|| panic!("model {name} is not registered"))
}

fn ms(d: Duration) -> String {
    format!("{:.0} ms", d.as_secs_f64() * 1000.0)
}

/// B's request latency for `n` requests, printed under `label`.
fn b_latency(registry: &ModelRegistry, label: &str, n: usize) {
    let b = sender(registry, "B");
    let lats: Vec<String> = (0..n)
        .map(|i| {
            let (text, d) = serve(&b, &format!("Name a colour, number {i}."));
            assert!(!text.is_empty() || d.as_millis() > 0, "B answered nothing");
            ms(d)
        })
        .collect();
    println!("  B latency {label}: {}", lats.join(", "));
}

/// The positive control: unload A while a request's sender clone is retained. It must
/// report `joined: false` and must not free A's memory.
fn control_trial(registry: &ModelRegistry, a_path: &str, ctx: usize, wait: Duration) {
    let with_a = load(registry, "A", a_path, ctx);
    let retained = sender(registry, "A"); // a running request's clone
    let out = unload_model(registry, "A", Duration::from_secs(3)).expect("unload");
    let after = settled_mib(wait);
    println!(
        "CONTROL (sender retained): joined {} released {:?}; used {with_a} -> {after} MiB \
         (must stay up: {})",
        out.joined,
        out.released.is_some(),
        if !out.joined && after >= with_a - 64 {
            "held, control valid"
        } else {
            "NOT held, control INVALID"
        }
    );
    drop(retained);
    std::thread::sleep(Duration::from_secs(2));
    let _ = lightbulb::model_fuel::device_pool::release_unused_device_memory();
    let _ = settled_mib(wait);
}

fn trial(
    i: usize,
    registry: &ModelRegistry,
    a: &str,
    b: &str,
    ctx: usize,
    wait: Duration,
    baseline: i64,
) {
    println!("--- trial {i}");
    let used_a = load(registry, "A", a, ctx);
    let used_ab = load(registry, "B", b, ctx);
    println!("  used: baseline {baseline}, +A {used_a}, +A+B {used_ab} MiB");
    b_latency(registry, "before the unload", 3);
    let t = Instant::now();
    let out = unload_model(registry, "A", wait).expect("unload A");
    let unload_secs = t.elapsed().as_secs_f64();
    let after_unload = settled_mib(wait);
    println!(
        "  unload A: joined {} released {:?} in {unload_secs:.1}s; used {used_ab} -> {after_unload} MiB (freed {} MiB)",
        out.joined,
        out.released.map(|r| (r.before, r.after)),
        used_ab - after_unload
    );
    b_latency(registry, "after the unload (first three)", 3);
    let out_b = unload_model(registry, "B", wait).expect("unload B");
    let end = settled_mib(wait);
    println!(
        "  unload B: joined {}; used {after_unload} -> {end} MiB (baseline {baseline}, left over {} MiB)",
        out_b.joined,
        end - baseline
    );
}

fn main() {
    let a = std::env::var("LB_MODEL_A").expect("set LB_MODEL_A to a .gguf path");
    let b = std::env::var("LB_MODEL_B").expect("set LB_MODEL_B to a .gguf path");
    let trials = env_usize("LB_TRIALS", 3);
    let ctx = env_usize("LB_CTX", 512);
    let wait = Duration::from_secs(env_usize("LB_WAIT_S", 30) as u64);

    // As the server does: selecting the device creates the release handle, before any load.
    let device = lightbulb::model_fuel::device::try_select().expect("device selection");
    println!(
        "device {device:?}; chosen {:?}",
        lightbulb::model_fuel::device::chosen()
    );
    let handle = lightbulb::model_fuel::device_pool::stats().expect("pool stats");
    println!("release handle present: {}", handle.is_some());
    std::thread::sleep(Duration::from_secs(2));
    let baseline = used_mib();
    println!(
        "A {a}\nB {b}\nctx {ctx}, trials {trials}; baseline (context + handle) {baseline} MiB\n"
    );

    let registry = ModelRegistry::default();
    for i in 1..=trials {
        trial(i, &registry, &a, &b, ctx, wait, baseline);
    }
    println!("--- control");
    control_trial(&registry, &a, ctx, wait);
}
