// SPDX-License-Identifier: MIT OR Apache-2.0
//! PROBE for issue #298 (not a test): with ONE model resident, how much does the CUDA
//! default pool reserve, and how much of it does `trim_to(0)` give back?
//!
//! Loads `LB_MODEL`, serves `LB_REQUESTS` (3) short jobs, and prints the pool's
//! used/reserved bytes plus device-wide `nvidia-smi` memory after the load, after the
//! requests, and after `release_unused_device_memory`. Run it for several models: if the
//! reserve that survives the trim is large for one model and not another, the cause is
//! that model's allocation pattern, not the unload.
//!
//! `LB_MODEL=<gguf> LIGHTBULB_DEVICE=cuda:0 cargo run --release --features fuel-cuda
//!   --example pool_residency_probe` (GPU runs go through `gpu-run.ps1`).

use lightbulb::engine::model_runner::{EngineOptions, InferenceJob, ModelRunner, ResponseMode};
use lightbulb::engine::{
    MemoryAwareScheduler, Priority, memory_aware_scheduler::MemoryAwareConfig,
};
use lightbulb::model_fuel::device_pool;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::Duration;

fn smi_mib() -> i64 {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .expect("running nvidia-smi");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()
        .and_then(|l| l.trim().parse().ok())
        .expect("nvidia-smi memory.used was not an integer")
}

fn mib(b: u64) -> String {
    format!("{} MiB", b / (1024 * 1024))
}

fn report(label: &str) {
    std::thread::sleep(Duration::from_secs(2));
    let pool = device_pool::stats().ok().flatten();
    match pool {
        Some(p) => println!(
            "{label:<28} pool used {:>7}  reserved {:>7}  | device used {} MiB",
            mib(p.used),
            mib(p.reserved),
            smi_mib()
        ),
        None => println!(
            "{label:<28} (no pool handle) | device used {} MiB",
            smi_mib()
        ),
    }
}

fn serve(tx: &lightbulb::engine::model_runner::InferenceRequestSender, i: usize) {
    let (otx, orx) = tokio::sync::oneshot::channel();
    tx.send(InferenceJob {
        id: format!("probe-{i}"),
        prompt: format!("Name a colour, number {i}."),
        max_new_tokens: 8,
        temperature: 0.0,
        add_special_tokens: true,
        priority: Priority::Normal,
        response_mode: ResponseMode::Complete(otx),
    })
    .expect("queueing a job");
    orx.blocking_recv()
        .expect("runner dropped the job")
        .expect("the job failed");
}

fn main() {
    let model = std::env::var("LB_MODEL").expect("set LB_MODEL to a .gguf path");
    let requests: usize = std::env::var("LB_REQUESTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let ctx: usize = std::env::var("LB_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    lightbulb::model_fuel::device::try_select().expect("device selection");
    println!("model {model}\nctx {ctx}, requests {requests}");
    report("baseline (handle only)");

    let scheduler = Arc::new(Mutex::new(MemoryAwareScheduler::new(
        MemoryAwareConfig::default(),
    )));
    let stats = Arc::new(Mutex::new(lightbulb::cache::PrefixCacheStats::default()));
    let (tx, _ctrl, join) = ModelRunner::start_with_handle(
        &model,
        4,
        ctx,
        Some("f32".into()),
        16,
        stats,
        scheduler,
        EngineOptions::default(),
    )
    .expect("starting a runner");
    serve(&tx, 0);
    report("after load + first request");
    for i in 1..requests {
        serve(&tx, i);
    }
    report("after the requests");

    let released = device_pool::release_unused_device_memory().expect("trim");
    println!("trim_to(0) while resident: {released:?}");
    report("after trim, model resident");
    serve(&tx, 99);
    report("after one more request");

    drop(tx);
    join.join().expect("runner thread");
    report("runner gone, before trim");
    device_pool::release_unused_device_memory().expect("trim");
    report("runner gone, after trim");
}
