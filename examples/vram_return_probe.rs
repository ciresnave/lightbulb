// SPDX-License-Identifier: MIT OR Apache-2.0
//! Does dropping a model's runner return its GPU memory? (gates F2 P2)
//!
//! `docs/superpowers/specs/2026-10-30-f2-multi-model-design.md` plans to unload
//! a model by dropping its `ModelRunner` senders, which ends the runner thread.
//! That only helps if the device memory actually comes back. This probe
//! measures it, n trials, with a positive control.
//!
//! Per trial, in one process that holds a CUDA device for its whole life (so
//! the CUDA context itself is not part of what is measured):
//!
//! * RELEASE: start a runner, serve one tiny job (so the model is loaded and
//!   its KV pool allocated), read device memory used, drop BOTH senders, poll
//!   until used falls back to the pre-load level or `LB_WAIT_S` passes.
//! * CONTROL: same, but keep a clone of the job sender alive (a deliberately
//!   retained handle: the runner thread cannot see a disconnect). Used memory
//!   must stay up. If it does not, the RELEASE result says nothing, because
//!   the probe could not tell "returned" from "never allocated".
//!
//! Device memory comes from `nvidia-smi --query-gpu=memory.used`, which is
//! device-wide: run it under the machine GPU lock so no other process moves
//! the number (`fuel/scripts/gpu-run.ps1 -Project lightbulb -- <cmd>`), under
//! `pwsh`, not Windows PowerShell 5.1 (see `prefix_sharing_bench.rs`).
//!
//! ```text
//! LB_GGUF=C:/Models/gguf-corpus/quants/SmolLM2-135M-Instruct-f16.gguf \
//!   cargo run --release --example vram_return_probe --features fuel-cuda
//! ```
//!
//! Knobs: `LB_GGUF` (required), `LB_TRIALS` (3), `LB_CTX` (2048),
//! `LB_WAIT_S` (30), `LB_TOL_MIB` (64, the slack on "back to baseline"),
//! `LB_HOLD_DEVICE` (1; 0 = hold no device, like the server).

use lightbulb::engine::model_runner::{EngineOptions, InferenceJob, ModelRunner, ResponseMode};
use lightbulb::engine::{
    MemoryAwareScheduler, Priority, memory_aware_scheduler::MemoryAwareConfig,
};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::oneshot;

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
        .expect("nvidia-smi printed nothing")
        .trim()
        .parse()
        .expect("nvidia-smi memory.used was not an integer")
}

/// Start a runner and serve one tiny job so the model and its KV pool are
/// really allocated before we look at memory.
fn start_loaded(
    path: &str,
    ctx: usize,
) -> (
    lightbulb::engine::model_runner::InferenceRequestSender,
    lightbulb::engine::model_runner::ControlSender,
) {
    let scheduler = Arc::new(Mutex::new(MemoryAwareScheduler::new(
        MemoryAwareConfig::default(),
    )));
    let stats = Arc::new(Mutex::new(lightbulb::cache::PrefixCacheStats::default()));
    let (tx, ctrl) = ModelRunner::start_with_options(
        path,
        4,
        ctx,
        Some("f32".into()),
        16,
        stats,
        scheduler,
        EngineOptions::default(),
    )
    .expect("starting runner");
    let (otx, orx) = oneshot::channel();
    tx.send(InferenceJob {
        id: "probe".into(),
        prompt: "The quick brown fox".into(),
        max_new_tokens: 2,
        temperature: 0.0,
        add_special_tokens: true,
        priority: Priority::Normal,
        response_mode: ResponseMode::Complete(otx),
    })
    .unwrap();
    orx.blocking_recv()
        .unwrap()
        .expect("probe job failed: model did not load");
    (tx, ctrl)
}

/// Poll until used <= target or `wait` elapses; returns (final used, seconds).
fn wait_back_to(target: i64, wait: Duration) -> (i64, f64) {
    let t0 = Instant::now();
    loop {
        let u = used_mib();
        if u <= target || t0.elapsed() >= wait {
            return (u, t0.elapsed().as_secs_f64());
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

/// One RELEASE + CONTROL trial; returns (released, control_held).
fn trial(
    i: usize,
    path: &str,
    ctx: usize,
    wait: Duration,
    baseline: i64,
    tol: i64,
) -> (bool, bool) {
    // RELEASE
    let (tx, ctrl) = start_loaded(path, ctx);
    let loaded = used_mib();
    drop(tx);
    drop(ctrl);
    let (after, secs) = wait_back_to(baseline + tol, wait);
    let returned = after <= baseline + tol;
    // CONTROL
    let (tx2, ctrl2) = start_loaded(path, ctx);
    let loaded2 = used_mib();
    let retained = tx2.clone();
    drop(tx2);
    drop(ctrl2);
    std::thread::sleep(wait.min(Duration::from_secs(10)));
    let held = used_mib();
    let control_held = held >= baseline + (loaded2 - baseline) * 9 / 10;
    drop(retained);
    let _ = wait_back_to(baseline + tol, wait); // clean up for the next trial
    println!(
        "trial {i}: RELEASE loaded {loaded} MiB (+{} over baseline) -> after drop {after} MiB \
         in {secs:.1}s: {}",
        loaded - baseline,
        if returned { "RETURNED" } else { "NOT RETURNED" }
    );
    println!(
        "         CONTROL loaded {loaded2} MiB -> with a retained handle {held} MiB: {}",
        if control_held {
            "held (control valid)"
        } else {
            "NOT held (control INVALID)"
        }
    );
    (returned, control_held)
}

fn main() {
    let path = std::env::var("LB_GGUF").expect("set LB_GGUF to a .gguf path");
    let trials = env_usize("LB_TRIALS", 3);
    let ctx = env_usize("LB_CTX", 2048);
    let wait = Duration::from_secs(env_usize("LB_WAIT_S", 30) as u64);
    let tol = env_usize("LB_TOL_MIB", 64) as i64;

    // LB_HOLD_DEVICE=1 (default): hold a device for the whole run, so the CUDA
    // context is in the baseline. =0: hold none, like the server (each runner
    // thread selects its own device and nothing else keeps one), and the
    // baseline is whatever the GPU shows before any CUDA work. The two modes
    // can differ on whether device-owned memory comes back at all, so a result
    // must name which one it was.
    let hold = env_usize("LB_HOLD_DEVICE", 1) != 0;
    let _ctx_holder = hold.then(lightbulb::model_fuel::device::select);
    std::thread::sleep(Duration::from_secs(2));
    let baseline = used_mib();
    println!("model {path}");
    println!("ctx {ctx}, trials {trials}, tolerance {tol} MiB, hold_device {hold}");
    let what = if hold {
        "context only"
    } else {
        "no CUDA work yet"
    };
    println!("baseline ({what}): {baseline} MiB used\n");

    let (mut release_ok, mut control_valid) = (0, 0);
    for i in 1..=trials {
        let (returned, control_held) = trial(i, &path, ctx, wait, baseline, tol);
        release_ok += returned as usize;
        control_valid += control_held as usize;
    }
    println!(
        "\nsummary: RELEASE returned {release_ok}/{trials}; CONTROL valid {control_valid}/{trials}"
    );
    if control_valid < trials {
        println!("the positive control failed in some trials: those RELEASE results prove nothing");
    }
}
