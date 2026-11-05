// SPDX-License-Identifier: MIT OR Apache-2.0
//! BENCH, not a test: which device a build serves on, and what one prefill and
//! each decode step cost, through the same loader and the same Fuel forward call
//! (`forward_with_kv_context_persistent`) the server uses, WITHOUT the paged
//! scheduler. Compare with the `bench_served_driver_steps` bench in
//! `model_fuel/scheduler_driver_paged.rs`, which drives the scheduler the server
//! uses. Measured 2026-11-05 (Qwen3-0.6B Q8_0, RTX 4070 Laptop): this path is
//! ~0.10 s/step on cuda:0 and ~0.25 s/step on cpu after a one-off ~5 s first step.
//!
//! ```text
//! LB_GGUF=<qwen3 .gguf> LIGHTBULB_DEVICE=cuda:0 //!   cargo run --release --features fuel-cuda --example decode_step_bench
//! ```
//! Knobs: `LB_GGUF` (required, a qwen3 GGUF), `LIGHTBULB_DEVICE`, `LB_PROMPT_REPS`
//! (6), `LB_STEPS` (6). GPU runs go through `fuel/scripts/gpu-run.ps1`.
use fuel::inference_context::{InferenceContext, KvCache};
use std::time::Instant;

fn env(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn main() {
    let path = std::env::var("LB_GGUF").expect("set LB_GGUF");
    let sel = std::env::var("LIGHTBULB_DEVICE").unwrap_or_else(|_| "(unset=auto)".into());
    println!("LIGHTBULB_DEVICE={sel}");
    println!(
        "devices fuel lists: {:?}",
        lightbulb::model_fuel::device::enumerate()
    );
    let chosen = lightbulb::model_fuel::device::try_select();
    println!(
        "try_select => {:?}",
        chosen.as_ref().map(|d| format!("{d:?}"))
    );

    let t = Instant::now();
    let loaded = lightbulb::model_fuel::loader_gguf_qwen3::load_quantized_qwen3_gguf(
        std::path::Path::new(&path),
    )
    .expect("load");
    println!(
        "load: {:?}  loaded.device = {:?}",
        t.elapsed(),
        loaded.device
    );

    let text = "The quick brown fox jumps over the lazy dog. ".repeat(env("LB_PROMPT_REPS", 6));
    let ids: Vec<u32> = loaded
        .tokenizer
        .encode(text, false)
        .expect("encode")
        .get_ids()
        .to_vec();
    let steps = env("LB_STEPS", 6);
    let c = &loaded.config;
    let device = loaded.device.clone();
    let mut cache = KvCache::with_capacity(
        c.num_hidden_layers,
        c.num_key_value_heads,
        c.head_dim,
        ids.len() + steps + 2,
        fuel::DType::F32,
        &device,
    )
    .expect("kv cache");
    let mut ctx = InferenceContext::new(device);
    let mut session = None;

    let t = Instant::now();
    let mut logits = loaded
        .model
        .forward_with_kv_context_persistent(&ids, &mut cache, &mut ctx, &mut session)
        .expect("prefill");
    println!("prefill {} tokens: {:?}", ids.len(), t.elapsed());
    for i in 0..steps {
        let next = lightbulb::model_fuel::generate::argmax(&logits);
        let t = Instant::now();
        logits = loaded
            .model
            .forward_with_kv_context_persistent(&[next], &mut cache, &mut ctx, &mut session)
            .expect("decode");
        println!("decode step {i}: {:?}", t.elapsed());
    }
}
