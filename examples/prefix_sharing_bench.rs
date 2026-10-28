// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP research-track item 1, slice 6: prefix sharing OFF vs ON.
//!
//! Serves one GGUF through the real `ModelRunner` twice -- once per arm,
//! `EngineOptions::prefix_sharing` false then true -- and sends the same
//! workload to each: a long-lived DONOR with a long prompt, then sequential
//! SHARERS whose prompts begin with the donor's prefix and differ in a short
//! tail, plus an UNRELATED control of equal length that shares nothing.
//!
//! Sharing needs a LIVE donor, so the donor decodes `LB_DONOR_NEW` tokens
//! while the sharers run; each sharer row records whether the donor was still
//! live when it was submitted. A sharer submitted after the donor finished is
//! cold in both arms and says nothing about sharing.
//!
//! Reported per arm: TTFT (submit -> first streamed token) and tokens/s
//! (post-first-token decode rate) per request kind, median over `LB_REPS`.
//! Greedy (temperature 0) completions of every sharer and the control are
//! compared between arms; sharing must be bit-identical to cold.
//!
//! Device: CPU by default, the CUDA device when built with `--features
//! fuel-cuda` (`model_fuel::device::select`).
//!
//! ```text
//! LB_GGUF=C:/Models/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
//!   cargo run --release --example prefix_sharing_bench [--features fuel-cuda]
//! ```
//!
//! Knobs (env): `LB_GGUF` (required), `LB_PREFIX_PARAS` (default 30),
//! `LB_SHARERS` (4), `LB_SHARER_NEW` (16), `LB_DONOR_NEW` (400),
//! `LB_REPS` (3), `LB_CTX` (2048), `LB_GATE=1` (+ `LB_GATE_MAX_S`),
//! `LB_ON_FIRST=1` (run the sharing arm first so a capped run still yields it).

use lightbulb::engine::model_runner::{
    EngineOptions, InferenceJob, ModelRunner, ResponseMode, StreamItem,
};
use lightbulb::engine::{
    MemoryAwareScheduler, Priority, memory_aware_scheduler::MemoryAwareConfig,
};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};

type Rx = mpsc::UnboundedReceiver<anyhow::Result<StreamItem>>;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// `paras` distinct paragraphs; `seed` shifts every number so two seeds share
/// no token prefix (the unrelated control).
fn paragraphs(paras: usize, seed: usize) -> String {
    (0..paras)
        .map(|i| {
            format!(
                "Section {} of report {}: the survey crew logged {} samples at station {} \
                 and recorded a drift of {} units before the {} inspection.\n",
                i,
                seed,
                i * 37 + seed * 11,
                (i * 5 + seed) % 97,
                i * 13 + seed * 3,
                ["morning", "noon", "evening", "midnight"][(i + seed) % 4],
            )
        })
        .collect()
}

/// (cpu %, free RAM GiB) sampled over ~1 s.
fn load() -> (f32, f64) {
    let mut sys = sysinfo::System::new();
    sys.refresh_cpu_usage();
    std::thread::sleep(Duration::from_millis(1100));
    sys.refresh_cpu_usage();
    sys.refresh_memory();
    (
        sys.global_cpu_usage(),
        sys.available_memory() as f64 / (1u64 << 30) as f64,
    )
}

fn report_load(tag: &str) {
    let (cpu, free) = load();
    println!("  [load {tag}] cpu {cpu:.0}%  free RAM {free:.1} GiB");
}

/// `LB_GATE=1`: wait (up to `LB_GATE_MAX_S`, default 4 h) until CPU < 25%
/// for 60 consecutive seconds and >= 24 GiB RAM is free, then run.
fn gate() {
    if std::env::var("LB_GATE").as_deref() != Ok("1") {
        return;
    }
    let max = Duration::from_secs(env_usize("LB_GATE_MAX_S", 4 * 3600) as u64);
    let t0 = Instant::now();
    let mut quiet = 0;
    while quiet < 60 {
        let (cpu, free) = load();
        quiet = if cpu < 25.0 && free >= 24.0 {
            quiet + 1
        } else {
            0
        };
        assert!(
            t0.elapsed() < max,
            "gate: box never went quiet (last cpu {cpu:.0}%, free {free:.1} GiB)"
        );
    }
    println!("gate passed after {:.0} s", t0.elapsed().as_secs_f64());
}

struct Submitted {
    rx: Rx,
    at: Instant,
}

fn submit(
    tx: &lightbulb::engine::model_runner::InferenceRequestSender,
    id: &str,
    prompt: String,
    max_new: usize,
) -> Submitted {
    let (stx, rx) = mpsc::unbounded_channel();
    let at = Instant::now();
    tx.send(InferenceJob {
        id: id.to_string(),
        prompt,
        max_new_tokens: max_new,
        temperature: 0.0,
        add_special_tokens: true,
        priority: Priority::Normal,
        response_mode: ResponseMode::Streaming(stx),
    })
    .expect("runner thread gone");
    Submitted { rx, at }
}

struct Outcome {
    ttft: Duration,
    /// Streamed tokens after the first, per second of wall time after it.
    decode_tps: f64,
    text: String,
}

/// Drain to `Done`, timing the first token.
fn drain(mut s: Submitted) -> Outcome {
    let mut first: Option<Instant> = None;
    let mut n = 0usize;
    let mut text = String::new();
    loop {
        match s.rx.blocking_recv().expect("channel closed early") {
            Ok(StreamItem::Token(t)) => {
                n += 1;
                first.get_or_insert_with(Instant::now);
                text.push_str(&t);
            }
            Ok(StreamItem::Done { .. }) => break,
            Err(e) => panic!("request failed: {e:#}"),
        }
    }
    let first = first.expect("no token streamed");
    let tail = first.elapsed().as_secs_f64();
    Outcome {
        ttft: first - s.at,
        decode_tps: if n > 1 && tail > 0.0 {
            (n - 1) as f64 / tail
        } else {
            f64::NAN
        },
        text,
    }
}

/// Block until the first token (donor prefill finished), return the receiver.
fn await_first_token(mut s: Submitted) -> (Submitted, String) {
    match s.rx.blocking_recv().expect("donor channel closed") {
        Ok(StreamItem::Token(t)) => (s, t),
        other => panic!("donor first item was {other:?}"),
    }
}

#[derive(Default)]
struct ArmResult {
    /// (label, donor_live, outcome)
    rows: Vec<(String, bool, Outcome)>,
    donor_text: String,
    hits: usize,
    prompt_tokens: usize,
}

type Sender = lightbulb::engine::model_runner::InferenceRequestSender;

fn start_runner(
    path: &str,
    sharing: bool,
    cfg: &Cfg,
) -> (Sender, Arc<Mutex<lightbulb::cache::PrefixCacheStats>>) {
    let scheduler = Arc::new(Mutex::new(MemoryAwareScheduler::new(
        MemoryAwareConfig::default(),
    )));
    let stats = Arc::new(Mutex::new(lightbulb::cache::PrefixCacheStats::default()));
    let (tx, _ctrl) = ModelRunner::start_with_options(
        path,
        4,
        cfg.ctx,
        Some("f32".into()),
        16,
        stats.clone(),
        scheduler,
        EngineOptions {
            prefix_sharing: sharing,
        },
    )
    .expect("starting runner");
    (tx, stats)
}

/// Token count of the prefix, via a warm-up (kernel/JIT/page-in) that is not
/// part of any timed row. Complete mode, two new tokens (1 is refused:
/// max_new must be > 0).
fn warm_up(tx: &Sender, prefix: &str) -> usize {
    let (otx, orx) = oneshot::channel();
    tx.send(InferenceJob {
        id: "warm".into(),
        prompt: format!("{prefix}warm-up tail"),
        max_new_tokens: 2,
        temperature: 0.0,
        add_special_tokens: true,
        priority: Priority::Normal,
        response_mode: ResponseMode::Complete(otx),
    })
    .unwrap();
    orx.blocking_recv()
        .unwrap()
        .expect("warm-up failed")
        .prompt_tokens
}

/// One repetition: an unrelated cold control, then a donor and its sharers.
fn run_rep(tx: &Sender, cfg: &Cfg, rep: usize, prefix: &str, unrelated: &str, res: &mut ArmResult) {
    // Unrelated control FIRST, with no donor live: a cold request of the
    // same length. Its TTFT is the cold baseline within this arm.
    let c = drain(submit(
        tx,
        "ctl",
        format!("{unrelated}control {rep}"),
        cfg.sharer_new,
    ));
    eprintln!(
        "  .. rep{rep} control done ttft {:.0} ms",
        c.ttft.as_secs_f64() * 1e3
    );
    res.rows.push((format!("rep{rep} control(cold)"), false, c));

    let donor = submit(
        tx,
        "donor",
        format!("{prefix}donor tail {rep}"),
        cfg.donor_new,
    );
    let (mut donor, first) = await_first_token(donor);
    res.donor_text = first;

    for k in 0..cfg.sharers {
        // `try_recv` Err(Empty) = donor still decoding; a `Done` item (or
        // closed channel) = it finished and this sharer is cold.
        let mut live = true;
        while let Ok(item) = donor.rx.try_recv() {
            match item {
                Ok(StreamItem::Token(t)) => res.donor_text.push_str(&t),
                _ => live = false,
            }
        }
        let o = drain(submit(
            tx,
            "sharer",
            format!("{prefix}sharer {k} of rep {rep}, answer briefly"),
            cfg.sharer_new,
        ));
        eprintln!(
            "  .. rep{rep} sharer{k} live={live} ttft {:.0} ms",
            o.ttft.as_secs_f64() * 1e3
        );
        res.rows.push((format!("rep{rep} sharer{k}"), live, o));
    }
    // Let the donor finish so the next rep starts with an idle engine.
    while let Some(Ok(StreamItem::Token(t))) = donor.rx.blocking_recv() {
        res.donor_text.push_str(&t);
    }
}

fn run_arm(path: &str, sharing: bool, cfg: &Cfg) -> ArmResult {
    let (tx, stats) = start_runner(path, sharing, cfg);
    let prefix = paragraphs(cfg.paras, 0);
    let unrelated = paragraphs(cfg.paras, 7);
    let mut res = ArmResult {
        prompt_tokens: warm_up(&tx, &prefix),
        ..Default::default()
    };
    for rep in 0..cfg.reps {
        run_rep(&tx, cfg, rep, &prefix, &unrelated, &mut res);
    }
    res.hits = stats.lock().unwrap().hits;
    res
}

struct Cfg {
    paras: usize,
    sharers: usize,
    sharer_new: usize,
    donor_new: usize,
    reps: usize,
    ctx: usize,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.retain(|x| x.is_finite());
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn summarise(label: &str, r: &ArmResult, pick: impl Fn(&str, bool) -> bool) -> (f64, f64, usize) {
    let sel: Vec<_> = r
        .rows
        .iter()
        .filter(|(l, live, _)| pick(l, *live))
        .collect();
    let ttft = median(
        sel.iter()
            .map(|(_, _, o)| o.ttft.as_secs_f64() * 1e3)
            .collect(),
    );
    let tps = median(sel.iter().map(|(_, _, o)| o.decode_tps).collect());
    println!(
        "  {label:<28} n={:<2} TTFT median {ttft:>9.1} ms   decode median {tps:>7.2} tok/s",
        sel.len()
    );
    (ttft, tps, sel.len())
}

impl Cfg {
    fn from_env() -> Self {
        Cfg {
            paras: env_usize("LB_PREFIX_PARAS", 30),
            sharers: env_usize("LB_SHARERS", 4),
            sharer_new: env_usize("LB_SHARER_NEW", 16),
            donor_new: env_usize("LB_DONOR_NEW", 400),
            reps: env_usize("LB_REPS", 3),
            ctx: env_usize("LB_CTX", 2048),
        }
    }
}

fn print_arm(r: &ArmResult) {
    println!(
        "  prefix prompt ~{} tokens; cache_stats.hits = {}",
        r.prompt_tokens, r.hits
    );
    for (l, live, o) in &r.rows {
        println!(
            "    {l:<22} donor_live={live:<5} ttft {:>9.1} ms  decode {:>7.2} tok/s",
            o.ttft.as_secs_f64() * 1e3,
            o.decode_tps
        );
    }
    summarise("control (cold, unrelated)", r, |l, _| l.contains("control"));
    summarise("sharers, donor live", r, |l, live| {
        l.contains("sharer") && live
    });
    summarise("sharers, donor gone (cold)", r, |l, live| {
        l.contains("sharer") && !live
    });
}

fn compare_arms(off: &ArmResult, on: &ArmResult) {
    println!(
        "
=== bit-identity, OFF vs ON (greedy) ==="
    );
    let mut all_same = off.rows.len() == on.rows.len();
    for ((l, _, a), (_, _, b)) in off.rows.iter().zip(on.rows.iter()) {
        let same = a.text == b.text;
        all_same &= same;
        if !same {
            println!(
                "  DIFFERS {l}:
    off: {:?}
    on : {:?}",
                a.text, b.text
            );
        }
    }
    println!(
        "  rows compared: {}; donor stream equal: {}",
        off.rows.len(),
        off.donor_text == on.donor_text
    );
    println!("  ALL SHARER/CONTROL COMPLETIONS BIT-IDENTICAL: {all_same}");
    let live = |l: &str, live: bool| l.contains("sharer") && live;
    let (o, _, _) = summarise("OFF sharers, donor live", off, live);
    let (o2, _, _) = summarise("ON  sharers, donor live", on, live);
    println!("  TTFT ON/OFF for live-donor sharers: {:.3}", o2 / o);
    let (c0, _, _) = summarise("OFF control", off, |l, _| l.contains("control"));
    let (c1, _, _) = summarise("ON  control", on, |l, _| l.contains("control"));
    println!("  TTFT ON/OFF for no-share control: {:.3}", c1 / c0);
}

fn main() {
    let path = std::env::var("LB_GGUF").expect("set LB_GGUF to a .gguf file");
    let cfg = Cfg::from_env();
    println!("model: {path}");
    println!(
        "device build: {}",
        if cfg!(feature = "fuel-cuda") {
            "fuel-cuda (CUDA if present)"
        } else {
            "CPU"
        }
    );

    gate();
    report_load("start");
    let mut arms = Vec::new();
    // `LB_ON_FIRST=1` runs the sharing arm first so a capped run still yields
    // the arm under test; `arms` is put back in OFF, ON order below.
    let on_first = std::env::var("LB_ON_FIRST").as_deref() == Ok("1");
    for sharing in if on_first {
        [true, false]
    } else {
        [false, true]
    } {
        println!(
            "
=== arm: prefix_sharing = {sharing} ==="
        );
        report_load("arm start");
        let r = run_arm(&path, sharing, &cfg);
        report_load("arm end");
        print_arm(&r);
        arms.push(r);
    }
    if on_first {
        arms.reverse();
    }
    compare_arms(&arms[0], &arms[1]);
    report_load("end");
}
