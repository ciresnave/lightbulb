//! Measurement for PR #4's hot-path objection: is the inactive-row guard's
//! `gather`/`where_cond` material next to a decode step, or noise?
//!
//! Two reviewers independently flagged that `ParallelKvCache::append` now does
//! a full gather plus a `keep` allocation and two `where_cond`s on any step
//! where at least one row is inactive. That establishes FREQUENCY — partial
//! batches are the common case in a batching server. It does not establish
//! MAGNITUDE, and nobody in the thread had numbers.
//!
//! `#[ignore]`d: this is a probe, not an assertion. Run with
//! `cargo test --release --test kv_append_perf_probe -- --ignored --nocapture`.
use candlelight::core::{DType, Device, Tensor};
use lightbulb::engine::ParallelCacheBuilder;

/// TinyLlama-1.1B-Chat geometry, which is what this project actually serves.
const LAYERS: usize = 22;
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 64;
const CONTEXT: usize = 2048;
const BATCH: usize = 8;
const ITERS: usize = 200;
const ROUNDS: usize = 9;

/// One configuration under test, with its own pre-allocated cache.
struct Arm {
    label: &'static str,
    active: Vec<bool>,
    builder: ParallelCacheBuilder,
    cache: lightbulb::engine::ParallelKvCache,
    samples: Vec<std::time::Duration>,
}

impl Arm {
    fn new(label: &'static str, active: Vec<bool>) -> Self {
        let dev = Device::Cpu;
        let mut builder = ParallelCacheBuilder::new(BATCH, CONTEXT, DType::F32, &dev).unwrap();
        let cache = builder.make_cache(KV_HEADS, HEAD_DIM).unwrap();
        Self { label, active, builder, cache, samples: Vec::new() }
    }
}

#[test]
#[ignore = "measurement probe, not an assertion — see PR #4's hot-path objection"]
fn append_cost_all_active_vs_partial() {
    let dev = Device::Cpu;
    let n = BATCH * KV_HEADS * HEAD_DIM;
    let k = Tensor::from_vec(vec![1.0f32; n], (BATCH, KV_HEADS, 1, HEAD_DIM), &dev).unwrap();
    let v = Tensor::from_vec(vec![2.0f32; n], (BATCH, KV_HEADS, 1, HEAD_DIM), &dev).unwrap();

    let mut one_off = vec![true; BATCH];
    one_off[BATCH - 1] = false;
    let mut half = vec![true; BATCH];
    for a in half.iter_mut().take(BATCH / 2) { *a = false; }

    let mut arms = vec![
        Arm::new("all active (fast path)", vec![true; BATCH]),
        Arm::new("1 of 8 inactive (guard path)", one_off),
        Arm::new("4 of 8 inactive (guard path)", half),
    ];

    // Warm EVERY arm before timing ANY of them. The cache is
    // batch*kv_heads*context*head_dim*4 bytes per tensor (~16 MB here, x2 for
    // k and v), so a cold arm pays page faults the others do not — measuring
    // the allocator instead of the code. An earlier version of this probe
    // timed each arm to completion in sequence and reported the GUARD path as
    // FASTER than the fast path, which is impossible: the guard path does
    // strictly more work. That was arm-order artefact, not a finding.
    for arm in arms.iter_mut() {
        for _ in 0..20 {
            let iam = arm.builder.indices_and_mask(1, &arm.active).unwrap();
            arm.cache.append(&k, &v, &iam).unwrap();
        }
    }

    // Interleave rounds so any drift hits all arms equally.
    for _ in 0..ROUNDS {
        for arm in arms.iter_mut() {
            let start = std::time::Instant::now();
            for _ in 0..ITERS {
                let iam = arm.builder.indices_and_mask(1, &arm.active).unwrap();
                arm.cache.append(&k, &v, &iam).unwrap();
            }
            arm.samples.push(start.elapsed() / ITERS as u32);
        }
    }

    println!("
TinyLlama geometry: {LAYERS} layers, {KV_HEADS} kv-heads, head_dim {HEAD_DIM}, context {CONTEXT}, batch {BATCH}");
    println!("{ROUNDS} interleaved rounds x {ITERS} iterations, all arms pre-warmed, CPU, --release
");
    for arm in arms.iter_mut() {
        arm.samples.sort();
        let med = arm.samples[arm.samples.len() / 2];
        let min = arm.samples[0];
        let max = arm.samples[arm.samples.len() - 1];
        println!(
            "{:<32} median {:>8.1?}  (min {:>8.1?}  max {:>8.1?})  -> {:>8.1?} / decode step",
            arm.label, med, min, max, med * LAYERS as u32
        );
    }
    println!();
}
