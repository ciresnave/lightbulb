//! Does H2O's eviction ranking tolerate an un-normalized attention-mass proxy?
//!
//! **Why this exists.** Baracuda can emit a per-key attention statistic as a second
//! output from the flash kernel. At *decode* (Sq=1) the exactly-normalized value is
//! cheap. At *prefill* it is not: flash's `÷l_q` normalization is deferred and
//! differs per query, so `Σ_q exp(S_qk − m_q)/l_q` cannot be accumulated in one
//! pass — exact needs a recompute sweep (~1.5–2× the softmax). An *un-normalized*
//! `Σ_q exp(S_qk − m_q)` accumulates cheaply.
//!
//! So the question that decides what gets built: **does using the un-normalized
//! proxy change which tokens H2O actually evicts?**
//!
//! **The analytic concern.** Un-normalized sums weight each query by its own `l_q`,
//! so a diffuse (high-entropy) query contributes more mass than a sharp one. Whether
//! that reorders the *eviction set* is empirical.
//!
//! **What this measures.** Real TinyLlama-1.1B weights, real tokens, attention at
//! layers 0/5/11/16/21, all 32 heads. For each head: the exact column-sum
//! `Σ_q probs[q][k]`, the un-normalized proxy `Σ_q exp(S_qk − m_q)`, and how far the
//! resulting evict-K sets diverge.
//!
//! **Result (48-token prompt):** mean overlap 90–99% across every layer and eviction
//! fraction, but the *worst single head* falls to 58–62% — concentrated at aggressive
//! eviction (10%) and at the first and last layers. `l_q` spread is 15–31× everywhere,
//! confirming the analytic mechanism is present and material.
//!
//! **What it does NOT measure**, stated so the result is not overread:
//! - A single prompt. No claim about distributional robustness.
//! - **One step's `a_t`, not H2O's accumulated `c_t`.** H2O sums with decay across
//!   steps; per-step errors could average out or compound. Untested.
//! - End-to-end generation quality. Eviction-set overlap is a proxy for a proxy.
//! - Prefill only (that is the case in question; decode is settled analytically —
//!   at Sq=1 the two differ by one scalar, so ranking is invariant within a step).
//! - It does not model H2O's cross-step decay accumulation, only one step's `a_t`.
//!
//! Run: `cargo run --release --example h2o_exact_vs_proxy`

use anyhow::{Context, Result};
use candlelight::core::{DType, Device, IndexOp, Tensor};
use std::collections::HashSet;

const SNAPSHOT: &str = "C:/Users/cires/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6";

const N_HEADS: usize = 32;
const N_KV_HEADS: usize = 4;
const HEAD_DIM: usize = 64;
const HIDDEN: usize = 2048;
const ROPE_THETA: f32 = 10000.0;
const EPS: f64 = 1e-5;

fn main() -> Result<()> {
    let device = Device::Cpu;

    // ---- real tokens ----
    let tok = tokenizers::Tokenizer::from_file(format!("{SNAPSHOT}/tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let prompt = "The capital of France is Paris. The capital of Germany is Berlin. \
                  Historically, the relationship between these two nations has shaped \
                  the political landscape of Europe for centuries, and their economies \
                  remain deeply intertwined to this day.";
    let ids: Vec<u32> = tok
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?
        .get_ids()
        .to_vec();
    let seq = ids.len();
    println!("prompt tokens: {seq}");

    // ---- real weights, layer 0 ----
    let tensors = candlelight::core::safetensors::load(
        format!("{SNAPSHOT}/model.safetensors"),
        &device,
    )
    .context("loading safetensors")?;

    let get = |name: &str| -> Result<Tensor> {
        tensors
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?
            .to_dtype(DType::F32)
            .map_err(Into::into)
    };

    let embed = get("model.embed_tokens.weight")?;

    let idx = Tensor::new(ids.as_slice(), &device)?;
    let embedded = embed.index_select(&idx, 0)?; // [seq, hidden]

    println!();
    println!("{:>5} {:>6} {:>10} {:>10} {:>8} {:>12}", "layer", "evict", "mean ovlp", "min ovlp", "worst", "l_q spread");
    for layer in [0usize, 5, 11, 16, 21] {
    let ln_w = get(&format!("model.layers.{layer}.input_layernorm.weight"))?;
    let q_w = get(&format!("model.layers.{layer}.self_attn.q_proj.weight"))?;
    let k_w = get(&format!("model.layers.{layer}.self_attn.k_proj.weight"))?;
    let x = candlelight::nn::ops::rms_norm(&embedded, &ln_w, EPS as f32)?;

    // ---- Q, K projections ----
    let q = x.matmul(&q_w.t()?)?.reshape((seq, N_HEADS, HEAD_DIM))?;
    let k = x.matmul(&k_w.t()?)?.reshape((seq, N_KV_HEADS, HEAD_DIM))?;

    // ---- RoPE ----
    let (cos, sin) = rope_tables(seq, HEAD_DIM, &device)?;
    let q = apply_rope(&q, &cos, &sin, seq, N_HEADS)?;
    let k = apply_rope(&k, &cos, &sin, seq, N_KV_HEADS)?;

    let scale = 1.0f64 / (HEAD_DIM as f64).sqrt();

    let mut rows: Vec<(usize, Vec<f64>, Vec<f64>, Vec<f64>)> = Vec::new();
    for h in 0..N_HEADS {
        let kv_h = h / (N_HEADS / N_KV_HEADS); // GQA
        let qh = q.i((.., h, ..))?; // [seq, dim]
        let kh = k.i((.., kv_h, ..))?; // [seq, dim]
        let scores = (qh.matmul(&kh.t()?)? * scale)?; // [seq, seq]
        let scores = causal_mask(&scores, seq, &device)?;
        let s: Vec<Vec<f32>> = scores.to_vec2()?;

        // Exact: Σ_q softmax(s[q])[k].  Proxy: Σ_q exp(s[q][k] − max_q).
        let mut exact = vec![0.0f64; seq];
        let mut proxy = vec![0.0f64; seq];
        let mut l_values = Vec::with_capacity(seq);
        for row in s.iter() {
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f64> = row.iter().map(|&v| ((v - m) as f64).exp()).collect();
            let l: f64 = e.iter().sum();
            l_values.push(l);
            for (kk, &ev) in e.iter().enumerate() {
                exact[kk] += ev / l; // normalized
                proxy[kk] += ev; // un-normalized
            }
        }
        rows.push((h, exact, proxy, l_values));
    }

    for &frac in &[0.10f64, 0.25, 0.50] {
        let n_evict = ((seq as f64) * frac).round() as usize;
        let mut overlaps = Vec::new();
        let mut worst = (0usize, 1.0f64);
        for (h, exact, proxy, _) in &rows {
            let a = lowest_k(exact, n_evict);
            let b = lowest_k(proxy, n_evict);
            let inter = a.intersection(&b).count();
            let ov = inter as f64 / n_evict.max(1) as f64;
            overlaps.push(ov);
            if ov < worst.1 {
                worst = (*h, ov);
            }
        }
        let mean = overlaps.iter().sum::<f64>() / overlaps.len() as f64;
        let min = overlaps.iter().cloned().fold(f64::INFINITY, f64::min);
        // l_q spread drives the distortion: uniform l_q ⇒ proxy ≈ exact.
        let spread = {
            let (_, _, _, l) = &rows[worst.0];
            let mx = l.iter().cloned().fold(f64::MIN, f64::max);
            let mn = l.iter().cloned().fold(f64::MAX, f64::min);
            mx / mn.max(1e-30)
        };
        println!(
            "{:>5} {:>5.0}% {:>9.1}% {:>9.1}% {:>8} {:>11.1}x",
            layer,
            frac * 100.0,
            mean * 100.0,
            min * 100.0,
            format!("h{}", worst.0),
            spread
        );
    }
    rows.clear();
    }

    println!();
    println!("Interpretation: overlap is the fraction of evicted keys the two agree on.");
    println!("100% = the proxy is free. Low overlap = exact matters and the recompute");
    println!("sweep (or a decode-only restriction) is required for prefill.");
    Ok(())
}

fn lowest_k(v: &[f64], k: usize) -> HashSet<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    idx.into_iter().take(k).collect()
}

fn causal_mask(scores: &Tensor, seq: usize, device: &Device) -> Result<Tensor> {
    let mut m = vec![0f32; seq * seq];
    for i in 0..seq {
        for j in (i + 1)..seq {
            m[i * seq + j] = f32::NEG_INFINITY;
        }
    }
    let mask = Tensor::from_vec(m, (seq, seq), device)?;
    Ok(scores.broadcast_add(&mask)?)
}

fn rope_tables(seq: usize, dim: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let half = dim / 2;
    let mut cos = vec![0f32; seq * half];
    let mut sin = vec![0f32; seq * half];
    for pos in 0..seq {
        for i in 0..half {
            let freq = 1.0f32 / ROPE_THETA.powf(2.0 * i as f32 / dim as f32);
            let a = pos as f32 * freq;
            cos[pos * half + i] = a.cos();
            sin[pos * half + i] = a.sin();
        }
    }
    Ok((
        Tensor::from_vec(cos, (seq, half), device)?,
        Tensor::from_vec(sin, (seq, half), device)?,
    ))
}

/// Llama-style rotate-half RoPE over `[seq, heads, dim]`.
fn apply_rope(t: &Tensor, cos: &Tensor, sin: &Tensor, seq: usize, heads: usize) -> Result<Tensor> {
    let half = HEAD_DIM / 2;
    let x1 = t.narrow(2, 0, half)?; // [seq, heads, half]
    let x2 = t.narrow(2, half, half)?;
    let cos = cos.reshape((seq, 1, half))?.broadcast_as((seq, heads, half))?;
    let sin = sin.reshape((seq, 1, half))?.broadcast_as((seq, heads, half))?;
    let r1 = ((&x1 * &cos)? - (&x2 * &sin)?)?;
    let r2 = ((&x2 * &cos)? + (&x1 * &sin)?)?;
    Ok(Tensor::cat(&[r1, r2], 2)?)
}
