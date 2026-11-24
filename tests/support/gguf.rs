// SPDX-License-Identifier: MIT OR Apache-2.0
// Shared test support: a synthetic GGUF header writer (ROADMAP C1).
//
// A valid GGUF v3 header with ZERO tensors is a few hundred bytes, so a metadata-only test needs no real model.
// Reached by `#[path = "support/gguf.rs"] mod gguf_writer;` from each integration test that needs it: one copy
// to keep correct, not one per test file. Plain `//` comments, as in `support/mod.rs`.

/// One metadata value to write into a synthetic header.
pub enum Kv {
    Str(&'static str),
    U32(u32),
    /// GGUF value-type 6 (`F32`).
    F32(f32),
    /// GGUF value-type 10 (`ValueType::U64`) — distinct from `U32` at the
    /// wire level. Exists because every other variant here emits `U32`
    /// (type 4), so `Value::to_u32()` and `Value::to_u64()` were
    /// indistinguishable on any fixture built without this — this is the
    /// one that discriminates them (`fuel-formats/src/gguf.rs:227,242` at
    /// pin `8771997e`: `to_u32` accepts ONLY `U32`, `to_u64` upcasts
    /// U8/U16/U32/U64/Bool).
    U64(u64),
    StrArray(Vec<&'static str>),
    /// Like `StrArray`, for entries built at run time.
    OwnedStrArray(Vec<String>),
    /// A homogeneous array of `u32`s — used to put a present-but-not-a-string
    /// entry into `tokenizer.ggml.tokens` without going through `StrArray`.
    U32Array(Vec<u32>),
}

fn put_str(b: &mut Vec<u8>, s: &str) {
    b.extend_from_slice(&(s.len() as u64).to_le_bytes());
    b.extend_from_slice(s.as_bytes());
}

/// Write a scalar value (type tag, then payload). `false` when `v` is an array, which `put_array` writes.
fn put_scalar(b: &mut Vec<u8>, v: &Kv) -> bool {
    match v {
        Kv::Str(s) => {
            b.extend_from_slice(&8u32.to_le_bytes());
            put_str(b, s);
        }
        Kv::U32(n) => {
            b.extend_from_slice(&4u32.to_le_bytes());
            b.extend_from_slice(&n.to_le_bytes());
        }
        Kv::F32(n) => {
            b.extend_from_slice(&6u32.to_le_bytes());
            b.extend_from_slice(&n.to_le_bytes());
        }
        Kv::U64(n) => {
            b.extend_from_slice(&10u32.to_le_bytes());
            b.extend_from_slice(&n.to_le_bytes());
        }
        _ => return false,
    }
    true
}

/// Write an array value: type 9, the element type, the count, then the elements.
fn put_array(b: &mut Vec<u8>, v: &Kv) {
    b.extend_from_slice(&9u32.to_le_bytes());
    match v {
        Kv::StrArray(items) => put_str_elements(b, items.iter().copied()),
        Kv::OwnedStrArray(items) => put_str_elements(b, items.iter().map(String::as_str)),
        Kv::U32Array(items) => {
            b.extend_from_slice(&4u32.to_le_bytes()); // element type: u32
            b.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for it in items {
                b.extend_from_slice(&it.to_le_bytes());
            }
        }
        _ => unreachable!("a scalar goes through put_scalar"),
    }
}

fn put_str_elements<'a>(b: &mut Vec<u8>, items: impl ExactSizeIterator<Item = &'a str>) {
    b.extend_from_slice(&8u32.to_le_bytes()); // element type: string
    b.extend_from_slice(&(items.len() as u64).to_le_bytes());
    for it in items {
        put_str(b, it);
    }
}

/// `tensor_count` is a parameter, not a constant, because Task 2 needs a
/// one-tensor file to reach the unsupported-dtype path. Pass `0` for every
/// metadata-only fixture; the tensor-info records themselves are appended by
/// the caller (see `gguf_fixture_with_tensor`).
pub fn gguf_bytes(kvs: &[(&str, Kv)], tensor_count: u64) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes()); // version
    b.extend_from_slice(&tensor_count.to_le_bytes());
    b.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for (k, v) in kvs {
        put_str(&mut b, k);
        if !put_scalar(&mut b, v) {
            put_array(&mut b, v);
        }
    }
    b
}

/// A header with exactly one tensor-info record (name, 1 dim of size 1, GGML dtype code, offset 0). The tensor's
/// DATA is never written: `Content::read` parses the info table and stops, which is what keeps this a few
/// hundred bytes. Layout per `fuel-formats/src/gguf.rs:415-434`: u64 name-len + name, u32 n_dims, n_dims x u64
/// dims, u32 dtype code, u64 offset.
pub fn gguf_bytes_with_tensor(kvs: &[(&str, Kv)], name: &str, dtype: u32) -> Vec<u8> {
    let mut body = gguf_bytes(kvs, 1);
    body.extend_from_slice(&(name.len() as u64).to_le_bytes());
    body.extend_from_slice(name.as_bytes());
    body.extend_from_slice(&1u32.to_le_bytes()); // n_dims
    body.extend_from_slice(&1u64.to_le_bytes()); // dims[0]
    body.extend_from_slice(&dtype.to_le_bytes());
    body.extend_from_slice(&0u64.to_le_bytes()); // offset
    body
}

// ---- A servable synthetic GGUF (ROADMAP C1) -------------------------------------------------------------------
//
// A tiny Llama-shaped GGUF with real F32 tensor data, so the GGUF serving path (header, tokenizer rebuild, chat
// template, weight load, generation) runs without a downloaded model. Like `write_constant_output_checkpoint`
// in `support/mod.rs`, its greedy output is one fixed token: every embedding row is the same constant vector,
// every projection is zero, and the output row for `token` is all ones, so that token's logit is the hidden
// size and every other row's is at most a tenth of it.

// 256 and 256: a multiple of the K-quant super-block size (256), so every quantised row (a projection's row is
// `hidden` or `ffn` long) is whole blocks of every dtype here, Q8_0's 32 and Q4_0's 32 included.
const SYN_HIDDEN: usize = 256;
const SYN_FFN: usize = 256;
const SYN_LAYERS: usize = 3;
const SYN_HEADS: usize = 4;

/// The synthetic GGUF's vocabulary: `<s>` (id 0), `</s>` (id 1), then the 94 printable ASCII characters `!`..`~`
/// (ids 2..=95, which GPT-2's byte-to-unicode map leaves as themselves), the byte-level space (id 96) and `<pad>`
/// (id 97). 98 entries, EVEN on purpose: a Q8_0 matrix with an odd row count is not a whole number of u32s
/// (34-byte blocks) and Fuel's `qmatmul` refuses it. A byte-level BPE with no merges encodes text over the
/// printable characters.
pub fn synthetic_gguf_tokens() -> Vec<String> {
    let mut tokens = vec!["<s>".to_string(), "</s>".to_string()];
    tokens.extend((b'!'..=b'~').map(|b| (b as char).to_string()));
    tokens.push(char::from_u32(0x120).expect("a scalar value").to_string());
    tokens.push("<pad>".to_string());
    tokens
}

const GGML_F32: u32 = 0;
const GGML_Q4_0: u32 = 2;
const GGML_Q8_0: u32 = 8;
const GGML_Q8_1: u32 = 9;
const GGML_Q6_K: u32 = 14;

/// IEEE half-precision bits of `x`, for the normal-range values a block scale takes here (`0.0` is `0`).
fn f16_bits(x: f32) -> u16 {
    if x == 0.0 {
        return 0;
    }
    assert!(x.is_normal(), "only normal scales");
    let sign = if x < 0.0 { 0x8000u16 } else { 0 };
    let bits = x.abs().to_bits();
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mut mant = ((bits & 0x7f_ffff) + 0x1000) >> 13; // round to nearest
    if mant == 0x400 {
        exp += 1;
        mant = 0;
    }
    assert!(
        (1..=30).contains(&exp),
        "scale {x} outside the half normal range"
    );
    sign | ((exp as u16) << 10) | mant as u16
}

/// Q8_0: per block of 32 values, an f16 scale `max|x| / 127` then 32 signed bytes `round(x / scale)`.
fn quantize_q8_0(values: &[f32]) -> Vec<u8> {
    assert!(
        values.len().is_multiple_of(32),
        "Q8_0 rows are whole blocks of 32"
    );
    let mut out = Vec::with_capacity(values.len() / 32 * 34);
    for block in values.chunks(32) {
        let scale = block.iter().fold(0.0f32, |m, v| m.max(v.abs())) / 127.0;
        // The scale is stored as an f16, so quantise against the value the reader will use.
        let bits = f16_bits(scale);
        out.extend_from_slice(&bits.to_le_bytes());
        for v in block {
            let q = if scale == 0.0 {
                0.0
            } else {
                (v / scale).round()
            };
            out.push(q as i8 as u8);
        }
    }
    out
}

/// Q4_0: per block of 32 values, an f16 scale `d = (the value of largest magnitude) / -8` then 16 bytes packing
/// 32 nibbles `q = clamp(round(x / d) + 8, 0, 15)`, element `i` in the low nibble of byte `i` and element
/// `i + 16` in its high nibble (llama.cpp's layout); a value reads back as `(q - 8) * d`.
fn quantize_q4_0(values: &[f32]) -> Vec<u8> {
    assert!(
        values.len().is_multiple_of(32),
        "Q4_0 rows are whole blocks of 32"
    );
    let mut out = Vec::with_capacity(values.len() / 32 * 18);
    for block in values.chunks(32) {
        let extreme = block
            .iter()
            .fold(0.0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
        let scale = extreme / -8.0;
        out.extend_from_slice(&f16_bits(scale).to_le_bytes());
        let nibble = |v: f32| -> u8 {
            let q = if scale == 0.0 {
                8.0
            } else {
                (v / scale).round() + 8.0
            };
            q.clamp(0.0, 15.0) as u8
        };
        for i in 0..16 {
            out.push(nibble(block[i]) | (nibble(block[i + 16]) << 4));
        }
    }
    out
}

/// Q8_1: per block of 32 values, an f16 scale `d = max|x| / 127`, an f16 `s = d * sum(q)`, then 32 signed bytes.
fn quantize_q8_1(values: &[f32]) -> Vec<u8> {
    assert!(
        values.len().is_multiple_of(32),
        "Q8_1 rows are whole blocks of 32"
    );
    let mut out = Vec::with_capacity(values.len() / 32 * 36);
    for block in values.chunks(32) {
        let scale = block.iter().fold(0.0f32, |m, v| m.max(v.abs())) / 127.0;
        let q: Vec<i8> = block
            .iter()
            .map(|v| {
                if scale == 0.0 {
                    0
                } else {
                    (v / scale).round() as i8
                }
            })
            .collect();
        let sum: i32 = q.iter().map(|&x| i32::from(x)).sum();
        out.extend_from_slice(&f16_bits(scale).to_le_bytes());
        out.extend_from_slice(&f16_bits(scale * sum as f32).to_le_bytes());
        out.extend(q.iter().map(|&x| x as u8));
    }
    out
}

/// Q6_K: super-blocks of 256 values, 210 bytes each: `ql[128]` (low 4 bits), `qh[64]` (high 2 bits), 16 `i8`
/// scales (one per 16 values) and an f16 `d`. A value reads back as `d * scale[group] * (q - 32)`, `q` in 0..64.
/// Within each half (128 values) element `l` (0..32) of the four quarters shares `ql[l]`/`ql[l + 32]` (low and
/// high nibble) and `qh[l]` (four 2-bit fields); quarter `k` uses scale `8h + 2k + l / 16` (llama.cpp's layout).
pub fn quantize_q6_k(values: &[f32]) -> Vec<u8> {
    assert!(
        values.len().is_multiple_of(256),
        "Q6_K rows are whole super-blocks of 256"
    );
    let mut out = Vec::with_capacity(values.len() / 256 * 210);
    for block in values.chunks(256) {
        out.extend(q6_k_block(block));
    }
    out
}

/// One Q6_K super-block (see [`quantize_q6_k`]).
fn q6_k_block(block: &[f32]) -> Vec<u8> {
    let group_scale: Vec<f32> = block
        .chunks(16)
        .map(|g| {
            let extreme = g
                .iter()
                .fold(0.0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
            extreme / -32.0
        })
        .collect();
    let d = group_scale.iter().fold(0.0f32, |m, v| m.max(v.abs())) / 127.0;
    // An f16 scale must be a normal half (>= 2^-14); a smaller one would need a subnormal this writer does not emit.
    let d = if d > 0.0 { d.max(1e-4) } else { 0.0 };
    let scales: Vec<i8> = group_scale
        .iter()
        .map(|&s| {
            if d == 0.0 {
                0
            } else {
                (s / d).round().clamp(-127.0, 127.0) as i8
            }
        })
        .collect();
    let q = |i: usize| -> u8 {
        let step = d * f32::from(scales[i / 16]);
        let v = if step == 0.0 {
            0.0
        } else {
            (block[i] / step).round()
        };
        (v + 32.0).clamp(0.0, 63.0) as u8
    };
    let (mut ql, mut qh) = ([0u8; 128], [0u8; 64]);
    for half in 0..2 {
        for l in 0..32 {
            let at = |k: usize| q(half * 128 + k * 32 + l);
            ql[half * 64 + l] = (at(0) & 0xF) | ((at(2) & 0xF) << 4);
            ql[half * 64 + l + 32] = (at(1) & 0xF) | ((at(3) & 0xF) << 4);
            qh[half * 32 + l] =
                (at(0) >> 4) | ((at(1) >> 4) << 2) | ((at(2) >> 4) << 4) | ((at(3) >> 4) << 6);
        }
    }
    let mut out = Vec::with_capacity(210);
    out.extend_from_slice(&ql);
    out.extend_from_slice(&qh);
    out.extend(scales.iter().map(|&x| x as u8));
    out.extend_from_slice(&f16_bits(d).to_le_bytes());
    out
}

// ---- Q4_K and Q5_K (ROADMAP C1b follow-up) ---------------------------------------------------------------------
//
// PROVENANCE. The block LAYOUTS (`block_q4_K`: f16 `d`, f16 `dmin`, 12 packed 6-bit scale/min bytes, 128 nibble
// bytes = 144 bytes; `block_q5_K`: the same plus 32 high-bit bytes = 176 bytes) and the scale/min packing are
// ggml's, from `ggml/src/ggml-common.h` and `get_scale_min_k4` / `dequantize_row_q4_K` / `dequantize_row_q5_K` in
// `ggml/src/ggml-quants.c`, read at ggml-org/llama.cpp commit 76f186df71a41abb3ee20fffedee31ca5dbc7d9f
// (2026-11-19). ggml is MIT, Copyright (c) 2023-2026 The ggml authors (licence text in
// `docs/THIRD_PARTY_NOTICES.md`). No ggml code is copied: the ENCODER below is written from the decode arithmetic
// (`x = d * sc * q - dmin * m`) with a plain min/max quantiser, not ggml's `make_qkx2_quants` search, so its output
// is a valid block that ggml (or Fuel) reads back, not ggml's byte-identical output.

/// Pack eight 6-bit scales and eight 6-bit mins into ggml's 12 bytes (the inverse of `get_scale_min_k4`).
fn pack_scale_min_k4(ls: &[u8; 8], lm: &[u8; 8]) -> [u8; 12] {
    let mut s = [0u8; 12];
    s[..4].copy_from_slice(&ls[..4]);
    s[4..8].copy_from_slice(&lm[..4]);
    for j in 4..8 {
        s[j + 4] = (ls[j] & 0xF) | ((lm[j] & 0xF) << 4);
        s[j - 4] |= (ls[j] >> 4) << 6;
        s[j] |= (lm[j] >> 4) << 6;
    }
    s
}

/// One 256-value "affine K" super-block: eight 32-value sub-blocks, each `x = d * sc * q - dmin * m` with 6-bit `sc`
/// and `m` and `q` in `0..=qmax`. Returns `(d, dmin, packed scales, q[256])`.
fn affine_k_block(block: &[f32], qmax: f32) -> (f32, f32, [u8; 12], [u8; 256]) {
    let mut sub_scale = [0f32; 8];
    let mut sub_min = [0f32; 8]; // stored positive: the value subtracted
    for (g, sub) in block.chunks(32).enumerate() {
        let lo = sub.iter().fold(0.0f32, |m, &v| m.min(v)); // include 0 so the min is never above it
        let hi = sub.iter().fold(0.0f32, |m, &v| m.max(v));
        sub_scale[g] = (hi - lo) / qmax;
        sub_min[g] = -lo;
    }
    let max_of = |a: &[f32; 8]| a.iter().fold(0.0f32, |m, &v| m.max(v));
    // A scale must be a normal half (>= 2^-14); a smaller one would need a subnormal this writer does not emit.
    let norm = |x: f32| if x > 0.0 { x.max(1e-4) } else { 0.0 };
    let d = norm(max_of(&sub_scale) / 63.0);
    let dmin = norm(max_of(&sub_min) / 63.0);
    let six = |x: f32, unit: f32| {
        if unit == 0.0 {
            0u8
        } else {
            (x / unit).round().clamp(0.0, 63.0) as u8
        }
    };
    let (mut ls, mut lm) = ([0u8; 8], [0u8; 8]);
    for g in 0..8 {
        ls[g] = six(sub_scale[g], d);
        lm[g] = six(sub_min[g], dmin);
    }
    let mut q = [0u8; 256];
    for (i, &v) in block.iter().enumerate() {
        let (step, off) = (d * f32::from(ls[i / 32]), dmin * f32::from(lm[i / 32]));
        q[i] = if step == 0.0 {
            0
        } else {
            ((v + off) / step).round().clamp(0.0, qmax) as u8
        };
    }
    (d, dmin, pack_scale_min_k4(&ls, &lm), q)
}

/// Q4_K: super-blocks of 256 values, 144 bytes each: f16 `d`, f16 `dmin`, 12 scale/min bytes, `qs[128]`. In each
/// 64-value chunk `c`, sub-block `2c` is the LOW nibbles of `qs[32c..32c + 32]` and sub-block `2c + 1` the HIGH ones.
pub fn quantize_q4_k(values: &[f32]) -> Vec<u8> {
    assert!(
        values.len().is_multiple_of(256),
        "Q4_K rows are whole super-blocks of 256"
    );
    let mut out = Vec::with_capacity(values.len() / 256 * 144);
    for block in values.chunks(256) {
        let (d, dmin, scales, q) = affine_k_block(block, 15.0);
        out.extend_from_slice(&f16_bits(d).to_le_bytes());
        out.extend_from_slice(&f16_bits(dmin).to_le_bytes());
        out.extend_from_slice(&scales);
        for c in 0..4 {
            for l in 0..32 {
                out.push(q[c * 64 + l] | (q[c * 64 + 32 + l] << 4));
            }
        }
    }
    out
}

/// Q5_K: super-blocks of 256 values, 176 bytes each: f16 `d`, f16 `dmin`, 12 scale/min bytes, `qh[32]`, `qs[128]`.
/// As Q4_K, plus the fifth bit of element `l` of sub-block `s` in bit `s` of `qh[l]`.
pub fn quantize_q5_k(values: &[f32]) -> Vec<u8> {
    assert!(
        values.len().is_multiple_of(256),
        "Q5_K rows are whole super-blocks of 256"
    );
    let mut out = Vec::with_capacity(values.len() / 256 * 176);
    for block in values.chunks(256) {
        let (d, dmin, scales, q) = affine_k_block(block, 31.0);
        let mut qh = [0u8; 32];
        for s in 0..8 {
            for l in 0..32 {
                qh[l] |= (q[s * 32 + l] >> 4) << s;
            }
        }
        out.extend_from_slice(&f16_bits(d).to_le_bytes());
        out.extend_from_slice(&f16_bits(dmin).to_le_bytes());
        out.extend_from_slice(&scales);
        out.extend_from_slice(&qh);
        for c in 0..4 {
            for l in 0..32 {
                out.push((q[c * 64 + l] & 0xF) | ((q[c * 64 + 32 + l] & 0xF) << 4));
            }
        }
    }
    out
}

/// How a tensor is stored in the synthetic file.
#[derive(Clone, Copy, PartialEq)]
pub enum Storage {
    F32,
    Q8_0,
    Q8_1,
    Q4_0,
    Q6K,
}

/// Append one tensor: its info record, and its data padded to the 32-byte alignment, encoded per `storage`.
fn push_tensor(
    infos: &mut Vec<u8>,
    data: &mut Vec<u8>,
    (name, dims, values): (&str, &[u64], &[f32]),
    storage: Storage,
) {
    put_str(infos, name);
    infos.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for d in dims {
        infos.extend_from_slice(&d.to_le_bytes());
    }
    let (code, bytes) = match storage {
        Storage::Q8_0 => (GGML_Q8_0, quantize_q8_0(values)),
        Storage::Q4_0 => (GGML_Q4_0, quantize_q4_0(values)),
        Storage::Q8_1 => (GGML_Q8_1, quantize_q8_1(values)),
        Storage::Q6K => (GGML_Q6_K, quantize_q6_k(values)),
        Storage::F32 => (
            GGML_F32,
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
    };
    infos.extend_from_slice(&code.to_le_bytes());
    infos.extend_from_slice(&(data.len() as u64).to_le_bytes()); // offset in the data region
    data.extend_from_slice(&bytes);
    while !data.len().is_multiple_of(32) {
        data.push(0);
    }
}

fn synthetic_kvs(
    tokens: Vec<String>,
    chat_template: Option<&'static str>,
) -> Vec<(&'static str, Kv)> {
    let mut kvs = vec![
        ("general.architecture", Kv::Str("llama")),
        ("general.name", Kv::Str("synthetic-constant-output")),
        ("llama.context_length", Kv::U32(64)),
        ("llama.embedding_length", Kv::U32(SYN_HIDDEN as u32)),
        ("llama.block_count", Kv::U32(SYN_LAYERS as u32)),
        ("llama.feed_forward_length", Kv::U32(SYN_FFN as u32)),
        ("llama.attention.head_count", Kv::U32(SYN_HEADS as u32)),
        ("llama.attention.head_count_kv", Kv::U32(SYN_HEADS as u32)),
        ("llama.attention.layer_norm_rms_epsilon", Kv::F32(1e-5)),
        ("llama.rope.freq_base", Kv::F32(10000.0)),
        ("llama.vocab_size", Kv::U32(tokens.len() as u32)),
        ("tokenizer.ggml.model", Kv::Str("gpt2")),
        ("tokenizer.ggml.pre", Kv::Str("gpt-2")),
        ("tokenizer.ggml.tokens", Kv::OwnedStrArray(tokens)),
        ("tokenizer.ggml.merges", Kv::OwnedStrArray(Vec::new())),
        ("tokenizer.ggml.bos_token_id", Kv::U32(0)),
        ("tokenizer.ggml.eos_token_id", Kv::U32(1)),
    ];
    if let Some(t) = chat_template {
        kvs.push(("tokenizer.chat_template", Kv::Str(t)));
    }
    kvs
}

/// The output matrix: small deterministic values in [-0.05, 0.05) (an LCG; no `rand`), so they cannot win, with
/// the row for `token` all ones.
fn output_matrix(token: u32, vocab: usize) -> Vec<f32> {
    let h = SYN_HIDDEN;
    let mut state = 0x2545_F491u32;
    let mut head: Vec<f32> = (0..h * vocab)
        .map(|_| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((state >> 16) as f32 / 65_536.0 - 0.5) * 0.1
        })
        .collect();
    head[token as usize * h..(token as usize + 1) * h].fill(1.0);
    head
}

/// Every tensor of the model as `(name, dims, values)`, in file order.
fn synthetic_tensors(token: u32, vocab: usize) -> Vec<(String, Vec<u64>, Vec<f32>)> {
    let (h, f) = (SYN_HIDDEN, SYN_FFN);
    let (hd, fd, vd) = (h as u64, f as u64, vocab as u64);
    let mut t = vec![(
        "token_embd.weight".to_string(),
        vec![hd, vd],
        vec![0.5; h * vocab],
    )];
    for i in 0..SYN_LAYERS {
        t.push((format!("blk.{i}.attn_norm.weight"), vec![hd], vec![1.0; h]));
        for n in ["attn_q", "attn_k", "attn_v", "attn_output"] {
            t.push((
                format!("blk.{i}.{n}.weight"),
                vec![hd, hd],
                vec![0.0; h * h],
            ));
        }
        t.push((format!("blk.{i}.ffn_norm.weight"), vec![hd], vec![1.0; h]));
        for n in ["ffn_gate", "ffn_up"] {
            t.push((
                format!("blk.{i}.{n}.weight"),
                vec![hd, fd],
                vec![0.0; h * f],
            ));
        }
        t.push((
            format!("blk.{i}.ffn_down.weight"),
            vec![fd, hd],
            vec![0.0; f * h],
        ));
    }
    t.push(("output_norm.weight".to_string(), vec![hd], vec![1.0; h]));
    t.push((
        "output.weight".to_string(),
        vec![hd, vd],
        output_matrix(token, vocab),
    ));
    t
}

/// The whole file, all tensors F32: header, metadata, tensor infos, padding to the 32-byte alignment, tensor data.
pub fn constant_output_gguf_bytes(token: u32) -> Vec<u8> {
    constant_output_gguf_bytes_with(token, Storage::F32)
}

/// The same file with every matrix quantised to Q8_0 (the norm vectors stay F32, as in llama.cpp's own files), so
/// the quantised load and matmul path runs.
pub fn constant_output_gguf_q8_0_bytes(token: u32) -> Vec<u8> {
    constant_output_gguf_bytes_with(token, Storage::Q8_0)
}

/// The same file with every matrix quantised to Q4_0 (norms F32).
pub fn constant_output_gguf_q4_0_bytes(token: u32) -> Vec<u8> {
    constant_output_gguf_bytes_with(token, Storage::Q4_0)
}

/// The same file with every matrix quantised to Q6_K, a K-quant (norms F32).
pub fn constant_output_gguf_q6_k_bytes(token: u32) -> Vec<u8> {
    constant_output_gguf_bytes_with(token, Storage::Q6K)
}

/// The same file with every matrix quantised to Q8_1 (norms F32).
pub fn constant_output_gguf_q8_1_bytes(token: u32) -> Vec<u8> {
    constant_output_gguf_bytes_with(token, Storage::Q8_1)
}

fn constant_output_gguf_bytes_with(token: u32, matrices: Storage) -> Vec<u8> {
    gguf_bytes_full(token, matrices, None)
}

/// The F32 file with `tokenizer.chat_template` set to `template` (it must stay within the synthetic vocabulary:
/// printable ASCII and spaces, no newline).
pub fn constant_output_gguf_with_template_bytes(token: u32, template: &'static str) -> Vec<u8> {
    gguf_bytes_full(token, Storage::F32, Some(template))
}

fn gguf_bytes_full(token: u32, matrices: Storage, chat_template: Option<&'static str>) -> Vec<u8> {
    let vocab = synthetic_gguf_tokens().len();
    assert!((token as usize) < vocab, "token out of vocabulary");
    assemble(synthetic_tensors(token, vocab), matrices, chat_template)
}

/// A file whose model emits `chain` in order and then the EOS (id 1), whatever the prompt: the answer's LENGTH is
/// known, so a stop that arrives after exactly `chain.len()` tokens (and a cap that cuts it short) can be asserted.
///
/// How: every projection is zero, so the final hidden state is the (RMS-normalised) embedding of the last token,
/// and the logits are the output matrix times it. With one-hot embeddings that makes the output matrix a lookup
/// table, token to next token: `chain[i]` goes to `chain[i + 1]`, the last goes to the EOS, and every other token
/// (the prompt's) goes to `chain[0]`.
pub fn chain_output_gguf_bytes(chain: &[u32]) -> Vec<u8> {
    chain_gguf(chain, None)
}

/// [`chain_output_gguf_bytes`] with `tokenizer.chat_template` set (same vocabulary limits as
/// [`constant_output_gguf_with_template_bytes`]).
pub fn chain_output_gguf_with_template_bytes(chain: &[u32], template: &'static str) -> Vec<u8> {
    chain_gguf(chain, Some(template))
}

fn chain_gguf(chain: &[u32], chat_template: Option<&'static str>) -> Vec<u8> {
    let vocab = synthetic_gguf_tokens().len();
    assert!(!chain.is_empty() && chain.iter().all(|&t| (t as usize) < vocab && t > 1));
    let h = SYN_HIDDEN;
    assert!(vocab <= h, "one-hot embeddings need hidden >= vocab");
    let mut next = vec![chain[0]; vocab];
    for (i, &t) in chain.iter().enumerate() {
        next[t as usize] = chain.get(i + 1).copied().unwrap_or(1);
    }
    let mut tensors = synthetic_tensors(chain[0], vocab);
    for (name, _, values) in &mut tensors {
        if name == "token_embd.weight" {
            values.fill(0.0);
            (0..vocab).for_each(|t| values[t * h + t] = 1.0);
        } else if name == "output.weight" {
            values.fill(0.0);
            (0..vocab).for_each(|t| values[next[t] as usize * h + t] = 1.0);
        }
    }
    assemble(tensors, Storage::F32, chat_template)
}

/// The file bytes for `tensors`, each matrix stored as `matrices` (norms F32).
fn assemble(
    tensors: Vec<(String, Vec<u64>, Vec<f32>)>,
    matrices: Storage,
    chat_template: Option<&'static str>,
) -> Vec<u8> {
    let tokens = synthetic_gguf_tokens();
    let (mut infos, mut data) = (Vec::new(), Vec::new());
    for (name, dims, values) in &tensors {
        let is_norm = name.ends_with("norm.weight");
        let storage = if is_norm { Storage::F32 } else { matrices };
        push_tensor(&mut infos, &mut data, (name, dims, values), storage);
    }
    let mut out = gguf_bytes(&synthetic_kvs(tokens, chat_template), tensors.len() as u64);
    out.extend_from_slice(&infos);
    while !out.len().is_multiple_of(32) {
        out.push(0);
    }
    out.extend_from_slice(&data);
    out
}
