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

const SYN_HIDDEN: usize = 16;
const SYN_FFN: usize = 40;
const SYN_LAYERS: usize = 3;
const SYN_HEADS: usize = 4;

/// The synthetic GGUF's vocabulary: `<s>` (id 0), `</s>` (id 1), then the 94 printable ASCII characters `!`..`~`
/// (ids 2..=95, which GPT-2's byte-to-unicode map leaves as themselves) and the byte-level space (id 96).
/// 97 entries; a byte-level BPE with no merges encodes text over these characters.
pub fn synthetic_gguf_tokens() -> Vec<String> {
    let mut tokens = vec!["<s>".to_string(), "</s>".to_string()];
    tokens.extend((b'!'..=b'~').map(|b| (b as char).to_string()));
    tokens.push(char::from_u32(0x120).expect("a scalar value").to_string());
    tokens
}

/// Append one F32 tensor: its info record, and its data padded to the 32-byte alignment.
fn push_f32_tensor(
    infos: &mut Vec<u8>,
    data: &mut Vec<u8>,
    name: &str,
    dims: &[u64],
    values: &[f32],
) {
    put_str(infos, name);
    infos.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for d in dims {
        infos.extend_from_slice(&d.to_le_bytes());
    }
    infos.extend_from_slice(&0u32.to_le_bytes()); // GGML_TYPE_F32
    infos.extend_from_slice(&(data.len() as u64).to_le_bytes()); // offset in the data region
    for v in values {
        data.extend_from_slice(&v.to_le_bytes());
    }
    while !data.len().is_multiple_of(32) {
        data.push(0);
    }
}

fn synthetic_kvs(tokens: Vec<String>) -> Vec<(&'static str, Kv)> {
    vec![
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
    ]
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

/// The whole file: header, metadata, tensor infos, padding to the 32-byte alignment, tensor data.
pub fn constant_output_gguf_bytes(token: u32) -> Vec<u8> {
    let tokens = synthetic_gguf_tokens();
    let vocab = tokens.len();
    assert!((token as usize) < vocab, "token out of vocabulary");
    let tensors = synthetic_tensors(token, vocab);
    let (mut infos, mut data) = (Vec::new(), Vec::new());
    for (name, dims, values) in &tensors {
        push_f32_tensor(&mut infos, &mut data, name, dims, values);
    }
    let mut out = gguf_bytes(&synthetic_kvs(tokens), tensors.len() as u64);
    out.extend_from_slice(&infos);
    while !out.len().is_multiple_of(32) {
        out.push(0);
    }
    out.extend_from_slice(&data);
    out
}
