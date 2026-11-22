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
    /// GGUF value-type 10 (`ValueType::U64`) — distinct from `U32` at the
    /// wire level. Exists because every other variant here emits `U32`
    /// (type 4), so `Value::to_u32()` and `Value::to_u64()` were
    /// indistinguishable on any fixture built without this — this is the
    /// one that discriminates them (`fuel-formats/src/gguf.rs:227,242` at
    /// pin `8771997e`: `to_u32` accepts ONLY `U32`, `to_u64` upcasts
    /// U8/U16/U32/U64/Bool).
    U64(u64),
    StrArray(Vec<&'static str>),
    /// A homogeneous array of `u32`s — used to put a present-but-not-a-string
    /// entry into `tokenizer.ggml.tokens` without going through `StrArray`.
    U32Array(Vec<u32>),
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
    // Not `let mut` — this closure captures nothing, so it is `Fn` and a
    // `mut` binding would only earn an `unused_mut` warning.
    let put_str = |b: &mut Vec<u8>, s: &str| {
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    };
    for (k, v) in kvs {
        put_str(&mut b, k);
        match v {
            Kv::Str(s) => {
                b.extend_from_slice(&8u32.to_le_bytes());
                put_str(&mut b, s);
            }
            Kv::U32(n) => {
                b.extend_from_slice(&4u32.to_le_bytes());
                b.extend_from_slice(&n.to_le_bytes());
            }
            Kv::U64(n) => {
                b.extend_from_slice(&10u32.to_le_bytes());
                b.extend_from_slice(&n.to_le_bytes());
            }
            Kv::StrArray(items) => {
                b.extend_from_slice(&9u32.to_le_bytes());
                b.extend_from_slice(&8u32.to_le_bytes()); // element type: string
                b.extend_from_slice(&(items.len() as u64).to_le_bytes());
                for it in items {
                    put_str(&mut b, it);
                }
            }
            Kv::U32Array(items) => {
                b.extend_from_slice(&9u32.to_le_bytes());
                b.extend_from_slice(&4u32.to_le_bytes()); // element type: u32
                b.extend_from_slice(&(items.len() as u64).to_le_bytes());
                for it in items {
                    b.extend_from_slice(&it.to_le_bytes());
                }
            }
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
