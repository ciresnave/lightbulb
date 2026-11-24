// SPDX-License-Identifier: MIT OR Apache-2.0
//! The synthetic Q4_K and Q5_K encoders (`support/gguf.rs`) against an INDEPENDENT decode: a straight port of
//! ggml's `get_scale_min_k4`, `dequantize_row_q4_K`, `_q5_K` and `_q3_K` (ggml-org/llama.cpp commit
//! 76f186df71a41abb3ee20fffedee31ca5dbc7d9f, `ggml/src/ggml-quants.c`; MIT, Copyright (c) 2023-2026 The ggml
//! authors, see `docs/THIRD_PARTY_NOTICES.md`). The decoder shares no code with the encoder: a layout slip on
//! either side shows as an error far above the bound.
//!
//! The bound is the format's own: per 32-value sub-block, half a quantisation step, plus the error of rounding
//! the 6-bit scale and min against their f16 super-scales. The `*_seeded_bug_*` tests corrupt a valid block
//! (high-bit plane dropped, scale and min swapped) and require the bound to be VIOLATED, so the bound is shown
//! to discriminate rather than assumed to.

#[path = "support/gguf.rs"]
#[allow(dead_code)]
mod gguf_writer;

fn f16(bits: u16) -> f32 {
    let (e, m) = (i32::from((bits >> 10) & 0x1f), f32::from(bits & 0x3ff));
    let v = if e == 0 {
        m / 1024.0 * 2f32.powi(-14)
    } else {
        (1.0 + m / 1024.0) * 2f32.powi(e - 15)
    };
    if bits & 0x8000 != 0 { -v } else { v }
}

/// ggml `get_scale_min_k4`, line for line.
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// ggml `dequantize_row_q4_K` / `_q5_K` over `blocks` (144 or 176 bytes each).
fn dequantize(blocks: &[u8], q5: bool) -> Vec<f32> {
    let size = if q5 { 176 } else { 144 };
    assert_eq!(blocks.len() % size, 0);
    let mut y = Vec::new();
    for b in blocks.chunks(size) {
        let d = f16(u16::from_le_bytes([b[0], b[1]]));
        let min = f16(u16::from_le_bytes([b[2], b[3]]));
        let scales = &b[4..16];
        let (qh, qs) = if q5 {
            (&b[16..48], &b[48..])
        } else {
            (&b[0..0], &b[16..])
        };
        let (mut is, mut u1, mut u2) = (0usize, 1u8, 2u8);
        for c in 0..4 {
            let ql = &qs[c * 32..c * 32 + 32];
            let (sc, m) = get_scale_min_k4(is, scales);
            let (d1, m1) = (d * f32::from(sc), min * f32::from(m));
            let (sc, m) = get_scale_min_k4(is + 1, scales);
            let (d2, m2) = (d * f32::from(sc), min * f32::from(m));
            for l in 0..32 {
                let hi = if q5 && qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                y.push(d1 * (f32::from(ql[l] & 0xF) + hi) - m1);
            }
            for l in 0..32 {
                let hi = if q5 && qh[l] & u2 != 0 { 16.0 } else { 0.0 };
                y.push(d2 * (f32::from(ql[l] >> 4) + hi) - m2);
            }
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    y
}

/// Deterministic varied input: signed, mixed magnitudes per sub-block, some positive-only and all-zero sub-blocks.
fn values(n: usize) -> Vec<f32> {
    let mut s = 0x9E37_79B9u32;
    let mut next = move || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (s >> 8) as f32 / (1u32 << 24) as f32 // [0, 1)
    };
    (0..n)
        .map(|i| {
            let sub = i / 32;
            let amp = 0.05 + (sub % 7) as f32 * 0.4;
            match sub % 5 {
                0 => next() * amp,               // positive only
                1 => 0.0,                        // all zero
                _ => (next() * 2.0 - 1.0) * amp, // signed
            }
        })
        .collect()
}

/// Per-element error bound (see the module comment), from the INPUT only.
fn bounds(vals: &[f32], qmax: f32) -> Vec<f32> {
    let range = |s: &[f32]| {
        let lo = s.iter().fold(0.0f32, |m, &v| m.min(v));
        let hi = s.iter().fold(0.0f32, |m, &v| m.max(v));
        (hi - lo, -lo)
    };
    let mut out = Vec::new();
    for sb in vals.chunks(256) {
        let subs: Vec<(f32, f32)> = sb.chunks(32).map(range).collect();
        let d = subs.iter().fold(0.0f32, |m, s| m.max(s.0 / qmax)) / 63.0;
        let dmin = subs.iter().fold(0.0f32, |m, s| m.max(s.1)) / 63.0;
        let widest = subs.iter().fold(0.0f32, |m, s| m.max(s.0 + s.1));
        for &(span, _) in &subs {
            let b =
                0.5 * span / qmax + d / 4.0 + dmin / 2.0 + qmax * d / 2.0 + widest * 2f32.powi(-9);
            out.extend(std::iter::repeat_n(b * 1.05 + 1e-5, 32));
        }
    }
    out
}

fn worst_excess(got: &[f32], want: &[f32], bound: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .zip(bound)
        .map(|((g, w), b)| (g - w).abs() - b)
        .fold(f32::MIN, f32::max)
}

fn check_round_trip(q5: bool) {
    let (qmax, size) = if q5 { (31.0, 176) } else { (15.0, 144) };
    let vals = values(1024);
    let bytes = if q5 {
        gguf_writer::quantize_q5_k(&vals)
    } else {
        gguf_writer::quantize_q4_k(&vals)
    };
    assert_eq!(bytes.len(), 4 * size);
    let got = dequantize(&bytes, q5);
    assert_eq!(got.len(), vals.len());
    let excess = worst_excess(&got, &vals, &bounds(&vals, qmax));
    assert!(excess <= 0.0, "worst element exceeds its bound by {excess}");
}

#[test]
fn the_q4_k_encoder_round_trips_through_the_ggml_decode() {
    check_round_trip(false);
}

#[test]
fn the_q5_k_encoder_round_trips_through_the_ggml_decode() {
    check_round_trip(true);
}

/// The decode is not vacuous and the bound is not slack: a valid Q5_K block with its high-bit plane dropped, or
/// its scale and min bytes swapped, must violate the bound.
#[test]
fn the_seeded_bugs_violate_the_bound() {
    let vals = values(1024);
    for q5 in [false, true] {
        let qmax = if q5 { 31.0 } else { 15.0 };
        let good = if q5 {
            gguf_writer::quantize_q5_k(&vals)
        } else {
            gguf_writer::quantize_q4_k(&vals)
        };
        let bound = bounds(&vals, qmax);
        assert!(
            worst_excess(&dequantize(&good, q5), &vals, &bound) <= 0.0,
            "control: the unmutated blocks pass"
        );
        let size = if q5 { 176 } else { 144 };
        let mut swapped = good.clone();
        for b in swapped.chunks_mut(size) {
            for j in 0..4 {
                b.swap(4 + j, 8 + j); // scales[j] <-> scales[j + 4]: scale and min swapped
            }
        }
        assert!(
            worst_excess(&dequantize(&swapped, q5), &vals, &bound) > 0.0,
            "q5={q5}: scale/min swap not caught"
        );
        if q5 {
            let mut dropped = good.clone();
            for b in dropped.chunks_mut(size) {
                b[16..48].fill(0); // the fifth-bit plane
            }
            assert!(
                worst_excess(&dequantize(&dropped, q5), &vals, &bound) > 0.0,
                "high-bit drop not caught"
            );
        }
    }
}

// ---- Q3_K ------------------------------------------------------------------------------------------------------

/// ggml `dequantize_row_q3_K` (110-byte blocks), including its `aux` scale unpacking, written over u32 words.
fn dequantize_q3_k(blocks: &[u8]) -> Vec<f32> {
    assert_eq!(blocks.len() % 110, 0);
    let (kmask1, kmask2) = (0x0303_0303u32, 0x0f0f_0f0fu32);
    let mut y = Vec::new();
    for b in blocks.chunks(110) {
        let (hm, rest) = b.split_at(32);
        let (qs, rest) = rest.split_at(64);
        let (raw, dbits) = rest.split_at(12);
        let d_all = f16(u16::from_le_bytes([dbits[0], dbits[1]]));
        let word = |i: usize| {
            u32::from_le_bytes([raw[4 * i], raw[4 * i + 1], raw[4 * i + 2], raw[4 * i + 3]])
        };
        let mut aux = [word(0), word(1), word(2), 0];
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | ((tmp & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        let scales: Vec<u8> = aux.iter().flat_map(|w| w.to_le_bytes()).collect();
        let (mut is, mut m) = (0usize, 1u8);
        for n in 0..2 {
            let q = &qs[n * 32..];
            let mut shift = 0;
            for _ in 0..4 {
                for half in 0..2 {
                    let dl = d_all * (f32::from(scales[is]) - 32.0);
                    is += 1;
                    for l in 0..16 {
                        let low = i32::from((q[half * 16 + l] >> shift) & 3);
                        let high = if hm[half * 16 + l] & m != 0 { 0 } else { 4 };
                        y.push(dl * (low - high) as f32);
                    }
                }
                shift += 2;
                m <<= 1;
            }
        }
    }
    y
}

/// Per-element bound for Q3_K, from the input only: a 3-bit step is `|extreme| / 4`, and the opposite-signed
/// extreme can clamp at +3 (one step), plus the 6-bit scale rounding against `d` and the f16 `d`.
fn q3_bounds(vals: &[f32]) -> Vec<f32> {
    let mut out = Vec::new();
    for sb in vals.chunks(256) {
        let ext: Vec<f32> = sb
            .chunks(16)
            .map(|g| g.iter().fold(0.0f32, |m, v| m.max(v.abs())))
            .collect();
        let d = ext.iter().fold(0.0f32, |m, &e| m.max(e)) / 4.0 / 31.0;
        let widest = ext.iter().fold(0.0f32, |m, &e| m.max(e));
        for e in ext {
            let b = e / 4.0 + 3.0 * d + widest * 2f32.powi(-9);
            out.extend(std::iter::repeat_n(b * 1.05 + 1e-5, 16));
        }
    }
    out
}

#[test]
fn the_q3_k_encoder_round_trips_through_the_ggml_decode() {
    let vals = values(1024);
    let bytes = gguf_writer::quantize_q3_k(&vals);
    assert_eq!(bytes.len(), 4 * 110);
    let got = dequantize_q3_k(&bytes);
    assert_eq!(got.len(), vals.len());
    let excess = worst_excess(&got, &vals, &q3_bounds(&vals));
    assert!(excess <= 0.0, "worst element exceeds its bound by {excess}");
}

/// The Q3_K bound discriminates: a valid block with its `hmask` zeroed (the term that carries "q >= 0"), with its
/// scale bytes' high bits cleared, or with two scale words swapped, must violate it.
#[test]
fn the_q3_k_seeded_bugs_violate_the_bound() {
    let vals = values(1024);
    let good = gguf_writer::quantize_q3_k(&vals);
    let bound = q3_bounds(&vals);
    assert!(
        worst_excess(&dequantize_q3_k(&good), &vals, &bound) <= 0.0,
        "control: unmutated blocks pass"
    );
    type Bug = (&'static str, fn(&mut [u8]));
    let bugs: [Bug; 3] = [
        ("hmask dropped", |b| b[0..32].fill(0)),
        ("scale high bits dropped", |b| b[104..108].fill(0)), // sc[8..12] (bytes 104..108) carry every scale's top two bits
        ("scale words swapped", |b| {
            for j in 0..4 {
                b.swap(96 + j, 100 + j); // sc[j] <-> sc[j + 4]
            }
        }),
    ];
    for (name, bug) in bugs {
        let mut bad = good.clone();
        for blk in bad.chunks_mut(110) {
            bug(blk);
        }
        assert!(
            worst_excess(&dequantize_q3_k(&bad), &vals, &bound) > 0.0,
            "{name}: not caught"
        );
    }
}

// ---- The published Fuel reader ----------------------------------------------------------------------------------

/// The differential the encoders exist for: the same bytes decoded by `fuel-quantized` 0.47.32 (crates.io,
/// `dequant_ggml_bytes` -> `BlockQ{3,4,5}K::to_float`) and by the ggml port above must agree, and Fuel's output must
/// itself be within the bound of the input. Agreement of two DIFFERENT decoders on bytes written by a third piece
/// of code is what pins the layout; either decoder alone only pins itself.
#[test]
fn fuel_quantized_decodes_the_encoders_output_like_the_ggml_port() {
    use fuel_quantized::{GgmlDType, dequant_ggml_bytes};
    let vals = values(1024);
    type Case = (&'static str, GgmlDType, Vec<u8>, Vec<f32>, Vec<f32>);
    let cases: [Case; 3] = [
        (
            "Q4_K",
            GgmlDType::Q4K,
            gguf_writer::quantize_q4_k(&vals),
            dequantize(&gguf_writer::quantize_q4_k(&vals), false),
            bounds(&vals, 15.0),
        ),
        (
            "Q5_K",
            GgmlDType::Q5K,
            gguf_writer::quantize_q5_k(&vals),
            dequantize(&gguf_writer::quantize_q5_k(&vals), true),
            bounds(&vals, 31.0),
        ),
        (
            "Q3_K",
            GgmlDType::Q3K,
            gguf_writer::quantize_q3_k(&vals),
            dequantize_q3_k(&gguf_writer::quantize_q3_k(&vals)),
            q3_bounds(&vals),
        ),
    ];
    for (name, dtype, bytes, port, bound) in cases {
        let fuel = dequant_ggml_bytes(&bytes, dtype, name).expect("fuel decode");
        assert_eq!(fuel.len(), vals.len(), "{name}: length");
        let disagree = fuel
            .iter()
            .zip(&port)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            disagree < 1e-5,
            "{name}: fuel and the ggml port disagree by {disagree}"
        );
        assert!(
            worst_excess(&fuel, &vals, &bound) <= 0.0,
            "{name}: fuel's decode leaves the bound"
        );
    }
}
