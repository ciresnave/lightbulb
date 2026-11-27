// SPDX-License-Identifier: MIT OR Apache-2.0
//! Unit tests for the `/v1/embeddings` core (`super`).

use super::*;
use base64::Engine;

fn spec() -> EmbeddingSpec {
    EmbeddingSpec {
        pooling: Pooling::Mean,
        normalize: true,
        dim: 4,
        matryoshka: vec![2],
        matryoshka_layer_norm: false,
        max_tokens: 8,
        max_items: 3,
    }
}
fn req(json: serde_json::Value) -> EmbeddingRequest {
    serde_json::from_value(json).unwrap()
}
/// Vector = [len, 1, 2, 3]; tokens = byte length.
struct Stub;
impl Embedder for Stub {
    fn count_tokens(&self, t: &str) -> Result<usize, EmbeddingError> {
        Ok(t.len())
    }
    fn embed(&self, t: &str) -> Result<Embedded, EmbeddingError> {
        Ok(Embedded {
            vector: vec![t.len() as f32, 1.0, 2.0, 3.0],
            tokens: t.len(),
        })
    }
}

#[test]
fn string_and_array_inputs_validate() {
    let a = validate(&req(serde_json::json!({"model":"m","input":"hi"})), &spec()).unwrap();
    assert_eq!(a.texts, vec!["hi"]);
    let b = validate(
        &req(serde_json::json!({"model":"m","input":["a","b"]})),
        &spec(),
    )
    .unwrap();
    assert_eq!(b.texts.len(), 2);
}

#[test]
fn refusals_name_their_param() {
    let s = spec();
    let e = |j| validate(&req(j), &s).unwrap_err();
    assert_eq!(
        e(serde_json::json!({"model":"m","input":[1,2,3]})).param,
        "input"
    );
    assert_eq!(
        e(serde_json::json!({"model":"m","input":[[1,2]]})).param,
        "input"
    );
    assert_eq!(
        e(serde_json::json!({"model":"m","input":[]})).param,
        "input"
    );
    assert_eq!(
        e(serde_json::json!({"model":"m","input":""})).param,
        "input"
    );
    assert_eq!(
        e(serde_json::json!({"model":"m","input":["a","b","c","d"]})).param,
        "input"
    );
    assert_eq!(
        e(serde_json::json!({"model":"m","input":"x","encoding_format":"hex"})).param,
        "encoding_format"
    );
    assert_eq!(
        e(serde_json::json!({"model":"m","input":"x","dimensions":3})).param,
        "dimensions"
    );
    assert_eq!(
        e(serde_json::json!({"model":"m","input":"x","dimensions":0})).param,
        "dimensions"
    );
}

#[test]
fn dimensions_on_a_non_matryoshka_model_is_refused_but_native_width_is_allowed() {
    let mut s = spec();
    s.matryoshka.clear();
    assert!(
        validate(
            &req(serde_json::json!({"model":"m","input":"x","dimensions":2})),
            &s
        )
        .is_err()
    );
    let ok = validate(
        &req(serde_json::json!({"model":"m","input":"x","dimensions":4})),
        &s,
    )
    .unwrap();
    assert_eq!(ok.dimensions, None);
}

#[test]
fn output_is_unit_norm_and_truncation_renormalises() {
    let s = spec();
    let v = validate(
        &req(serde_json::json!({"model":"m","input":"ab","dimensions":2})),
        &s,
    )
    .unwrap();
    let r = build_response("m", &v, &s, &Stub).unwrap();
    let EmbeddingValue::Float(f) = &r.data[0].embedding else {
        panic!()
    };
    assert_eq!(f.len(), 2);
    assert!(((f.iter().map(|x| x * x).sum::<f32>()) - 1.0).abs() < 1e-6);
    // Truncate-then-normalise differs from normalise-then-truncate: [2,1] -> [.894,.447].
    let want = [2.0 / 5f32.sqrt(), 1.0 / 5f32.sqrt()];
    for (g, w) in f.iter().zip(want) {
        assert!((g - w).abs() < 1e-6, "{f:?} vs {want:?}");
    }
}

#[test]
fn non_normalising_spec_returns_raw_values() {
    let mut s = spec();
    s.normalize = false;
    let v = validate(&req(serde_json::json!({"model":"m","input":"ab"})), &s).unwrap();
    let r = build_response("m", &v, &s, &Stub).unwrap();
    let EmbeddingValue::Float(f) = &r.data[0].embedding else {
        panic!()
    };
    assert_eq!(f, &vec![2.0, 1.0, 2.0, 3.0]);
}

#[test]
fn base64_round_trips_to_the_float_form() {
    for n in 0..7 {
        let v: Vec<f32> = (0..n).map(|i| i as f32 * 0.37 - 1.0).collect();
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encode_base64_f32(&v))
            .unwrap();
        let back: Vec<f32> = bytes
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(back, v, "n={n}");
    }
}

#[test]
fn usage_sums_tokens_and_order_is_kept() {
    let s = spec();
    let v = validate(
        &req(serde_json::json!({"model":"m","input":["abc","de"],"encoding_format":"base64"})),
        &s,
    )
    .unwrap();
    let r = build_response("m", &v, &s, &Stub).unwrap();
    assert_eq!((r.usage.prompt_tokens, r.usage.total_tokens), (5, 5));
    assert_eq!(
        r.data.iter().map(|d| d.index).collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert!(matches!(r.data[0].embedding, EmbeddingValue::Base64(_)));
    let j = serde_json::to_value(&r).unwrap();
    assert_eq!(j["object"], "list");
    assert!(j["data"][0]["embedding"].is_string());
}

#[test]
fn over_long_input_and_wrong_width_are_errors_not_truncation() {
    let s = spec();
    let v = validate(
        &req(serde_json::json!({"model":"m","input":"123456789"})),
        &s,
    )
    .unwrap();
    assert_eq!(
        build_response("m", &v, &s, &Stub).unwrap_err().param,
        "input"
    );
    struct Wrong;
    impl Embedder for Wrong {
        fn count_tokens(&self, _: &str) -> Result<usize, EmbeddingError> {
            Ok(1)
        }
        fn embed(&self, _: &str) -> Result<Embedded, EmbeddingError> {
            Ok(Embedded {
                vector: vec![1.0],
                tokens: 1,
            })
        }
    }
    let v = validate(&req(serde_json::json!({"model":"m","input":"a"})), &s).unwrap();
    assert_eq!(
        build_response("m", &v, &s, &Wrong).unwrap_err().param,
        "model"
    );
}

#[test]
fn zero_vector_stays_finite() {
    let mut z = vec![0.0f32; 3];
    l2_normalize(&mut z);
    assert!(z.iter().all(|x| x.is_finite()));
}

/// Counts `embed` calls and returns whatever vector it was given.
struct Probe {
    vector: Vec<f32>,
    tokens: usize,
    calls: std::cell::Cell<usize>,
}
impl Embedder for Probe {
    fn count_tokens(&self, _: &str) -> Result<usize, EmbeddingError> {
        Ok(self.tokens)
    }
    fn embed(&self, _: &str) -> Result<Embedded, EmbeddingError> {
        self.calls.set(self.calls.get() + 1);
        Ok(Embedded {
            vector: self.vector.clone(),
            tokens: self.tokens,
        })
    }
}
fn probe(vector: Vec<f32>, tokens: usize) -> Probe {
    Probe {
        vector,
        tokens,
        calls: std::cell::Cell::new(0),
    }
}

#[test]
fn an_over_long_input_never_reaches_inference() {
    let s = spec();
    let v = validate(&req(serde_json::json!({"model":"m","input":"a"})), &s).unwrap();
    let p = probe(vec![1.0; 4], s.max_tokens + 1);
    let e = build_response("m", &v, &s, &p).unwrap_err();
    assert_eq!((e.param, e.kind), ("input", ErrorKind::Invalid));
    assert_eq!(p.calls.get(), 0, "embed ran on an input over the cap");
    // Control: at the cap it does run.
    let ok = probe(vec![1.0; 4], s.max_tokens);
    build_response("m", &v, &s, &ok).unwrap();
    assert_eq!(ok.calls.get(), 1);
}

#[test]
fn a_non_finite_vector_is_an_internal_error() {
    let s = spec();
    let v = validate(&req(serde_json::json!({"model":"m","input":"a"})), &s).unwrap();
    for bad_v in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let e = build_response("m", &v, &s, &probe(vec![1.0, bad_v, 0.0, 0.0], 1)).unwrap_err();
        assert_eq!(e.kind, ErrorKind::Internal, "{bad_v}");
    }
    // Control: the same shape with finite values passes.
    build_response("m", &v, &s, &probe(vec![1.0, 2.0, 0.0, 0.0], 1)).unwrap();
}

#[test]
fn a_wrong_width_is_the_models_fault_not_the_callers() {
    let s = spec();
    let v = validate(&req(serde_json::json!({"model":"m","input":"a"})), &s).unwrap();
    let e = build_response("m", &v, &s, &probe(vec![1.0], 1)).unwrap_err();
    assert_eq!(e.kind, ErrorKind::Internal);
}

#[test]
fn a_matryoshka_width_at_or_above_the_native_width_is_refused() {
    let mut s = spec();
    s.matryoshka = vec![2, 8];
    let r = |d| {
        validate(
            &req(serde_json::json!({"model":"m","input":"x","dimensions":d})),
            &s,
        )
    };
    assert_eq!(r(2).unwrap().dimensions, Some(2), "control: a valid width");
    assert!(
        r(8).is_err(),
        "8 > native 4 would return a shorter vector than asked"
    );
}

#[test]
fn a_reduced_width_is_renormalised_even_when_the_model_does_not_normalise() {
    let mut s = spec();
    s.normalize = false;
    let v = validate(
        &req(serde_json::json!({"model":"m","input":"ab","dimensions":2})),
        &s,
    )
    .unwrap();
    let r = build_response("m", &v, &s, &Stub).unwrap();
    let EmbeddingValue::Float(f) = &r.data[0].embedding else {
        panic!()
    };
    let want = [2.0 / 5f32.sqrt(), 1.0 / 5f32.sqrt()];
    for (g, w) in f.iter().zip(want) {
        assert!((g - w).abs() < 1e-6, "{f:?}");
    }
}

#[test]
fn l2_normalize_survives_values_whose_squares_overflow() {
    let mut big = vec![3.0e30f32, 4.0e30];
    l2_normalize(&mut big);
    assert!(
        (big[0] - 0.6).abs() < 1e-6 && (big[1] - 0.8).abs() < 1e-6,
        "{big:?}"
    );
    // Control: the naive sum of squares of the same input is infinite.
    assert!((3.0e30f32 * 3.0e30 + 4.0e30 * 4.0e30).is_infinite());
    let mut zero = vec![0.0f32; 3];
    l2_normalize(&mut zero);
    assert_eq!(zero, vec![0.0; 3]);
}

fn nomic_like() -> EmbeddingSpec {
    let mut s = spec();
    s.matryoshka_layer_norm = true;
    s
}

#[test]
fn a_reduced_width_layer_norms_the_full_vector_before_truncating() {
    // [1,2,3,4]: mean 2.5, biased variance 1.25, so the layer norm is [-1.5,-0.5,0.5,1.5] / sqrt(1.25 + 1e-5).
    // Slice to 2: [-1.5,-0.5] direction, L2 -> [-0.948683, -0.316228].
    let out = finish(vec![1.0, 2.0, 3.0, 4.0], &nomic_like(), Some(2));
    assert!((out[0] + 0.948_683).abs() < 1e-5, "{out:?}");
    assert!((out[1] + 0.316_228).abs() < 1e-5, "{out:?}");
    // Control: without the flag the same input gives the plain truncate-then-L2 vector [1,2]/sqrt5.
    let plain = finish(vec![1.0, 2.0, 3.0, 4.0], &spec(), Some(2));
    assert!((plain[0] - 0.447_214).abs() < 1e-5, "{plain:?}");
    // Order matters: layer-norming the truncated vector instead is a different vector ([-1,1] -> [-0.707, 0.707]).
    let mut wrong = vec![1.0f32, 2.0];
    layer_norm(&mut wrong);
    l2_normalize(&mut wrong);
    assert!(
        (wrong[0] - out[0]).abs() > 0.1,
        "truncate-then-layer-norm must differ"
    );
}

#[test]
fn the_native_width_never_takes_the_layer_norm() {
    let v = vec![1.0, 2.0, 3.0, 4.0];
    let native = finish(v.clone(), &nomic_like(), None);
    let want = finish(v, &spec(), None);
    assert_eq!(native, want, "the full width is mean pooling then L2 only");
}
