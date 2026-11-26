// SPDX-License-Identifier: MIT OR Apache-2.0
//! `/v1/embeddings` core (ROADMAP F3, `docs/F3-EMBEDDINGS-DESIGN.md`), slice 1: request types, validation,
//! pooling table, float/base64 encoding, `dimensions` truncation and the response body. Route-agnostic on
//! purpose: an Ollama-native `/api/embed` (board 514, not decided) would wrap [`build_response`] unchanged.
//! No model is wired in yet; [`Embedder`] is the seam a Fuel-backed encoder implements in slice 2.

use serde::{Deserialize, Serialize};

/// OpenAI `input`: a string, an array of strings, or (rejected) token ids.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    One(String),
    Many(Vec<String>),
    /// Token-id forms the OpenAI spec allows. Parsed only so they can be refused with a clear 400.
    Tokens(Vec<serde_json::Value>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
    #[serde(default)]
    pub encoding_format: Option<String>,
    #[serde(default)]
    pub dimensions: Option<usize>,
    #[serde(default)]
    pub user: Option<String>,
}

/// An OpenAI-shaped validation failure: HTTP status, `param`, message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingError {
    pub kind: ErrorKind,
    pub param: &'static str,
    pub message: String,
}

/// Who is at fault, so the route can answer 400 for a bad request and 500 for an inference failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// The request is wrong (HTTP 400).
    Invalid,
    /// The model or engine failed or returned something unusable (HTTP 500).
    Internal,
}

fn bad(param: &'static str, message: impl Into<String>) -> EmbeddingError {
    EmbeddingError {
        kind: ErrorKind::Invalid,
        param,
        message: message.into(),
    }
}

/// A model-side failure: not the caller's fault.
pub(crate) fn internal(message: impl Into<String>) -> EmbeddingError {
    EmbeddingError {
        kind: ErrorKind::Internal,
        param: "model",
        message: message.into(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingFormat {
    Float,
    Base64,
}

/// How a model's token vectors become one vector. Taken from the model's own card, never defaulted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    /// Mean over non-padding tokens.
    Mean,
    /// First (`[CLS]`) token.
    Cls,
    /// Last token.
    LastToken,
}

/// The per-model row of the design note's pooling table. A model without a row is refused at load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingSpec {
    pub pooling: Pooling,
    pub normalize: bool,
    /// Native output width.
    pub dim: usize,
    /// Smaller widths the model was trained for (Matryoshka); empty = `dimensions` is refused.
    pub matryoshka: Vec<usize>,
    /// Per-request token cap and array-size cap; over either is a 400, never silent truncation.
    pub max_tokens: usize,
    pub max_items: usize,
}

/// A validated request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Validated {
    pub texts: Vec<String>,
    pub format: EncodingFormat,
    pub dimensions: Option<usize>,
}

pub fn validate(req: &EmbeddingRequest, spec: &EmbeddingSpec) -> Result<Validated, EmbeddingError> {
    let format = parse_format(req.encoding_format.as_deref())?;
    let texts = parse_texts(&req.input, spec)?;
    let dimensions = parse_dimensions(req.dimensions, spec)?;
    Ok(Validated {
        texts,
        format,
        dimensions,
    })
}

fn parse_format(f: Option<&str>) -> Result<EncodingFormat, EmbeddingError> {
    match f {
        None | Some("float") => Ok(EncodingFormat::Float),
        Some("base64") => Ok(EncodingFormat::Base64),
        Some(other) => Err(bad(
            "encoding_format",
            format!("'{other}' is not supported; use 'float' or 'base64'"),
        )),
    }
}

fn parse_texts(
    input: &EmbeddingInput,
    spec: &EmbeddingSpec,
) -> Result<Vec<String>, EmbeddingError> {
    let texts = match input {
        EmbeddingInput::One(s) => vec![s.clone()],
        EmbeddingInput::Many(v) => v.clone(),
        EmbeddingInput::Tokens(_) => {
            return Err(bad(
                "input",
                "token-id input is not supported; send a string or an array of strings",
            ));
        }
    };
    if texts.is_empty() {
        return Err(bad("input", "input must not be an empty array"));
    }
    if texts.len() > spec.max_items {
        return Err(bad(
            "input",
            format!(
                "{} inputs exceeds the limit of {}",
                texts.len(),
                spec.max_items
            ),
        ));
    }
    if texts.iter().any(|t| t.is_empty()) {
        return Err(bad("input", "input must not contain an empty string"));
    }
    Ok(texts)
}

fn parse_dimensions(
    requested: Option<usize>,
    spec: &EmbeddingSpec,
) -> Result<Option<usize>, EmbeddingError> {
    match requested {
        None => Ok(None),
        Some(d) if d == spec.dim => Ok(None),
        Some(0) => Err(bad("dimensions", "dimensions must be positive")),
        Some(d) if d < spec.dim && spec.matryoshka.contains(&d) => Ok(Some(d)),
        Some(d) => {
            let ok = if spec.matryoshka.is_empty() {
                format!("this model only produces {}", spec.dim)
            } else {
                format!("supported: {} or {:?}", spec.dim, spec.matryoshka)
            };
            Err(bad("dimensions", format!("{d} is not supported; {ok}")))
        }
    }
}

/// What a model-backed encoder returns for one text.
#[derive(Debug, Clone, PartialEq)]
pub struct Embedded {
    /// Pooled vector of the model's native width, NOT yet normalised.
    pub vector: Vec<f32>,
    /// Real tokenizer tokens including special tokens.
    pub tokens: usize,
}

/// The seam slice 2 fills with a Fuel encoder: one sequence per call (Fuel has no padded batch).
pub trait Embedder {
    /// Real tokenizer tokens including special tokens, WITHOUT running the model, so an over-long input
    /// is refused before any inference is spent on it.
    fn count_tokens(&self, text: &str) -> Result<usize, EmbeddingError>;
    fn embed(&self, text: &str) -> Result<Embedded, EmbeddingError>;
}

/// L2-normalise in place. A zero vector is left as zeros (no NaN).
pub fn l2_normalize(v: &mut [f32]) {
    // Scale by the largest magnitude first so squaring cannot overflow f32 for large finite values.
    let m = v.iter().fold(0.0f32, |a, x| a.max(x.abs()));
    if m == 0.0 || !m.is_finite() {
        return;
    }
    let n = m * v.iter().map(|x| (x / m) * (x / m)).sum::<f32>().sqrt();
    v.iter_mut().for_each(|x| *x /= n);
}

/// Post-process a pooled vector per the spec: truncate to `dimensions` (Matryoshka), then normalise.
/// Truncate BEFORE normalising, so the result has unit norm at the requested width.
pub fn finish(mut v: Vec<f32>, spec: &EmbeddingSpec, dimensions: Option<usize>) -> Vec<f32> {
    if let Some(d) = dimensions {
        v.truncate(d);
    }
    // A reduced width is always re-normalised: a truncated vector has the wrong magnitude whatever the
    // model's native setting.
    if spec.normalize || dimensions.is_some() {
        l2_normalize(&mut v);
    }
    v
}

/// Standard base64 of the little-endian f32 bytes (OpenAI's `encoding_format: "base64"`).
pub fn encode_base64_f32(v: &[f32]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (u32::from(c[0]) << 16)
            | (u32::from(*c.get(1).unwrap_or(&0)) << 8)
            | u32::from(*c.get(2).unwrap_or(&0));
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 {
            T[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if c.len() > 2 {
            T[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum EmbeddingValue {
    Float(Vec<f32>),
    Base64(String),
}

#[derive(Debug, Serialize)]
pub struct EmbeddingObject {
    pub object: &'static str,
    pub index: usize,
    pub embedding: EmbeddingValue,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

fn over_limit(index: usize, tokens: usize, spec: &EmbeddingSpec) -> EmbeddingError {
    bad(
        "input",
        format!(
            "input {index} has {tokens} tokens, over the limit of {}",
            spec.max_tokens
        ),
    )
}

/// What a model's output must meet: under the token cap, at the native width, and finite.
fn check_embedded(index: usize, e: &Embedded, spec: &EmbeddingSpec) -> Result<(), EmbeddingError> {
    if e.tokens > spec.max_tokens {
        return Err(over_limit(index, e.tokens, spec));
    }
    if e.vector.len() != spec.dim {
        return Err(internal(format!(
            "model returned width {}, spec says {}",
            e.vector.len(),
            spec.dim
        )));
    }
    if let Some(at) = e.vector.iter().position(|x| !x.is_finite()) {
        return Err(internal(format!(
            "model returned a non-finite value at component {at} of input {index}"
        )));
    }
    Ok(())
}

/// Run a validated request through an embedder, in order, one sequence at a time.
pub fn build_response(
    model: &str,
    v: &Validated,
    spec: &EmbeddingSpec,
    embedder: &dyn Embedder,
) -> Result<EmbeddingResponse, EmbeddingError> {
    let mut data = Vec::with_capacity(v.texts.len());
    let mut tokens = 0usize;
    for (index, text) in v.texts.iter().enumerate() {
        let counted = embedder.count_tokens(text)?;
        if counted > spec.max_tokens {
            return Err(over_limit(index, counted, spec));
        }
        let e = embedder.embed(text)?;
        check_embedded(index, &e, spec)?;
        tokens += e.tokens;
        let out = finish(e.vector, spec, v.dimensions);
        let embedding = match v.format {
            EncodingFormat::Float => EmbeddingValue::Float(out),
            EncodingFormat::Base64 => EmbeddingValue::Base64(encode_base64_f32(&out)),
        };
        data.push(EmbeddingObject {
            object: "embedding",
            index,
            embedding,
        });
    }
    Ok(EmbeddingResponse {
        object: "list",
        data,
        model: model.to_string(),
        usage: EmbeddingUsage {
            prompt_tokens: tokens,
            total_tokens: tokens,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;

    fn spec() -> EmbeddingSpec {
        EmbeddingSpec {
            pooling: Pooling::Mean,
            normalize: true,
            dim: 4,
            matryoshka: vec![2],
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
}
