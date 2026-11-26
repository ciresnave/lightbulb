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
    pub param: &'static str,
    pub message: String,
}

fn bad(param: &'static str, message: impl Into<String>) -> EmbeddingError {
    EmbeddingError { param, message: message.into() }
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
    let format = match req.encoding_format.as_deref() {
        None | Some("float") => EncodingFormat::Float,
        Some("base64") => EncodingFormat::Base64,
        Some(other) => {
            return Err(bad("encoding_format", format!("'{other}' is not supported; use 'float' or 'base64'")));
        }
    };
    let texts = match &req.input {
        EmbeddingInput::One(s) => vec![s.clone()],
        EmbeddingInput::Many(v) => v.clone(),
        EmbeddingInput::Tokens(_) => {
            return Err(bad("input", "token-id input is not supported; send a string or an array of strings"));
        }
    };
    if texts.is_empty() {
        return Err(bad("input", "input must not be an empty array"));
    }
    if texts.len() > spec.max_items {
        return Err(bad("input", format!("{} inputs exceeds the limit of {}", texts.len(), spec.max_items)));
    }
    if texts.iter().any(|t| t.is_empty()) {
        return Err(bad("input", "input must not contain an empty string"));
    }
    let dimensions = match req.dimensions {
        None => None,
        Some(d) if d == spec.dim => None,
        Some(0) => return Err(bad("dimensions", "dimensions must be positive")),
        Some(d) if spec.matryoshka.contains(&d) => Some(d),
        Some(d) => {
            let ok = if spec.matryoshka.is_empty() {
                format!("this model only produces {}", spec.dim)
            } else {
                format!("supported: {} or {:?}", spec.dim, spec.matryoshka)
            };
            return Err(bad("dimensions", format!("{d} is not supported; {ok}")));
        }
    };
    Ok(Validated { texts, format, dimensions })
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
    fn embed(&self, text: &str) -> Result<Embedded, EmbeddingError>;
}

/// L2-normalise in place. A zero vector is left as zeros (no NaN).
pub fn l2_normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

/// Post-process a pooled vector per the spec: truncate to `dimensions` (Matryoshka), then normalise.
/// Truncate BEFORE normalising, so the result has unit norm at the requested width.
pub fn finish(mut v: Vec<f32>, spec: &EmbeddingSpec, dimensions: Option<usize>) -> Vec<f32> {
    if let Some(d) = dimensions {
        v.truncate(d);
    }
    if spec.normalize {
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
        let n = (u32::from(c[0]) << 16) | (u32::from(*c.get(1).unwrap_or(&0)) << 8) | u32::from(*c.get(2).unwrap_or(&0));
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
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
        let e = embedder.embed(text)?;
        if e.tokens > spec.max_tokens {
            return Err(bad("input", format!("input {index} has {} tokens, over the limit of {}", e.tokens, spec.max_tokens)));
        }
        if e.vector.len() != spec.dim {
            return Err(bad("model", format!("model returned width {}, spec says {}", e.vector.len(), spec.dim)));
        }
        tokens += e.tokens;
        let out = finish(e.vector, spec, v.dimensions);
        let embedding = match v.format {
            EncodingFormat::Float => EmbeddingValue::Float(out),
            EncodingFormat::Base64 => EmbeddingValue::Base64(encode_base64_f32(&out)),
        };
        data.push(EmbeddingObject { object: "embedding", index, embedding });
    }
    Ok(EmbeddingResponse {
        object: "list",
        data,
        model: model.to_string(),
        usage: EmbeddingUsage { prompt_tokens: tokens, total_tokens: tokens },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;

    fn spec() -> EmbeddingSpec {
        EmbeddingSpec { pooling: Pooling::Mean, normalize: true, dim: 4, matryoshka: vec![2], max_tokens: 8, max_items: 3 }
    }
    fn req(json: serde_json::Value) -> EmbeddingRequest {
        serde_json::from_value(json).unwrap()
    }
    /// Vector = [len, 1, 2, 3]; tokens = byte length.
    struct Stub;
    impl Embedder for Stub {
        fn embed(&self, t: &str) -> Result<Embedded, EmbeddingError> {
            Ok(Embedded { vector: vec![t.len() as f32, 1.0, 2.0, 3.0], tokens: t.len() })
        }
    }

    #[test]
    fn string_and_array_inputs_validate() {
        let a = validate(&req(serde_json::json!({"model":"m","input":"hi"})), &spec()).unwrap();
        assert_eq!(a.texts, vec!["hi"]);
        let b = validate(&req(serde_json::json!({"model":"m","input":["a","b"]})), &spec()).unwrap();
        assert_eq!(b.texts.len(), 2);
    }

    #[test]
    fn refusals_name_their_param() {
        let s = spec();
        let e = |j| validate(&req(j), &s).unwrap_err();
        assert_eq!(e(serde_json::json!({"model":"m","input":[1,2,3]})).param, "input");
        assert_eq!(e(serde_json::json!({"model":"m","input":[[1,2]]})).param, "input");
        assert_eq!(e(serde_json::json!({"model":"m","input":[]})).param, "input");
        assert_eq!(e(serde_json::json!({"model":"m","input":""})).param, "input");
        assert_eq!(e(serde_json::json!({"model":"m","input":["a","b","c","d"]})).param, "input");
        assert_eq!(e(serde_json::json!({"model":"m","input":"x","encoding_format":"hex"})).param, "encoding_format");
        assert_eq!(e(serde_json::json!({"model":"m","input":"x","dimensions":3})).param, "dimensions");
        assert_eq!(e(serde_json::json!({"model":"m","input":"x","dimensions":0})).param, "dimensions");
    }

    #[test]
    fn dimensions_on_a_non_matryoshka_model_is_refused_but_native_width_is_allowed() {
        let mut s = spec();
        s.matryoshka.clear();
        assert!(validate(&req(serde_json::json!({"model":"m","input":"x","dimensions":2})), &s).is_err());
        let ok = validate(&req(serde_json::json!({"model":"m","input":"x","dimensions":4})), &s).unwrap();
        assert_eq!(ok.dimensions, None);
    }

    #[test]
    fn output_is_unit_norm_and_truncation_renormalises() {
        let s = spec();
        let v = validate(&req(serde_json::json!({"model":"m","input":"ab","dimensions":2})), &s).unwrap();
        let r = build_response("m", &v, &s, &Stub).unwrap();
        let EmbeddingValue::Float(f) = &r.data[0].embedding else { panic!() };
        assert_eq!(f.len(), 2);
        assert!(((f.iter().map(|x| x * x).sum::<f32>()) - 1.0).abs() < 1e-6);
        // Truncate-then-normalise differs from normalise-then-truncate: [2,1] -> [.894,.447].
        assert!((f[0] - 2.0 / 5f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn non_normalising_spec_returns_raw_values() {
        let mut s = spec();
        s.normalize = false;
        let v = validate(&req(serde_json::json!({"model":"m","input":"ab"})), &s).unwrap();
        let r = build_response("m", &v, &s, &Stub).unwrap();
        let EmbeddingValue::Float(f) = &r.data[0].embedding else { panic!() };
        assert_eq!(f, &vec![2.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn base64_round_trips_to_the_float_form() {
        for n in 0..7 {
            let v: Vec<f32> = (0..n).map(|i| i as f32 * 0.37 - 1.0).collect();
            let bytes = base64::engine::general_purpose::STANDARD.decode(encode_base64_f32(&v)).unwrap();
            let back: Vec<f32> = bytes.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            assert_eq!(back, v, "n={n}");
        }
    }

    #[test]
    fn usage_sums_tokens_and_order_is_kept() {
        let s = spec();
        let v = validate(&req(serde_json::json!({"model":"m","input":["abc","de"],"encoding_format":"base64"})), &s).unwrap();
        let r = build_response("m", &v, &s, &Stub).unwrap();
        assert_eq!((r.usage.prompt_tokens, r.usage.total_tokens), (5, 5));
        assert_eq!(r.data.iter().map(|d| d.index).collect::<Vec<_>>(), vec![0, 1]);
        assert!(matches!(r.data[0].embedding, EmbeddingValue::Base64(_)));
        let j = serde_json::to_value(&r).unwrap();
        assert_eq!(j["object"], "list");
        assert!(j["data"][0]["embedding"].is_string());
    }

    #[test]
    fn over_long_input_and_wrong_width_are_errors_not_truncation() {
        let s = spec();
        let v = validate(&req(serde_json::json!({"model":"m","input":"123456789"})), &s).unwrap();
        assert_eq!(build_response("m", &v, &s, &Stub).unwrap_err().param, "input");
        struct Wrong;
        impl Embedder for Wrong {
            fn embed(&self, _: &str) -> Result<Embedded, EmbeddingError> {
                Ok(Embedded { vector: vec![1.0], tokens: 1 })
            }
        }
        let v = validate(&req(serde_json::json!({"model":"m","input":"a"})), &s).unwrap();
        assert_eq!(build_response("m", &v, &s, &Wrong).unwrap_err().param, "model");
    }

    #[test]
    fn zero_vector_stays_finite() {
        let mut z = vec![0.0f32; 3];
        l2_normalize(&mut z);
        assert!(z.iter().all(|x| x.is_finite()));
    }
}
