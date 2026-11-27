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

pub(crate) fn bad(param: &'static str, message: impl Into<String>) -> EmbeddingError {
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
        Some(d) => Err(unsupported_dimension(d, spec)),
    }
}

fn unsupported_dimension(d: usize, spec: &EmbeddingSpec) -> EmbeddingError {
    let ok = if spec.matryoshka.is_empty() {
        format!("this model only produces {}", spec.dim)
    } else {
        format!("supported: {} or {:?}", spec.dim, spec.matryoshka)
    };
    bad("dimensions", format!("{d} is not supported; {ok}"))
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

/// A loaded embedding model: its pooling-table row and the encoder that follows it. The model registry
/// holds one per embedding model; a chat entry has none.
pub struct LoadedEmbedder {
    pub spec: EmbeddingSpec,
    pub embedder: Box<dyn Embedder + Send + Sync>,
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
mod tests;
