// SPDX-License-Identifier: MIT OR Apache-2.0
//! OpenAI `logprobs` on chat and completions: request validation and the two response shapes.
//!
//! The values are Fuel's per-token logprobs under the model's own distribution (before temperature, filters and
//! penalties), one per generated token. A token's text is decoded on its own, so a token that is part of a
//! multi-byte character shows the bytes the decoder could make of it.

use crate::engine::model_runner::TokenLogprob;
use serde_json::{Value, json};

/// Largest number of alternatives per token this server accepts (OpenAI's chat limit).
pub(crate) const MAX_TOP_LOGPROBS: u32 = 20;

/// The alternatives count a chat request asks Fuel to record, or `None` when it did not ask for logprobs.
/// `top_logprobs` without `logprobs: true` is a 400, as at OpenAI.
pub(crate) fn validate_chat(
    logprobs: Option<bool>,
    top_logprobs: Option<u32>,
) -> Result<Option<usize>, String> {
    if let Some(n) = top_logprobs {
        if logprobs != Some(true) {
            return Err("`top_logprobs` requires `logprobs: true`".to_string());
        }
        if n > MAX_TOP_LOGPROBS {
            return Err(format!(
                "`top_logprobs`={n} is out of range: it must be in [0, {MAX_TOP_LOGPROBS}]"
            ));
        }
    }
    Ok((logprobs == Some(true)).then(|| top_logprobs.unwrap_or(0) as usize))
}

/// The alternatives count a completions request asks Fuel to record (`logprobs: N` is the integer form there).
pub(crate) fn validate_completions(logprobs: Option<u32>) -> Result<Option<usize>, String> {
    match logprobs {
        Some(n) if n > MAX_TOP_LOGPROBS => Err(format!(
            "`logprobs`={n} is out of range: it must be in [0, {MAX_TOP_LOGPROBS}]"
        )),
        other => Ok(other.map(|n| n as usize)),
    }
}

/// The entries that belong to `text`: stop-sequence truncation shortens the text but not what was generated, so
/// logprobs for tokens wholly past the cut are dropped.
pub(crate) fn within_text<'a>(logprobs: &'a [TokenLogprob], text: &str) -> &'a [TokenLogprob] {
    let mut consumed = 0usize;
    let mut keep = 0usize;
    for lp in logprobs {
        if consumed >= text.len() {
            break;
        }
        consumed += lp.token.len();
        keep += 1;
    }
    &logprobs[..keep]
}

fn token_entry(token: &str, logprob: f32) -> Value {
    json!({"token": token, "logprob": logprob, "bytes": token.as_bytes()})
}

/// Chat shape: `{"content": [{token, logprob, bytes, top_logprobs: [...]}]}`.
pub(crate) fn chat_json(logprobs: &[TokenLogprob]) -> Value {
    let content: Vec<Value> = logprobs
        .iter()
        .map(|lp| {
            let mut e = token_entry(&lp.token, lp.logprob);
            e["top_logprobs"] =
                Value::Array(lp.top.iter().map(|(t, p)| token_entry(t, *p)).collect());
            e
        })
        .collect();
    json!({ "content": content })
}

/// Completions shape: parallel `tokens`, `token_logprobs`, `top_logprobs` (token -> logprob maps) and
/// `text_offset` (byte offset of each token in the generated text).
pub(crate) fn completions_json(logprobs: &[TokenLogprob]) -> Value {
    let mut offset = 0usize;
    let mut offsets = Vec::with_capacity(logprobs.len());
    for lp in logprobs {
        offsets.push(offset);
        offset += lp.token.len();
    }
    json!({
        "tokens": logprobs.iter().map(|l| l.token.as_str()).collect::<Vec<_>>(),
        "token_logprobs": logprobs.iter().map(|l| l.logprob).collect::<Vec<_>>(),
        "top_logprobs": logprobs.iter().map(|l| {
            l.top.iter().map(|(t, p)| (t.clone(), json!(p))).collect::<serde_json::Map<_, _>>()
        }).collect::<Vec<_>>(),
        "text_offset": offsets,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lp(token: &str, logprob: f32, top: &[(&str, f32)]) -> TokenLogprob {
        TokenLogprob {
            token: token.to_string(),
            logprob,
            top: top.iter().map(|(t, p)| (t.to_string(), *p)).collect(),
        }
    }

    #[test]
    fn chat_validation_follows_openai() {
        assert_eq!(validate_chat(None, None), Ok(None));
        assert_eq!(validate_chat(Some(false), None), Ok(None));
        assert_eq!(validate_chat(Some(true), None), Ok(Some(0)));
        assert_eq!(validate_chat(Some(true), Some(5)), Ok(Some(5)));
        assert!(validate_chat(None, Some(2)).is_err());
        assert!(validate_chat(Some(false), Some(2)).is_err());
        assert!(validate_chat(Some(true), Some(21)).is_err());
        assert_eq!(validate_chat(Some(true), Some(20)), Ok(Some(20)));
    }

    #[test]
    fn completions_validation_is_the_integer_form() {
        assert_eq!(validate_completions(None), Ok(None));
        assert_eq!(validate_completions(Some(0)), Ok(Some(0)));
        assert_eq!(validate_completions(Some(20)), Ok(Some(20)));
        assert!(validate_completions(Some(21)).is_err());
    }

    #[test]
    fn within_text_drops_tokens_past_the_cut() {
        let all = [
            lp("Hel", -0.1, &[]),
            lp("lo", -0.2, &[]),
            lp(" there", -0.3, &[]),
        ];
        assert_eq!(within_text(&all, "Hello there").len(), 3);
        assert_eq!(within_text(&all, "Hello").len(), 2);
        assert_eq!(within_text(&all, "He").len(), 1);
        assert_eq!(within_text(&all, "").len(), 0);
    }

    #[test]
    fn chat_shape_carries_token_bytes_and_alternatives() {
        let v = chat_json(&[lp("Hi", -0.5, &[("Hi", -0.5), ("Hey", -1.5)])]);
        let e = &v["content"][0];
        assert_eq!(e["token"], "Hi");
        assert_eq!(e["bytes"], json!([72, 105]));
        assert_eq!(e["top_logprobs"][1]["token"], "Hey");
        assert_eq!(e["top_logprobs"][1]["logprob"], json!(-1.5));
        // No alternatives requested: an empty list, never absent.
        let v = chat_json(&[lp("Hi", -0.5, &[])]);
        assert_eq!(v["content"][0]["top_logprobs"], json!([]));
    }

    #[test]
    fn completions_shape_is_parallel_arrays_with_offsets() {
        let v = completions_json(&[lp("ab", -0.1, &[("ab", -0.1)]), lp("c", -0.2, &[])]);
        assert_eq!(v["tokens"], json!(["ab", "c"]));
        assert_eq!(v["text_offset"], json!([0, 2]));
        assert_eq!(v["token_logprobs"], json!([-0.1f32, -0.2f32]));
        assert_eq!(v["top_logprobs"][0]["ab"], json!(-0.1f32));
        assert_eq!(v["top_logprobs"][1], json!({}));
    }
}
