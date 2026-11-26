// SPDX-License-Identifier: MIT OR Apache-2.0
//! The GGUF serving path end to end on a SYNTHETIC GGUF (ROADMAP C1): header and metadata parse, tokenizer
//! rebuild, weight load into Fuel, the runner, generation, and the OpenAI response, with no downloaded model.
//!
//! The file is `support::gguf::constant_output_gguf_bytes`: tiny, F32, and its greedy output is one fixed token,
//! so the assertions are exact (a trained model's text is not). It proves the PATH works and reports honestly
//! (`finish_reason`, `usage`); it says nothing about a real checkpoint's quality, and nothing about quantised
//! dtypes (the tensors are F32).

use std::path::Path;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use lightbulb::api::{ApiConfig, AppState};
use lightbulb::engine::{MemoryAwareConfig, MemoryAwareScheduler, ModelRunner};

#[path = "support/gguf.rs"]
#[allow(dead_code)]
mod gguf_writer;

fn app_state_for(path: &Path) -> AppState {
    let scheduler = Arc::new(std::sync::Mutex::new(MemoryAwareScheduler::new(
        MemoryAwareConfig::default(),
    )));
    let (tx, ctrl_tx) = ModelRunner::start(
        path,
        1,
        512,
        Some("f32".to_string()),
        8,
        Arc::new(std::sync::Mutex::new(
            lightbulb::cache::PrefixCacheStats::default(),
        )),
        scheduler.clone(),
    )
    .expect("starting the model runner on the synthetic GGUF");
    AppState {
        scheduler,
        config: ApiConfig::default(),
        pg_backend: None,
        manager: None,
        models: lightbulb::api::ModelRegistry::single(
            "test-model",
            lightbulb::api::ModelEntry {
                inference_tx: Some(tx),
                cache_control: Some(ctrl_tx),
                chat_template: lightbulb::api::chat_template::resolve_for_serving(path),
                ..Default::default()
            },
        ),
        eos_monitor: Default::default(),
        security_sinks: Vec::new(),
        cache_stats: Arc::new(std::sync::Mutex::new(
            lightbulb::cache::PrefixCacheStats::default(),
        )),
        live_system: lightbulb::api::new_shared_system_metrics(),
        queue_depth: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
    }
}

/// POST a chat request for a synthetic GGUF whose output token is `token`; return the parsed body.
async fn chat(token: u32, max_tokens: u32) -> serde_json::Value {
    let body = chat_body(max_tokens, serde_json::json!({}));
    let bytes = post_chat(token, body).await;
    serde_json::from_slice(&bytes).expect("response was not JSON")
}

fn chat_body(max_tokens: u32, extra: serde_json::Value) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": "lightbulb-default",
        "messages": [{"role": "user", "content": "ab cd"}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    });
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    body
}

/// The response body (200 asserted) of `body` posted to a server over the synthetic GGUF whose output is `token`.
async fn post_chat(token: u32, body: serde_json::Value) -> Vec<u8> {
    post_chat_gguf(gguf_writer::constant_output_gguf_bytes(token), body).await
}

/// [`post_chat`] over the given GGUF file contents (200 asserted).
async fn post_chat_gguf(gguf: Vec<u8>, body: serde_json::Value) -> Vec<u8> {
    let (status, bytes) = post_chat_gguf_status(gguf, body).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    bytes
}

/// The status and body of `body` posted to a server over the given GGUF file contents.
async fn post_chat_gguf_status(gguf: Vec<u8>, body: serde_json::Value) -> (StatusCode, Vec<u8>) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("synthetic.gguf");
    std::fs::write(&path, gguf).expect("writing the gguf");
    let app = lightbulb::api::openai::routes().with_state(app_state_for(&path));
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .expect("router returned no response");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    (status, bytes.to_vec())
}

/// A GGUF whose model never emits the EOS runs to its cap and is served as such: `finish_reason: "length"`,
/// `completion_tokens` exactly the cap, and the content is the favoured token's text, six times (token 7 is
/// `&`), so the tokenizer rebuilt from the file's own vocabulary decoded what the weights produced.
#[tokio::test]
async fn a_synthetic_gguf_is_served_to_its_cap() {
    let v = chat(7, 6).await;
    assert_eq!(v["choices"][0]["finish_reason"], "length", "{v}");
    assert_eq!(v["usage"]["completion_tokens"], 6, "{v}");
    assert_eq!(v["choices"][0]["message"]["content"], "&&&&&&", "{v}");
}

/// The stop arm: the same file with the EOS as its output token stops on the first token.
#[tokio::test]
async fn a_synthetic_gguf_whose_model_emits_eos_is_served_as_stopped() {
    let v = chat(1, 32).await;
    assert_eq!(v["choices"][0]["finish_reason"], "stop", "{v}");
    let completion = v["usage"]["completion_tokens"]
        .as_u64()
        .expect("completion_tokens");
    assert!(
        (1..32).contains(&completion),
        "stopped early, not at the cap: {v}"
    );
}

/// Logprobs end to end through real Fuel on the synthetic GGUF, non-streaming: one entry per generated token, the
/// token text decoded from the file's own vocabulary, and, because the model's output row for `&` dominates
/// (logit 16 against at most 0.8), a logprob within a hair of 0 whose top alternative is that token.
#[tokio::test]
async fn a_synthetic_gguf_returns_logprobs() {
    let body = chat_body(4, serde_json::json!({"logprobs": true, "top_logprobs": 2}));
    let v: serde_json::Value =
        serde_json::from_slice(&post_chat(7, body).await).expect("response was not JSON");
    let content = v["choices"][0]["logprobs"]["content"]
        .as_array()
        .expect("logprobs.content");
    assert_eq!(content.len(), 4, "one entry per generated token: {v}");
    for entry in content {
        assert_eq!(entry["token"], "&", "{v}");
        let logprob = entry["logprob"].as_f64().expect("logprob");
        assert!(
            (-1e-3..=0.0).contains(&logprob),
            "a dominant token has logprob ~0: {v}"
        );
        assert_eq!(
            entry["top_logprobs"]
                .as_array()
                .expect("top_logprobs")
                .len(),
            2,
            "{v}"
        );
        assert_eq!(entry["top_logprobs"][0]["token"], "&", "{v}");
    }
}

/// The same file with every matrix quantised to Q8_0 (norms F32): the quantised load and matmul path serves it,
/// and the output is the same fixed token with the same logprob shape (a dominant token, logprob near 0), so the
/// dequantised weights are close enough that the decision does not move. Fails if the file does not load, or the
/// quantised path decodes anything but `&`.
#[tokio::test]
async fn a_q8_0_synthetic_gguf_is_served_and_decodes_the_same_token() {
    let body = chat_body(4, serde_json::json!({"logprobs": true, "top_logprobs": 1}));
    let bytes = post_chat_gguf(gguf_writer::constant_output_gguf_q8_0_bytes(7), body).await;
    let v: serde_json::Value = serde_json::from_slice(&bytes).expect("response was not JSON");
    assert_eq!(v["choices"][0]["finish_reason"], "length", "{v}");
    assert_eq!(v["choices"][0]["message"]["content"], "&&&&", "{v}");
    let first = &v["choices"][0]["logprobs"]["content"][0];
    assert!(first["logprob"].as_f64().expect("logprob") > -1e-2, "{v}");
}

/// Q4_0, the format of this repo's own real TinyLlama file: same file, 4-bit matrices, still the same fixed token
/// (the dominant row survives 4-bit quantisation; the logprob is looser than Q8_0's, so the bound is too).
#[tokio::test]
async fn a_q4_0_synthetic_gguf_is_served_and_decodes_the_same_token() {
    let body = chat_body(4, serde_json::json!({"logprobs": true, "top_logprobs": 1}));
    let bytes = post_chat_gguf(gguf_writer::constant_output_gguf_q4_0_bytes(7), body).await;
    let v: serde_json::Value = serde_json::from_slice(&bytes).expect("response was not JSON");
    assert_eq!(v["choices"][0]["finish_reason"], "length", "{v}");
    assert_eq!(v["choices"][0]["message"]["content"], "&&&&", "{v}");
    let first = &v["choices"][0]["logprobs"]["content"][0];
    assert!(first["logprob"].as_f64().expect("logprob") > -5e-2, "{v}");
}

/// Q6_K, a K-quant (256-value super-blocks, 6-bit values, 8-bit group scales): the loader's K-quant path serves it
/// and the fixed token survives. The bound is the loosest of the three: the group scales are 8-bit and the weights
/// are 6-bit.
#[tokio::test]
async fn a_q6_k_synthetic_gguf_is_served_and_decodes_the_same_token() {
    let body = chat_body(4, serde_json::json!({"logprobs": true, "top_logprobs": 1}));
    let bytes = post_chat_gguf(gguf_writer::constant_output_gguf_q6_k_bytes(7), body).await;
    let v: serde_json::Value = serde_json::from_slice(&bytes).expect("response was not JSON");
    assert_eq!(v["choices"][0]["finish_reason"], "length", "{v}");
    assert_eq!(v["choices"][0]["message"]["content"], "&&&&", "{v}");
    let first = &v["choices"][0]["logprobs"]["content"][0];
    assert!(first["logprob"].as_f64().expect("logprob") > -5e-2, "{v}");
}

/// The Q6_K encoder against an independent reading of llama.cpp's `dequantize_row_q6_K`: varied values (not the
/// fixture's constants, which any layout slip would still decode) come back within the quantisation step. A wrong
/// nibble, high-bit or scale placement moves values by far more than the bound.
#[test]
fn the_q6_k_encoder_round_trips_through_the_reference_layout() {
    let values: Vec<f32> = (0..512)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
        .collect();
    let bytes = gguf_writer::quantize_q6_k(&values);
    assert_eq!(bytes.len(), 2 * 210);
    for (b, block) in bytes.chunks(210).zip(values.chunks(256)) {
        let (ql, rest) = b.split_at(128);
        let (qh, rest) = rest.split_at(64);
        let (sc, d) = rest.split_at(16);
        let d = half_to_f32(u16::from_le_bytes([d[0], d[1]]));
        for half in 0..2 {
            for l in 0..32 {
                let (lo, hi) = (ql[half * 64 + l], ql[half * 64 + l + 32]);
                let h = qh[half * 32 + l];
                let q = [
                    (lo & 0xF) | ((h & 3) << 4),
                    (hi & 0xF) | (((h >> 2) & 3) << 4),
                    (lo >> 4) | (((h >> 4) & 3) << 4),
                    (hi >> 4) | (((h >> 6) & 3) << 4),
                ];
                for (k, q) in q.iter().enumerate() {
                    let scale = f32::from(sc[half * 8 + 2 * k + l / 16] as i8);
                    let got = d * scale * (f32::from(*q) - 32.0);
                    let want = block[half * 128 + k * 32 + l];
                    assert!(
                        (got - want).abs() < 0.05,
                        "{got} vs {want} (half {half} k {k} l {l})"
                    );
                }
            }
        }
    }
}

/// An f16 (normal range) to f32.
fn half_to_f32(bits: u16) -> f32 {
    let exp = i32::from((bits >> 10) & 0x1f);
    let mant = f32::from(bits & 0x3ff);
    let magnitude = (1.0 + mant / 1024.0) * 2f32.powi(exp - 15);
    if bits & 0x8000 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// The quantisers are not no-ops: the Q8_0 file is under half the F32 one (34 bytes per 32 values against 128 for
/// the matrices) and the Q4_0 file smaller still (18), so the tests above ran different files, not the F32 one under
/// other names.
#[test]
fn the_quantised_files_are_smaller_than_the_f32_one() {
    let f32_len = gguf_writer::constant_output_gguf_bytes(7).len();
    let q8_len = gguf_writer::constant_output_gguf_q8_0_bytes(7).len();
    let q4_len = gguf_writer::constant_output_gguf_q4_0_bytes(7).len();
    assert!(q8_len * 2 < f32_len, "q8_0 {q8_len} vs f32 {f32_len} bytes");
    assert!(q4_len < q8_len, "q4_0 {q4_len} vs q8_0 {q8_len} bytes");
    let q6_len = gguf_writer::constant_output_gguf_q6_k_bytes(7).len();
    assert!(q6_len < q8_len, "q6_k {q6_len} vs q8_0 {q8_len} bytes");
    assert!(q4_len < q6_len, "q4_0 {q4_len} vs q6_k {q6_len} bytes");
}

/// A chat template within the synthetic vocabulary (printable ASCII and spaces): not the legacy `role: content`
/// join, so a prompt built from it has a different token count.
const TEMPLATE: &str = "{% for m in messages %}<|{{ m.role }}|> {{ m.content }} {% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}";

/// The GGUF's own chat template reaches the prompt on BOTH the non-streaming and the streaming path
/// (`chat.rs` has two prompt-building sites; fixing one leaves the other green-looking and wrong). Observed
/// through `usage.prompt_tokens`: it must equal the token count of the template rendered over the messages and
/// encoded with the tokenizer rebuilt from the same file, and (the control) differ from the legacy
/// `user: <content>` join's count, so a handler that ignored the template would fail.
#[tokio::test]
async fn the_ggufs_chat_template_reaches_the_prompt_on_both_paths() {
    let gguf = gguf_writer::constant_output_gguf_with_template_bytes(7, TEMPLATE);
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("synthetic.gguf");
    std::fs::write(&path, &gguf).expect("writing the gguf");

    let tokenizer = lightbulb::gguf::Content::read(&path)
        .expect("reading the gguf")
        .extract_tokenizer()
        .expect("rebuilding the tokenizer from the file");
    let count = |text: &str| tokenizer.encode(text, false).expect("encoding").len() as u64;
    let resolved = lightbulb::api::chat_template::resolve_for_serving(&path)
        .expect("the file's own template resolves");
    let rendered = resolved
        .render(&[lightbulb::contracts::validation::RawMessage {
            role: "user".into(),
            content: "ab cd".into(),
        }])
        .expect("rendering");
    let expected = count(&rendered);
    let legacy = count("user: ab cd");
    assert_ne!(
        expected, legacy,
        "the control must tell the template from the join: {rendered:?}"
    );

    let plain = post_chat_gguf(gguf.clone(), chat_body(2, serde_json::json!({}))).await;
    let plain: serde_json::Value = serde_json::from_slice(&plain).expect("response was not JSON");
    assert_eq!(
        plain["usage"]["prompt_tokens"], expected,
        "non-streaming: {plain}"
    );

    let streamed = post_chat_gguf(
        gguf,
        chat_body(
            2,
            serde_json::json!({"stream": true, "stream_options": {"include_usage": true}}),
        ),
    )
    .await;
    let text = String::from_utf8(streamed).expect("SSE body is UTF-8");
    let usage = text
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .filter_map(|d| serde_json::from_str::<serde_json::Value>(d).ok())
        .find_map(|c| c.get("usage").filter(|u| !u.is_null()).cloned())
        .expect("a usage chunk");
    assert_eq!(usage["prompt_tokens"], expected, "streaming: {text}");
}

/// ROADMAP B12, re-verified on Fuel 0.47.32: a Q8_1 matrix is REFUSED at model load with an error that names the
/// dtype, not served as garbage. Pins the typed refusal (`GAP-125`: `BlockQ8_1::to_float` has no body upstream); if
/// Fuel gains Q8_1 dequantisation this fails and the row should be closed, not the test silenced.
#[tokio::test]
async fn a_q8_1_gguf_is_refused_by_name_not_served() {
    let body = chat_body(2, serde_json::json!({}));
    let (status, bytes) =
        post_chat_gguf_status(gguf_writer::constant_output_gguf_q8_1_bytes(7), body).await;
    let text = String::from_utf8_lossy(&bytes);
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{text}");
    assert!(
        text.contains("Q8_1") && text.contains("GAP-125"),
        "the refusal must name the dtype and the upstream gap: {text}"
    );
}

/// ROADMAP H8: a model that produces a NaN logit must fail the request, not answer with a token picked from the
/// row. Fuel 0.57.0 made `sample_logits` (the path a default-sampling request takes: no top_p, penalties or
/// logprobs) a typed `Err` on a NaN row; before it the greedy argmax and the temperature draw both returned a token.
/// Both default paths are exercised, each against the same request on a NaN-free file as the control.
#[tokio::test]
async fn a_nan_logit_row_fails_a_default_sampling_request_instead_of_picking_a_token() {
    for temperature in [0.0, 0.7] {
        let body = chat_body(3, serde_json::json!({ "temperature": temperature }));

        // Control: the same request over the NaN-free file is served.
        let ok = post_chat_gguf(gguf_writer::constant_output_gguf_bytes(7), body.clone()).await;
        let ok: serde_json::Value = serde_json::from_slice(&ok).unwrap();
        assert_eq!(
            ok["usage"]["completion_tokens"], 3,
            "temperature {temperature}"
        );

        let (status, bytes) =
            post_chat_gguf_status(gguf_writer::nan_logit_gguf_bytes(7), body).await;
        let text = String::from_utf8_lossy(&bytes);
        assert_eq!(
            status,
            StatusCode::INTERNAL_SERVER_ERROR,
            "temperature {temperature}: {text}"
        );
        assert!(
            text.to_lowercase().contains("nan") || text.contains("not finite"),
            "the error must say the logits were NaN or not finite (temperature {temperature}): {text}"
        );
    }
}

/// The same request streamed: each token chunk carries its own entry, from the real driver through the runner and
/// the SSE handler (not a stub on any hop).
#[tokio::test]
async fn a_synthetic_gguf_streams_logprobs_on_each_token_chunk() {
    let body = chat_body(
        4,
        serde_json::json!({"logprobs": true, "top_logprobs": 1, "stream": true}),
    );
    let text = String::from_utf8(post_chat(7, body).await).expect("SSE body is UTF-8");
    let frames: Vec<serde_json::Value> = text
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).expect("each SSE frame carries JSON"))
        .collect();
    let with_entry: Vec<&serde_json::Value> = frames
        .iter()
        .filter(|f| f["choices"][0].get("logprobs").is_some())
        .collect();
    assert_eq!(
        with_entry.len(),
        4,
        "one chunk per generated token carries an entry: {text}"
    );
    for f in with_entry {
        let entry = &f["choices"][0]["logprobs"]["content"][0];
        assert_eq!(entry["token"], "&", "{f}");
        assert_eq!(f["choices"][0]["delta"]["content"], "&", "{f}");
        assert!(entry["logprob"].as_f64().expect("logprob") > -1e-3, "{f}");
    }
}

/// EOS TIMING: a model that emits `&'(` and then the EOS stops after exactly three tokens (`finish_reason: "stop"`,
/// `completion_tokens: 4` (the EOS counts), the text is the chain), whatever the cap above that; a cap of two cuts it to `&'` with
/// `"length"`. The constant-output model can only stop at once or never; here the stop arrives mid-generation, so a
/// stop that is dropped, early or late changes the text or the reason.
#[tokio::test]
async fn a_chain_model_stops_exactly_when_its_chain_ends() {
    let gguf = gguf_writer::chain_output_gguf_bytes(&[7, 8, 9]);
    let v: serde_json::Value = serde_json::from_slice(
        &post_chat_gguf(gguf.clone(), chat_body(10, serde_json::json!({}))).await,
    )
    .expect("response was not JSON");
    assert_eq!(v["choices"][0]["message"]["content"], "&'(", "{v}");
    assert_eq!(v["choices"][0]["finish_reason"], "stop", "{v}");
    // The EOS is generated, so it is counted: three chain tokens and the stop.
    assert_eq!(v["usage"]["completion_tokens"], 4, "{v}");

    let cut: serde_json::Value =
        serde_json::from_slice(&post_chat_gguf(gguf, chat_body(2, serde_json::json!({}))).await)
            .expect("response was not JSON");
    assert_eq!(cut["choices"][0]["message"]["content"], "&'", "{cut}");
    assert_eq!(cut["choices"][0]["finish_reason"], "length", "{cut}");
}

/// The synthetic counterpart of `chat_template_e2e`'s two stop tests: a GGUF WITH a chat template, whose model ends
/// its answer with the EOS, stops on it on both the non-streaming and the streaming path: `finish_reason: "stop"`,
/// the text exactly the chain (nothing after the stop, no EOS text in it), one terminal chunk on the stream.
#[tokio::test]
async fn a_templated_chat_stops_on_the_eos_on_both_paths() {
    let gguf = gguf_writer::chain_output_gguf_with_template_bytes(&[7, 8, 9], TEMPLATE);
    let plain: serde_json::Value = serde_json::from_slice(
        &post_chat_gguf(gguf.clone(), chat_body(10, serde_json::json!({}))).await,
    )
    .expect("response was not JSON");
    assert_eq!(plain["choices"][0]["message"]["content"], "&'(", "{plain}");
    assert_eq!(plain["choices"][0]["finish_reason"], "stop", "{plain}");

    let streamed = post_chat_gguf(gguf, chat_body(10, serde_json::json!({"stream": true}))).await;
    let text = String::from_utf8(streamed).expect("SSE body is UTF-8");
    let chunks: Vec<serde_json::Value> = text
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .filter_map(|d| serde_json::from_str(d).ok())
        .collect();
    let content: String = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(content, "&'(", "{text}");
    let reasons: Vec<&serde_json::Value> = chunks
        .iter()
        .map(|c| &c["choices"][0]["finish_reason"])
        .filter(|r| !r.is_null())
        .collect();
    assert_eq!(reasons, ["stop"], "{text}");
}
