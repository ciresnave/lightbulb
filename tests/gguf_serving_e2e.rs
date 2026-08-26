//! Serving a real GGUF end to end.
//!
//! Known-red baseline, measured 2026-08-14 BEFORE this work, on
//! `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` @ `52e7645b`, Q4_0, over HTTP at
//! `temperature: 0.0`:
//!
//! ```text
//! content: "| ass istant | <0x0A> | ass istant | ass istant | | ass istant |"
//! Chat template … resolved via Registry (bos "", eos "")
//! ```
//!
//! The file declares `tokenizer.chat_template` and ids 1/2 → `<s>`/`</s>`;
//! resolution simply never looked inside it. Re-running this test against a
//! commit before that fix must reproduce the garbage above — that is what makes
//! this gate a gate rather than a description.
//!
//! # What actually happened AFTER this work (measured 2026-08-19)
//!
//! Resolution is fixed, and verified two independent ways below: through
//! direct assertions on `special_tokens`/`resolve`, and by rendering this
//! checkpoint's own template against the exact fixed message this file sends
//! and asserting the resulting string byte-for-byte. Both confirm the epic's
//! claim for this checkpoint: BOS `"<s>"`, EOS `"</s>"`,
//! `Resolution::GgufMetadata`, and a correctly rendered prompt —
//! `"<|user|>\nName the capital of France.</s>\n<|assistant|>\n"`, carrying
//! the end-of-turn marker the pre-fix prompt lacked.
//!
//! Despite that, the served completion is STILL garbage of the same shape as
//! the pre-fix baseline above. See
//! [`a_gguf_is_served_a_coherent_answer`] for the
//! measured text, its reproducibility, and why that is a separate, already-
//! recorded defect rather than something this epic failed to fix.
//!
//! **This falsified the spec's root-cause claim, not its observation.**
//! Spec §1 attributed the garbage output to the rendered prompt carrying no
//! end-of-turn marker, so the model free-associates. The prompt demonstrably
//! carried the marker and the model free-associated anyway — so "no marker"
//! was not the mechanism. The observation was real; the explanation for it did
//! not survive measurement.
//!
//! # The actual mechanism, found 2026-08-26
//!
//! **The tokenizer, not the prompt.** `Content::extract_tokenizer` built a
//! `Unigram` model whose scores were INVENTED as `-(id as f64)` — the negative
//! token index — while the GGUF's own `tokenizer.ggml.merges` (61249) and
//! `tokenizer.ggml.tokens` (32000), which are BYTE-IDENTICAL to this
//! checkpoint's `tokenizer.json`, sat unread. `byte_fallback` was off, so every
//! newline became id 0, the UNK token; and there was no post-processor, so BOS
//! was never added and `add_special_tokens` was inert.
//!
//! The model was being fed UNK for every newline and shattered subwords
//! throughout — `capital` as `c`+`ap`+`it`+`al`, `France` as `F`+`ran`+`ce`.
//! **28 ids where the reference produces 22.** That is not a prompt defect at
//! all, which is why three rounds of asserting on the rendered STRING never
//! found it: a correct rendered string is not a correct token sequence.
//!
//! `tests/gguf_tokenizer_fidelity.rs` now gates the token ids against the
//! checkpoint's own tokenizer, id for id, on both `add_special_tokens` arms.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use lightbulb::api::chat_template::{self, Resolution};
use lightbulb::api::{ApiConfig, AppState};
use lightbulb::contracts::validation::RawMessage;
use lightbulb::engine::model_runner::ModelRunner;
use lightbulb::engine::{MemoryAwareScheduler, memory_aware_scheduler::MemoryAwareConfig};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tower::ServiceExt;

const MISSING: &str = "no GGUF checkpoint. Set LIGHTBULB_GGUF to a .gguf file.";

/// The checkpoint, from `LIGHTBULB_GGUF` only.
///
/// **Deliberately no fallback path.** An earlier version fell back to a
/// hard-coded absolute path containing a developer's username, which put a
/// local filesystem detail — and that username — in a public repository. Both
/// tests here are already `#[ignore]`d, so requiring the variable costs a
/// reader nothing and removes the leak. Fixed by deletion rather than by
/// parameterising: there is no correct default for a 637 MB file.
fn gguf_path() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var_os("LIGHTBULB_GGUF")?);
    p.is_file().then_some(p)
}

/// The one request every test in this file sends, through the real router
/// with a real runner.
///
/// Shared rather than duplicated in each test: the whole point of both tests
/// is that they are asking about the SAME request — one about what
/// resolution did with it, the other about what the model did with it — so
/// letting the two copies drift apart would silently invalidate the
/// comparison between them. `chat_template.rs`'s own docs call out exactly
/// this failure mode for the four copies `resolve_for_serving` replaced.
async fn serve_the_fixed_prompt(path: &Path) -> (StatusCode, serde_json::Value) {
    let tx = ModelRunner::start(path, 1, 512, Some("f32".to_string()))
        .expect("starting the model runner");
    let state = AppState {
        scheduler: Arc::new(MemoryAwareScheduler::new(MemoryAwareConfig::default())),
        config: ApiConfig::default(),
        db_pool: None,
        inference_tx: Some(tx),
        chat_template: chat_template::resolve_for_serving(path),
        eos_monitor: Default::default(),
    };
    let app = lightbulb::api::openai::routes().with_state(state);
    let body = serde_json::json!({
        "model": "tinyllama",
        "messages": [{"role": "user", "content": "Name the capital of France."}],
        "max_tokens": 24,
        "temperature": 0.0,
    });
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
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    (status, v)
}

/// Everything this epic (chat-template resolution) actually owns, and no
/// more.
///
/// Deliberately does NOT assert on the completion's text. An earlier version
/// of this test did, and failed: not because resolution is wrong — the three
/// assertions below all pass — but because of a separate, downstream defect
/// recorded in the sibling test
/// [`a_gguf_is_served_a_coherent_answer`]. Mixing
/// that assertion into this test would make a correct fix look broken every
/// time this file runs, which is worse than not gating the completion text
/// at all.
#[tokio::test]
#[ignore = "needs the GGUF checkpoint"]
async fn a_gguf_is_served_with_its_own_template() {
    let path = gguf_path().expect(MISSING);

    // 1. The file's own declaration is what resolution used. Exact strings,
    //    not `!is_empty()`: empty is precisely what the defect produced.
    let tk = chat_template::special_tokens(&path);
    assert_eq!(tk.bos, "<s>", "BOS did not come from GGUF metadata");
    assert_eq!(tk.eos, "</s>", "EOS did not come from GGUF metadata");
    assert_eq!(
        chat_template::resolve(&path).resolved_by,
        Resolution::GgufMetadata,
        "resolution fell through to a guess instead of reading the file"
    );

    // 2. Not just which tier resolved — the exact prompt text it renders for
    //    the request this file sends. Promoted from a throwaway `eprintln!`
    //    added during the round-1 fix investigation (in
    //    `build_prompt_from_raw`, `src/api/openai/chat.rs`, reverted before
    //    commit) into a durable assertion, because nothing else in the suite
    //    pins the rendered STRING that actually reaches the model — the
    //    synthetic tests in `tests/chat_template_render.rs` assert
    //    resolution, not this. `build_prompt_from_raw` itself is
    //    `pub(crate)` and unreachable from an integration test, and no
    //    production code was changed to expose it; this renders directly
    //    against the resolved template instead, which is the same
    //    `ResolvedTemplate::render` call `build_prompt_from_raw` makes for a
    //    `Some(t)` template, over the same message list.
    let resolved_template = chat_template::resolve_for_serving(&path)
        .expect("no template resolved for a GGUF that has one");
    let messages = vec![RawMessage {
        role: "user".to_string(),
        content: "Name the capital of France.".to_string(),
    }];
    let rendered = resolved_template
        .render(&messages)
        .expect("rendering the checkpoint's own template");
    assert_eq!(
        rendered, "<|user|>\nName the capital of France.</s>\n<|assistant|>\n",
        "the rendered prompt drifted from the checkpoint's own template"
    );

    // 3. And the HTTP path itself answers with a normal response — status
    //    and shape only. What the model does with this exact, correctly
    //    rendered prompt is a separate claim; see the sibling test.
    let (status, v) = serve_the_fixed_prompt(&path).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        v["choices"][0]["message"]["content"].as_str().is_some(),
        "response carried no string content: {v}"
    );
}

/// The whole point: a real GGUF, served over HTTP, answers the question.
///
/// **This was the recorded-defect test and it was EXPECTED TO FAIL.** Given a
/// provably correct rendered prompt, the served completion was still garbage:
///
/// ```text
/// "| ass istant | i | user | user | ass istant | | | | | ass istant | < | user |"
/// ```
///
/// byte-identical across four independent runs at `temperature: 0.0`. It is
/// flipped to a live assertion here because it now genuinely passes, not
/// because anything it asserts was weakened — the assertion is the same
/// `contains("Paris")` it always made. Measured 2026-08-26 on
/// `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` @ `52e7645b`, Q4_0, CUDA:
///
/// ```text
/// completion: "France| Paris, France</s>"
/// ```
///
/// **The cause was the tokenizer, not the prompt** — see this file's header.
/// The rendered prompt had been correct for a week; the ids built from it were
/// not, and every earlier gate asserted on the string.
///
/// The completion is coherent and terminates on EOS, but it is not pristine:
/// the leading `"France|"` is an artifact worth a look. It is recorded here
/// rather than asserted against, because the assertion this test owns is that
/// the model ANSWERS, and inventing a stricter one to match today's exact bytes
/// would make an unrelated sampling change look like a regression.
#[tokio::test]
#[ignore = "needs the GGUF checkpoint (637 MB); set LIGHTBULB_GGUF"]
async fn a_gguf_is_served_a_coherent_answer() {
    let path = gguf_path().expect(MISSING);

    let (status, v) = serve_the_fixed_prompt(&path).await;
    assert_eq!(status, StatusCode::OK);
    let content = v["choices"][0]["message"]["content"]
        .as_str()
        .expect("response carried no string content");
    eprintln!("completion: {content:?}");

    assert!(
        content.contains("Paris"),
        "recorded defect, downstream of chat-template resolution (see this \
         test's doc comment): the model did not answer the question: {content:?}"
    );
}
