//! Real-model integration tests for Output Contracts.
//!
//! Tests are `#[ignore]` by default. Run them explicitly:
//!
//! ```sh
//! cargo test --test contracts_live -- --ignored --nocapture
//!
//! # Point at a different model:
//! LIGHTBULB_TEST_MODEL_DIR=../models/llama-3b cargo test --test contracts_live -- --ignored --nocapture
//! ```
//!
//! A **single** ModelRunner is shared across all tests via `OnceLock` so the
//! model loads exactly once.  Without this, parallel tests each load the model
//! on a different CUDA device simultaneously, exhausting GPU memory.
//!
//! If the model path cannot be found or the model fails to load, the tests
//! **fail** with a clear panic message.

use std::path::PathBuf;
use std::sync::OnceLock;

use lightbulb::contracts::{
    ContractOutput, OutputContractSpec,
    executor::{ContractExecutionResult, execute_contract_with_runner},
    validation::RawMessage,
};
use lightbulb::engine::model_runner::{InferenceJob, ModelRunner, ResponseMode};

type RunnerTx = std::sync::mpsc::Sender<InferenceJob>;

// ─── Shared runner (loaded once, shared by all tests) ─────────────────────

static RUNNER: OnceLock<RunnerTx> = OnceLock::new();

/// Clone the shared sender.  Panics (FAIL) if the runner could not be
/// initialized — the test should not silently pass when the model is broken.
fn get_runner() -> RunnerTx {
    RUNNER.get_or_init(init_runner).clone()
}

/// Start the runner, wait for it to finish loading, and verify with a 1-token
/// probe.  Panics on any failure so the test shows as FAILED rather than as a
/// silent "ok" that hides a real problem.
fn init_runner() -> RunnerTx {
    let model_dir = find_model_dir();

    let tx = ModelRunner::start(&model_dir, 1, 512, Some("f32".to_string()))
        .expect("ModelRunner::start failed");

    // ModelRunner::start always returns Ok(tx) immediately — the real load
    // happens on a spawned thread.  Wait, then probe to detect failures.
    std::thread::sleep(std::time::Duration::from_secs(5));

    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
    tx.send(InferenceJob {
        id: "probe".to_string(),
        prompt: "user: hello\nassistant:".to_string(),
        max_new_tokens: 1,
        temperature: 0.0,
        response_mode: ResponseMode::Complete(resp_tx),
    })
    .expect("probe send failed — model thread exited before responding");

    // Block on the probe using a fresh OS thread with its own mini-runtime so
    // we never call block_on inside an existing tokio runtime (which panics).
    let probe_result = std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(resp_rx)
    })
    .join()
    .expect("probe thread panicked");

    match probe_result {
        Ok(Ok(_)) => {
            println!("[contracts_live] model runner ready");
            tx
        }
        Ok(Err(e)) => panic!("[contracts_live] model load failed: {e}"),
        Err(e) => panic!("[contracts_live] probe channel dropped: {e}"),
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────

/// Locate the model directory or panic.  Set `LIGHTBULB_TEST_MODEL_DIR` to
/// override the default search path.
fn find_model_dir() -> PathBuf {
    if let Ok(val) = std::env::var("LIGHTBULB_TEST_MODEL_DIR") {
        let p = PathBuf::from(&val);
        assert!(
            p.exists(),
            "LIGHTBULB_TEST_MODEL_DIR is set but path does not exist: {val}"
        );
        return p;
    }
    for candidate in &["../models/llama-3b", "../../models/llama-3b", "models/llama-3b"] {
        let p = PathBuf::from(candidate);
        if p.join("config.json").exists() {
            println!(
                "[contracts_live] Found model at {}",
                p.canonicalize().unwrap_or_else(|_| p.clone()).display()
            );
            return p;
        }
    }
    panic!(
        "[contracts_live] no model directory found — set LIGHTBULB_TEST_MODEL_DIR \
         or place the model at ../models/llama-3b"
    );
}

fn user_msgs(content: &str) -> Vec<RawMessage> {
    vec![RawMessage { role: "user".to_string(), content: content.to_string() }]
}

// ─── Tests ────────────────────────────────────────────────────────────────

/// Smoke: model returns non-empty text for a plain prompt (no contract).
#[ignore]
#[tokio::test]
async fn live_smoke_plain_generation() {
    let tx = get_runner();

    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
    tx.send(InferenceJob {
        id: "smoke".to_string(),
        prompt: "user: Say the word hello\nassistant:".to_string(),
        max_new_tokens: 20,
        temperature: 0.0,
        response_mode: ResponseMode::Complete(resp_tx),
    })
    .expect("send failed");

    match resp_rx.await {
        Ok(Ok(text)) => {
            println!("[smoke] model output: {text:?}");
            assert!(!text.is_empty(), "model should produce at least one token");
        }
        Ok(Err(e)) => panic!("inference error: {e}"),
        Err(e) => panic!("channel error: {e}"),
    }
}

/// EnumChoice yes/no: "Is water wet?"
#[ignore]
#[tokio::test]
async fn live_enum_choice_yes_no() {
    let tx = get_runner();

    let spec = OutputContractSpec::EnumChoice {
        choices: vec!["yes".to_string(), "no".to_string()],
        case_sensitive: false,
        allow_index: true,
    };

    let result: ContractExecutionResult = execute_contract_with_runner(
        tx,
        &user_msgs("Is water wet? Answer with yes or no."),
        "llama-3b",
        &spec,
        3,
        &[],
        40,
        0.0,
    )
    .await
    .unwrap();

    println!(
        "[yes/no] parse_success={} attempts={} output={:?}",
        result.result.parse_success, result.result.attempts, result.result.output
    );

    assert!(
        result.result.parse_success,
        "EnumChoice should parse within 3 attempts; raw: {:?}",
        result.final_text
    );
    assert!(
        matches!(&result.result.output, ContractOutput::EnumChoice { choice } if choice == "yes" || choice == "no"),
        "choice must be yes or no; got: {:?}",
        result.result.output
    );
}

/// EnumChoice three-way: sentiment classification.
#[ignore]
#[tokio::test]
async fn live_enum_choice_three_way() {
    let tx = get_runner();

    let spec = OutputContractSpec::EnumChoice {
        choices: vec!["positive".to_string(), "negative".to_string(), "neutral".to_string()],
        case_sensitive: false,
        allow_index: true,
    };

    let result = execute_contract_with_runner(
        tx,
        &user_msgs("The movie was absolutely fantastic! What is the sentiment?"),
        "llama-3b",
        &spec,
        3,
        &[],
        40,
        0.0,
    )
    .await
    .unwrap();

    println!(
        "[3-way] parse_success={} attempts={} output={:?}",
        result.result.parse_success, result.result.attempts, result.result.output
    );

    assert!(
        result.result.parse_success,
        "3-way EnumChoice should parse within 3 attempts; raw: {:?}",
        result.final_text
    );
}

/// TaggedFields: classify a Java NullPointerException.
/// Small models often fail to format tags correctly, so we assert leniently.
#[ignore]
#[tokio::test]
async fn live_tagged_fields_error_classification() {
    let tx = get_runner();

    let spec = OutputContractSpec::TaggedFields {
        allowed_tags: vec!["CATEGORY".to_string(), "SEVERITY".to_string()],
        required_tags: vec![],
        repeatable_tags: vec![],
    };

    let result = execute_contract_with_runner(
        tx,
        &user_msgs("Classify this error: \"NullPointerException at line 42 in UserService.java\""),
        "llama-3b",
        &spec,
        3,
        &[],
        80,
        0.0,
    )
    .await
    .unwrap();

    println!(
        "[tagged_fields] parse_success={} attempts={} output={:?}",
        result.result.parse_success, result.result.attempts, result.result.output
    );
    if let ContractOutput::TaggedFields { ref tags } = result.result.output {
        println!("  CATEGORY={:?}  SEVERITY={:?}", tags.get("CATEGORY"), tags.get("SEVERITY"));
    }

    assert!(!result.final_text.is_empty(), "must return some text");
    assert_eq!(result.result.model_id, "llama-3b");
    assert!(result.result.attempts <= 3);
    assert!(result.result.latency_ms > 0);
}

/// Fallback chain: 3-label primary (2 attempts) → yes/no fallback.
#[ignore]
#[tokio::test]
async fn live_fallback_chain() {
    let tx = get_runner();

    let primary = OutputContractSpec::EnumChoice {
        choices: vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()],
        case_sensitive: false,
        allow_index: true,
    };
    let fallback = OutputContractSpec::EnumChoice {
        choices: vec!["yes".to_string(), "no".to_string()],
        case_sensitive: false,
        allow_index: true,
    };

    let result = execute_contract_with_runner(
        tx,
        &user_msgs("Is this a test? Pick from: yes, no, alpha, beta, or gamma."),
        "llama-3b",
        &primary,
        2,
        &[fallback],
        40,
        0.0,
    )
    .await
    .unwrap();

    println!(
        "[fallback] contract_type={} parse_success={} attempts={} output={:?}",
        result.result.contract_type,
        result.result.parse_success,
        result.result.attempts,
        result.result.output
    );

    assert!(result.result.attempts <= 4, "≤2 primary + ≤2 fallback");
    assert!(!result.final_text.is_empty());
}
