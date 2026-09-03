use std::path::Path;

/// A local-generation smoke test.
///
/// # Why this is `#[ignore]`d and fails rather than skipping
///
/// It used to run in the ORDINARY `cargo test`, print
/// "[skipped] models/llama-3b not found", and report `ok` in 0.00s — a passing
/// test in every green suite report that executed no assertion. Its name said
/// "if_present", which is honest about the condition and does nothing about the
/// verdict.
///
/// `#[ignore]` is the honest gate: it does not run and does not claim to.
/// Running it is then deliberate, so a missing checkpoint is a setup error.
/// `LIGHTBULB_LLAMA3B_MODEL` overrides the path, matching the convention in
/// `api_result_metadata.rs`, whose own constant already says these "fail rather
/// than skipping" — the discipline existed here, it just was not applied to
/// this file.
#[test]
#[ignore = "needs a local llama-3b checkpoint; set LIGHTBULB_LLAMA3B_MODEL"]
fn local_model_smoke() {
    let dir =
        std::env::var("LIGHTBULB_LLAMA3B_MODEL").unwrap_or_else(|_| "models/llama-3b".to_string());
    let model_dir = Path::new(&dir);
    assert!(
        model_dir.exists(),
        "checkpoint not found at {dir:?}. This test is #[ignore]d, so running it is deliberate and a missing checkpoint is a setup error rather than a reason to pass. Set LIGHTBULB_LLAMA3B_MODEL."
    );
    let prompt = "Hello from Lightbulb! ";
    let out = lightbulb::local_llama_generate(
        model_dir.to_str().unwrap(),
        prompt,
        3,
        0.0, // argmax for determinism
        None,
        42,
    )
    .expect("local generation should succeed");
    assert!(out.len() > 0, "should generate at least some text");
}
