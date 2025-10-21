use std::path::Path;

#[test]
fn local_model_smoke_if_present() {
    let model_dir = Path::new("models/llama-3b");
    if !model_dir.exists() {
        eprintln!("[skipped] models/llama-3b not found; skipping local generation smoke test");
        return;
    }
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
