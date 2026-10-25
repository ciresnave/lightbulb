//! AWQ slice 2: `AwqQwen3`/`load_awq_llama` generate the same tokens as an
//! independent reference implementation, on the real checkpoint.
//!
//! A third-party quantizer's own output is a claim until an independent
//! comparison passes, not ground truth (`ROADMAP.md`'s "AWQ loader wiring:
//! slice plan"). The reference here is HF `transformers` + `autoawq`, pinned
//! versions, generated ONCE in WSL2 (autoawq has no Windows wheel; its
//! `triton` dependency has no Windows PyPI wheel at all) and committed as a
//! fixture -- see `scripts/awq_reference/README.md`. This test needs neither
//! WSL2 nor Python: it only reads the committed JSON and runs Lightbulb's own
//! GPU path.
//!
//! **Not `#[ignore]`d**, same policy as `tests/parallel_model_manager_
//! integration.rs::cache_index_reused_by_a_later_unrelated_request_is_not_
//! corrupted` and `src/engine/model_runner.rs::a_cancelled_request_leaks_
//! its_cache_slot_and_the_next_request_panics` -- CireSnave's EXPECTATIONS
//! §5.1g/5.1h. Hosted CI's `gates` job must `--skip` it by name pending a
//! self-hosted GPU runner. Needs the real checkpoint AND a CUDA device
//! (`AwqLinear`'s Marlin kernel has no CPU path), so it panics rather than
//! skipping when either is missing -- a skip that returns success is how a
//! test that cannot run gets counted as passing (see `parallel_model_manager_
//! integration.rs`'s own `require_model` doc for the measured instance of
//! that failure mode in this repo).

use candlelight::core::{DType, Tensor};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct ReferenceFixture {
    provenance: Provenance,
    prompt: String,
    add_special_tokens: bool,
    max_new_tokens: usize,
    generated_token_ids: Vec<u32>,
    generated_text: String,
}

#[derive(Debug, Deserialize)]
struct Provenance {
    model_repo: String,
    model_revision: String,
    model_license: String,
    base_model_repo: String,
    base_model_license: String,
}

fn fixture_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("awq_reference")
        .join("v1")
        .join("qwen3_0_6b_awq_greedy.json")
}

fn load_fixture() -> ReferenceFixture {
    let path = fixture_path();
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "failed to read reference fixture {}: {e}. Regenerate with \
             scripts/awq_reference/reproduce-reference.sh (see its README.md) -- this fixture \
             is committed, so its absence means the checkout is incomplete, not that generation \
             is needed.",
            path.display()
        )
    });
    serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("reference fixture {} did not parse: {e}", path.display()))
}

/// The one tokenizer-and-generate step this test needs. A free function, not
/// a method on anything `load_awq_llama` returns, because `AwqQwen3::forward`
/// already does the KV-cache bookkeeping -- this only drives it and argmaxes.
fn generate_greedy(
    model: &mut lightbulb::model::AwqQwen3,
    device: &candlelight::core::Device,
    prompt_ids: &[u32],
    max_new_tokens: usize,
) -> Vec<u32> {
    let mut offset = 0usize;
    let mut generated = Vec::with_capacity(max_new_tokens);

    let mut next_input: Vec<u32> = prompt_ids.to_vec();
    for _ in 0..max_new_tokens {
        let seq_len = next_input.len();
        let input = Tensor::from_vec(next_input.clone(), (1, seq_len), device)
            .expect("building the input tensor")
            .to_dtype(DType::U32)
            .expect("casting input ids to U32");

        // `offset` is the number of tokens ALREADY in the KV cache before this
        // call -- it feeds both RoPE's position and (indirectly, via how much
        // the cache already holds) the causal mask. It is NOT the decode step
        // index: those agree only on the very first call. Getting this wrong
        // is exactly the bug `examples/test_awq_inference.rs`'s dead manual
        // loop has (`model.forward(&input, step)` with `step` from `0..`),
        // which is why this function does not reuse it.
        let logits = model
            .forward(&input, offset)
            .expect("forward pass during greedy generation");
        offset += seq_len;

        let logits: Vec<f32> = logits
            .squeeze(0)
            .expect("squeezing the batch dim")
            .to_dtype(DType::F32)
            .expect("casting logits to f32")
            .to_vec1()
            .expect("flattening logits");
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx as u32)
            .expect("logits must be non-empty");

        generated.push(next_token);
        next_input = vec![next_token];
    }
    generated
}

#[test]
fn awq_qwen3_generation_matches_the_independent_reference_implementation() {
    let fixture = load_fixture();

    // Cross-check the fixture against the same facts recorded in
    // `test_models.rs`, so a drift between the two provenance records (one
    // generated by a Python script, one hand-written) is caught rather than
    // silently diverging.
    assert_eq!(
        fixture.provenance.model_repo,
        lightbulb::test_models::AWQ_MODEL_REPO
    );
    assert_eq!(
        fixture.provenance.model_revision,
        lightbulb::test_models::AWQ_PINNED_REVISION
    );
    assert_eq!(fixture.provenance.model_license, "gpl-3.0");
    assert_eq!(
        fixture.provenance.base_model_repo,
        lightbulb::test_models::AWQ_BASE_MODEL_REPO
    );
    assert_eq!(fixture.provenance.base_model_license, "apache-2.0");

    let model_dir = lightbulb::test_models::awq_qwen3_dir().expect(
        "AWQ checkpoint not found -- set LIGHTBULB_AWQ_MODEL_DIR, or place \
         Qwen3-0.6B-AWQ under the models directory (see test_models::awq_qwen3_dir)",
    );
    let model_dir_str = model_dir
        .to_str()
        .expect("model_dir must be valid UTF-8 for load_awq_llama's &str signature");

    let (mut model, _config, device, _name_mapper) =
        lightbulb::loaders::load_awq_llama(model_dir_str, Some("f16"), false).unwrap_or_else(|e| {
            panic!(
                "failed to load the AWQ checkpoint at {model_dir_str}: {e:#}. This test requires \
                 CUDA (AwqLinear's Marlin kernel has no CPU path) -- if this box has no usable \
                 CUDA device, it must be --skip'd by name, not left to fail here."
            )
        });

    let tokenizer =
        tokenizers::Tokenizer::from_file(Path::new(model_dir_str).join("tokenizer.json"))
            .expect("loading the checkpoint's own tokenizer.json");
    let prompt_ids: Vec<u32> = tokenizer
        .encode(fixture.prompt.as_str(), fixture.add_special_tokens)
        .expect("tokenizing the fixture's prompt")
        .get_ids()
        .to_vec();

    let generated = generate_greedy(&mut model, &device, &prompt_ids, fixture.max_new_tokens);

    assert_eq!(
        generated, fixture.generated_token_ids,
        "Lightbulb's AWQ generation diverged from the independent reference. Reference decoded \
         text: {:?}. If `scripts/awq_reference/reproduce-reference.sh` was re-run with different \
         pinned versions, check whether this is a reference-side change before assuming a \
         Lightbulb regression (see the fixture README's \"What to do if the output changes\").",
        fixture.generated_text
    );
}
