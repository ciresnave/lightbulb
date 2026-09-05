//! The LIVE GGUF loader must refuse a non-llama checkpoint BY NAME.
//!
//! # Why this test exists at the integration level
//!
//! `crate::gguf::require_llama_architecture` is unit-tested against all thirteen
//! architectures the local corpus declares. Those tests prove the DECISION is
//! right. They cannot prove the live loader CALLS it.
//!
//! ⚠️ That distinction is the whole reason this file exists. There are two GGUF
//! config readers in this repo — `loaders::load_gguf_llama` and
//! `ParallelModelManager::load_gguf` — and **only the second is reachable**.
//! `load_gguf_llama`'s own doc comment says so, and warns in as many words:
//! *"a correct fix applied to the wrong caller is indistinguishable from a wrong
//! fix"*. A check wired into the dead path alone would pass every unit test in
//! the crate and change nothing that runs.
//!
//! So this drives the loader people actually use, with a real file off disk.
//!
//! # A vocabulary-only GGUF is a complete fixture here
//!
//! The architecture is read from the KV header, before any tensor is touched, so
//! `ggml-vocab-qwen2.gguf` — zero tensors — exercises the check exactly as a
//! full qwen2 checkpoint would.
//!
//! This was not obvious and I got it wrong first: the corpus has 11 files with
//! tensors and every one declares `llama`, which reads as "the corpus cannot
//! answer architecture questions". It cannot answer *loading* questions. It
//! answers this one thirteen times over, because the fixture set was built for
//! tokenizers and nothing in it announces which questions it can still serve.
//!
//! ```text
//!   LIGHTBULB_GGUF_CORPUS=<dir> cargo test --test gguf_architecture_refusal -- --ignored --nocapture
//! ```

use lightbulb::model::parallel_model_manager::ParallelModelManager;
use std::path::PathBuf;

/// Locate one corpus file by name, or `None` if the corpus is not provisioned.
fn corpus_file(name: &str) -> Option<PathBuf> {
    let root = std::env::var_os("LIGHTBULB_GGUF_CORPUS")?;
    let mut stack = vec![PathBuf::from(root)];
    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(&dir).ok()?;
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|f| f == name) {
                return Some(p);
            }
        }
    }
    None
}

#[test]
#[ignore = "needs a local GGUF corpus; set LIGHTBULB_GGUF_CORPUS"]
fn the_live_loader_refuses_a_qwen2_checkpoint_by_name() {
    let Some(path) = corpus_file("ggml-vocab-qwen2.gguf") else {
        lightbulb::test_notice::skip_unless_required(
            "LIGHTBULB_REQUIRE_CORPUS",
            "no ggml-vocab-qwen2.gguf under LIGHTBULB_GGUF_CORPUS",
        );
        return;
    };

    // `ParallelModelManager` is not `Debug`, so `expect_err` is unavailable.
    let err = match ParallelModelManager::load_gguf(
        &path,
        1,
        512,
        Some(candlelight::core::Device::Cpu),
        None,
    ) {
        Ok(_) => panic!(
            "a qwen2 checkpoint must be refused by the live loader, and it LOADED -- \
             which means llama-shaped config was built from qwen2 metadata"
        ),
        Err(e) => e.to_string(),
    };

    eprintln!("live loader refused with: {err}");

    assert!(
        err.contains("qwen2"),
        "the refusal must NAME the declared architecture, not a key: {err}"
    );
    assert!(
        !err.contains("Missing or invalid metadata key: llama."),
        "⚠️ the loader is still reporting a missing llama.* key. The key is not \
         missing -- it is under the qwen2. prefix -- and that message sends a reader \
         hunting for a corrupt GGUF. This is the exact defect the check exists to \
         remove: {err}"
    );
}

/// The control, and it is not optional: without it the assertion above is
/// satisfied by a loader that refuses EVERYTHING, which would be a far worse
/// defect than the one being fixed.
///
/// Uses TinyLlama's own vocabulary file rather than the 637 MB checkpoint, so
/// this needs no model download. It asserts only that the refusal is not the
/// ARCHITECTURE refusal — a llama file may still fail later for having no
/// tensors, which is a true and different statement.
#[test]
#[ignore = "needs a local GGUF corpus; set LIGHTBULB_GGUF_CORPUS"]
fn a_llama_checkpoint_is_not_refused_for_its_architecture() {
    let Some(path) = corpus_file("ggml-vocab-llama-spm.gguf") else {
        lightbulb::test_notice::skip_unless_required(
            "LIGHTBULB_REQUIRE_CORPUS",
            "no ggml-vocab-llama-spm.gguf under LIGHTBULB_GGUF_CORPUS",
        );
        return;
    };

    let outcome =
        ParallelModelManager::load_gguf(&path, 1, 512, Some(candlelight::core::Device::Cpu), None);
    let message = match &outcome {
        Ok(_) => String::new(),
        Err(e) => e.to_string(),
    };
    eprintln!("llama vocabulary file outcome: {message:?}");

    assert!(
        !message.contains("general.architecture"),
        "a file declaring `llama` must NOT be refused for its architecture, or the \
         check is a permanent refusal rather than a gate: {message}"
    );
}
