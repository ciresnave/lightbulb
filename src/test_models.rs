//! Locating the TinyLlama checkpoint the test suite uses, in one place.
//!
//! # Why this exists
//!
//! `fn tinyllama_dir()` was defined TWELVE times across `src/` and `tests/`,
//! each carrying this literal:
//!
//! ```text
//! C:/Users/cires/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1...
//! ```
//!
//! An absolute path into one person's home directory. On any other machine
//! every one of those tests can only ever skip, and `LIGHTBULB_REQUIRE_MODEL=1`
//! -- the arming from `crate::test_notice` that turns a skip into a failure --
//! would be correct-but-permanently-red anywhere else. The mechanism for making
//! skips honest is useless while the thing it looks for is unfindable.
//!
//! # The copies had already diverged, which is the argument against twelve of them
//!
//! - Ten checked `model.safetensors`; two checked `config.json`.
//! - ONE of the twelve -- `tests/paged_plan_once.rs` -- already honoured a
//!   `TINYLLAMA_DIR` environment variable.
//!
//! That last one matters. The override is not an idea nobody had; it is an idea
//! ONE author had, which copy-and-paste did not carry to the other eleven. So
//! this adopts `TINYLLAMA_DIR` rather than inventing a new name: the convention
//! already existed in the repo and deserves to be the one that survives.
//!
//! # Resolution order
//!
//! 1. `$TINYLLAMA_DIR` -- explicit, wins, and is not validated beyond the
//!    sentinel check so that pointing it at a copy is always possible.
//! 2. The HuggingFace cache, located from `$HF_HUB_CACHE`, else `$HF_HOME/hub`,
//!    else `<home>/.cache/huggingface/hub`, where `<home>` is `$USERPROFILE` or
//!    `$HOME`. The pinned revision is preferred; if it is absent, any snapshot
//!    directory carrying the sentinel is accepted.
//! 3. The portfolio models chain: `$LIGHTBULB_MODELS_DIR`, else `$MODELS_DIR`,
//!    else `C:\Models`, joined with the model directory name.
//!
//! # The pinned revision is preferred but not required
//!
//! `PINNED_REVISION` is the snapshot every recorded measurement in this repo was
//! taken against, including the committed golden fixture. Preferring it keeps
//! those numbers comparable. Falling back to any snapshot keeps the tests
//! runnable on a machine that fetched a different revision -- the architecture
//! assertions (dim 2048, 22 layers, 32 heads, 4 KV heads) hold across revisions
//! of the same model, so a mismatched snapshot degrades reproducibility rather
//! than correctness. Which one was used is reported by [`describe`].

use std::path::PathBuf;

/// The HuggingFace repo id, in cache-directory form.
pub const MODEL_REPO_DIR: &str = "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0";

/// The plain directory name, for the models-chain layout.
pub const MODEL_DIR_NAME: &str = "TinyLlama-1.1B-Chat-v1.0";

/// The snapshot every recorded measurement in this repo was taken against.
pub const PINNED_REVISION: &str = "fe8a4ea1ffedaf415f4da2f062534de366a451e6";

/// The file whose presence means "this really is an unpacked checkpoint".
pub const WEIGHTS_SENTINEL: &str = "model.safetensors";

/// The weaker sentinel, for tests that only need the config.
pub const CONFIG_SENTINEL: &str = "config.json";

fn home() -> Option<PathBuf> {
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
}

fn hf_hub_root() -> Option<PathBuf> {
    if let Some(v) = std::env::var_os("HF_HUB_CACHE") {
        return Some(PathBuf::from(v));
    }
    if let Some(v) = std::env::var_os("HF_HOME") {
        return Some(PathBuf::from(v).join("hub"));
    }
    Some(home()?.join(".cache").join("huggingface").join("hub"))
}

fn models_root() -> PathBuf {
    for var in ["LIGHTBULB_MODELS_DIR", "MODELS_DIR"] {
        if let Some(v) = std::env::var_os(var) {
            return PathBuf::from(v);
        }
    }
    PathBuf::from("C:/Models")
}

/// The checkpoint directory, requiring `model.safetensors`.
pub fn tinyllama_dir() -> Option<PathBuf> {
    tinyllama_dir_with(WEIGHTS_SENTINEL)
}

/// The checkpoint directory, requiring a caller-chosen sentinel file.
///
/// Two call sites need only `config.json`. Preserved as a distinct entry point
/// rather than silently tightened to the weights: strengthening their
/// precondition would change which machines those tests run on, which is a
/// behaviour change dressed as a refactor.
pub fn tinyllama_dir_with(sentinel: &str) -> Option<PathBuf> {
    let has = |p: &PathBuf| p.join(sentinel).is_file();

    if let Some(v) = std::env::var_os("TINYLLAMA_DIR") {
        let p = PathBuf::from(v);
        return has(&p).then_some(p);
    }

    if let Some(hub) = hf_hub_root() {
        let snapshots = hub.join(MODEL_REPO_DIR).join("snapshots");
        let pinned = snapshots.join(PINNED_REVISION);
        if has(&pinned) {
            return Some(pinned);
        }
        // Any other revision, so a machine that fetched a different one is not
        // stuck. Sorted, so the choice is deterministic rather than
        // filesystem-order dependent -- an arbitrary pick would make a
        // reproducibility problem intermittent instead of visible.
        if let Ok(entries) = std::fs::read_dir(&snapshots) {
            let mut candidates: Vec<PathBuf> =
                entries.flatten().map(|e| e.path()).filter(&has).collect();
            candidates.sort();
            if let Some(first) = candidates.into_iter().next() {
                return Some(first);
            }
        }
    }

    let chained = models_root().join(MODEL_DIR_NAME);
    has(&chained).then_some(chained)
}

/// Where the checkpoint was found and whether it is the pinned revision.
///
/// Recorded measurements are comparable only against `PINNED_REVISION`, so a
/// run on a different snapshot should say so rather than silently produce
/// numbers that look like the fixture's.
pub fn describe() -> String {
    match tinyllama_dir() {
        None => format!(
            "no TinyLlama checkpoint found (set TINYLLAMA_DIR, or place {MODEL_DIR_NAME} \
             under the models directory)"
        ),
        Some(p) => {
            let pinned = p.file_name().is_some_and(|n| n == PINNED_REVISION);
            let which = if pinned {
                "pinned revision"
            } else {
                "NOT the pinned revision -- recorded numbers are not comparable"
            };
            format!("{} ({which})", p.display())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// The environment is process-wide and cargo runs tests in parallel, so a
    /// test that sets TINYLLAMA_DIR can be observed by one that reads it. That
    /// would make `locates_the_checkpoint_where_one_exists` SKIP -- and a skip
    /// is a pass, so the race would be invisible rather than flaky.
    ///
    /// Every test here that touches the variable takes this first.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn pinned_revision_is_a_full_sha1() {
        assert_eq!(PINNED_REVISION.len(), 40, "a HF revision is a 40-char sha1");
        assert!(
            PINNED_REVISION.bytes().all(|b| b.is_ascii_hexdigit()),
            "PINNED_REVISION must be hex: {PINNED_REVISION:?}"
        );
    }

    #[test]
    fn an_override_pointing_at_nothing_resolves_to_none() {
        // The override must still be validated: a stale TINYLLAMA_DIR should
        // read as "no checkpoint", not as a directory that will fail later
        // inside a loader with a worse message.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let var = "TINYLLAMA_DIR";
        let previous = std::env::var_os(var);
        // SAFETY: this test owns the variable for its duration and restores it.
        unsafe { std::env::set_var(var, "Z:/definitely/not/here") };
        let got = tinyllama_dir();
        match previous {
            Some(v) => unsafe { std::env::set_var(var, v) },
            None => unsafe { std::env::remove_var(var) },
        }
        assert!(
            got.is_none(),
            "a non-existent override must not resolve: {got:?}"
        );
    }

    /// On a machine that has the checkpoint, the locator must actually find it.
    ///
    /// This is the case the twelve hardcoded copies got right and everything
    /// else wrong, so it is worth an explicit arm. It uses the loud-skip helper:
    /// where `LIGHTBULB_REQUIRE_MODEL=1` says the environment provides a
    /// checkpoint, failing to locate it is a failure rather than a skip.
    #[test]
    fn locates_the_checkpoint_where_one_exists() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(dir) = tinyllama_dir() else {
            crate::test_notice::skip_unless_required(
                "LIGHTBULB_REQUIRE_MODEL",
                "no TinyLlama snapshot reachable by any resolution rule",
            );
            return;
        };
        assert!(
            dir.join(WEIGHTS_SENTINEL).is_file(),
            "resolved to {} but it has no {WEIGHTS_SENTINEL}",
            dir.display()
        );
        println!("{}", describe());
    }
}
