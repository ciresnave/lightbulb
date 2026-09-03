//! Loud, machine-detectable test skips — and a way to make a skip a failure
//! where the precondition was supposed to be provisioned.
//!
//! # A test that prints a skip message and returns is a PASS
//!
//! MEASURED 2026-09-03, and it is the whole reason this module exists. Running
//! `loads_tinyllama_from_local_dir`, which ends in an `eprintln!`:
//!
//! ```text
//! cargo test ... -- --ignored                 ok    message NOT shown
//! cargo test ... -- --ignored --nocapture     ok    message shown
//! ```
//!
//! Same test, same passing result, same `eprintln!`. `cargo test` captures
//! stdout and stderr for tests that PASS and prints them only for failures, so
//! by default the skip notice is written and then discarded. The control is the
//! second row: the message is genuinely emitted, so its absence in the first row
//! is the harness capturing it, not a print that never happened.
//!
//! **So "skips loudly" is not a property a `println!` can have.** The harness
//! decides who sees it, and by default it decides nobody. Several tests in this
//! repo describe themselves as skipping loudly; none of them do, through no
//! fault of their authors.
//!
//! # Two remedies, because they answer different questions
//!
//! [`NOTICE_TOKEN`] makes a skip **findable** — one fixed ASCII string a runner
//! can grep for under `--nocapture`, so "did anything skip?" has an answer that
//! does not depend on a human reading scrollback.
//!
//! [`skip_unless_required`] makes a skip **impossible where it would be a lie**.
//! When the named variable is set to `1`, the precondition was supposed to be
//! there, so its absence is a failure rather than a skip.
//!
//! # Arming is keyed on PROVISIONING, not on a list of suites
//!
//! The variable is set by whatever step supplies the precondition. A job with a
//! "fetch the TinyLlama snapshot" step sets `LIGHTBULB_REQUIRE_MODEL=1`; a job
//! without one does not. The arming and the provisioning are then the same
//! declaration, and the question "can this environment satisfy the
//! precondition?" stops being a judgement someone maintains and becomes a fact
//! about the workflow.
//!
//! An enumerated allowlist goes stale the first time somebody adds a suite, and
//! it can be optimistically wrong for free — you add a name. Under
//! provisioning-keyed arming, being optimistic means adding a fetch step and
//! being pessimistic means deleting one. Both are visible in a diff and neither
//! happens by inattention.
//!
//! # What this does not do
//!
//! It does not make skips visible in a default `cargo test` run. Nothing can:
//! the capture is the harness's, and the only lever is `--nocapture`. This makes
//! the notice *greppable when captured output is shown* and *fatal when the
//! precondition was promised*. A skip in an environment that genuinely cannot
//! satisfy the precondition is a true statement about the machine and stays a
//! pass, which is the point.

/// The fixed marker every skip notice carries.
///
/// Printable ASCII with no whitespace, so it survives a grep, a CI log viewer
/// and a regex without quoting. `notice_token_is_greppable` asserts that,
/// because a token that acquires a unicode character or a space would break
/// every runner reading for it while every test still passed.
pub const NOTICE_TOKEN: &str = "LIGHTBULB-SKIP";

/// Emit a skip notice, or panic if this environment promised the precondition.
///
/// `require_var` is the environment variable whose presence means "this
/// environment provisions the thing, so a skip here is a lie". `precondition`
/// names what is missing, in the terms someone would need to supply it.
///
/// ```no_run
/// # use lightbulb::test_notice::skip_unless_required;
/// # fn tinyllama_dir() -> Option<std::path::PathBuf> { None }
/// let Some(dir) = tinyllama_dir() else {
///     skip_unless_required("LIGHTBULB_REQUIRE_MODEL", "no TinyLlama snapshot");
///     return;
/// };
/// # let _ = dir;
/// ```
pub fn skip_unless_required(require_var: &str, precondition: &str) {
    let required = std::env::var(require_var).is_ok_and(|v| v == "1");
    if required {
        panic!(
            "{NOTICE_TOKEN}: {precondition}\n\
             {require_var}=1 says this environment provisions it, so its absence is a \
             failure rather than a skip. Either supply the precondition or unset \
             {require_var} in the job that does not provide it."
        );
    }
    eprintln!(
        "{NOTICE_TOKEN}: {precondition} (set {require_var}=1 where this is provisioned \
         to make it a failure)"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The token is the runner's interface. A unicode character, a space, or an
    /// empty string would break every grep reading for it while every test in
    /// the repo still passed — the failure would appear as "nothing ever skips".
    #[test]
    fn notice_token_is_greppable() {
        assert!(
            !NOTICE_TOKEN.is_empty(),
            "the token is the runner's interface"
        );
        assert!(
            NOTICE_TOKEN
                .bytes()
                .all(|b| b.is_ascii_graphic() && b != b'"'),
            "NOTICE_TOKEN must be printable ASCII with no whitespace or quotes, got {NOTICE_TOKEN:?}"
        );
    }

    #[test]
    fn unarmed_environment_skips_rather_than_panicking() {
        // A name no job sets. The point of the arming is that it is opt-in.
        skip_unless_required(
            "LIGHTBULB_REQUIRE_NOTHING_AT_ALL",
            "a precondition nobody provisions",
        );
    }

    #[test]
    fn armed_environment_turns_a_skip_into_a_failure() {
        // Scoped to this test's own variable name so it cannot race another
        // test through the process-wide environment.
        let var = "LIGHTBULB_REQUIRE_TEST_NOTICE_SELFTEST";
        // SAFETY: single-threaded within this test, and the variable is unique
        // to it, so no other test observes the mutation.
        unsafe { std::env::set_var(var, "1") };
        let caught = std::panic::catch_unwind(|| {
            skip_unless_required(var, "a precondition this test claims to provision");
        });
        unsafe { std::env::remove_var(var) };
        assert!(
            caught.is_err(),
            "with {var}=1 the skip must fail the test, not print and pass"
        );
    }

    /// The arming must be exactly `1`, not merely "the variable exists".
    /// `FOO=0` and `FOO=` are the shapes a CI expression produces when it means
    /// "not armed" — vulkane's `runner.os == 'Linux' && '1' || ''` yields the
    /// empty string on the unarmed leg, and an `is_ok()` check would read that
    /// as armed.
    #[test]
    fn empty_or_zero_does_not_arm() {
        let var = "LIGHTBULB_REQUIRE_TEST_NOTICE_EMPTYARM";
        for value in ["", "0", "false"] {
            unsafe { std::env::set_var(var, value) };
            let caught = std::panic::catch_unwind(|| {
                skip_unless_required(var, "a precondition nobody provisions");
            });
            assert!(
                caught.is_ok(),
                "{var}={value:?} must NOT arm the requirement — an empty string is what a \
                 CI ternary emits on its unarmed branch"
            );
        }
        unsafe { std::env::remove_var(var) };
    }
}
