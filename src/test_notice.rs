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
//! When the named variable is set to an ON value — `1`, `true`, `yes` or `on`,
//! trimmed and case-insensitive — the precondition was supposed to be there, so
//! its absence is a failure rather than a skip. `0`, `false`, `no`, `off`, empty
//! and unset are OFF; anything else panics rather than guessing. See
//! [`armed_by`] for why the OFF list is explicit and why silence in either
//! direction is the failure.
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

/// The arming decision, separated from where the value comes from.
///
/// Pure over the VALUE so it is testable WITHOUT mutating the environment:
/// `set_var` is `unsafe` in edition 2024 and races every other test in the
/// binary. The `ENV_LOCK` elsewhere in this repo exists for exactly that.
///
/// ```text
/// on:  1 true yes on      (trimmed, ASCII-case-insensitive)
/// off: 0 false no off, unset, or empty
/// anything else: panic
/// ```
///
/// A REQUIRE gate that silently fails to arm turns every skip back into a
/// pass, and a typo is how that happens. Silence in EITHER direction is the
/// failure.
///
/// The OFF list is explicit rather than "anything not on the ON list", because
/// the fourth arm is the point: an unrecognised value must be LOUD, and a
/// catch-all off-arm silently swallows every typo.
///
/// # Rejected forms, recorded so nobody re-derives them
///
/// ```text
/// !v.is_empty()               vulkane's   -- arms on "0" and "false"
/// v != "0" && !v.is_empty()   mlmf's      -- fixes "0" only; false/no/off still arm
/// v == "1"                    lightbulb's -- silently ignores true/yes/on
/// ```
///
/// The last was ours, and it is the direction that matters most: somebody sets
/// `FOO=true` meaning ON, we ignore it, and the gate they think they enabled is
/// off while every skip passes quietly.
///
/// Agreed across vulkane / lightbulb / mlmf on 2026-09-05. Shared TEXT, not a
/// shared crate — mlmf's workspace refuses `dev-dependencies` for gated members,
/// and a crate would put test-only code in a published artifact. Variable NAMES
/// stay distinct: a shared name would make arming a device requirement silently
/// arm a corpus requirement, a failure mode invented by the standardisation
/// itself.
fn armed_by(var: &str, value: Option<&str>) -> bool {
    let Some(raw) = value else { return false };
    // `raw` UNTRIMMED in the message: the reader needs to see the trailing
    // space that caused it.
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "0" | "false" | "no" | "off" => false,
        "1" | "true" | "yes" | "on" => true,
        _ => panic!(
            "{var}={raw:?} is not a recognised on/off value \
             (on: 1 true yes on; off: 0 false no off; unset or empty = off)"
        ),
    }
}

/// The only caller. `var` is used ONCE, for both the lookup and the message, so
/// a mismatched variable/value pair is not expressible.
///
/// ⚠️ Deliberately NOT `std::env::var(var).ok().as_deref()`. `.ok()` maps
/// `VarError::NotUnicode` to `None`, where it is indistinguishable from unset
/// and reads as OFF — the value is lost THROUGH THE TYPE before it can reach
/// `armed_by`'s fourth arm, so the loudness we just built never fires on it.
/// The old `is_ok_and(|v| v == "1")` had the same hole. On Windows the
/// environment is UTF-16 and can carry unpaired surrogates, so this is
/// reachable rather than theoretical.
///
/// Found by vulkane while implementing their copy of the agreed text, which the
/// agreed text did not cover — noticed only on writing it.
fn armed(var: &str) -> bool {
    match std::env::var(var) {
        Ok(v) => armed_by(var, Some(&v)),
        Err(std::env::VarError::NotPresent) => armed_by(var, None),
        Err(std::env::VarError::NotUnicode(raw)) => panic!(
            "{var}={raw:?} is not valid UTF-8, so it cannot be a recognised on/off value \
             (on: 1 true yes on; off: 0 false no off; unset or empty = off)"
        ),
    }
}

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
    if armed(require_var) {
        panic!(
            "{NOTICE_TOKEN}: {precondition}\n\
             {require_var} is set to an ON value, which says this environment provisions \
             it, so its absence is a failure rather than a skip. Either supply the \
             precondition or unset {require_var} in the job that does not provide it."
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

    /// The OFF list, at the predicate rather than through the environment.
    ///
    /// `FOO=0` and `FOO=` are the shapes a CI expression produces when it means
    /// "not armed" — vulkane's `runner.os == 'Linux' && '1' || ''` yields the
    /// empty string on the unarmed leg, and an `is_ok()` check would read that
    /// as armed.
    #[test]
    fn the_off_list_does_not_arm() {
        assert!(!armed_by("V", None), "unset must not arm");
        for value in ["", "0", "false", "no", "off", " OFF ", "False"] {
            assert!(
                !armed_by("V", Some(value)),
                "{value:?} must NOT arm — an empty string is what a CI ternary emits on \
                 its unarmed branch, and 0/false/no/off are what a person writes meaning off"
            );
        }
    }

    /// The ON list, including the three values our old `v == "1"` silently
    /// ignored. That direction is the dangerous one: the gate someone thinks
    /// they enabled is off, and every skip goes back to passing quietly.
    #[test]
    fn the_on_list_arms() {
        for value in ["1", "true", "yes", "on", "TRUE", " On ", "\tyes\n"] {
            assert!(
                armed_by("V", Some(value)),
                "{value:?} must arm — a REQUIRE gate that silently fails to arm turns \
                 every skip back into a pass"
            );
        }
    }

    /// The fourth arm, and the reason the OFF list is explicit rather than a
    /// catch-all: a typo must be LOUD.
    ///
    /// Asserts the MESSAGE, not merely that something panicked. A panic with
    /// different text would mean some other failure and would still satisfy a
    /// bare `should_panic`.
    #[test]
    #[should_panic(expected = "is not a recognised on/off value")]
    fn an_unrecognised_value_panics_rather_than_silently_declining() {
        armed_by("LIGHTBULB_REQUIRE_SOMETHING", Some("ture"));
    }

    /// The env plumbing, kept deliberately thin.
    ///
    /// The decision lives in `armed_by` and is tested there. This covers only
    /// that `armed` reads the right variable — WITHOUT it, moving the decision
    /// into a pure function would leave the lookup untested while the
    /// value-level tests all passed. Vulkane's `armed_by` refactor is right and
    /// this is the half it would otherwise have dropped.
    ///
    /// ⚠️ WHAT THIS DOES NOT COVER, stated so a pass is not read as more than
    /// it is: only the `Ok` and `NotPresent` arms. `VarError::NotUnicode`
    /// cannot be produced through `set_var` with a `&str`, so the panic arm for
    /// it is unexercised here. That is the cost of the by-value split, and it
    /// is smaller than the process-global mutation the split avoids.
    #[test]
    fn the_env_lookup_reaches_the_predicate() {
        let var = "LIGHTBULB_REQUIRE_TEST_NOTICE_LOOKUP";
        // SAFETY: single-threaded within this test, and the variable is unique
        // to it, so no other test observes the mutation.
        unsafe { std::env::set_var(var, "yes") };
        let armed_when_set = armed(var);
        unsafe { std::env::remove_var(var) };
        let armed_when_unset = armed(var);
        assert!(armed_when_set, "{var}=yes must reach armed_by and arm");
        assert!(!armed_when_unset, "unset must not arm");
    }
}
