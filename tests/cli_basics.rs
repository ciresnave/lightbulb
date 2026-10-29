// SPDX-License-Identifier: MIT OR Apache-2.0
//! Every shipped binary answers `--version`/`-V`/`--help`/`-h` with exit 0 and
//! output, rejects an unknown flag with exit 2, and never writes ANSI escapes
//! to a pipe (CodeRipper OPS-011 `cli-basics`). The server is run with no
//! `DATABASE_URL` to prove flag handling happens before any configuration or
//! database access.

use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

const BINS: [(&str, &str); 3] = [
    ("lightbulb", env!("CARGO_BIN_EXE_lightbulb")),
    ("lightbulb-cli", env!("CARGO_BIN_EXE_lightbulb-cli")),
    ("lightbulb-probe", env!("CARGO_BIN_EXE_lightbulb-probe")),
];

fn run(exe: &str, args: &[&str]) -> Output {
    let mut cmd = Command::new(exe);
    cmd.args(args)
        .env_remove("DATABASE_URL")
        .env_remove("NO_COLOR")
        .stdin(Stdio::null());
    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // A binary that ignores the flag and starts serving would hang the test.
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if child.try_wait().unwrap().is_some() {
            return child.wait_with_output().unwrap();
        }
        if Instant::now() > deadline {
            child.kill().unwrap();
            let out = child.wait_with_output().unwrap();
            panic!(
                "{exe} {args:?} did not exit within 30s; stderr: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn assert_no_ansi(name: &str, args: &[&str], out: &Output) {
    for (stream, bytes) in [("stdout", &out.stdout), ("stderr", &out.stderr)] {
        assert!(
            !bytes.contains(&0x1b),
            "{name} {args:?}: {stream} contains an ANSI escape: {:?}",
            String::from_utf8_lossy(bytes)
        );
    }
}

#[test]
fn version_and_help_exit_zero_with_output_and_no_ansi() {
    for (name, exe) in BINS {
        for flag in ["--version", "-V", "--help", "-h"] {
            let out = run(exe, &[flag]);
            assert_eq!(out.status.code(), Some(0), "{name} {flag}: {out:?}");
            assert!(!out.stdout.is_empty(), "{name} {flag}: empty stdout");
            assert_no_ansi(name, &[flag], &out);
        }
    }
}

#[test]
fn version_names_the_crate_version() {
    for (name, exe) in BINS {
        let out = run(exe, &["--version"]);
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(
            text.contains(env!("CARGO_PKG_VERSION")),
            "{name} --version printed {text:?}"
        );
    }
}

#[test]
fn unknown_flag_exits_two() {
    for (name, exe) in BINS {
        let out = run(exe, &["--definitely-not-a-flag"]);
        assert_eq!(out.status.code(), Some(2), "{name}: {out:?}");
        assert_no_ansi(name, &["--definitely-not-a-flag"], &out);
    }
}

/// With no `DATABASE_URL` and no `--no-auth` the server refuses to start; its
/// (piped) log output must carry no ANSI escapes either.
#[test]
fn server_startup_refusal_writes_no_ansi_to_a_pipe() {
    let out = run(BINS[0].1, &[]);
    assert_ne!(out.status.code(), Some(0), "{out:?}");
    assert_no_ansi("lightbulb", &[], &out);
    assert!(
        !out.stdout.is_empty() || !out.stderr.is_empty(),
        "expected a refusal message"
    );
}
