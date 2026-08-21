#!/usr/bin/env bash
# Local quality gates. ARMED, NOT AUTOMATIC.
#
# .github/workflows/ci.yml runs these same gates on every push to main and on
# every pull request, so they are no longer convention-only. This script stays
# as the local form: it is what you run BEFORE pushing, and it is the thing CI
# mirrors rather than the other way round. If the two ever diverge, this file
# is not the authority — CI is what actually gates.
#
# TOOLCHAIN: pinned. rust-toolchain.toml pins 1.98.0 by explicit version, so
# these gates report a claim about a NAMED compiler rather than an ambient one.
# Before that file, the default on this machine was NIGHTLY 1.99.0 and every
# green this repo reported was a claim about a compiler nobody had stated.
#
# Usage: bash scripts/check.sh
set -uo pipefail
cd "$(dirname "$0")/.."
fail=0

# ── Build parallelism ────────────────────────────────────────────────────────
# -j 4 is mandatory here: full parallelism ICEs rustc on this machine and
# surfaces as spurious `rlib format` / `E0786 paging file` errors naming
# different crates each run. Settled by elimination; do not re-diagnose.
J="-j 4"

echo "── cargo fmt --all --check"
cargo fmt --all --check >/dev/null 2>&1 || { echo "   FAIL: formatting differs"; fail=1; }

echo "── cargo doc (rustdoc lints only — see note)"
# CRITICAL: deny the RUSTDOC lints specifically, NOT `-D warnings`.
#
# With `RUSTDOCFLAGS=-D warnings`, any ordinary rustc lint — an unused import,
# an unused `mut` — aborts the run BEFORE rustdoc reaches link resolution. The
# command then emits no link diagnostics at all, and a grep for them returns
# zero. Zero reads identically to "none found".
#
# That is not hypothetical: it happened here on 2026-08-20 and produced a
# reported "0 broken intra-doc links" when there were 24. The gate could stop
# before the thing it measures. Scoping the deny to rustdoc's own lints means
# a stray rustc warning can no longer mask a broken link. Verified by
# reintroducing an `unused_mut` and confirming the link count was unchanged.
RUSTDOCFLAGS="-D rustdoc::broken_intra_doc_links -D rustdoc::private_intra_doc_links" \
  cargo doc --workspace --no-deps >/dev/null 2>&1 \
  || { echo "   FAIL: broken or private intra-doc links"; fail=1; }

echo "── cargo test --lib"
cargo test $J --lib 2>&1 | grep -E "^test result" || { echo "   FAIL: lib tests"; fail=1; }

echo
[ $fail -eq 0 ] && echo "all gates passed" || echo "GATES FAILED"
exit $fail
