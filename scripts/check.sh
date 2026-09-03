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
# The deny lives in src/lib.rs as a CRATE-LEVEL ATTRIBUTE, not here.
#
# It used to be RUSTDOCFLAGS on this line. That applies to EVERY rustdoc
# invocation including dependencies, defeating cargo's lint-cap for registry
# crates: CI's first run failed on 16 broken links inside hardware-query-0.2.1
# -- someone else's crate, on Linux-only paths this machine never compiles,
# which is exactly why it passed locally and failed on ubuntu-latest.
#
# The old comment below is kept because its point still holds and is why the
# attribute names the two rustdoc lints rather than using `deny(warnings)`.
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
cargo doc --workspace --no-deps >/dev/null 2>&1 \
  || { echo "   FAIL: broken or private intra-doc links"; fail=1; }

echo "── cargo test --lib"
cargo test $J --lib 2>&1 | grep -E "^test result" || { echo "   FAIL: lib tests"; fail=1; }


# The clippy gate. Ratchets per-lint counts against scripts/clippy-baseline.tsv
# rather than denying all warnings, because this crate emits 620 diagnostics
# across 52 lints and a gate deferred until that cleanup finishes is a gate that
# does not exist. Fails on a NEW lint kind or an INCREASED count.
#
# It also catches the deny-by-default group for free -- `cargo clippy` exits
# non-zero on clippy::correctness with no flags. That is not theoretical: clippy
# was RED on main with clippy::eq_op, and nothing ran clippy, so the red was
# invisible for as long as it existed.
echo "-- clippy (per-lint ratchet, see scripts/clippy_gate.py)"
python scripts/clippy_gate.py || fail=1

echo
# ── What these gates do NOT cover ────────────────────────────────────────────
# PRINTED UNCONDITIONALLY, INCLUDING ON SUCCESS. That is the whole point: the
# line below is read at the moment someone is about to trust a green result.
#
# CireSnave ruled on 2026-08-27 that Lightbulb gets no GPU CI runner — a cost
# constraint, not a judgement about these tests. So the two behavioural
# acceptance gates in this repo are verified by a person running them on
# demand, and by nothing else.
#
# The hazard that creates is the one this repo keeps meeting in other forms:
# "nobody ran it this month" and "it passes" are INDISTINGUISHABLE from the
# outside. A silent absence looks exactly like coverage. So this reports the
# gap every run rather than leaving it to be discovered — the same treatment
# `gpu_paged_vs_contiguous` got in its own file, made routine.
echo "── NOT RUN HERE (require a GPU and a ~2.2 GB checkpoint)"
echo "   chat_template_e2e   a chat model answers its own template AND stops"
echo "   fuel_engine_http    the completion is coherent English, not a 200 with noise"
echo "   gguf_serving_e2e    a real GGUF is served end to end"
echo "   These are NEVER CI-verified: no GPU runner exists, by decision."
echo "   Run them yourself on a GPU box:"
echo "     LIGHTBULB_GGUF=<path>.gguf cargo test --release --features fuel-engine \\"
echo "       --test chat_template_e2e --test fuel_engine_http --test gguf_serving_e2e \\"
echo "       -- --ignored --nocapture --test-threads=1"
echo "   --test-threads=1 is load-bearing: each test loads its own ~2.2 GB copy"
echo "   of the checkpoint and in parallel they exhaust host memory."

echo
[ $fail -eq 0 ] && echo "all gates passed (see NOT RUN above — it is not empty)" || echo "GATES FAILED"
exit $fail
