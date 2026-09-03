#!/usr/bin/env python3
"""Clippy gate with a per-lint baseline.

WHY THIS IS NOT JUST `cargo clippy -D warnings`.

As of 2026-09-03 this crate emits 620 clippy/rustc warnings across 52 distinct
lints. Denying all warnings today would mean triaging 620 sites before any
clippy gate could exist at all, and a gate deferred until a cleanup finishes is
a gate that does not exist. So this ratchets instead: the current counts are
recorded in clippy-baseline.tsv, and the gate fails when a lint gets WORSE or a
NEW kind of lint appears. The 620 stay visible, counted, and deferrable.

WHAT THIS CATCHES

  1. Any lint kind not in the baseline at all.
  2. Any baselined lint whose count increased.
  3. Anything clippy itself treats as an error -- the `clippy::correctness`
     group is deny-by-default, so `cargo clippy` exits non-zero on it with no
     flags from us. That is not hypothetical: `cargo clippy` was failing on
     main with `clippy::eq_op` at src/engine/reasoning_controls.rs:218, and no
     gate ran clippy, so nothing noticed.

WHAT THIS DOES NOT CATCH, stated because a guard trusted past its reach is
worse than no guard:

  - Fixing one `collapsible_if` and adding another leaves the count equal and
    passes. This is a ratchet on counts, not on sites.
  - Default features only, same as the `test` gate. `--features fuel-engine`
    code is not linted here.
  - A count that goes DOWN is allowed silently and does not auto-update the
    baseline. The file is only tightened when someone runs --update, so the
    recorded numbers are a ceiling, not a measurement of today.

WHY THERE IS A POSITIVE CONTROL -- AND WHAT WAS ACTUALLY MEASURED

The worry was that a warm target dir makes cargo skip re-emitting diagnostics,
so `cargo clippy` prints nothing, exits 0, and a count-based gate reads zero
warnings as a clean pass. This script therefore used to run
`cargo clean -p lightbulb` first, to force a recheck.

THAT PREMISE IS FALSE ON THIS TOOLCHAIN, and it was a claim nobody had checked.
Measured 2026-09-03, rustc/clippy 1.98.0: with the clean removed and the target
dir warm from an immediately preceding run, clippy still reported all 620
diagnostics. cargo REPLAYS cached diagnostics for fresh units. So the clean was
buying nothing and cost a full recheck on every run.

The zero-guard below stays anyway, and is defensible on its own terms rather
than on that story: if this ever does produce zero against a non-empty baseline
-- a cargo change, a cache restored oddly, a target dir in a state we have not
seen -- then zero and "clean tree" are the same output, and this repo has been
bitten by that shape repeatedly. The guard makes the gate fail loudly instead.
It is a control against an unknown, not a fix for an observed defect.

WHAT THE COUNTS COUNT

Diagnostic INSTANCES, not source sites. `--all-targets` checks the lib and the
test targets separately, so a line compiled into both is reported twice.
Measured: adding one `let y = ...; y` moved `let_and_return` from 2 to 4, and
one unused fn moved `dead_code` from 39 to 41. So 620 is roughly 2x the distinct
sites for code that lives in both targets. The ratchet is unaffected -- the
doubling is consistent between runs -- but do not read 620 as "620 places".

Usage:
    python scripts/clippy_gate.py            # check
    python scripts/clippy_gate.py --update   # rewrite the baseline
"""

import json
import os
import subprocess
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, "clippy-baseline.tsv")

# Must stay identical between --update and a check, or the two measure
# different things and the comparison is meaningless.
CLIPPY_CMD = [
    "cargo",
    "clippy",
    "--all-targets",
    "--message-format=json",
]


def measure():
    """Return (Counter of lint -> count, clippy's exit code, error text)."""
    # No `cargo clean` here: measured on 1.98.0, cargo replays cached
    # diagnostics for fresh units, so a warm target dir still reports all 620.
    # See the docstring -- the clean was justified by a premise that turned out
    # to be false, and only cost a full recheck per run.
    #
    # encoding pinned explicitly: text=True alone decodes with the locale
    # codec, which on Windows is cp1252 and dies on the UTF-8 box-drawing
    # characters rustc renders in its diagnostics. errors="replace" so a stray
    # byte degrades one character rather than losing the whole measurement.
    proc = subprocess.run(
        CLIPPY_CMD,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    counts = Counter()
    errors = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if msg.get("reason") != "compiler-message":
            continue
        m = msg.get("message", {})
        level = m.get("level")
        if level not in ("warning", "error"):
            continue
        code = (m.get("code") or {}).get("code") or "<uncoded>"
        counts[code] += 1
        if level == "error":
            errors.append(m.get("rendered") or m.get("message", ""))

    return counts, proc.returncode, errors, proc.stderr


def read_baseline():
    if not os.path.exists(BASELINE):
        return None
    out = Counter()
    with open(BASELINE, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            count, name = line.split("\t", 1)
            out[name] = int(count)
    return out


def write_baseline(counts):
    lines = [
        "# Per-lint clippy/rustc warning counts. Generated by:",
        "#     python scripts/clippy_gate.py --update",
        "#",
        "# A ceiling, not a census: counts may only go down. See the module",
        "# docstring in clippy_gate.py for what this gate does and does not catch.",
        "#",
        "# count\tlint",
    ]
    for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(str(n) + "\t" + name)
    with open(BASELINE, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


# A measured total below this fraction of the baseline is treated as a partial
# measurement rather than an improvement. See `verdict`.
PLAUSIBLE_FLOOR = 0.5


# The four outcomes. Named rather than returned as bare exit codes, so that the
# selftest asserts WHICH conclusion was reached rather than only whether the
# process would fail -- "partial measurement" and "regressed" are both exit 1,
# and a test that cannot tell them apart passes when the gate reaches the right
# verdict for the wrong reason.
NOTHING_MEASURED = "nothing_measured"
PARTIAL = "partial"
REGRESSED = "regressed"
CLEAN = "clean"

EXIT_CODE = {NOTHING_MEASURED: 1, PARTIAL: 1, REGRESSED: 1, CLEAN: 0}


def verdict(counts, base):
    """The pass/fail decision, and nothing else. Returns (outcome, detail).

    Four conditions, in order:

      measured 0 against a non-empty baseline -> NOTHING_MEASURED
      measured at or below the floor          -> PARTIAL
      new lint kind or increased count        -> REGRESSED
      otherwise                               -> CLEAN

    THE SECOND CHECK IS THE ONE THAT WAS MISSING, and it is the direction that
    matters. The gate ratchets counts DOWN silently, so any measurement that
    sees less than the real amount -- a dropped --all-targets, a feature change,
    a clippy version that reports differently, a target that did not build --
    looks exactly like a cleanup. Measured: halving every count passes and
    prints "52 lints improved". A gate that fails open on the cheap direction is
    worse than no gate, because it also supplies the reassurance.

    Only the exactly-zero case was caught before, and zero is the one value a
    real partial measurement is least likely to take.

    No message text here. Rendering lives in `render`, so that this function
    reads as its four conditions -- previously they were interleaved with the
    comprehensions and f-strings that build the failure output, which made the
    decision the minority of the function by volume.
    """
    total = sum(counts.values())
    base_total = sum(base.values())

    if base_total > 0 and total == 0:
        return NOTHING_MEASURED, {"total": total, "base_total": base_total}

    # `<=`, not `<`. Exactly half is not an arbitrary boundary here: --all-targets
    # double-counts code compiled into both lib and test, so dropping it yields
    # almost exactly 50%. The most likely partial measurement lands precisely on
    # the boundary, and a strict `<` would let it through.
    if base_total > 0 and total <= base_total * PLAUSIBLE_FLOOR:
        return PARTIAL, {"total": total, "base_total": base_total}

    new = sorted(k for k in counts if k not in base)
    worse = sorted((k, base[k], counts[k]) for k in counts if k in base and counts[k] > base[k])
    if new or worse:
        return REGRESSED, {"new": new, "worse": worse, "counts": counts}

    better = sorted((k, base[k], counts.get(k, 0)) for k in base if counts.get(k, 0) < base[k])
    return CLEAN, {"total": total, "lints": len(counts), "better": better}


def render(outcome, detail):
    """Turn a verdict into the lines a reader sees. No decisions here."""
    if outcome == NOTHING_MEASURED:
        return [
            f"FAIL: measured 0 diagnostics against a baseline of {detail['base_total']}.",
            "      Nothing was measured -- this is an instrument failure, not a",
            "      clean tree. Zero is not a result here.",
        ]

    if outcome == PARTIAL:
        total, base_total = detail["total"], detail["base_total"]
        return [
            f"FAIL: measured {total} diagnostics against a baseline of {base_total}",
            f"      ({total * 100 // base_total}%), which is too far down to be a",
            "      cleanup. Treated as a PARTIAL MEASUREMENT: a dropped",
            "      --all-targets, a feature change, or a target that did not",
            "      build all look like an improvement to a ratchet.",
            "",
            "      If the reduction is real, acknowledge it explicitly:",
            "          python scripts/clippy_gate.py --update",
        ]

    if outcome == REGRESSED:
        counts = detail["counts"]
        lines = ["FAIL: clippy regressed against scripts/clippy-baseline.tsv"]
        lines += [f"  NEW LINT   {k}  x{counts[k]}" for k in detail["new"]]
        lines += [f"  INCREASED  {k}  {was} -> {now}" for k, was, now in detail["worse"]]
        lines += [
            "",
            "Fix the new sites. If the increase is genuinely intended, run",
            "    python scripts/clippy_gate.py --update",
            "and say in the commit message why the ceiling went up.",
        ]
        return lines

    if outcome == CLEAN:
        lines = [
            f"clippy gate OK: {detail['total']} diagnostics, {detail['lints']} lints, "
            "none new or increased"
        ]
        better = detail["better"]
        if better:
            lines.append(f"  {len(better)} lint(s) improved -- tighten the ceiling with --update:")
            lines += [f"    {k}  {was} -> {now}" for k, was, now in better[:10]]
        return lines

    # Explicit, because the CLEAN branch used to be the fall-through. Deleting
    # any earlier branch then routed that outcome here, and the failure surfaced
    # as `KeyError: 'lints'` from a detail dict shaped for a different verdict --
    # a real failure wearing a confusing cause. Measured by doing it.
    raise AssertionError(f"render has no branch for outcome {outcome!r}")


def _selftest():
    """Assert every path through `verdict`. Runs on every invocation.

    Microseconds, and it means the decision logic is checked at the moment it is
    used rather than at some point in the past. The partial-measurement case in
    particular passed for the gate's whole first life, and nothing would have
    told anyone.
    """
    base = {"a": 100, "b": 100}
    # Asserted on the OUTCOME, not the exit code. PARTIAL and REGRESSED are both
    # exit 1, so a test comparing exit codes passes when the gate reaches the
    # right verdict for the wrong reason -- which is the failure mode this whole
    # script exists to catch, and it would have been sitting in its own selftest.
    cases = [
        ("clean", {"a": 100, "b": 100}, CLEAN),
        ("improved", {"a": 90, "b": 100}, CLEAN),
        ("new lint", {"a": 100, "b": 100, "c": 1}, REGRESSED),
        ("increased", {"a": 101, "b": 100}, REGRESSED),
        ("measured zero", {}, NOTHING_MEASURED),
        ("half measured", {"a": 50, "b": 50}, PARTIAL),
        ("just above floor", {"a": 100, "b": 1}, CLEAN),
    ]
    for name, counts, want in cases:
        got, detail = verdict(counts, base)
        assert got == want, f"verdict({name}) returned {got}, expected {want}"
        # Rendering must survive every outcome: an unrenderable verdict would
        # crash the gate at the moment it had something to report.
        lines = render(got, detail)
        assert lines and all(isinstance(x, str) for x in lines), f"render({name}) produced {lines!r}"
    # An empty baseline must not make everything a partial measurement.
    assert verdict({}, {})[0] == CLEAN, "empty baseline against empty counts must pass"
    # Every outcome must map to an exit code, or the gate cannot report itself.
    for outcome in (NOTHING_MEASURED, PARTIAL, REGRESSED, CLEAN):
        assert outcome in EXIT_CODE, f"{outcome} has no exit code"


def main():
    _selftest()
    update = "--update" in sys.argv

    # A gate that can be disarmed by adding a flag is a gate that still writes
    # its "I ran" marker. --update rewrites the ceiling and returns 0 for any
    # input, so in CI it would be a green tick over an unconditional pass.
    if update and os.environ.get("CI"):
        print("FAIL: --update rewrites the baseline and always succeeds, so it is")
        print("      never a gate. Run it locally and commit the result.")
        return 1

    counts, rc, errors, stderr = measure()
    total = sum(counts.values())

    if errors:
        print("FAIL: clippy reported errors, not just warnings.")
        print("      The clippy::correctness group is deny-by-default, so this")
        print("      is a lint cargo itself refuses to let pass.")
        for e in errors[:5]:
            print(e)
        return 1

    if rc != 0:
        print("FAIL: clippy exited " + str(rc) + " with no parsed error message.")
        print("      This is a build failure, not a lint result. stderr tail:")
        print(stderr[-2000:])
        return 1

    if update:
        write_baseline(counts)
        print("baseline updated: " + str(total) + " diagnostics, " + str(len(counts)) + " lints")
        return 0

    base = read_baseline()
    if base is None:
        print("FAIL: no baseline at " + BASELINE)
        print("      Create one with: python scripts/clippy_gate.py --update")
        return 1

    outcome, detail = verdict(counts, base)
    for line in render(outcome, detail):
        print(line)
    return EXIT_CODE[outcome]


if __name__ == "__main__":
    sys.exit(main())
