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


def main():
    update = "--update" in sys.argv
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

    # POSITIVE CONTROL. A warm target dir makes cargo skip re-emitting
    # diagnostics entirely; the result is zero warnings and a clean exit, which
    # is indistinguishable from a genuinely clean crate. The baseline is
    # non-empty, so zero here means the measurement did not happen.
    if sum(base.values()) > 0 and total == 0:
        print("FAIL: measured 0 diagnostics against a baseline of "
              + str(sum(base.values())) + ".")
        print("      That is an instrument failure, not a clean tree: cargo did")
        print("      not recheck the crate, so nothing was measured. Zero is not")
        print("      a result here.")
        return 1

    new = sorted(k for k in counts if k not in base)
    worse = sorted(
        (k, base[k], counts[k]) for k in counts if k in base and counts[k] > base[k]
    )

    if new or worse:
        print("FAIL: clippy regressed against scripts/clippy-baseline.tsv")
        for k in new:
            print("  NEW LINT   " + k + "  x" + str(counts[k]))
        for k, was, now in worse:
            print("  INCREASED  " + k + "  " + str(was) + " -> " + str(now))
        print("")
        print("Fix the new sites. If the increase is genuinely intended, run")
        print("    python scripts/clippy_gate.py --update")
        print("and say in the commit message why the ceiling went up.")
        return 1

    better = sorted(
        (k, base[k], counts.get(k, 0)) for k in base if counts.get(k, 0) < base[k]
    )
    print("clippy gate OK: " + str(total) + " diagnostics, " + str(len(counts))
          + " lints, none new or increased")
    if better:
        print("  " + str(len(better)) + " lint(s) improved -- tighten the ceiling with --update:")
        for k, was, now in better[:10]:
            print("    " + k + "  " + str(was) + " -> " + str(now))
    return 0


if __name__ == "__main__":
    sys.exit(main())
