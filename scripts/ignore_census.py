#!/usr/bin/env python3
"""Fail when the number of `#[ignore]` attributes changes without `docs/ignore-count.txt` changing with it.

WHY: `#[ignore]` is a last resort (CireSnave 2026-10-16, EXPECTATIONS 5.1g). A new one must be a deliberate edit of the
recorded number, in the same change, so it is seen in review; a removed one lowers the number (the ROADMAP C1/C1b rows
say which). The count is attribute lines only (`#[ignore]`, `#[ignore = "..."]`, `#[ignore] // why`), not prose that
mentions `#[ignore]`, over `git ls-files '*.rs'` (the index, never a disk walk: other lanes' worktrees live beside this
one). It says nothing about WHAT is ignored, so swapping one ignore for another is not caught; the recorded number is a
ratchet on the total, not a list.

`--selftest` runs the matcher over a synthetic source with a known count, so "0 found" and "the matcher is broken" are
not the same output.
"""
import re
import subprocess  # nosec B404 - fixed-argv git call below, no untrusted input
import sys

ATTR = re.compile(r"^\s*#\[ignore\b")
RECORD = "docs/ignore-count.txt"


def count(text: str) -> int:
    return sum(1 for line in text.splitlines() if ATTR.match(line))


def selftest() -> None:
    sample = '#[test]\n#[ignore]\nfn a() {}\n    #[ignore = "x"]\n/// mentions #[ignore] in prose\n// #[ignore]\nfn b() {}\n'
    got = count(sample)
    if got != 2:
        sys.exit(f"selftest: matcher counted {got} in a sample with exactly 2 attributes")


def main() -> None:
    selftest()
    # nosec B603 B607 - literal argv, `git` resolved from PATH like every other script here; nothing user-supplied
    files = subprocess.run(  # nosec B603 B607
        ["git", "ls-files", "*.rs"], capture_output=True, text=True, check=True
    ).stdout.split()
    if not files:
        sys.exit("git ls-files returned no .rs files: the census would read 0 from an empty population")
    total = 0
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            total += count(fh.read())
    recorded = int(open(RECORD, encoding="utf-8").read().split()[0])
    print(f"#[ignore] attributes: {total} over {len(files)} .rs files; recorded {recorded} in {RECORD}")
    if total != recorded:
        sys.exit(
            f"the ignore count is {total} but {RECORD} says {recorded}. Set it to {total} in this change, and say why in "
            "the ROADMAP C1/C1b rows: a new #[ignore] is a last resort (EXPECTATIONS 5.1g) and must be a visible edit; a "
            "removed one is progress to record."
        )


if __name__ == "__main__":
    main()
