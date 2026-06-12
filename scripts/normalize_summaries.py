#!/usr/bin/env python3
"""
normalize_summaries.py

Renames all files in docs/summaries/ to canonical slug format (lowercase, hyphen-separated),
updates docs/literature/index.md links, and prints a report of changes.
"""
import os
import re
from pathlib import Path

SUMMARIES_DIR = Path("docs/summaries")
INDEX_MD = Path("docs/literature/index.md")

slugify = lambda s: re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def normalize_filenames():
    changes = []
    for f in SUMMARIES_DIR.glob("*.md"):
        base = f.stem
        canonical = slugify(base)
        if canonical != base:
            new_path = f.parent / (canonical + ".md")
            if new_path.exists():
                # Skip renaming if target file already exists
                continue
            os.rename(f, new_path)
            changes.append((f.name, new_path.name))
    return changes


def update_index(changes):
    with open(INDEX_MD, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        for old, new in changes:
            if old in line:
                lines[i] = line.replace(old, new)
    with open(INDEX_MD, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    changes = normalize_filenames()
    if changes:
        update_index(changes)
        print("Renamed files:")
        for old, new in changes:
            print(f"{old} -> {new}")
    else:
        print("No changes needed.")


if __name__ == "__main__":
    main()
