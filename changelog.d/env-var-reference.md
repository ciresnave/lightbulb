kind: docs: environment-variable reference (ROADMAP D2)
---
Adds `docs/CONFIGURATION.md`, a table of every `LIGHTBULB_*` variable read under `src/` (35 names; 14 were documented before), with defaults and which values refuse to start, and links it from QUICKSTART. Checked by diffing `git grep -o LIGHTBULB_[A-Z0-9_]+ -- src` against the file both ways: 0 missing, 0 extra (the same diff against the old docs found 21 missing). Docs only; no code changed.
