kind: CI: CodeRipper 0.47.5 pin
---
Moves the portfolio workflow pin in `.github/workflows/ci.yml` from coderipper#307 (0.43.49) to coderipper#356 (`c72a6e0`, installs 0.47.5). The rule-id migration tool (`tools/migrate-rule-ids.py` at c72a6e0) reports nothing to migrate: `.coderipper.toml` and the CI files hold no old rule names (only `SEC-002` ids).
