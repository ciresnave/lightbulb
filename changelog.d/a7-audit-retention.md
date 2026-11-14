kind: audit-log retention (ROADMAP A7)
---
### Added
- `LIGHTBULB_AUDIT_RETENTION_DAYS=N` deletes `audit_logs` rows older than N days, once an hour in a background task. Unset keeps every row, as before. Zero or a non-number stops startup with a message instead of being read as "delete everything". Tests cover the parsing and pin the purge statement to `audit_logs` only; the delete itself needs PostgreSQL and was not run here.
