kind: model pull: disk-space check and admin route (ROADMAP F1)
---
### Added
`lightbulb-cli pull` and the new `POST /v1/lightbulb/admin/models/pull` (`{"model":"repo[:quant]"}`, admin role) now refuse before downloading when the models directory's disk cannot hold the files not already present (typed `InsufficientSpace`, HTTP 507). The route answers when the download ends and reports no progress; other codes: 400 bad spec, 404 no matching `.gguf`, 409 repo already being pulled, 501 no models directory, 502 Hub or download failure. `sysinfo` gains its `disk` feature (no new crate in `Cargo.lock`). Verified with 17 unit tests against a localhost stub Hub (disk refusal with an exactly-enough control, route refusals 400/401/403/409/501); not run against the real Hub.
